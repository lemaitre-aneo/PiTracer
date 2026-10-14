//! Runs against an in-process fake of the ArmoniK control plane, whose
//! tasks follow the iterative protocol of rust-tracer-worker (through the
//! same lib functions): passes accumulate, intermediate display images are
//! published, and the block's final results come last.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use armonik::events::{ResultStatusUpdate, Update, subscribe};
use armonik::reexports::tokio_stream::wrappers::TcpListenerStream;
use armonik::reexports::tonic;
use armonik::results::filter::Condition;
use armonik::server::{RequestContext, ResultsService, SessionsService, TasksService};
use armonik::{FilterStringOperator, events, results, sessions, tasks};
use clap::Parser;
use raytracer::{
    BlockResult, Convergence, DenoisePayload, LINEAR_NAME, Scene, TaskType, Tile, accumulate,
    decode_scene, encode_scene, render_block_row,
};
use tokio::sync::{broadcast, mpsc};

use super::*;
use crate::pool::PoolConfig;
use crate::watcher::{EventsWatcher, PollingWatcher};

/// What the fake's tasks do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Behavior {
    /// Follow the protocol to the end.
    Render,
    /// Render the first pass, publish it, then never finish.
    Stall,
    Abort,
    /// Never finish.
    Hang,
}

#[derive(Debug, Clone)]
struct FakeResult {
    session_id: String,
    name: String,
    status: ResultStatus,
    data: Option<Vec<u8>>,
}

struct Fake {
    behavior: Behavior,
    results: Mutex<HashMap<String, FakeResult>>,
    next_id: Mutex<u64>,
    /// Session calls, in order (`create`, `close`, ...).
    calls: Mutex<Vec<String>>,
    /// Submission-level options of every submission.
    submitted_options: Mutex<Vec<Option<TaskOptions>>>,
    /// Result status changes: (session, result, status).
    events: broadcast::Sender<(String, String, ResultStatus)>,
    /// Downloads of final (denoised) display images still to fail
    /// (UNAVAILABLE) before they work: the run cannot end without them.
    failing_downloads: std::sync::atomic::AtomicUsize,
}

impl Fake {
    fn new(behavior: Behavior) -> Arc<Self> {
        Arc::new(Self {
            behavior,
            results: Mutex::default(),
            next_id: Mutex::default(),
            calls: Mutex::default(),
            submitted_options: Mutex::default(),
            events: broadcast::channel(4096).0,
            failing_downloads: Default::default(),
        })
    }

    fn id(&self, prefix: &str) -> String {
        let mut next = self.next_id.lock().unwrap();
        *next += 1;
        format!("{prefix}{next:04}")
    }

    fn call(&self, call: &str) {
        self.calls.lock().unwrap().push(call.to_owned());
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }

    fn add_result(&self, session_id: &str, name: String, data: Option<Vec<u8>>) -> results::Raw {
        let result_id = self.id("result");
        let status = if data.is_some() {
            ResultStatus::Completed
        } else {
            ResultStatus::Created
        };
        self.results.lock().unwrap().insert(
            result_id.clone(),
            FakeResult {
                session_id: session_id.to_owned(),
                name: name.clone(),
                status: status.clone(),
                data,
            },
        );
        if status == ResultStatus::Completed {
            _ = self
                .events
                .send((session_id.to_owned(), result_id.clone(), status.clone()));
        }
        results::Raw {
            session_id: session_id.to_owned(),
            name,
            status,
            result_id,
            ..Default::default()
        }
    }

    fn finish(&self, result_id: &str, status: ResultStatus, data: Option<Vec<u8>>) {
        let session_id = {
            let mut results = self.results.lock().unwrap();
            let result = results.get_mut(result_id).unwrap();
            result.status = status.clone();
            result.data = data;
            result.session_id.clone()
        };
        _ = self.events.send((session_id, result_id.to_owned(), status));
    }

    fn data(&self, result_id: &str) -> Vec<u8> {
        self.results.lock().unwrap()[result_id]
            .data
            .clone()
            .unwrap()
    }

    /// Whether result `id` passes `filters`: session id, name prefix and
    /// status, the filters the watchers use. No filter matches everything.
    fn matches(&self, filters: &results::filter::Or, id: &str) -> bool {
        let results = self.results.lock().unwrap();
        let Some(result) = results.get(id) else {
            return false;
        };
        let matches = |field: &results::filter::Field| match (&field.field, &field.condition) {
            (results::Field::SessionId, Condition::String(filter)) => {
                filter.value == result.session_id
            }
            (results::Field::Name, Condition::String(filter)) => match filter.operator {
                FilterStringOperator::Equal => result.name == filter.value,
                FilterStringOperator::StartsWith => result.name.starts_with(&filter.value),
                ref operator => panic!("unsupported name operator {operator:?}"),
            },
            (results::Field::Status, Condition::Status(filter)) => filter.value == result.status,
            filter => panic!("unsupported filter {filter:?}"),
        };
        filters.or.is_empty() || filters.or.iter().any(|and| and.and.iter().all(matches))
    }

    /// The status of `result_id` once it is final: ArmoniK only runs a
    /// task once its data dependencies are.
    fn wait_final(&self, result_id: &str) -> ResultStatus {
        loop {
            let status = self.results.lock().unwrap()[result_id].status.clone();
            if matches!(status, ResultStatus::Completed | ResultStatus::Aborted) {
                return status;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Runs one task the way rust-tracer-worker does: a path-tracing task
    /// runs its block's chain of passes (inline instead of as submitted
    /// follow-ups), a denoising task its block's tile through
    /// [`stand_in_denoise`] instead of OIDN.
    fn execute(&self, session_id: &str, options: &TaskOptions, task: &tasks::submit::RequestItem) {
        let statuses: Vec<_> = task
            .data_dependencies
            .iter()
            .map(|id| self.wait_final(id))
            .collect();
        if statuses.contains(&ResultStatus::Aborted) {
            for output in &task.expected_output_keys {
                self.finish(output, ResultStatus::Aborted, None);
            }
            return;
        }
        if TaskType::from_options(&options.options).unwrap() == TaskType::Denoising {
            self.denoise(options, task);
            return;
        }
        match self.behavior {
            Behavior::Hang => (),
            Behavior::Abort => {
                for output in &task.expected_output_keys {
                    self.finish(output, ResultStatus::Aborted, None);
                }
            }
            Behavior::Render | Behavior::Stall => {
                let scene_id = &options.options[SCENE_ID_OPTION];
                assert!(task.data_dependencies.contains(scene_id));
                let scene = Scene::from(decode_scene(&self.data(scene_id)).unwrap());
                let convergence = Convergence::from_options(&options.options).unwrap();
                let mut payload = BlockPayload::decode(&self.data(&task.payload_id)).unwrap();
                payload.validate(&scene.camera).unwrap();
                assert_eq!(
                    task.expected_output_keys,
                    [
                        payload.outputs.linear.clone(),
                        payload.outputs.display.clone()
                    ]
                );
                loop {
                    let previous = payload
                        .previous
                        .as_ref()
                        .map(|id| BlockResult::decode(&self.data(id)).unwrap());
                    let accumulation =
                        accumulate(&scene, previous, render(&scene, &payload), &convergence)
                            .unwrap();
                    let linear = accumulation.result.encode();
                    let display = accumulation.result.to_display(&scene).encode();
                    if accumulation.done {
                        let outputs = &payload.outputs;
                        self.finish(&outputs.linear, ResultStatus::Completed, Some(linear));
                        self.finish(&outputs.display, ResultStatus::Completed, Some(display));
                        return;
                    }
                    let block = &payload.block;
                    let linear_id = self
                        .add_result(session_id, name(LINEAR_NAME, block), Some(linear))
                        .result_id;
                    self.add_result(session_id, name(DISPLAY_NAME, block), Some(display));
                    if self.behavior == Behavior::Stall {
                        return;
                    }
                    payload = payload.next(linear_id);
                }
            }
        }
    }
}

impl Fake {
    fn denoise(&self, options: &TaskOptions, task: &tasks::submit::RequestItem) {
        let scene_id = &options.options[SCENE_ID_OPTION];
        let scene = Scene::from(decode_scene(&self.data(scene_id)).unwrap());
        let payload = DenoisePayload::decode(&self.data(&task.payload_id)).unwrap();
        assert_eq!(
            task.expected_output_keys,
            [
                payload.outputs.linear.clone(),
                payload.outputs.display.clone()
            ]
        );
        let inputs: Vec<_> = payload
            .inputs
            .iter()
            .map(|input| {
                assert!(task.data_dependencies.contains(&input.result));
                BlockResult::decode(&self.data(&input.result)).unwrap()
            })
            .collect();
        let denoised = stand_in_denoise(&scene, &payload.block, &inputs);
        self.finish(
            &payload.outputs.linear,
            ResultStatus::Completed,
            Some(denoised.encode()),
        );
        self.finish(
            &payload.outputs.display,
            ResultStatus::Completed,
            Some(denoised.to_display(&scene).encode()),
        );
    }
}

/// The denoising tasks' stand-in for OIDN: a 3x3 box blur over the tile, so
/// that a block's result depends on its neighbours' pixels.
fn stand_in_denoise(scene: &Scene, block: &Block, inputs: &[BlockResult]) -> BlockResult {
    let (width, height) = (scene.camera.width, scene.camera.height);
    let mut tile = Tile::assemble(block, width, height, inputs).unwrap();
    let (w, h) = (tile.area.width as i64, tile.area.height as i64);
    let source = tile.color.clone();
    for y in 0..h {
        for x in 0..w {
            for channel in 0..3 {
                let (mut sum, mut count) = (0f32, 0f32);
                for (dx, dy) in (-1..=1).flat_map(|dx| (-1..=1).map(move |dy| (dx, dy))) {
                    let (nx, ny) = (x + dx, y + dy);
                    if (0..w).contains(&nx) && (0..h).contains(&ny) {
                        sum += source[((ny * w + nx) * 3 + channel) as usize];
                        count += 1.0;
                    }
                }
                tile.color[((y * w + x) * 3 + channel) as usize] = sum / count;
            }
        }
    }
    tile.crop(block)
}

/// One pass, as the worker renders it.
fn render(scene: &Scene, payload: &BlockPayload) -> BlockResult {
    BlockResult {
        block: payload.block,
        samples: payload.samples,
        pixels: (0..payload.block.height)
            .flat_map(|row| render_block_row(scene, payload, row))
            .collect(),
    }
}

/// Methods the client never calls.
macro_rules! unused_rpcs {
    ($($service:ident::$method:ident),* $(,)?) => {$(
        async fn $method(
            self: Arc<Self>,
            _request: $service::$method::Request,
            _context: RequestContext,
        ) -> Result<$service::$method::Response, tonic::Status> {
            Err(tonic::Status::unimplemented(stringify!($method)))
        }
    )*};
}

/// Session calls the fake only records.
macro_rules! recorded_session_calls {
    ($($method:ident),* $(,)?) => {$(
        async fn $method(
            self: Arc<Self>,
            request: sessions::$method::Request,
            _context: RequestContext,
        ) -> Result<sessions::$method::Response, tonic::Status> {
            self.call(stringify!($method));
            Ok(sessions::$method::Response {
                session: sessions::Raw {
                    session_id: request.session_id,
                    ..Default::default()
                },
            })
        }
    )*};
}

impl SessionsService for Fake {
    unused_rpcs!(sessions::list, sessions::get, sessions::stop_submission);
    recorded_session_calls!(cancel, pause, resume, close, purge, delete);

    async fn create(
        self: Arc<Self>,
        _request: sessions::create::Request,
        _context: RequestContext,
    ) -> Result<sessions::create::Response, tonic::Status> {
        self.call("create");
        Ok(sessions::create::Response {
            session_id: self.id("session"),
        })
    }
}

impl ResultsService for Fake {
    unused_rpcs!(
        results::get,
        results::get_owner_task_id,
        results::import,
        results::delete_data,
        results::get_service_configuration,
    );

    async fn list(
        self: Arc<Self>,
        request: results::list::Request,
        _context: RequestContext,
    ) -> Result<results::list::Response, tonic::Status> {
        let ids: Vec<String> = self.results.lock().unwrap().keys().cloned().collect();
        let mut found: Vec<_> = ids
            .into_iter()
            .filter(|id| self.matches(&request.filters, id))
            .map(|id| {
                let result = self.results.lock().unwrap()[&id].clone();
                results::Raw {
                    session_id: result.session_id,
                    name: result.name,
                    status: result.status,
                    result_id: id,
                    ..Default::default()
                }
            })
            .collect();
        found.sort_by(|a, b| a.result_id.cmp(&b.result_id));
        let total = found.len() as i32;
        let page: Vec<_> = found
            .into_iter()
            .skip((request.page * request.page_size) as usize)
            .take(request.page_size as usize)
            .collect();
        Ok(results::list::Response {
            results: page,
            page: request.page,
            page_size: request.page_size,
            total,
        })
    }

    async fn create_metadata(
        self: Arc<Self>,
        request: results::create_metadata::Request,
        _context: RequestContext,
    ) -> Result<results::create_metadata::Response, tonic::Status> {
        let results = request
            .results
            .into_iter()
            .map(|item| self.add_result(&request.session_id, item.name, None))
            .collect();
        Ok(results::create_metadata::Response { results })
    }

    async fn create(
        self: Arc<Self>,
        request: results::create::Request,
        _context: RequestContext,
    ) -> Result<results::create::Response, tonic::Status> {
        let results = request
            .results
            .into_iter()
            .map(|item| self.add_result(&request.session_id, item.name, Some(item.data)))
            .collect();
        Ok(results::create::Response { results })
    }

    async fn download(
        self: Arc<Self>,
        request: results::download::Request,
        _context: RequestContext,
    ) -> Result<
        impl tonic::codegen::tokio_stream::Stream<
            Item = Result<results::download::Response, tonic::Status>,
        > + Send,
        tonic::Status,
    > {
        use std::sync::atomic::Ordering;
        let name = self.results.lock().unwrap()[&request.result_id]
            .name
            .clone();
        let failing = name.starts_with(raytracer::DENOISED_DISPLAY_NAME)
            && self
                .failing_downloads
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
                .is_ok();
        if failing {
            return Err(tonic::Status::unavailable("fake ingress busy"));
        }
        let data = self.data(&request.result_id);
        // Two chunks, to exercise the reassembly.
        let (head, tail) = data.split_at(data.len() / 2);
        Ok(futures::stream::iter([head.to_vec(), tail.to_vec()].map(
            |data_chunk| Ok(results::download::Response { data_chunk }),
        )))
    }

    async fn upload(
        self: Arc<Self>,
        _request: impl tonic::codegen::tokio_stream::Stream<
            Item = Result<results::upload::Request, tonic::Status>,
        > + Send
        + 'static,
        _context: RequestContext,
    ) -> Result<results::upload::Response, tonic::Status> {
        Err(tonic::Status::unimplemented("upload"))
    }
}

impl TasksService for Fake {
    unused_rpcs!(
        tasks::list,
        tasks::list_detailed,
        tasks::get,
        tasks::cancel,
        tasks::get_result_ids,
        tasks::count_status,
    );

    /// Runs the tasks in the background, in reverse order so results do not
    /// come back in submission order.
    async fn submit(
        self: Arc<Self>,
        request: tasks::submit::Request,
        _context: RequestContext,
    ) -> Result<tasks::submit::Response, tonic::Status> {
        self.submitted_options
            .lock()
            .unwrap()
            .push(request.task_options.clone());
        let items = request
            .items
            .iter()
            .map(|item| tasks::submit::ResponseItem {
                task_id: self.id("task"),
                expected_output_ids: item.expected_output_keys.clone(),
                data_dependencies: item.data_dependencies.clone(),
                payload_id: item.payload_id.clone(),
            })
            .collect();
        let fake = self.clone();
        tokio::task::spawn_blocking(move || {
            let options = request.task_options.clone().unwrap_or_default();
            for task in request.items.iter().rev() {
                fake.execute(&request.session_id, &options, task);
            }
        });
        Ok(tasks::submit::Response { items })
    }
}

impl armonik::server::EventsService for Fake {
    async fn subscribe(
        self: Arc<Self>,
        request: events::subscribe::Request,
        _context: RequestContext,
    ) -> Result<
        impl tonic::codegen::tokio_stream::Stream<
            Item = Result<events::subscribe::Response, tonic::Status>,
        > + Send,
        tonic::Status,
    > {
        let mut receiver = self.events.subscribe();
        Ok(async_stream::stream! {
            while let Ok((session_id, result_id, status)) = receiver.recv().await {
                if session_id == request.session_id
                    && self.matches(&request.result_filters, &result_id)
                {
                    yield Ok(subscribe::Response {
                        session_id,
                        update: Update::ResultStatusUpdate(ResultStatusUpdate { result_id, status }),
                    });
                }
            }
        })
    }
}

#[derive(Parser)]
struct TransportCli {
    #[command(flatten)]
    transport: crate::transport::Transport,
}

/// Serves `fake`, and a pool of connections to it through the command-line
/// transport options.
async fn serve(fake: Arc<Fake>) -> Arc<Pool> {
    serve_with(fake, PoolConfig::default()).await
}

async fn serve_with(fake: Arc<Fake>, pool: PoolConfig) -> Arc<Pool> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(
                armonik::api::v3::sessions::sessions_server::SessionsServer::from_arc(fake.clone()),
            )
            .add_service(
                armonik::api::v3::results::results_server::ResultsServer::from_arc(fake.clone()),
            )
            .add_service(armonik::api::v3::tasks::tasks_server::TasksServer::from_arc(fake.clone()))
            .add_service(armonik::api::v3::events::events_server::EventsServer::from_arc(fake))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    let cli = TransportCli::try_parse_from([
        "test",
        "--endpoint",
        &address.to_string(),
        "--connect-timeout",
        "5s",
    ])
    .unwrap();
    Arc::new(Pool::new(cli.transport.config().unwrap(), pool))
}

fn scene_bytes() -> Vec<u8> {
    let definition = decode_scene(
        br#"{
            "camera": {"origin": [0, 0, 0], "direction": [0, 0, -1], "width": 16, "height": 12},
            "recursion_depth": 4,
            "objects": [
                {"type": "sphere", "radius": 1, "position": [0, 3, -5], "emission": [5, 5, 5]},
                {"type": "sphere", "radius": 100, "position": [0, -101, -5], "color": [0.5, 0.5, 0.5]}
            ]
        }"#,
    )
    .unwrap();
    encode_scene(&definition)
}

/// Never converges, so every block takes exactly 3 passes of 2 samples
/// (except the black ones, final after their first).
fn job() -> RenderJob {
    RenderJob {
        scene: scene_bytes(),
        width: 16,
        height: 12,
        // Uneven last column and row.
        blocks: raytracer::tile_grid(16, 12, 5, 5),
        samples: 2,
        convergence: Convergence {
            error_threshold: 0.0,
            max_samples: 6,
        },
        denoise: false,
        task_options: TaskOptions::default(),
        partitions: Vec::new(),
        denoise_partition: None,
    }
}

/// The final accumulation of `block`, replaying its passes.
fn expected_final(scene: &Scene, job: &RenderJob, block: Block) -> BlockResult {
    let convergence = job.convergence;
    let mut payload = BlockPayload {
        block,
        samples: job.samples,
        seed: SEED,
        ..Default::default()
    };
    let mut previous = None;
    loop {
        let accumulation =
            accumulate(scene, previous, render(scene, &payload), &convergence).unwrap();
        if accumulation.done {
            return accumulation.result;
        }
        previous = Some(accumulation.result);
        // The seed sequence does not depend on the accumulation's id.
        payload = payload.next(String::new());
    }
}

fn drain(events: &mut mpsc::UnboundedReceiver<RunEvent>) -> Vec<RunEvent> {
    std::iter::from_fn(|| events.try_recv().ok()).collect()
}

const SEED: u64 = 7;

/// Fails, instead of hanging, when what a test waits for never comes.
async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(30), future)
        .await
        .expect("timed out")
}

async fn renders_every_block(watcher: &dyn ResultWatcher, denoise: bool) {
    renders_every_block_through(watcher, denoise, 0, PoolConfig::default()).await;
}

/// The first `failing_downloads` downloads of final images fail
/// (transiently), and the connections are pooled as `pool` says.
async fn renders_every_block_through(
    watcher: &dyn ResultWatcher,
    denoise: bool,
    failing_downloads: usize,
    pool: PoolConfig,
) {
    let fake = Fake::new(Behavior::Render);
    fake.failing_downloads
        .store(failing_downloads, std::sync::atomic::Ordering::Relaxed);
    let pool = serve_with(fake.clone(), pool).await;
    // Denoising on a partition of its own.
    let job = RenderJob {
        denoise,
        task_options: TaskOptions {
            partition_id: "cpu".into(),
            ..Default::default()
        },
        partitions: vec!["cpu".into(), "gpu".into()],
        denoise_partition: Some("gpu".into()),
        ..job()
    };
    let (events, mut received) = mpsc::unbounded_channel();
    let (_commands, mut commands) = mpsc::unbounded_channel();

    let end = within(run(&pool, &job, watcher, SEED, &events, &mut commands)).await;
    assert_eq!(end, RunEnd::Ended);

    let received = drain(&mut received);
    assert!(matches!(&received[0], RunEvent::Reset { .. }));
    assert!(matches!(
        received.last(),
        Some(RunEvent::Finished { error: None })
    ));
    let scene = Scene::from(decode_scene(&job.scene).unwrap());
    let mut images: HashMap<Block, Vec<(u32, bool, &DisplayBlock)>> = HashMap::new();
    for event in &received {
        if let RunEvent::Block { image, done } = event {
            images
                .entry(image.block)
                .or_default()
                .push((image.samples, *done, image));
        }
    }
    assert_eq!(images.len(), job.blocks.len());
    let finals: HashMap<Block, BlockResult> = job
        .blocks
        .iter()
        .map(|block| (*block, expected_final(&scene, &job, *block)))
        .collect();
    let mut final_passes = 0;
    for block in &job.blocks {
        let shown = &images[block];
        // Better and better (a final finishing before its submission is
        // recorded shows first as progress, at the same count), ending with
        // the final image, shown once.
        let counts: Vec<_> = shown
            .iter()
            .map(|(samples, done, _)| (*samples, *done))
            .collect();
        assert!(
            counts.windows(2).all(|pair| pair[0].0 <= pair[1].0),
            "{block:?}: {counts:?}"
        );
        let (_, done, image) = shown.last().unwrap();
        assert!(*done, "{block:?}: {counts:?}");
        assert_eq!(shown.iter().filter(|(_, done, _)| *done).count(), 1);
        let expected = if denoise {
            let neighbours: Vec<_> =
                raytracer::neighborhood(&job.blocks, block, job.width, job.height)
                    .iter()
                    .map(|neighbour| finals[neighbour].clone())
                    .collect();
            stand_in_denoise(&scene, block, &neighbours)
        } else {
            finals[block].clone()
        };
        assert_eq!(**image, expected.to_display(&scene));
        // 3 passes, or 1 for the black ones.
        match finals[block].samples {
            6 => final_passes += 1,
            2 => assert!(finals[block].is_black()),
            samples => panic!("{block:?}: {samples} samples"),
        }
    }
    assert!(final_passes > 0, "the scene should not be all black");
    assert!(matches!(
        received
            .iter()
            .rev()
            .find(|e| matches!(e, RunEvent::Progress { .. })),
        Some(RunEvent::Progress {
            done: 12,
            total: 12
        })
    ));

    assert_eq!(fake.calls(), ["create", "close", "purge", "delete"]);
    // Every injected failure was hit, and recovered from.
    assert_eq!(
        fake.failing_downloads
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    let options = fake.submitted_options.lock().unwrap();
    let (rendering, denoising): (Vec<_>, Vec<_>) = options
        .iter()
        .map(|o| o.as_ref().unwrap())
        .partition(|o| TaskType::from_options(&o.options).unwrap() == TaskType::PathTracing);
    assert!(!rendering.is_empty());
    assert_eq!(denoising.is_empty(), !denoise);
    assert!(rendering.iter().chain(&denoising).all(|options| {
        options.options.contains_key(SCENE_ID_OPTION)
            && Convergence::from_options(&options.options).unwrap() == job.convergence
    }));
    assert!(
        denoising
            .iter()
            .all(|options| options.priority == raytracer::DENOISE_PRIORITY)
    );
    assert!(
        rendering
            .iter()
            .all(|options| options.partition_id == "cpu")
    );
    assert!(
        denoising
            .iter()
            .all(|options| options.partition_id == "gpu")
    );
}

#[tokio::test]
async fn renders_every_block_with_events() {
    renders_every_block(&EventsWatcher, false).await;
}

#[tokio::test]
async fn renders_every_block_with_polling() {
    renders_every_block(
        &PollingWatcher {
            interval: Duration::from_millis(10),
        },
        false,
    )
    .await;
}

#[tokio::test]
async fn denoises_every_block_with_events() {
    renders_every_block(&EventsWatcher, true).await;
}

#[tokio::test]
async fn denoises_every_block_with_polling() {
    renders_every_block(
        &PollingWatcher {
            interval: Duration::from_millis(10),
        },
        true,
    )
    .await;
}

/// Connections spent after two requests, at most two of them at a time
/// (one held by the events subscription), multiplexed or not: the run goes
/// through many reconnections and waits for connections, unharmed.
#[tokio::test]
async fn runs_survive_connection_recycling() {
    for multiplex in [false, true] {
        let pool = PoolConfig {
            pool_size: Some(2),
            requests_per_connection: Some(2),
            multiplex,
        };
        renders_every_block_through(&EventsWatcher, true, 0, pool).await;
    }
}

/// A run survives transient download failures of its final images.
#[tokio::test]
async fn runs_survive_failed_downloads() {
    renders_every_block_through(
        &EventsWatcher,
        true,
        DOWNLOAD_ATTEMPTS as usize - 1,
        PoolConfig::default(),
    )
    .await;
}

/// A download is retried until its last attempt.
#[tokio::test]
async fn downloads_retry_transient_failures() {
    let fake = Fake::new(Behavior::Render);
    let pool = serve(fake.clone()).await;
    let image = DisplayBlock {
        block: Block {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        },
        samples: 4,
        pixels: vec![[1, 2, 3]],
    };
    let result_id = fake
        .add_result(
            "session",
            name(raytracer::DENOISED_DISPLAY_NAME, &image.block),
            Some(image.encode()),
        )
        .result_id;
    let finished = Finished::Final {
        result_id,
        block: image.block,
        status: ResultStatus::Completed,
    };
    let set_failures = |n: u32| {
        fake.failing_downloads
            .store(n as usize, std::sync::atomic::Ordering::Relaxed)
    };

    set_failures(DOWNLOAD_ATTEMPTS - 1);
    let started = std::time::Instant::now();
    let (_, downloaded) = within(download(&pool, "session", finished.clone())).await;
    assert_eq!(downloaded.unwrap(), image);
    // One delay per failed attempt.
    assert!(started.elapsed() >= Duration::from_millis(500) * (DOWNLOAD_ATTEMPTS - 1));

    set_failures(DOWNLOAD_ATTEMPTS);
    let (_, downloaded) = within(download(&pool, "session", finished)).await;
    let error = format!("{:#}", downloaded.unwrap_err());
    assert!(error.contains("could not download"), "{error}");
}

/// Blocks still refining show their intermediate images (whose first pass
/// is all they get here).
#[tokio::test]
async fn intermediate_images_show_progress() {
    let fake = Fake::new(Behavior::Stall);
    let pool = serve(fake.clone()).await;
    let job = job();
    let (events, mut received) = mpsc::unbounded_channel();
    let (sender, mut commands) = mpsc::unbounded_channel();
    let scene = Scene::from(decode_scene(&job.scene).unwrap());

    let watch = async {
        let mut first_passes = HashMap::new();
        while first_passes.len() < job.blocks.len() {
            if let Some(RunEvent::Block { image, done }) = received.recv().await {
                // Intermediate, except black blocks: final after one pass.
                assert!(!done || image.pixels.iter().all(|pixel| *pixel == [0; 3]));
                first_passes.insert(image.block, image);
            }
        }
        sender.send(Command::Quit).unwrap();
        first_passes
    };
    let (end, first_passes) = within(async {
        tokio::join!(
            run(&pool, &job, &EventsWatcher, SEED, &events, &mut commands),
            watch
        )
    })
    .await;
    assert_eq!(end, RunEnd::Interrupted(Command::Quit));
    for block in &job.blocks {
        let payload = BlockPayload {
            block: *block,
            samples: job.samples,
            seed: SEED,
            ..Default::default()
        };
        assert_eq!(
            first_passes[block],
            render(&scene, &payload).to_display(&scene)
        );
    }
    assert_eq!(fake.calls(), ["create", "cancel", "purge", "delete"]);
}

#[tokio::test]
async fn aborted_result_fails_and_cancels_the_session() {
    let fake = Fake::new(Behavior::Abort);
    let pool = serve(fake.clone()).await;
    let (events, mut received) = mpsc::unbounded_channel();
    let (_commands, mut commands) = mpsc::unbounded_channel();

    let job = RenderJob {
        denoise: true,
        ..job()
    };
    let end = within(run(
        &pool,
        &job,
        &EventsWatcher,
        SEED,
        &events,
        &mut commands,
    ))
    .await;
    assert_eq!(end, RunEnd::Ended);
    match drain(&mut received).last() {
        Some(RunEvent::Finished { error: Some(error) }) => {
            assert!(error.contains("aborted"), "{error}")
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(fake.calls(), ["create", "cancel", "purge", "delete"]);
}

#[tokio::test]
async fn commands_pause_and_interrupt_the_run() {
    let fake = Fake::new(Behavior::Hang);
    let pool = serve(fake.clone()).await;
    let (events, mut received) = mpsc::unbounded_channel();
    let (sender, mut commands) = mpsc::unbounded_channel();
    for command in [Command::TogglePause, Command::TogglePause, Command::Quit] {
        sender.send(command).unwrap();
    }

    let end = within(run(
        &pool,
        &job(),
        &EventsWatcher,
        SEED,
        &events,
        &mut commands,
    ))
    .await;
    assert_eq!(end, RunEnd::Interrupted(Command::Quit));
    assert_eq!(
        fake.calls(),
        ["create", "pause", "resume", "cancel", "purge", "delete"]
    );
    // An interrupted run does not report an outcome.
    assert!(
        !drain(&mut received)
            .iter()
            .any(|e| matches!(e, RunEvent::Finished { .. }))
    );
}

#[tokio::test]
async fn demo_loop_reruns_until_quit() {
    let fake = Fake::new(Behavior::Render);
    let pool = serve(fake.clone()).await;
    let (events, mut received) = mpsc::unbounded_channel();
    let (sender, commands) = mpsc::unbounded_channel();
    let demo = tokio::spawn(demo_loop(
        pool,
        job(),
        || Ok(job()),
        Box::new(EventsWatcher),
        Some(Duration::from_millis(10)),
        events,
        commands,
    ));

    let mut finished = 0;
    while finished < 2 {
        if let Some(RunEvent::Finished { error }) = within(received.recv()).await {
            assert_eq!(error, None);
            finished += 1;
        }
    }
    sender.send(Command::Quit).unwrap();
    within(demo).await.unwrap();
    let calls = fake.calls();
    assert!(
        calls.iter().filter(|c| *c == "create").count() >= 2,
        "{calls:?}"
    );
}

/// Each new run renders the reloaded scene, or the previous one when the
/// reload fails.
#[tokio::test]
async fn demo_loop_reloads_the_scene_for_each_run() {
    let fake = Fake::new(Behavior::Render);
    let pool = serve(fake.clone()).await;
    let (events, mut received) = mpsc::unbounded_channel();
    let (sender, commands) = mpsc::unbounded_channel();
    let smaller = || {
        let mut definition = decode_scene(&scene_bytes()).unwrap();
        definition.camera.width = 8;
        definition.camera.height = 6;
        RenderJob {
            scene: encode_scene(&definition),
            width: 8,
            height: 6,
            blocks: raytracer::tile_grid(8, 6, 5, 5),
            ..job()
        }
    };
    let mut reloads = vec![Ok(smaller()), Err(eyre::eyre!("bad scene"))];
    let demo = tokio::spawn(demo_loop(
        pool,
        job(),
        move || reloads.pop().unwrap_or_else(|| Ok(job())),
        Box::new(EventsWatcher),
        Some(Duration::from_millis(10)),
        events,
        commands,
    ));

    // Run 1 (initial), 2 (reload failed: same scene), 3 (smaller scene).
    let mut sizes = Vec::new();
    let mut reloads = Vec::new();
    let mut finished = 0;
    while finished < 3 {
        match within(received.recv()).await.unwrap() {
            RunEvent::Reset { width, height, .. } => sizes.push((width, height)),
            RunEvent::SceneReload { error } => reloads.push(error),
            RunEvent::Finished { error } => {
                assert_eq!(error, None);
                finished += 1;
            }
            _ => (),
        }
    }
    sender.send(Command::Quit).unwrap();
    within(demo).await.unwrap();
    assert_eq!(sizes, [(16, 12), (16, 12), (8, 6)]);
    assert_eq!(reloads, [Some("bad scene".to_owned()), None]);
}

/// Results finished before the subscription come from the catch-up
/// listing; both it and the events keep to the name prefix.
#[tokio::test]
async fn events_watcher_catches_up_and_filters_by_name() {
    let fake = Fake::new(Behavior::Render);
    let pool = serve(fake.clone()).await;
    fake.add_result("session", "payload-early".into(), Some(vec![1]));
    let early = fake.add_result("session", "display-early".into(), Some(vec![1]));
    fake.add_result("other", "display-other".into(), Some(vec![1]));
    let unwanted = fake.add_result("session", "linear-late".into(), None);
    let late = fake.add_result("session", "display-late".into(), None);

    let mut updates = EventsWatcher.watch(pool, "session".into(), "display-".into());
    let first = within(updates.next()).await.unwrap();
    assert_eq!(first.result_id, early.result_id);
    assert_eq!(first.status, ResultStatus::Completed);

    fake.finish(&unwanted.result_id, ResultStatus::Completed, Some(vec![2]));
    fake.finish(&late.result_id, ResultStatus::Completed, Some(vec![2]));
    let second = within(updates.next()).await.unwrap();
    assert_eq!(second.result_id, late.result_id);
}

fn block(x: u32) -> Block {
    Block {
        x,
        y: 0,
        width: 1,
        height: 1,
    }
}

fn update(result_id: &str, status: ResultStatus) -> ResultUpdate {
    ResultUpdate {
        result_id: result_id.into(),
        status,
    }
}

fn final_(result_id: &str, x: u32, status: ResultStatus) -> Option<Finished> {
    Some(Finished::Final {
        result_id: result_id.into(),
        block: block(x),
        status,
    })
}

#[test]
fn tracker_sorts_finals_from_progress_in_either_order() {
    let mut tracker = Tracker::default();
    // Submitted, then finished.
    assert_eq!(tracker.expect("a".into(), block(0)), None);
    assert_eq!(
        tracker.update(update("a", ResultStatus::Completed)),
        final_("a", 0, ResultStatus::Completed)
    );
    // An unknown image is progress, once.
    let progress = Some(Finished::Progress {
        result_id: "p".into(),
    });
    assert_eq!(
        tracker.update(update("p", ResultStatus::Completed)),
        progress
    );
    assert_eq!(tracker.update(update("p", ResultStatus::Completed)), None);
    // A final finishing before its submission is recorded: shown as
    // progress, then claimed as final.
    assert_eq!(
        tracker.update(update("b", ResultStatus::Completed)),
        Some(Finished::Progress {
            result_id: "b".into()
        })
    );
    assert_eq!(
        tracker.expect("b".into(), block(1)),
        final_("b", 1, ResultStatus::Completed)
    );
    // An early abort is not progress, but still fails its block.
    assert_eq!(tracker.update(update("c", ResultStatus::Aborted)), None);
    assert_eq!(
        tracker.expect("c".into(), block(2)),
        final_("c", 2, ResultStatus::Aborted)
    );
    // Finals are reported once.
    assert_eq!(tracker.update(update("a", ResultStatus::Completed)), None);
    assert_eq!(tracker.update(update("b", ResultStatus::Completed)), None);
    assert!(tracker.pending.is_empty());
}
