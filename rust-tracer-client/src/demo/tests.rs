//! Runs against an in-process fake of the ArmoniK control plane, whose
//! tasks render their block with the lib like rust-tracer-worker does.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use armonik::events::{ResultStatusUpdate, Update, subscribe};
use armonik::reexports::tokio_stream::wrappers::TcpListenerStream;
use armonik::reexports::tonic;
use armonik::results::filter::Condition;
use armonik::server::{RequestContext, ResultsService, SessionsService, TasksService};
use armonik::{events, results, sessions, tasks};
use clap::Parser;
use raytracer::{Scene, decode_scene, encode_scene, render_block_row};
use tokio::sync::{broadcast, mpsc};

use super::*;
use crate::watcher::{EventsWatcher, PollingWatcher};

/// What the fake's tasks do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Behavior {
    Render,
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

    /// Runs one task the way rust-tracer-worker does.
    fn execute(&self, options: &TaskOptions, task: &tasks::submit::RequestItem) {
        let output = &task.expected_output_keys[0];
        match self.behavior {
            Behavior::Hang => (),
            Behavior::Abort => self.finish(output, ResultStatus::Aborted, None),
            Behavior::Render => {
                let scene_id = &options.options[SCENE_ID_OPTION];
                assert!(task.data_dependencies.contains(scene_id));
                let scene = Scene::from(decode_scene(&self.data(scene_id)).unwrap());
                let payload = BlockPayload::decode(&self.data(&task.payload_id)).unwrap();
                payload.validate(&scene.camera).unwrap();
                let result = BlockResult {
                    block: payload.block,
                    samples: payload.samples,
                    pixels: (0..payload.block.height)
                        .flat_map(|row| render_block_row(&scene, &payload, row))
                        .collect(),
                };
                self.finish(output, ResultStatus::Completed, Some(result.encode()));
            }
        }
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

    /// Supports the filters the watchers use: session id and status
    /// equality.
    async fn list(
        self: Arc<Self>,
        request: results::list::Request,
        _context: RequestContext,
    ) -> Result<results::list::Response, tonic::Status> {
        let matches = |id: &str, result: &FakeResult, field: &results::filter::Field| match (
            &field.field,
            &field.condition,
        ) {
            (results::Field::SessionId, Condition::String(filter)) => {
                filter.value == result.session_id
            }
            (results::Field::Status, Condition::Status(filter)) => filter.value == result.status,
            (results::Field::ResultId, Condition::String(filter)) => filter.value == id,
            filter => panic!("unsupported filter {filter:?}"),
        };
        let mut found: Vec<_> = self
            .results
            .lock()
            .unwrap()
            .iter()
            .filter(|(id, result)| {
                request
                    .filters
                    .or
                    .iter()
                    .any(|and| and.and.iter().all(|field| matches(id, result, field)))
            })
            .map(|(id, result)| results::Raw {
                session_id: result.session_id.clone(),
                name: result.name.clone(),
                status: result.status.clone(),
                result_id: id.clone(),
                ..Default::default()
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
            let options = request.task_options.unwrap_or_default();
            for task in request.items.iter().rev() {
                fake.execute(&options, task);
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
                if session_id == request.session_id {
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

/// Serves `fake` and connects a client to it through the command-line
/// transport options.
async fn serve(fake: Arc<Fake>) -> Client {
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
    Client::with_config(cli.transport.config().unwrap())
        .await
        .unwrap()
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

fn job() -> RenderJob {
    RenderJob {
        scene: scene_bytes(),
        // Uneven last column and row.
        blocks: crate::blocks::tile(16, 12, 5, 5),
        samples: 2,
        task_options: TaskOptions::default(),
        partitions: Vec::new(),
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

async fn renders_every_block(watcher: &dyn ResultWatcher) {
    let fake = Fake::new(Behavior::Render);
    let client = serve(fake.clone()).await;
    let job = job();
    let (events, mut received) = mpsc::unbounded_channel();
    let (_commands, mut commands) = mpsc::unbounded_channel();

    let end = within(run(&client, &job, watcher, SEED, &events, &mut commands)).await;
    assert_eq!(end, RunEnd::Ended);

    let received = drain(&mut received);
    assert!(matches!(&received[0], RunEvent::Reset { .. }));
    assert!(matches!(
        received.last(),
        Some(RunEvent::Finished { error: None })
    ));
    let scene = Scene::from(decode_scene(&job.scene).unwrap());
    let mut blocks: Vec<_> = received
        .iter()
        .filter_map(|event| match event {
            RunEvent::Block(result) => Some(result),
            _ => None,
        })
        .collect();
    blocks.sort_by_key(|result| (result.block.y, result.block.x));
    let mut expected = job.blocks.clone();
    expected.sort_by_key(|block| (block.y, block.x));
    assert_eq!(blocks.len(), expected.len());
    for (result, block) in blocks.into_iter().zip(expected) {
        let payload = BlockPayload {
            block,
            samples: job.samples,
            seed: SEED,
        };
        let pixels: Vec<_> = (0..block.height)
            .flat_map(|row| render_block_row(&scene, &payload, row))
            .collect();
        assert_eq!(result.block, block);
        assert_eq!(result.samples, job.samples);
        assert_eq!(result.pixels, pixels);
    }
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
    let options = fake.submitted_options.lock().unwrap();
    assert!(
        options
            .iter()
            .all(|o| o.as_ref().unwrap().options.contains_key(SCENE_ID_OPTION))
    );
}

#[tokio::test]
async fn renders_every_block_with_events() {
    renders_every_block(&EventsWatcher).await;
}

#[tokio::test]
async fn renders_every_block_with_polling() {
    renders_every_block(&PollingWatcher {
        interval: Duration::from_millis(10),
    })
    .await;
}

#[tokio::test]
async fn aborted_result_fails_and_cancels_the_session() {
    let fake = Fake::new(Behavior::Abort);
    let client = serve(fake.clone()).await;
    let (events, mut received) = mpsc::unbounded_channel();
    let (_commands, mut commands) = mpsc::unbounded_channel();

    let end = within(run(
        &client,
        &job(),
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
    let client = serve(fake.clone()).await;
    let (events, mut received) = mpsc::unbounded_channel();
    let (sender, mut commands) = mpsc::unbounded_channel();
    for command in [Command::TogglePause, Command::TogglePause, Command::Quit] {
        sender.send(command).unwrap();
    }

    let end = within(run(
        &client,
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
    let client = serve(fake.clone()).await;
    let (events, mut received) = mpsc::unbounded_channel();
    let (sender, commands) = mpsc::unbounded_channel();
    let demo = tokio::spawn(demo_loop(
        client,
        job(),
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

#[tokio::test]
async fn events_watcher_catches_up_on_earlier_results() {
    let fake = Fake::new(Behavior::Render);
    let client = serve(fake.clone()).await;
    let early = fake.add_result("session", "early".into(), Some(vec![1]));
    let late = fake.add_result("session", "late".into(), None);
    fake.add_result("other", "other".into(), Some(vec![1]));

    let mut updates = EventsWatcher.watch(client, "session".into());
    let first = within(updates.next()).await.unwrap();
    assert_eq!(first.result_id, early.result_id);
    assert_eq!(first.status, ResultStatus::Completed);

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

#[test]
fn tracker_matches_finishes_in_either_order() {
    let mut tracker = Tracker::default();
    // Submitted, then finished.
    assert_eq!(tracker.expect("a".into(), block(0)), None);
    assert_eq!(
        tracker.update(update("a", ResultStatus::Completed)),
        Some(("a".into(), block(0), ResultStatus::Completed))
    );
    // Finished before its submission was recorded.
    assert_eq!(tracker.update(update("b", ResultStatus::Aborted)), None);
    assert_eq!(
        tracker.expect("b".into(), block(1)),
        Some(("b".into(), block(1), ResultStatus::Aborted))
    );
    // Duplicates and foreign results are dropped.
    assert_eq!(tracker.update(update("a", ResultStatus::Completed)), None);
    assert_eq!(tracker.update(update("b", ResultStatus::Aborted)), None);
    assert_eq!(
        tracker.update(update("payload", ResultStatus::Completed)),
        None
    );
    assert!(tracker.pending.is_empty());
}
