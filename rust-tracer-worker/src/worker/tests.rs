//! End-to-end `Process` calls against a fake agent, with the data folder in
//! a temporary directory.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use armonik::agent::{
    create_results, create_results_metadata, create_tasks, get_common_data, get_direct_data,
    get_resource_data, notify_result_data, submit_tasks,
};
use armonik::api::v3::agent::agent_server::AgentServer;
use armonik::reexports::tokio_stream::wrappers::{TcpListenerStream, UnixListenerStream};
use armonik::reexports::{tokio, tonic};
use armonik::server::{AgentService, RequestContext, WorkerService};
use armonik::{Output, TaskOptions, worker};
use raytracer::{
    Block, BlockOutputs, BlockPayload, BlockResult, Color, Convergence, DenoiseInput,
    DenoisePayload, DisplayBlock, Scene, SceneDefinition, TaskType, Tile, decode_scene,
    encode_scene, render_block_row, tile_grid,
};

use super::*;

/// Records the calls it receives; answers the result/task creations with
/// made-up ids.
#[derive(Debug, Default)]
struct FakeAgent {
    notified: Mutex<Vec<notify_result_data::Request>>,
    created_metadata: Mutex<Vec<create_results_metadata::Request>>,
    created: Mutex<Vec<create_results::Request>>,
    submitted: Mutex<Vec<submit_tasks::Request>>,
}

impl AgentService for FakeAgent {
    async fn create_results_metadata(
        self: Arc<Self>,
        request: create_results_metadata::Request,
        _context: RequestContext,
    ) -> Result<create_results_metadata::Response, tonic::Status> {
        let results = request
            .results
            .iter()
            .map(|result| armonik::agent::ResultMetaData {
                session_id: request.session_id.clone(),
                result_id: format!("id-{}", result.name),
                name: result.name.clone(),
                ..Default::default()
            })
            .collect();
        self.created_metadata.lock().unwrap().push(request);
        Ok(create_results_metadata::Response {
            communication_token: String::new(),
            results,
        })
    }

    async fn create_results(
        self: Arc<Self>,
        request: create_results::Request,
        _context: RequestContext,
    ) -> Result<create_results::Response, tonic::Status> {
        let results = request
            .results
            .iter()
            .map(|result| armonik::agent::ResultMetaData {
                session_id: request.session_id.clone(),
                result_id: format!("id-{}", result.name),
                name: result.name.clone(),
                ..Default::default()
            })
            .collect();
        self.created.lock().unwrap().push(request);
        Ok(create_results::Response {
            communication_token: String::new(),
            results,
        })
    }

    async fn notify_result_data(
        self: Arc<Self>,
        request: notify_result_data::Request,
        _context: RequestContext,
    ) -> Result<notify_result_data::Response, tonic::Status> {
        let result_ids = request.result_ids.clone();
        self.notified.lock().unwrap().push(request);
        Ok(notify_result_data::Response { result_ids })
    }

    async fn submit_tasks(
        self: Arc<Self>,
        request: submit_tasks::Request,
        _context: RequestContext,
    ) -> Result<submit_tasks::Response, tonic::Status> {
        let items = request
            .items
            .iter()
            .enumerate()
            .map(|(i, item)| submit_tasks::ResponseItem {
                task_id: format!("task-{i}"),
                expected_output_ids: item.expected_output_keys.clone(),
                data_dependencies: item.data_dependencies.clone(),
                payload_id: item.payload_id.clone(),
            })
            .collect();
        self.submitted.lock().unwrap().push(request);
        Ok(submit_tasks::Response {
            communication_token: String::new(),
            items,
        })
    }

    async fn get_resource_data(
        self: Arc<Self>,
        _request: get_resource_data::Request,
        _context: RequestContext,
    ) -> Result<get_resource_data::Response, tonic::Status> {
        Err(tonic::Status::unimplemented("fake agent"))
    }

    async fn get_common_data(
        self: Arc<Self>,
        _request: get_common_data::Request,
        _context: RequestContext,
    ) -> Result<get_common_data::Response, tonic::Status> {
        Err(tonic::Status::unimplemented("fake agent"))
    }

    async fn get_direct_data(
        self: Arc<Self>,
        _request: get_direct_data::Request,
        _context: RequestContext,
    ) -> Result<get_direct_data::Response, tonic::Status> {
        Err(tonic::Status::unimplemented("fake agent"))
    }

    async fn create_tasks(
        self: Arc<Self>,
        _request: impl tonic::codegen::tokio_stream::Stream<
            Item = Result<create_tasks::Request, tonic::Status>,
        > + Send
        + 'static,
        _context: RequestContext,
    ) -> Result<create_tasks::Response, tonic::Status> {
        Err(tonic::Status::unimplemented("fake agent"))
    }
}

struct Harness {
    agent: Arc<FakeAgent>,
    worker: Arc<Worker>,
    folder: tempfile::TempDir,
}

impl Harness {
    /// With the agent on TCP.
    async fn new() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        Self::serve(
            TcpListenerStream::new(listener),
            crate::GrpcChannel {
                address: format!("http://{address}"),
                socket_type: crate::SocketType::Tcp,
            },
            tempfile::tempdir().unwrap(),
        )
    }

    /// With the agent on a Unix socket in the data folder.
    fn with_unix_socket() -> Self {
        let folder = tempfile::tempdir().unwrap();
        let path = path_string(&folder.path().join("agent.sock"));
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        Self::serve(
            UnixListenerStream::new(listener),
            crate::GrpcChannel {
                address: path,
                socket_type: crate::SocketType::UnixDomainSocket,
            },
            folder,
        )
    }

    fn serve<S, IO, E>(incoming: S, channel: crate::GrpcChannel, folder: tempfile::TempDir) -> Self
    where
        S: tonic::codegen::tokio_stream::Stream<Item = Result<IO, E>> + Send + 'static,
        IO: tonic::transport::server::Connected
            + tokio::io::AsyncRead
            + tokio::io::AsyncWrite
            + Unpin
            + Send
            + 'static,
        E: Into<Box<dyn std::error::Error + Send + Sync>> + Send + 'static,
    {
        let agent = Arc::new(FakeAgent::default());
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(AgentServer::from_arc(agent.clone()))
                .serve_with_incoming(incoming),
        );
        Self {
            agent,
            worker: Arc::new(Worker::new(channel).unwrap()),
            folder,
        }
    }

    fn write(&self, id: &str, data: impl AsRef<[u8]>) {
        std::fs::write(self.folder.path().join(id), data).unwrap();
    }

    fn read(&self, id: &str) -> Vec<u8> {
        std::fs::read(self.folder.path().join(id)).unwrap()
    }

    /// A request rendering `payload_id` with the scene `scene_id`, expecting
    /// the block's final results, with the given convergence options.
    fn request(&self, scene_id: &str, payload_id: &str) -> worker::process::Request {
        let mut options = HashMap::from([(SCENE_ID_OPTION.into(), scene_id.into())]);
        Convergence::default().to_options(&mut options);
        worker::process::Request {
            communication_token: "token".into(),
            session_id: "session".into(),
            task_id: "task".into(),
            task_options: TaskOptions {
                options,
                ..Default::default()
            },
            expected_output_keys: vec![FINAL_LINEAR.into(), FINAL_DISPLAY.into()],
            payload_id: payload_id.into(),
            data_dependencies: vec![scene_id.into()],
            data_folder: path_string(self.folder.path()),
            configuration: Default::default(),
        }
    }

    async fn process(&self, request: worker::process::Request) -> Output {
        self.worker
            .clone()
            .process(request, RequestContext::default())
            .await
            .unwrap()
            .output
    }

    fn notified(&self) -> Vec<notify_result_data::Request> {
        self.agent.notified.lock().unwrap().clone()
    }

    /// The `create_results` calls: (name, data) of every created result.
    fn created(&self) -> Vec<(String, Vec<u8>)> {
        self.agent
            .created
            .lock()
            .unwrap()
            .iter()
            .flat_map(|request| &request.results)
            .map(|result| (result.name.clone(), result.data.to_vec()))
            .collect()
    }

    fn submitted(&self) -> Vec<submit_tasks::Request> {
        self.agent.submitted.lock().unwrap().clone()
    }
}

fn path_string(path: &Path) -> String {
    path.to_str().unwrap().to_owned()
}

const FINAL_LINEAR: &str = "final-linear";
const FINAL_DISPLAY: &str = "final-display";

fn scene_definition() -> SceneDefinition {
    decode_scene(
        br#"{
            "camera": {"origin": [0, 0, 0], "direction": [0, 0, -1], "width": 16, "height": 12},
            "samples": 1000,
            "recursion_depth": 5,
            "objects": [
                {"type": "sphere", "radius": 1, "position": [0, 3, -5], "emission": [5, 5, 5]},
                {"type": "sphere", "radius": 100, "position": [0, -101, -5], "color": [0.5, 0.5, 0.5]}
            ]
        }"#,
    )
    .unwrap()
}

const BLOCK: Block = Block {
    x: 3,
    y: 2,
    width: 5,
    height: 7,
};

fn payload() -> BlockPayload {
    BlockPayload {
        block: BLOCK,
        samples: 3,
        seed: 42,
        outputs: BlockOutputs {
            linear: FINAL_LINEAR.into(),
            display: FINAL_DISPLAY.into(),
        },
        previous: None,
    }
}

/// The pass `payload` renders, sequentially.
fn pass(payload: &BlockPayload) -> BlockResult {
    let scene = Scene::from(scene_definition());
    BlockResult {
        block: payload.block,
        samples: payload.samples,
        pixels: (0..payload.block.height)
            .flat_map(|row| render_block_row(&scene, payload, row))
            .collect(),
    }
}

/// A harness with the scene (`scene`) and the first pass's payload
/// (`payload`) in its data folder.
async fn harness() -> Harness {
    let harness = Harness::new().await;
    harness.write("scene", encode_scene(&scene_definition()));
    harness.write("payload", payload().encode());
    harness
}

/// A request for a later pass over the block, starting from `previous`
/// (written as `accumulation`), with the given convergence.
fn later_pass(
    harness: &Harness,
    previous: &BlockResult,
    convergence: Convergence,
) -> (worker::process::Request, BlockPayload) {
    harness.write("accumulation", previous.encode());
    let next = payload().next("accumulation".into());
    harness.write("next", next.encode());
    let mut request = harness.request("scene", "next");
    request.data_dependencies.push("accumulation".into());
    convergence.to_options(&mut request.task_options.options);
    (request, next)
}

fn assert_finals(harness: &Harness, expected: &BlockResult) {
    let linear = BlockResult::decode(&harness.read(FINAL_LINEAR)).unwrap();
    assert_eq!(&linear, expected);
    let display = DisplayBlock::decode(&harness.read(FINAL_DISPLAY)).unwrap();
    assert_eq!(
        display,
        expected.to_display(&Scene::from(scene_definition()))
    );
    let notified: Vec<_> = harness
        .notified()
        .into_iter()
        .flat_map(|request| request.result_ids)
        .collect();
    assert_eq!(notified, [FINAL_LINEAR, FINAL_DISPLAY]);
    assert!(harness.submitted().is_empty());
}

#[tokio::test]
async fn a_first_pass_delegates_to_a_follow_up() {
    let harness = harness().await;
    assert_eq!(
        harness.process(harness.request("scene", "payload")).await,
        Output::Ok
    );

    // The accumulation and its display image, then the follow-up payload.
    let created = harness.created();
    let names: Vec<_> = created.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, ["linear-3-2", "display-3-2", "payload-3-2"]);
    // Rows rendered in parallel land in order, and match a sequential
    // render sample for sample.
    let expected = pass(&payload());
    assert!(expected.pixels.iter().any(|p| p.r > 0.0));
    assert_eq!(BlockResult::decode(&created[0].1).unwrap(), expected);
    assert_eq!(
        DisplayBlock::decode(&created[1].1).unwrap(),
        expected.to_display(&Scene::from(scene_definition()))
    );
    let next = BlockPayload::decode(&created[2].1).unwrap();
    assert_eq!(next, payload().next("id-linear-3-2".into()));

    let submitted = harness.submitted();
    let [submission] = submitted.as_slice() else {
        panic!("{submitted:?}");
    };
    assert_eq!(submission.communication_token, "token");
    let [task] = submission.items.as_slice() else {
        panic!("{submission:?}");
    };
    assert_eq!(task.expected_output_keys, [FINAL_LINEAR, FINAL_DISPLAY]);
    assert_eq!(task.data_dependencies, ["scene", "id-linear-3-2"]);
    assert_eq!(task.payload_id, "id-payload-3-2");
    let options = task.task_options.as_ref().unwrap();
    assert_eq!(options.priority, FOLLOW_UP_PRIORITY);
    // The session-wide options carry over.
    assert_eq!(
        options.options,
        harness.request("scene", "payload").task_options.options
    );
    // Nothing final yet.
    assert_eq!(harness.notified(), []);
}

#[tokio::test]
async fn a_black_block_is_final_after_one_pass() {
    let harness = Harness::new().await;
    let mut empty = scene_definition();
    empty.objects.clear();
    harness.write("scene", encode_scene(&empty));
    harness.write("payload", payload().encode());

    assert_eq!(
        harness.process(harness.request("scene", "payload")).await,
        Output::Ok
    );
    let black = BlockResult {
        block: BLOCK,
        samples: 3,
        pixels: vec![Default::default(); BLOCK.len()],
    };
    assert_finals(&harness, &black);
}

#[tokio::test]
async fn a_converged_pass_sends_both_final_results() {
    let harness = harness().await;
    let previous = pass(&payload());
    let (request, next) = later_pass(
        &harness,
        &previous,
        Convergence {
            error_threshold: f32::MAX,
            ..Default::default()
        },
    );
    assert_eq!(harness.process(request).await, Output::Ok);
    assert_finals(&harness, &previous.merge(&pass(&next)));
}

#[tokio::test]
async fn an_unconverged_pass_delegates_again() {
    let harness = harness().await;
    let previous = pass(&payload());
    let (request, next) = later_pass(
        &harness,
        &previous,
        Convergence {
            error_threshold: 0.0,
            ..Default::default()
        },
    );
    assert_eq!(harness.process(request).await, Output::Ok);

    let created = harness.created();
    let merged = BlockResult::decode(&created[0].1).unwrap();
    assert_eq!(merged, previous.merge(&pass(&next)));
    assert_eq!(merged.samples, 6);
    let follow_up = BlockPayload::decode(&created[2].1).unwrap();
    assert_eq!(follow_up, next.next("id-linear-3-2".into()));
    assert_eq!(harness.notified(), []);
}

#[tokio::test]
async fn the_sample_cap_ends_the_block() {
    let harness = harness().await;
    let previous = pass(&payload());
    let (request, next) = later_pass(
        &harness,
        &previous,
        Convergence {
            error_threshold: 0.0,
            max_samples: 6,
        },
    );
    assert_eq!(harness.process(request).await, Output::Ok);
    assert_finals(&harness, &previous.merge(&pass(&next)));
}

#[tokio::test]
async fn agent_on_unix_socket() {
    let harness = Harness::with_unix_socket();
    harness.write("scene", encode_scene(&scene_definition()));
    harness.write("payload", payload().encode());

    let output = harness.process(harness.request("scene", "payload")).await;
    assert_eq!(output, Output::Ok);
    assert_eq!(harness.submitted().len(), 1);
}

#[tokio::test]
async fn scene_is_parsed_once_per_scene_id() {
    let harness = harness().await;
    assert_eq!(
        harness.process(harness.request("scene", "payload")).await,
        Output::Ok
    );

    // Served from the cache: the file is not read again.
    std::fs::remove_file(harness.folder.path().join("scene")).unwrap();
    assert_eq!(
        harness.process(harness.request("scene", "payload")).await,
        Output::Ok
    );

    // Another scene id is read, even when the cached one is still around.
    let mut smaller = scene_definition();
    smaller.camera.width = 4;
    harness.write("other", encode_scene(&smaller));
    assert!(matches!(
        harness.process(harness.request("other", "payload")).await,
        Output::Error { details } if details.contains("exceeds the 4x12 image")
    ));
}

#[tokio::test]
async fn failures_are_reported_in_the_output() {
    let harness = harness().await;
    harness.write(
        "no-samples",
        BlockPayload {
            samples: 0,
            ..payload()
        }
        .encode(),
    );
    harness.write("garbage", "not json");
    harness.write("next", payload().next("accumulation".into()).encode());
    let elsewhere = BlockResult {
        block: Block { x: 0, ..BLOCK },
        ..pass(&payload())
    };
    harness.write("accumulation", elsewhere.encode());

    type Edit = Box<dyn Fn(&mut worker::process::Request)>;
    let cases: [(&str, Edit); 11] = [
        (
            "could not read payload",
            Box::new(|r| r.payload_id = "missing".into()),
        ),
        (
            "invalid payload",
            Box::new(|r| r.payload_id = "garbage".into()),
        ),
        (
            "at least one sample",
            Box::new(|r| r.payload_id = "no-samples".into()),
        ),
        (
            "sceneId task option is missing",
            Box::new(|r| {
                r.task_options.options.remove(SCENE_ID_OPTION);
            }),
        ),
        (
            "not a data dependency",
            Box::new(|r| r.data_dependencies.clear()),
        ),
        (
            "invalid scene",
            Box::new(|r| {
                r.task_options
                    .options
                    .insert(SCENE_ID_OPTION.into(), "garbage".into());
                r.data_dependencies = vec!["garbage".into()];
            }),
        ),
        (
            "names",
            Box::new(|r| r.expected_output_keys.push("other".into())),
        ),
        (
            "names",
            Box::new(|r| r.expected_output_keys = vec![FINAL_LINEAR.into()]),
        ),
        (
            "invalid task option",
            Box::new(|r| {
                r.task_options
                    .options
                    .insert(raytracer::MAX_SAMPLES_OPTION.into(), "many".into());
            }),
        ),
        // The previous accumulation must be declared.
        (
            "accumulation is not a data dependency",
            Box::new(|r| r.payload_id = "next".into()),
        ),
        (
            "expected a result for block",
            Box::new(|r| {
                r.payload_id = "next".into();
                r.data_dependencies.push("accumulation".into());
            }),
        ),
    ];
    for (message, edit) in cases {
        let mut request = harness.request("scene", "payload");
        edit(&mut request);
        match harness.process(request).await {
            Output::Error { details } => assert!(details.contains(message), "{message}: {details}"),
            Output::Ok => panic!("{message}: task succeeded"),
        }
    }
    assert_eq!(harness.notified(), []);
    assert!(harness.submitted().is_empty());
}

/// Noisy blocks of a uniform 0.4 grey over the scene's 16x12 image: the
/// 96-pixel halo makes any block's tile the whole image.
fn noisy_grid() -> Vec<BlockResult> {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::SmallRng::seed_from_u64(3);
    tile_grid(16, 12, 8, 6)
        .into_iter()
        .map(|block| BlockResult {
            block,
            samples: 10 + block.x,
            pixels: (0..block.len())
                .map(|_| {
                    let mut value = || 0.4 + rng.random_range(-0.2..0.2);
                    Color {
                        r: value(),
                        g: value(),
                        b: value(),
                    }
                })
                .collect(),
        })
        .collect()
}

const DENOISED_LINEAR: &str = "denoised-linear";
const DENOISED_DISPLAY: &str = "denoised-display";

/// A denoising request for `block` over `inputs`, written to the harness.
fn denoise_request(
    harness: &Harness,
    block: Block,
    inputs: &[BlockResult],
) -> worker::process::Request {
    let inputs: Vec<_> = inputs
        .iter()
        .map(|input| {
            let id = format!("input-{}-{}", input.block.x, input.block.y);
            harness.write(&id, input.encode());
            DenoiseInput {
                block: input.block,
                result: id,
            }
        })
        .collect();
    let payload = DenoisePayload {
        block,
        inputs: inputs.clone(),
        outputs: BlockOutputs {
            linear: DENOISED_LINEAR.into(),
            display: DENOISED_DISPLAY.into(),
        },
    };
    harness.write("denoise", payload.encode());
    let mut request = harness.request("scene", "denoise");
    TaskType::Denoising.to_options(&mut request.task_options.options);
    request.expected_output_keys = vec![DENOISED_LINEAR.into(), DENOISED_DISPLAY.into()];
    request
        .data_dependencies
        .extend(inputs.into_iter().map(|input| input.result));
    request
}

#[tokio::test]
async fn a_denoising_task_denoises_its_block_from_the_neighbourhood() {
    let harness = harness().await;
    let inputs = noisy_grid();
    let block = inputs[3].block;
    assert_eq!(
        harness
            .process(denoise_request(&harness, block, &inputs))
            .await,
        Output::Ok
    );

    let denoised = BlockResult::decode(&harness.read(DENOISED_LINEAR)).unwrap();
    assert_eq!(denoised.block, block);
    assert_eq!(denoised.samples, inputs[3].samples);
    // OIDN over the assembled tile, cropped.
    let mut tile = Tile::assemble(&block, 16, 12, &inputs).unwrap();
    raytracer::Denoiser::default()
        .denoise(&mut tile.color, 16, 12)
        .unwrap();
    assert_eq!(denoised, tile.crop(&block));
    let error = |result: &BlockResult| {
        result
            .pixels
            .iter()
            .flat_map(|p| [p.r, p.g, p.b])
            .map(|c| ((c - 0.4) as f64).powi(2))
            .sum::<f64>()
    };
    let (before, after) = (error(&inputs[3]), error(&denoised));
    println!("squared error {before} -> {after}");
    assert!(after < before / 4.0, "{before} -> {after}");

    let display = DisplayBlock::decode(&harness.read(DENOISED_DISPLAY)).unwrap();
    assert_eq!(
        display,
        denoised.to_display(&Scene::from(scene_definition()))
    );
    let notified: Vec<_> = harness
        .notified()
        .into_iter()
        .flat_map(|request| request.result_ids)
        .collect();
    assert_eq!(notified, [DENOISED_LINEAR, DENOISED_DISPLAY]);
}

#[tokio::test]
async fn denoising_failures_are_reported_in_the_output() {
    let harness = harness().await;
    let inputs = noisy_grid();
    let block = inputs[3].block;
    // Each case is processed right away: they share the data folder.
    let expect_error = async |message: &str, request| match harness.process(request).await {
        Output::Error { details } => assert!(details.contains(message), "{message}: {details}"),
        Output::Ok => panic!("{message}: task succeeded"),
    };

    let mut request = denoise_request(&harness, block, &inputs);
    request.data_dependencies.pop();
    expect_error("is not a data dependency", request).await;

    let request = denoise_request(&harness, block, &inputs);
    // The input declared as block (0, 0) holds another one.
    let other = BlockResult {
        block: inputs[1].block,
        ..inputs[0].clone()
    };
    harness.write("input-0-0", other.encode());
    expect_error("holds block", request).await;

    let request = denoise_request(&harness, block, &inputs[1..]);
    expect_error("do not cover the denoising tile", request).await;

    let mut request = denoise_request(&harness, block, &inputs);
    request.expected_output_keys.pop();
    expect_error("names", request).await;

    let mut request = denoise_request(&harness, block, &inputs);
    request
        .task_options
        .options
        .insert(raytracer::TASK_TYPE_OPTION.into(), "sharpening".into());
    expect_error("invalid task option", request).await;

    assert_eq!(harness.notified(), []);
}

#[tokio::test]
async fn dropping_the_blocking_future_cancels_the_work() {
    let (observed, cancellation) = std::sync::mpsc::channel();
    let work = spawn_blocking_cancellable(move |cancelled| {
        // Bounded, so a regression fails the test instead of hanging it: the
        // runtime waits for its blocking tasks when the test ends.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !cancelled.load(Ordering::Relaxed) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        observed.send(cancelled.load(Ordering::Relaxed)).unwrap();
    });
    // The timeout drops the future while the work is still spinning.
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), work)
            .await
            .is_err()
    );
    assert!(
        cancellation.recv().unwrap(),
        "the blocking work never saw the cancellation"
    );
}

#[test]
fn render_stops_once_cancelled() {
    let scene = Scene::from(scene_definition());
    let cancelled = AtomicBool::new(true);
    let err = render_block(&scene, &payload(), &cancelled).unwrap_err();
    assert!(err.to_string().contains("cancelled"), "{err}");
}

#[tokio::test]
async fn task_handler_forwards_agent_calls() {
    let harness = Harness::new().await;
    harness.write("payload", "data");
    harness.write("dependency", "dependency data");
    let mut request = harness.request("dependency", "payload");
    request.task_options.priority = 7;
    let mut handler = TaskHandler::new(request, harness.worker.agent.clone())
        .await
        .unwrap();

    assert_eq!(handler.payload(), b"data");
    assert_eq!(handler.session_id(), "session");
    assert_eq!(handler.task_id(), "task");
    assert_eq!(handler.task_options().priority, 7);
    assert_eq!(handler.expected_results(), [FINAL_LINEAR, FINAL_DISPLAY]);
    assert_eq!(
        handler.data_dependency("dependency").await.unwrap(),
        b"dependency data"
    );
    assert!(handler.data_dependency("payload").await.is_err());

    let created = handler
        .create_results([("next", b"payload".to_vec())])
        .await
        .unwrap();
    assert_eq!(created[0].result_id, "id-next");
    let created = handler.create_results_metadata(["out"]).await.unwrap();
    assert_eq!(created[0].result_id, "id-out");
    let submitted = handler
        .submit_tasks(
            Some(handler.task_options().clone()),
            [submit_tasks::RequestItem {
                expected_output_keys: vec!["id-out".into()],
                data_dependencies: vec!["dependency".into()],
                payload_id: "id-next".into(),
                task_options: None,
            }],
        )
        .await
        .unwrap();
    assert_eq!(submitted[0].payload_id, "id-next");

    let agent = &harness.agent;
    let created = &agent.created.lock().unwrap()[0];
    assert_eq!(
        (
            created.communication_token.as_str(),
            created.session_id.as_str()
        ),
        ("token", "session")
    );
    assert_eq!(&created.results[0].data[..], b"payload");
    let submitted = &agent.submitted.lock().unwrap()[0];
    assert_eq!(submitted.communication_token, "token");
    assert_eq!(submitted.task_options.as_ref().unwrap().priority, 7);
    assert_eq!(
        agent.created_metadata.lock().unwrap()[0].results[0].name,
        "out"
    );
}
