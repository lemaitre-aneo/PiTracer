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
    Block, BlockPayload, BlockResult, Scene, SceneDefinition, decode_scene, encode_scene,
    render_block_row,
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

    /// A request rendering `payload_id` with the scene `scene_id` into
    /// `result`.
    fn request(&self, scene_id: &str, payload_id: &str) -> worker::process::Request {
        worker::process::Request {
            communication_token: "token".into(),
            session_id: "session".into(),
            task_id: "task".into(),
            task_options: TaskOptions {
                options: HashMap::from([(SCENE_ID_OPTION.into(), scene_id.into())]),
                ..Default::default()
            },
            expected_output_keys: vec!["result".into()],
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
}

fn path_string(path: &Path) -> String {
    path.to_str().unwrap().to_owned()
}

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

const PAYLOAD: BlockPayload = BlockPayload {
    block: Block {
        x: 3,
        y: 2,
        width: 5,
        height: 7,
    },
    samples: 3,
    seed: 42,
};

#[tokio::test]
async fn renders_block_and_sends_result() {
    let harness = Harness::new().await;
    harness.write("scene", encode_scene(&scene_definition()));
    harness.write("payload", PAYLOAD.encode());

    let output = harness.process(harness.request("scene", "payload")).await;
    assert_eq!(output, Output::Ok);

    let result = BlockResult::decode(&harness.read("result")).unwrap();
    assert_eq!(result.block, PAYLOAD.block);
    assert_eq!(result.samples, PAYLOAD.samples);
    // Rows rendered in parallel land in order, and match a sequential
    // render sample for sample.
    let scene = Scene::from(scene_definition());
    let expected: Vec<_> = (0..PAYLOAD.block.height)
        .flat_map(|row| render_block_row(&scene, &PAYLOAD, row))
        .collect();
    assert_eq!(result.pixels, expected);
    assert!(result.pixels.iter().any(|p| p.r > 0.0));

    assert_eq!(
        harness.notified(),
        [notify_result_data::Request {
            communication_token: "token".into(),
            session_id: "session".into(),
            result_ids: vec!["result".into()],
        }]
    );
}

#[tokio::test]
async fn agent_on_unix_socket() {
    let harness = Harness::with_unix_socket();
    harness.write("scene", encode_scene(&scene_definition()));
    harness.write("payload", PAYLOAD.encode());

    let output = harness.process(harness.request("scene", "payload")).await;
    assert_eq!(output, Output::Ok);
    assert_eq!(harness.notified()[0].result_ids, ["result"]);
}

#[tokio::test]
async fn scene_is_parsed_once_per_scene_id() {
    let harness = Harness::new().await;
    harness.write("scene", encode_scene(&scene_definition()));
    harness.write("payload", PAYLOAD.encode());
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
    let harness = Harness::new().await;
    harness.write("scene", encode_scene(&scene_definition()));
    harness.write("payload", PAYLOAD.encode());
    harness.write(
        "no-samples",
        BlockPayload {
            samples: 0,
            ..PAYLOAD
        }
        .encode(),
    );
    harness.write("garbage", "not json");

    type Edit = Box<dyn Fn(&mut worker::process::Request)>;
    let cases: [(&str, Edit); 7] = [
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
            Box::new(|r| r.task_options.options.clear()),
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
            "expected a single result",
            Box::new(|r| r.expected_output_keys.push("other".into())),
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
    let err = render_block(&scene, &PAYLOAD, &cancelled).unwrap_err();
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
    assert_eq!(handler.expected_results(), ["result"]);
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
