use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use armonik::client::Agent;
use armonik::reexports::tonic::transport::{Channel, Endpoint, Uri};
use armonik::reexports::{tokio, tonic};
use armonik::{Output, server, worker};
use eyre::{Context, OptionExt, ensure};
use hyper_util::rt::TokioIo;
use rayon::prelude::*;
use raytracer::{
    BlockOutputs, BlockPayload, BlockResult, Color, Convergence, DISPLAY_NAME, DenoisePayload,
    Denoiser, LINEAR_NAME, PAYLOAD_NAME, SCENE_ID_OPTION, Scene, TaskType, Tile, accumulate,
    decode_scene, name, render_block_row, validate_denoise,
};
use rust_tracer_worker::task_handler::TaskHandler;

/// Priority of the follow-up passes, above the client's first passes (1):
/// blocks already started finish before new ones start, so the finished
/// area grows from the center as a disc (the C# worker's heuristic).
pub const FOLLOW_UP_PRIORITY: i32 = 8;

pub struct Worker {
    agent: Agent<Channel>,
    /// Last scene used, by result id: consecutive tasks of a session share
    /// their scene, so it is parsed once per worker rather than per task.
    scene: Mutex<Option<(String, Arc<Scene>)>>,
    denoiser: Arc<Denoiser>,
}

impl std::fmt::Debug for Worker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Worker").finish_non_exhaustive()
    }
}

impl Worker {
    /// The agent connection is lazy: established on the first request, so
    /// the worker can start before the agent listens.
    pub fn new(channel_config: crate::GrpcChannel) -> Result<Self, eyre::Report> {
        let channel = match channel_config.socket_type {
            crate::SocketType::Tcp => Endpoint::from_shared(channel_config.address)?.connect_lazy(),
            crate::SocketType::UnixDomainSocket => {
                let path = channel_config.address;
                // The URI is required by the API but unused: the connector
                // ignores it.
                Endpoint::from_static("http://localhost").connect_with_connector_lazy(
                    tower::service_fn(move |_: Uri| {
                        let path = path.clone();
                        async move {
                            Ok::<_, std::io::Error>(TokioIo::new(
                                tokio::net::UnixStream::connect(path).await?,
                            ))
                        }
                    }),
                )
            }
        };

        Ok(Self {
            agent: Agent::with_channel(channel),
            scene: Mutex::default(),
            denoiser: Arc::default(),
        })
    }

    /// Renders one pass over the task's block and merges it into the
    /// previous ones; sends the block's final results once done, or else
    /// delegates them to a follow-up pass.
    async fn render_task(&self, mut handler: TaskHandler) -> eyre::Result<()> {
        let (scene_id, scene) = self.task_scene(&handler).await?;
        let payload = BlockPayload::decode(handler.payload()).wrap_err("invalid payload")?;
        payload
            .validate(&scene.camera)
            .wrap_err("invalid payload")?;
        check_outputs(&handler, &payload.outputs)?;
        let convergence = Convergence::from_options(&handler.task_options().options)?;
        let previous = match &payload.previous {
            Some(previous) => Some(
                BlockResult::decode(&handler.data_dependency(previous).await?)
                    .wrap_err_with(|| format!("invalid accumulation {previous}"))?,
            ),
            None => None,
        };

        let pass = spawn_blocking_cancellable({
            let (scene, payload) = (scene.clone(), payload.clone());
            move |cancelled| render_block(&scene, &payload, cancelled)
        })
        .await??;
        let accumulation = accumulate(
            &scene,
            previous,
            BlockResult {
                block: payload.block,
                samples: payload.samples,
                pixels: pass,
            },
            &convergence,
        )?;
        tracing::debug!(
            block = ?payload.block,
            samples = accumulation.result.samples,
            error = ?accumulation.error,
            done = accumulation.done,
            "pass rendered"
        );

        let linear = accumulation.result.encode();
        let display = accumulation.result.to_display(&scene).encode();
        if accumulation.done {
            handler.send_result(&payload.outputs.linear, linear).await?;
            handler.send_result(&payload.outputs.display, display).await
        } else {
            delegate(&mut handler, &scene_id, &payload, linear, display).await
        }
    }

    /// Denoises the task's block from the final linear results of its
    /// neighbourhood, and sends the block's denoised results.
    async fn denoise_task(&self, mut handler: TaskHandler) -> eyre::Result<()> {
        let (_, scene) = self.task_scene(&handler).await?;
        let payload = DenoisePayload::decode(handler.payload()).wrap_err("invalid payload")?;
        validate_denoise(&payload, &scene.camera).wrap_err("invalid payload")?;
        check_outputs(&handler, &payload.outputs)?;
        let mut inputs = Vec::with_capacity(payload.inputs.len());
        for input in &payload.inputs {
            let result = BlockResult::decode(&handler.data_dependency(&input.result).await?)
                .wrap_err_with(|| format!("invalid input {}", input.result))?;
            ensure!(
                result.block == input.block,
                "input {} holds block {:?}, expected {:?}",
                input.result,
                result.block,
                input.block
            );
            inputs.push(result);
        }

        let block = payload.block;
        let (width, height) = (scene.camera.width, scene.camera.height);
        let denoiser = self.denoiser.clone();
        let denoised = spawn_blocking_cancellable(move |_| -> eyre::Result<BlockResult> {
            let mut tile = Tile::assemble(&block, width, height, &inputs)?;
            let area = tile.area;
            denoiser.denoise(&mut tile.color, area.width as usize, area.height as usize)?;
            Ok(tile.crop(&block))
        })
        .await??;
        tracing::debug!(?block, "block denoised");

        handler
            .send_result(&payload.outputs.linear, denoised.encode())
            .await?;
        handler
            .send_result(
                &payload.outputs.display,
                denoised.to_display(&scene).encode(),
            )
            .await
    }

    /// The task's scene, named by its `sceneId` option, with that id.
    async fn task_scene(&self, handler: &TaskHandler) -> eyre::Result<(String, Arc<Scene>)> {
        let scene_id = handler
            .task_options()
            .options
            .get(SCENE_ID_OPTION)
            .ok_or_eyre("the sceneId task option is missing")?
            .clone();
        let scene = self.scene(handler, &scene_id).await?;
        Ok((scene_id, scene))
    }

    /// The scene `scene_id`, from the cache or else parsed from the task's
    /// data dependency.
    async fn scene(&self, handler: &TaskHandler, scene_id: &str) -> eyre::Result<Arc<Scene>> {
        // Checked even on a cache hit: a task that does not declare its
        // scene would fail on any worker that has not cached it.
        ensure!(
            handler.data_dependencies().iter().any(|id| id == scene_id),
            "scene {scene_id} is not a data dependency of the task"
        );

        if let Some((id, scene)) = &*self.scene.lock().unwrap_or_else(PoisonError::into_inner)
            && id == scene_id
        {
            return Ok(scene.clone());
        }

        let definition = handler.data_dependency(scene_id).await?;
        let scene = Arc::new(Scene::from(
            decode_scene(&definition).wrap_err_with(|| format!("invalid scene {scene_id}"))?,
        ));
        tracing::info!(scene_id, "changing scene");
        *self.scene.lock().unwrap_or_else(PoisonError::into_inner) =
            Some((scene_id.to_owned(), scene.clone()));
        Ok(scene)
    }
}

/// Checks that the task is expected to produce exactly the results its
/// payload names.
fn check_outputs(handler: &TaskHandler, outputs: &BlockOutputs) -> eyre::Result<()> {
    let mut expected = handler.expected_results().to_vec();
    expected.sort();
    let mut named = [&outputs.linear, &outputs.display];
    named.sort();
    ensure!(
        expected.iter().eq(named),
        "the task expects {expected:?}, but its payload names {outputs:?} as its results",
    );
    Ok(())
}

/// Hands the block over to a follow-up pass: creates the accumulation
/// `linear` (the follow-up's input) and its `display` image (for the client
/// to follow the progress), then submits the follow-up, expecting the
/// block's final results in place of this task.
async fn delegate(
    handler: &mut TaskHandler,
    scene_id: &str,
    payload: &BlockPayload,
    linear: Vec<u8>,
    display: Vec<u8>,
) -> eyre::Result<()> {
    let block = &payload.block;
    let linear_name = name(LINEAR_NAME, block);
    let linear_id = handler
        .create_results([
            (linear_name.clone(), linear),
            (name(DISPLAY_NAME, block), display),
        ])
        .await?
        .into_iter()
        .find(|result| result.name == linear_name)
        .ok_or_eyre("the accumulation was not created")?
        .result_id;
    let next = payload.next(linear_id.clone());
    let payload_id = handler
        .create_results([(name(PAYLOAD_NAME, block), next.encode())])
        .await?
        .pop()
        .ok_or_eyre("the follow-up payload was not created")?
        .result_id;
    let mut task_options = handler.task_options().clone();
    task_options.priority = FOLLOW_UP_PRIORITY;
    handler
        .submit_tasks(
            None,
            [armonik::agent::submit_tasks::RequestItem {
                expected_output_keys: vec![
                    payload.outputs.linear.clone(),
                    payload.outputs.display.clone(),
                ],
                data_dependencies: vec![scene_id.to_owned(), linear_id],
                payload_id,
                task_options: Some(task_options),
            }],
        )
        .await?;
    Ok(())
}

/// Runs `f` on the blocking thread pool. Dropping the returned future
/// (which is how a cancelled `Process` call is cancelled) raises the flag
/// passed to `f`: unlike async code, blocking code is not stopped by the drop
/// itself, so `f` must poll the flag to give up early.
async fn spawn_blocking_cancellable<T: Send + 'static>(
    f: impl FnOnce(&AtomicBool) -> T + Send + 'static,
) -> eyre::Result<T> {
    struct CancelOnDrop(Arc<AtomicBool>);

    impl Drop for CancelOnDrop {
        fn drop(&mut self) {
            // Harmless after completion: nothing polls the flag any more.
            self.0.store(true, Ordering::Relaxed);
        }
    }

    let guard = CancelOnDrop(Arc::default());
    let cancelled = guard.0.clone();
    Ok(tokio::task::spawn_blocking(move || f(&cancelled)).await?)
}

/// Renders the rows of the block in parallel, giving up (between rows) once
/// `cancelled` is raised.
fn render_block(
    scene: &Scene,
    payload: &BlockPayload,
    cancelled: &AtomicBool,
) -> eyre::Result<Vec<Color>> {
    let rows = (0..payload.block.height)
        .into_par_iter()
        .map(|row| {
            (!cancelled.load(Ordering::Relaxed)).then(|| render_block_row(scene, payload, row))
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_eyre("task cancelled")?;
    Ok(rows.concat())
}

impl server::WorkerService for Worker {
    async fn health_check(
        self: Arc<Self>,
        _request: worker::health_check::Request,
        _context: server::RequestContext,
    ) -> std::result::Result<worker::health_check::Response, tonic::Status> {
        Ok(worker::health_check::Response::Serving)
    }

    /// Task failures are reported in the output (so the agent can retry the
    /// task), never as gRPC errors.
    async fn process(
        self: Arc<Self>,
        request: worker::process::Request,
        _context: server::RequestContext,
    ) -> std::result::Result<worker::process::Response, tonic::Status> {
        let span = tracing::info_span!(
            "task",
            session_id = request.session_id,
            task_id = request.task_id
        );
        let result = tracing::Instrument::instrument(
            async {
                let handler = TaskHandler::new(request, self.agent.clone()).await?;
                match TaskType::from_options(&handler.task_options().options)? {
                    TaskType::PathTracing => self.render_task(handler).await,
                    TaskType::Denoising => self.denoise_task(handler).await,
                }
            },
            span.clone(),
        )
        .await;

        let output = match result {
            Ok(()) => Output::Ok,
            Err(err) => {
                span.in_scope(|| tracing::error!("task failed: {err:?}"));
                Output::Error {
                    details: format!("{err:?}"),
                }
            }
        };
        Ok(worker::process::Response { output })
    }
}

#[cfg(test)]
mod tests;
