use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use armonik::client::Agent;
use armonik::reexports::tonic::transport::{Channel, Endpoint, Uri};
use armonik::reexports::{tokio, tonic};
use armonik::{Output, server, worker};
use eyre::{Context, OptionExt, bail, ensure};
use hyper_util::rt::TokioIo;
use rayon::prelude::*;
use raytracer::{BlockPayload, BlockResult, Color, Scene, decode_scene, render_block_row};
use rust_tracer_worker::task_handler::TaskHandler;

/// Task option naming the result that holds the session's scene.
pub const SCENE_ID_OPTION: &str = "sceneId";

pub struct Worker {
    agent: Agent<Channel>,
    /// Last scene used, by result id: consecutive tasks of a session share
    /// their scene, so it is parsed once per worker rather than per task.
    scene: Mutex<Option<(String, Arc<Scene>)>>,
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
        })
    }

    /// Renders the task's block and sends it as the task's single result.
    async fn render_task(&self, mut handler: TaskHandler) -> eyre::Result<()> {
        let [result_id] = handler.expected_results() else {
            bail!(
                "expected a single result to produce, got {}",
                handler.expected_results().len()
            );
        };
        let result_id = result_id.clone();
        let scene = self.scene(&handler).await?;
        let payload = BlockPayload::decode(handler.payload()).wrap_err("invalid payload")?;
        payload
            .validate(&scene.camera)
            .wrap_err("invalid payload")?;
        tracing::debug!(?payload, "rendering block");

        let pixels =
            spawn_blocking_cancellable(move |cancelled| render_block(&scene, &payload, cancelled))
                .await??;

        let result = BlockResult {
            block: payload.block,
            samples: payload.samples,
            pixels,
        };
        handler.send_result(&result_id, result.encode()).await
    }

    /// The scene named by the task's `sceneId` option, from the cache or
    /// else parsed from the task's data dependency.
    async fn scene(&self, handler: &TaskHandler) -> eyre::Result<Arc<Scene>> {
        let scene_id = handler
            .task_options()
            .options
            .get(SCENE_ID_OPTION)
            .ok_or_eyre("the sceneId task option is missing")?;
        // Checked even on a cache hit: a task that does not declare its
        // scene would fail on any worker that has not cached it.
        ensure!(
            handler.data_dependencies().contains(scene_id),
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
            Some((scene_id.clone(), scene.clone()));
        Ok(scene)
    }
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
                self.render_task(handler).await
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
