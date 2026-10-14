//! Rendering an image on ArmoniK, one session per run: upload the scene,
//! submit the first pass of each block, then show each block's display
//! images as they come: the intermediate ones while its passes refine it
//! (found by name, see `raytracer::task`), and the final one, which the run
//! waits for.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use armonik::{ResultStatus, TaskOptions};
use eyre::{Context, OptionExt, bail, ensure};
use futures::stream::FuturesUnordered;
use futures::{StreamExt, TryStreamExt};
use raytracer::{
    Block, BlockOutputs, BlockPayload, Convergence, DENOISE_PAYLOAD_NAME, DENOISE_PRIORITY,
    DENOISED_DISPLAY_NAME, DENOISED_LINEAR_NAME, DISPLAY_NAME, DenoiseInput, DenoisePayload,
    DisplayBlock, FINAL_DISPLAY_NAME, FINAL_LINEAR_NAME, PAYLOAD_NAME, SCENE_ID_OPTION, TaskType,
    name, neighborhood,
};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::pool::Pool;
use crate::watcher::{ResultUpdate, ResultWatcher};

/// Everything a run needs.
pub struct RenderJob {
    /// Encoded scene definition, shared by all the tasks of a session.
    pub scene: Vec<u8>,
    /// Image size.
    pub width: u32,
    pub height: u32,
    /// Blocks to render, in submission order.
    pub blocks: Vec<Block>,
    /// Samples per pixel of each pass.
    pub samples: u32,
    pub convergence: Convergence,
    /// Denoise each block from its neighbourhood once final: the denoised
    /// display images are then the ones the run waits for (the converged
    /// ones show as progress meanwhile).
    pub denoise: bool,
    /// Default options of the session's tasks (the scene and the
    /// convergence settings are added at submission).
    pub task_options: TaskOptions,
    /// Partitions the session may use (empty: the default partition).
    pub partitions: Vec<String>,
    /// Partition of the denoising tasks, if not that of `task_options`
    /// (workers with a GPU, say).
    pub denoise_partition: Option<String>,
}

/// What a run reports to the display.
#[derive(Debug)]
pub enum RunEvent {
    /// A new run starts: clear the image, which is `width` x `height`.
    Reset {
        session_id: String,
        width: u32,
        height: u32,
    },
    /// The scene was reloaded for the coming run, or `error` tells why it
    /// could not be (the previous scene is used again).
    SceneReload { error: Option<String> },
    /// A display image of a block: an intermediate one, or the final one
    /// when `done`.
    Block { image: DisplayBlock, done: bool },
    /// `done` of the `total` blocks are displayed.
    Progress { done: usize, total: usize },
    /// The run ended; `error` tells why it did not complete.
    Finished { error: Option<String> },
}

/// What the display asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Pause the running session, or resume it.
    TogglePause,
    /// Start a new run (abandoning the current one, if any).
    Rerun,
    Quit,
}

/// How a run ended, as far as the demo loop is concerned.
#[derive(Debug, PartialEq, Eq)]
pub enum RunEnd {
    /// Completed or failed (already reported): wait before the next run.
    Ended,
    /// Interrupted by a command.
    Interrupted(Command),
}

/// Requests per batch when creating results and submitting tasks.
const BATCH_SIZE: usize = 100;
/// Downloads in flight at once.
const DOWNLOADS: usize = 16;
const DOWNLOAD_ATTEMPTS: u32 = 3;
/// Bound on the session cleanup, so an unreachable control plane cannot
/// hold the shutdown hostage.
const CLEANUP_TIMEOUT: Duration = Duration::from_secs(10);

/// Runs until a command quits, starting a new run `rerun_delay` after each
/// one ends (or only on [`Command::Rerun`] without a delay). Every run after
/// the first renders `reload()`, or the previous job again if it fails.
pub async fn demo_loop(
    pool: Arc<Pool>,
    mut job: RenderJob,
    mut reload: impl FnMut() -> eyre::Result<RenderJob> + Send,
    watcher: Box<dyn ResultWatcher>,
    rerun_delay: Option<Duration>,
    events: UnboundedSender<RunEvent>,
    mut commands: UnboundedReceiver<Command>,
) {
    loop {
        match run(
            &pool,
            &job,
            &*watcher,
            rand::random(),
            &events,
            &mut commands,
        )
        .await
        {
            RunEnd::Interrupted(Command::Quit) => return,
            RunEnd::Interrupted(_) => (),
            RunEnd::Ended => {
                if !wait_for_rerun(rerun_delay, &mut commands).await {
                    return;
                }
            }
        }
        match reload() {
            Ok(reloaded) => {
                job = reloaded;
                _ = events.send(RunEvent::SceneReload { error: None });
            }
            Err(err) => {
                tracing::warn!("could not reload the scene, rendering the previous one: {err:#}");
                _ = events.send(RunEvent::SceneReload {
                    error: Some(format!("{err:#}")),
                });
            }
        }
    }
}

/// Waits until the next run is due; `false` to quit instead.
async fn wait_for_rerun(
    rerun_delay: Option<Duration>,
    commands: &mut UnboundedReceiver<Command>,
) -> bool {
    if let Some(delay) = rerun_delay {
        tracing::info!("next run in {delay:?} (R to start it now, Esc to quit)");
    } else {
        tracing::info!("press R to render again, Esc to quit");
    }
    let delay = async {
        match rerun_delay {
            Some(delay) => tokio::time::sleep(delay).await,
            None => std::future::pending().await,
        }
    };
    let mut delay = std::pin::pin!(delay);
    loop {
        tokio::select! {
            () = &mut delay => return true,
            command = commands.recv() => match command {
                Some(Command::Rerun) => return true,
                Some(Command::Quit) | None => return false,
                // Nothing to pause between runs.
                Some(Command::TogglePause) => (),
            },
        }
    }
}

/// One run, in its own session, which it cleans up whatever happens: a
/// failure is reported (log and [`RunEvent::Finished`]), not returned.
pub async fn run(
    pool: &Arc<Pool>,
    job: &RenderJob,
    watcher: &dyn ResultWatcher,
    seed: u64,
    events: &UnboundedSender<RunEvent>,
    commands: &mut UnboundedReceiver<Command>,
) -> RunEnd {
    let session = async {
        let session_id = pool
            .sessions()
            .await?
            .create(job.partitions.clone(), job.task_options.clone())
            .await?;
        Ok::<_, eyre::Report>(session_id)
    };
    let session_id = match session.await {
        Ok(session_id) => session_id,
        Err(err) => {
            let error = format!("could not create a session: {err:#}");
            tracing::error!("{error}");
            _ = events.send(RunEvent::Finished { error: Some(error) });
            return RunEnd::Ended;
        }
    };
    tracing::info!(session_id, "rendering {} blocks", job.blocks.len());
    _ = events.send(RunEvent::Reset {
        session_id: session_id.clone(),
        width: job.width,
        height: job.height,
    });

    // Scoped so the render (its watcher and downloads) is dropped before the
    // session is torn down.
    let outcome = {
        let render = render(pool, &session_id, job, watcher, seed, events);
        let mut render = std::pin::pin!(render);
        let mut paused = false;
        loop {
            tokio::select! {
                result = &mut render => break result.map(|()| RunEnd::Ended),
                command = commands.recv() => match command {
                    Some(Command::TogglePause) => {
                        paused = !paused;
                        toggle_pause(pool, &session_id, paused).await;
                    }
                    Some(command) => break Ok(RunEnd::Interrupted(command)),
                    None => break Ok(RunEnd::Interrupted(Command::Quit)),
                },
            }
        }
    };

    let completed = matches!(outcome, Ok(RunEnd::Ended));
    let error = match &outcome {
        Ok(_) => None,
        Err(err) => {
            tracing::error!(session_id, "render failed: {err:?}");
            Some(format!("{err}"))
        }
    };
    if tokio::time::timeout(CLEANUP_TIMEOUT, cleanup(pool, &session_id, completed))
        .await
        .is_err()
    {
        tracing::warn!(session_id, "session cleanup timed out");
    }
    if !matches!(outcome, Ok(RunEnd::Interrupted(_))) {
        _ = events.send(RunEvent::Finished { error });
    }
    outcome.unwrap_or(RunEnd::Ended)
}

async fn toggle_pause(pool: &Pool, session_id: &str, pause: bool) {
    let result = async {
        let mut sessions = pool.sessions().await?;
        if pause {
            sessions.pause(session_id).await?;
        } else {
            sessions.resume(session_id).await?;
        }
        Ok::<_, eyre::Report>(())
    };
    match result.await {
        Ok(()) => tracing::info!(session_id, "{}", if pause { "paused" } else { "resumed" }),
        Err(err) => tracing::warn!(session_id, "could not pause or resume: {err:#}"),
    }
}

/// Closes a completed session, cancels an abandoned one, then purges and
/// deletes it (its results are no longer needed).
async fn cleanup(pool: &Pool, session_id: &str, completed: bool) {
    for step in [
        if completed { "close" } else { "cancel" },
        "purge",
        "delete",
    ] {
        // One pooled connection per request.
        let result = async {
            let mut sessions = pool.sessions().await?;
            match step {
                "close" => sessions.close(session_id).await?,
                "cancel" => sessions.cancel(session_id).await?,
                "purge" => sessions.purge(session_id).await?,
                _ => sessions.delete(session_id).await?,
            };
            Ok::<_, eyre::Report>(())
        };
        if let Err(err) = result.await {
            tracing::warn!(session_id, "could not {step} the session: {err:#}");
        }
    }
}

/// Submits the job's tasks and forwards the display images until every
/// block is final. Submission and downloads overlap, so the first blocks
/// show up while later ones are still being submitted.
async fn render(
    pool: &Arc<Pool>,
    session_id: &str,
    job: &RenderJob,
    watcher: &dyn ResultWatcher,
    seed: u64,
    events: &UnboundedSender<RunEvent>,
) -> eyre::Result<()> {
    let (submitted, mut submissions) = tokio::sync::mpsc::unbounded_channel();
    let submit = async move { submit(pool, session_id, job, seed, submitted).await };
    let collect = async {
        let total = job.blocks.len();
        let blocks: HashSet<Block> = job.blocks.iter().copied().collect();
        let mut updates = watcher.watch(
            pool.clone(),
            session_id.to_owned(),
            format!("{DISPLAY_NAME}-"),
        );
        let mut tracker = Tracker::default();
        let mut queued = VecDeque::new();
        let mut downloads = FuturesUnordered::new();
        // Sample count of the image shown for each block (`u32::MAX` once
        // final), so a late download never replaces a better image.
        let mut shown: HashMap<Block, u32> = HashMap::new();
        let mut submitting = true;
        let mut done = 0;
        while done < total {
            while downloads.len() < DOWNLOADS
                && let Some(finished) = queued.pop_front()
            {
                downloads.push(download(pool, session_id, finished));
            }
            let finished = tokio::select! {
                submission = submissions.recv(), if submitting => match submission {
                    Some((result_id, block)) => tracker.expect(result_id, block),
                    None => {
                        submitting = false;
                        None
                    }
                },
                Some(update) = updates.next() => tracker.update(update),
                Some((finished, image)) = downloads.next() => {
                    match (finished, image) {
                        (Finished::Final { result_id, block, .. }, image) => {
                            let image = image?;
                            ensure!(
                                image.block == block,
                                "result {result_id} holds block {:?}, expected {block:?}",
                                image.block
                            );
                            shown.insert(block, u32::MAX);
                            done += 1;
                            _ = events.send(RunEvent::Block { image, done: true });
                            _ = events.send(RunEvent::Progress { done, total });
                        }
                        (Finished::Progress { result_id }, Ok(image)) => {
                            let shown = shown.entry(image.block).or_default();
                            if !blocks.contains(&image.block) {
                                tracing::warn!("result {result_id} holds unknown block {:?}", image.block);
                            } else if image.samples > *shown {
                                *shown = image.samples;
                                _ = events.send(RunEvent::Block { image, done: false });
                            }
                        }
                        // Progress is best effort.
                        (Finished::Progress { result_id }, Err(err)) => {
                            tracing::warn!("skipping progress image {result_id}: {err:?}");
                        }
                    }
                    None
                }
                else => bail!("stopped waiting with {} blocks missing", total - done),
            };
            match finished {
                Some(Finished::Final {
                    result_id,
                    status: ResultStatus::Aborted,
                    ..
                }) => bail!("result {result_id} was aborted (its task failed)"),
                Some(finished) => queued.push_back(finished),
                None => (),
            }
        }
        tracing::info!(session_id, "all {total} blocks rendered");
        Ok(())
    };
    tokio::try_join!(submit, collect).map(|_| ())
}

/// Uploads the scene and submits the first pass of each block, sending each
/// block's final display result id to `submitted` once submitted.
async fn submit(
    pool: &Pool,
    session_id: &str,
    job: &RenderJob,
    seed: u64,
    submitted: UnboundedSender<(String, Block)>,
) -> eyre::Result<()> {
    // One pooled connection per request.

    let scene_id = pool
        .results()
        .await?
        .create(
            session_id,
            [armonik::results::create::RequestItem {
                name: "scene".into(),
                data: job.scene.clone(),
                ..Default::default()
            }],
        )
        .await
        .wrap_err("could not upload the scene")?
        .pop()
        .ok_or_eyre("no scene result created")?
        .result_id;
    let mut task_options = job.task_options.clone();
    task_options
        .options
        .insert(SCENE_ID_OPTION.into(), scene_id.clone());
    job.convergence.to_options(&mut task_options.options);
    TaskType::PathTracing.to_options(&mut task_options.options);
    // The final linear result of each block, the denoising tasks' input.
    let mut finals = HashMap::new();

    let by_name = |raws: Vec<armonik::results::Raw>| -> HashMap<String, String> {
        raws.into_iter()
            .map(|raw| (raw.name, raw.result_id))
            .collect()
    };
    let id = |ids: &HashMap<String, String>, prefix: &str, block: &Block| {
        ids.get(&name(prefix, block))
            .cloned()
            .ok_or_else(|| eyre::eyre!("missing result {prefix} for block {block:?}"))
    };
    for blocks in job.blocks.chunks(BATCH_SIZE) {
        let outputs = by_name(
            pool.results()
                .await?
                .create_metadata(
                    session_id,
                    blocks.iter().flat_map(|block| {
                        [FINAL_LINEAR_NAME, FINAL_DISPLAY_NAME].map(|prefix| {
                            armonik::results::create_metadata::RequestItem {
                                name: name(prefix, block),
                                ..Default::default()
                            }
                        })
                    }),
                )
                .await
                .wrap_err("could not create the results")?,
        );
        let payloads = blocks
            .iter()
            .map(|&block| {
                Ok(BlockPayload {
                    block,
                    samples: job.samples,
                    seed,
                    outputs: BlockOutputs {
                        linear: id(&outputs, FINAL_LINEAR_NAME, &block)?,
                        display: id(&outputs, FINAL_DISPLAY_NAME, &block)?,
                    },
                    previous: None,
                })
            })
            .collect::<eyre::Result<Vec<_>>>()?;
        let payload_ids = by_name(
            pool.results()
                .await?
                .create(
                    session_id,
                    payloads
                        .iter()
                        .map(|payload| armonik::results::create::RequestItem {
                            name: name(PAYLOAD_NAME, &payload.block),
                            data: payload.encode(),
                            ..Default::default()
                        }),
                )
                .await
                .wrap_err("could not upload the payloads")?,
        );
        let items = payloads
            .iter()
            .map(|payload| {
                Ok(armonik::tasks::submit::RequestItem {
                    payload_id: id(&payload_ids, PAYLOAD_NAME, &payload.block)?,
                    expected_output_keys: vec![
                        payload.outputs.linear.clone(),
                        payload.outputs.display.clone(),
                    ],
                    data_dependencies: vec![scene_id.clone()],
                    task_options: None,
                })
            })
            .collect::<eyre::Result<Vec<_>>>()?;

        pool.tasks()
            .await?
            .submit(session_id, Some(task_options.clone()), items)
            .await
            .wrap_err("could not submit the tasks")?;
        for payload in payloads {
            if job.denoise {
                finals.insert(payload.block, payload.outputs.linear);
            } else {
                // The receiver only goes away once the run is over.
                _ = submitted.send((payload.outputs.display, payload.block));
            }
        }
    }
    if !job.denoise {
        return Ok(());
    }

    // Denoising tasks, once every block's final results exist: each depends
    // on those of its neighbourhood, so it starts as soon as they are done.
    TaskType::Denoising.to_options(&mut task_options.options);
    task_options.priority = DENOISE_PRIORITY;
    if let Some(partition) = &job.denoise_partition {
        task_options.partition_id = partition.clone();
    }
    for blocks in job.blocks.chunks(BATCH_SIZE) {
        let outputs = by_name(
            pool.results()
                .await?
                .create_metadata(
                    session_id,
                    blocks.iter().flat_map(|block| {
                        [DENOISED_LINEAR_NAME, DENOISED_DISPLAY_NAME].map(|prefix| {
                            armonik::results::create_metadata::RequestItem {
                                name: name(prefix, block),
                                ..Default::default()
                            }
                        })
                    }),
                )
                .await
                .wrap_err("could not create the denoised results")?,
        );
        let payloads = blocks
            .iter()
            .map(|&block| {
                let inputs = neighborhood(&job.blocks, &block, job.width, job.height)
                    .into_iter()
                    .map(|neighbour| DenoiseInput {
                        block: neighbour,
                        result: finals[&neighbour].clone(),
                    })
                    .collect();
                Ok(DenoisePayload {
                    block,
                    inputs,
                    outputs: BlockOutputs {
                        linear: id(&outputs, DENOISED_LINEAR_NAME, &block)?,
                        display: id(&outputs, DENOISED_DISPLAY_NAME, &block)?,
                    },
                })
            })
            .collect::<eyre::Result<Vec<_>>>()?;
        let payload_ids = by_name(
            pool.results()
                .await?
                .create(
                    session_id,
                    payloads
                        .iter()
                        .map(|payload| armonik::results::create::RequestItem {
                            name: name(DENOISE_PAYLOAD_NAME, &payload.block),
                            data: payload.encode(),
                            ..Default::default()
                        }),
                )
                .await
                .wrap_err("could not upload the denoising payloads")?,
        );
        let items = payloads
            .iter()
            .map(|payload| {
                Ok(armonik::tasks::submit::RequestItem {
                    payload_id: id(&payload_ids, DENOISE_PAYLOAD_NAME, &payload.block)?,
                    expected_output_keys: vec![
                        payload.outputs.linear.clone(),
                        payload.outputs.display.clone(),
                    ],
                    data_dependencies: std::iter::once(scene_id.clone())
                        .chain(payload.inputs.iter().map(|input| input.result.clone()))
                        .collect(),
                    task_options: None,
                })
            })
            .collect::<eyre::Result<Vec<_>>>()?;
        pool.tasks()
            .await?
            .submit(session_id, Some(task_options.clone()), items)
            .await
            .wrap_err("could not submit the denoising tasks")?;
        for payload in payloads {
            _ = submitted.send((payload.outputs.display, payload.block));
        }
    }
    Ok(())
}

/// Downloads and decodes a display image, retrying transient errors.
async fn download(
    pool: &Pool,
    session_id: &str,
    finished: Finished,
) -> (Finished, eyre::Result<DisplayBlock>) {
    let result_id = finished.result_id().to_owned();
    let mut attempt = 1;
    let bytes = loop {
        let data = async {
            // The connection is held until the whole result is read.
            let mut results = pool.results().await?;
            let chunks = results.download(session_id, &result_id).await?;
            Ok::<_, eyre::Report>(chunks.try_concat().await?)
        };
        match data.await {
            Ok(bytes) => break bytes,
            Err(err) if attempt < DOWNLOAD_ATTEMPTS => {
                tracing::warn!("download of {result_id} failed (attempt {attempt}): {err:#}");
                attempt += 1;
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            Err(err) => {
                let err = Err(err).wrap_err_with(|| format!("could not download {result_id}"));
                return (finished, err);
            }
        }
    };
    let image = DisplayBlock::decode(&bytes)
        .wrap_err_with(|| format!("result {result_id} is not a display image"));
    (finished, image)
}

/// A finished result worth downloading.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Finished {
    /// The final display image of `block` (completed or aborted).
    Final {
        result_id: String,
        block: Block,
        status: ResultStatus,
    },
    /// An intermediate display image (self-describing: block and samples).
    Progress { result_id: String },
}

impl Finished {
    fn result_id(&self) -> &str {
        match self {
            Finished::Final { result_id, .. } | Finished::Progress { result_id } => result_id,
        }
    }
}

/// Sorts the finished display images the watcher reports into the final
/// ones the run waits for, which it learns from the submission, and the
/// intermediate ones. Either side can come first: a final may finish before
/// its submission has been recorded, so an unknown image is shown as
/// progress but remembered, in case it turns out to be a final.
#[derive(Debug, Default)]
struct Tracker {
    /// Submitted final images not finished yet.
    pending: HashMap<String, Block>,
    /// Finished images not known to be finals: intermediate ones, and early
    /// finishes of finals.
    unclaimed: HashMap<String, ResultStatus>,
    /// Finals already reported, to drop duplicates.
    claimed: HashSet<String>,
}

impl Tracker {
    /// Records a submitted final; returns it if it already finished.
    fn expect(&mut self, result_id: String, block: Block) -> Option<Finished> {
        match self.unclaimed.remove(&result_id) {
            Some(status) => {
                self.claimed.insert(result_id.clone());
                Some(Finished::Final {
                    result_id,
                    block,
                    status,
                })
            }
            None => {
                self.pending.insert(result_id, block);
                None
            }
        }
    }

    /// Records a finished image; returns it the first time only.
    fn update(&mut self, update: ResultUpdate) -> Option<Finished> {
        let ResultUpdate { result_id, status } = update;
        if let Some(block) = self.pending.remove(&result_id) {
            self.claimed.insert(result_id.clone());
            Some(Finished::Final {
                result_id,
                block,
                status,
            })
        } else if self.claimed.contains(&result_id) || self.unclaimed.contains_key(&result_id) {
            None
        } else {
            let completed = status == ResultStatus::Completed;
            self.unclaimed.insert(result_id.clone(), status);
            completed.then_some(Finished::Progress { result_id })
        }
    }
}

#[cfg(test)]
mod tests;
