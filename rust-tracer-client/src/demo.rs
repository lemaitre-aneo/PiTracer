//! Rendering an image on ArmoniK, one session per run: upload the scene,
//! submit one task per block, then download and forward each block as soon
//! as its result completes.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Duration;

use armonik::{Client, ResultStatus, TaskOptions};
use eyre::{Context, OptionExt, bail, ensure};
use futures::stream::FuturesUnordered;
use futures::{StreamExt, TryStreamExt};
use raytracer::{Block, BlockPayload, BlockResult, SCENE_ID_OPTION};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::watcher::{ResultUpdate, ResultWatcher};

/// Everything a run needs, identical from one run to the next.
pub struct RenderJob {
    /// Encoded scene definition, shared by all the tasks of a session.
    pub scene: Vec<u8>,
    /// Blocks to render, in submission order.
    pub blocks: Vec<Block>,
    /// Samples per pixel.
    pub samples: u32,
    /// Default options of the session's tasks.
    pub task_options: TaskOptions,
    /// Partitions the session may use (empty: the default partition).
    pub partitions: Vec<String>,
}

/// What a run reports to the display.
#[derive(Debug)]
pub enum RunEvent {
    /// A new run starts: clear the image.
    Reset { session_id: String },
    /// A block was rendered.
    Block(BlockResult),
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
/// one ends (or only on [`Command::Rerun`] without a delay).
pub async fn demo_loop(
    client: Client,
    job: RenderJob,
    watcher: Box<dyn ResultWatcher>,
    rerun_delay: Option<Duration>,
    events: UnboundedSender<RunEvent>,
    mut commands: UnboundedReceiver<Command>,
) {
    loop {
        match run(
            &client,
            &job,
            &*watcher,
            rand::random(),
            &events,
            &mut commands,
        )
        .await
        {
            RunEnd::Interrupted(Command::Quit) => return,
            RunEnd::Interrupted(_) => continue,
            RunEnd::Ended => (),
        }
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
                () = &mut delay => break,
                command = commands.recv() => match command {
                    Some(Command::Rerun) => break,
                    Some(Command::Quit) | None => return,
                    // Nothing to pause between runs.
                    Some(Command::TogglePause) => (),
                },
            }
        }
    }
}

/// One run, in its own session, which it cleans up whatever happens: a
/// failure is reported (log and [`RunEvent::Finished`]), not returned.
pub async fn run(
    client: &Client,
    job: &RenderJob,
    watcher: &dyn ResultWatcher,
    seed: u64,
    events: &UnboundedSender<RunEvent>,
    commands: &mut UnboundedReceiver<Command>,
) -> RunEnd {
    let session_id = match client
        .clone()
        .into_sessions()
        .create(job.partitions.clone(), job.task_options.clone())
        .await
    {
        Ok(session_id) => session_id,
        Err(err) => {
            let error = format!("could not create a session: {err}");
            tracing::error!("{error}");
            _ = events.send(RunEvent::Finished { error: Some(error) });
            return RunEnd::Ended;
        }
    };
    tracing::info!(session_id, "rendering {} blocks", job.blocks.len());
    _ = events.send(RunEvent::Reset {
        session_id: session_id.clone(),
    });

    // Scoped so the render (its watcher and downloads) is dropped before the
    // session is torn down.
    let outcome = {
        let render = render(client, &session_id, job, watcher, seed, events);
        let mut render = std::pin::pin!(render);
        let mut paused = false;
        loop {
            tokio::select! {
                result = &mut render => break result.map(|()| RunEnd::Ended),
                command = commands.recv() => match command {
                    Some(Command::TogglePause) => {
                        paused = !paused;
                        toggle_pause(client, &session_id, paused).await;
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
    if tokio::time::timeout(CLEANUP_TIMEOUT, cleanup(client, &session_id, completed))
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

async fn toggle_pause(client: &Client, session_id: &str, pause: bool) {
    let mut sessions = client.clone().into_sessions();
    let result = if pause {
        sessions.pause(session_id).await
    } else {
        sessions.resume(session_id).await
    };
    match result {
        Ok(_) => tracing::info!(session_id, "{}", if pause { "paused" } else { "resumed" }),
        Err(err) => tracing::warn!(session_id, "could not pause or resume: {err}"),
    }
}

/// Closes a completed session, cancels an abandoned one, then purges and
/// deletes it (its results are no longer needed).
async fn cleanup(client: &Client, session_id: &str, completed: bool) {
    let mut sessions = client.clone().into_sessions();
    let first = if completed {
        sessions.close(session_id).await
    } else {
        sessions.cancel(session_id).await
    };
    let steps = [
        (if completed { "close" } else { "cancel" }, first),
        ("purge", sessions.purge(session_id).await),
        ("delete", sessions.delete(session_id).await),
    ];
    for (step, result) in steps {
        if let Err(err) = result {
            tracing::warn!(session_id, "could not {step} the session: {err}");
        }
    }
}

/// Submits the job's tasks and forwards their results until every block is
/// displayed. Submission and downloads overlap, so the first blocks show up
/// while later ones are still being submitted.
async fn render(
    client: &Client,
    session_id: &str,
    job: &RenderJob,
    watcher: &dyn ResultWatcher,
    seed: u64,
    events: &UnboundedSender<RunEvent>,
) -> eyre::Result<()> {
    let (submitted, mut submissions) = tokio::sync::mpsc::unbounded_channel();
    let submit = async move { submit(client, session_id, job, seed, submitted).await };
    let collect = async {
        let total = job.blocks.len();
        let mut updates = watcher.watch(client.clone(), session_id.to_owned());
        let mut tracker = Tracker::default();
        let mut queued = VecDeque::new();
        let mut downloads = FuturesUnordered::new();
        let mut submitting = true;
        let mut done = 0;
        while done < total {
            while downloads.len() < DOWNLOADS
                && let Some((result_id, block)) = queued.pop_front()
            {
                downloads.push(download(client, session_id, result_id, block));
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
                Some(result) = downloads.next() => {
                    let result = result?;
                    done += 1;
                    _ = events.send(RunEvent::Block(result));
                    _ = events.send(RunEvent::Progress { done, total });
                    None
                }
                else => bail!("stopped waiting with {} blocks missing", total - done),
            };
            match finished {
                Some((result_id, _, ResultStatus::Aborted)) => {
                    bail!("result {result_id} was aborted (its task failed)")
                }
                Some((result_id, block, _)) => queued.push_back((result_id, block)),
                None => (),
            }
        }
        tracing::info!(session_id, "all {total} blocks rendered");
        Ok(())
    };
    tokio::try_join!(submit, collect).map(|_| ())
}

/// Uploads the scene and submits one task per block, sending each task's
/// result id (and block) to `submitted` once submitted.
async fn submit(
    client: &Client,
    session_id: &str,
    job: &RenderJob,
    seed: u64,
    submitted: UnboundedSender<(String, Block)>,
) -> eyre::Result<()> {
    let mut results = client.clone().into_results();
    let mut tasks = client.clone().into_tasks();

    let scene_id = results
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

    for blocks in job.blocks.chunks(BATCH_SIZE) {
        let name = |prefix: &str, block: &Block| format!("{prefix}-{}-{}", block.x, block.y);
        let payloads = results
            .create(
                session_id,
                blocks
                    .iter()
                    .map(|&block| armonik::results::create::RequestItem {
                        name: name("payload", &block),
                        data: BlockPayload {
                            block,
                            samples: job.samples,
                            seed,
                        }
                        .encode(),
                        ..Default::default()
                    }),
            )
            .await
            .wrap_err("could not upload the payloads")?;
        let outputs = results
            .create_metadata(
                session_id,
                blocks
                    .iter()
                    .map(|block| armonik::results::create_metadata::RequestItem {
                        name: name("block", block),
                        ..Default::default()
                    }),
            )
            .await
            .wrap_err("could not create the results")?;
        let by_name = |raws: Vec<armonik::results::Raw>| -> HashMap<String, String> {
            raws.into_iter()
                .map(|raw| (raw.name, raw.result_id))
                .collect()
        };
        let (payloads, outputs) = (by_name(payloads), by_name(outputs));
        let items = blocks
            .iter()
            .map(|block| {
                let payload_id = payloads.get(&name("payload", block));
                let output_id = outputs.get(&name("block", block));
                match (payload_id, output_id) {
                    (Some(payload_id), Some(output_id)) => {
                        Ok((*block, payload_id.clone(), output_id.clone()))
                    }
                    _ => Err(eyre::eyre!("missing results for block {block:?}")),
                }
            })
            .collect::<eyre::Result<Vec<_>>>()?;

        tasks
            .submit(
                session_id,
                Some(task_options.clone()),
                items.iter().map(|(_, payload_id, output_id)| {
                    armonik::tasks::submit::RequestItem {
                        payload_id: payload_id.clone(),
                        expected_output_keys: vec![output_id.clone()],
                        data_dependencies: vec![scene_id.clone()],
                        task_options: None,
                    }
                }),
            )
            .await
            .wrap_err("could not submit the tasks")?;
        for (block, _, output_id) in items {
            // The receiver only goes away once the run is over.
            _ = submitted.send((output_id, block));
        }
    }
    Ok(())
}

/// Downloads and decodes the result of `block`, retrying transient errors.
async fn download(
    client: &Client,
    session_id: &str,
    result_id: String,
    block: Block,
) -> eyre::Result<BlockResult> {
    let mut attempt = 1;
    let bytes = loop {
        let data = async {
            let chunks = client
                .clone()
                .into_results()
                .download(session_id, &result_id)
                .await?;
            chunks.try_concat().await
        };
        match data.await {
            Ok(bytes) => break bytes,
            Err(err) if attempt < DOWNLOAD_ATTEMPTS => {
                tracing::warn!("download of {result_id} failed (attempt {attempt}): {err}");
                attempt += 1;
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            Err(err) => {
                return Err(err).wrap_err_with(|| format!("could not download {result_id}"));
            }
        }
    };
    let result = BlockResult::decode(&bytes)
        .wrap_err_with(|| format!("result {result_id} is not a block result"))?;
    ensure!(
        result.block == block,
        "result {result_id} holds block {:?}, expected {block:?}",
        result.block
    );
    Ok(result)
}

/// Matches the stream of finished results against the results the run
/// waits for. Either side can come first: a task may finish before the
/// submission that created it has been recorded.
#[derive(Debug, Default)]
struct Tracker {
    /// Submitted results not finished yet.
    pending: HashMap<String, Block>,
    /// Finished results not (yet) known to be ours: early finishes of ours,
    /// and everything else the watcher reports (scene, payloads).
    unclaimed: HashMap<String, ResultStatus>,
    /// Results already reported as finished, to drop duplicates.
    claimed: HashSet<String>,
}

impl Tracker {
    /// Records a submitted result; returns it if it already finished.
    fn expect(&mut self, result_id: String, block: Block) -> Option<(String, Block, ResultStatus)> {
        match self.unclaimed.remove(&result_id) {
            Some(status) => {
                self.claimed.insert(result_id.clone());
                Some((result_id, block, status))
            }
            None => {
                self.pending.insert(result_id, block);
                None
            }
        }
    }

    /// Records a finished result; returns it if it is one of ours, the first
    /// time only.
    fn update(&mut self, update: ResultUpdate) -> Option<(String, Block, ResultStatus)> {
        let ResultUpdate { result_id, status } = update;
        if let Some(block) = self.pending.remove(&result_id) {
            self.claimed.insert(result_id.clone());
            Some((result_id, block, status))
        } else {
            if !self.claimed.contains(&result_id) {
                self.unclaimed.insert(result_id, status);
            }
            None
        }
    }
}

#[cfg(test)]
mod tests;
