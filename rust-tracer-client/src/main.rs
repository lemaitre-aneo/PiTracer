use std::collections::HashMap;
use std::time::Duration;

use armonik::TaskOptions;
use clap::Parser;
use eyre::{Context, ensure};
use raytracer::{Convergence, DENOISE_ALIGNMENT, SceneDefinition, encode_scene};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::demo::{Command, RenderJob, RunEvent};
use crate::watcher::WatchMode;

mod bag;
mod blocks;
mod demo;
mod pool;
mod transport;
mod viewer;
mod watcher;

#[cfg(not(miri))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Renders a scene on ArmoniK: the image is split into blocks, each rendered
/// by a rust-tracer-worker task, and shown as the blocks come back.
#[derive(Debug, Clone, Parser)]
pub struct Cli {
    /// Width of the scene
    #[arg(short = 'W', long, default_value = "0")]
    pub width: u32,

    /// Height of the scene
    #[arg(short = 'H', long, default_value = "0")]
    pub height: u32,

    /// Scene file
    #[arg(short, long, default_value = "scene.yml")]
    pub scene: String,

    /// Samples per pixel of each pass (one task renders one pass over its
    /// block)
    #[arg(short = 'S', long, default_value = "0")]
    pub samples: u32,

    /// Recursion depth
    #[arg(short, long, default_value = "0")]
    pub depth: u32,

    /// Gamma
    #[arg(short, long, default_value = "0")]
    pub gamma: f32,

    /// Light Sampling [default: true]
    #[arg(long)]
    pub light_sampling: Option<bool>,

    /// Mixture continuation sampling (overrides NEE when > 0): draw the single
    /// continuation direction from the light/BRDF mixture, aiming at a light
    /// with this probability, one ray per bounce. 0 disables. Meaningful in
    /// [0,1). Negative = use the scene file's value.
    #[arg(long, default_value = "-1")]
    pub mixture_sampling: f32,

    /// Geometric decay of the mixture light probability per bounce
    /// (c_l = mixture_sampling * decay^depth). 1.0 = constant (deep
    /// fireflies); < 1 concentrates aiming on early bounces and bounds
    /// throughput. Negative = use the scene file's value.
    #[arg(long, default_value = "-1")]
    pub mixture_decay: f32,

    /// Width of a task's block, in pixels (a multiple of 16 when denoising)
    #[arg(long, default_value = "64")]
    pub task_width: u32,

    /// Height of a task's block, in pixels (a multiple of 16 when denoising)
    #[arg(long, default_value = "64")]
    pub task_height: u32,

    /// Do not denoise the blocks (Open Image Denoise, in denoising tasks,
    /// each over the neighbourhood of its block) once they are final
    #[arg(long)]
    pub no_denoise: bool,

    /// A block is done once the estimated mean squared error of its display
    /// image, in 8-bit levels, is at most this
    #[arg(long, default_value_t = Convergence::default().error_threshold)]
    pub error_threshold: f32,

    /// A block is done once it has this many samples per pixel, converged or
    /// not
    #[arg(long, default_value_t = Convergence::default().max_samples)]
    pub max_samples: u32,

    /// How to wait for the results
    #[arg(long, value_enum, default_value_t = WatchMode::Events)]
    pub watch: WatchMode,

    /// Delay between two result listings with `--watch polling`, in seconds
    #[arg(long, default_value = "3")]
    pub poll_interval: f64,

    /// Partition running rust-tracer-worker [default: the default partition]
    #[arg(long)]
    pub partition: Option<String>,

    /// Partition running the denoising tasks, such as one whose workers have
    /// a GPU [default: --partition]
    #[arg(long, requires = "partition")]
    pub denoise_partition: Option<String>,

    /// Do not start a new render 30 s after one ends; wait for R (or SIGUSR1)
    /// instead
    #[arg(long)]
    pub no_auto_rerun: bool,

    /// Keep the sessions at the end of each run instead of deleting them, so
    /// they can be inspected (their results are still purged)
    #[arg(long)]
    pub keep_session: bool,

    #[command(flatten)]
    pub transport: transport::Transport,
}

/// Displays an error followed by its causes, `: `-separated: what went wrong
/// with a gRPC request (its status code and message) is in the source of the
/// `armonik` error, whose own message only says that a request failed.
pub(crate) struct Chain<'a>(pub &'a dyn std::error::Error);

impl std::fmt::Display for Chain<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)?;
        let mut source = self.0.source();
        while let Some(cause) = source {
            write!(f, ": {cause}")?;
            source = cause.source();
        }
        Ok(())
    }
}

/// Delay between the end of a run and the next one, unless
/// `--no-auto-rerun`.
const RERUN_DELAY: Duration = Duration::from_secs(30);

/// The scene file, with the command-line overrides applied (`0`, or a
/// negative value for the mixture knobs, keeps the file's value).
fn scene(cli: &Cli) -> Result<SceneDefinition, eyre::Report> {
    let mut scene = config::Config::builder()
        .add_source(config::File::with_name(&cli.scene))
        .build()?
        .try_deserialize::<SceneDefinition>()
        .wrap_err_with(|| format!("could not load the scene {}", cli.scene))?;
    // Texture files are relative to the scene file, and travel in the scene.
    let folder = std::path::Path::new(&cli.scene)
        .parent()
        .unwrap_or(".".as_ref());
    scene
        .load_textures(folder)
        .wrap_err_with(|| format!("could not load the textures of the scene {}", cli.scene))?;
    if cli.width != 0 {
        scene.camera.width = cli.width;
    }
    if cli.height != 0 {
        scene.camera.height = cli.height;
    }
    if cli.samples != 0 {
        scene.samples = cli.samples;
    }
    if cli.depth != 0 {
        scene.recursion_depth = cli.depth;
    }
    if cli.gamma != 0f32 {
        scene.gamma = cli.gamma;
    }
    if let Some(light_sampling) = cli.light_sampling {
        scene.light_sampling = light_sampling;
    }
    if cli.mixture_sampling >= 0f32 {
        scene.mixture_sampling = cli.mixture_sampling;
    }
    if cli.mixture_decay >= 0f32 {
        scene.mixture_decay = cli.mixture_decay;
    }
    Ok(scene)
}

fn convergence(cli: &Cli) -> Convergence {
    Convergence {
        error_threshold: cli.error_threshold,
        max_samples: cli.max_samples,
    }
}

/// What a run renders: the scene file as it is now, with the command-line
/// overrides. Called again before every run, so the scene can be edited
/// while the demo runs.
fn job(cli: &Cli) -> Result<RenderJob, eyre::Report> {
    let definition = scene(cli)?;
    let (width, height) = (definition.camera.width, definition.camera.height);
    ensure!(width > 0 && height > 0, "the image is empty");
    ensure!(definition.samples > 0, "no samples per pixel");
    Ok(RenderJob {
        scene: encode_scene(&definition),
        width,
        height,
        blocks: raytracer::tile_grid(width, height, cli.task_width, cli.task_height),
        samples: definition.samples,
        convergence: convergence(cli),
        denoise: !cli.no_denoise,
        task_options: TaskOptions {
            max_duration: armonik::reexports::prost_types::Duration {
                seconds: 300,
                nanos: 0,
            },
            max_retries: 1,
            priority: 1,
            partition_id: cli.partition.clone().unwrap_or_default(),
            ..Default::default()
        },
        partitions: cli
            .partition
            .iter()
            .chain(&cli.denoise_partition)
            .cloned()
            .collect(),
        denoise_partition: cli.denoise_partition.clone(),
    })
}

/// Resolves on the first SIGINT or SIGTERM (Ctrl-C on Windows). The handlers
/// are registered by the call (within a runtime), not on the first poll; one
/// that cannot be is logged, and its signal keeps its default action.
#[cfg(unix)]
fn interrupted() -> impl Future<Output = ()> {
    use futures::{StreamExt, stream::FuturesUnordered};
    use tokio::signal::unix::{SignalKind, signal};
    let mut signals: Vec<_> = [
        (SignalKind::interrupt(), "SIGINT"),
        (SignalKind::terminate(), "SIGTERM"),
    ]
    .into_iter()
    .filter_map(|(kind, name)| {
        signal(kind)
            .inspect_err(|err| tracing::error!("could not handle {name}: {err}"))
            .ok()
    })
    .collect();
    async move {
        let mut received = signals
            .iter_mut()
            .map(|signal| signal.recv())
            .collect::<FuturesUnordered<_>>();
        // No handler at all: nothing will ever be received.
        if received.next().await.is_none() {
            std::future::pending().await
        }
    }
}

#[cfg(windows)]
fn interrupted() -> impl Future<Output = ()> {
    let ctrl_c = tokio::signal::windows::ctrl_c()
        .inspect_err(|err| tracing::error!("could not handle Ctrl-C: {err}"));
    async move {
        match ctrl_c {
            Ok(mut ctrl_c) => _ = ctrl_c.recv().await,
            Err(_) => std::future::pending().await,
        }
    }
}

fn main() -> Result<(), eyre::Report> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    // Settings that do not depend on the scene file: checked once.
    ensure!(
        cli.task_width > 0 && cli.task_height > 0,
        "the task blocks are empty"
    );
    // Denoising tiles are grown from blocks, and must stay aligned.
    ensure!(
        cli.no_denoise
            || (cli.task_width % DENOISE_ALIGNMENT == 0
                && cli.task_height % DENOISE_ALIGNMENT == 0),
        "denoising needs task blocks whose sizes are multiples of {DENOISE_ALIGNMENT} pixels \
         (or --no-denoise)"
    );
    let poll_interval =
        Duration::try_from_secs_f64(cli.poll_interval).wrap_err("invalid --poll-interval")?;
    let convergence = convergence(&cli);
    let mut options = HashMap::new();
    convergence.to_options(&mut options);
    // What the workers will read back, so invalid values fail here.
    Convergence::from_options(&options).wrap_err("invalid convergence settings")?;
    // Without a previous scene to fall back on, the first load must succeed.
    let first = job(&cli)?;
    let (width, height) = (first.width, first.height);

    let runtime = tokio::runtime::Runtime::new()?;
    // Connect before opening the window, so a bad endpoint fails right away
    // (the connection then goes back to the pool for the first requests).
    let config = cli.transport.config()?;
    let pool = runtime.block_on(async {
        let pool = std::sync::Arc::new(pool::Pool::new(config, cli.transport.pool_config()));
        drop(
            pool.client()
                .await
                .wrap_err("could not connect to ArmoniK")?,
        );
        Ok::<_, eyre::Report>(pool)
    })?;

    // Ctrl-C (or SIGTERM) closes the window, so the session is cleaned up
    // as on Esc. Registered now, so a signal can no longer kill the process
    // once a session may exist, and before SDL, which must not take over.
    let interrupted = {
        let _runtime = runtime.enter();
        interrupted()
    };
    sdl2::hint::set("SDL_NO_SIGNAL_HANDLERS", "1");
    let sdl = sdl2::init().map_err(eyre::Report::msg)?;
    let sdl_events = sdl.event().map_err(eyre::Report::msg)?;
    sdl_events
        .register_custom_event::<RunEvent>()
        .map_err(eyre::Report::msg)?;
    let sender = sdl_events.event_sender();
    let quit_sender = sdl_events.event_sender();
    runtime.spawn(async move {
        interrupted.await;
        if let Err(err) = quit_sender.push_event(sdl2::event::Event::Quit { timestamp: 0 }) {
            tracing::warn!("could not forward Ctrl-C to the window: {err}");
        }
    });

    let (commands, command_receiver) = tokio::sync::mpsc::unbounded_channel();
    // SIGUSR1 renders the scene file again, as R. Registered before the
    // first session too: SIGUSR1 kills by default. Optional: R still works.
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let rerun = {
            let _runtime = runtime.enter();
            signal(SignalKind::user_defined1())
        };
        match rerun {
            Ok(mut rerun) => {
                let commands = commands.clone();
                runtime.spawn(async move {
                    while rerun.recv().await.is_some() {
                        tracing::info!("SIGUSR1: rendering again");
                        if commands.send(Command::Rerun).is_err() {
                            break;
                        }
                    }
                });
            }
            Err(err) => tracing::error!("could not handle SIGUSR1: {err}"),
        }
    }
    let watcher = cli.watch.watcher(poll_interval);
    let demo = std::thread::spawn(move || {
        runtime.block_on(async move {
            let (events, mut event_receiver) = tokio::sync::mpsc::unbounded_channel();
            let forward = tokio::spawn(async move {
                while let Some(event) = event_receiver.recv().await {
                    if let Err(err) = sender.push_custom_event(event) {
                        tracing::warn!("could not forward an event to the window: {err}");
                    }
                }
            });
            let options = demo::RunOptions {
                rerun_delay: (!cli.no_auto_rerun).then_some(RERUN_DELAY),
                keep_session: cli.keep_session,
            };
            let reload = move || job(&cli);
            demo::demo_loop(
                pool,
                first,
                reload,
                watcher,
                options,
                events,
                command_receiver,
            )
            .await;
            _ = forward.await;
        })
    });

    let shown = viewer::run(&sdl, width, height, &commands);

    // Let the demo cancel and clean its session up before SDL goes away.
    _ = commands.send(Command::Quit);
    demo.join()
        .map_err(|_| eyre::eyre!("the demo thread panicked"))?;
    shown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chains_show_the_causes() {
        let status = armonik::reexports::tonic::Status::unavailable("ingress busy");
        let io = std::io::Error::other(status);
        let shown = Chain(&io).to_string();
        assert!(shown.contains("ingress busy"), "{shown}");
        assert_eq!(
            Chain(&std::fmt::Error).to_string(),
            std::fmt::Error.to_string()
        );
    }

    const SCENE: &str = "
camera:
  origin: [0, 0, 0]
  direction: [0, 0, -1]
  width: 40
  height: 30
samples: 8
objects:
  - type: sphere
    radius: 1
    position: [0, 0, -5]
    emission: [1, 1, 1]
";

    fn cli(scene: &std::path::Path, extra: &[&str]) -> Cli {
        let scene = scene.to_str().unwrap();
        let args = [
            "client",
            "-e",
            "localhost:1",
            "-s",
            scene,
            "--task-width",
            "16",
        ];
        Cli::try_parse_from(args.iter().chain(extra)).unwrap()
    }

    #[test]
    fn job_reflects_the_scene_file_as_it_is_now() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("scene.yml");
        std::fs::write(&path, SCENE).unwrap();

        let first = job(&cli(&path, &[])).unwrap();
        assert_eq!((first.width, first.height, first.samples), (40, 30, 8));
        assert_eq!(first.blocks.len(), 3);
        // Neither the scene file nor the command line: 50 samples per pass.
        std::fs::write(&path, SCENE.replace("samples: 8\n", "")).unwrap();
        assert_eq!(job(&cli(&path, &[])).unwrap().samples, 50);
        std::fs::write(&path, SCENE).unwrap();
        // Command-line overrides apply on every load.
        let overridden = job(&cli(&path, &["-W", "64", "-S", "3"])).unwrap();
        assert_eq!((overridden.width, overridden.samples), (64, 3));

        // Edited while the demo runs.
        std::fs::write(&path, SCENE.replace("width: 40", "width: 20")).unwrap();
        let edited = job(&cli(&path, &[])).unwrap();
        assert_eq!(edited.width, 20);
        assert_ne!(edited.scene, first.scene);
    }

    /// Texture files are read relative to the scene file, and travel inside
    /// the scene the workers get.
    #[test]
    fn texture_files_travel_in_the_scene() {
        let folder = tempfile::tempdir().unwrap();
        std::fs::create_dir(folder.path().join("textures")).unwrap();
        std::fs::copy(
            "../rust-tracer-standalone/armonik.svg",
            folder.path().join("textures/logo.svg"),
        )
        .unwrap();
        let logo = "
  - type: quad
    position: [0, 0, -1]
    u: [1, 0, 0]
    v: [0, 1, 0]
    texture: image
    file: textures/logo.svg
";
        let path = folder.path().join("scene.yml");
        std::fs::write(&path, format!("{SCENE}{logo}")).unwrap();

        let received = raytracer::decode_scene(&job(&cli(&path, &[])).unwrap().scene).unwrap();
        let raytracer::TextureEnum::Image(image) = &received.objects.last().unwrap().texture else {
            panic!("not an image texture");
        };
        assert_eq!(image.size(), Some((1234.0, 327.0)));

        std::fs::remove_file(folder.path().join("textures/logo.svg")).unwrap();
        assert!(job(&cli(&path, &[])).is_err());
    }

    #[test]
    fn broken_scene_files_are_errors() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("scene.yml");
        for broken in [
            "objects: [",
            "camera: {width: lots}",
            &SCENE.replace("samples: 8", "samples: 0"),
            &SCENE.replace("width: 40", "width: 0"),
        ] {
            std::fs::write(&path, broken).unwrap();
            assert!(job(&cli(&path, &[])).is_err(), "{broken}");
        }
        std::fs::remove_file(&path).unwrap();
        assert!(job(&cli(&path, &[])).is_err());
    }
}
