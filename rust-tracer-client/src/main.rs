use std::collections::HashMap;
use std::time::Duration;

use armonik::TaskOptions;
use clap::Parser;
use eyre::{Context, ensure};
use raytracer::{Scene, SceneDefinition, encode_scene};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::demo::{Command, RenderJob, RunEvent};
use crate::watcher::WatchMode;

mod blocks;
mod demo;
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

    /// Samples per pixel
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

    /// Width of a task's block, in pixels
    #[arg(long, default_value = "60")]
    pub task_width: u32,

    /// Height of a task's block, in pixels
    #[arg(long, default_value = "60")]
    pub task_height: u32,

    /// How to wait for the results
    #[arg(long, value_enum, default_value_t = WatchMode::Events)]
    pub watch: WatchMode,

    /// Delay between two result listings with `--watch polling`, in seconds
    #[arg(long, default_value = "3")]
    pub poll_interval: f64,

    /// Partition running rust-tracer-worker [default: the default partition]
    #[arg(long)]
    pub partition: Option<String>,

    /// Do not start a new render 30 s after one ends; wait for R instead
    #[arg(long)]
    pub no_auto_rerun: bool,

    #[command(flatten)]
    pub transport: transport::Transport,
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

fn main() -> Result<(), eyre::Report> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    let definition = scene(&cli)?;
    let (width, height) = (definition.camera.width, definition.camera.height);
    ensure!(width > 0 && height > 0, "the image is empty");
    ensure!(
        cli.task_width > 0 && cli.task_height > 0,
        "the task blocks are empty"
    );
    ensure!(definition.samples > 0, "no samples per pixel");
    let poll_interval =
        Duration::try_from_secs_f64(cli.poll_interval).wrap_err("invalid --poll-interval")?;

    let job = RenderJob {
        scene: encode_scene(&definition),
        blocks: blocks::tile(width, height, cli.task_width, cli.task_height),
        samples: definition.samples,
        task_options: TaskOptions {
            options: HashMap::new(),
            max_duration: armonik::reexports::prost_types::Duration {
                seconds: 300,
                nanos: 0,
            },
            max_retries: 1,
            priority: 1,
            partition_id: cli.partition.clone().unwrap_or_default(),
            ..Default::default()
        },
        partitions: cli.partition.iter().cloned().collect(),
    };
    // Only used for its color conversion (gamma).
    let display = Scene::from(definition);

    let runtime = tokio::runtime::Runtime::new()?;
    // Connect before opening the window, so a bad endpoint fails right away.
    let client = runtime
        .block_on(armonik::Client::with_config(cli.transport.config()?))
        .wrap_err("could not connect to ArmoniK")?;

    let sdl = sdl2::init().map_err(eyre::Report::msg)?;
    let sdl_events = sdl.event().map_err(eyre::Report::msg)?;
    sdl_events
        .register_custom_event::<RunEvent>()
        .map_err(eyre::Report::msg)?;
    let sender = sdl_events.event_sender();

    let (commands, command_receiver) = tokio::sync::mpsc::unbounded_channel();
    let watcher = cli.watch.watcher(poll_interval);
    let rerun_delay = (!cli.no_auto_rerun).then_some(RERUN_DELAY);
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
            demo::demo_loop(client, job, watcher, rerun_delay, events, command_receiver).await;
            _ = forward.await;
        })
    });

    let shown = viewer::run(&sdl, width, height, &display, &commands);

    // Let the demo cancel and clean its session up before SDL goes away.
    _ = commands.send(Command::Quit);
    demo.join()
        .map_err(|_| eyre::eyre!("the demo thread panicked"))?;
    shown
}
