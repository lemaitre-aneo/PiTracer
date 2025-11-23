extern crate sdl2;

use ::raytracer::{CameraDefinition, Scene};
use clap::Parser;
use rand::SeedableRng;
use sdl2::event::Event;
use sdl2::event::WindowEvent;
use sdl2::keyboard::Keycode;
use sdl2::rect::Rect;
use std::sync::Arc;
use std::time::Duration;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::raytracer::Raytracer;

mod raytracer;

#[cfg(not(miri))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug, Clone, Default, Parser)]
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
}

fn scene(cli: &mut Cli) -> Result<Scene, eyre::Report> {
    let mut scene = config::Config::builder()
        .add_source(config::File::with_name(&cli.scene))
        .build()?
        .try_deserialize::<Scene<CameraDefinition>>()?;
    if cli.width == 0 {
        cli.width = scene.camera.width;
    } else {
        scene.camera.width = cli.width;
    }
    if cli.height == 0 {
        cli.height = scene.camera.height;
    } else {
        scene.camera.height = cli.height;
    }
    if cli.samples == 0 {
        cli.samples = scene.samples;
    } else {
        scene.samples = cli.samples;
    }
    if cli.depth == 0 {
        cli.depth = scene.recursion_depth;
    } else {
        scene.recursion_depth = cli.depth;
    }
    if cli.gamma == 0f32 {
        cli.gamma = scene.gamma;
    } else {
        scene.gamma = cli.gamma;
    }
    let scene = scene.map_camera(From::from);
    tracing::debug!("{scene:?}");
    Ok(scene)
}

pub fn main() -> Result<(), eyre::Report> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_span_events(
            tracing_subscriber::fmt::format::FmtSpan::NEW
                | tracing_subscriber::fmt::format::FmtSpan::CLOSE,
        ))
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let mut cli = Cli::parse();
    let scene = scene(&mut cli)?;

    let sdl = sdl2::init().map_err(eyre::Report::msg)?;
    let video_subsystem = sdl.video().map_err(eyre::Report::msg)?;

    let raytracer = Raytracer::new(scene, &sdl).map_err(eyre::Report::msg)?;

    let window = video_subsystem
        .window("Rust Tracer", cli.width, cli.height)
        .position_centered()
        .resizable()
        .build()?;

    let mut canvas = window.into_canvas().build()?;
    let texture_creator = canvas.texture_creator();

    let mut event_pump = sdl.event_pump().map_err(eyre::Report::msg)?;

    let mut rng = rand::rngs::StdRng::from_os_rng();

    let mut paused = false;
    let mut started = None;

    loop {
        let mut render = false;
        for event in event_pump.poll_iter() {
            if let Some(ctx) = event.as_user_event_type::<Arc<Raytracer>>() {
                ctx.acknowledge();
                render = true;
                continue;
            }
            match event {
                Event::Window {
                    win_event:
                        WindowEvent::Maximized
                        | WindowEvent::Minimized
                        | WindowEvent::Resized(..)
                        | WindowEvent::SizeChanged(..)
                        | WindowEvent::Restored,
                    ..
                } => {
                    render = true;
                }
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => {
                    // Signal that the rendering must be stopped
                    tracing::info!("Stopping Requested");

                    raytracer.stop();
                    return Ok(());
                }
                Event::KeyDown {
                    keycode: Some(Keycode::Space),
                    ..
                } => {
                    paused = !paused;
                    let title = if paused {
                        tracing::info!("Paused");
                        "Rust Tracer (paused)"
                    } else {
                        tracing::info!("Resumed");
                        "Rust Tracer"
                    };
                    canvas
                        .window_mut()
                        .set_title(title)
                        .map_err(eyre::Report::msg)?;
                }
                Event::KeyDown {
                    keycode: Some(Keycode::F11),
                    ..
                } => {
                    let window = canvas.window_mut();

                    let state = match window.fullscreen_state() {
                        sdl2::video::FullscreenType::Off => sdl2::video::FullscreenType::True,
                        _ => sdl2::video::FullscreenType::Off,
                    };
                    window.set_fullscreen(state).map_err(eyre::Report::msg)?;
                    render = true;
                }
                _ => {}
            }
        }

        if Arc::strong_count(&raytracer) == 1 {
            if let Some(started) = started.take() {
                let finished = std::time::Instant::now();
                tracing::info!(
                    "Iteration #{}: finished in {:#?}",
                    raytracer.iter(),
                    finished - started
                );
            }
            if !paused {
                started = Some(std::time::Instant::now());
                raytracer.clone().send_process(&mut rng);
                tracing::debug!("iteration #{}: started", raytracer.iter());
            }
        }

        if render {
            let surface = raytracer.get_surface().map_err(eyre::Report::msg)?;
            let texture = surface
                .as_texture(&texture_creator)
                .map_err(eyre::Report::msg)?;

            let (w, h) = canvas.output_size().unwrap();
            let target = match (cli.width * h).cmp(&(w * cli.height)) {
                std::cmp::Ordering::Greater => {
                    let sh = cli.height * w / cli.width;
                    let y = (h - sh) / 2;
                    Some(Rect::new(0, y as i32, w, sh))
                }
                std::cmp::Ordering::Less => {
                    let sw = cli.width * h / cli.height;
                    let x = (w - sw) / 2;
                    Some(Rect::new(x as i32, 0, sw, h))
                }
                std::cmp::Ordering::Equal => None,
            };

            canvas.clear();

            canvas
                .copy(&texture, None, target)
                .map_err(eyre::Report::msg)?;
            canvas.present();
        }

        std::thread::sleep(Duration::from_secs_f64(1. / 30.));
    }
}
