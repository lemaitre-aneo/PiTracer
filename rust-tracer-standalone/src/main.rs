extern crate sdl2;

use clap::Parser;
use rand::SeedableRng;
use raytracer::{Camera, CameraDefinition, Color, Scene, Sphere, SphereVec, Vector};
use sdl2::event::EventSender;
use sdl2::keyboard::Keycode;
use sdl2::{EventSubsystem, Sdl};
use sdl2::{event::Event, surface::Surface};
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

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

struct MailBox {
    pixels: Mutex<Vec<u32>>,
    event_sender: EventSender,
}

impl MailBox {
    fn new(events: &EventSubsystem) -> Self {
        events.register_custom_event::<Arc<MailBox>>().unwrap();
        Self {
            pixels: Default::default(),
            event_sender: events.event_sender(),
        }
    }

    fn push(self: Arc<Self>, row: u32) {
        let mut pixels = self.pixels.lock().unwrap();
        pixels.push(row);

        if pixels.len() == 1 {
            self.event_sender.push_custom_event(self.clone()).unwrap();
        }
    }

    fn extract_pixels(&self) -> Vec<u32> {
        std::mem::take(&mut self.pixels.lock().unwrap())
    }
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
    tracing::error!("{scene:?}");
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
    let scene = Arc::new(scene(&mut cli)?);

    let sdl_context = sdl2::init().map_err(eyre::Report::msg)?;
    let video_subsystem = sdl_context.video().map_err(eyre::Report::msg)?;

    let window = video_subsystem
        .window("rust-sdl2 demo", cli.width, cli.height)
        .position_centered()
        .build()?;

    let mut canvas = window.into_canvas().build()?;
    let mut surface = Surface::new(cli.width, cli.height, sdl2::pixels::PixelFormatEnum::RGB888)
        .map_err(eyre::Report::msg)?;
    let mailbox = Arc::new(MailBox::new(
        &sdl_context.event().map_err(eyre::Report::msg)?,
    ));

    canvas.present();
    let mut event_pump = sdl_context.event_pump().map_err(eyre::Report::msg)?;

    let mut rng = rand::rngs::StdRng::from_os_rng();

    let total_size = cli.width as usize * cli.height as usize * 3;
    let mut radiance = Vec::<AtomicU32>::with_capacity(total_size);
    radiance.resize_with(total_size, Default::default);
    let radiance = Arc::new(radiance);

    let mut iter = 0;

    loop {
        if Arc::strong_count(&mailbox) == 1 {
            iter += 1;
            tracing::error!("iteration {iter}");

            for y in 0..cli.height {
                let scene = scene.clone();
                let mailbox = mailbox.clone();
                let mut rng = rand::rngs::SmallRng::from_rng(&mut rng);
                let radiance = radiance.clone();

                rayon::spawn(move || {
                    for x in 0..cli.width {
                        let pixel = scene.pixel_radiance(x, cli.height - y - 1, &mut rng);
                        let r = &radiance[(y as usize * cli.width as usize + x as usize) * 3];
                        let g = &radiance[(y as usize * cli.width as usize + x as usize) * 3 + 1];
                        let b = &radiance[(y as usize * cli.width as usize + x as usize) * 3 + 2];

                        let old = Color {
                            r: f32::from_bits(r.load(std::sync::atomic::Ordering::Relaxed)),
                            g: f32::from_bits(g.load(std::sync::atomic::Ordering::Relaxed)),
                            b: f32::from_bits(b.load(std::sync::atomic::Ordering::Relaxed)),
                        };

                        let pixel = (old * (iter - 1) as f32 + pixel) / iter as f32;

                        r.store(pixel.r.to_bits(), std::sync::atomic::Ordering::Relaxed);
                        g.store(pixel.g.to_bits(), std::sync::atomic::Ordering::Relaxed);
                        b.store(pixel.b.to_bits(), std::sync::atomic::Ordering::Relaxed);
                    }

                    mailbox.push(y);
                });
            }
        }
        let mut render = false;
        for event in event_pump.poll_iter() {
            if let Some(mailbox) = event.as_user_event_type::<Arc<MailBox>>() {
                for y in mailbox.extract_pixels() {
                    for x in 0..cli.width {
                        let r = &radiance[(y as usize * cli.width as usize + x as usize) * 3];
                        let g = &radiance[(y as usize * cli.width as usize + x as usize) * 3 + 1];
                        let b = &radiance[(y as usize * cli.width as usize + x as usize) * 3 + 2];

                        let color = Color {
                            r: f32::from_bits(r.load(std::sync::atomic::Ordering::Relaxed)),
                            g: f32::from_bits(g.load(std::sync::atomic::Ordering::Relaxed)),
                            b: f32::from_bits(b.load(std::sync::atomic::Ordering::Relaxed)),
                        };

                        let color = scene.to_u8(color);
                        canvas.set_draw_color((color[0], color[1], color[2]));
                        canvas
                            .draw_point((x as i32, y as i32))
                            .map_err(eyre::Report::msg)?;
                        // unsafe {
                        //     let surface = surface.raw();
                        //     let pixels = (*surface).pixels as *mut [u8; 3];
                        //     let pitch = (*surface).pitch;
                        //     let pixel = pixels.byte_offset(y as isize * pitch as isize + x as isize);
                        //     *pixel = color;
                        // }
                    }
                }
                render = true;
                continue;
            }
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => return Ok(()),
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
                _ => {
                    // tracing::error!("{event:?}");
                }
            }
        }

        // The rest of the game loop goes here...

        canvas.present();

        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
    }
}
