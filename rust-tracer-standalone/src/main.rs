extern crate sdl2;

use clap::Parser;
use rand::SeedableRng;
use raytracer::{Camera, CameraDefinition, Scene, Sphere, SphereVec, Vector};
use sdl2::event::EventSender;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::{EventSubsystem, Sdl};
use sdl2::{event::Event, surface::Surface};
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
    #[arg(short = 'd', long, default_value = "0")]
    pub depth: u32,
}

struct Pixel {
    x: u32,
    y: u32,
    color: [u8; 3],
}

struct MailBox {
    pixels: Mutex<Vec<Pixel>>,
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

    fn push(self: Arc<Self>, pixel: Pixel) {
        let mut pixels = self.pixels.lock().unwrap();
        pixels.push(pixel);

        if pixels.len() == 1 {
            self.event_sender.push_custom_event(self.clone()).unwrap();
        }
    }

    fn extract_pixels(&self) -> Vec<Pixel> {
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

    for x in 0..cli.width {
        for y in 0..cli.height {
            let scene = scene.clone();
            let mailbox = mailbox.clone();
            let mut rng = rand::rngs::SmallRng::from_rng(&mut rng);

            rayon::spawn(move || {
                let pixel = scene.pixel_radiance(x, cli.height - y - 1, &mut rng);
                let color = [
                    pixel.r.clamp(0f32, 1f32),
                    pixel.g.clamp(0f32, 1f32),
                    pixel.b.clamp(0f32, 1f32),
                ]
                .map(|l| (l.powf(1f32 / 2.5f32) * 255f32 + 0.5f32) as u8);

                mailbox.push(Pixel { x, y, color });
            });
        }
    }

    let mut finished = false;

    loop {
        if Arc::strong_count(&mailbox) == 1 && !finished {
            tracing::error!("Finished!");
            finished = true;
        }
        let mut render = false;
        for event in event_pump.poll_iter() {
            if let Some(mailbox) = event.as_user_event_type::<Arc<MailBox>>() {
                for Pixel { x, y, color } in mailbox.extract_pixels() {
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
