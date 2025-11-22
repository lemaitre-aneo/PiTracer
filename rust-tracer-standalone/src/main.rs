extern crate sdl2;

use clap::Parser;
use rand::SeedableRng;
use raytracer::{CameraDefinition, Color, Scene};
use sdl2::EventSubsystem;
use sdl2::event::{EventSender, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::rect::Rect;
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
    pixels: Mutex<bool>,
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

    fn signal(self: Arc<Self>) {
        let mut pending = self.pixels.lock().unwrap();

        if !*pending {
            self.event_sender.push_custom_event(self.clone()).unwrap();
        }

        *pending = true;
    }

    fn acknowledge(&self) {
        *self.pixels.lock().unwrap() = false;
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
    let scene = Arc::new(scene(&mut cli)?);

    let sdl = sdl2::init().map_err(eyre::Report::msg)?;
    let video_subsystem = sdl.video().map_err(eyre::Report::msg)?;

    let window = video_subsystem
        .window("Rust Tracer", cli.width, cli.height)
        .position_centered()
        .build()?;

    let mut canvas = window.into_canvas().build()?;
    let texture_creator = canvas.texture_creator();
    let mailbox = Arc::new(MailBox::new(&sdl.event().map_err(eyre::Report::msg)?));

    canvas.present();
    let mut event_pump = sdl.event_pump().map_err(eyre::Report::msg)?;

    let mut rng = rand::rngs::StdRng::from_os_rng();

    let total_size = cli.width as usize * cli.height as usize;
    let mut radiance = Vec::<AtomicU32>::with_capacity(total_size * 3);
    radiance.resize_with(total_size * 3, Default::default);
    let radiance = Arc::new(radiance);
    let mut image = Vec::<AtomicU32>::with_capacity(total_size);
    image.resize_with(total_size, Default::default);
    let image = Arc::new(image);

    let mut iter = 0;
    let mut paused = false;
    let mut started = None;

    loop {
        let mut render = false;
        let mut clear = false;
        for event in event_pump.poll_iter() {
            if let Some(mailbox) = event.as_user_event_type::<Arc<MailBox>>() {
                mailbox.acknowledge();
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
                    clear = true;
                    render = true;
                }
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => return Ok(()),
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
                    clear = true;
                    render = true;
                }
                _ => {}
            }
        }

        if Arc::strong_count(&mailbox) == 1 {
            if let Some(started) = started.take() {
                let finished = std::time::Instant::now();
                tracing::info!("Iteration #{iter}: finished in {:#?}", finished - started);
            }
            if !paused {
                iter += 1;
                tracing::debug!("iteration #{iter}: started");
                started = Some(std::time::Instant::now());

                for y in 0..cli.height {
                    let scene = scene.clone();
                    let mailbox = mailbox.clone();
                    let mut rng = rand::rngs::SmallRng::from_rng(&mut rng);
                    let radiance = radiance.clone();
                    let image = image.clone();

                    rayon::spawn(move || {
                        for x in 0..cli.width {
                            let pixel = scene.pixel_radiance(x, cli.height - y - 1, &mut rng);
                            let r = &radiance[(y as usize * cli.width as usize + x as usize) * 3];
                            let g =
                                &radiance[(y as usize * cli.width as usize + x as usize) * 3 + 1];
                            let b =
                                &radiance[(y as usize * cli.width as usize + x as usize) * 3 + 2];

                            let old = Color {
                                r: f32::from_bits(r.load(std::sync::atomic::Ordering::Relaxed)),
                                g: f32::from_bits(g.load(std::sync::atomic::Ordering::Relaxed)),
                                b: f32::from_bits(b.load(std::sync::atomic::Ordering::Relaxed)),
                            };

                            let pixel = (old * (iter - 1) as f32 + pixel) / iter as f32;

                            r.store(pixel.r.to_bits(), std::sync::atomic::Ordering::Relaxed);
                            g.store(pixel.g.to_bits(), std::sync::atomic::Ordering::Relaxed);
                            b.store(pixel.b.to_bits(), std::sync::atomic::Ordering::Relaxed);

                            let color = scene.to_u8(pixel);
                            let color = (255 << 24)
                                | ((color[0] as u32) << 16)
                                | ((color[1] as u32) << 8)
                                | (color[2] as u32);

                            image[y as usize * cli.width as usize + x as usize]
                                .store(color, std::sync::atomic::Ordering::Relaxed);
                        }

                        mailbox.signal();
                    });
                }
            }
        }

        if render {
            let data = unsafe {
                std::slice::from_raw_parts_mut(image.as_slice().as_ptr() as *mut u8, total_size * 4)
            };
            let surface = Surface::from_data(
                data,
                cli.width,
                cli.height,
                cli.width * 4,
                sdl2::pixels::PixelFormatEnum::ARGB8888,
            )
            .map_err(eyre::Report::msg)?;
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

            if clear {
                canvas.clear();
            }

            canvas
                .copy(&texture, None, target)
                .map_err(eyre::Report::msg)?;
            canvas.present();
        }

        ::std::thread::sleep(Duration::from_secs_f64(1./30.));
    }
}
