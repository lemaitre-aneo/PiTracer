extern crate sdl2;

use ::raytracer::Scene;
use ::raytracer::SceneDefinition;
use clap::Parser;
use sdl2::event::Event;
use sdl2::event::WindowEvent;
use sdl2::keyboard::Keycode;
use sdl2::rect::Rect;
use std::time::{Duration, Instant};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::raytracer::{Control, ViewerEvent};

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

    /// Light Sampling [default: true]
    #[arg(long)]
    pub light_sampling: Option<bool>,
}

fn scene(cli: &mut Cli) -> Result<Scene, eyre::Report> {
    let mut scene = config::Config::builder()
        .add_source(config::File::with_name(&cli.scene))
        .build()?
        .try_deserialize::<SceneDefinition>()?;
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
    if let Some(ls) = cli.light_sampling {
        scene.light_sampling = ls;
    } else {
        cli.light_sampling = Some(scene.light_sampling);
    }
    let scene = Scene::from(scene);
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
    let events = sdl.event().map_err(eyre::Report::msg)?;
    events
        .register_custom_event::<ViewerEvent>()
        .map_err(eyre::Report::msg)?;

    let window = video_subsystem
        .window("Rust Tracer", cli.width, cli.height)
        .position_centered()
        .resizable()
        .build()?;

    let mut canvas = window.into_canvas().build()?;
    let texture_creator = canvas.texture_creator();

    let mut event_pump = sdl.event_pump().map_err(eyre::Report::msg)?;

    // The UI thread exclusively owns the displayed framebuffer; the render
    // thread streams finished rows into it through the SDL event queue.
    let width = cli.width as usize;
    let height = cli.height as usize;
    let pitch = width * 3;
    let mut pixels = vec![0u8; pitch * height];

    let control = Control::new();
    let render_thread = std::thread::spawn({
        let control = control.clone();
        let sender = events.event_sender();
        move || crate::raytracer::render_thread(scene, control, sender)
    });

    const PRESENT_INTERVAL: Duration = Duration::from_millis(15);
    let mut dirty = true;
    let mut last_present = Instant::now() - PRESENT_INTERVAL;

    'running: loop {
        // Block until something happens, then drain the whole queue so a burst
        // of row events is coalesced into a single present.
        let mut pending = Vec::new();
        if let Some(event) = event_pump.wait_event_timeout(100) {
            pending.push(event);
            pending.extend(event_pump.poll_iter());
        }

        for event in pending {
            // Single registered custom type: extract at most once per event
            // (the payload is moved out of a raw box; never clone user events).
            match event.as_user_event_type::<ViewerEvent>() {
                Some(ViewerEvent::Row { y, pixels: row }) => {
                    pixels[y * pitch..(y + 1) * pitch].copy_from_slice(&row);
                    dirty = true;
                    continue;
                }
                Some(ViewerEvent::IterationDone { iter }) => {
                    tracing::debug!("iteration #{iter}: displayed");
                    continue;
                }
                None => {}
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
                    dirty = true;
                }
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => {
                    tracing::info!("Stopping Requested");
                    break 'running;
                }
                Event::KeyDown {
                    keycode: Some(Keycode::Space),
                    ..
                } => {
                    let title = if control.toggle_pause() {
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
                    dirty = true;
                }
                _ => {}
            }
        }

        if dirty && last_present.elapsed() >= PRESENT_INTERVAL {
            let surface = sdl2::surface::Surface::from_data(
                &mut pixels,
                cli.width,
                cli.height,
                pitch as u32,
                sdl2::pixels::PixelFormatEnum::RGB24,
            )
            .map_err(eyre::Report::msg)?;
            let texture = surface
                .as_texture(&texture_creator)
                .map_err(eyre::Report::msg)?;

            let (w, h) = canvas.output_size().map_err(eyre::Report::msg)?;
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

            dirty = false;
            last_present = Instant::now();
        }
    }

    // Wind the render thread down before tearing SDL down: workers must not
    // push into a dead queue, and a clean join instead of a mid-render return
    // is what keeps a worker panic from aborting the process.
    control.stop();
    render_thread
        .join()
        .map_err(|_| eyre::eyre!("render thread panicked"))?;

    Ok(())
}

#[cfg(test)]
mod test {
    use rand::Rng as _;
    use rand::SeedableRng;

    fn scene(light_sampling: bool) -> ::raytracer::Scene {
        let mut definition = config::Config::builder()
            .add_source(config::File::with_name("pitracer.yml"))
            .build()
            .unwrap()
            .try_deserialize::<::raytracer::SceneDefinition>()
            .unwrap();
        definition.light_sampling = light_sampling;
        ::raytracer::Scene::from(definition)
    }

    fn parse(yaml: &str) -> Result<::raytracer::SceneDefinition, config::ConfigError> {
        config::Config::builder()
            .add_source(config::File::from_str(yaml, config::FileFormat::Yaml))
            .build()
            .unwrap()
            .try_deserialize::<::raytracer::SceneDefinition>()
    }

    /// The shipped scene must survive the whole deserialization stack (config
    /// crate → serde flatten → the `type` tag → the geometry definition
    /// modules): 13 objects (light + 6 balls, then 6 quad walls), and a
    /// spot-checked wall whose geometry and color arrived intact.
    #[test]
    fn pitracer_yaml_parses_objects() {
        let definition = config::Config::builder()
            .add_source(config::File::with_name("pitracer.yml"))
            .build()
            .unwrap()
            .try_deserialize::<::raytracer::SceneDefinition>()
            .unwrap();
        assert_eq!(definition.objects.len(), 13);
        let quads = definition
            .objects
            .iter()
            .filter(|o| matches!(o.geometry, ::raytracer::GeometryEnum::Quad(_)))
            .count();
        assert_eq!(quads, 6);

        let left = &definition.objects[7];
        assert_eq!(
            left.geometry,
            ::raytracer::GeometryEnum::Quad(::raytracer::Quad::new(
                (1f32, -5f32, -5f32).into(),
                (0f32, 91.6f32, 0f32).into(),
                (0f32, 0f32, 310f32).into(),
            ))
        );
        assert_eq!(left.texture.color, (0.75f32, 0.25f32, 0.25f32).into());
    }

    /// One quad entry carries geometry, material, and texture keys in a
    /// single YAML map.
    #[test]
    fn quad_entry_carries_material_and_texture() {
        let definition = parse(
            "\
objects:
  - type: quad
    position: [1, 2, 3]
    u: [1, 0, 0]
    v: [0, 1, 0]
    material: Metal
    fuzziness: 0.25
    color: [0.1, 0.2, 0.3]
    emission: [1, 2, 3]
",
        )
        .unwrap();
        let quad = &definition.objects[0];
        assert_eq!(
            quad.material,
            ::raytracer::MaterialEnum::Metal(::raytracer::Metal { fuzziness: 0.25 })
        );
        assert_eq!(quad.texture.emission, (1f32, 2f32, 3f32).into());
    }

    /// The pre-rename `fuziness` spelling stays accepted as an alias; both
    /// spellings land in the same field.
    #[test]
    fn fuzziness_accepts_legacy_spelling() {
        for spelling in ["fuzziness", "fuziness"] {
            let definition = parse(&format!(
                "\
objects:
  - type: sphere
    radius: 1
    material: Metal
    {spelling}: 0.25
"
            ))
            .unwrap();
            assert_eq!(
                definition.objects[0].material,
                ::raytracer::MaterialEnum::Metal(::raytracer::Metal { fuzziness: 0.25 }),
                "spelling `{spelling}`"
            );
        }
    }

    /// An object entry without a `type` field must be a hard parse error, not
    /// a silent default — geometry has no sensible fallback.
    #[test]
    fn untyped_object_is_rejected() {
        let result = parse(
            "\
objects:
  - radius: 10
    position: [50, 81.6, 61.6]
    emission: [10, 10, 10]
",
        );
        assert!(result.is_err(), "missing `type` must fail: {result:?}");
    }

    /// An unknown `type` value must be a hard parse error.
    #[test]
    fn unknown_type_is_rejected() {
        let result = parse(
            "\
objects:
  - type: cube
    position: [1, 2, 3]
",
        );
        assert!(result.is_err(), "unknown `type` must fail: {result:?}");
    }

    /// Title-case tags are tolerated via serde aliases.
    #[test]
    fn title_case_type_accepted() {
        let definition = parse(
            "\
objects:
  - type: Sphere
    radius: 1
    position: [0, 0, 5]
  - type: Quad
    position: [0, 0, 5]
    u: [1, 0, 0]
    v: [0, 1, 0]
",
        )
        .unwrap();
        assert_eq!(definition.objects.len(), 2);
        assert!(matches!(
            definition.objects[0].geometry,
            ::raytracer::GeometryEnum::Sphere(_)
        ));
        assert!(matches!(
            definition.objects[1].geometry,
            ::raytracer::GeometryEnum::Quad(_)
        ));
    }

    /// An object entry without a `material:` key defaults to Lambertian.
    #[test]
    fn absent_material_defaults_to_lambertian() {
        let definition = parse(
            "\
objects:
  - type: sphere
    radius: 1
",
        )
        .unwrap();
        assert_eq!(
            definition.objects[0].material,
            ::raytracer::MaterialEnum::Lambertian(::raytracer::Lambertian {})
        );
    }

    /// Documented migration failure mode: the old top-level `spheres:` /
    /// `quads:` keys are unknown to `SceneDefinition` and silently dropped
    /// (`#[serde(default)]` container) — an unmigrated file yields an empty
    /// scene rather than an error. Pinned here; catalogued in BACKLOG P0.
    #[test]
    fn legacy_top_level_keys_are_dropped() {
        let definition = parse(
            "\
spheres:
  - radius: 1
    position: [50, 81.6, 61.6]
    emission: [10, 10, 10]
",
        )
        .unwrap();
        assert!(definition.objects.is_empty());
    }

    /// Regression: on the real scene, every radiance sample must be finite
    /// and below the physical path bound 2·E_max·(depth+1) — fireflies from
    /// estimator reweighting showed up here as saturated specks.
    #[test]
    fn pitracer_radiance_finite_and_bounded() {
        for light_sampling in [false, true] {
            let scene = scene(light_sampling);
            let bound = 2.0 * 10.0 * (scene.recursion_depth + 1) as f32;
            let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
            let mut max = 0f32;
            for py in (0..scene.camera.height).step_by(4) {
                for px in (0..scene.camera.width).step_by(4) {
                    for _ in 0..8 {
                        let x = px as f32 + rng.random::<f32>();
                        let y = py as f32 + rng.random::<f32>();
                        let c = scene.radiance(x, y, &mut rng);
                        for v in [c.r, c.g, c.b] {
                            assert!(
                                v.is_finite(),
                                "non-finite radiance at ({px},{py}), light_sampling={light_sampling}: {c:?}",
                            );
                            max = max.max(v);
                        }
                    }
                }
            }
            eprintln!("light_sampling={light_sampling}: max sample = {max:.1} (bound {bound})");
            assert!(
                max <= bound,
                "firefly at light_sampling={light_sampling}: max {max} exceeds bound {bound}",
            );
        }
    }
}
