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

use crate::denoiser::DenoiseThread;
use crate::display::Display;
use crate::raytracer::{Control, ViewerEvent};

mod denoiser;
mod display;
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

    /// Render one frame (the scene's samples per pixel, or -S) into this
    /// image file (format from the extension: png, jpg) and exit, without
    /// opening a window
    #[arg(long, value_name = "FILE")]
    pub save: Option<std::path::PathBuf>,

    /// No noise reduction: in the saved image, and at the viewer's start
    /// (D still toggles it)
    #[arg(long)]
    pub no_denoise: bool,

    /// Device that runs the noise reduction
    #[arg(long, value_enum, default_value_t)]
    pub denoise_device: denoiser::Device,
}

fn scene(cli: &mut Cli) -> Result<Scene, eyre::Report> {
    let mut scene = config::Config::builder()
        .add_source(config::File::with_name(&cli.scene))
        .build()?
        .try_deserialize::<SceneDefinition>()?;
    // Texture files are relative to the scene file.
    scene.load_textures(
        std::path::Path::new(&cli.scene)
            .parent()
            .unwrap_or(".".as_ref()),
    )?;
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
    if cli.mixture_sampling < 0f32 {
        cli.mixture_sampling = scene.mixture_sampling;
    } else {
        scene.mixture_sampling = cli.mixture_sampling;
    }
    if cli.mixture_decay < 0f32 {
        cli.mixture_decay = scene.mixture_decay;
    } else {
        scene.mixture_decay = cli.mixture_decay;
    }
    let scene = Scene::from(scene);
    tracing::debug!("{scene:?}");
    Ok(scene)
}

/// Renders one frame of `scene`, noise-reduced on `device` if `denoise` (and
/// built with the `denoise` feature), into the image file at `path`.
fn save(
    scene: &Scene,
    path: &std::path::Path,
    denoise: bool,
    #[allow(unused_variables)] device: denoiser::Device,
) -> Result<(), eyre::Report> {
    let started = Instant::now();
    #[allow(unused_mut)]
    let mut radiance = crate::raytracer::render_frame(scene);
    tracing::info!("rendered in {:#?}", started.elapsed());
    let (width, height) = (scene.camera.width, scene.camera.height);
    if denoise {
        #[cfg(feature = "denoise")]
        {
            let started = Instant::now();
            let denoiser = ::raytracer::Denoiser::new(device.into());
            denoiser
                .denoise(&mut radiance, width as usize, height as usize)
                .map_err(|err| eyre::eyre!("noise reduction failed: {err}"))?;
            let device = denoiser
                .device()
                .map_or("unknown".into(), |d| d.to_string());
            tracing::info!(
                "denoised on the {device} device in {:#?}",
                started.elapsed()
            );
        }
        #[cfg(not(feature = "denoise"))]
        tracing::warn!(
            "built without the denoise feature: saving the image without noise reduction"
        );
    }
    let bytes = radiance.iter().map(|&value| scene.to_u8(value)).collect();
    let image = image::RgbImage::from_raw(width, height, bytes)
        .ok_or_else(|| eyre::eyre!("the image is empty"))?;
    image
        .save(path)
        .map_err(|err| eyre::eyre!("could not save {}: {err}", path.display()))?;
    tracing::info!("saved {}", path.display());
    Ok(())
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
    if let Some(path) = &cli.save {
        return save(&scene, path, !cli.no_denoise, cli.denoise_device);
    }

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
    // thread streams finished rows into it through the SDL event queue, and
    // the denoising thread the denoised frames.
    let width = cli.width as usize;
    let height = cli.height as usize;
    let pitch = width * 3;
    let scene = std::sync::Arc::new(scene);
    let mut display = Display::new(scene.clone(), denoiser::AVAILABLE);
    if cli.no_denoise && display.enabled() {
        display.toggle();
    }
    let denoise_thread =
        DenoiseThread::spawn(width, height, cli.denoise_device, events.event_sender());
    let mut paused = false;
    let title = |paused: bool, display: &Display| {
        let mut title = String::from("Rust Tracer");
        if display.enabled() {
            title.push_str(" (denoised)");
        }
        if paused {
            title.push_str(" (paused)");
        }
        title
    };
    canvas
        .window_mut()
        .set_title(&title(paused, &display))
        .map_err(eyre::Report::msg)?;

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
                Some(ViewerEvent::Row { y, radiance }) => {
                    display.row(y, &radiance);
                    dirty = true;
                    continue;
                }
                Some(ViewerEvent::IterationDone { iter, snapshot }) => {
                    tracing::debug!("iteration #{iter}: displayed");
                    if let Some(request) = display.frame_done(iter, snapshot) {
                        denoise_thread.request(request);
                    }
                    continue;
                }
                Some(ViewerEvent::Denoised(denoised)) => {
                    dirty |= display.denoised(denoised);
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
                    paused = control.toggle_pause();
                    tracing::info!("{}", if paused { "Paused" } else { "Resumed" });
                    canvas
                        .window_mut()
                        .set_title(&title(paused, &display))
                        .map_err(eyre::Report::msg)?;
                }
                Event::KeyDown {
                    keycode: Some(Keycode::D),
                    ..
                } => {
                    if !denoiser::AVAILABLE {
                        tracing::info!("noise reduction is not built in (the `denoise` feature)");
                    }
                    if let Some(request) = display.toggle() {
                        denoise_thread.request(request);
                    }
                    let state = if display.enabled() { "on" } else { "off" };
                    tracing::info!("noise reduction {state}");
                    canvas
                        .window_mut()
                        .set_title(&title(paused, &display))
                        .map_err(eyre::Report::msg)?;
                    dirty = true;
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
                display.rgb(),
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
    denoise_thread.stop();

    Ok(())
}

#[cfg(test)]
mod test {
    use rand::RngExt as _;
    use rand::SeedableRng;

    fn scene(light_sampling: bool) -> ::raytracer::Scene {
        let mut definition = config::Config::builder()
            .add_source(config::File::with_name("pitracer.yml"))
            .build()
            .unwrap()
            .try_deserialize::<::raytracer::SceneDefinition>()
            .unwrap();
        definition.load_textures(".".as_ref()).unwrap();
        definition.light_sampling = light_sampling;
        // This helper feeds the NEE/PT firefly-bound test, whose
        // 2*E*(depth+1) bound is the NEE invariant; disable the (now-default)
        // mixture estimator so the test stays about NEE/PT.
        definition.mixture_sampling = 0.0;
        ::raytracer::Scene::from(definition)
    }

    /// The camera inside an emissive sphere that reflects nothing: every
    /// sample of every pixel is exactly the emission.
    #[test]
    fn save_renders_one_frame_into_a_file() {
        let definition = parse(
            "
camera: {origin: [0, 0, 0], direction: [0, 0, 1], width: 4, height: 3}
samples: 2
objects:
  - type: sphere
    radius: 10
    position: [0, 0, 0]
    color: [0, 0, 0]
    emission: [0.25, 0.5, 0.75]
",
        )
        .unwrap();
        let scene = ::raytracer::Scene::from(definition);
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("render.png");
        super::save(&scene, &path, false, Default::default()).unwrap();

        let image = image::open(&path).unwrap().into_rgb8();
        assert_eq!(image.dimensions(), (4, 3));
        let expected = [0.25f32, 0.5, 0.75].map(|value| scene.to_u8(value));
        assert!(image.pixels().all(|pixel| pixel.0 == expected));

        // The format comes from the extension.
        assert!(
            super::save(
                &scene,
                &folder.path().join("render.unknown"),
                false,
                Default::default()
            )
            .is_err()
        );
    }

    fn parse(yaml: &str) -> Result<::raytracer::SceneDefinition, config::ConfigError> {
        config::Config::builder()
            .add_source(config::File::from_str(yaml, config::FileFormat::Yaml))
            .build()
            .unwrap()
            .try_deserialize::<::raytracer::SceneDefinition>()
    }

    /// The shipped scene must survive the whole deserialization stack (config
    /// crate -> serde flatten -> the `type` tag -> the geometry definition
    /// modules): 14 objects (ellipse light + 6 ball spheres, then 6 quad
    /// walls and the logo quad), the marble ball's and the logo's `texture:`
    /// tags, and a spot-checked wall whose geometry and color arrived intact.
    #[test]
    fn pitracer_yaml_parses_objects() {
        let definition = config::Config::builder()
            .add_source(config::File::with_name("pitracer.yml"))
            .build()
            .unwrap()
            .try_deserialize::<::raytracer::SceneDefinition>()
            .unwrap();
        assert_eq!(definition.objects.len(), 14);
        let quads = definition
            .objects
            .iter()
            .filter(|o| matches!(o.geometry, ::raytracer::GeometryEnum::Quad(_)))
            .count();
        assert_eq!(quads, 7);
        assert!(matches!(
            definition.objects[0].geometry,
            ::raytracer::GeometryEnum::Ellipse(_)
        ));
        // the white ball and the logo carry the scene's `texture:` tags
        assert!(matches!(
            definition.objects[3].texture,
            ::raytracer::TextureEnum::Marble(_)
        ));
        assert!(matches!(
            definition.objects[10].texture,
            ::raytracer::TextureEnum::Image(_)
        ));

        let left = &definition.objects[7];
        assert_eq!(
            left.geometry,
            ::raytracer::GeometryEnum::Quad(::raytracer::Quad::new(
                (1f32, -5f32, -5f32).into(),
                (0f32, 91.6f32, 0f32).into(),
                (0f32, 0f32, 310f32).into(),
            ))
        );
        assert_eq!(
            left.texture,
            ::raytracer::TextureEnum::Flat(::raytracer::Flat {
                color: (0.75f32, 0.25f32, 0.25f32).into(),
                emission: Default::default(),
            })
        );
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
        assert_eq!(
            quad.texture,
            ::raytracer::TextureEnum::Flat(::raytracer::Flat {
                color: (0.1f32, 0.2f32, 0.3f32).into(),
                emission: (1f32, 2f32, 3f32).into(),
            })
        );
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
    /// a silent default: geometry has no sensible fallback.
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
  - type: Ellipse
    position: [0, 0, 5]
    u: [1, 0, 0]
    v: [0, 1, 0]
",
        )
        .unwrap();
        assert_eq!(definition.objects.len(), 3);
        assert!(matches!(
            definition.objects[0].geometry,
            ::raytracer::GeometryEnum::Sphere(_)
        ));
        assert!(matches!(
            definition.objects[1].geometry,
            ::raytracer::GeometryEnum::Quad(_)
        ));
        assert!(matches!(
            definition.objects[2].geometry,
            ::raytracer::GeometryEnum::Ellipse(_)
        ));
    }

    /// An ellipse entry round-trips through the flatten + definition path:
    /// geometry (center + semi-axes) and texture arrive intact.
    #[test]
    fn ellipse_entry_parses() {
        let definition = parse(
            "\
objects:
  - type: ellipse
    position: [50, 81.5, 61.6]
    u: [12, 0, 0]
    v: [0, 0, 8]
    color: [0.1, 0.2, 0.3]
",
        )
        .unwrap();
        let light = &definition.objects[0];
        assert_eq!(
            light.geometry,
            ::raytracer::GeometryEnum::Ellipse(::raytracer::Ellipse::new(
                (50f32, 81.5f32, 61.6f32).into(),
                (12f32, 0f32, 0f32).into(),
                (0f32, 0f32, 8f32).into(),
            ))
        );
        assert_eq!(
            light.texture,
            ::raytracer::TextureEnum::Flat(::raytracer::Flat {
                color: (0.1f32, 0.2f32, 0.3f32).into(),
                emission: Default::default(),
            })
        );
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

    /// A `texture: marble` entry lands in the Marble variant with its knobs
    /// intact (lowercase tag; Title case tolerated via alias like `type:`).
    #[test]
    fn marble_texture_parses() {
        for tag in ["marble", "Marble"] {
            let definition = parse(&format!(
                "\
objects:
  - type: sphere
    radius: 10
    position: [15, 45, 112]
    texture: {tag}
    color: [0.9, 0.9, 0.9]
    vein_color: [0.3, 0.25, 0.35]
    scale: 1.5
    turbulence: 4
    octaves: 3
    seed: 7
"
            ))
            .unwrap();
            assert_eq!(
                definition.objects[0].texture,
                ::raytracer::TextureEnum::Marble(::raytracer::Marble {
                    color: (0.9f32, 0.9f32, 0.9f32).into(),
                    vein_color: (0.3f32, 0.25f32, 0.35f32).into(),
                    scale: 1.5,
                    turbulence: 4.0,
                    octaves: 3,
                    seed: 7,
                    ..Default::default()
                }),
                "tag `{tag}`"
            );
        }
    }

    /// An object entry without a `texture:` key defaults to Flat, so plain
    /// `color:`/`emission:` entries keep working.
    #[test]
    fn absent_texture_defaults_to_flat() {
        let definition = parse(
            "\
objects:
  - type: sphere
    radius: 1
    color: [0.1, 0.2, 0.3]
",
        )
        .unwrap();
        assert_eq!(
            definition.objects[0].texture,
            ::raytracer::TextureEnum::Flat(::raytracer::Flat {
                color: (0.1f32, 0.2f32, 0.3f32).into(),
                emission: Default::default(),
            })
        );
    }

    /// Documented migration failure mode: the old top-level `spheres:` /
    /// `quads:` keys are unknown to `SceneDefinition` and silently dropped
    /// (`#[serde(default)]` container), so an unmigrated file yields an empty
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
    /// and below the physical path bound 2*E_max*(depth+1). Fireflies from
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
