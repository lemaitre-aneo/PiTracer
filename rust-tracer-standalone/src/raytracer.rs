//! Progressive row-streaming renderer.
//!
//! A dedicated render thread owns the radiance accumulator and loops full-image
//! iterations; rayon workers each own one row at a time (`chunks_mut` +
//! `par_bridge`, handing rows out in top-to-bottom order), so all rendering
//! math runs on exclusively-owned plain memory. Finished rows are
//! published to the UI thread through the SDL event queue (`ViewerEvent`),
//! which exclusively owns the surface it displays. There is no shared
//! framebuffer, no lock on the render path, and no `unsafe` anywhere: partial
//! frames remain the intended normal state, with row granularity.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use rand::rngs::SmallRng;
use rand::{Rng as _, SeedableRng};
use rayon::prelude::*;
use raytracer::{Color, Scene};
use sdl2::event::EventSender;

/// Messages streamed from the render thread to the SDL main loop through the
/// SDL event queue.
///
/// This is the single registered custom event type: the main loop extracts the
/// payload exactly once per received event with
/// `Event::as_user_event_type::<ViewerEvent>()`. Never clone a user event and
/// never extract twice — the payload is moved out of a raw box.
pub enum ViewerEvent {
    /// One finished row of pixels (tight `width * 3` RGB24 bytes).
    Row { y: usize, pixels: Vec<u8> },
    /// A full pass over the image completed.
    IterationDone { iter: u32 },
}

/// Control surface shared between the UI and the render thread.
pub struct Control {
    stopping: AtomicBool,
    paused: Mutex<bool>,
    unpause: Condvar,
}

impl Control {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            stopping: AtomicBool::new(false),
            paused: Mutex::new(false),
            unpause: Condvar::new(),
        })
    }

    /// Requests the render thread to wind down (also wakes it from a pause).
    pub fn stop(&self) {
        self.stopping.store(true, Ordering::Relaxed);
        let _paused = self.paused.lock().unwrap();
        self.unpause.notify_all();
    }

    pub fn stopping(&self) -> bool {
        self.stopping.load(Ordering::Relaxed)
    }

    /// Toggles pause and returns the new state.
    pub fn toggle_pause(&self) -> bool {
        let mut paused = self.paused.lock().unwrap();
        *paused = !*paused;
        self.unpause.notify_all();
        *paused
    }

    /// Blocks between iterations while paused (returns early when stopping).
    fn wait_if_paused(&self) {
        let mut paused = self.paused.lock().unwrap();
        while *paused && !self.stopping() {
            paused = self.unpause.wait(paused).unwrap();
        }
    }
}

/// Accumulates one more sample pass of row `y` into its running average.
/// Returns false when interrupted by a stop request (row left incomplete).
fn render_row(
    scene: &Scene,
    y: usize,
    iter: u32,
    accumulator: &mut [f32],
    control: &Control,
    rng: &mut SmallRng,
) -> bool {
    let height = scene.camera.height as usize;
    for x in 0..scene.camera.width as usize {
        if control.stopping() {
            return false;
        }
        // Row y of the buffer displays image row y from the top; the camera's
        // v axis points up, hence the flip.
        let pixel = scene.pixel_radiance(x as u32, (height - y - 1) as u32, rng);

        let accumulator = &mut accumulator[x * 3..x * 3 + 3];
        let old = Color {
            r: accumulator[0],
            g: accumulator[1],
            b: accumulator[2],
        };
        let new = (old * (iter - 1) as f32 + pixel) / iter as f32;
        accumulator[0] = new.r;
        accumulator[1] = new.g;
        accumulator[2] = new.b;
    }
    true
}

/// Converts a radiance row to tight RGB24 bytes.
fn convert_row(scene: &Scene, accumulator: &[f32]) -> Vec<u8> {
    accumulator
        .iter()
        .map(|&value| scene.to_u8(value))
        .collect()
}

/// Body of the render thread: loops full-image iterations until stopped,
/// streaming each finished row to the UI through `sender`.
pub fn render_thread(scene: Scene, control: Arc<Control>, sender: EventSender) {
    let width = scene.camera.width as usize;
    let height = scene.camera.height as usize;
    // Per-(iteration, row) rngs are seeded through SplitMix (`seed_from_u64`),
    // never from raw constant state — see the test-suite seeding convention.
    let seed_base = rand::rngs::StdRng::from_os_rng().random::<u64>();

    let mut radiance = vec![0f32; width * 3 * height];

    for iter in 1u32.. {
        control.wait_if_paused();
        if control.stopping() {
            break;
        }

        let started = std::time::Instant::now();
        // par_bridge (not par_chunks_mut): workers pull the NEXT row from one
        // ordered queue instead of splitting the image into per-worker
        // regions, so rows complete top-to-bottom with only thread-skew
        // reordering — the intended progressive feel. The serialized pull is
        // one lock per row, noise against a row's render cost.
        radiance
            .chunks_mut(width * 3)
            .enumerate()
            .par_bridge()
            .for_each(|(y, accumulator)| {
                let mut rng = SmallRng::seed_from_u64(seed_base ^ ((iter as u64) << 32) ^ y as u64);
                if render_row(&scene, y, iter, accumulator, &control, &mut rng) {
                    let pixels = convert_row(&scene, accumulator);
                    // A full queue only costs a stale row until the next
                    // iteration repaints it.
                    if sender
                        .push_custom_event(ViewerEvent::Row { y, pixels })
                        .is_err()
                    {
                        tracing::debug!("dropped row {y} (event queue full or shutting down)");
                    }
                }
            });

        if control.stopping() {
            break;
        }
        tracing::info!("iteration #{iter}: finished in {:#?}", started.elapsed());
        sender
            .push_custom_event(ViewerEvent::IterationDone { iter })
            .ok();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ::raytracer::{
        CameraDefinition, Flat, GeometryEnum, Lambertian, MaterialEnum, Object, SceneDefinition,
        Sphere, TextureEnum, Vector,
    };

    /// Camera enclosed in a purely emissive sphere: every sample of every
    /// pixel is exactly E = (0.25, 0.5, 0.75) — deterministic rows.
    fn test_scene() -> Scene {
        Scene::from(SceneDefinition {
            camera: CameraDefinition {
                origin: Vector::default(),
                direction: Vector::from((0f32, 0f32, 1f32)),
                field_of_view: 90f32,
                width: 4,
                height: 4,
                world_height: 0.1,
            },
            objects: vec![Object::new(
                GeometryEnum::Sphere(Sphere {
                    radius: 10f32,
                    position: Vector::default(),
                }),
                MaterialEnum::Lambertian(Lambertian {}),
                TextureEnum::Flat(Flat {
                    color: (0f32, 0f32, 0f32).into(),
                    emission: (0.25f32, 0.5f32, 0.75f32).into(),
                }),
            )],
            recursion_depth: 5,
            samples: 4,
            gamma: 2.2,
            light_sampling: false,
        })
    }

    #[test]
    fn first_iteration_writes_samples_exactly() {
        let scene = test_scene();
        let control = Control::new();
        let mut accumulator = vec![0f32; 4 * 3];
        let mut rng = SmallRng::seed_from_u64(1);

        assert!(render_row(
            &scene,
            0,
            1,
            &mut accumulator,
            &control,
            &mut rng
        ));
        for pixel in accumulator.chunks(3) {
            assert_eq!(pixel, [0.25, 0.5, 0.75]);
        }
    }

    #[test]
    fn running_average_matches_closed_form() {
        let scene = test_scene();
        let control = Control::new();
        // pre-fill with a different value: iteration 2 must blend (old + E)/2
        let mut accumulator = vec![0.5f32; 4 * 3];
        let mut rng = SmallRng::seed_from_u64(2);

        assert!(render_row(
            &scene,
            1,
            2,
            &mut accumulator,
            &control,
            &mut rng
        ));
        for pixel in accumulator.chunks(3) {
            assert_eq!(pixel[0], (0.5 + 0.25) / 2.0);
            assert_eq!(pixel[1], (0.5 + 0.5) / 2.0);
            assert_eq!(pixel[2], (0.5 + 0.75) / 2.0);
        }
    }

    #[test]
    fn stop_interrupts_row() {
        let scene = test_scene();
        let control = Control::new();
        control.stop();
        let mut accumulator = vec![0f32; 4 * 3];
        let mut rng = SmallRng::seed_from_u64(3);

        assert!(!render_row(
            &scene,
            0,
            1,
            &mut accumulator,
            &control,
            &mut rng
        ));
        assert!(
            accumulator.iter().all(|&v| v == 0f32),
            "stopped row must stay untouched"
        );
    }

    #[test]
    fn convert_row_matches_scene_to_u8() {
        let scene = test_scene();
        let accumulator = [0f32, 0.25, 0.5, 0.75, 1.0, 2.0];
        let converted = convert_row(&scene, &accumulator);
        for (byte, value) in converted.iter().zip(accumulator) {
            assert_eq!(*byte, scene.to_u8(value));
        }
    }

    #[test]
    fn stop_wakes_a_paused_control() {
        let control = Control::new();
        assert!(control.toggle_pause());
        control.stop();
        // must return instead of blocking forever
        control.wait_if_paused();
        assert!(control.stopping());

        let control = Control::new();
        assert!(control.toggle_pause());
        assert!(!control.toggle_pause());
        control.wait_if_paused(); // unpaused: returns immediately
    }
}
