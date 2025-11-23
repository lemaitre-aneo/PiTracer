use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32},
    },
};

use rand::{RngCore, SeedableRng};
use raytracer::{Color, Scene};
use sdl2::{Sdl, event::EventSender, surface::Surface};

pub struct Raytracer {
    pub scene: Scene,
    pending: Mutex<bool>,
    event_sender: EventSender,
    radiance: *mut f32,
    pixels: *mut u8,
    stopping: AtomicBool,
    iter: AtomicU32,
}

unsafe impl Send for Raytracer {}
unsafe impl Sync for Raytracer {}

impl Drop for Raytracer {
    fn drop(&mut self) {
        let layout = self.layout();
        let total_size = layout.height * layout.pitch;

        unsafe {
            std::alloc::dealloc(
                self.radiance as *mut u8,
                std::alloc::Layout::from_size_align(total_size * 4, 16).unwrap_unchecked(),
            );
            std::alloc::dealloc(
                self.pixels,
                std::alloc::Layout::from_size_align(total_size, 16).unwrap_unchecked(),
            );
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Layout {
    height: usize,
    width: usize,
    pitch: usize,
}

impl Raytracer {
    fn layout_from_scene(scene: &Scene) -> Layout {
        let width = scene.camera.width as usize;
        let height = scene.camera.height as usize;
        let pitch = (width * 3).next_multiple_of(16);

        Layout {
            height,
            width,
            pitch,
        }
    }

    fn layout(&self) -> Layout {
        Self::layout_from_scene(&self.scene)
    }

    pub fn new(scene: Scene, sdl: &Sdl) -> Result<Arc<Self>, String> {
        let events = sdl.event()?;
        events.register_custom_event::<Arc<Raytracer>>()?;

        let layout = Self::layout_from_scene(&scene);
        let total_size = layout.height * layout.pitch;

        Ok(Arc::new(Self {
            scene,
            pending: Mutex::new(false),
            event_sender: events.event_sender(),
            radiance: unsafe {
                std::alloc::alloc(
                    std::alloc::Layout::from_size_align(
                        total_size * std::mem::size_of::<f32>(),
                        16,
                    )
                    .unwrap_unchecked(),
                ) as *mut f32
            },
            pixels: unsafe {
                std::alloc::alloc(
                    std::alloc::Layout::from_size_align(
                        total_size * std::mem::size_of::<u8>(),
                        16,
                    )
                    .unwrap_unchecked(),
                )
            },
            stopping: AtomicBool::new(false),
            iter: AtomicU32::new(0),
        }))
    }

    pub fn iter(&self) -> u32 {
        self.iter.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn signal(self: Arc<Self>) {
        let mut pending = self.pending.lock().unwrap();

        if !*pending {
            self.event_sender.push_custom_event(self.clone()).unwrap();
        }

        *pending = true;
    }

    pub fn acknowledge(&self) {
        *self.pending.lock().unwrap() = false;
    }

    pub fn get_surface(&self) -> Result<Surface<'_>, String> {
        let layout = self.layout();

        Surface::from_data(
            unsafe { std::slice::from_raw_parts_mut(self.pixels, layout.height * layout.pitch) },
            layout.width as u32,
            layout.height as u32,
            layout.pitch as u32,
            sdl2::pixels::PixelFormatEnum::RGB24,
        )
    }

    pub fn send_process(self: Arc<Self>, mut rng: &mut dyn RngCore) {
        let layout = self.layout();
        let iter = self.iter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

        for y in 0..layout.height {
            let mut rng = rand::rngs::SmallRng::from_rng(&mut rng);
            let this = self.clone();

            rayon::spawn(move || {
                for x in 0..layout.width {
                    if this.stopping() {
                        return;
                    }
                    let pixel = this.scene.pixel_radiance(
                        x as u32,
                        (layout.height - y - 1) as u32,
                        &mut rng,
                    );
                    unsafe {
                        let r = this.radiance.add(y * layout.pitch + x * 3);
                        let g = this.radiance.add(y * layout.pitch + x * 3 + 1);
                        let b = this.radiance.add(y * layout.pitch + x * 3 + 2);

                        let old = Color {
                            r: *r,
                            g: *g,
                            b: *b,
                        };

                        let pixel = (old * (iter - 1) as f32 + pixel) / iter as f32;

                        *r = pixel.r;
                        *g = pixel.g;
                        *b = pixel.b;
                    }
                }

                for x in 0..(layout.width * 3) {
                    unsafe {
                        let value = *this.radiance.add(y * layout.pitch + x);
                        *this.pixels.add(y * layout.pitch + x) = this.scene.to_u8(value);
                    }
                }

                this.signal();
            });
        }
    }

    pub fn stop(self: &Arc<Self>) {
        self.stopping
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn stopping(&self) -> bool {
        self.stopping
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}
