use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32},
    },
    time::Duration,
};

use rand::{RngCore, SeedableRng};
use raytracer::{Color, Scene};
use sdl2::{Sdl, event::EventSender, surface::Surface};

pub struct Raytracer {
    pub scene: Scene,
    pending: Mutex<bool>,
    event_sender: EventSender,
    radiance: Vec<AtomicU32>,
    pixels: Vec<AtomicU32>,
    stopping: AtomicBool,
    iter: AtomicU32,
}

impl Raytracer {
    pub fn new(scene: Scene, sdl: &Sdl) -> Result<Arc<Self>, String> {
        let events = sdl.event()?;
        events.register_custom_event::<Arc<Raytracer>>()?;
        let total_size = scene.camera.width as usize * scene.camera.height as usize;
        let mut radiance = Vec::with_capacity(total_size * 3);
        radiance.resize_with(total_size * 3, Default::default);
        let mut pixels = Vec::with_capacity(total_size);
        pixels.resize_with(total_size, Default::default);

        Ok(Arc::new(Self {
            scene,
            pending: Mutex::new(false),
            event_sender: events.event_sender(),
            radiance,
            pixels,
            stopping: AtomicBool::new(false),
            iter: AtomicU32::new(0),
        }))
    }

    pub fn width(&self) -> u32 {
        self.scene.camera.width
    }
    pub fn height(&self) -> u32 {
        self.scene.camera.height
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
        let data = unsafe {
            std::slice::from_raw_parts_mut(
                self.pixels.as_slice().as_ptr() as *mut u8,
                self.width() as usize * self.height() as usize * 4,
            )
        };
        Surface::from_data(
            data,
            self.width(),
            self.height(),
            self.width() * 4,
            sdl2::pixels::PixelFormatEnum::ARGB8888,
        )
    }

    pub fn send_process(self: Arc<Self>, mut rng: &mut dyn RngCore) {
        let iter = self.iter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

        for y in 0..self.height() {
            let mut rng = rand::rngs::SmallRng::from_rng(&mut rng);
            let this = self.clone();

            rayon::spawn(move || {
                for x in 0..this.width() {
                    if this.stopping.load(std::sync::atomic::Ordering::Acquire) {
                        return;
                    }
                    let pixel = this
                        .scene
                        .pixel_radiance(x, this.height() - y - 1, &mut rng);
                    let r = &this.radiance[(y as usize * this.width() as usize + x as usize) * 3];
                    let g =
                        &this.radiance[(y as usize * this.width() as usize + x as usize) * 3 + 1];
                    let b =
                        &this.radiance[(y as usize * this.width() as usize + x as usize) * 3 + 2];

                    let old = Color {
                        r: f32::from_bits(r.load(std::sync::atomic::Ordering::Relaxed)),
                        g: f32::from_bits(g.load(std::sync::atomic::Ordering::Relaxed)),
                        b: f32::from_bits(b.load(std::sync::atomic::Ordering::Relaxed)),
                    };

                    let pixel = (old * (iter - 1) as f32 + pixel) / iter as f32;

                    r.store(pixel.r.to_bits(), std::sync::atomic::Ordering::Relaxed);
                    g.store(pixel.g.to_bits(), std::sync::atomic::Ordering::Relaxed);
                    b.store(pixel.b.to_bits(), std::sync::atomic::Ordering::Relaxed);

                    let color = this.scene.to_u8(pixel);
                    let color = (255 << 24)
                        | ((color[0] as u32) << 16)
                        | ((color[1] as u32) << 8)
                        | (color[2] as u32);

                    this.pixels[y as usize * this.width() as usize + x as usize]
                        .store(color, std::sync::atomic::Ordering::Relaxed);
                }

                this.signal();
            });
        }
    }

    pub fn stop(self: &Arc<Self>) {
        self.stopping
            .store(true, std::sync::atomic::Ordering::Relaxed);

        // Wait for all processing tasks to finish their rendering
        while Arc::strong_count(self) > 1 {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
