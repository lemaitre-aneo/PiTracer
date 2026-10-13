//! The denoising thread (the `denoise` feature): denoises the frames the
//! window asks for, off the UI and render threads, and posts the results
//! through the SDL event queue. Without the feature, a stand-in that never
//! denoises, so the window code is the same either way.

pub use imp::DenoiseThread;

/// Whether noise reduction is built in.
pub const AVAILABLE: bool = cfg!(feature = "denoise");

/// The device that denoises (`--denoise-device`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Device {
    /// A supported GPU if any, else the CPU
    #[default]
    Auto,
    Cpu,
    /// NVIDIA GPU
    Cuda,
    /// AMD GPU
    Hip,
    /// Intel GPU
    Sycl,
    /// Apple GPU
    Metal,
}

#[cfg(feature = "denoise")]
impl From<Device> for raytracer::DenoiseDevice {
    fn from(device: Device) -> Self {
        match device {
            Device::Auto => Self::Auto,
            Device::Cpu => Self::Cpu,
            Device::Cuda => Self::Cuda,
            Device::Hip => Self::Hip,
            Device::Sycl => Self::Sycl,
            Device::Metal => Self::Metal,
        }
    }
}

#[cfg(feature = "denoise")]
mod imp {
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread::JoinHandle;

    use raytracer::Denoiser;
    use sdl2::event::EventSender;

    use super::Device;
    use crate::display::{DenoiseRequest, Denoised};
    use crate::raytracer::ViewerEvent;

    #[derive(Default)]
    struct Slot {
        /// Only the latest request matters: a newer frame supersedes an older
        /// one still waiting.
        request: Option<DenoiseRequest>,
        stopping: bool,
    }

    pub struct DenoiseThread {
        slot: Arc<(Mutex<Slot>, Condvar)>,
        handle: JoinHandle<()>,
    }

    impl DenoiseThread {
        pub fn spawn(width: usize, height: usize, device: Device, sender: EventSender) -> Self {
            let slot = Arc::new((Mutex::new(Slot::default()), Condvar::new()));
            let handle = std::thread::spawn({
                let slot = slot.clone();
                move || run(&slot, width, height, device, sender)
            });
            Self { slot, handle }
        }

        pub fn request(&self, request: DenoiseRequest) {
            let (slot, wake) = &*self.slot;
            slot.lock().unwrap().request = Some(request);
            wake.notify_one();
        }

        /// Stops the thread, after the frame being denoised if any (an OIDN
        /// call cannot be interrupted).
        pub fn stop(self) {
            let (slot, wake) = &*self.slot;
            slot.lock().unwrap().stopping = true;
            wake.notify_one();
            if self.handle.join().is_err() {
                tracing::error!("the denoising thread panicked");
            }
        }
    }

    fn run(
        slot: &(Mutex<Slot>, Condvar),
        width: usize,
        height: usize,
        device: Device,
        sender: EventSender,
    ) {
        let (slot, wake) = slot;
        let denoiser = Denoiser::new(device.into());
        let mut first = true;
        loop {
            let request = {
                let mut slot = slot.lock().unwrap();
                loop {
                    if slot.stopping {
                        return;
                    }
                    if let Some(request) = slot.request.take() {
                        break request;
                    }
                    slot = wake.wait(slot).unwrap();
                }
            };
            let started = std::time::Instant::now();
            let mut denoised = (*request.snapshot).clone();
            if let Err(err) = denoiser.denoise(&mut denoised, width, height) {
                tracing::warn!("could not denoise frame #{}: {err}", request.iter);
                continue;
            }
            if std::mem::take(&mut first) {
                let device = denoiser
                    .device()
                    .map_or("unknown".into(), |d| d.to_string());
                tracing::info!("denoising on the {device} device");
            }
            tracing::debug!(
                "frame #{}: denoised in {:#?}",
                request.iter,
                started.elapsed()
            );
            let event = ViewerEvent::Denoised(Denoised {
                iter: request.iter,
                denoised,
                snapshot: request.snapshot,
            });
            if sender.push_custom_event(event).is_err() {
                tracing::debug!("dropped denoised frame #{}", request.iter);
            }
        }
    }
}

#[cfg(not(feature = "denoise"))]
mod imp {
    use sdl2::event::EventSender;

    use super::Device;
    use crate::display::DenoiseRequest;

    pub struct DenoiseThread;

    impl DenoiseThread {
        pub fn spawn(_width: usize, _height: usize, _device: Device, _sender: EventSender) -> Self {
            Self
        }

        pub fn request(&self, _request: DenoiseRequest) {}

        pub fn stop(self) {}
    }
}
