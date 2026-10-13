//! Open Image Denoise (the `denoise` feature), as used by the worker's
//! denoising tasks and the standalone viewer.

use std::sync::{Mutex, PoisonError};

/// The kind of device Open Image Denoise runs on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DenoiseDevice {
    /// The CPU, available everywhere.
    #[default]
    Cpu,
    /// The fastest device OIDN finds: a supported GPU if any, else the CPU.
    Auto,
    /// An NVIDIA GPU.
    Cuda,
    /// An AMD GPU.
    Hip,
    /// An Intel GPU.
    Sycl,
    /// An Apple GPU.
    Metal,
}

impl DenoiseDevice {
    fn create(self) -> Result<oidn::Device, oidn::Error> {
        match self {
            Self::Cpu => oidn::Device::cpu(),
            Self::Auto => oidn::Device::new(),
            Self::Cuda => oidn::Device::cuda(),
            Self::Hip => oidn::Device::hip(),
            Self::Sycl => oidn::Device::sycl(),
            Self::Metal => oidn::Device::metal(),
        }
    }

    /// The kind of `device`, which `Auto` only knows once created.
    fn of(device: &oidn::Device) -> Option<Self> {
        use oidn::sys::*;
        let kind = unsafe { oidnGetDeviceInt(device.raw(), c"type".as_ptr()) };
        [
            (OIDNDeviceType_OIDN_DEVICE_TYPE_CPU, Self::Cpu),
            (OIDNDeviceType_OIDN_DEVICE_TYPE_CUDA, Self::Cuda),
            (OIDNDeviceType_OIDN_DEVICE_TYPE_HIP, Self::Hip),
            (OIDNDeviceType_OIDN_DEVICE_TYPE_SYCL, Self::Sycl),
            (OIDNDeviceType_OIDN_DEVICE_TYPE_METAL, Self::Metal),
        ]
        .into_iter()
        .find_map(|(raw, this)| (raw == kind).then_some(this))
    }
}

impl std::fmt::Display for DenoiseDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Cpu => "CPU",
            Self::Auto => "automatic",
            Self::Cuda => "CUDA",
            Self::Hip => "HIP",
            Self::Sycl => "SYCL",
            Self::Metal => "Metal",
        })
    }
}

/// A lazily created OIDN device, of the kind given at construction (the CPU
/// by default). Devices are costly to create, and not `Sync`: denoising
/// calls are serialized, which costs nothing, since OIDN already spreads one
/// call over every core, or over the whole GPU.
#[derive(Default)]
pub struct Denoiser {
    kind: DenoiseDevice,
    device: Mutex<Option<oidn::Device>>,
}

impl std::fmt::Debug for Denoiser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Denoiser")
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

impl Denoiser {
    /// A denoiser that runs on a device of the given kind, created on the
    /// first call.
    pub fn new(kind: DenoiseDevice) -> Self {
        Self {
            kind,
            device: Mutex::default(),
        }
    }

    /// The kind of the device, once created by a call to [`Self::denoise`]:
    /// for [`DenoiseDevice::Auto`], the one OIDN picked.
    pub fn device(&self) -> Option<DenoiseDevice> {
        let device = self.device.lock().unwrap_or_else(PoisonError::into_inner);
        device.as_ref().and_then(DenoiseDevice::of)
    }

    /// Denoises, in place, the `width` x `height` linear RGB image `color`
    /// (f32 triplets, row-major). The input scale is fixed, as OIDN requires
    /// for HDR images denoised in tiles (an automatic one would differ from
    /// tile to tile and show seams): 1.0, which OIDN maps to 100 cd/m², about
    /// display white, as in our radiance.
    pub fn denoise(
        &self,
        color: &mut [f32],
        width: usize,
        height: usize,
    ) -> Result<(), DenoiseError> {
        let mut device = self.device.lock().unwrap_or_else(PoisonError::into_inner);
        // A failed creation is not kept: the next call tries again.
        if device.is_none() {
            let created = self.kind.create().map_err(|err| {
                DenoiseError(format!(
                    "OIDN could not create a {} device: {err}",
                    self.kind
                ))
            })?;
            *device = Some(created);
        }
        let device = device.as_ref().expect("created above");
        // Device storage, copied to and from explicitly: the default storage
        // of GPU devices can be managed memory, which some platforms (WSL)
        // emulate, more than 20 times slower. On the CPU, it is host memory.
        let image = device
            .create_buffer_with_storage(color.len(), oidn::Storage::Device)
            .map_err(|err| DenoiseError(format!("OIDN could not allocate the image: {err}")))?;
        image
            .write(color)
            .map_err(|err| DenoiseError(format!("OIDN could not upload the image: {err}")))?;
        // The crate checks the device's error after running the filter.
        oidn::RayTracing::try_new(device)
            .map_err(|err| DenoiseError(format!("OIDN could not create a filter: {err}")))?
            .hdr(true)
            .input_scale(1.0)
            .image_dimensions(width, height)
            .filter_in_place_buffer(&image)
            .map_err(|err| DenoiseError(format!("OIDN could not filter: {err}")))?;
        image
            .read_to_slice(color)
            .map_err(|err| DenoiseError(format!("OIDN could not download the image: {err}")))
    }
}

/// Why Open Image Denoise could not denoise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenoiseError(pub String);

impl std::fmt::Display for DenoiseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DenoiseError {}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use rand::{RngExt, SeedableRng};

    use super::*;

    #[test]
    fn denoising_reduces_noise() {
        let (width, height) = (64, 48);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
        let exact = 0.4f32;
        let mut color: Vec<f32> = (0..width * height * 3)
            .map(|_| exact + rng.random_range(-0.2..0.2))
            .collect();
        let error = |color: &[f32]| {
            color
                .iter()
                .map(|&c| ((c - exact) as f64).powi(2))
                .sum::<f64>()
                / color.len() as f64
        };
        let before = error(&color);
        Denoiser::default()
            .denoise(&mut color, width, height)
            .unwrap();
        let after = error(&color);
        assert!(after < before / 10.0, "MSE {before} -> {after}");
    }

    /// `Auto` works everywhere (falling back to the CPU without a supported
    /// GPU), and reports the device OIDN picked once created.
    #[test]
    fn automatic_device() {
        let denoiser = Denoiser::new(DenoiseDevice::Auto);
        assert_eq!(denoiser.device(), None);
        let mut color = vec![0.5f32; 16 * 16 * 3];
        denoiser.denoise(&mut color, 16, 16).unwrap();
        assert!(color.iter().all(|&c| (c - 0.5).abs() < 0.05), "{color:?}");
        let device = denoiser.device().expect("a known device");
        assert_ne!(device, DenoiseDevice::Auto);
    }

    #[test]
    fn wrong_sizes_are_errors() {
        let mut color = vec![0f32; 10];
        assert!(Denoiser::default().denoise(&mut color, 4, 4).is_err());
    }

    /// The tiling constants the client plans denoising tasks with must
    /// cover what the linked OIDN requires, for the filter as `denoise`
    /// configures it (the constants are only set once the filter is
    /// committed with its images, which select the model).
    #[test]
    fn halo_covers_the_tile_overlap() {
        use oidn::sys::*;
        let int = |filter, name: &CStr| unsafe { oidnGetFilterInt(filter, name.as_ptr()) };
        let (width, height) = (64, 64);
        let (overlap, alignment) = unsafe {
            let device = oidnNewDevice(OIDNDeviceType_OIDN_DEVICE_TYPE_CPU);
            oidnCommitDevice(device);
            let bytes = width * height * 3 * size_of::<f32>();
            let (color, output) = (oidnNewBuffer(device, bytes), oidnNewBuffer(device, bytes));
            let filter = oidnNewFilter(device, c"RT".as_ptr());
            for (name, buffer) in [(c"color", color), (c"output", output)] {
                let format = OIDNFormat_OIDN_FORMAT_FLOAT3;
                oidnSetFilterImage(
                    filter,
                    name.as_ptr(),
                    buffer,
                    format,
                    width,
                    height,
                    0,
                    0,
                    0,
                );
            }
            oidnSetFilterBool(filter, c"hdr".as_ptr(), true);
            oidnSetFilterFloat(filter, c"inputScale".as_ptr(), 1.0);
            oidnCommitFilter(filter);
            let values = (int(filter, c"tileOverlap"), int(filter, c"tileAlignment"));
            oidnReleaseFilter(filter);
            oidnReleaseBuffer(color);
            oidnReleaseBuffer(output);
            oidnReleaseDevice(device);
            values
        };
        assert!(overlap > 0 && alignment > 0, "{overlap} {alignment}");
        assert!(
            crate::DENOISE_HALO >= overlap as u32,
            "OIDN needs {overlap} px"
        );
        assert_eq!(
            crate::DENOISE_ALIGNMENT % alignment as u32,
            0,
            "OIDN needs {alignment} px"
        );
        assert_eq!(crate::DENOISE_HALO % crate::DENOISE_ALIGNMENT, 0);
    }
}
