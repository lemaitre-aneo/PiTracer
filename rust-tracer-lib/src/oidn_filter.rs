//! Open Image Denoise (the `denoise` feature), as used by the worker's
//! denoising tasks and the standalone viewer.

use std::sync::{Mutex, PoisonError};

/// A lazily created OIDN CPU device. Devices are costly to create, and not
/// `Sync`: denoising calls are serialized, which costs nothing, since OIDN
/// already spreads one call over every core.
#[derive(Default)]
pub struct Denoiser {
    device: Mutex<Option<oidn::Device>>,
}

impl std::fmt::Debug for Denoiser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Denoiser").finish_non_exhaustive()
    }
}

impl Denoiser {
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
            let created = oidn::Device::cpu().map_err(|err| {
                DenoiseError(format!("OIDN could not create a CPU device: {err}"))
            })?;
            *device = Some(created);
        }
        let device = device.as_ref().expect("created above");
        // The crate checks the device's error after running the filter.
        oidn::RayTracing::try_new(device)
            .map_err(|err| DenoiseError(format!("OIDN could not create a filter: {err}")))?
            .hdr(true)
            .input_scale(1.0)
            .image_dimensions(width, height)
            .filter_in_place(color)
            .map_err(|err| DenoiseError(format!("OIDN could not filter: {err}")))
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
