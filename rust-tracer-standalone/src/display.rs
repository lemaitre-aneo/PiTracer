//! What the window shows: the raw radiance, or its noise-reduced version.
//!
//! Noise reduction never feeds back into the rendering: each completed frame
//! (a full pass over the image) is snapshotted, and a denoised frame `D` is
//! computed from scratch from such a snapshot `S`. Rows keep streaming in as
//! they are rendered: with noise reduction on, a row whose current raw
//! radiance is `A` is shown as `D + (A - S)`, the denoised frame plus what
//! the passes since its snapshot have contributed (a running mean moves by
//! a fraction of a pass's noise, so little noise comes back), until the
//! next denoised frame replaces it.

use std::sync::Arc;

use raytracer::Scene;

/// A frame to denoise: the raw radiance of frame `iter`, RGB f32 triplets,
/// row-major.
// Only read by the denoising thread.
#[cfg_attr(not(feature = "denoise"), allow(dead_code))]
pub struct DenoiseRequest {
    pub iter: u32,
    pub snapshot: Arc<Vec<f32>>,
}

/// The denoised version of frame `iter`, and the snapshot it was computed
/// from.
pub struct Denoised {
    pub iter: u32,
    pub denoised: Vec<f32>,
    pub snapshot: Arc<Vec<f32>>,
}

pub struct Display {
    scene: Arc<Scene>,
    /// Floats per row.
    stride: usize,
    /// Latest radiance of every row, as streamed.
    raw: Vec<f32>,
    /// What the window shows, RGB24.
    rgb: Vec<u8>,
    /// Whether noise reduction can be used at all (built in).
    available: bool,
    /// Noise reduction is on.
    enabled: bool,
    /// The denoised frame is shown: noise reduction is on and the frame is
    /// recent enough (see [`Display::toggle`]).
    active: bool,
    denoised: Option<Denoised>,
    /// Snapshot of the last completed frame.
    last_frame: Option<(u32, Arc<Vec<f32>>)>,
}

impl Display {
    /// Noise reduction starts on when `available`.
    pub fn new(scene: Arc<Scene>, available: bool) -> Self {
        let (width, height) = (scene.camera.width as usize, scene.camera.height as usize);
        Self {
            scene,
            stride: width * 3,
            raw: vec![0f32; width * height * 3],
            rgb: vec![0u8; width * height * 3],
            available,
            enabled: available,
            active: false,
            denoised: None,
            last_frame: None,
        }
    }

    pub fn rgb(&mut self) -> &mut [u8] {
        &mut self.rgb
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Row `y` was rendered: shows it.
    pub fn row(&mut self, y: usize, radiance: &[f32]) {
        self.raw[y * self.stride..(y + 1) * self.stride].copy_from_slice(radiance);
        self.compose_row(y);
    }

    /// A frame was completed: the frame to denoise, if noise reduction is
    /// on.
    pub fn frame_done(&mut self, iter: u32, snapshot: Arc<Vec<f32>>) -> Option<DenoiseRequest> {
        self.last_frame = Some((iter, snapshot.clone()));
        self.enabled.then_some(DenoiseRequest { iter, snapshot })
    }

    /// A denoised frame is ready: whether the whole image changed.
    pub fn denoised(&mut self, denoised: Denoised) -> bool {
        if self
            .denoised
            .as_ref()
            .is_some_and(|current| current.iter > denoised.iter)
        {
            return false;
        }
        self.denoised = Some(denoised);
        if self.enabled {
            self.active = true;
            self.compose_all();
        }
        self.enabled
    }

    /// Toggles noise reduction, and returns the frame to denoise, if any.
    /// Turning it off shows the raw image at once. Turning it on shows the
    /// denoised frame at once if it is the last completed frame's;
    /// otherwise that frame is denoised, and the raw image shown until it
    /// is ready.
    pub fn toggle(&mut self) -> Option<DenoiseRequest> {
        if !self.available {
            return None;
        }
        self.enabled = !self.enabled;
        let current = match (&self.denoised, &self.last_frame) {
            (Some(denoised), Some((iter, _))) => denoised.iter == *iter,
            _ => false,
        };
        self.active = self.enabled && current;
        self.compose_all();
        if self.enabled && !current {
            self.last_frame
                .as_ref()
                .map(|(iter, snapshot)| DenoiseRequest {
                    iter: *iter,
                    snapshot: snapshot.clone(),
                })
        } else {
            None
        }
    }

    fn compose_all(&mut self) {
        for y in 0..self.raw.len() / self.stride {
            self.compose_row(y);
        }
    }

    fn compose_row(&mut self, y: usize) {
        let span = y * self.stride..(y + 1) * self.stride;
        let (raw, rgb) = (&self.raw[span.clone()], &mut self.rgb[span.clone()]);
        match self.denoised.as_ref().filter(|_| self.active) {
            Some(denoised) => {
                let (d, s) = (&denoised.denoised[span.clone()], &denoised.snapshot[span]);
                for (i, byte) in rgb.iter_mut().enumerate() {
                    *byte = self.scene.to_u8(d[i] + (raw[i] - s[i]));
                }
            }
            None => {
                for (byte, &value) in rgb.iter_mut().zip(raw) {
                    *byte = self.scene.to_u8(value);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2x2 image, gamma 1 so that bytes are 255 * radiance.
    fn display(available: bool) -> Display {
        let mut definition = raytracer::decode_scene(b"{}").unwrap();
        (definition.camera.width, definition.camera.height) = (2, 2);
        definition.gamma = 1.0;
        Display::new(Arc::new(Scene::from(definition)), available)
    }

    fn frame(value: f32) -> Arc<Vec<f32>> {
        Arc::new(vec![value; 12])
    }

    fn denoised(iter: u32, value: f32, snapshot: f32) -> Denoised {
        Denoised {
            iter,
            denoised: vec![value; 12],
            snapshot: frame(snapshot),
        }
    }

    fn shown(display: &mut Display) -> Vec<u8> {
        display.rgb().to_vec()
    }

    /// Byte of radiance `value` (gamma 1).
    fn byte(value: f32) -> u8 {
        (value * 255.0 + 0.5) as u8
    }

    #[test]
    fn noise_reduction_starts_on_and_follows_the_frames() {
        let mut display = display(true);
        assert!(display.enabled());
        // No denoised frame yet: raw.
        display.row(0, &[0.2; 6]);
        assert_eq!(&shown(&mut display)[..6], [byte(0.2); 6]);

        let request = display.frame_done(1, frame(0.2)).unwrap();
        assert_eq!(request.iter, 1);
        assert!(display.denoised(denoised(1, 0.5, 0.2)));
        // Row 0: D + (A - S) = 0.5; row 1 never rendered: 0.5 + (0 - 0.2).
        assert_eq!(
            shown(&mut display),
            [[byte(0.5); 6], [byte(0.3); 6]].concat()
        );

        // The next frame's rows add their contribution on top.
        display.row(1, &[0.4; 6]);
        assert_eq!(&shown(&mut display)[6..], [byte(0.5 + 0.4 - 0.2); 6]);
    }

    #[test]
    fn turning_it_off_shows_the_raw_image_at_once() {
        let mut display = display(true);
        display.row(0, &[0.2; 6]);
        display.frame_done(1, frame(0.2));
        display.denoised(denoised(1, 0.5, 0.2));
        assert!(display.toggle().is_none());
        assert!(!display.enabled());
        assert_eq!(&shown(&mut display)[..6], [byte(0.2); 6]);
        // Off: no denoising, rows raw.
        assert!(display.frame_done(2, frame(0.3)).is_none());
        display.row(0, &[0.3; 6]);
        assert_eq!(&shown(&mut display)[..6], [byte(0.3); 6]);
    }

    #[test]
    fn turning_it_on_reuses_a_current_denoised_frame() {
        let mut display = display(true);
        display.row(0, &[0.2; 6]);
        display.frame_done(1, frame(0.2));
        display.denoised(denoised(1, 0.5, 0.2));
        display.toggle();
        // Still frame 1: back on at once.
        assert!(display.toggle().is_none());
        assert_eq!(&shown(&mut display)[..6], [byte(0.5); 6]);
    }

    #[test]
    fn turning_it_on_after_newer_frames_denoises_the_last_one() {
        let mut display = display(true);
        display.row(0, &[0.2; 6]);
        display.frame_done(1, frame(0.2));
        display.denoised(denoised(1, 0.5, 0.2));
        display.toggle();
        display.frame_done(2, frame(0.25));
        display.row(0, &[0.3; 6]);

        let request = display.toggle().unwrap();
        assert_eq!(request.iter, 2);
        // Raw until the denoised frame 2 is ready: the stale one is not used.
        assert_eq!(&shown(&mut display)[..6], [byte(0.3); 6]);
        assert!(display.denoised(denoised(2, 0.6, 0.25)));
        assert_eq!(&shown(&mut display)[..6], [byte(0.6 + 0.3 - 0.25); 6]);
    }

    #[test]
    fn late_denoised_frames_do_not_replace_newer_ones() {
        let mut display = display(true);
        display.row(0, &[0.2; 6]);
        display.denoised(denoised(3, 0.5, 0.2));
        assert!(!display.denoised(denoised(2, 0.9, 0.2)));
        assert_eq!(&shown(&mut display)[..6], [byte(0.5); 6]);
    }

    #[test]
    fn without_noise_reduction_the_image_stays_raw() {
        let mut display = display(false);
        assert!(!display.enabled());
        assert!(display.toggle().is_none());
        assert!(!display.enabled());
        assert!(display.frame_done(1, frame(0.2)).is_none());
        display.row(0, &[0.2; 6]);
        assert_eq!(&shown(&mut display)[..6], [byte(0.2); 6]);
    }
}
