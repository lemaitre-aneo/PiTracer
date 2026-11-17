use serde::{Deserialize, Serialize};

use crate::{Color, Hit, Texture};

use super::TextureProperty;

/// The Mandelbrot set mapped onto the surface coordinates: `(u, v)` in
/// [0, 1]^2 covers the rectangle of the complex plane of size `size`
/// centered on `center`, `u` along the real axis and `v` along the imaginary
/// one (quads map their corner `position` to `(0, 0)`, `u` along their `u`
/// edge and `v` along their `v` edge, like the image texture). An ellipse's
/// `(u, v)` spans [-1, 1]^2 instead, centered on 0: there, the ellipse's
/// center shows `center - size / 2` and each semi-axis spans `size`, so a
/// disc of radius `r` around `c` takes `center: c + r / 2`, `size: r`.
///
/// The point `c` belongs to the set when `z -> z^2 + c` (from `z = 0`)
/// stays bounded for `iterations` steps; it is then `inside_color`.
/// Otherwise its color comes from the smooth (continuous) escape count
///
///   mu = n + 1 - log2(log2 |z_n|)
///
/// (`n` the first step with `|z_n| > 256`), which has no bands between
/// iteration counts, and does not depend on the escape radius (it is about
/// the step at which `|z|` crosses 2): the palette is a closed loop of colors, linearly
/// interpolated, and one turn of it spans `period` iterations.
///
/// Computed in f64: deep zooms (small `size`) run out of f32 precision long
/// before the iteration count matters. The texture is a pure function of
/// `(u, v)`; the path tracer's sub-pixel jitter anti-aliases it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Mandelbrot {
    /// Complex number at the center of the surface, `[real, imaginary]`.
    pub center: [f64; 2],
    /// Extent of the complex plane covered by the surface, `[real,
    /// imaginary]`: match the surface's aspect ratio for an undistorted set.
    pub size: [f64; 2],
    /// Maximum number of steps before a point is deemed inside the set.
    pub iterations: u32,
    /// Color (albedo) of the points inside the set.
    pub inside_color: Color,
    /// Colors (albedo) of the escaping points, cycled through in order and
    /// back to the first. Empty: the escaping points are black.
    pub palette: Vec<Color>,
    /// Iterations per turn of the palette (> 0).
    pub period: f32,
}

impl Default for Mandelbrot {
    fn default() -> Self {
        Self {
            center: [-0.75, 0.0],
            size: [3.0, 3.0],
            iterations: 500,
            inside_color: Color::splat(0.02),
            palette: vec![
                Color::from((0.0, 0.02, 0.15)),
                Color::from((0.05, 0.3, 0.75)),
                Color::from((0.9, 0.9, 0.8)),
                Color::from((0.95, 0.55, 0.0)),
                Color::from((0.25, 0.04, 0.0)),
            ],
            period: 40.0,
        }
    }
}

/// Escape radius: large, so that `mu` is smooth to well below a palette step
/// (the error of the `log2(log2 |z|)` correction shrinks with the radius).
const BAILOUT: f64 = 256.0;

/// Whether `c = re + i * im` lies in the main cardioid or the period-2 bulb:
/// closed-form membership, which spares the full iteration count over most
/// of the set's area.
fn known_inside(re: f64, im: f64) -> bool {
    let q = (re - 0.25) * (re - 0.25) + im * im;
    q * (q + (re - 0.25)) <= 0.25 * im * im || (re + 1.0) * (re + 1.0) + im * im <= 0.0625
}

impl Mandelbrot {
    /// Smooth escape count of `c = re + i * im`, or `None` inside the set.
    fn escape(&self, re: f64, im: f64) -> Option<f64> {
        if known_inside(re, im) {
            return None;
        }
        let (mut x, mut y) = (0f64, 0f64);
        for n in 0..self.iterations {
            let (x2, y2) = (x * x, y * y);
            if x2 + y2 > BAILOUT * BAILOUT {
                // log2 |z| = log2(|z|^2) / 2
                let log_modulus = 0.5 * (x2 + y2).log2();
                return Some(f64::from(n) + 1.0 - log_modulus.log2());
            }
            y = 2.0 * x * y + im;
            x = x2 - y2 + re;
        }
        None
    }

    /// Palette color at smooth escape count `mu`.
    fn color(&self, mu: f64) -> Color {
        let len = self.palette.len();
        if len == 0 {
            return Color::default();
        }
        let t = (mu / f64::from(self.period)).rem_euclid(1.0) * len as f64;
        // rem_euclid can round up to exactly 1.0 for tiny negative inputs
        let i = (t.floor() as usize).min(len - 1);
        let f = (t - i as f64) as f32;
        self.palette[i] * (1.0 - f) + self.palette[(i + 1) % len] * f
    }
}

impl<R: crate::Rng + ?Sized> Texture<R> for Mandelbrot {
    fn get(&self, hit: Hit, _rng: &mut R) -> TextureProperty {
        let re = self.center[0] + (f64::from(hit.u) - 0.5) * self.size[0];
        let im = self.center[1] + (f64::from(hit.v) - 0.5) * self.size[1];
        TextureProperty {
            attenuation: match self.escape(re, im) {
                Some(mu) => self.color(mu),
                None => self.inside_color,
            },
            emission: Color::default(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::UnitVector;
    use crate::test_utils::{assert_close, hit_at, rng};

    fn at(texture: &Mandelbrot, u: f32, v: f32) -> Color {
        let hit = Hit {
            u,
            v,
            ..hit_at(Default::default(), UnitVector::Y, true)
        };
        texture.get(hit, &mut rng(1)).attenuation
    }

    /// Plain iteration, without the cardioid/bulb shortcut, as a reference.
    fn bounded(re: f64, im: f64, iterations: u32) -> bool {
        let (mut x, mut y) = (0f64, 0f64);
        for _ in 0..iterations {
            if x * x + y * y > 4.0 {
                return false;
            }
            (x, y) = (x * x - y * y + re, 2.0 * x * y + im);
        }
        true
    }

    /// The surface center maps to `center`, the corners to `center +/-
    /// size / 2`: 0 (inside the cardioid) is in the set, 1 + i is not.
    #[test]
    fn surface_maps_onto_the_complex_plane() {
        let texture = Mandelbrot {
            center: [0.0, 0.0],
            size: [2.0, 2.0],
            inside_color: Color::from((0.1, 0.2, 0.3)),
            ..Default::default()
        };
        assert_eq!(at(&texture, 0.5, 0.5), texture.inside_color);
        assert_ne!(at(&texture, 1.0, 1.0), texture.inside_color);
        // -1 + 0i (center of the period-2 bulb) at u = 0, v = 0.5
        assert_eq!(at(&texture, 0.0, 0.5), texture.inside_color);
    }

    /// The closed-form cardioid/bulb test only says "inside" for points the
    /// plain iteration keeps bounded, and does say it for most of them.
    #[test]
    fn shortcut_is_sound() {
        let n = 400;
        let (mut shortcut, mut total) = (0, 0);
        for i in 0..=n {
            for j in 0..=n {
                let re = -2.0 + 2.5 * f64::from(i) / f64::from(n);
                let im = -1.25 + 2.5 * f64::from(j) / f64::from(n);
                if known_inside(re, im) {
                    assert!(bounded(re, im, 10_000), "{re} + {im}i said inside");
                    shortcut += 1;
                }
                total += usize::from(bounded(re, im, 1000));
            }
        }
        // the cardioid and the bulb are ~90% of the set's area
        assert!(shortcut * 10 > total * 8, "{shortcut} of {total}");
    }

    /// Points escaping well before the iteration cap get a count, points
    /// inside the set get none.
    #[test]
    fn escape_matches_iteration() {
        let texture = Mandelbrot::default();
        for (re, im) in [(0.5, 0.5), (-2.1, 0.0), (0.3, 0.0), (-0.75, 0.2)] {
            assert!(!bounded(re, im, 100));
            assert!(texture.escape(re, im).is_some(), "{re} + {im}i");
        }
        // period-3 bulbs, outside the shortcut: real iterations
        for (re, im) in [(0.0, 0.0), (-1.0, 0.0), (-0.1226, 0.7449), (-1.7549, 0.0)] {
            assert!(bounded(re, im, 10_000));
            assert!(texture.escape(re, im).is_none(), "{re} + {im}i");
        }
    }

    /// The smooth count is continuous: two nearby escaping points get
    /// nearby counts, even across an integer escape step.
    #[test]
    fn escape_count_is_smooth() {
        let texture = Mandelbrot::default();
        let mut previous = texture.escape(-2.5, 0.5).unwrap();
        for k in 1..=1000 {
            let re = -2.5 + 0.5 * f64::from(k) / 1000.0;
            let mu = texture.escape(re, 0.5).unwrap();
            assert!(
                (mu - previous).abs() < 0.05,
                "jump {previous} -> {mu} at {re}"
            );
            previous = mu;
        }
    }

    /// The palette is a closed loop: one period later the color is the same,
    /// and between two entries it interpolates linearly.
    #[test]
    fn palette_cycles_and_interpolates() {
        let texture = Mandelbrot {
            palette: vec![Color::splat(0.0), Color::splat(1.0)],
            period: 10.0,
            ..Default::default()
        };
        assert_close(texture.color(0.0).r, 0.0, 1e-6);
        assert_close(texture.color(2.5).r, 0.5, 1e-6);
        assert_close(texture.color(5.0).r, 1.0, 1e-6);
        assert_close(texture.color(7.5).r, 0.5, 1e-6);
        assert_close(texture.color(13.0).r, texture.color(3.0).r, 1e-6);
        assert_eq!(
            Mandelbrot {
                palette: vec![],
                ..Default::default()
            }
            .color(3.0),
            Color::default()
        );
    }
}
