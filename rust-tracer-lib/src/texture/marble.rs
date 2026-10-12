use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use serde::{Deserialize, Serialize};

use crate::{Color, Hit, Texture, Vector};

use super::TextureProperty;

/// Classic Perlin marble: sine stripes along `direction`, phase-perturbed by
/// 3D OpenSimplex2 fbm noise evaluated at the world-space hit point (a solid
/// texture: the pattern carves through the object, it is not surface-mapped).
///
///   t = (1 + sin(scale * dot(p, dir) + turbulence * fbm(p))) / 2
///   attenuation = lerp(color, vein_color, t)
///
/// with dir = normalize(direction). `sin` folds any noise amplitude into
/// t in [0, 1], so the attenuation always stays inside the hull of the two
/// colors no matter how the noise is tuned.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Marble {
    pub color: Color,
    pub vein_color: Color,
    pub emission: Color,
    /// Stripe axis; only its direction matters (normalized before use).
    pub direction: Vector,
    /// Stripe angular frequency in rad/world-unit (wavelength =
    /// 2*pi/scale). The noise feature size follows the stripe wavelength.
    pub scale: f32,
    /// Phase distortion strength in radians. The fbm term peaks around
    /// +/-0.8 (see `noise_range_probe`), so the default 4 lets veins wander
    /// by up to ~half a stripe (~pi radians of phase); 0 gives perfectly
    /// straight stripes, and large values scramble them into blobs.
    pub turbulence: f32,
    pub octaves: usize,
    pub seed: i64,
}

impl Default for Marble {
    fn default() -> Self {
        Self {
            color: Color::splat(0.75),
            vein_color: Color::splat(0.2),
            emission: Color::default(),
            direction: Vector::from((1f32, 0f32, 0f32)),
            scale: 1f32,
            turbulence: 4f32,
            octaves: 4,
            seed: 0,
        }
    }
}

impl Marble {
    /// Fbm OpenSimplex2 noise at `p` (frequency tied to the stripe wavelength).
    ///
    /// fastnoise-lite is f32-native and normalizes its fbm by the fractal
    /// bounding factor, so the output is zero-mean and documented to stay
    /// within [-1, 1] (empirically +/-0.9; see `noise_range_probe`). The
    /// `FastNoiseLite` value holds only scalar config, so constructing one per
    /// call is cheap. Its default lacunarity 2.0 / gain 0.5 match our octave
    /// schedule and are left implicit.
    fn fbm(&self, p: Vector) -> f32 {
        let mut noise = FastNoiseLite::with_seed(self.seed as i32);
        noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        noise.set_fractal_type(Some(FractalType::FBm));
        noise.set_fractal_octaves(Some(self.octaves as i32));
        noise.set_frequency(Some(self.scale / std::f32::consts::TAU));
        noise.get_noise_3d(p.x, p.y, p.z)
    }
}

impl<R: crate::Rng + ?Sized> Texture<R> for Marble {
    fn get(&self, hit: Hit, _rng: &mut R) -> TextureProperty {
        let norm = self.direction.norm();
        // Degenerate (zero/non-finite) direction: drop the stripe term
        // instead of poisoning the color with NaN.
        let stripes = if norm > 0f32 && norm.is_finite() {
            self.scale * hit.point.dot(self.direction) / norm
        } else {
            0f32
        };
        let t = 0.5f32 * (1f32 + (stripes + self.turbulence * self.fbm(hit.point)).sin());
        TextureProperty {
            attenuation: self.color * (1f32 - t) + self.vein_color * t,
            emission: self.emission,
        }
    }
}

#[cfg(test)]
mod test {
    use rand::RngExt as _;

    use super::*;
    use crate::UnitVector;
    use crate::test_utils::{assert_close, hit_at, rng, seeds};

    fn colors() -> (Color, Color) {
        (Color::from((0.9, 0.8, 0.7)), Color::from((0.1, 0.2, 0.3)))
    }

    fn random_point(rng: &mut rand::rngs::SmallRng) -> Vector {
        Vector::from((
            rng.random::<f32>() * 200f32 - 100f32,
            rng.random::<f32>() * 200f32 - 100f32,
            rng.random::<f32>() * 200f32 - 100f32,
        ))
    }

    /// With turbulence 0 the texture is a pure sine stripe pattern with the
    /// closed form t = (1 + sin(scale * dot(p, dir)))/2, checked in f64.
    #[test]
    fn turbulence_zero_gives_exact_stripes() {
        let (color, vein_color) = colors();
        let cases = [
            // (direction, scale, point)
            ((1f32, 0f32, 0f32), 2f32, (0.7f32, 1f32, -2f32)),
            ((1f32, 0f32, 0f32), 2f32, (-3.2f32, 0f32, 5f32)),
            ((1f32, 0f32, 0f32), 0.5f32, (2.9f32, -1f32, 0f32)),
            // Non-axis, non-unit direction: must be normalized internally.
            ((0f32, 3f32, 4f32), 1.5f32, (1f32, 2f32, -0.5f32)),
        ];
        for (direction, scale, point) in cases {
            let marble = Marble {
                color,
                vein_color,
                direction: direction.into(),
                scale,
                turbulence: 0f32,
                ..Default::default()
            };
            let got = marble.get(hit_at(point.into(), UnitVector::X, true), &mut rng(1));
            let d = Vector::from(direction);
            let t =
                0.5f64 * (1f64 + f64::from(scale * Vector::from(point).dot(d) / d.norm()).sin());
            assert_close(got.attenuation.r, (1f64 - t) * 0.9 + t * 0.1, 1e-6);
            assert_close(got.attenuation.g, (1f64 - t) * 0.8 + t * 0.2, 1e-6);
            assert_close(got.attenuation.b, (1f64 - t) * 0.7 + t * 0.3, 1e-6);
        }
    }

    /// Pins the fbm range the `turbulence` doc relies on. fastnoise-lite
    /// documents its fbm output as bounded within [-1, 1] and normalizes by
    /// the fractal-bounding factor; measured here at +/-0.99 (1 octave) and
    /// +/-0.83 (4 octaves). Asserting the documented bound trips if a crate
    /// update changes its normalization.
    #[test]
    fn noise_range_probe() {
        for octaves in [1usize, 4] {
            let marble = Marble {
                octaves,
                ..Default::default()
            };
            let mut r = rng(1);
            let (mut min, mut max) = (f32::MAX, f32::MIN);
            let mut mean = 0f64;
            let n = 10_000;
            for _ in 0..n {
                let v = marble.fbm(random_point(&mut r));
                assert!(v.is_finite(), "fbm returned {v}");
                min = min.min(v);
                max = max.max(v);
                mean += f64::from(v) / f64::from(n);
            }
            // documented bound is +/-1; the small slack absorbs f32 rounding
            assert!(
                (-1.01..0f32).contains(&min) && (0f32..1.01).contains(&max),
                "fbm({octaves} octaves) range [{min}, {max}] escapes ±1 or is one-sided",
            );
            assert!(max - min > 0.5, "fbm nearly constant: [{min}, {max}]");
            assert!(mean.abs() < 0.02, "fbm not zero-mean: {mean}");
        }
    }

    /// t in [0, 1] means every channel stays inside the hull of the two
    /// colors, whatever the noise does.
    #[test]
    fn attenuation_stays_in_color_hull() {
        let (color, vein_color) = colors();
        for seed in seeds() {
            let marble = Marble {
                color,
                vein_color,
                turbulence: 20f32,
                seed: seed as i64,
                ..Default::default()
            };
            let mut r = rng(seed);
            for _ in 0..1_000 {
                let got = marble.get(hit_at(random_point(&mut r), UnitVector::X, true), &mut r);
                for (channel, lo, hi) in [
                    (got.attenuation.r, 0.1f32, 0.9f32),
                    (got.attenuation.g, 0.2f32, 0.8f32),
                    (got.attenuation.b, 0.3f32, 0.7f32),
                ] {
                    assert!(
                        (lo - 1e-6..=hi + 1e-6).contains(&channel),
                        "channel {channel} outside [{lo}, {hi}]",
                    );
                }
            }
        }
    }

    /// The texture is a pure function of the hit point: the RNG must not
    /// influence it (repeated hits at one point must agree exactly).
    #[test]
    fn texture_is_deterministic() {
        let marble = Marble::default();
        let mut r = rng(1);
        for _ in 0..100 {
            let hit = hit_at(random_point(&mut r), UnitVector::X, true);
            let first = marble.get(hit, &mut r);
            assert_eq!(first, marble.get(hit, &mut r));
            assert_eq!(first, marble.get(hit, &mut rng(2)));
        }
    }

    #[test]
    fn emission_is_constant_and_seeds_differ() {
        let emission = Color::from((1f32, 2f32, 3f32));
        let a = Marble {
            emission,
            ..Default::default()
        };
        let b = Marble { seed: 1, ..a };
        let mut r = rng(1);
        let mut differ = false;
        for _ in 0..20 {
            let hit = hit_at(random_point(&mut r), UnitVector::X, true);
            assert_eq!(a.get(hit, &mut r).emission, emission);
            differ |= a.get(hit, &mut r).attenuation != b.get(hit, &mut r).attenuation;
        }
        assert!(differ, "seeds 0 and 1 produced identical patterns");
    }
}
