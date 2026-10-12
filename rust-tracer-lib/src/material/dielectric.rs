use rand::RngExt;
use serde::{Deserialize, Serialize};

use crate::{Hit, Material, UnitVector, select};

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(default)]
pub struct Dielectric {
    refraction_index: f32,
    #[serde(skip)]
    refraction_index_reciprocal: f32,
    #[serde(skip)]
    base_reflectance: f32,
}

impl Default for Dielectric {
    fn default() -> Self {
        Self::new(1.5f32)
    }
}

impl Dielectric {
    pub fn new(refraction_index: f32) -> Self {
        let diff = refraction_index - 1f32;
        let sum = refraction_index + 1f32;
        Dielectric {
            refraction_index,
            refraction_index_reciprocal: 1f32 / refraction_index,
            base_reflectance: diff * diff / (sum * sum),
        }
    }
}

impl<'de> Deserialize<'de> for Dielectric {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let index = {
            #[derive(Default, Deserialize)]
            #[serde(default)]
            struct Dielectric {
                refraction_index: Option<f32>,
            }

            Dielectric::deserialize(deserializer)?
                .refraction_index
                .unwrap_or(1.5f32)
        };

        Ok(Dielectric::new(index))
    }
}

impl<R: crate::Rng + ?Sized> Material<R> for Dielectric {
    fn scatter(&self, direction: UnitVector, hit: Hit, rng: &mut R) -> Option<UnitVector> {
        let refraction_factor = select(
            hit.front,
            self.refraction_index_reciprocal,
            self.refraction_index,
        );

        let cost = direction.dot(-hit.normal).min(1f32);
        let sint = (1f32 - cost * cost).sqrt();

        let cannot_refract = refraction_factor * sint > 1f32;

        let reflectance =
            self.base_reflectance + (1f32 - self.base_reflectance) * (1f32 - cost).powi(5);

        if cannot_refract || reflectance > rng.random::<f32>() {
            Some(direction.reflect(hit.normal))
        } else {
            let perp = refraction_factor * (direction + cost * hit.normal);
            let parallel = -(1f32 - perp.norm2()).abs().sqrt() * hit.normal;
            Some((perp + parallel).normalized())
        }
    }
}

// Ground truth (all derived from optics, not from the code):
// - Snell: entering a medium of index eta, sin(theta_t) = sin(theta_i)/eta;
//   the refracted ray is coplanar with (d, n), on the far side of the
//   surface, tangentially forward. Exiting, the relative index is eta and
//   refraction is impossible beyond the critical angle theta_c = asin(1/eta).
// - Fresnel (Schlick): reflect with probability
//   R(theta) = R0 + (1 - R0)*(1 - cos(theta))^5, R0 = ((eta-1)/(eta+1))^2.
#[cfg(test)]
mod test {
    use super::*;
    use crate::Vector;
    use crate::test_utils::{
        Stats, assert_close, assert_stat_close, assert_vec_close, hit_at, rng, seeds,
    };

    const NORMAL: UnitVector = UnitVector::Y;

    /// Incident direction at angle `theta` from the surface normal, opposing it.
    fn incident(theta: f32) -> UnitVector {
        Vector::from((theta.sin(), -theta.cos(), 0f32)).normalized()
    }

    fn schlick(eta: f32, theta: f32) -> f32 {
        let r0 = ((eta - 1f32) / (eta + 1f32)).powi(2);
        r0 + (1f32 - r0) * (1f32 - theta.cos()).powi(5)
    }

    /// Refracted rays (on the far side: dot(t, n) < 0) satisfy Snell's law
    /// and stay in the plane of incidence.
    #[test]
    fn snell_angle() {
        for (theta, eta) in [
            (45f32.to_radians(), 1.5f32),
            (20f32.to_radians(), 1.5f32),
            (45f32.to_radians(), 1.33f32),
        ] {
            let material: &dyn Material = &Dielectric::new(eta);
            let incoming = incident(theta);
            let hit = hit_at(Vector::default(), NORMAL, true);
            let mut rng = rng(seeds()[0]);

            let mut refracted = 0;
            for _ in 0..10_000 {
                let scattered = material
                    .scatter(incoming, hit, &mut rng)
                    .expect("must scatter");
                if scattered.dot(NORMAL) > 0f32 {
                    continue;
                }
                refracted += 1;

                let sin_t = (1f32 - scattered.dot(NORMAL).powi(2)).sqrt();
                assert_close(sin_t, theta.sin() / eta, 1e-5);
                // coplanar with the plane of incidence
                assert_close(
                    scattered.dot(Vector::from(incoming).cross(NORMAL)),
                    0f32,
                    1e-6,
                );
                // tangentially forward
                let tangential = incoming - incoming.dot(NORMAL) * NORMAL;
                assert!(scattered.dot(tangential) > 0f32);

                if refracted >= 100 {
                    break;
                }
            }
            assert!(
                refracted >= 100,
                "only {refracted} refracted samples in 10⁴ draws"
            );
        }
    }

    #[test]
    fn reflected_branch_is_mirror() {
        let material: &dyn Material = &Dielectric::new(1.5f32);
        let incoming = incident(60f32.to_radians());
        let hit = hit_at(Vector::default(), NORMAL, true);
        let mirror = incoming.reflect(NORMAL);
        let mut rng = rng(seeds()[1]);

        let mut reflected = 0;
        for _ in 0..10_000 {
            let scattered = material
                .scatter(incoming, hit, &mut rng)
                .expect("must scatter");
            if scattered.dot(NORMAL) > 0f32 {
                reflected += 1;
                assert_vec_close(Vector::from(scattered), Vector::from(mirror), 1e-7);
            }
        }
        // R(60 degrees) = 0.07: ~700 reflections expected in 1e4 draws
        assert!(reflected > 0, "no reflected sample in 10⁴ draws");
    }

    /// Exiting the medium (relative index eta), refraction is impossible when
    /// eta * sin(theta) > 1, i.e. beyond theta_c = asin(1/1.5) ~ 41.81 deg.
    #[test]
    fn total_internal_reflection() {
        let material: &dyn Material = &Dielectric::new(1.5f32);
        let hit = hit_at(Vector::default(), NORMAL, false);
        let mut rng = rng(seeds()[2]);

        let incoming = incident(50f32.to_radians());
        let mirror = incoming.reflect(NORMAL);
        for _ in 0..10_000 {
            let scattered = material
                .scatter(incoming, hit, &mut rng)
                .expect("must scatter");
            assert_vec_close(Vector::from(scattered), Vector::from(mirror), 1e-7);
        }

        // below the critical angle transmission resumes (1 - R(30 deg) ~ 96%)
        let incoming = incident(30f32.to_radians());
        let refracted = (0..10_000)
            .filter(|_| {
                let scattered = material
                    .scatter(incoming, hit, &mut rng)
                    .expect("must scatter");
                scattered.dot(NORMAL) < 0f32
            })
            .count();
        assert!(refracted > 0, "no refraction below the critical angle");
    }

    /// The reflect/refract branch frequency must follow the Schlick
    /// reflectance (binomial statistics, 4 sigma).
    #[test]
    fn schlick_branch_statistics() {
        let eta = 1.5f32;
        let material: &dyn Material = &Dielectric::new(eta);
        let hit = hit_at(Vector::default(), NORMAL, true);

        for (theta, seed) in [
            (0f32, seeds()[0]),
            (60f32.to_radians(), seeds()[1]),
            (80f32.to_radians(), seeds()[2]),
        ] {
            let incoming = incident(theta);
            let expected = schlick(eta, theta);
            let mut rng = rng(seed);

            let n = 100_000;
            let mut stats = Stats::new();
            for _ in 0..n {
                let scattered = material
                    .scatter(incoming, hit, &mut rng)
                    .expect("must scatter");
                stats.push(crate::select(scattered.dot(NORMAL) > 0f32, 1f32, 0f32));
            }
            let se = (expected * (1f32 - expected) / n as f32).sqrt();
            assert_stat_close(stats.mean(), se, expected, 4f32, 0f32);
        }
    }

    /// Glass absorbs nothing: scatter never returns None, entering or exiting.
    #[test]
    fn always_scatters() {
        let material: &dyn Material = &Dielectric::new(1.5f32);
        let mut rng = rng(seeds()[0]);
        for front in [true, false] {
            let hit = hit_at(Vector::default(), NORMAL, front);
            for degrees in 0..90 {
                let incoming = incident((degrees as f32).to_radians());
                for _ in 0..100 {
                    assert!(material.scatter(incoming, hit, &mut rng).is_some());
                }
            }
        }
    }
}
