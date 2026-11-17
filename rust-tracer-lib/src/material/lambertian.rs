use serde::{Deserialize, Serialize};

use crate::{Hit, Material, UnitVector};

#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Lambertian {}

impl<R: crate::Rng + ?Sized> Material<R> for Lambertian {
    fn scatter(&self, _direction: UnitVector, hit: Hit, rng: &mut R) -> Option<UnitVector> {
        // Cosine sampling is load-bearing, not a variance tweak: the renderer
        // multiplies throughput by the albedo alone, which is only correct
        // when p = cos(theta)/pi (see the ground-truth note below).
        let scattered = UnitVector::random_cosine(hit.normal, rng);
        // With a shading normal (smooth mesh), the lobe can dip below the
        // actual surface: such a ray would start inside the object (see
        // `Hit::spawn`) and, in a closed mesh, never get out. It is absorbed
        // instead.
        (scattered.dot(hit.geometric) > 0f32).then_some(scattered)
    }

    fn pdf(&self, _direction: UnitVector, hit: Hit, scattered: UnitVector) -> f32 {
        // cos(theta)/pi above the horizon; the max clamps below-horizon
        // directions to 0.
        let cost = scattered.dot(hit.normal);
        (cost * std::f32::consts::FRAC_1_PI).max(0f32)
    }
}

// Ground truth: the BRDF-sampling path tracer multiplies the throughput by
// the albedo only, which is unbiased iff the scatter density is exactly
// p(w) = cos(theta)/pi (then f*cos/p = (rho/pi)*cos/(cos/pi) = rho). MIS
// additionally requires pdf() to return the true density of scatter().
#[cfg(test)]
mod test {
    use std::f32::consts::PI;

    use super::*;
    use crate::Vector;
    use crate::test_utils::{
        assert_close, assert_sampler_matches_pdf, assert_stat_close, at_angle, chi2_directions,
        hit_at, rng, sampler_mean, seeds, sphere_integral,
    };

    const MATERIAL: &dyn Material = &Lambertian {};

    #[test]
    fn scatter_in_upper_hemisphere() {
        let normals = [
            UnitVector::Y,
            Vector::from((1f32, 2f32, 3f32)).normalized(),
            Vector::from((-0.3f32, -0.9f32, 0.6f32)).normalized(),
        ];
        let mut rng = rng(seeds()[0]);
        for normal in normals {
            let hit = hit_at(Vector::default(), normal, true);
            let incoming = at_angle(-normal, 0.7f64);
            for _ in 0..10_000 {
                let scattered = MATERIAL
                    .scatter(incoming, hit, &mut rng)
                    .expect("must scatter");
                assert!(
                    scattered.dot(normal) > 0f32,
                    "scattered below the horizon of {normal:?}: {scattered:?}",
                );
            }
        }
    }

    /// With a shading normal tilted from the geometric one (smooth mesh),
    /// the part of the cosine lobe below the actual surface is absorbed
    /// rather than sent into the object; the rest is unchanged. Tilt 0.5 rad:
    /// the lobe's mass below the plane is (1 - cos 0.5) / 2 ~ 6%.
    #[test]
    fn absorbed_below_the_surface() {
        let geometric = UnitVector::Y;
        let normal = at_angle(geometric, 0.5f64);
        let hit = Hit {
            geometric,
            ..hit_at(Vector::default(), normal, true)
        };
        let incoming = -normal;
        let mut rng = rng(seeds()[0]);
        let n = 100_000;
        let mut absorbed = 0;
        for _ in 0..n {
            match MATERIAL.scatter(incoming, hit, &mut rng) {
                Some(scattered) => assert!(scattered.dot(geometric) > 0f32, "{scattered:?}"),
                None => absorbed += 1,
            }
        }
        let fraction = f64::from(absorbed) / f64::from(n);
        let expected = (1f64 - 0.5f64.cos()) / 2f64;
        let se = (expected * (1f64 - expected) / f64::from(n)).sqrt();
        assert_stat_close(fraction, se, expected, 4f32, 0f32);
    }

    #[test]
    fn pdf_closed_form() {
        let normal = UnitVector::Y;
        let hit = hit_at(Vector::default(), normal, true);
        let incoming = at_angle(-normal, 0.5f64);

        // pdf(w) = cos(theta)/pi at 0, 60, 89.9 degrees; 0 below the horizon
        assert_close(MATERIAL.pdf(incoming, hit, normal), 1f32 / PI, 1e-6);
        assert_close(
            MATERIAL.pdf(incoming, hit, at_angle(normal, 60f64.to_radians())),
            0.5f32 / PI,
            1e-5,
        );
        assert_close(
            MATERIAL.pdf(incoming, hit, at_angle(normal, 89.9f64.to_radians())),
            89.9f32.to_radians().cos() / PI,
            1e-5,
        );
        assert_eq!(
            MATERIAL.pdf(incoming, hit, at_angle(normal, 120f64.to_radians())),
            0f32
        );
        assert_eq!(MATERIAL.pdf(incoming, hit, -normal), 0f32);
    }

    #[test]
    fn pdf_integrates_to_one() {
        let normal = Vector::from((1f32, 2f32, 3f32)).normalized();
        let hit = hit_at(Vector::default(), normal, true);
        let incoming = at_angle(-normal, 0.7f64);

        let mut rng = rng(seeds()[0]);
        let (integral, se) =
            sphere_integral(1_000_000, &mut rng, |w| MATERIAL.pdf(incoming, hit, w));
        assert_stat_close(integral, se, 1f32, 4f32, 0f32);
    }

    /// scatter() must be distributed according to pdf(), independently of the
    /// incoming direction, and match the absolute anchor E[dot(w, n)] = 2/3
    /// of the cosine density (see vector.rs for the derivation).
    #[test]
    fn scatter_matches_pdf() {
        let normal = Vector::from((0.5f32, 1f32, -0.25f32)).normalized();
        let hit = hit_at(Vector::default(), normal, true);

        for (seed, angle) in [(seeds()[0], 0.3f64), (seeds()[1], 1.2f64)] {
            let incoming = at_angle(-normal, angle);
            let mut rng = rng(seed);
            assert_sampler_matches_pdf(
                100_000,
                1_000_000,
                &mut rng,
                normal,
                |rng| MATERIAL.scatter(incoming, hit, rng).expect("must scatter"),
                |w| MATERIAL.pdf(incoming, hit, w),
            );

            let (mean, se) = sampler_mean(
                100_000,
                &mut rng,
                |rng| MATERIAL.scatter(incoming, hit, rng).expect("must scatter"),
                |w| w.dot(normal),
            );
            assert_stat_close(mean, se, 2f32 / 3f32, 4f32, 0f32);

            chi2_directions(
                50_000,
                &mut rng,
                normal,
                0f64,
                10,
                10,
                |rng| MATERIAL.scatter(incoming, hit, rng).expect("must scatter"),
                |z0, z1| z1 * z1 - z0 * z0,
            );
        }
    }
}
