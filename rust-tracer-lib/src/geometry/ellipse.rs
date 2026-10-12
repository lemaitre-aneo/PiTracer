use rand::RngExt;
use serde::{Deserialize, Serialize};

use crate::{GeometryPrimitive, Hit, UnitVector, Vector, select};

mod definition {
    use serde::{Deserialize, Serialize};

    use crate::Vector;

    /// No `deny_unknown_fields`: an ellipse is flattened into an `Object`
    /// next to material and texture keys, which serde would otherwise reject.
    #[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[serde(default)]
    pub(super) struct Ellipse {
        pub position: Vector,
        pub u: Vector,
        pub v: Vector,
    }

    impl From<super::Ellipse> for Ellipse {
        fn from(value: super::Ellipse) -> Self {
            Self {
                position: value.position,
                u: value.u,
                v: value.v,
            }
        }
    }
}

/// Ellipse: center `position` (C) and semi-axis vectors `u`, `v`; surface
/// points are C + alpha*u + beta*v with alpha^2 + beta^2 <= 1 (the boundary
/// passes through C +/- u and C +/- v). The semi-axes need not be orthogonal:
/// the shape is the affine image of the unit disk, with area pi*|u x v|.
/// Two-sided (the hit normal opposes the incident ray, like every geometry).
///
/// Plane quantities are cached at construction; the definition/runtime split
/// keeps the fields consistent by making [`Ellipse::new`] (or serde) the only
/// way to build one. A degenerate ellipse (u x v vanishing or non-finite,
/// which includes `Ellipse::default()`) stores `area = 0` and is inert: never
/// hit, never sampled, never NaN. That is the safe failure mode for a YAML
/// entry whose semi-axes were mistyped away.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(from = "definition::Ellipse", into = "definition::Ellipse")]
pub struct Ellipse {
    position: Vector,
    u: Vector,
    v: Vector,
    /// normalize(u x v); arbitrary placeholder when `area == 0`.
    normal: UnitVector,
    /// (u x v) / |u x v|^2: dotted with p x v and u x p it yields
    /// (alpha, beta).
    w: Vector,
    /// pi*|u x v|; the pi is baked in so target_pdf's t^2 / (area * |cos|)
    /// stays the plain solid-angle conversion of uniform-area sampling.
    area: f32,
    /// |C| + |u| + |v| bounds the coordinate magnitude of every point on the
    /// ellipse: the scale the hit guard and spawn offset are relative to (the
    /// analogue of the sphere's radius + |center|).
    scale: f32,
}

impl Ellipse {
    pub fn new(position: Vector, u: Vector, v: Vector) -> Self {
        let n = u.cross(v);
        let n2 = n.norm2();
        if n2 > 0f32 && n2.is_finite() {
            Self {
                position,
                u,
                v,
                normal: n.normalized(),
                w: n / n2,
                area: std::f32::consts::PI * n2.sqrt(),
                scale: position.norm() + u.norm() + v.norm(),
            }
        } else {
            Self {
                position,
                u,
                v,
                normal: UnitVector::default(),
                w: Vector::default(),
                area: 0f32,
                scale: 0f32,
            }
        }
    }

    // The `!(>)` and `!(<=)` comparisons are deliberate: they keep NaN (and
    // the degenerate area = 0) on the reject path, which the positive forms
    // would let through.
    #[expect(clippy::neg_cmp_op_on_partial_ord)]
    fn hit(&self, origin: Vector, direction: UnitVector) -> Option<f32> {
        if !(self.area > 0f32) {
            return None;
        }
        let denom = direction.dot(self.normal);
        if denom == 0f32 {
            return None;
        }
        // Subtract-then-dot mirrors the sphere's oc = C - O: for an
        // axis-aligned disk the signed plane distance is one exact f32
        // subtraction; caching dot(n, C) would trade that for a cancellation.
        let s = self.normal.dot(self.position - origin);
        let t = s / denom;

        // Relative guard against re-hitting the plane an origin numerically
        // sits on; same rationale as the quad's (see quad.rs). Grazing is
        // Hit::spawn's job. `!(>)` also rejects a NaN t.
        if !(t > crate::HIT_EPSILON_REL * self.scale) {
            return None;
        }

        let p = origin + t * direction - self.position;
        let alpha = self.w.dot(p.cross(self.v));
        let beta = self.w.dot(self.u.cross(p));
        // Interior of the affine unit disk; `!(<=)` rejects NaN coordinates
        // (t = inf from a near-parallel denominator ends up here).
        if !(alpha * alpha + beta * beta <= 1f32) {
            return None;
        }
        Some(t)
    }

    fn target_pdf(&self, origin: Vector, scattered: UnitVector) -> f32 {
        let Some(t) = self.hit(origin, scattered) else {
            return 0f32;
        };
        // Uniform-area sampling seen in solid angle: p(w) = t^2 / (A * |cos|)
        // with A = pi*|u x v|. hit() succeeding implies cos != 0, so the
        // density is finite; it is 0 for every direction hit() rejects,
        // including the in-plane pencil from a shading point on the ellipse
        // itself (measure-zero; the NEE direct term skips it via its pl > 0
        // gate).
        t * t / (self.area * scattered.dot(self.normal).abs())
    }
}

impl From<definition::Ellipse> for Ellipse {
    fn from(value: definition::Ellipse) -> Self {
        Self::new(value.position, value.u, value.v)
    }
}

impl Default for Ellipse {
    fn default() -> Self {
        definition::Ellipse::default().into()
    }
}

impl<R: crate::Rng + ?Sized> GeometryPrimitive<R> for Ellipse {
    fn hit(&self, origin: Vector, direction: UnitVector, _rng: &mut R) -> Option<f32> {
        self.hit(origin, direction)
    }

    fn to_hit(&self, origin: Vector, direction: UnitVector, distance: f32) -> Hit {
        let point = origin + distance * direction;
        // Same crossing-direction convention as the sphere: `front` means the
        // ray came from the side the geometric normal (u x v) points toward.
        let front = direction.dot(self.normal) < 0f32;
        let normal = select(front, self.normal, -self.normal);

        let p = point - self.position;
        let alpha = self.w.dot(p.cross(self.v));
        let beta = self.w.dot(self.u.cross(p));

        let offset = crate::SURFACE_OFFSET_REL * self.scale;

        Hit {
            point,
            normal,
            front,
            // Disk coordinates in [-1, 1]: (0, 0) at the center, +/-1 on the
            // boundary along each semi-axis.
            u: alpha,
            v: beta,
            distance,
            offset,
        }
    }

    // Deliberate `!(>)`: NaN-safe degenerate check, see `Ellipse::hit`.
    #[expect(clippy::neg_cmp_op_on_partial_ord)]
    fn target_random(&self, origin: Vector, rng: &mut R) -> UnitVector {
        if !(self.area > 0f32) {
            return UnitVector::default();
        }
        // Uniform over the ellipse: uniform on the unit disk (r = sqrt(r1)
        // makes the radial density linear), then the affine map
        // C + alpha*u + beta*v, which preserves area-uniformity.
        let r = rng.random::<f32>().sqrt();
        let phi = 2f32 * std::f32::consts::PI * rng.random::<f32>();
        let alpha = r * phi.cos();
        let beta = r * phi.sin();
        (self.position + alpha * self.u + beta * self.v - origin).normalized()
    }

    fn target_pdf(&self, origin: Vector, scattered: UnitVector) -> f32 {
        self.target_pdf(origin, scattered)
    }
}

#[cfg(test)]
mod test {
    use rand::Rng;

    use super::*;
    use crate::test_utils::{
        assert_close, assert_sampler_matches_pdf, assert_stat_close, assert_vec_close, at_angle,
        chi2_directions, rng, seeds, sphere_integral,
    };

    fn ellipse_at(position: (f32, f32, f32), u: (f32, f32, f32), v: (f32, f32, f32)) -> Ellipse {
        Ellipse::new(position.into(), u.into(), v.into())
    }

    /// The pitracer.yml ceiling light this primitive exists for.
    fn pitracer_light() -> Ellipse {
        ellipse_at(
            (50f32, 81.5f32, 61.6f32),
            (12f32, 0f32, 0f32),
            (0f32, 0f32, 8f32),
        )
    }

    /// Ray-ellipse ground truth: the plane hit is
    /// t = dot(n, C - O) / dot(n, d), inside iff alpha^2 + beta^2 <= 1;
    /// aiming at a known surface point makes the expected distance simply
    /// |aim - origin| (f64).
    #[test]
    fn hit_distances_analytic() {
        // semi-axes 2 (x) and 4 (y) in the plane z = 5, centered at (1, -1)
        let ellipse = ellipse_at((1f32, -1f32, 5f32), (2f32, 0f32, 0f32), (0f32, 4f32, 0f32));
        let origin = Vector::default();
        let z = UnitVector::Z;

        // head-on through the center
        let t = ellipse
            .hit(Vector::from((1f32, -1f32, 0f32)), z)
            .expect("must hit");
        assert_close(t, 5f32, 1e-6);

        // oblique at a known interior point C + 0.5u + 0.5v = (2, 1, 5)
        let aim = Vector::from((2f32, 1f32, 5f32));
        let t = ellipse
            .hit(origin, (aim - origin).normalized())
            .expect("must hit");
        assert_close(t, 30f64.sqrt(), 1e-6);

        // THE quad-vs-ellipse discriminator: inside the bounding
        // parallelogram but outside the disk (alpha^2 + beta^2 = 1.62 > 1)
        let aim = Vector::from((1f32, -1f32, 5f32))
            + 0.9f32 * Vector::from((2f32, 0f32, 0f32))
            + 0.9f32 * Vector::from((0f32, 4f32, 0f32));
        assert_eq!(ellipse.hit(origin, (aim - origin).normalized()), None);

        // just inside / just outside the boundary along +u
        let boundary = Vector::from((2f32, 0f32, 0f32));
        let center = Vector::from((1f32, -1f32, 5f32));
        let aim = center + 0.999f32 * boundary;
        assert!(ellipse.hit(origin, (aim - origin).normalized()).is_some());
        let aim = center + 1.001f32 * boundary;
        assert_eq!(ellipse.hit(origin, (aim - origin).normalized()), None);

        // ellipse behind the ray: no phantom negative-t hit
        assert_eq!(ellipse.hit(Vector::from((1f32, -1f32, 10f32)), z), None);

        // direction parallel to the plane, origin off-plane
        assert_eq!(ellipse.hit(origin, UnitVector::X), None);

        // origin and direction both in the plane
        assert_eq!(
            ellipse.hit(Vector::from((1f32, -1f32, 5f32)), UnitVector::X),
            None
        );

        // the Default ellipse is degenerate and inert
        assert_eq!(Ellipse::default().hit(origin, z), None);
    }

    /// Pins the hit conventions materials rely on (normal opposes the ray,
    /// `front` = came from the u x v side) and the (alpha, beta) disk
    /// parametrization, including a sheared (non-orthogonal semi-axes) case.
    #[test]
    fn to_hit_normal_orientation_and_uv() {
        // u x v = +Z for this ellipse
        let ellipse = ellipse_at((1f32, -1f32, 5f32), (2f32, 0f32, 0f32), (0f32, 4f32, 0f32));

        // approaching from z < 5 travels along +Z, against the grain: back face
        let origin = Vector::from((1f32, -1f32, 0f32));
        let t = ellipse.hit(origin, UnitVector::Z).expect("must hit");
        let hit = GeometryPrimitive::<dyn Rng>::to_hit(&ellipse, origin, UnitVector::Z, t);
        assert!(!hit.front);
        assert_vec_close(hit.point, Vector::from((1f32, -1f32, 5f32)), 1e-6);
        assert_vec_close(
            Vector::from(hit.normal),
            Vector::from((0f32, 0f32, -1f32)),
            1e-6,
        );
        assert!(hit.normal.dot(UnitVector::Z) < 0f32);
        assert_close(hit.distance, t, 1e-6);
        // center maps to (0, 0)
        assert_close(hit.u, 0f32, 1e-6);
        assert_close(hit.v, 0f32, 1e-6);

        // approaching from z > 5: front face, normal +Z, still opposing
        let origin = Vector::from((1f32, -1f32, 10f32));
        let t = ellipse.hit(origin, -UnitVector::Z).expect("must hit");
        let hit = GeometryPrimitive::<dyn Rng>::to_hit(&ellipse, origin, -UnitVector::Z, t);
        assert!(hit.front);
        assert_vec_close(
            Vector::from(hit.normal),
            Vector::from((0f32, 0f32, 1f32)),
            1e-6,
        );
        assert!(hit.normal.dot(-UnitVector::Z) < 0f32);

        // sheared, tilted semi-axes: aim at C + 0.3u + 0.6v (interior:
        // 0.09 + 0.36 <= 1) and read back the disk coordinates
        let ellipse = ellipse_at((1f32, 1f32, 1f32), (1f32, 0f32, 1f32), (1f32, 1f32, 0f32));
        let aim = Vector::from((1f32, 1f32, 1f32))
            + 0.3f32 * Vector::from((1f32, 0f32, 1f32))
            + 0.6f32 * Vector::from((1f32, 1f32, 0f32));
        let origin = Vector::from((5f32, -3f32, 4f32));
        let direction = (aim - origin).normalized();
        let t = ellipse.hit(origin, direction).expect("must hit");
        let hit = GeometryPrimitive::<dyn Rng>::to_hit(&ellipse, origin, direction, t);
        assert_vec_close(hit.point, aim, 1e-5);
        assert_close(hit.u, 0.3f32, 1e-5);
        assert_close(hit.v, 0.6f32, 1e-5);
        assert!(hit.normal.dot(direction) < 0f32);
    }

    /// An origin lying exactly in the ellipse's plane must never hit it: the
    /// relative guard eats the t ~ 0 root for every non-parallel direction
    /// (parallel ones die on denom == 0).
    #[test]
    fn hit_guard_rejects_on_plane_origin() {
        let ellipse = pitracer_light();
        let origin = Vector::from((50f32, 81.5f32, 61f32)); // in-plane, interior
        for direction in [
            Vector::from((0f32, 1f32, 0f32)),
            Vector::from((0f32, -1f32, 0f32)),
            Vector::from((1f32, 1f32, 0.3f32)),
            Vector::from((-0.2f32, -1f32, 0.5f32)),
            Vector::from((1f32, 0.01f32, 0f32)), // shallow but not parallel
        ] {
            assert_eq!(ellipse.hit(origin, direction.normalized()), None);
        }
    }

    /// A ray spawned via Hit::spawn cannot re-intersect the plane it leaves
    /// at any angle: the origin sits strictly on the side the ray travels.
    #[test]
    fn spawn_clears_own_surface() {
        let mut rng = rng(seeds()[1]);
        for ellipse in [
            pitracer_light(),
            ellipse_at(
                (1f32, 2f32, 3f32),
                (0.25f32, 0f32, 0f32),
                (0f32, 0f32, 0.25f32),
            ),
        ] {
            let aim = ellipse.position + 0.3f32 * ellipse.u + 0.6f32 * ellipse.v;
            let origin = aim + 40f32 * ellipse.normal + 0.1f32 * ellipse.u;
            let direction = (aim - origin).normalized();
            let t = ellipse.hit(origin, direction).expect("must hit");
            let hit = GeometryPrimitive::<dyn Rng>::to_hit(&ellipse, origin, direction, t);

            let mut directions = vec![hit.normal];
            for angle in [30f64, 60f64, 85f64, 89.9f64, 89.99f64] {
                directions.push(at_angle(hit.normal, angle.to_radians()));
            }
            for _ in 0..200 {
                let w = UnitVector::random(&mut rng);
                directions.push(crate::select(w.dot(hit.normal) >= 0f32, w, -w));
            }

            for scattered in directions {
                let spawned = ellipse.hit(hit.spawn(scattered), scattered);
                assert!(
                    spawned.is_none(),
                    "self re-hit on ellipse at {:?}: t = {spawned:?}, dir·n = {}",
                    ellipse.position,
                    scattered.dot(hit.normal),
                );
            }
        }
    }

    /// The offset reported by to_hit must bound the hit point's true (f64)
    /// distance from the plane while staying far below the extent.
    #[test]
    fn to_hit_offset_bounds_placement_noise() {
        let mut rng = rng(seeds()[2]);
        for ellipse in [
            pitracer_light(),
            ellipse_at(
                (1f32, 2f32, 3f32),
                (0.25f32, 0f32, 0f32),
                (0f32, 0f32, 0.25f32),
            ),
        ] {
            let (u, v, c) = (ellipse.u, ellipse.v, ellipse.position);
            let n64 = [
                f64::from(u.y) * f64::from(v.z) - f64::from(u.z) * f64::from(v.y),
                f64::from(u.z) * f64::from(v.x) - f64::from(u.x) * f64::from(v.z),
                f64::from(u.x) * f64::from(v.y) - f64::from(u.y) * f64::from(v.x),
            ];
            let n64_norm = n64.iter().map(|c| c * c).sum::<f64>().sqrt();

            let extent = u.norm().min(v.norm());
            for _ in 0..50 {
                // interior aims: (a, b) in the disk of radius 0.7*sqrt(2) <= 0.99
                let a = (2f32 * rng.random::<f32>() - 1f32) * 0.7f32;
                let b = (2f32 * rng.random::<f32>() - 1f32) * 0.7f32;
                let aim = c + a * u + b * v;
                let side = crate::select(rng.random::<f32>() < 0.5f32, 1f32, -1f32);
                let origin = aim
                    + (10f32 + 30f32 * rng.random::<f32>()) * side * ellipse.normal
                    + (rng.random::<f32>() - 0.5f32) * u
                    + (rng.random::<f32>() - 0.5f32) * v;
                let direction = (aim - origin).normalized();
                let Some(t) = ellipse.hit(origin, direction) else {
                    continue; // lateral jitter may push the aim off the disk
                };
                let hit = GeometryPrimitive::<dyn Rng>::to_hit(&ellipse, origin, direction, t);

                assert!(hit.offset > 0f32);
                assert!(
                    hit.offset < 1e-4f32 * extent,
                    "offset {} not small vs extent {extent}",
                    hit.offset
                );

                let err = [
                    (f64::from(hit.point.x) - f64::from(c.x)) * n64[0],
                    (f64::from(hit.point.y) - f64::from(c.y)) * n64[1],
                    (f64::from(hit.point.z) - f64::from(c.z)) * n64[2],
                ]
                .iter()
                .sum::<f64>()
                    / n64_norm;
                assert!(
                    err.abs() < f64::from(hit.offset),
                    "placement noise {err:e} exceeds offset {:e}",
                    hit.offset,
                );
            }
        }
    }

    /// Every sampled direction re-hits the ellipse with a positive pdf, from
    /// origins on both sides; directions toward the bounding-parallelogram
    /// corner, away from the plane, or parallel to it score 0.
    #[test]
    fn target_random_hits_ellipse() {
        let ellipse = ellipse_at((1f32, -1f32, 5f32), (2f32, 0f32, 0f32), (0f32, 4f32, 0f32));
        for seed in seeds() {
            let mut rng = rng(seed);
            for origin in [
                Vector::default(),
                Vector::from((3f32, 1f32, -2f32)),
                Vector::from((0.5f32, -0.5f32, 8f32)), // other side of the plane
            ] {
                for _ in 0..1000 {
                    let direction = ellipse.target_random(origin, &mut rng);
                    assert!(
                        ellipse.hit(origin, direction).is_some(),
                        "sample misses the ellipse from {origin:?}: {direction:?}"
                    );
                    assert!(ellipse.target_pdf(origin, direction) > 0f32);
                }
            }
        }

        let origin = Vector::default();
        // inside the bounding parallelogram, outside the disk
        let corner = Vector::from((1f32, -1f32, 5f32))
            + 0.9f32 * Vector::from((2f32, 0f32, 0f32))
            + 0.9f32 * Vector::from((0f32, 4f32, 0f32));
        assert_eq!(
            ellipse.target_pdf(origin, (corner - origin).normalized()),
            0f32
        );
        // away from the plane
        assert_eq!(ellipse.target_pdf(origin, -UnitVector::Z), 0f32);
        // parallel to the plane
        assert_eq!(ellipse.target_pdf(origin, UnitVector::X), 0f32);
    }

    /// Uniform-area sampling in solid angle:
    /// p(w) = t^2 / (pi * |u x v| * |cos|). For the unit disk at distance 1:
    /// head-on p = 1/pi; toward (0.5, 0, 1), t^2 = 1.25 and
    /// cos = 1/sqrt(1.25), so p = 1.25^1.5 / pi. Two-sided: the same value
    /// approaching from behind.
    #[test]
    fn target_pdf_closed_form() {
        let disk = ellipse_at((0f32, 0f32, 1f32), (1f32, 0f32, 0f32), (0f32, 1f32, 0f32));
        let origin = Vector::default();

        assert_close(
            disk.target_pdf(origin, UnitVector::Z),
            std::f64::consts::FRAC_1_PI,
            1e-6,
        );
        assert_close(
            disk.target_pdf(origin, Vector::from((0.5f32, 0f32, 1f32)).normalized()),
            1.25f64.powf(1.5f64) / std::f64::consts::PI,
            1e-6,
        );
        assert_eq!(
            disk.target_pdf(origin, Vector::from((1.5f32, 0f32, 1f32)).normalized()),
            0f32
        );

        // from the other side of the plane
        let behind = Vector::from((0f32, 0f32, 2f32));
        assert_close(
            disk.target_pdf(behind, -UnitVector::Z),
            std::f64::consts::FRAC_1_PI,
            1e-6,
        );
    }

    /// target_pdf must integrate to 1 over the sphere of directions. This
    /// also pins the pi in the area: with area = |u x v| the integral would
    /// read pi.
    #[test]
    fn target_pdf_normalizes() {
        let disk = ellipse_at((0f32, 0f32, 1f32), (1f32, 0f32, 0f32), (0f32, 1f32, 0f32));
        let origin = Vector::default();

        let mut rng = rng(seeds()[1]);
        let (integral, se) = sphere_integral(1_000_000, &mut rng, |w| disk.target_pdf(origin, w));
        assert_stat_close(integral, se, 1f32, 4f32, 0f32);
    }

    /// target_random must be distributed according to target_pdf.
    ///
    /// Probe-based consistency on an elliptic (unequal axes) case, which is
    /// not phi-symmetric about any axis. Then a chi-squared goodness-of-fit
    /// on the CIRCULAR on-axis case, which (unlike any quad) IS phi-symmetric:
    /// for a disk of radius R at distance d on the axis, the fraction of the
    /// (area-uniform) samples with direction cosine >= z is the area fraction
    /// of radius <= d*tan(theta), i.e. d^2*(1 - z^2)/(R^2*z^2), giving the
    /// slice probability bin_prob(z0, z1) = (d^2/R^2)*(1/z0^2 - 1/z1^2) over
    /// the support z in [d/sqrt(d^2 + R^2), 1] (the full range sums to
    /// exactly 1).
    #[test]
    fn target_sampler_matches_target_pdf() {
        let ellipse = ellipse_at((0f32, 0f32, 1f32), (2f32, 0f32, 0f32), (0f32, 1f32, 0f32));
        let origin = Vector::default();

        let mut rng = rng(seeds()[0]);
        assert_sampler_matches_pdf(
            100_000,
            1_000_000,
            &mut rng,
            UnitVector::Z,
            |rng| ellipse.target_random(origin, rng),
            |w| ellipse.target_pdf(origin, w),
        );

        // circular case: R = 1 at d = 1 => z_min = 1/sqrt(2)
        let disk = ellipse_at((0f32, 0f32, 1f32), (1f32, 0f32, 0f32), (0f32, 1f32, 0f32));
        let z_min = 1f64 / 2f64.sqrt();
        chi2_directions(
            50_000,
            &mut rng,
            UnitVector::Z,
            z_min,
            10,
            10,
            |rng| disk.target_random(origin, rng),
            |z0, z1| 1f64 / (z0 * z0) - 1f64 / (z1 * z1),
        );
    }

    /// Degenerate ellipses (Default, parallel semi-axes, zero semi-axis) are
    /// inert: no hits, zero pdf, and a finite placeholder sampling direction.
    #[test]
    fn degenerate_ellipse_inert() {
        let mut rng = rng(seeds()[0]);
        for ellipse in [
            Ellipse::default(),
            ellipse_at((1f32, 2f32, 3f32), (1f32, 1f32, 0f32), (2f32, 2f32, 0f32)),
            ellipse_at((1f32, 2f32, 3f32), (0f32, 0f32, 0f32), (0f32, 4f32, 0f32)),
        ] {
            for _ in 0..10 {
                let origin = 5f32 * Vector::from(UnitVector::random(&mut rng));
                let direction = UnitVector::random(&mut rng);
                assert_eq!(ellipse.hit(origin, direction), None);
                assert_eq!(ellipse.target_pdf(origin, direction), 0f32);
                let sample = ellipse.target_random(origin, &mut rng);
                assert!(sample.x.is_finite() && sample.y.is_finite() && sample.z.is_finite());
            }
        }
    }
}
