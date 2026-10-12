use rand::RngExt;
use serde::{Deserialize, Serialize};

use crate::{GeometryPrimitive, Hit, UnitVector, Vector, select};

mod definition {
    use serde::{Deserialize, Serialize};

    use crate::Vector;

    /// No `deny_unknown_fields`: a quad is flattened into an `Object` next to
    /// material and texture keys, which serde would otherwise reject here.
    #[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[serde(default)]
    pub(super) struct Quad {
        pub position: Vector,
        pub u: Vector,
        pub v: Vector,
    }

    impl From<super::Quad> for Quad {
        fn from(value: super::Quad) -> Self {
            Self {
                position: value.position,
                u: value.u,
                v: value.v,
            }
        }
    }
}

/// Parallelogram: corner `position` (Q) and edge vectors `u`, `v`; surface
/// points are Q + alpha*u + beta*v with (alpha, beta) in [0, 1]^2. Two-sided
/// (the hit normal opposes the incident ray, like every geometry).
///
/// Plane quantities are cached at construction; the definition/runtime split
/// keeps the fields consistent by making [`Quad::new`] (or serde) the only way
/// to build one. A degenerate quad (u x v vanishing or non-finite, which
/// includes `Quad::default()`) stores `area = 0` and is inert: never hit,
/// never sampled, never NaN. That is the safe failure mode for a YAML entry
/// whose edges were mistyped away.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(from = "definition::Quad", into = "definition::Quad")]
pub struct Quad {
    position: Vector,
    u: Vector,
    v: Vector,
    /// normalize(u x v); arbitrary placeholder when `area == 0`.
    normal: UnitVector,
    /// (u x v) / |u x v|^2: dotted with p x v and u x p it yields
    /// (alpha, beta).
    w: Vector,
    area: f32,
    /// |Q| + |u| + |v| bounds the coordinate magnitude of every point on the
    /// quad: the scale the hit guard and spawn offset are relative to (the
    /// analogue of the sphere's radius + |center|).
    scale: f32,
}

impl Quad {
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
                area: n2.sqrt(),
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

    // The `!(>)` comparisons are deliberate: they keep NaN (and the
    // degenerate area = 0) on the reject path, which `<=` would let through.
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
        // axis-aligned wall the signed plane distance is one exact f32
        // subtraction; caching dot(n, Q) would trade that for a cancellation.
        let s = self.normal.dot(self.position - origin);
        let t = s / denom;

        // Relative guard against re-hitting the plane an origin numerically
        // sits on (|origin| <= ~scale there, so the rounding of `s` is a few
        // ulps of `scale` and the phantom t falls below it -- e.g. a camera
        // placed exactly in a wall's plane). Origins far away but coplanar
        // produce tiny-t phantoms whose hit point ~= origin lands far outside
        // the (alpha, beta) bounds instead. Grazing is Hit::spawn's job: a
        // ray spawned strictly off a plane, traveling away from it, can never
        // re-cross it. `!(>)` also rejects a NaN t.
        if !(t > crate::HIT_EPSILON_REL * self.scale) {
            return None;
        }

        let p = origin + t * direction - self.position;
        let alpha = self.w.dot(p.cross(self.v));
        let beta = self.w.dot(self.u.cross(p));
        // `!contains` rejects NaN coordinates (t = inf from a near-parallel
        // denominator ends up here as NaN/inf arithmetic).
        if !(0f32..=1f32).contains(&alpha) || !(0f32..=1f32).contains(&beta) {
            return None;
        }
        Some(t)
    }

    fn target_pdf(&self, origin: Vector, scattered: UnitVector) -> f32 {
        let Some(t) = self.hit(origin, scattered) else {
            return 0f32;
        };
        // Uniform-area sampling seen in solid angle:
        // p(w) = t^2 / (A * |cos|). hit() succeeding implies cos != 0, so the
        // density is finite; it is 0 for every direction hit() rejects,
        // including the whole in-plane pencil from a shading point on the
        // quad itself, where the sampler can only produce directions this pdf
        // scores 0 (measure-zero, the planar analogue of the sphere's
        // exact-tangency corner; the NEE direct term is skipped there by its
        // pl > 0 gate).
        t * t / (self.area * scattered.dot(self.normal).abs())
    }
}

impl From<definition::Quad> for Quad {
    fn from(value: definition::Quad) -> Self {
        Self::new(value.position, value.u, value.v)
    }
}

impl Default for Quad {
    fn default() -> Self {
        definition::Quad::default().into()
    }
}

impl<R: crate::Rng + ?Sized> GeometryPrimitive<R> for Quad {
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

        // Placement noise scales with the quad's coordinate magnitude, not
        // with a wall-sized bounding sphere: for the pitracer walls this is
        // ~4e-4 where the r = 1e5 spheres forced ~0.2, i.e. 500x less
        // receiver displacement bias per bounce.
        let offset = crate::SURFACE_OFFSET_REL * self.scale;

        Hit {
            point,
            normal,
            front,
            u: alpha,
            v: beta,
            distance,
            offset,
        }
    }

    // Deliberate `!(>)`: NaN-safe degenerate check, see `Quad::hit`.
    #[expect(clippy::neg_cmp_op_on_partial_ord)]
    fn target_random(&self, origin: Vector, rng: &mut R) -> UnitVector {
        if !(self.area > 0f32) {
            return UnitVector::default();
        }
        let s = rng.random::<f32>();
        let t = rng.random::<f32>();
        (self.position + s * self.u + t * self.v - origin).normalized()
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
        rng, seeds, sphere_integral,
    };

    fn quad_at(position: (f32, f32, f32), u: (f32, f32, f32), v: (f32, f32, f32)) -> Quad {
        Quad::new(position.into(), u.into(), v.into())
    }

    /// The actual pitracer.yml walls the quad primitive exists to replace.
    fn pitracer_floor() -> Quad {
        quad_at(
            (-4f32, 0f32, -5f32),
            (108f32, 0f32, 0f32),
            (0f32, 0f32, 310f32),
        )
    }
    fn pitracer_front_wall() -> Quad {
        quad_at(
            (-4f32, -5f32, 300f32),
            (108f32, 0f32, 0f32),
            (0f32, 91.6f32, 0f32),
        )
    }

    /// Ray-parallelogram ground truth: the plane hit is
    /// t = dot(n, Q - O) / dot(n, d), inside iff (alpha, beta) in [0, 1]^2;
    /// aiming at a known surface point makes the expected distance simply
    /// |aim - origin| (f64).
    #[test]
    fn hit_distances_analytic() {
        // 2x4 quad in the plane z = 5, x in [-1, 1], y in [-2, 2]
        let quad = quad_at((-1f32, -2f32, 5f32), (2f32, 0f32, 0f32), (0f32, 4f32, 0f32));
        let origin = Vector::default();
        let z = UnitVector::Z;

        // head-on through the interior
        let t = quad.hit(origin, z).expect("must hit");
        assert_close(t, 5f32, 1e-6);

        // oblique at a known interior point (0, 1, 5): t = sqrt(26)
        let aim = Vector::from((0f32, 1f32, 5f32));
        let t = quad
            .hit(origin, (aim - origin).normalized())
            .expect("must hit");
        assert_close(t, 26f64.sqrt(), 1e-6);

        // corner is inclusive
        let aim = Vector::from((-1f32, -2f32, 5f32));
        let t = quad
            .hit(origin, (aim - origin).normalized())
            .expect("must hit");
        assert_close(t, 30f64.sqrt(), 1e-6);

        // just outside either bound: the plane is hit, the quad is not
        for aim in [(1.01f32, 0f32, 5f32), (0f32, 2.01f32, 5f32)] {
            let aim = Vector::from(aim);
            assert_eq!(quad.hit(origin, (aim - origin).normalized()), None);
        }

        // quad behind the ray: no phantom negative-t hit
        assert_eq!(quad.hit(origin, -z), None);
        assert_eq!(quad.hit(Vector::from((0f32, 0f32, 10f32)), z), None);

        // direction parallel to the plane, origin off-plane
        assert_eq!(quad.hit(origin, UnitVector::X), None);

        // origin and direction both in the plane
        assert_eq!(
            quad.hit(Vector::from((0f32, 0f32, 5f32)), UnitVector::X),
            None
        );

        // the Default quad is degenerate and inert
        assert_eq!(Quad::default().hit(origin, z), None);
    }

    /// Pins the hit conventions materials rely on (normal opposes the ray,
    /// `front` = came from the u x v side) and the (alpha, beta)
    /// parametrization, including a sheared parallelogram where (u, v) are
    /// not orthogonal.
    #[test]
    fn to_hit_normal_orientation_and_uv() {
        // u x v = +Z for this quad
        let quad = quad_at((-1f32, -2f32, 5f32), (2f32, 0f32, 0f32), (0f32, 4f32, 0f32));

        // approaching from z < 5 travels along +Z, against-the-grain: back face
        let origin = Vector::default();
        let t = quad.hit(origin, UnitVector::Z).expect("must hit");
        let hit = GeometryPrimitive::<dyn Rng>::to_hit(&quad, origin, UnitVector::Z, t);
        assert!(!hit.front);
        assert_vec_close(hit.point, Vector::from((0f32, 0f32, 5f32)), 1e-6);
        assert_vec_close(
            Vector::from(hit.normal),
            Vector::from((0f32, 0f32, -1f32)),
            1e-6,
        );
        assert!(hit.normal.dot(UnitVector::Z) < 0f32);
        assert_close(hit.distance, t, 1e-6);
        assert_close(hit.u, 0.5f32, 1e-6);
        assert_close(hit.v, 0.5f32, 1e-6);

        // approaching from z > 5: front face, normal +Z, still opposing
        let origin = Vector::from((0f32, 0f32, 10f32));
        let t = quad.hit(origin, -UnitVector::Z).expect("must hit");
        let hit = GeometryPrimitive::<dyn Rng>::to_hit(&quad, origin, -UnitVector::Z, t);
        assert!(hit.front);
        assert_vec_close(
            Vector::from(hit.normal),
            Vector::from((0f32, 0f32, 1f32)),
            1e-6,
        );
        assert!(hit.normal.dot(-UnitVector::Z) < 0f32);

        // (alpha, beta) at the far corner Q + u + v
        let origin = Vector::from((1f32, 2f32, 0f32));
        let t = quad.hit(origin, UnitVector::Z).expect("must hit");
        let hit = GeometryPrimitive::<dyn Rng>::to_hit(&quad, origin, UnitVector::Z, t);
        assert_close(hit.u, 1f32, 1e-6);
        assert_close(hit.v, 1f32, 1e-6);

        // sheared parallelogram in a tilted plane: aim at Q + 0.3u + 0.6v
        let quad = quad_at((1f32, 1f32, 1f32), (1f32, 0f32, 1f32), (1f32, 1f32, 0f32));
        let aim = Vector::from((1f32, 1f32, 1f32))
            + 0.3f32 * Vector::from((1f32, 0f32, 1f32))
            + 0.6f32 * Vector::from((1f32, 1f32, 0f32));
        let origin = Vector::from((5f32, -3f32, 4f32));
        let direction = (aim - origin).normalized();
        let t = quad.hit(origin, direction).expect("must hit");
        let hit = GeometryPrimitive::<dyn Rng>::to_hit(&quad, origin, direction, t);
        assert_vec_close(hit.point, aim, 1e-5);
        assert_close(hit.u, 0.3f32, 1e-5);
        assert_close(hit.v, 0.6f32, 1e-5);
        assert!(hit.normal.dot(direction) < 0f32);
    }

    /// Plane distances at the coordinates the pitracer walls actually live at
    /// must match the f64 ground truth: aiming at a known surface point makes
    /// it |aim - origin| exactly.
    #[test]
    fn hit_distance_accurate_at_wall_scale() {
        // 45 degrees onto the floor from y = 30: t = 30*sqrt(2)
        let origin = Vector::from((50f32, 30f32, 100f32));
        let direction = Vector::from((1f32, -1f32, 0f32)).normalized();
        let t = pitracer_floor().hit(origin, direction).expect("must hit");
        assert_close(t, 30f64 * 2f64.sqrt(), 1e-6);

        // front wall from the camera position, straight on: t = 300 - z0
        let origin = Vector::from((50f32, 52f32, 295.6f32));
        let t = pitracer_front_wall()
            .hit(origin, UnitVector::Z)
            .expect("must hit");
        assert_close(t, 300f64 - f64::from(295.6f32), 1e-6);

        // and obliquely, aiming at a known point on the wall
        let aim = Vector::from((20f32, 40f32, 300f32));
        let direction = (aim - origin).normalized();
        let t = pitracer_front_wall()
            .hit(origin, direction)
            .expect("must hit");
        let expected = [aim.x - origin.x, aim.y - origin.y, aim.z - origin.z]
            .iter()
            .map(|d| f64::from(*d) * f64::from(*d))
            .sum::<f64>()
            .sqrt();
        assert_close(t, expected, 1e-6);
    }

    /// An origin lying exactly in the quad's plane must never hit it: the
    /// relative guard eats the t ~ 0 root for every non-parallel direction
    /// (parallel ones die on denom == 0). This is the scene.yml camera, which
    /// sits exactly on the front wall's plane; its primary rays must survive.
    #[test]
    fn hit_guard_rejects_on_plane_origin() {
        // on the pitracer floor plane (y = 0), interior of the quad
        let quad = pitracer_floor();
        let origin = Vector::from((50f32, 0f32, 100f32));
        for direction in [
            Vector::from((0f32, 1f32, 0f32)),
            Vector::from((0f32, -1f32, 0f32)),
            Vector::from((1f32, 1f32, 0.3f32)),
            Vector::from((-0.2f32, -1f32, 0.5f32)),
            Vector::from((1f32, 0.01f32, 0f32)), // shallow but not parallel
        ] {
            assert_eq!(quad.hit(origin, direction.normalized()), None);
        }

        // the scene.yml front wall with the camera origin exactly on it
        let wall = quad_at(
            (-13f32, -13f32, -10f32),
            (26f32, 0f32, 0f32),
            (0f32, 26f32, 0f32),
        );
        let camera = Vector::from((0f32, 0f32, -10f32));
        for direction in [
            Vector::from((0f32, 0f32, 1f32)),
            Vector::from((0.3f32, 0.2f32, 1f32)),
            Vector::from((-0.4f32, 0.1f32, -1f32)),
        ] {
            assert_eq!(wall.hit(camera, direction.normalized()), None);
        }
    }

    /// A ray spawned via Hit::spawn cannot re-intersect the plane it leaves at
    /// any angle: the origin sits strictly on the side the ray travels, so it
    /// never crosses back. This is the planar (and even stronger) version of
    /// the sphere's convexity argument.
    #[test]
    fn spawn_clears_own_surface() {
        let mut rng = rng(seeds()[1]);
        for quad in [
            pitracer_floor(),
            pitracer_front_wall(),
            quad_at(
                (1f32, 2f32, 3f32),
                (0.5f32, 0f32, 0f32),
                (0f32, 0f32, 0.5f32),
            ), // small
        ] {
            let aim = quad.position + 0.3f32 * quad.u + 0.6f32 * quad.v;
            let origin = aim + 40f32 * quad.normal + 0.1f32 * quad.u;
            let direction = (aim - origin).normalized();
            let t = quad.hit(origin, direction).expect("must hit");
            let hit = GeometryPrimitive::<dyn Rng>::to_hit(&quad, origin, direction, t);

            let mut directions = vec![hit.normal];
            for angle in [30f64, 60f64, 85f64, 89.9f64, 89.99f64] {
                directions.push(at_angle(hit.normal, angle.to_radians()));
            }
            for _ in 0..200 {
                let w = UnitVector::random(&mut rng);
                directions.push(crate::select(w.dot(hit.normal) >= 0f32, w, -w));
            }

            for scattered in directions {
                let spawned = quad.hit(hit.spawn(scattered), scattered);
                assert!(
                    spawned.is_none(),
                    "self re-hit on quad at {:?}: t = {spawned:?}, dir·n = {}",
                    quad.position,
                    scattered.dot(hit.normal),
                );
            }
        }
    }

    /// The offset reported by to_hit must bound the hit point's true (f64)
    /// distance from the plane (the invariant spawn relies on) while staying
    /// far below the quad's extent.
    #[test]
    fn to_hit_offset_bounds_placement_noise() {
        let mut rng = rng(seeds()[2]);
        for quad in [
            pitracer_floor(),
            pitracer_front_wall(),
            quad_at(
                (1f32, 2f32, 3f32),
                (0.5f32, 0f32, 0f32),
                (0f32, 0f32, 0.5f32),
            ),
        ] {
            // f64 plane basis recomputed from the f32 edges
            let (u, v, q) = (quad.u, quad.v, quad.position);
            let n64 = [
                f64::from(u.y) * f64::from(v.z) - f64::from(u.z) * f64::from(v.y),
                f64::from(u.z) * f64::from(v.x) - f64::from(u.x) * f64::from(v.z),
                f64::from(u.x) * f64::from(v.y) - f64::from(u.y) * f64::from(v.x),
            ];
            let n64_norm = n64.iter().map(|c| c * c).sum::<f64>().sqrt();

            let extent = u.norm().min(v.norm());
            for _ in 0..50 {
                let aim = q
                    + (0.05f32 + 0.9f32 * rng.random::<f32>()) * u
                    + (0.05f32 + 0.9f32 * rng.random::<f32>()) * v;
                let side = crate::select(rng.random::<f32>() < 0.5f32, 1f32, -1f32);
                let origin = aim
                    + (10f32 + 30f32 * rng.random::<f32>()) * side * quad.normal
                    + (rng.random::<f32>() - 0.5f32) * u
                    + (rng.random::<f32>() - 0.5f32) * v;
                let direction = (aim - origin).normalized();
                let Some(t) = quad.hit(origin, direction) else {
                    continue; // lateral jitter may push the aim off the quad
                };
                let hit = GeometryPrimitive::<dyn Rng>::to_hit(&quad, origin, direction, t);

                assert!(hit.offset > 0f32);
                assert!(
                    hit.offset < 1e-4f32 * extent,
                    "offset {} not small vs extent {extent}",
                    hit.offset
                );

                let err = [
                    (f64::from(hit.point.x) - f64::from(q.x)) * n64[0],
                    (f64::from(hit.point.y) - f64::from(q.y)) * n64[1],
                    (f64::from(hit.point.z) - f64::from(q.z)) * n64[2],
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

    /// Every sampled direction re-hits the quad with a positive pdf, from
    /// origins on both sides; directions past the edges, away from the plane,
    /// or parallel to it score 0.
    #[test]
    fn target_random_hits_quad() {
        let quad = quad_at((-1f32, -2f32, 5f32), (2f32, 0f32, 0f32), (0f32, 4f32, 0f32));
        for seed in seeds() {
            let mut rng = rng(seed);
            for origin in [
                Vector::default(),
                Vector::from((3f32, 1f32, -2f32)),
                Vector::from((0.5f32, -0.5f32, 8f32)), // other side of the plane
            ] {
                for _ in 0..1000 {
                    let direction = quad.target_random(origin, &mut rng);
                    assert!(
                        quad.hit(origin, direction).is_some(),
                        "sample misses the quad from {origin:?}: {direction:?}"
                    );
                    assert!(quad.target_pdf(origin, direction) > 0f32);
                }
            }
        }

        let origin = Vector::default();
        // past the +u edge (plane hit at x = 1.5 > 1)
        let miss = (Vector::from((1.5f32, 0f32, 5f32)) - origin).normalized();
        assert_eq!(quad.target_pdf(origin, miss), 0f32);
        // away from the plane
        assert_eq!(quad.target_pdf(origin, -UnitVector::Z), 0f32);
        // parallel to the plane
        assert_eq!(quad.target_pdf(origin, UnitVector::X), 0f32);
    }

    /// Uniform-area sampling in solid angle: p(w) = t^2 / (A * |cos|). For
    /// the 2x2 quad at distance 1 (A = 4): head-on p = 1/4; toward
    /// (0.5, 0, 1), t^2 = 1.25 and cos = 1/sqrt(1.25), so p = 1.25^1.5 / 4.
    /// Two-sided: the same values hold approaching from behind.
    #[test]
    fn target_pdf_closed_form() {
        let quad = quad_at((-1f32, -1f32, 1f32), (2f32, 0f32, 0f32), (0f32, 2f32, 0f32));
        let origin = Vector::default();

        assert_close(quad.target_pdf(origin, UnitVector::Z), 0.25f32, 1e-6);
        assert_close(
            quad.target_pdf(origin, Vector::from((0.5f32, 0f32, 1f32)).normalized()),
            1.25f64.powf(1.5f64) / 4f64,
            1e-6,
        );
        assert_eq!(
            quad.target_pdf(origin, Vector::from((1.5f32, 0f32, 1f32)).normalized()),
            0f32
        );

        // from the other side of the plane
        let behind = Vector::from((0f32, 0f32, 2f32));
        assert_close(quad.target_pdf(behind, -UnitVector::Z), 0.25f32, 1e-6);
    }

    /// target_pdf must integrate to 1 over the sphere of directions (the quad
    /// subtends 4*atan(1/sqrt(3)) ~ 2.09 sr here: plenty of MC power at 1e6).
    #[test]
    fn target_pdf_normalizes() {
        let quad = quad_at((-1f32, -1f32, 1f32), (2f32, 0f32, 0f32), (0f32, 2f32, 0f32));
        let origin = Vector::default();

        let mut rng = rng(seeds()[1]);
        let (integral, se) = sphere_integral(1_000_000, &mut rng, |w| quad.target_pdf(origin, w));
        assert_stat_close(integral, se, 1f32, 4f32, 0f32);
    }

    /// target_random must be distributed according to target_pdf. Probe-based
    /// consistency only: `chi2_directions` assumes a phi-symmetric target
    /// around the axis, which no quad is (even a square is only 4-fold
    /// symmetric); the probe set + normalization + every-sample-hits pin the
    /// sampler/pdf pair without that assumption.
    #[test]
    fn target_sampler_matches_target_pdf() {
        let quad = quad_at((-1f32, -1f32, 1f32), (2f32, 0f32, 0f32), (0f32, 2f32, 0f32));
        let origin = Vector::default();

        let mut rng = rng(seeds()[0]);
        assert_sampler_matches_pdf(
            100_000,
            1_000_000,
            &mut rng,
            UnitVector::Z,
            |rng| quad.target_random(origin, rng),
            |w| quad.target_pdf(origin, w),
        );
    }

    /// Degenerate quads (Default, parallel edges, zero edge) are inert: no
    /// hits, zero pdf, and a finite placeholder sampling direction.
    #[test]
    fn degenerate_quad_inert() {
        let mut rng = rng(seeds()[0]);
        for quad in [
            Quad::default(),
            quad_at((1f32, 2f32, 3f32), (1f32, 1f32, 0f32), (2f32, 2f32, 0f32)),
            quad_at((1f32, 2f32, 3f32), (0f32, 0f32, 0f32), (0f32, 4f32, 0f32)),
        ] {
            for _ in 0..10 {
                let origin = 5f32 * Vector::from(UnitVector::random(&mut rng));
                let direction = UnitVector::random(&mut rng);
                assert_eq!(quad.hit(origin, direction), None);
                assert_eq!(quad.target_pdf(origin, direction), 0f32);
                let sample = quad.target_random(origin, &mut rng);
                assert!(sample.x.is_finite() && sample.y.is_finite() && sample.z.is_finite());
            }
        }
    }
}
