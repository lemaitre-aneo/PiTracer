use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{GeometryPrimitive, Hit, Onb, UnitVector, Vector, select};

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sphere {
    pub radius: f32,
    pub position: Vector,
}

impl<R: crate::Rng + ?Sized> GeometryPrimitive<R> for Sphere {
    fn hit(&self, origin: Vector, direction: UnitVector, _rng: &mut R) -> Option<f32> {
        self.hit(origin, direction)
    }

    fn to_hit(&self, origin: Vector, direction: UnitVector, distance: f32) -> Hit {
        let point = origin + distance * direction;
        let outward_normal = (point - self.position).normalized();
        // The face is decided by the crossing direction at the hit point:
        // robust when the ray originates on the surface itself, where an
        // origin-inside-or-outside test is at the mercy of rounding.
        let front = direction.dot(outward_normal) < 0f32;
        let normal = select(front, outward_normal, -outward_normal);

        // No reprojection onto the sphere: its arithmetic rounds at the same
        // ulp(radius + |position|) scale as the error it would correct, so it
        // cannot control which side of the surface the point lands on (still
        // ~1e-2 off at r = 1e5). Instead, report the placement noise bound;
        // rays spawned `offset` off the surface clear it robustly.
        let offset = crate::SURFACE_OFFSET_REL * (self.radius + self.position.norm());

        Hit {
            point,
            normal,
            front,
            u: point.x,
            v: point.y,
            distance,
            offset,
        }
    }

    fn target_random(&self, origin: Vector, rng: &mut R) -> UnitVector {
        let direction = self.position - origin;
        let sint2 = self.radius * self.radius / direction.norm2();

        if sint2 >= 1f32 {
            return UnitVector::random(rng);
        }

        let base = Onb::new(direction.normalized());

        let r1 = rng.random::<f32>();
        let r2 = rng.random::<f32>();
        // 1 - z is uniform on [0, 1 - cos(theta_max)]; keeping it as its own
        // variable sidesteps the cancellations in z - 1 and 1 - z^2
        // (= u*(2 - u)) that erased small cones in f32.
        let u = r2 * Self::one_minus_cos_max(sint2);
        let z = 1f32 - u;
        let sin_theta = (u * (2f32 - u)).sqrt();

        let phi = 2f32 * std::f32::consts::PI * r1;
        let x = phi.cos() * sin_theta;
        let y = phi.sin() * sin_theta;

        base.transform(Vector { x, y, z }).normalized()
    }
    fn target_pdf(&self, origin: Vector, scattered: UnitVector) -> f32 {
        self.target_pdf(origin, scattered)
    }
}

impl Sphere {
    /// 1 - cos(theta_max) of the tangent cone, from sin^2(theta_max) =
    /// r^2/d^2. The naive `1 - sqrt(1 - sint2)` cancels in f32 for small
    /// angular radii (-7% at r/d ~ 1e-3, 0 below ~1.7e-4, where the pdf
    /// explodes to +inf and the sampler collapses onto the axis); this
    /// algebraically equal form keeps full relative precision at every scale.
    fn one_minus_cos_max(sint2: f32) -> f32 {
        sint2 / (1f32 + (1f32 - sint2).sqrt())
    }

    fn hit(&self, origin: Vector, direction: UnitVector) -> Option<f32> {
        let oc = self.position - origin;
        let a = direction.norm2();
        let h = direction.dot(oc);
        let oc2 = oc.norm2();
        let r2 = self.radius * self.radius;
        let c = oc2 - r2;

        // Vector-rejection discriminant (PBRT / Ray Tracing Gems): h^2 - a*c
        // cancels catastrophically when r << |oc| (a distant light is
        // randomly missed) and near tangency; r^2 - |oc_perp|^2 compares like
        // against like.
        let perp = oc - (h / a) * direction;
        let delta = a * (r2 - perp.norm2());
        if delta < 0f32 {
            return None;
        }
        let sqrtd = delta.sqrt();

        // Citardauq pairing: the root q/a adds same-sign quantities, and c/q
        // recovers the other from the exact product t0*t1 = c/a, avoiding the
        // (h - sqrt(delta)) cancellation (t-error 0.40 at r = 1e6 with the
        // naive form).
        let q = h + sqrtd.copysign(h);
        let (near, far) = if h >= 0f32 {
            (c / q, q / a)
        } else {
            (q / a, c / q)
        };

        // Smallest root beyond the self-intersection guard: a ray leaving the
        // surface into the sphere must report the far root (the chord), and a
        // sphere entirely behind the ray must report nothing. The guard is
        // relative (see HIT_EPSILON_REL); origins spawned via Hit::spawn sit
        // off the surface and don't rely on it.
        let guard2 = crate::HIT_EPSILON_REL * crate::HIT_EPSILON_REL * oc2;
        if near > 0f32 && near * near > guard2 {
            return Some(near);
        }
        if far > 0f32 && far * far > guard2 {
            return Some(far);
        }
        None
    }

    fn target_pdf(&self, origin: Vector, scattered: UnitVector) -> f32 {
        let direction = self.position - origin;
        let sint2 = self.radius * self.radius / direction.norm2();

        if sint2 >= 1f32 {
            return 1f32 / (4f32 * std::f32::consts::PI);
        }

        if self.hit(origin, scattered).is_none() {
            return 0f32;
        }

        1f32 / (2f32 * std::f32::consts::PI * Self::one_minus_cos_max(sint2))
    }
}

#[cfg(test)]
mod test {
    use rand::{RngCore, SeedableRng};

    use super::*;
    #[test]
    fn random() {
        let sphere = Sphere {
            radius: 1f32,
            position: Vector {
                x: 1f32,
                y: 2f32,
                z: 3f32,
            },
        };
        let origin = Vector::<f32>::default();

        let mut rng = rand::rngs::SmallRng::from_seed(Default::default());

        for _ in 0..100 {
            let direction = sphere.target_random(origin, &mut rng);
            assert!(sphere.hit(origin, direction).is_some());
            let pdf = sphere.target_pdf(origin, direction);
            eprintln!("{direction:?} -> {pdf:?}");
            assert!(pdf > 0f32);
        }

        assert_eq!(sphere.target_pdf(origin, UnitVector::X), 0f32);
        assert_eq!(sphere.target_pdf(origin, UnitVector::Y), 0f32);
        assert_eq!(sphere.target_pdf(origin, UnitVector::Z), 0f32);
    }

    #[test]
    fn hit() {
        let sphere = Sphere {
            radius: 1f32,
            position: Vector {
                x: 1f32,
                y: 2f32,
                z: 3f32,
            },
        };

        let origin = Vector {
            x: 0f32,
            y: 1f32,
            z: 2f32,
        };
        let direction = Vector::splat(1f32).normalized();

        let distance = sphere.hit(origin, direction).expect("Must hit");
        let hit = GeometryPrimitive::<dyn RngCore>::to_hit(&sphere, origin, direction, distance);

        eprintln!("{hit:?}");
    }

    use crate::test_utils::{
        Stats, assert_close, assert_sampler_matches_pdf, assert_stat_close, assert_vec_close,
        at_angle, chi2_directions, rng, sampler_mean, seeds, sphere_integral,
    };

    fn sphere_at(position: (f32, f32, f32), radius: f32) -> Sphere {
        Sphere {
            radius,
            position: position.into(),
        }
    }

    /// Analytic ray-sphere ground truth: with proj = dot(C - O, d) and impact
    /// parameter b (b^2 = |C - O|^2 - proj^2), a forward hit exists iff
    /// b <= r, at t = proj - sqrt(r^2 - b^2) from outside and
    /// t = proj + sqrt(r^2 - b^2) from inside.
    #[test]
    fn hit_distances_analytic() {
        let origin = Vector::default();
        let z = UnitVector::Z;

        // head-on from outside: t = distance - radius
        let t = sphere_at((0f32, 0f32, 5f32), 2f32)
            .hit(origin, z)
            .expect("must hit");
        assert_close(t, 3f32, 1e-6);

        // offset b = 1, proj = 5: t = 5 - sqrt(3)
        let t = sphere_at((0f32, 1f32, 5f32), 2f32)
            .hit(origin, z)
            .expect("must hit");
        assert_close(t, 5f32 - 3f32.sqrt(), 1e-6);

        // from the center: t = r in any direction
        let sphere = sphere_at((1f32, 2f32, 3f32), 2f32);
        for direction in [
            UnitVector::X,
            UnitVector::Y,
            Vector::splat(-1f32).normalized(),
        ] {
            let t = sphere.hit(sphere.position, direction).expect("must hit");
            assert_close(t, 2f32, 1e-6);
        }

        // from inside: exit at t = proj + sqrt(r^2 - b^2)
        let t = sphere_at((0f32, 0f32, 5f32), 2f32)
            .hit(Vector::from((0f32, 0f32, 4f32)), z)
            .expect("must hit");
        assert_close(t, 3f32, 1e-6);

        // miss: b = 3 > r = 2
        assert_eq!(sphere_at((0f32, 3f32, 5f32), 2f32).hit(origin, z), None);

        // sphere behind the ray: no positive phantom distance may be reported
        let t = sphere_at((0f32, 0f32, -5f32), 2f32).hit(origin, z);
        assert!(t.is_none_or(|t| t <= 0f32), "phantom forward hit at {t:?}");

        // near-tangent: inside graze hits, outside graze misses
        let b = 2f32 * (1f32 - 1e-6);
        let t = sphere_at((0f32, b, 5f32), 2f32)
            .hit(origin, z)
            .expect("must hit");
        assert!(t > 0f32);
        let b = 2f32 * (1f32 + 1e-6);
        assert_eq!(sphere_at((0f32, b, 5f32), 2f32).hit(origin, z), None);
    }

    /// Pins the hit convention every material relies on: the normal always
    /// opposes the incident ray; `front` means the origin was outside.
    #[test]
    fn to_hit_normal_orientation() {
        let sphere = sphere_at((0f32, 0f32, 5f32), 2f32);
        let direction = UnitVector::Z;

        // from outside
        let origin = Vector::default();
        let t = sphere.hit(origin, direction).expect("must hit");
        let hit = GeometryPrimitive::<dyn RngCore>::to_hit(&sphere, origin, direction, t);
        assert!(hit.front);
        assert_vec_close(hit.point, Vector::from((0f32, 0f32, 3f32)), 1e-6);
        assert_vec_close(
            Vector::from(hit.normal),
            Vector::from((0f32, 0f32, -1f32)),
            1e-6,
        );
        assert!(hit.normal.dot(direction) < 0f32);
        assert_close((hit.point - sphere.position).norm(), 2f32, 1e-5);
        assert_close(hit.distance, t, 1e-6);

        // from inside: back face, normal flipped inward, still opposing
        let origin = Vector::from((0f32, 0f32, 4f32));
        let t = sphere.hit(origin, direction).expect("must hit");
        let hit = GeometryPrimitive::<dyn RngCore>::to_hit(&sphere, origin, direction, t);
        assert!(!hit.front);
        assert_vec_close(hit.point, Vector::from((0f32, 0f32, 7f32)), 1e-6);
        assert_vec_close(
            Vector::from(hit.normal),
            Vector::from((0f32, 0f32, -1f32)),
            1e-6,
        );
        assert!(hit.normal.dot(direction) < 0f32);
    }

    /// Entering distance vs ground truth at wall scale: the naive
    /// (h - sqrt(delta))/a subtraction lost ~0.4 units at r = 1e6; the
    /// Citardauq pairing must stay within f32 rounding of the true root.
    #[test]
    fn hit_distance_accurate_at_wall_scale() {
        // head-on: r = 1e6, origin 100 above the surface; exact answer 100
        let t = sphere_at((0f32, 0f32, 0f32), 1e6f32)
            .hit(Vector::from((0f32, 0f32, 1_000_100f32)), -UnitVector::Z)
            .expect("must hit");
        assert_close(t, 100f32, 1.5e-3);

        // 45 degrees onto an r = 1e5 floor; f64 reference root
        let sphere = sphere_at((0f32, -1e5f32, 0f32), 1e5f32);
        let origin = Vector::from((0f32, 30f32, 0f32));
        let direction = Vector::from((1f32, -1f32, 0f32)).normalized();
        let t = sphere.hit(origin, direction).expect("must hit");

        let dy = f64::from(Vector::from(direction).y);
        let h = dy * -100_030f64; // dot(d, C - O), x and z terms are 0
        let c = 100_030f64 * 100_030f64 - 1e10f64;
        let expected = c / (h + (h * h - c).sqrt());
        assert_close(t, expected, 5e-4);
    }

    /// The naive discriminant h^2 - a*c cancels to rounding noise for a
    /// distant small sphere (r << |oc|): hits get missed or land a full
    /// radius off. The rejection form must find them at the exact distance.
    #[test]
    fn hit_finds_far_small_spheres_exactly() {
        for d in [1e4f32, 1e6f32] {
            let t = sphere_at((0f32, 0f32, d), 1f32)
                .hit(Vector::default(), UnitVector::Z)
                .expect("must hit");
            assert_close(t, d - 1f32, 1e-6);
        }

        // off-axis: impact parameter 0.6, entering root from the f64 quadratic
        let t = sphere_at((0.6f32, 0f32, 1e4f32), 1f32)
            .hit(Vector::default(), UnitVector::Z)
            .expect("must hit");
        let c = 0.6f64 * 0.6f64 + 1e8f64 - 1f64;
        let expected = c / (1e4f64 + (1e8f64 - c).sqrt());
        assert_close(t, expected, 1e-6);
    }

    /// Transmission through a small sphere: spawned just inside the entry
    /// point, the ray must find the exit chord. The old absolute 7e-2 guard
    /// erased it for r <= 0.035 (the chord itself is <= 0.07), killing glass
    /// beads entirely.
    #[test]
    fn small_sphere_exit_hit_preserved() {
        let sphere = sphere_at((73f32, 46.5f32, 88f32), 0.035f32);
        let origin = Vector::from((70f32, 46.5f32, 88f32));
        let direction = UnitVector::X;

        let t = sphere.hit(origin, direction).expect("must enter");
        let hit = GeometryPrimitive::<dyn RngCore>::to_hit(&sphere, origin, direction, t);
        assert!(hit.front);

        // straight-through transmission: spawn on the inside
        let inside = hit.spawn(direction);
        let exit = sphere.hit(inside, direction).expect("exit hit erased");
        assert_close(exit, 2f32 * 0.035f32 - hit.offset, 1e-4);
    }

    /// A ray spawned via Hit::spawn cannot re-intersect the surface it
    /// leaves, at any scale and any outward angle, grazing included. This
    /// invariant replaces the absolute guard: grazing phantoms reach
    /// sqrt(2*r*delta) (about 45 at wall scale for delta = 1e-2), beyond any
    /// usable threshold, but an origin strictly outside a convex surface
    /// cannot re-hit it geometrically.
    #[test]
    fn spawn_clears_own_surface() {
        let mut rng = rng(seeds()[1]);
        for (radius, position) in [
            (16.5f32, (40f32, 16.5f32, 47f32)),  // pitracer ball
            (1e5f32, (50f32, -1e5f32, 81.6f32)), // pitracer wall
            (0.035f32, (73f32, 46.5f32, 88f32)), // glass bead
        ] {
            let sphere = sphere_at(position, radius);
            let axis = Vector::from((1f32, 2f32, 3f32)).normalized();
            let origin = sphere.position + (1.5f32 * radius + 100f32) * axis;
            let t = sphere.hit(origin, -axis).expect("must hit");
            let hit = GeometryPrimitive::<dyn RngCore>::to_hit(&sphere, origin, -axis, t);

            let mut directions = vec![hit.normal];
            for angle in [30f64, 60f64, 85f64, 89.9f64, 89.99f64] {
                directions.push(at_angle(hit.normal, angle.to_radians()));
            }
            for _ in 0..200 {
                let w = UnitVector::random(&mut rng);
                directions.push(crate::select(w.dot(hit.normal) >= 0f32, w, -w));
            }

            for scattered in directions {
                let spawned = sphere.hit(hit.spawn(scattered), scattered);
                assert!(
                    spawned.is_none(),
                    "self re-hit at r = {radius}: t = {spawned:?}, dir·n = {}",
                    scattered.dot(hit.normal),
                );
            }
        }
    }

    /// The offset reported by to_hit must bound the point's true distance
    /// from the surface (measured in f64; that is the invariant spawn relies
    /// on) while staying far below the geometry itself.
    #[test]
    fn to_hit_offset_bounds_placement_noise() {
        let mut rng = rng(seeds()[2]);
        for (radius, position) in [
            (16.5f32, (40f32, 16.5f32, 47f32)),
            (1e5f32, (50f32, -1e5f32, 81.6f32)),
            (0.035f32, (73f32, 46.5f32, 88f32)),
        ] {
            let sphere = sphere_at(position, radius);
            for _ in 0..50 {
                let towards = UnitVector::random(&mut rng);
                let origin = sphere.position + (2f32 * radius + 50f32) * towards;
                let aim = sphere.position + (0.3f32 * radius) * UnitVector::random(&mut rng);
                let direction = (aim - origin).normalized();
                let t = sphere.hit(origin, direction).expect("must hit");
                let hit = GeometryPrimitive::<dyn RngCore>::to_hit(&sphere, origin, direction, t);

                assert!(hit.offset > 0f32);
                assert!(
                    hit.offset < 0.02f32 * radius,
                    "offset {} not small vs r = {radius}",
                    hit.offset
                );

                let to_center = [
                    f64::from(hit.point.x) - f64::from(sphere.position.x),
                    f64::from(hit.point.y) - f64::from(sphere.position.y),
                    f64::from(hit.point.z) - f64::from(sphere.position.z),
                ];
                let err = to_center.iter().map(|d| d * d).sum::<f64>().sqrt() - f64::from(radius);
                assert!(
                    err.abs() < f64::from(hit.offset),
                    "placement noise {err:e} exceeds offset {:e} at r = {radius}",
                    hit.offset,
                );
            }
        }
    }

    /// Sampling the sphere by solid angle from outside is uniform over the
    /// cone with cos(theta_max) = sqrt(1 - r^2/d^2):
    /// pdf = 1 / (2*pi*(1 - cos(theta_max))) inside the cone, 0 outside.
    #[test]
    fn target_pdf_closed_form_in_cone() {
        let sphere = sphere_at((1f32, 2f32, 3f32), 1f32);
        let origin = Vector::default();
        let axis = sphere.position.normalized();
        let cos_max = (13f64 / 14f64).sqrt();
        let theta_max = cos_max.acos();
        let expected = 1f64 / (2f64 * std::f64::consts::PI * (1f64 - cos_max));

        assert_close(sphere.target_pdf(origin, axis), expected, 1e-6);
        assert_close(
            sphere.target_pdf(origin, at_angle(axis, 0.5f64 * theta_max)),
            expected,
            1e-5,
        );
        assert_eq!(
            sphere.target_pdf(origin, at_angle(axis, 1.01f64 * theta_max)),
            0f32
        );
    }

    /// Uniform-in-cone means z := dot(w, axis) is uniform on
    /// [cos(theta_max), 1]: E[z] = (1 + c)/2, E[z^2] = (1 + c + c^2)/3;
    /// every sample hits the sphere.
    #[test]
    fn target_random_within_cone_moments() {
        let sphere = sphere_at((1f32, 2f32, 3f32), 1f32);
        let origin = Vector::default();
        let axis = sphere.position.normalized();
        let c = (13f64 / 14f64).sqrt();

        for seed in seeds() {
            let mut rng = rng(seed);
            let mut stats = [Stats::new(); 2];
            for _ in 0..100_000 {
                let w = sphere.target_random(origin, &mut rng);
                let z = w.dot(axis);
                assert!(f64::from(z) >= c - 1e-5, "sample outside the cone: z = {z}");
                assert!(
                    sphere.hit(origin, w).is_some(),
                    "sample misses the sphere: {w:?}"
                );
                stats[0].push(z);
                stats[1].push(z * z);
            }
            assert_stat_close(
                stats[0].mean(),
                stats[0].se(),
                (1f64 + c) / 2f64,
                4f32,
                0f32,
            );
            assert_stat_close(
                stats[1].mean(),
                stats[1].se(),
                (1f64 + c + c * c) / 3f64,
                4f32,
                0f32,
            );
        }
    }

    /// target_random must be distributed according to target_pdf (wide cone
    /// so the goodness-of-fit bins fill: r = 1, d = 1.25 => c = 0.6 exactly).
    #[test]
    fn target_sampler_matches_target_pdf() {
        let sphere = sphere_at((0f32, 0f32, 1.25f32), 1f32);
        let origin = Vector::default();
        let axis = UnitVector::Z;
        let c = 0.6f64;

        let mut rng = rng(seeds()[0]);
        assert_sampler_matches_pdf(
            100_000,
            1_000_000,
            &mut rng,
            axis,
            |rng| sphere.target_random(origin, rng),
            |w| sphere.target_pdf(origin, w),
        );

        chi2_directions(
            50_000,
            &mut rng,
            axis,
            c,
            10,
            10,
            |rng| sphere.target_random(origin, rng),
            |z0, z1| (z1 - z0) / (1f64 - c),
        );
    }

    /// target_pdf must integrate to 1 over the sphere of directions.
    #[test]
    fn target_pdf_normalizes() {
        let sphere = sphere_at((0f32, 0f32, 1.5f32), 1f32);
        let origin = Vector::default();

        let mut rng = rng(seeds()[1]);
        let (integral, se) = sphere_integral(1_000_000, &mut rng, |w| sphere.target_pdf(origin, w));
        assert_stat_close(integral, se, 1f32, 4f32, 0f32);
    }

    /// target_pdf must track the f64 ground truth at every angular radius:
    /// the naive `1 - sqrt(1 - sint2)` denominator cancels in f32 (-7% at
    /// r/d ~ 1e-3, +inf below ~1.7e-4, where the light silently vanished
    /// from diffuse lighting).
    #[test]
    fn target_pdf_exact_at_small_angular_radius() {
        for d in [2f32, 10f32, 1e3f32, 1e4f32, 1e6f32] {
            let sphere = sphere_at((0f32, 0f32, d), 1f32);
            let pdf = sphere.target_pdf(Vector::default(), UnitVector::Z);

            // f64 reference through the same stable algebra: the naive f64
            // form loses digits itself by d = 1e6.
            let sint2 = 1f64 / (f64::from(d) * f64::from(d));
            let omc = sint2 / (1f64 + (1f64 - sint2).sqrt());
            let expected = 1f64 / (2f64 * std::f64::consts::PI * omc);

            assert_close(pdf, expected, 1e-5);
        }
    }

    /// Below the old cancellation threshold (angular radius 1e-4) the sampler
    /// must still spread over the tangent cone instead of collapsing onto the
    /// axis. At this scale z rounds to exactly 1.0 in f32, so the spread is
    /// measured on the transverse components, which keep full relative
    /// precision: sin^2 = (1 - z)(1 + z) with 1 - z uniform on [0, omc] gives
    /// E[sin^2] = omc*(1 - omc/3) and max sin^2 = omc*(2 - omc)
    /// = sin^2(theta_max) exactly.
    #[test]
    fn target_random_small_cone_spread() {
        let sphere = sphere_at((0f32, 0f32, 1e4f32), 1f32);
        let origin = Vector::default();
        let sint2 = 1e-8f64;
        let omc = sint2 / (1f64 + (1f64 - sint2).sqrt());

        let mut rng = rng(seeds()[0]);
        let mut stats = Stats::new();
        for _ in 0..100_000 {
            let direction = sphere.target_random(origin, &mut rng);
            let w = Vector::from(direction);
            let t2 = f64::from(w.x) * f64::from(w.x) + f64::from(w.y) * f64::from(w.y);
            assert!(
                t2 <= sint2 * (1f64 + 1e-5),
                "sample outside the cone: sin²θ = {t2:e}"
            );
            // clearly-interior samples must reach the sphere (guards a wrong
            // aim axis or hemisphere; edge samples may graze-miss by rounding)
            if t2 < sint2 * 0.999 {
                assert!(
                    sphere.hit(origin, direction).is_some(),
                    "interior sample misses the sphere: {w:?}"
                );
            }
            stats.push(t2);
        }
        assert_stat_close(
            stats.mean(),
            stats.se(),
            omc * (1f64 - omc / 3f64),
            4f32,
            0f32,
        );
    }

    /// From inside the sphere every direction hits it: the only consistent
    /// strategy is uniform over the full sphere, pdf = 1/(4*pi) everywhere.
    #[test]
    fn target_inside_sphere_uniform() {
        let sphere = sphere_at((1f32, 0f32, 0f32), 2f32);
        let origin = Vector::default();
        let expected = 1f32 / (4f32 * std::f32::consts::PI);

        let mut rng = rng(seeds()[2]);
        for _ in 0..100 {
            assert_close(
                sphere.target_pdf(origin, UnitVector::random(&mut rng)),
                expected,
                1e-6,
            );
        }

        let (mean, se) = sampler_mean(
            100_000,
            &mut rng,
            |rng| sphere.target_random(origin, rng),
            |w| w.dot(UnitVector::X),
        );
        assert_stat_close(mean, se, 0f32, 4f32, 0f32);

        chi2_directions(
            50_000,
            &mut rng,
            UnitVector::Z,
            -1f64,
            10,
            10,
            |rng| sphere.target_random(origin, rng),
            |z0, z1| (z1 - z0) / 2.0,
        );
    }

    /// Heavy tier: cone sampler goodness-of-fit at 10x samples, finer grid.
    #[test]
    #[ignore = "heavy convergence tier"]
    fn heavy_chi2_target_sampler() {
        let sphere = sphere_at((0f32, 0f32, 1.25f32), 1f32);
        let origin = Vector::default();
        let c = 0.6f64;

        let mut rng = rng(seeds()[2]);
        chi2_directions(
            1_000_000,
            &mut rng,
            UnitVector::Z,
            c,
            20,
            20,
            |rng| sphere.target_random(origin, rng),
            |z0, z1| (z1 - z0) / (1f64 - c),
        );
    }
}
