use serde::{Deserialize, Serialize};

use crate::Onb;

mod definition {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct Empty {}

    #[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
    #[serde(untagged)]
    pub(super) enum Vector<T = f32> {
        #[default]
        Default,
        Empty(Empty),
        List(T, T, T),
        Map {
            x: T,
            y: T,
            z: T,
        },
    }

    impl<T> From<super::Vector<T>> for Vector<T> {
        fn from(value: super::Vector<T>) -> Self {
            Self::List(value.x, value.y, value.z)
        }
    }

    impl<T: Default> From<Vector<T>> for super::Vector<T> {
        fn from(value: Vector<T>) -> Self {
            match value {
                Vector::List(x, y, z) => Self { x, y, z },
                Vector::Map { x, y, z } => Self { x, y, z },
                _ => Self::default(),
            }
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(from = "definition::Vector<T>", into = "definition::Vector<T>")]
#[serde(bound = "for <'a> T: Default + Clone + Serialize + Deserialize<'a>")]
pub struct Vector<T = f32> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub type VectorVec = Vec<Vector>;

/// Direction with unit norm by construction (`Vector::normalized`, the axis
/// constants, or the exact-norm `TryFrom`); derefs to `Vector` for reads, so
/// `norm`/`norm2` short-circuit to 1.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct UnitVector<T = f32>(Vector<T>);

// Componentwise operator impls over every Vector/UnitVector/f32 pairing. The
// assign form is the primitive; the value forms are derived from it and
// always return a plain Vector (arithmetic does not preserve unit norm).
macro_rules! impl_op {
    ($op:ident($opassign:ident): $trait:ident($traitassign:ident): $assign:tt) => {
        impl_op!($opassign(Vector, f32): $traitassign: $assign);
        impl_op!($opassign(Vector, Vector): $traitassign: $assign {.});
        impl_op!($opassign(Vector, UnitVector): $traitassign: $assign {.0.});


        impl_op!($op(Vector, f32) -> Vector: $trait: $assign);
        impl_op!($op(Vector, Vector) -> Vector: $trait: $assign);
        impl_op!($op(Vector, UnitVector) -> Vector: $trait: $assign);
        impl_op!($op(UnitVector, f32) -> Vector: $trait: $assign);
        impl_op!($op(UnitVector, Vector) -> Vector: $trait: $assign);
        impl_op!($op(UnitVector, UnitVector) -> Vector: $trait: $assign);
        impl_op!($op(f32, Vector) -> Vector: $trait: $assign);
        impl_op!($op(f32, UnitVector) -> Vector: $trait: $assign);
    };

    ($op:ident($({$dereflhs:tt})? $lhs:ty, $({$derefrhs:tt})? $rhs:ty): $trait:ident : $assign:tt $({$($member:tt)*})?) => {
        impl std::ops::$trait<$rhs> for $lhs {
            fn $op(&mut self, rhs: $rhs) {
                $($dereflhs)? self.x $assign $($derefrhs)? rhs $($($member)* x)?;
                $($dereflhs)? self.y $assign $($derefrhs)? rhs $($($member)* y)?;
                $($dereflhs)? self.z $assign $($derefrhs)? rhs $($($member)* z)?;
            }
        }
    };

    ($op:ident($lhs:ty, $rhs:ty) -> $ret:ty: $trait:ident : $assign:tt ) => {
        impl std::ops::$trait<$rhs> for $lhs {
            type Output = $ret;

            fn $op(self, rhs: $rhs) -> Self::Output {
                let mut res = Self::Output::from(self);
                res $assign rhs;
                res
            }
        }
    };
}

impl_op!(add(add_assign) : Add(AddAssign) : +=);
impl_op!(sub(sub_assign) : Sub(SubAssign) : -=);
impl_op!(mul(mul_assign) : Mul(MulAssign) : *=);
impl_op!(div(div_assign) : Div(DivAssign) : /=);

impl std::ops::Neg for Vector {
    type Output = Vector;

    fn neg(self) -> Self::Output {
        Vector {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}
impl std::ops::Neg for UnitVector {
    type Output = UnitVector;

    fn neg(self) -> Self::Output {
        UnitVector(-self.0)
    }
}

impl From<f32> for Vector {
    fn from(value: f32) -> Self {
        Self {
            x: value,
            y: value,
            z: value,
        }
    }
}

impl From<(f32, f32, f32)> for Vector {
    fn from(value: (f32, f32, f32)) -> Self {
        Self {
            x: value.0,
            y: value.1,
            z: value.2,
        }
    }
}

impl From<UnitVector> for Vector {
    fn from(value: UnitVector) -> Self {
        value.0
    }
}

impl TryFrom<Vector> for UnitVector {
    type Error = Vector;

    fn try_from(value: Vector) -> Result<Self, Self::Error> {
        if value.norm2() == 1f32 {
            Ok(Self(value))
        } else {
            Err(value)
        }
    }
}

impl AsRef<Vector> for UnitVector {
    fn as_ref(&self) -> &Vector {
        &self.0
    }
}

impl std::ops::Deref for UnitVector {
    type Target = Vector;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::borrow::Borrow<Vector> for UnitVector {
    fn borrow(&self) -> &Vector {
        &self.0
    }
}

impl Default for UnitVector {
    fn default() -> Self {
        Self(Vector {
            x: 0f32,
            y: 1f32,
            z: 0f32,
        })
    }
}

impl Vector {
    pub fn splat(value: f32) -> Self {
        value.into()
    }
    pub fn sum(self) -> f32 {
        self.x + self.y + self.z
    }
    pub fn norm2(self) -> f32 {
        self.dot(self)
    }
    pub fn norm(self) -> f32 {
        self.norm2().sqrt()
    }
    pub fn normalized(self) -> UnitVector {
        let sum = self.dot(self);
        let normalization = 1f32 / sum.sqrt();
        UnitVector(self * normalization)
    }
    pub fn reflect(self, normal: UnitVector) -> Vector {
        let an = 2f32 * (self * normal).sum();
        self - an * normal
    }
    pub fn dot(self, rhs: impl std::borrow::Borrow<Vector>) -> f32 {
        let rhs = *rhs.borrow();
        (self * rhs).sum()
    }
    pub fn cross(self, rhs: impl std::borrow::Borrow<Vector>) -> Vector {
        let rhs = *rhs.borrow();
        Vector {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
}

impl UnitVector {
    pub const X: UnitVector = UnitVector(Vector {
        x: 1f32,
        y: 0f32,
        z: 0f32,
    });
    pub const Y: UnitVector = UnitVector(Vector {
        x: 0f32,
        y: 1f32,
        z: 0f32,
    });
    pub const Z: UnitVector = UnitVector(Vector {
        x: 0f32,
        y: 0f32,
        z: 1f32,
    });

    pub fn norm(self) -> f32 {
        1f32
    }
    pub fn norm2(self) -> f32 {
        1f32
    }
    pub fn normalized(self) -> UnitVector {
        self
    }
    pub fn reflect(self, normal: UnitVector) -> UnitVector {
        UnitVector(self.0.reflect(normal))
    }
    /// Uniform on the sphere: rejection-sample the unit ball (excluding tiny
    /// norms whose normalization would amplify component noise), normalize.
    pub fn random<R: rand::Rng + ?Sized>(rng: &mut R) -> UnitVector {
        loop {
            let v = Vector {
                x: rng.random(),
                y: rng.random(),
                z: rng.random(),
            } * 2f32
                - 1f32;
            let norm2 = v.norm2();
            if (0.001f32..=1f32).contains(&norm2) {
                return v.normalized();
            }
        }
    }
    pub fn random_hemisphere<R: rand::Rng + ?Sized>(normal: UnitVector, rng: &mut R) -> UnitVector {
        let vec = Self::random(rng);
        crate::select(vec.dot(normal) > 0f32, vec, -vec)
    }

    /// Cosine-weighted direction around local +z, p(w) = cos(theta)/pi: a
    /// uniform point on the unit disk (r = sqrt(r2)) lifted onto the
    /// hemisphere (z = sqrt(1 - r^2)).
    pub fn random_cosine_local<R: rand::Rng + ?Sized>(rng: &mut R) -> UnitVector {
        let r1 = rng.random::<f32>();
        let r2 = rng.random::<f32>();
        let r2_sqrt = r2.sqrt();

        let phi = 2f32 * std::f32::consts::PI * r1;
        let x = phi.cos() * r2_sqrt;
        let y = phi.sin() * r2_sqrt;
        let z = (1f32 - r2).sqrt();

        UnitVector(Vector { x, y, z })
    }

    /// [`random_cosine_local`](Self::random_cosine_local) rotated so local +z
    /// lands on `normal`.
    pub fn random_cosine<R: rand::Rng + ?Sized>(normal: UnitVector, rng: &mut R) -> UnitVector {
        UnitVector(Onb::new(normal).transform(UnitVector::random_cosine_local(rng).0))
    }
}

#[cfg(test)]
mod test {
    use rand::Rng as _;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::test_utils::{
        Stats, assert_close, assert_stat_close, assert_vec_close, chi2_directions, rng, seeds,
    };

    fn random_vector(rng: &mut SmallRng) -> Vector {
        Vector {
            x: rng.random::<f32>() * 10f32 - 5f32,
            y: rng.random::<f32>() * 10f32 - 5f32,
            z: rng.random::<f32>() * 10f32 - 5f32,
        }
    }

    /// Reflection off a plane with unit normal n is the Householder map
    /// r = v - 2*(v dot n)*n: an isometric involution that flips the normal
    /// component and preserves the tangential one.
    #[test]
    fn reflect_involution_and_norm() {
        let mut rng = rng(seeds()[0]);
        for _ in 0..1000 {
            let v = random_vector(&mut rng);
            let n = UnitVector::random(&mut rng);
            let r = v.reflect(n);

            assert_vec_close(r.reflect(n), v, 1e-5);
            assert_close(r.norm(), v.norm(), 1e-5);
            assert_close(r.dot(n), -v.dot(n), 1e-5);
            assert_vec_close(r - r.dot(n) * n, v - v.dot(n) * n, 1e-5);
        }

        assert_eq!(
            Vector::from((1f32, -1f32, 0f32)).reflect(UnitVector::Y),
            Vector::from((1f32, 1f32, 0f32)),
        );
        assert_eq!(
            Vector::from((-UnitVector::Y).reflect(UnitVector::Y)),
            Vector::from(UnitVector::Y)
        );
        // vector orthogonal to the normal is unchanged
        assert_eq!(
            Vector::from((1f32, 0f32, 0f32)).reflect(UnitVector::Y),
            Vector::from((1f32, 0f32, 0f32)),
        );
    }

    #[test]
    fn normalized_is_unit() {
        let mut rng = rng(seeds()[0]);
        for _ in 0..1000 {
            let v = random_vector(&mut rng);
            if v.norm() < 1e-6 {
                continue;
            }
            let u = v.normalized();
            assert_close(Vector::from(u).norm2(), 1f32, 1e-6);
            // same direction: dot(u, v) == |v|
            assert_close(u.dot(v), v.norm(), 1e-5);
        }
    }

    /// UnitVector::random must be uniform on the sphere: density 1/(4*pi),
    /// so E[w_i] = 0, E[w_i^2] = 1/3, E[w_i*w_j] = 0 (i != j).
    #[test]
    fn unit_random_is_uniform_sphere() {
        for seed in seeds() {
            let mut rng = rng(seed);
            let mut stats = [Stats::new(); 9];
            for _ in 0..100_000 {
                let w = UnitVector::random(&mut rng);
                for (stat, value) in stats.iter_mut().zip([
                    w.x,
                    w.y,
                    w.z,
                    w.x * w.x,
                    w.y * w.y,
                    w.z * w.z,
                    w.x * w.y,
                    w.y * w.z,
                    w.z * w.x,
                ]) {
                    stat.push(value);
                }
            }
            let expected = [
                0f32,
                0f32,
                0f32,
                1f32 / 3f32,
                1f32 / 3f32,
                1f32 / 3f32,
                0f32,
                0f32,
                0f32,
            ];
            for (stat, expected) in stats.iter().zip(expected) {
                assert_stat_close(stat.mean(), stat.se(), expected, 4f32, 0f32);
            }
        }

        // goodness-of-fit: z-marginal of the uniform sphere is uniform on [-1, 1]
        let mut rng = rng(seeds()[0]);
        chi2_directions(
            50_000,
            &mut rng,
            UnitVector::Z,
            -1f64,
            10,
            10,
            UnitVector::random,
            |z0, z1| (z1 - z0) / 2.0,
        );
    }

    #[test]
    fn unit_random_norm() {
        let mut rng = rng(seeds()[0]);
        for _ in 0..1000 {
            let w = UnitVector::random(&mut rng);
            assert_close(Vector::from(w).norm2(), 1f32, 1e-6);
        }
    }

    /// Cosine-weighted hemisphere sampling has density p(w) = z/pi (z >= 0),
    /// so E[z^k] = 2/(k+2): E[z] = 2/3, E[z^2] = 1/2, and by symmetry
    /// E[x] = E[y] = 0, E[x^2] = E[y^2] = (1 - E[z^2])/2 = 1/4.
    /// The z-slice [z0, z1] carries probability z1^2 - z0^2.
    #[test]
    fn random_cosine_local_matches_cosine_density() {
        for seed in seeds() {
            let mut rng = rng(seed);
            let mut stats = [Stats::new(); 5];
            for _ in 0..100_000 {
                let w = UnitVector::random_cosine_local(&mut rng);
                assert!(w.z >= 0f32, "cosine sample below the horizon: {w:?}");
                for (stat, value) in stats.iter_mut().zip([w.z, w.z * w.z, w.x, w.y, w.x * w.x]) {
                    stat.push(value);
                }
            }
            let expected = [2f32 / 3f32, 0.5f32, 0f32, 0f32, 0.25f32];
            for (stat, expected) in stats.iter().zip(expected) {
                assert_stat_close(stat.mean(), stat.se(), expected, 4f32, 0f32);
            }
        }

        let mut rng = rng(seeds()[0]);
        chi2_directions(
            50_000,
            &mut rng,
            UnitVector::Z,
            0f64,
            10,
            10,
            UnitVector::random_cosine_local,
            |z0, z1| z1 * z1 - z0 * z0,
        );
    }

    /// Same distribution around arbitrary (non-axis) normals, with
    /// z := dot(w, n). This closes the local-sample + ONB-transform loop end
    /// to end.
    #[test]
    fn random_cosine_respects_normal() {
        let normals = [
            Vector::from((1f32, 2f32, 3f32)).normalized(),
            Vector::from((-1f32, 0.1f32, 0.5f32)).normalized(),
            Vector::from((0.3f32, -0.9f32, 0.6f32)).normalized(),
            Vector::from((-2f32, -1f32, -3f32)).normalized(),
            Vector::from((5f32, 0.1f32, -0.1f32)).normalized(),
        ];
        for normal in normals {
            let mut rng = rng(seeds()[0]);
            let mut stats = [Stats::new(); 2];
            for _ in 0..50_000 {
                let w = UnitVector::random_cosine(normal, &mut rng);
                let z = w.dot(normal);
                assert!(
                    z >= -1e-6,
                    "cosine sample below the horizon of {normal:?}: {w:?}"
                );
                stats[0].push(z);
                stats[1].push(z * z);
            }
            assert_stat_close(stats[0].mean(), stats[0].se(), 2f32 / 3f32, 4f32, 0f32);
            assert_stat_close(stats[1].mean(), stats[1].se(), 0.5f32, 4f32, 0f32);
        }

        let normal = normals[0];
        let mut rng = rng(seeds()[1]);
        chi2_directions(
            50_000,
            &mut rng,
            normal,
            0f64,
            10,
            10,
            |rng| UnitVector::random_cosine(normal, rng),
            |z0, z1| z1 * z1 - z0 * z0,
        );
    }

    /// Uniform hemisphere sampling has density 1/(2*pi) on dot(w, n) >= 0, so
    /// E[z^k] = 1/(k+1): E[z] = 1/2, E[z^2] = 1/3.
    #[test]
    fn random_hemisphere_is_uniform_hemisphere() {
        let normal = Vector::from((1f32, 2f32, 3f32)).normalized();
        for seed in seeds() {
            let mut rng = rng(seed);
            let mut stats = [Stats::new(); 2];
            for _ in 0..100_000 {
                let w = UnitVector::random_hemisphere(normal, &mut rng);
                let z = w.dot(normal);
                assert!(z >= -1e-6, "hemisphere sample below the horizon: {w:?}");
                stats[0].push(z);
                stats[1].push(z * z);
            }
            assert_stat_close(stats[0].mean(), stats[0].se(), 0.5f32, 4f32, 0f32);
            assert_stat_close(stats[1].mean(), stats[1].se(), 1f32 / 3f32, 4f32, 0f32);
        }

        let mut rng = rng(seeds()[0]);
        chi2_directions(
            50_000,
            &mut rng,
            normal,
            0f64,
            10,
            10,
            |rng| UnitVector::random_hemisphere(normal, rng),
            |z0, z1| z1 - z0,
        );
    }

    /// Heavy tier: same goodness-of-fit checks with 10x the samples and a
    /// finer grid. Run with `cargo test -p rust-tracer-lib -- --ignored`.
    #[test]
    #[ignore = "heavy convergence tier"]
    fn heavy_chi2_samplers() {
        let mut rng = rng(seeds()[2]);
        chi2_directions(
            1_000_000,
            &mut rng,
            UnitVector::Z,
            -1f64,
            20,
            20,
            UnitVector::random,
            |z0, z1| (z1 - z0) / 2.0,
        );
        chi2_directions(
            1_000_000,
            &mut rng,
            UnitVector::Z,
            0f64,
            20,
            20,
            UnitVector::random_cosine_local,
            |z0, z1| z1 * z1 - z0 * z0,
        );
    }
}
