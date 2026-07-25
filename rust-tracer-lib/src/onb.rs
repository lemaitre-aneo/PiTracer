use crate::{UnitVector, Vector, select};

/// Orthonormal basis around `w`: `transform` maps local coordinates (z along
/// `w`) to world space. Handedness is unspecified; every distribution sampled
/// through this basis is phi-symmetric, so it never matters.
pub struct Onb {
    u: UnitVector,
    v: UnitVector,
    w: UnitVector,
}

impl Onb {
    pub fn new(w: UnitVector) -> Self {
        let a = select(w.x.abs() > 0.9f32, UnitVector::Y, UnitVector::X);
        let v = w.cross(a).normalized();
        let u = w.cross(v).normalized();
        Self { u, v, w }
    }

    pub fn transform(&self, vec: Vector) -> Vector {
        vec.x * self.u + vec.y * self.v + vec.z * self.w
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_utils::{assert_close, assert_vec_close, rng, seeds};

    /// The basis must be orthonormal, map local +z to w, and be a rotation up
    /// to handedness (|det| = 1). Handedness itself is not part of the spec:
    /// every distribution sampled through Onb is phi-symmetric.
    fn check_orthonormal(w: UnitVector) {
        let onb = Onb::new(w);

        assert_close(onb.u.dot(onb.v), 0f32, 1e-6);
        assert_close(onb.u.dot(onb.w), 0f32, 1e-6);
        assert_close(onb.v.dot(onb.w), 0f32, 1e-6);
        assert_close(Vector::from(onb.u).norm2(), 1f32, 1e-6);
        assert_close(Vector::from(onb.v).norm2(), 1f32, 1e-6);
        assert_close(Vector::from(onb.w).norm2(), 1f32, 1e-6);

        assert_vec_close(
            onb.transform(Vector::from((0f32, 0f32, 1f32))),
            Vector::from(w),
            1e-6,
        );

        let tx = onb.transform(Vector::from((1f32, 0f32, 0f32)));
        let ty = onb.transform(Vector::from((0f32, 1f32, 0f32)));
        let tz = onb.transform(Vector::from((0f32, 0f32, 1f32)));
        assert_close(tx.cross(ty).dot(tz).abs(), 1f32, 1e-6);
    }

    #[test]
    fn orthonormal_for_random_w() {
        let mut rng = rng(seeds()[0]);
        for _ in 0..1000 {
            check_orthonormal(UnitVector::random(&mut rng));
        }

        // both sides of the w.x branch threshold
        for dx in [-1e-6f32, 1e-6f32] {
            let x = 0.9f32 + dx;
            check_orthonormal(Vector::from((x, (1f32 - x * x).sqrt(), 0f32)).normalized());
        }

        for w in [
            UnitVector::X,
            UnitVector::Y,
            UnitVector::Z,
            -UnitVector::Y,
            -UnitVector::Z,
        ] {
            check_orthonormal(w);
        }
    }

    /// -X is a realistic normal (axis-aligned sphere hit on its +X side) and
    /// must produce a valid basis like any other direction.
    #[test]
    fn orthonormal_for_negative_x() {
        check_orthonormal(-UnitVector::X);
    }

    /// The transform is orthonormal, hence an isometry: it preserves dot
    /// products and norms. This is what makes cosine sampling valid in world
    /// space after being derived in the local frame.
    #[test]
    fn transform_is_isometry() {
        use rand::Rng as _;

        let mut rng = rng(seeds()[1]);
        for _ in 0..1000 {
            let onb = Onb::new(UnitVector::random(&mut rng));
            let mut random_vector = || Vector {
                x: rng.random::<f32>() * 10f32 - 5f32,
                y: rng.random::<f32>() * 10f32 - 5f32,
                z: rng.random::<f32>() * 10f32 - 5f32,
            };
            let a = random_vector();
            let b = random_vector();
            let ta = onb.transform(a);
            let tb = onb.transform(b);
            assert_close(ta.dot(tb), a.dot(b), 1e-4);
            assert_close(ta.norm(), a.norm(), 1e-5);
        }
    }

    #[test]
    fn aligned_x() {
        let onb = Onb::new(UnitVector::X);
        assert_eq!(onb.u, -UnitVector::Y);
        assert_eq!(onb.v, UnitVector::Z);
        assert_eq!(onb.w, UnitVector::X);
    }
    #[test]
    fn aligned_y() {
        let onb = Onb::new(UnitVector::Y);
        assert_eq!(onb.u, -UnitVector::X);
        assert_eq!(onb.v, -UnitVector::Z);
        assert_eq!(onb.w, UnitVector::Y);
    }
    #[test]
    fn aligned_z() {
        let onb = Onb::new(UnitVector::Z);
        assert_eq!(onb.u, -UnitVector::X);
        assert_eq!(onb.v, UnitVector::Y);
        assert_eq!(onb.w, UnitVector::Z);
    }
}
