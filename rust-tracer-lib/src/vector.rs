use serde::{Deserialize, Serialize};

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

#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
pub struct UnitVector<T = f32>(Vector<T>);

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
}
