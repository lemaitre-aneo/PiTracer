use serde::{Deserialize, Serialize, de::Visitor};
use soa_derive::StructOfArray;

#[derive(StructOfArray, Debug, Default, Clone, Copy, PartialEq, PartialOrd, Serialize)]
#[soa_derive(Debug, Default, Clone, PartialEq, Serialize)]
#[serde(default)]
pub struct Vector {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl<'de> Deserialize<'de> for Vector {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct FieldVisitor;

        impl<'de> Visitor<'de> for FieldVisitor {
            type Value = Vector;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("either {x: f32, y: f32, z: f32} or (f32, f32, f32)")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                Ok(Vector {
                    x: seq.next_element()?.unwrap_or_default(),
                    y: seq.next_element()?.unwrap_or_default(),
                    z: seq.next_element()?.unwrap_or_default(),
                })
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut vec = Vector::default();

                while let Some((key, value)) = map.next_entry::<String, f32>()? {
                    match key.as_str() {
                        "x" => vec.x = value,
                        "y" => vec.y = value,
                        "z" => vec.z = value,
                        _ => Err(serde::de::Error::custom("Expected either `x`, `y`, `z`"))?,
                    }
                }
                Ok(vec)
            }
        }

        deserializer.deserialize_any(FieldVisitor)
    }
}

impl<'de> Deserialize<'de> for VectorVec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct FieldVisitor;

        impl<'de> Visitor<'de> for FieldVisitor {
            type Value = VectorVec;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("either [{x: f32, y: f32, z: f32}], [(f32, f32, f32)], or {x: [f32], y: [f32], z: [f32]}")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut vectors = Vec::new();

                while let Some(vec) = seq.next_element()? {
                    vectors.push(vec);
                }

                Ok(vectors.into_iter().collect())
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut vec = VectorVec::default();

                while let Some((key, value)) = map.next_entry::<String, Vec<f32>>()? {
                    match key.as_str() {
                        "x" => vec.x = value,
                        "y" => vec.y = value,
                        "z" => vec.z = value,
                        _ => Err(serde::de::Error::custom("Expected either `x`, `y`, `z`"))?,
                    }
                }
                Ok(vec)
            }
        }

        deserializer.deserialize_any(FieldVisitor)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(default)]
pub struct UnitVector(Vector);

macro_rules! impl_op {
    ($op:ident($opassign:ident): $trait:ident($traitassign:ident): $assign:tt) => {
        impl_op!($opassign(Vector, f32): $traitassign: $assign);
        impl_op!($opassign(Vector, Vector): $traitassign: $assign {.});
        impl_op!($opassign(Vector, UnitVector): $traitassign: $assign {.0.});
        impl_op!($opassign(Vector, VectorRef<'_>): $traitassign: $assign {.});
        impl_op!($opassign(Vector, {*} VectorRefMut<'_>): $traitassign: $assign {.});
        impl_op!($opassign({*} VectorRefMut<'_>, f32): $traitassign: $assign);
        impl_op!($opassign({*} VectorRefMut<'_>, Vector): $traitassign: $assign {.});
        impl_op!($opassign({*} VectorRefMut<'_>, UnitVector): $traitassign: $assign {.0.});
        impl_op!($opassign({*} VectorRefMut<'_>, VectorRef<'_>): $traitassign: $assign {.});
        impl_op!($opassign({*} VectorRefMut<'_>, {*} VectorRefMut<'_>): $traitassign: $assign {.});


        impl_op!($op(Vector, f32) -> Vector: $trait: $assign);
        impl_op!($op(Vector, Vector) -> Vector: $trait: $assign);
        impl_op!($op(Vector, UnitVector) -> Vector: $trait: $assign);
        impl_op!($op(Vector, VectorRef<'_>) -> Vector: $trait: $assign);
        impl_op!($op(Vector, VectorRefMut<'_>) -> Vector: $trait: $assign);
        impl_op!($op(UnitVector, f32) -> Vector: $trait: $assign);
        impl_op!($op(UnitVector, Vector) -> Vector: $trait: $assign);
        impl_op!($op(UnitVector, UnitVector) -> Vector: $trait: $assign);
        impl_op!($op(UnitVector, VectorRef<'_>) -> Vector: $trait: $assign);
        impl_op!($op(UnitVector, VectorRefMut<'_>) -> Vector: $trait: $assign);
        impl_op!($op(VectorRef<'_>, f32) -> Vector: $trait: $assign);
        impl_op!($op(VectorRef<'_>, Vector) -> Vector: $trait: $assign);
        impl_op!($op(VectorRef<'_>, UnitVector) -> Vector: $trait: $assign);
        impl_op!($op(VectorRef<'_>, VectorRef<'_>) -> Vector: $trait: $assign);
        impl_op!($op(VectorRef<'_>, VectorRefMut<'_>) -> Vector: $trait: $assign);
        impl_op!($op(VectorRefMut<'_>, f32) -> Vector: $trait: $assign);
        impl_op!($op(VectorRefMut<'_>, Vector) -> Vector: $trait: $assign);
        impl_op!($op(VectorRefMut<'_>, UnitVector) -> Vector: $trait: $assign);
        impl_op!($op(VectorRefMut<'_>, VectorRef<'_>) -> Vector: $trait: $assign);
        impl_op!($op(VectorRefMut<'_>, VectorRefMut<'_>) -> Vector: $trait: $assign);
        impl_op!($op(f32, Vector) -> Vector: $trait: $assign);
        impl_op!($op(f32, UnitVector) -> Vector: $trait: $assign);
        impl_op!($op(f32, VectorRef<'_>) -> Vector: $trait: $assign);
        impl_op!($op(f32, VectorRefMut<'_>) -> Vector: $trait: $assign);
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
impl std::ops::Neg for VectorRef<'_> {
    type Output = Vector;

    fn neg(self) -> Self::Output {
        -self.to_owned()
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
        UnitVector(self / self.norm())
    }
    pub fn reflect(self, normal: UnitVector) -> Vector {
        let an = 2f32 * (self * normal).sum();
        self - an * normal
    }
    pub fn cross(self, rhs: impl Into<Vector>) -> Vector {
        let rhs = rhs.into();
        Vector {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
}

impl VectorRef<'_> {
    pub fn sum(self) -> f32 {
        self.to_owned().sum()
    }
    pub fn norm2(self) -> f32 {
        self.to_owned().norm2()
    }
    pub fn norm(self) -> f32 {
        self.to_owned().norm()
    }
    pub fn normalized(self) -> UnitVector {
        self.to_owned().normalized()
    }
    pub fn reflect(self, normal: UnitVector) -> Vector {
        self.to_owned().reflect(normal)
    }
    pub fn cross(self, rhs: impl Into<Vector>) -> Vector {
        self.to_owned().cross(rhs)
    }
}

impl VectorRefMut<'_> {
    pub fn sum(&self) -> f32 {
        self.to_owned().sum()
    }
    pub fn norm2(&self) -> f32 {
        self.to_owned().norm2()
    }
    pub fn norm(&self) -> f32 {
        self.to_owned().norm()
    }
    pub fn normalized(&self) -> UnitVector {
        self.to_owned().normalized()
    }
    pub fn reflect(&self, normal: UnitVector) -> Vector {
        self.to_owned().reflect(normal)
    }
    pub fn cross(self, rhs: impl Into<Vector>) -> Vector {
        self.to_owned().cross(rhs)
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
    pub fn random(rng: &mut impl rand::Rng) -> UnitVector {
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
    pub fn random_hemisphere(normal: UnitVector, rng: &mut impl rand::Rng) -> UnitVector {
        let vec = Self::random(rng);
        std::hint::select_unpredictable(vec.dot(normal) > 0f32, vec, -vec)
    }
}

pub trait Dot<Rhs = Self> {
    type Output;
    fn dot(self, rhs: Rhs) -> Self::Output;
}

impl Dot<Vector> for Vector {
    type Output = f32;

    fn dot(self, rhs: Vector) -> Self::Output {
        (self * rhs).sum()
    }
}
impl Dot<UnitVector> for Vector {
    type Output = f32;

    fn dot(self, rhs: UnitVector) -> Self::Output {
        (self * rhs).sum()
    }
}
impl Dot<VectorRef<'_>> for Vector {
    type Output = f32;

    fn dot(self, rhs: VectorRef<'_>) -> Self::Output {
        (self * rhs).sum()
    }
}
impl Dot<VectorRefMut<'_>> for Vector {
    type Output = f32;

    fn dot(self, rhs: VectorRefMut<'_>) -> Self::Output {
        (self * rhs).sum()
    }
}

impl Dot<Vector> for UnitVector {
    type Output = f32;

    fn dot(self, rhs: Vector) -> Self::Output {
        (self * rhs).sum()
    }
}
impl Dot<UnitVector> for UnitVector {
    type Output = f32;

    fn dot(self, rhs: UnitVector) -> Self::Output {
        (self * rhs).sum()
    }
}
impl Dot<VectorRef<'_>> for UnitVector {
    type Output = f32;

    fn dot(self, rhs: VectorRef<'_>) -> Self::Output {
        (self * rhs).sum()
    }
}
impl Dot<&VectorRefMut<'_>> for UnitVector {
    type Output = f32;

    fn dot(self, rhs: &VectorRefMut<'_>) -> Self::Output {
        (self * rhs.to_owned()).sum()
    }
}

impl Dot<Vector> for VectorRef<'_> {
    type Output = f32;

    fn dot(self, rhs: Vector) -> Self::Output {
        (self * rhs).sum()
    }
}
impl Dot<UnitVector> for VectorRef<'_> {
    type Output = f32;

    fn dot(self, rhs: UnitVector) -> Self::Output {
        (self * rhs).sum()
    }
}
impl Dot<VectorRef<'_>> for VectorRef<'_> {
    type Output = f32;

    fn dot(self, rhs: VectorRef<'_>) -> Self::Output {
        (self * rhs).sum()
    }
}
impl Dot<&VectorRefMut<'_>> for VectorRef<'_> {
    type Output = f32;

    fn dot(self, rhs: &VectorRefMut<'_>) -> Self::Output {
        (self * rhs.to_owned()).sum()
    }
}

impl Dot<Vector> for &VectorRefMut<'_> {
    type Output = f32;

    fn dot(self, rhs: Vector) -> Self::Output {
        (self.to_owned() * rhs).sum()
    }
}
impl Dot<UnitVector> for &VectorRefMut<'_> {
    type Output = f32;

    fn dot(self, rhs: UnitVector) -> Self::Output {
        (self.to_owned() * rhs).sum()
    }
}
impl Dot<VectorRef<'_>> for &VectorRefMut<'_> {
    type Output = f32;

    fn dot(self, rhs: VectorRef<'_>) -> Self::Output {
        (self.to_owned() * rhs).sum()
    }
}
impl Dot<&VectorRefMut<'_>> for &VectorRefMut<'_> {
    type Output = f32;

    fn dot(self, rhs: &VectorRefMut<'_>) -> Self::Output {
        (self.to_owned() * rhs.to_owned()).sum()
    }
}
