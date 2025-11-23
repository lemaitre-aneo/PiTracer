use serde::{Deserialize, Serialize, de::Visitor};

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize)]
#[serde(default)]
pub struct Color<T = f32> {
    pub r: T,
    pub g: T,
    pub b: T,
}

pub type ColorVec = Vec<Color>;

impl<'de> Deserialize<'de> for Color {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct FieldVisitor;

        impl<'de> Visitor<'de> for FieldVisitor {
            type Value = Color;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("either {r: f32, g: f32, b: f32} or (f32, f32, f32)")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                Ok(Color {
                    r: seq.next_element()?.unwrap_or_default(),
                    g: seq.next_element()?.unwrap_or_default(),
                    b: seq.next_element()?.unwrap_or_default(),
                })
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut color = Color::default();

                while let Some((key, value)) = map.next_entry::<String, f32>()? {
                    match key.as_ref() {
                        "r" => color.r = value,
                        "g" => color.g = value,
                        "b" => color.b = value,
                        _ => Err(serde::de::Error::custom("Expected either `r`, `g`, `b`"))?,
                    }
                }
                Ok(color)
            }
        }

        deserializer.deserialize_any(FieldVisitor)
    }
}


macro_rules! impl_op {
    ($op:ident($opassign:ident): $trait:ident($traitassign:ident): $assign:tt) => {
        impl_op!($opassign(Color, f32): $traitassign: $assign);
        impl_op!($opassign(Color, Color): $traitassign: $assign {.});

        impl_op!($op(Color, f32) -> Color: $trait: $assign);
        impl_op!($op(Color, Color) -> Color: $trait: $assign);
        impl_op!($op(f32, Color) -> Color: $trait: $assign);
    };

    ($op:ident($({$dereflhs:tt})? $lhs:ty, $({$derefrhs:tt})? $rhs:ty): $trait:ident : $assign:tt $({$member:tt})?) => {
        impl std::ops::$trait<$rhs> for $lhs {
            fn $op(&mut self, rhs: $rhs) {
                $($dereflhs)? self.r $assign $($derefrhs)? rhs $($member r)?;
                $($dereflhs)? self.g $assign $($derefrhs)? rhs $($member g)?;
                $($dereflhs)? self.b $assign $($derefrhs)? rhs $($member b)?;
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

impl std::ops::Neg for Color {
    type Output = Color;

    fn neg(self) -> Self::Output {
        Color {
            r: -self.r,
            g: -self.g,
            b: -self.b,
        }
    }
}
impl From<f32> for Color {
    fn from(value: f32) -> Self {
        Self {
            r: value,
            g: value,
            b: value,
        }
    }
}

impl From<(f32, f32, f32)> for Color {
    fn from(value: (f32, f32, f32)) -> Self {
        Self {
            r: value.0,
            g: value.1,
            b: value.2,
        }
    }
}

impl Color {
    pub fn splat(value: f32) -> Self {
        value.into()
    }
    pub fn sum(self) -> f32 {
        self.r + self.g + self.b
    }
    pub fn norm2(self) -> f32 {
        (self * self).sum()
    }
    pub fn norm(self) -> f32 {
        self.norm2().sqrt()
    }
}
