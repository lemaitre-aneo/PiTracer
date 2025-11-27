use serde::{Deserialize, Serialize};

mod definition {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct Empty {}

    #[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
    #[serde(untagged)]
    pub(super) enum Color<T = f32> {
        #[default]
        Default,
        Empty(Empty),
        List(T, T, T),
        Map {
            r: T,
            g: T,
            b: T,
        },
    }

    impl<T> From<super::Color<T>> for Color<T> {
        fn from(value: super::Color<T>) -> Self {
            Self::List(value.r, value.g, value.b)
        }
    }

    impl<T: Default> From<Color<T>> for super::Color<T> {
        fn from(value: Color<T>) -> Self {
            match value {
                Color::List(r, g, b) => Self { r, g, b },
                Color::Map { r, g, b } => Self { r, g, b },
                _ => Self::default(),
            }
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(from = "definition::Color<T>", into = "definition::Color<T>")]
#[serde(bound = "for <'a> T: Default + Clone + Serialize + Deserialize<'a>")]
#[serde(default)]
pub struct Color<T = f32> {
    pub r: T,
    pub g: T,
    pub b: T,
}

pub type ColorVec = Vec<Color>;

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
