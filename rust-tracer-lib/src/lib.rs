#[cfg(test)]
pub(crate) mod test_utils;

mod camera;
mod color;
mod geometry;
mod hittable;
mod material;
mod onb;
mod scene;
mod texture;
mod vector;

pub use camera::*;
pub use color::*;
pub use geometry::*;
pub use hittable::*;
pub use material::*;
pub use onb::*;
pub use scene::*;
pub use texture::*;
pub use vector::*;

/// Minimum hit distance as a fraction of the ray origin's distance to the
/// sphere center: roots with t ≤ HIT_EPSILON_REL·|oc| are rejected. This is
/// only a safety net for rays whose origin lies (numerically) on a surface —
/// the primary self-intersection defense is [`Hit::offset`]: rays spawned off
/// the surface cannot re-find it geometrically, grazing included, which no
/// t-guard can promise (grazing phantom t reaches √(2·r·δ), ≈ 45 for a δ≈1e-2
/// point error on an r=1e5 wall — far beyond any usable threshold).
pub const HIT_EPSILON_REL: f32 = 1e-6;

/// Ray-spawn offset as a fraction of the geometry's coordinate scale
/// (radius + |center| for a sphere): a hit point can only be placed on its
/// surface to within ~1.5 ulps of that scale, so spawned rays start this far
/// off the surface, on the side they travel (see [`Hit::offset`]). Kept as
/// tight as the noise allows (≈ 8× the f32 epsilon): the offset displaces
/// the receiver radiometrically, biasing incident solid angles by
/// 2·offset/distance, so every unnecessary factor here is bias.
pub const SURFACE_OFFSET_REL: f32 = 1e-6;

pub fn select<T>(cond: bool, then: T, otherwise: T) -> T {
    // unsafe {
    //     let mut array = [
    //         std::mem::ManuallyDrop::new(otherwise),
    //         std::mem::ManuallyDrop::new(then),
    //     ];
    //     std::mem::ManuallyDrop::drop(&mut array[(!cond) as usize]);
    //     std::mem::ManuallyDrop::take(&mut array[cond as usize])
    // }
    if cond { then } else { otherwise }
}

pub trait Rng: rand::RngCore + sealed::Sealed {}

mod sealed {
    pub trait Sealed {}

    impl Sealed for rand::rngs::SmallRng {}
    impl Sealed for rand::rngs::StdRng {}
    impl Sealed for rand::rngs::ThreadRng {}
    impl Sealed for dyn rand::RngCore {}
}

impl<R: rand::RngCore + sealed::Sealed + ?Sized> Rng for R {}
