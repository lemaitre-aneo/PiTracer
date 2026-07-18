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

/// Minimum hit distance: intersections closer than this are treated as
/// self-intersections of the ray with its own originating surface.
/// Sized for f32 geometry (the classic scene has radius-1e5 walls, where
/// |oc|² carries an absolute rounding error of order 1e3).
pub const HIT_EPSILON: f32 = 7e-2;

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
