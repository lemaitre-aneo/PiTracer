#[cfg(test)]
pub(crate) mod test_utils;

mod camera;
mod color;
mod denoise;
mod geometry;
mod hittable;
mod material;
#[cfg(feature = "denoise")]
mod oidn_filter;
mod onb;
mod scene;
mod task;
mod texture;
mod vector;

pub use camera::*;
pub use color::*;
pub use denoise::*;
pub use geometry::*;
pub use hittable::*;
pub use material::*;
#[cfg(feature = "denoise")]
pub use oidn_filter::{DenoiseError, Denoiser};
pub use onb::*;
pub use scene::*;
pub use task::*;
pub use texture::*;
pub use vector::*;

/// Minimum hit distance as a fraction of the ray origin's distance to the
/// sphere center: roots with t <= HIT_EPSILON_REL * |oc| are rejected. Only a
/// safety net for rays whose origin lies (numerically) on a surface; the
/// primary self-intersection defense is [`Hit::offset`]: rays spawned off the
/// surface cannot re-find it geometrically, grazing included, which no
/// t-guard can promise (a grazing phantom t reaches sqrt(2*r*delta), about 45
/// for a delta = 1e-2 point error on an r = 1e5 wall, far beyond any usable
/// threshold).
pub const HIT_EPSILON_REL: f32 = 1e-6;

/// Ray-spawn offset as a fraction of the geometry's coordinate scale
/// (radius + |center| for a sphere): a hit point can only be placed on its
/// surface to within ~1.5 ulps of that scale, so spawned rays start this far
/// off the surface, on the side they travel (see [`Hit::offset`]). Kept as
/// tight as the noise allows (about 8x the f32 epsilon): the offset displaces
/// the receiver radiometrically, biasing incident solid angles by
/// 2 * offset / distance, so every unnecessary factor here is bias.
pub const SURFACE_OFFSET_REL: f32 = 1e-6;

/// Plain branchy select, kept as a function so a branchless implementation
/// could be swapped in behind one seam if profiling ever justified it.
pub fn select<T>(cond: bool, then: T, otherwise: T) -> T {
    if cond { then } else { otherwise }
}

/// RNG bound used by all the behavior traits: sealed so that both the
/// concrete RNGs (fast, monomorphized) and `dyn rand::Rng` (object-safe
/// default) satisfy it, and nothing else can.
pub trait Rng: rand::Rng + sealed::Sealed {}

mod sealed {
    pub trait Sealed {}

    impl Sealed for rand::rngs::SmallRng {}
    impl Sealed for rand::rngs::StdRng {}
    impl Sealed for rand::rngs::ThreadRng {}
    impl Sealed for dyn rand::Rng {}
}

impl<R: rand::Rng + sealed::Sealed + ?Sized> Rng for R {}
