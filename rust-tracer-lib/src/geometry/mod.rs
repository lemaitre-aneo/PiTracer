use rand::RngCore;

use crate::{Material, Texture, UnitVector, Vector};

mod sphere;

pub use sphere::*;

pub(crate) trait GeometryPrimitive<R: crate::Rng + ?Sized = dyn RngCore>:
    std::fmt::Debug
{
    fn hit(&self, origin: Vector, direction: UnitVector, rng: &mut R) -> Option<f32>;

    fn to_hit(&self, origin: Vector, direction: UnitVector, distance: f32) -> Hit;

    fn target_random(&self, origin: Vector, rng: &mut R) -> UnitVector {
        _ = (origin, rng);
        UnitVector::default()
    }
    fn target_pdf(&self, origin: Vector, scattered: UnitVector) -> f32 {
        _ = (origin, scattered);
        0.0
    }
}

pub trait Geometry<R: crate::Rng + ?Sized = dyn RngCore>: std::fmt::Debug {
    fn hit<'a>(
        &'a self,
        origin: Vector,
        direction: UnitVector,
        rng: &mut R,
    ) -> Option<(f32, &'a dyn HitCaster<R>)>;

    fn to_hit(&self, origin: Vector, direction: UnitVector, distance: f32) -> Hit;

    fn target_random(&self, origin: Vector, rng: &mut R) -> UnitVector {
        _ = (origin, rng);
        UnitVector::default()
    }
    fn target_pdf(&self, origin: Vector, scattered: UnitVector) -> f32 {
        _ = (origin, scattered);
        0.0
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Hit {
    pub point: Vector,
    pub normal: UnitVector,
    pub front: bool,
    pub u: f32,
    pub v: f32,
    pub distance: f32,
    /// How far a ray spawned at this hit must start off the surface: a few
    /// ulps of the geometry's coordinate scale, i.e. just above the rounding
    /// noise of `point`'s placement on the surface, and far below any
    /// resolvable feature. Applied by [`Hit::spawn`].
    pub offset: f32,
}

impl Hit {
    /// Origin for a secondary ray leaving this hit: `point` pushed `offset`
    /// off the surface toward the side `direction` travels (outside for
    /// reflection/diffuse, inside for transmission). An origin strictly off a
    /// convex surface cannot re-intersect it on its way out — grazing
    /// included — which is what makes the tiny relative hit guard
    /// ([`crate::HIT_EPSILON_REL`]) sufficient.
    pub fn spawn(&self, direction: UnitVector) -> Vector {
        self.point + self.offset.copysign(direction.dot(self.normal)) * self.normal
    }
}

pub trait HitCaster<R: crate::Rng + ?Sized = dyn RngCore>: std::fmt::Debug {
    fn as_geometry(&self) -> &dyn Geometry<R>;
    fn as_material<'a>(&'a self, default: &'a dyn Material<R>) -> &'a dyn Material<R> {
        default
    }
    fn as_texture<'a>(&'a self, default: &'a dyn Texture<R>) -> &'a dyn Texture<R> {
        default
    }
}

impl<R: crate::Rng + ?Sized, G: GeometryPrimitive<R>> Geometry<R> for G {
    fn hit<'a>(
        &'a self,
        origin: Vector,
        direction: UnitVector,
        rng: &mut R,
    ) -> Option<(f32, &'a dyn HitCaster<R>)> {
        self.hit(origin, direction, rng)
            .map(|distance| (distance, self as &dyn HitCaster<R>))
    }

    fn to_hit(&self, origin: Vector, direction: UnitVector, distance: f32) -> Hit {
        self.to_hit(origin, direction, distance)
    }

    fn target_random(&self, origin: Vector, rng: &mut R) -> UnitVector {
        self.target_random(origin, rng)
    }
    fn target_pdf(&self, origin: Vector, scattered: UnitVector) -> f32 {
        self.target_pdf(origin, scattered)
    }
}

impl<R: crate::Rng + ?Sized, G: GeometryPrimitive<R>> HitCaster<R> for G {
    fn as_geometry(&self) -> &dyn Geometry<R> {
        self
    }
}
