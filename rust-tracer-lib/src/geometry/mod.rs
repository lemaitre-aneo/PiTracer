use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::{Material, Texture, UnitVector, Vector};

mod ellipse;
mod quad;
mod sphere;

pub use ellipse::*;
pub use quad::*;
pub use sphere::*;

/// The geometry side of a scene-definition object, discriminated by an
/// explicit `type:` field (`type: sphere`, `type: quad`) in the same
/// flattened YAML map that carries the material tag and texture keys.
///
/// Unlike `MaterialEnum` (where a missing `material:` deliberately falls back
/// to Lambertian), a missing or unknown `type` is a hard deserialization
/// error: geometry has no sensible default, and failing loudly is what keeps
/// a typo from silently dropping an object.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum GeometryEnum {
    #[serde(alias = "Sphere")]
    Sphere(Sphere),
    #[serde(alias = "Quad")]
    Quad(Quad),
    #[serde(alias = "Ellipse")]
    Ellipse(Ellipse),
}

/// Only for `Object`'s container-level `#[serde(default)]` bound — the YAML
/// path always goes through the tag and can never produce this.
impl Default for GeometryEnum {
    fn default() -> Self {
        Self::Sphere(Sphere::default())
    }
}

impl<R: crate::Rng + ?Sized> GeometryPrimitive<R> for GeometryEnum {
    fn hit(&self, origin: Vector, direction: UnitVector, rng: &mut R) -> Option<f32> {
        match self {
            GeometryEnum::Sphere(sphere) => {
                GeometryPrimitive::<R>::hit(sphere, origin, direction, rng)
            }
            GeometryEnum::Quad(quad) => GeometryPrimitive::<R>::hit(quad, origin, direction, rng),
            GeometryEnum::Ellipse(ellipse) => {
                GeometryPrimitive::<R>::hit(ellipse, origin, direction, rng)
            }
        }
    }

    fn to_hit(&self, origin: Vector, direction: UnitVector, distance: f32) -> Hit {
        match self {
            GeometryEnum::Sphere(sphere) => {
                GeometryPrimitive::<R>::to_hit(sphere, origin, direction, distance)
            }
            GeometryEnum::Quad(quad) => {
                GeometryPrimitive::<R>::to_hit(quad, origin, direction, distance)
            }
            GeometryEnum::Ellipse(ellipse) => {
                GeometryPrimitive::<R>::to_hit(ellipse, origin, direction, distance)
            }
        }
    }

    fn target_random(&self, origin: Vector, rng: &mut R) -> UnitVector {
        match self {
            GeometryEnum::Sphere(sphere) => {
                GeometryPrimitive::<R>::target_random(sphere, origin, rng)
            }
            GeometryEnum::Quad(quad) => GeometryPrimitive::<R>::target_random(quad, origin, rng),
            GeometryEnum::Ellipse(ellipse) => {
                GeometryPrimitive::<R>::target_random(ellipse, origin, rng)
            }
        }
    }

    fn target_pdf(&self, origin: Vector, scattered: UnitVector) -> f32 {
        match self {
            GeometryEnum::Sphere(sphere) => {
                GeometryPrimitive::<R>::target_pdf(sphere, origin, scattered)
            }
            GeometryEnum::Quad(quad) => GeometryPrimitive::<R>::target_pdf(quad, origin, scattered),
            GeometryEnum::Ellipse(ellipse) => {
                GeometryPrimitive::<R>::target_pdf(ellipse, origin, scattered)
            }
        }
    }
}

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
