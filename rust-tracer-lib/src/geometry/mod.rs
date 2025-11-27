use rand::RngCore;

use crate::{Material, Texture, UnitVector, Vector};

mod sphere;

pub use sphere::*;

pub(crate) trait GeometryPrimitive<R: crate::Rng + ?Sized = dyn RngCore>: std::fmt::Debug {
    fn hit(&self, origin: Vector, direction: UnitVector, rng: &mut R) -> Option<f32>;

    fn to_hit(&self, origin: Vector, direction: UnitVector, distance: f32) -> Hit;
}

pub trait Geometry<R: crate::Rng + ?Sized = dyn RngCore>: std::fmt::Debug {
    fn hit<'a>(
        &'a self,
        origin: Vector,
        direction: UnitVector,
        rng: &mut R,
    ) -> Option<(f32, &'a dyn HitCaster<R>)>;

    fn to_hit(&self, origin: Vector, direction: UnitVector, distance: f32) -> Hit;
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Hit {
    pub point: Vector,
    pub normal: UnitVector,
    pub front: bool,
    pub u: f32,
    pub v: f32,
    pub distance: f32,
}

pub trait HitCaster<R: crate::Rng + ?Sized = dyn RngCore> {
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
}

impl<R: crate::Rng + ?Sized, G: GeometryPrimitive<R>> HitCaster<R> for G {
    fn as_geometry(&self) -> &dyn Geometry<R> {
        self
    }
}
