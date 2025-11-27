use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::{
    Color, Geometry, Hit, HitCaster, Material, Texture, TextureProperty, UnitVector, Vector,
};

pub trait Hittable<R: crate::Rng + ?Sized = dyn RngCore>:
    Geometry<R> + Material<R> + Texture<R>
{
}

impl<R: crate::Rng + ?Sized, H: Geometry<R> + Material<R> + Texture<R> + ?Sized> Hittable<R> for H {}

pub struct Radiance {
    pub emission: Color,
    pub absorption: Color,
    pub origin: Vector,
    pub direction: UnitVector,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Object<G, M, T> {
    #[serde(flatten)]
    pub geometry: G,
    #[serde(flatten)]
    pub material: M,
    #[serde(flatten)]
    pub texture: T,
}

impl<G, M, T> Object<G, M, T> {
    pub fn new(geometry: G, material: M, texture: T) -> Self {
        Self {
            geometry,
            material,
            texture,
        }
    }
}

impl<G: Geometry, M: std::fmt::Debug, T: std::fmt::Debug> Geometry for Object<G, M, T> {
    fn hit<'a, 'b>(
        &'a self,
        origin: Vector,
        direction: UnitVector,
        rng: &'b mut (dyn RngCore + 'static),
    ) -> Option<(f32, &'a dyn HitCaster)> {
        self.geometry.hit(origin, direction, rng)
    }

    fn to_hit(&self, origin: Vector, direction: UnitVector, distance: f32) -> Hit {
        self.geometry.to_hit(origin, direction, distance)
    }
}

impl<R: crate::Rng + ?Sized, G: std::fmt::Debug, M: Material<R>, T: std::fmt::Debug> Material<R>
    for Object<G, M, T>
{
    fn scatter(&self, direction: UnitVector, hit: Hit, rng: &mut R) -> Option<UnitVector> {
        self.material.scatter(direction, hit, rng)
    }
}

impl<R: crate::Rng + ?Sized, G: std::fmt::Debug, M: std::fmt::Debug, T: Texture<R>> Texture<R>
    for Object<G, M, T>
{
    fn get(&self, hit: Hit, rng: &mut R) -> TextureProperty {
        self.texture.get(hit, rng)
    }
}
