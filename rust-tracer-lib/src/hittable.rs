use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};

use crate::{
    Color, Geometry, Hit, HitCaster, Material, Texture, TextureProperty, UnitVector, Vector,
};

/// Union of the three behavior traits a scene object provides. Blanket
/// implemented for anything satisfying all three; `Scene` stores objects as
/// `Arc<dyn Hittable>`.
pub trait Hittable<R: crate::Rng + ?Sized = dyn RngCore>:
    Geometry<R> + Material<R> + Texture<R> + std::fmt::Debug
{
}

impl<R: crate::Rng + ?Sized, H: Geometry<R> + Material<R> + Texture<R> + std::fmt::Debug + ?Sized>
    Hittable<R> for H
{
}

pub struct Radiance {
    pub emission: Color,
    pub absorption: Color,
    pub origin: Vector,
    pub direction: UnitVector,
}

/// A scene object: geometry + material + texture, each trait impl forwarding
/// to its component. The `#[serde(flatten)]` fields make one YAML `objects:`
/// entry carry all three aspects in a single map (geometry fields, the
/// `material:` tag, texture keys side by side).
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

/// The texture's transparency is applied here, at intersection, so that
/// every ray (camera, continuation, shadow) sees through it alike: a hit is
/// kept with probability [`Texture::opacity`], otherwise the ray goes on to
/// the surface's next crossing (the far side of a sphere, nothing for a
/// quad). Stochastic alpha is unbiased: on average a ray is stopped by the
/// fraction of coverage, which also anti-aliases the image's edges.
impl<G: Geometry, M: std::fmt::Debug, T: Texture> Geometry for Object<G, M, T> {
    fn hit<'a, 'b>(
        &'a self,
        origin: Vector,
        direction: UnitVector,
        rng: &'b mut (dyn RngCore + 'static),
    ) -> Option<(f32, &'a dyn HitCaster)> {
        let (mut distance, mut caster) = self.geometry.hit(origin, direction, rng)?;
        if !self.texture.transparent() {
            return Some((distance, caster));
        }
        loop {
            let hit = self.geometry.to_hit(origin, direction, distance);
            let opacity = self.texture.opacity(&hit);
            if opacity >= 1.0 || rng.random::<f32>() < opacity {
                return Some((distance, caster));
            }
            let beyond = hit.spawn(direction);
            let (further, next) = self.geometry.hit(beyond, direction, rng)?;
            distance = (beyond + further * direction - origin).dot(direction);
            caster = next;
        }
    }

    fn to_hit(&self, origin: Vector, direction: UnitVector, distance: f32) -> Hit {
        self.geometry.to_hit(origin, direction, distance)
    }

    fn target_random(&self, origin: Vector, rng: &mut (dyn RngCore + 'static)) -> UnitVector {
        self.geometry.target_random(origin, rng)
    }

    fn target_pdf(&self, origin: Vector, scattered: UnitVector) -> f32 {
        self.geometry.target_pdf(origin, scattered)
    }
}

impl<R: crate::Rng + ?Sized, G: std::fmt::Debug, M: Material<R>, T: std::fmt::Debug> Material<R>
    for Object<G, M, T>
{
    fn scatter(&self, direction: UnitVector, hit: Hit, rng: &mut R) -> Option<UnitVector> {
        self.material.scatter(direction, hit, rng)
    }

    fn pdf(&self, direction: UnitVector, hit: Hit, scattered: UnitVector) -> f32 {
        self.material.pdf(direction, hit, scattered)
    }
}

impl<R: crate::Rng + ?Sized, G: std::fmt::Debug, M: std::fmt::Debug, T: Texture<R>> Texture<R>
    for Object<G, M, T>
{
    fn get(&self, hit: Hit, rng: &mut R) -> TextureProperty {
        self.texture.get(hit, rng)
    }

    fn transparent(&self) -> bool {
        self.texture.transparent()
    }

    fn opacity(&self, hit: &Hit) -> f32 {
        self.texture.opacity(hit)
    }
}
