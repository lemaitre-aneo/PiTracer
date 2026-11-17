use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{Material, Texture, UnitVector, Vector};

mod ellipse;
mod mesh;
mod quad;
mod sphere;

pub use ellipse::*;
pub use mesh::*;
pub use quad::*;
pub use sphere::*;

/// The geometry side of a scene-definition object, discriminated by an
/// explicit `type:` field (`type: sphere` / `quad` / `ellipse` / `mesh`) in
/// the same flattened YAML map that carries the material tag and texture
/// keys.
///
/// Unlike `MaterialEnum` (where a missing `material:` deliberately falls back
/// to Lambertian), a missing or unknown `type` is a hard deserialization
/// error: geometry has no sensible default, and failing loudly is what keeps
/// a typo from silently dropping an object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum GeometryEnum {
    #[serde(alias = "Sphere")]
    Sphere(Sphere),
    #[serde(alias = "Quad")]
    Quad(Quad),
    #[serde(alias = "Ellipse")]
    Ellipse(Ellipse),
    #[serde(alias = "Mesh")]
    Mesh(Mesh),
}

/// Only for `Object`'s container-level `#[serde(default)]` bound; the YAML
/// path always goes through the tag and can never produce this.
impl Default for GeometryEnum {
    fn default() -> Self {
        Self::Sphere(Sphere::default())
    }
}

impl GeometryEnum {
    /// Reads the geometry's file if it has one (a mesh `model`, relative to
    /// `base`).
    pub fn load(&mut self, base: &std::path::Path) -> Result<(), MeshError> {
        match self {
            GeometryEnum::Mesh(mesh) => mesh.load(base),
            GeometryEnum::Sphere(_) | GeometryEnum::Quad(_) | GeometryEnum::Ellipse(_) => Ok(()),
        }
    }
}

/// For the scene's RNG only (`dyn Rng`), like [`Mesh`]'s: a generic `R`
/// would overlap, for the coherence checker, with the blanket impl lifting
/// leaf primitives.
impl Geometry for GeometryEnum {
    fn hit<'a>(
        &'a self,
        origin: Vector,
        direction: UnitVector,
        rng: &mut (dyn Rng + 'static),
    ) -> Option<(f32, &'a dyn HitCaster)> {
        match self {
            GeometryEnum::Sphere(sphere) => Geometry::hit(sphere, origin, direction, rng),
            GeometryEnum::Quad(quad) => Geometry::hit(quad, origin, direction, rng),
            GeometryEnum::Ellipse(ellipse) => Geometry::hit(ellipse, origin, direction, rng),
            GeometryEnum::Mesh(mesh) => Geometry::hit(mesh, origin, direction, rng),
        }
    }

    fn to_hit(&self, origin: Vector, direction: UnitVector, distance: f32) -> Hit {
        match self {
            GeometryEnum::Sphere(sphere) => {
                Geometry::<dyn Rng>::to_hit(sphere, origin, direction, distance)
            }
            GeometryEnum::Quad(quad) => {
                Geometry::<dyn Rng>::to_hit(quad, origin, direction, distance)
            }
            GeometryEnum::Ellipse(ellipse) => {
                Geometry::<dyn Rng>::to_hit(ellipse, origin, direction, distance)
            }
            GeometryEnum::Mesh(mesh) => {
                Geometry::<dyn Rng>::to_hit(mesh, origin, direction, distance)
            }
        }
    }

    fn target_random(&self, origin: Vector, rng: &mut (dyn Rng + 'static)) -> UnitVector {
        match self {
            GeometryEnum::Sphere(sphere) => Geometry::target_random(sphere, origin, rng),
            GeometryEnum::Quad(quad) => Geometry::target_random(quad, origin, rng),
            GeometryEnum::Ellipse(ellipse) => Geometry::target_random(ellipse, origin, rng),
            GeometryEnum::Mesh(mesh) => Geometry::target_random(mesh, origin, rng),
        }
    }

    fn target_pdf(&self, origin: Vector, scattered: UnitVector) -> f32 {
        match self {
            GeometryEnum::Sphere(sphere) => {
                Geometry::<dyn Rng>::target_pdf(sphere, origin, scattered)
            }
            GeometryEnum::Quad(quad) => Geometry::<dyn Rng>::target_pdf(quad, origin, scattered),
            GeometryEnum::Ellipse(ellipse) => {
                Geometry::<dyn Rng>::target_pdf(ellipse, origin, scattered)
            }
            GeometryEnum::Mesh(mesh) => Geometry::<dyn Rng>::target_pdf(mesh, origin, scattered),
        }
    }
}

/// Leaf geometry: `hit` returns a bare distance. The blanket impls below lift
/// any primitive to [`Geometry`] and [`HitCaster`] (a leaf is its own caster).
pub(crate) trait GeometryPrimitive<R: crate::Rng + ?Sized = dyn Rng>:
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

/// Object-level geometry. `hit` returns the distance plus the [`HitCaster`]
/// for what was actually struck; the caller resolves `to_hit`, material, and
/// texture on the winner only. `target_random`/`target_pdf` are the
/// solid-angle light-sampling pair; the defaults (constant direction, pdf 0)
/// suit geometry that is never used as a light target.
pub trait Geometry<R: crate::Rng + ?Sized = dyn Rng>: std::fmt::Debug {
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

/// Surface interaction. `normal` (the shading normal materials scatter
/// around) and `geometric` (the true surface normal) both oppose the incident
/// ray; they differ only on smooth-shaded meshes. `front` records whether the
/// ray came from the side the geometric normal points toward; `u`/`v` are the
/// geometry's own surface parameters.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Hit {
    pub point: Vector,
    pub normal: UnitVector,
    /// The surface's own normal, oriented like `normal`. Spawned rays are
    /// offset along it (see [`Hit::spawn`]): which side of the surface a ray
    /// starts on is a geometric question, which an interpolated shading
    /// normal cannot answer.
    pub geometric: UnitVector,
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
    /// convex surface cannot re-intersect it on its way out, grazing
    /// included, which is what makes the tiny relative hit guard
    /// ([`crate::HIT_EPSILON_REL`]) sufficient. The side is decided by the
    /// geometric normal: with a shading normal, a direction between the two
    /// tangent planes would start on the wrong side and re-hit the surface.
    pub fn spawn(&self, direction: UnitVector) -> Vector {
        self.point + self.offset.copysign(direction.dot(self.geometric)) * self.geometric
    }
}

/// Handle to the sub-geometry a `hit` actually struck. Every leaf returns
/// itself, and the `as_material`/`as_texture` defaults hand back the owning
/// object's components; composite geometry whose material varies across its
/// surface (per-face meshes, CSG, a BVH returning a child leaf) overrides
/// them so the winner supplies its own.
pub trait HitCaster<R: crate::Rng + ?Sized = dyn Rng>: std::fmt::Debug {
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
