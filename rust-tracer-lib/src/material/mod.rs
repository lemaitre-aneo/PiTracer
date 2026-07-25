use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::{Hit, UnitVector};

mod dielectric;
mod lambertian;
mod metal;

pub use dielectric::*;
pub use lambertian::*;
pub use metal::*;

pub trait Material<R: crate::Rng + ?Sized = dyn RngCore>: std::fmt::Debug {
    /// Samples the outgoing direction of a ray scattering at `hit`; `None`
    /// means the ray is absorbed.
    fn scatter(&self, direction: UnitVector, hit: Hit, rng: &mut R) -> Option<UnitVector>;

    /// Solid-angle density of `scatter`, with two contracts attached:
    ///
    /// - `0` (the default) marks a delta/specular BSDF, which both light
    ///   sampling estimators exclude entirely (mixing a delta lobe into pdf
    ///   ratios is ill-defined);
    /// - for pdf > 0 materials the renderer assumes
    ///   f * cos(theta) = attenuation * pdf (true for the cosine-sampled
    ///   Lambertian). NEE's direct term and the mixture reweighting both
    ///   lean on it, so any new non-specular material must keep this
    ///   proportionality or those estimators become biased (the scene-level
    ///   statistical tests do catch violations).
    fn pdf(&self, direction: UnitVector, hit: Hit, scattered: UnitVector) -> f32 {
        _ = (direction, hit, scattered);
        0f32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(from = "Option<definition::Material>", into = "definition::Material")]
pub enum MaterialEnum {
    Lambertian(Lambertian),
    Metal(Metal),
    Dielectric(Dielectric),
}

impl Default for MaterialEnum {
    fn default() -> Self {
        Self::Lambertian(Lambertian {})
    }
}

impl<R: crate::Rng + ?Sized> Material<R> for MaterialEnum {
    fn scatter(&self, direction: UnitVector, hit: Hit, rng: &mut R) -> Option<UnitVector> {
        match self {
            MaterialEnum::Lambertian(lambertian) => lambertian.scatter(direction, hit, rng),
            MaterialEnum::Metal(metal) => metal.scatter(direction, hit, rng),
            MaterialEnum::Dielectric(dielectric) => dielectric.scatter(direction, hit, rng),
        }
    }

    fn pdf(&self, direction: UnitVector, hit: Hit, scattered: UnitVector) -> f32 {
        match self {
            MaterialEnum::Lambertian(lambertian) => {
                Material::<R>::pdf(lambertian, direction, hit, scattered)
            }
            MaterialEnum::Metal(metal) => Material::<R>::pdf(metal, direction, hit, scattered),
            MaterialEnum::Dielectric(dielectric) => {
                Material::<R>::pdf(dielectric, direction, hit, scattered)
            }
        }
    }
}

mod definition {
    use serde::{Deserialize, Serialize};

    use super::{Dielectric, Lambertian, Metal};

    #[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct Empty {}

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "material")]
    pub(super) enum MaterialEnum {
        #[serde(alias = "Diffuse")]
        Lambertian(Lambertian),
        #[serde(alias = "Specular")]
        Metal(Metal),
        #[serde(alias = "Refraction")]
        Dielectric(Dielectric),
    }

    /// A missing `material:` tag falls back to Lambertian; so does an unknown
    /// tag value, silently, via the untagged `Empty`/`Default` arms (the
    /// BACKLOG P0 typo trap: `material: Metall` renders as Lambertian).
    #[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[allow(clippy::enum_variant_names)]
    #[serde(untagged)]
    pub(super) enum Material {
        Empty(Empty),
        Material(MaterialEnum),
        #[default]
        Default,
    }

    impl From<Option<Material>> for super::MaterialEnum {
        fn from(value: Option<Material>) -> Self {
            match value.unwrap_or_default() {
                Material::Material(material) => match material {
                    MaterialEnum::Lambertian(lambertian) => Self::Lambertian(lambertian),
                    MaterialEnum::Metal(metal) => Self::Metal(metal),
                    MaterialEnum::Dielectric(dielectric) => Self::Dielectric(dielectric),
                },
                _ => Self::default(),
            }
        }
    }

    impl From<super::MaterialEnum> for Material {
        fn from(value: super::MaterialEnum) -> Self {
            match value {
                super::MaterialEnum::Lambertian(lambertian) => {
                    Self::Material(MaterialEnum::Lambertian(lambertian))
                }
                super::MaterialEnum::Metal(metal) => Self::Material(MaterialEnum::Metal(metal)),
                super::MaterialEnum::Dielectric(dielectric) => {
                    Self::Material(MaterialEnum::Dielectric(dielectric))
                }
            }
        }
    }
}
