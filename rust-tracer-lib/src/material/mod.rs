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
    fn scatter(&self, direction: UnitVector, hit: Hit, rng: &mut R) -> Option<UnitVector>;
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
