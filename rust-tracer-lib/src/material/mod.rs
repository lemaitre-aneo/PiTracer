use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::{Hit, UnitVector};

mod dielectric;
mod lambertian;
mod specular;

pub use dielectric::*;
pub use lambertian::*;
pub use specular::*;

pub trait Material<R: crate::Rng + ?Sized = dyn RngCore>: std::fmt::Debug {
    fn scatter(&self, direction: UnitVector, hit: Hit, rng: &mut R) -> Option<UnitVector>;
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(from = "Option<definition::Material>", into = "definition::Material")]
pub enum MaterialEnum {
    Lambertian(Lambertian),
    Specular(Specular),
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
            MaterialEnum::Specular(specular) => specular.scatter(direction, hit, rng),
            MaterialEnum::Dielectric(dielectric) => dielectric.scatter(direction, hit, rng),
        }
    }
}

mod definition {
    use serde::{Deserialize, Serialize};

    use super::{Dielectric, Lambertian, Specular};

    #[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct Empty {}

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "reflexivity")]
    pub(super) enum ReflexivityEnum {
        #[serde(alias = "Diffuse")]
        Lambertian(Lambertian),
        Specular(Specular),
        #[serde(alias = "Refraction")]
        Dielectric(Dielectric),
    }
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "material")]
    pub(super) enum MaterialEnum {
        #[serde(alias = "Diffuse")]
        Lambertian(Lambertian),
        Specular(Specular),
        #[serde(alias = "Refraction")]
        Dielectric(Dielectric),
    }

    #[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[allow(clippy::enum_variant_names)]
    #[serde(untagged)]
    pub(super) enum Material {
        Empty(Empty),
        Material(MaterialEnum),
        Reflexivity(ReflexivityEnum),
        #[default]
        Default,
    }

    impl From<Option<Material>> for super::MaterialEnum {
        fn from(value: Option<Material>) -> Self {
            match value.unwrap_or_default() {
                Material::Material(material) => match material {
                    MaterialEnum::Lambertian(lambertian) => Self::Lambertian(lambertian),
                    MaterialEnum::Specular(specular) => Self::Specular(specular),
                    MaterialEnum::Dielectric(dielectric) => Self::Dielectric(dielectric),
                },
                Material::Reflexivity(reflexivity) => match reflexivity {
                    ReflexivityEnum::Lambertian(lambertian) => Self::Lambertian(lambertian),
                    ReflexivityEnum::Specular(specular) => Self::Specular(specular),
                    ReflexivityEnum::Dielectric(dielectric) => Self::Dielectric(dielectric),
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
                super::MaterialEnum::Specular(specular) => {
                    Self::Material(MaterialEnum::Specular(specular))
                }
                super::MaterialEnum::Dielectric(dielectric) => {
                    Self::Material(MaterialEnum::Dielectric(dielectric))
                }
            }
        }
    }
}
