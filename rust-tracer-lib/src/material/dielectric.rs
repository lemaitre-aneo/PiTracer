use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{Hit, Material, UnitVector, select};

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(default)]
pub struct Dielectric {
    refraction_index: f32,
    #[serde(skip)]
    refraction_index_reciprocal: f32,
    #[serde(skip)]
    base_reflectance: f32,
}

impl Default for Dielectric {
    fn default() -> Self {
        Self::new(1.5f32)
    }
}

impl Dielectric {
    pub fn new(refraction_index: f32) -> Self {
        let diff = refraction_index - 1f32;
        let sum = refraction_index + 1f32;
        Dielectric {
            refraction_index,
            refraction_index_reciprocal: 1f32 / refraction_index,
            base_reflectance: diff * diff / (sum * sum),
        }
    }
}

impl<'de> Deserialize<'de> for Dielectric {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let index = {
            #[derive(Default, Deserialize)]
            #[serde(default)]
            struct Dielectric {
                refraction_index: Option<f32>,
            }

            Dielectric::deserialize(deserializer)?.refraction_index.unwrap_or(1.5f32)
        };

        Ok(Dielectric::new(index))
    }
}

impl<R: crate::Rng + ?Sized> Material<R> for Dielectric {
    fn scatter(&self, direction: UnitVector, hit: Hit, rng: &mut R) -> Option<UnitVector> {
        let refraction_factor = select(
            hit.front,
            self.refraction_index_reciprocal,
            self.refraction_index,
        );

        let cost = direction.dot(-hit.normal).min(1f32);
        let sint = (1f32 - cost * cost).sqrt();

        let cannot_refract = refraction_factor * sint > 1f32;

        let reflectance =
            self.base_reflectance + (1f32 - self.base_reflectance) * (1f32 - cost).powi(5);

        if cannot_refract || reflectance > rng.random::<f32>() {
            Some(direction.reflect(hit.normal))
        } else {
            let perp = refraction_factor * (direction + cost * hit.normal);
            let parallel = -(1f32 - perp.norm2()).abs().sqrt() * hit.normal;
            Some((perp + parallel).normalized())
        }
    }
}
