use serde::{Deserialize, Serialize};

use crate::{Color, Hit, Texture};

use super::TextureProperty;

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Flat {
    pub color: Color,
    pub emission: Color,
}

impl<R: crate::Rng + ?Sized> Texture<R> for Flat {
    fn get(&self, _hit: Hit, _rng: &mut R) -> TextureProperty {
        TextureProperty {
            attenuation: self.color,
            emission: self.emission,
        }
    }
}
