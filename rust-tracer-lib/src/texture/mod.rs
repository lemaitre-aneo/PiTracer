use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::{Color, Hit};

mod flat;
mod marble;

pub use flat::*;
pub use marble::*;

pub trait Texture<R: RngCore + ?Sized = dyn RngCore>: std::fmt::Debug {
    fn get(&self, hit: Hit, rng: &mut R) -> TextureProperty;
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TextureProperty {
    pub emission: Color,
    pub attenuation: Color,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(from = "Option<definition::Texture>", into = "definition::Texture")]
pub enum TextureEnum {
    Flat(Flat),
    Marble(Marble),
}

impl Default for TextureEnum {
    fn default() -> Self {
        Self::Flat(Flat::default())
    }
}

impl TextureEnum {
    /// Constant emission of the texture, used by the scene build to weight
    /// light sampling before the enum is monomorphized away.
    pub fn emission(&self) -> Color {
        match self {
            TextureEnum::Flat(flat) => flat.emission,
            TextureEnum::Marble(marble) => marble.emission,
        }
    }
}

impl<R: crate::Rng + ?Sized> Texture<R> for TextureEnum {
    fn get(&self, hit: Hit, rng: &mut R) -> TextureProperty {
        match self {
            TextureEnum::Flat(flat) => flat.get(hit, rng),
            TextureEnum::Marble(marble) => marble.get(hit, rng),
        }
    }
}

mod definition {
    use serde::{Deserialize, Serialize};

    use super::{Flat, Marble};

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "texture", rename_all = "lowercase")]
    pub(super) enum TextureEnum {
        #[serde(alias = "Flat")]
        Flat(Flat),
        #[serde(alias = "Marble")]
        Marble(Marble),
    }

    /// A missing `texture:` tag falls back to `Flat` (whose fields all
    /// default), so plain `color:`/`emission:` entries keep parsing. Like the
    /// `material:` tag, an unknown tag value silently takes the fallback too
    /// (BACKLOG P0).
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[serde(untagged)]
    pub(super) enum Texture {
        Texture(TextureEnum),
        Flat(Flat),
    }

    impl Default for Texture {
        fn default() -> Self {
            Self::Flat(Flat::default())
        }
    }

    impl From<Option<Texture>> for super::TextureEnum {
        fn from(value: Option<Texture>) -> Self {
            match value.unwrap_or_default() {
                Texture::Texture(TextureEnum::Flat(flat)) | Texture::Flat(flat) => Self::Flat(flat),
                Texture::Texture(TextureEnum::Marble(marble)) => Self::Marble(marble),
            }
        }
    }

    impl From<super::TextureEnum> for Texture {
        fn from(value: super::TextureEnum) -> Self {
            match value {
                super::TextureEnum::Flat(flat) => Self::Texture(TextureEnum::Flat(flat)),
                super::TextureEnum::Marble(marble) => Self::Texture(TextureEnum::Marble(marble)),
            }
        }
    }
}
