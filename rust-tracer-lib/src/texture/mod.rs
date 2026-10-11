use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::{Color, Hit};

mod flat;
mod image;
mod marble;

pub use self::image::{DecodeError, Image, ImageError, SvgError};
pub use flat::*;
pub use marble::*;

pub trait Texture<R: RngCore + ?Sized = dyn RngCore>: std::fmt::Debug {
    fn get(&self, hit: Hit, rng: &mut R) -> TextureProperty;

    /// Whether the surface can be transparent anywhere: when false,
    /// [`Texture::opacity`] is never called.
    fn transparent(&self) -> bool {
        false
    }

    /// Probability that a ray stops at `hit` rather than crossing the
    /// surface (alpha), in [0, 1]. Applied at intersection (see
    /// [`crate::Object`]), so every ray, shadow rays included, sees through
    /// the transparent parts. A transparent texture must not emit: light
    /// sampling assumes emitters are opaque.
    fn opacity(&self, _hit: &Hit) -> f32 {
        1.0
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TextureProperty {
    pub emission: Color,
    pub attenuation: Color,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "Option<definition::Texture>", into = "definition::Texture")]
pub enum TextureEnum {
    Flat(Flat),
    Marble(Marble),
    Image(Image),
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
            TextureEnum::Image(_) => Color::default(),
        }
    }

    /// Reads the texture's file if it has one (relative to `base`).
    pub fn load(&mut self, base: &std::path::Path) -> Result<(), ImageError> {
        match self {
            TextureEnum::Image(image) => image.load(base),
            TextureEnum::Flat(_) | TextureEnum::Marble(_) => Ok(()),
        }
    }
}

impl<R: crate::Rng + ?Sized> Texture<R> for TextureEnum {
    fn get(&self, hit: Hit, rng: &mut R) -> TextureProperty {
        match self {
            TextureEnum::Flat(flat) => flat.get(hit, rng),
            TextureEnum::Marble(marble) => marble.get(hit, rng),
            TextureEnum::Image(image) => image.get(hit, rng),
        }
    }

    fn transparent(&self) -> bool {
        match self {
            TextureEnum::Image(image) => Texture::<R>::transparent(image),
            TextureEnum::Flat(_) | TextureEnum::Marble(_) => false,
        }
    }

    fn opacity(&self, hit: &Hit) -> f32 {
        match self {
            TextureEnum::Image(image) => Texture::<R>::opacity(image, hit),
            TextureEnum::Flat(_) | TextureEnum::Marble(_) => 1.0,
        }
    }
}

mod definition {
    use serde::{Deserialize, Serialize};

    use super::{Flat, Image, Marble};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "texture", rename_all = "lowercase")]
    pub(super) enum TextureEnum {
        #[serde(alias = "Flat")]
        Flat(Flat),
        #[serde(alias = "Marble")]
        Marble(Marble),
        #[serde(alias = "Image")]
        Image(Image),
    }

    /// A missing `texture:` tag falls back to `Flat` (whose fields all
    /// default), so plain `color:`/`emission:` entries keep parsing. Like the
    /// `material:` tag, an unknown tag value silently takes the fallback too
    /// (BACKLOG P0).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
                Texture::Texture(TextureEnum::Image(image)) => Self::Image(image),
            }
        }
    }

    impl From<super::TextureEnum> for Texture {
        fn from(value: super::TextureEnum) -> Self {
            match value {
                super::TextureEnum::Flat(flat) => Self::Texture(TextureEnum::Flat(flat)),
                super::TextureEnum::Marble(marble) => Self::Texture(TextureEnum::Marble(marble)),
                super::TextureEnum::Image(image) => Self::Texture(TextureEnum::Image(image)),
            }
        }
    }
}
