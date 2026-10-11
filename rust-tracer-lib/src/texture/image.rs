use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};

use crate::{Color, Hit, Texture};

use super::TextureProperty;

mod definition {
    use std::path::PathBuf;

    use serde::{Deserialize, Serialize};

    /// `file` is what scene files use; `data` (base64 of the encoded file) is
    /// what the scene carries once loaded, so that it travels to ArmoniK
    /// workers inside the scene JSON.
    #[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(default)]
    pub(super) struct Image {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub file: Option<PathBuf>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub data: Option<String>,
    }
}

/// A raster image (PNG or JPEG) mapped onto the surface coordinates: the
/// whole image covers `(u, v)` in [0, 1]^2, `u` left to right and `v` bottom
/// to top, clamped outside (quads map their corner `position` to the image's
/// bottom-left, `u` along their `u` edge and `v` along their `v` edge).
///
/// Texels are sRGB, decoded to linear albedo; there is no emission. Alpha
/// makes the surface transparent: a ray crosses it where the image is
/// transparent, with probability `1 - alpha` (see [`Texture::opacity`]).
/// Filtering is bilinear, on alpha-premultiplied colors.
///
/// The image is read by [`Image::load`] (through
/// [`crate::SceneDefinition::load_textures`]), and carried by the scene from
/// then on. Until loaded, the surface is black and opaque.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "definition::Image", into = "definition::Image")]
pub struct Image {
    file: Option<PathBuf>,
    data: Option<Arc<Encoded>>,
}

/// An image file and its decoded texels.
#[derive(PartialEq)]
struct Encoded {
    bytes: Vec<u8>,
    texels: Texels,
}

impl std::fmt::Debug for Encoded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encoded")
            .field("bytes", &self.bytes.len())
            .field("width", &self.texels.width)
            .field("height", &self.texels.height)
            .finish()
    }
}

/// Straight (not premultiplied) sRGB RGBA, row-major, top row first.
#[derive(PartialEq)]
struct Texels {
    width: usize,
    height: usize,
    rgba: Vec<[u8; 4]>,
    /// Some texel is not fully opaque.
    transparent: bool,
}

/// sRGB-encoded byte to linear value (IEC 61966-2-1).
fn srgb_to_linear(byte: u8) -> f32 {
    let c = f32::from(byte) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

static SRGB_TO_LINEAR: std::sync::LazyLock<[f32; 256]> =
    std::sync::LazyLock::new(|| std::array::from_fn(|i| srgb_to_linear(i as u8)));

impl Texels {
    fn decode(bytes: &[u8]) -> Result<Self, ::image::ImageError> {
        let image = ::image::load_from_memory(bytes)?.into_rgba8();
        let (width, height) = (image.width() as usize, image.height() as usize);
        let rgba: Vec<[u8; 4]> = image.pixels().map(|pixel| pixel.0).collect();
        let transparent = rgba.iter().any(|texel| texel[3] < 255);
        Ok(Self {
            width,
            height,
            rgba,
            transparent,
        })
    }

    /// Bilinear sample at `(u, v)`: linear color premultiplied by alpha, and
    /// alpha.
    fn sample(&self, u: f32, v: f32) -> ([f32; 3], f32) {
        // Texel centers sit at (i + 0.5) / size; clamp to the edge texels.
        let x = (u * self.width as f32 - 0.5).clamp(0.0, (self.width - 1) as f32);
        let y = ((1.0 - v) * self.height as f32 - 0.5).clamp(0.0, (self.height - 1) as f32);
        // Also maps NaN coordinates to texel 0.
        let (x0, y0) = (x as usize, y as usize);
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let lut = &*SRGB_TO_LINEAR;
        let mut color = [0f32; 3];
        let mut alpha = 0f32;
        for (x, y, weight) in [
            (x0, y0, (1.0 - fx) * (1.0 - fy)),
            (x1, y0, fx * (1.0 - fy)),
            (x0, y1, (1.0 - fx) * fy),
            (x1, y1, fx * fy),
        ] {
            let texel = self.rgba[y * self.width + x];
            let a = f32::from(texel[3]) / 255.0 * weight;
            for (channel, &byte) in color.iter_mut().zip(&texel[..3]) {
                *channel += lut[byte as usize] * a;
            }
            alpha += a;
        }
        (color, alpha)
    }
}

impl Image {
    /// An image from the bytes of an image file.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, ImageError> {
        let texels = Texels::decode(&bytes).map_err(ImageError::Decode)?;
        Ok(Self {
            file: None,
            data: Some(Arc::new(Encoded { bytes, texels })),
        })
    }

    /// Reads the image file, relative to `base` (the scene file's folder),
    /// unless the image is already loaded.
    pub fn load(&mut self, base: &Path) -> Result<(), ImageError> {
        if self.data.is_some() {
            return Ok(());
        }
        let file = self.file.as_ref().ok_or(ImageError::NoFile)?;
        let path = base.join(file);
        let bytes = std::fs::read(&path).map_err(|source| ImageError::Read {
            path: path.clone(),
            source,
        })?;
        let texels =
            Texels::decode(&bytes).map_err(|source| ImageError::DecodeFile { path, source })?;
        self.data = Some(Arc::new(Encoded { bytes, texels }));
        Ok(())
    }

    /// Width and height in texels, once loaded.
    pub fn size(&self) -> Option<(usize, usize)> {
        self.data
            .as_ref()
            .map(|data| (data.texels.width, data.texels.height))
    }
}

impl<R: crate::Rng + ?Sized> Texture<R> for Image {
    fn get(&self, hit: Hit, _rng: &mut R) -> TextureProperty {
        let attenuation = match &self.data {
            Some(data) => {
                let ([r, g, b], alpha) = data.texels.sample(hit.u, hit.v);
                // The color of the opaque part: a hit on the surface is one
                // the opacity test kept.
                if alpha > 0.0 {
                    Color {
                        r: r / alpha,
                        g: g / alpha,
                        b: b / alpha,
                    }
                } else {
                    Color::default()
                }
            }
            None => Color::default(),
        };
        TextureProperty {
            attenuation,
            emission: Color::default(),
        }
    }

    fn transparent(&self) -> bool {
        self.data
            .as_ref()
            .is_some_and(|data| data.texels.transparent)
    }

    fn opacity(&self, hit: &Hit) -> f32 {
        match &self.data {
            Some(data) => data.texels.sample(hit.u, hit.v).1,
            None => 1.0,
        }
    }
}

impl TryFrom<definition::Image> for Image {
    type Error = ImageError;

    fn try_from(value: definition::Image) -> Result<Self, Self::Error> {
        match value.data {
            Some(data) => {
                let bytes = BASE64.decode(data).map_err(ImageError::Base64)?;
                Ok(Self {
                    file: value.file,
                    ..Self::from_bytes(bytes)?
                })
            }
            None => Ok(Self {
                file: value.file,
                data: None,
            }),
        }
    }
}

impl From<Image> for definition::Image {
    fn from(value: Image) -> Self {
        match value.data {
            // Loaded: the data travels, the path would not resolve elsewhere.
            Some(data) => Self {
                file: None,
                data: Some(BASE64.encode(&data.bytes)),
            },
            None => Self {
                file: value.file,
                data: None,
            },
        }
    }
}

#[derive(Debug)]
pub enum ImageError {
    /// An image texture without `file`.
    NoFile,
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    DecodeFile {
        path: PathBuf,
        source: ::image::ImageError,
    },
    Decode(::image::ImageError),
    Base64(base64::DecodeError),
}

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageError::NoFile => write!(f, "an image texture needs a `file`"),
            ImageError::Read { path, source } => {
                write!(f, "could not read the image {}: {source}", path.display())
            }
            ImageError::DecodeFile { path, source } => {
                write!(f, "could not decode the image {}: {source}", path.display())
            }
            ImageError::Decode(source) => write!(f, "could not decode the image: {source}"),
            ImageError::Base64(source) => write!(f, "invalid image data: {source}"),
        }
    }
}

impl std::error::Error for ImageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ImageError::NoFile => None,
            ImageError::Read { source, .. } => Some(source),
            ImageError::DecodeFile { source, .. } | ImageError::Decode(source) => Some(source),
            ImageError::Base64(source) => Some(source),
        }
    }
}

#[cfg(test)]
mod tests;
