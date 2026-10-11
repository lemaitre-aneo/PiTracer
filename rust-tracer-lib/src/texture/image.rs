use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};

use crate::{Color, Hit, Texture};

use super::TextureProperty;

mod svg;
#[cfg(feature = "svg-text")]
mod text;

use svg::Svg;
pub use svg::SvgError;

mod definition {
    use std::path::PathBuf;

    use serde::{Deserialize, Serialize};

    /// `file` (with `fonts` and `system_fonts` for SVG text) is what scene
    /// files use; `data` (base64 of the encoded file, text turned into
    /// paths) is what the scene carries once loaded, so that it travels to
    /// ArmoniK workers inside the scene JSON.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(default)]
    pub(super) struct Image {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub file: Option<PathBuf>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        pub fonts: Vec<PathBuf>,
        #[serde(skip_serializing_if = "std::clone::Clone::clone")]
        pub system_fonts: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub data: Option<String>,
    }

    impl Default for Image {
        fn default() -> Self {
            Self {
                file: None,
                fonts: Vec::new(),
                system_fonts: true,
                data: None,
            }
        }
    }
}

/// An image mapped onto the surface coordinates: the whole image covers
/// `(u, v)` in [0, 1]^2, `u` left to right and `v` bottom to top (quads map
/// their corner `position` to the image's bottom-left, `u` along their `u`
/// edge and `v` along their `v` edge).
///
/// The file is a raster image (PNG or JPEG) or an SVG, told apart by its
/// content. A raster image is filtered bilinearly (on alpha-premultiplied
/// colors) and clamped to its edge texels outside [0, 1]^2. An SVG is
/// evaluated exactly from its curves (see the `svg` module), and transparent
/// outside its canvas.
///
/// Colors are sRGB, decoded to linear albedo; there is no emission. Alpha
/// makes the surface transparent: a ray crosses it where the image is
/// transparent, with probability `1 - alpha` (see [`Texture::opacity`]).
///
/// Text in an SVG needs the `svg-text` feature: it is laid out with the
/// font files of `fonts` (relative to the scene file) and, unless
/// `system_fonts` is false, the system's fonts; generic families
/// (`sans-serif`, ...) resolve to the first listed font. A font or character
/// no font provides is an error.
///
/// The image is read by [`Image::load`] (through
/// [`crate::SceneDefinition::load_textures`]), and carried by the scene from
/// then on, its text turned into paths. Until loaded, the surface is black
/// and opaque.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "definition::Image", into = "definition::Image")]
pub struct Image {
    file: Option<PathBuf>,
    fonts: Vec<PathBuf>,
    system_fonts: bool,
    data: Option<Arc<Encoded>>,
}

impl Default for Image {
    fn default() -> Self {
        Self {
            file: None,
            fonts: Vec::new(),
            system_fonts: true,
            data: None,
        }
    }
}

/// An image file and its decoded picture.
struct Encoded {
    /// For an SVG, text has been turned into paths.
    bytes: Vec<u8>,
    picture: Picture,
}

impl Encoded {
    /// An SVG is first made self-contained (text turned into paths, image
    /// files embedded) with its `sources`.
    fn decode(bytes: Vec<u8>, sources: &svg::Sources) -> Result<Self, DecodeError> {
        if ::image::guess_format(&bytes).is_ok() {
            let texels = Texels::decode(&bytes).map_err(DecodeError::Raster)?;
            return Ok(Self {
                bytes,
                picture: Picture::Raster(texels),
            });
        }
        let bytes = if svg::has_text(&bytes) || svg::has_external_images(&bytes) {
            svg::self_contained(&bytes, sources).map_err(DecodeError::Svg)?
        } else {
            bytes
        };
        let svg = Svg::parse(&bytes).map_err(DecodeError::Svg)?;
        Ok(Self {
            bytes,
            picture: Picture::Vector(svg),
        })
    }
}

/// The picture is decoded from the bytes.
impl PartialEq for Encoded {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}

impl std::fmt::Debug for Encoded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encoded")
            .field("bytes", &self.bytes.len())
            .field("size", &self.picture.size())
            .finish()
    }
}

enum Picture {
    Raster(Texels),
    Vector(Svg),
}

impl Picture {
    /// Linear color premultiplied by alpha, and alpha.
    fn sample(&self, u: f32, v: f32) -> ([f32; 3], f32) {
        match self {
            Picture::Raster(texels) => texels.sample(u, v),
            Picture::Vector(svg) => svg.sample(u, v),
        }
    }

    fn transparent(&self) -> bool {
        match self {
            Picture::Raster(texels) => texels.transparent,
            // At least outside its canvas.
            Picture::Vector(_) => true,
        }
    }

    fn size(&self) -> (f64, f64) {
        match self {
            Picture::Raster(texels) => (texels.width as f64, texels.height as f64),
            Picture::Vector(svg) => svg.size(),
        }
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

/// sRGB-encoded value to linear value, in [0, 1] (IEC 61966-2-1).
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

static SRGB_TO_LINEAR: std::sync::LazyLock<[f32; 256]> =
    std::sync::LazyLock::new(|| std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0)));

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
        let lut = &*SRGB_TO_LINEAR;
        self.filter(
            u * self.width as f32,
            (1.0 - v) * self.height as f32,
            true,
            |byte| lut[byte as usize],
        )
    }

    /// Sample at texel coordinates `(x, y)` (x right, y down, texel i
    /// covering [i, i + 1)), bilinear if `smooth` and nearest otherwise,
    /// clamped to the edge texels: color premultiplied by alpha (each byte
    /// mapped by `channel`), and alpha.
    fn filter(&self, x: f32, y: f32, smooth: bool, channel: impl Fn(u8) -> f32) -> ([f32; 3], f32) {
        // Texel centers sit at i + 0.5.
        let (x, y) = if smooth {
            (x - 0.5, y - 0.5)
        } else {
            (x.floor(), y.floor())
        };
        let x = x.clamp(0.0, (self.width - 1) as f32);
        let y = y.clamp(0.0, (self.height - 1) as f32);
        // Also maps NaN coordinates to texel 0.
        let (x0, y0) = (x as usize, y as usize);
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
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
            for (value, &byte) in color.iter_mut().zip(&texel[..3]) {
                *value += channel(byte) * a;
            }
            alpha += a;
        }
        (color, alpha)
    }
}

impl Image {
    /// An image from the bytes of an image file (SVG text uses the system's
    /// fonts, and SVG images referring to files are errors).
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, ImageError> {
        let sources = svg::Sources {
            folder: None,
            fonts: &[],
            system_fonts: true,
        };
        let encoded = Encoded::decode(bytes, &sources).map_err(ImageError::Decode)?;
        Ok(Self {
            data: Some(Arc::new(encoded)),
            ..Self::default()
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
        let fonts: Vec<PathBuf> = self.fonts.iter().map(|font| base.join(font)).collect();
        let sources = svg::Sources {
            folder: path.parent(),
            fonts: &fonts,
            system_fonts: self.system_fonts,
        };
        let encoded =
            Encoded::decode(bytes, &sources).map_err(|source| ImageError::DecodeFile {
                path: path.clone(),
                source,
            })?;
        self.data = Some(Arc::new(encoded));
        Ok(())
    }

    /// Width and height once loaded: in texels for a raster image, in user
    /// units for an SVG.
    pub fn size(&self) -> Option<(f64, f64)> {
        self.data.as_ref().map(|data| data.picture.size())
    }
}

impl<R: crate::Rng + ?Sized> Texture<R> for Image {
    fn get(&self, hit: Hit, _rng: &mut R) -> TextureProperty {
        let attenuation = match &self.data {
            Some(data) => {
                let ([r, g, b], alpha) = data.picture.sample(hit.u, hit.v);
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
            .is_some_and(|data| data.picture.transparent())
    }

    fn opacity(&self, hit: &Hit) -> f32 {
        match &self.data {
            Some(data) => data.picture.sample(hit.u, hit.v).1,
            None => 1.0,
        }
    }
}

impl TryFrom<definition::Image> for Image {
    type Error = ImageError;

    fn try_from(value: definition::Image) -> Result<Self, Self::Error> {
        let data = match value.data {
            Some(data) => {
                let bytes = BASE64.decode(data).map_err(ImageError::Base64)?;
                Self::from_bytes(bytes)?.data
            }
            None => None,
        };
        Ok(Self {
            file: value.file,
            fonts: value.fonts,
            system_fonts: value.system_fonts,
            data,
        })
    }
}

impl From<Image> for definition::Image {
    fn from(value: Image) -> Self {
        match value.data {
            // Loaded: the data travels, the path would not resolve elsewhere.
            Some(data) => Self {
                data: Some(BASE64.encode(&data.bytes)),
                ..Self::default()
            },
            None => Self {
                file: value.file,
                fonts: value.fonts,
                system_fonts: value.system_fonts,
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
        source: DecodeError,
    },
    Decode(DecodeError),
    Base64(base64::DecodeError),
}

/// Why image bytes could not be decoded.
#[derive(Debug)]
pub enum DecodeError {
    Raster(::image::ImageError),
    /// Not a raster image, and not a usable SVG either.
    Svg(SvgError),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::Raster(source) => source.fmt(f),
            DecodeError::Svg(source) => source.fmt(f),
        }
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DecodeError::Raster(source) => source.source(),
            DecodeError::Svg(source) => source.source(),
        }
    }
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
