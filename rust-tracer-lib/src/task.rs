//! Wire format and convergence logic of the distributed (ArmoniK) rendering
//! tasks, shared by the worker and the client.
//!
//! A render session uploads one scene ([`encode_scene`]), shared by every
//! task as a data dependency named by the [`SCENE_ID_OPTION`] task option.
//! Each block of the image is refined by a chain of tasks, one pass of
//! [`BlockPayload::samples`] samples per pixel each:
//!
//! - the client submits the first task of each block, expecting the block's
//!   two final results ([`BlockOutputs`]): the linear radiance
//!   ([`BlockResult`], f32) and the display image ([`DisplayBlock`], u8,
//!   gamma applied);
//! - a task merges its pass into the accumulation of the previous passes
//!   ([`accumulate`]); converged, it sends both final results; otherwise it
//!   creates the new accumulation (`linear-x-y`) and its display image
//!   (`display-x-y`), and delegates the final results to a follow-up task
//!   that takes the accumulation as input ([`BlockPayload::next`]).
//!
//! The client therefore knows from the start which results end the render,
//! and follows the progress through the intermediate display images, found
//! by name ([`DISPLAY_NAME`]) and self-describing (block, sample count).
//!
//! Coordinates are in display space: `(0, 0)` is the top-left pixel and
//! `y` grows downwards (the camera's own pixel rows are bottom-up; the flip
//! happens in [`render_block_row`]).

use std::collections::HashMap;

use rand::SeedableRng;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};

use crate::{Camera, Color, Scene, SceneDefinition};

/// Task option naming the result that holds the session's scene; every
/// task lists that result among its data dependencies.
pub const SCENE_ID_OPTION: &str = "sceneId";
/// Task option: a block is converged once the estimated mean squared error
/// of its display image (8-bit levels, see [`estimated_display_error`]) is
/// at most this value.
pub const ERROR_THRESHOLD_OPTION: &str = "errorMetricThreshold";
/// Task option: a block is final once it accumulates this many samples per
/// pixel, converged or not.
pub const MAX_SAMPLES_OPTION: &str = "maxSamples";

/// Result name prefixes; a result name is `<prefix>-<x>-<y>` ([`name`]).
pub const PAYLOAD_NAME: &str = "payload";
/// Intermediate accumulation of a block.
pub const LINEAR_NAME: &str = "linear";
/// Display images, intermediate (`display-x-y`) and final
/// (`display-final-x-y`): every name starting with `display-`.
pub const DISPLAY_NAME: &str = "display";
pub const FINAL_LINEAR_NAME: &str = "linear-final";
pub const FINAL_DISPLAY_NAME: &str = "display-final";

/// Name of the `prefix` result of `block`.
pub fn name(prefix: &str, block: &Block) -> String {
    format!("{prefix}-{}-{}", block.x, block.y)
}

/// Rectangle of pixels, in display space (`y = 0` is the top row).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Block {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Block {
    /// Number of pixels in the block.
    pub fn len(&self) -> usize {
        self.width as usize * self.height as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Tiles a `width` x `height` image with blocks of at most
/// `block_width` x `block_height` pixels (the last row and column are
/// clipped to the image), ordered from the image center outwards so the
/// middle of the picture shows up first.
pub fn tile_grid(width: u32, height: u32, block_width: u32, block_height: u32) -> Vec<Block> {
    let mut blocks: Vec<_> = (0..height)
        .step_by(block_height as usize)
        .flat_map(|y| {
            (0..width)
                .step_by(block_width as usize)
                .map(move |x| Block {
                    x,
                    y,
                    width: block_width.min(width - x),
                    height: block_height.min(height - y),
                })
        })
        .collect();
    // Twice the distance between centers, to stay in integers.
    let distance2 = |block: &Block| {
        let dx = (2 * block.x + block.width) as i64 - width as i64;
        let dy = (2 * block.y + block.height) as i64 - height as i64;
        dx * dx + dy * dy
    };
    blocks.sort_by_key(distance2);
    blocks
}

/// Ids of the two final results of a block, expected by every task of its
/// chain (each delegating them to the next one).
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockOutputs {
    /// Final [`BlockResult`].
    pub linear: String,
    /// Final [`DisplayBlock`].
    pub display: String,
}

/// Payload of one rendering task: one pass over a block.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockPayload {
    pub block: Block,
    /// Samples per pixel of the pass (overrides the scene's `samples`).
    pub samples: u32,
    /// Seed of the pass's random streams. Two passes over the same block
    /// must use different seeds, or they compute the exact same samples
    /// (different blocks get independent streams even with equal seeds).
    pub seed: u64,
    pub outputs: BlockOutputs,
    /// Accumulation of the previous passes ([`BlockResult`], a data
    /// dependency of the task); absent on the first pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous: Option<String>,
}

impl BlockPayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("a block payload always serializes")
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, WireError> {
        Ok(serde_json::from_slice(bytes)?)
    }

    /// Checks that the payload can be rendered through `camera`.
    pub fn validate(&self, camera: &Camera) -> Result<(), WireError> {
        if self.samples == 0 {
            return Err(WireError::NoSamples);
        }
        let Block {
            x,
            y,
            width,
            height,
        } = self.block;
        let fits = |start: u32, extent: u32, size: u32| {
            start.checked_add(extent).is_some_and(|end| end <= size)
        };
        if fits(x, width, camera.width) && fits(y, height, camera.height) {
            Ok(())
        } else {
            Err(WireError::OutOfImage {
                block: self.block,
                width: camera.width,
                height: camera.height,
            })
        }
    }

    /// Payload of the follow-up pass, continuing from the accumulation
    /// `previous` with a fresh seed.
    pub fn next(&self, previous: String) -> Self {
        Self {
            seed: splitmix64(self.seed),
            previous: Some(previous),
            ..self.clone()
        }
    }
}

/// When a block is done, read from the task options (with defaults).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Convergence {
    /// See [`ERROR_THRESHOLD_OPTION`].
    pub error_threshold: f32,
    /// See [`MAX_SAMPLES_OPTION`].
    pub max_samples: u32,
}

impl Default for Convergence {
    fn default() -> Self {
        Self {
            // Fast enough for a demo: about 7 levels of RMS error left, and
            // at most 20 passes of 50 samples.
            error_threshold: 50.0,
            max_samples: 1_000,
        }
    }
}

impl Convergence {
    /// Reads the options, defaulting the missing ones.
    pub fn from_options(options: &HashMap<String, String>) -> Result<Self, WireError> {
        fn parse<T: std::str::FromStr>(
            options: &HashMap<String, String>,
            name: &'static str,
            default: T,
        ) -> Result<T, WireError> {
            match options.get(name) {
                None => Ok(default),
                Some(value) => value.parse().map_err(|_| WireError::InvalidOption {
                    name,
                    value: value.clone(),
                }),
            }
        }
        let default = Self::default();
        let convergence = Self {
            error_threshold: parse(options, ERROR_THRESHOLD_OPTION, default.error_threshold)?,
            max_samples: parse(options, MAX_SAMPLES_OPTION, default.max_samples)?,
        };
        if convergence.error_threshold.is_nan() || convergence.error_threshold < 0.0 {
            return Err(WireError::InvalidOption {
                name: ERROR_THRESHOLD_OPTION,
                value: convergence.error_threshold.to_string(),
            });
        }
        Ok(convergence)
    }

    /// Writes the options [`Convergence::from_options`] reads.
    pub fn to_options(&self, options: &mut HashMap<String, String>) {
        options.insert(
            ERROR_THRESHOLD_OPTION.into(),
            self.error_threshold.to_string(),
        );
        options.insert(MAX_SAMPLES_OPTION.into(), self.max_samples.to_string());
    }
}

/// Little-endian header shared by the binary results: a 4-byte format tag,
/// then the block (x, y, width, height) and the sample count as u32.
const HEADER_LEN: usize = 4 + 5 * 4;

fn encode_header(magic: [u8; 4], block: &Block, samples: u32, pixel_len: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER_LEN + block.len() * pixel_len);
    bytes.extend_from_slice(&magic);
    for value in [block.x, block.y, block.width, block.height, samples] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Splits `bytes` into block, sample count and pixel data, checking the tag
/// and that the data holds exactly one `pixel_len` record per pixel.
fn decode_header(
    bytes: &[u8],
    magic: [u8; 4],
    pixel_len: usize,
) -> Result<(Block, u32, &[u8]), WireError> {
    let (header, data) = bytes
        .split_at_checked(HEADER_LEN)
        .ok_or(WireError::Length {
            expected: HEADER_LEN,
            actual: bytes.len(),
        })?;
    let (tag, fields) = header.split_at(magic.len());
    if tag != magic {
        return Err(WireError::BadMagic);
    }
    let mut fields = fields
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()));
    let mut field = || fields.next().unwrap();
    let block = Block {
        x: field(),
        y: field(),
        width: field(),
        height: field(),
    };
    let samples = field();

    let expected = block.len() * pixel_len;
    if data.len() != expected {
        return Err(WireError::Length {
            expected: HEADER_LEN + expected,
            actual: bytes.len(),
        });
    }
    Ok((block, samples, data))
}

/// Mean linear radiance of each pixel of `block` over `samples` samples,
/// row-major, top row first.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct BlockResult {
    pub block: Block,
    pub samples: u32,
    pub pixels: Vec<Color>,
}

impl BlockResult {
    /// Format tag and version; bump it on any layout change.
    const MAGIC: [u8; 4] = *b"PTB1";
    const PIXEL_LEN: usize = 3 * 4;

    /// The common header, then the pixels as r, g, b f32 triplets.
    pub fn encode(&self) -> Vec<u8> {
        assert_eq!(
            self.pixels.len(),
            self.block.len(),
            "pixel count does not match the block size"
        );
        let mut bytes = encode_header(Self::MAGIC, &self.block, self.samples, Self::PIXEL_LEN);
        for pixel in &self.pixels {
            for value in [pixel.r, pixel.g, pixel.b] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, WireError> {
        let (block, samples, data) = decode_header(bytes, Self::MAGIC, Self::PIXEL_LEN)?;
        let pixels = data
            .chunks_exact(Self::PIXEL_LEN)
            .map(|pixel| {
                let mut values = pixel
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()));
                let mut value = || values.next().unwrap();
                Color {
                    r: value(),
                    g: value(),
                    b: value(),
                }
            })
            .collect();
        Ok(Self {
            block,
            samples,
            pixels,
        })
    }

    /// Whether every pixel is exactly black. Radiance is never negative, so
    /// a zero mean means every sample was zero: the block is not noisy, it
    /// is genuinely black.
    pub fn is_black(&self) -> bool {
        self.pixels.iter().all(|pixel| *pixel == Color::default())
    }

    /// The mean over the samples of both results (of the same block).
    pub fn merge(&self, other: &BlockResult) -> BlockResult {
        let samples = self.samples + other.samples;
        let weight = other.samples as f32 / samples as f32;
        BlockResult {
            block: self.block,
            samples,
            pixels: self
                .pixels
                .iter()
                .zip(&other.pixels)
                .map(|(&mine, &theirs)| mine + (theirs - mine) * weight)
                .collect(),
        }
    }

    /// The display image, with the scene's gamma.
    pub fn to_display(&self, scene: &Scene) -> DisplayBlock {
        DisplayBlock {
            block: self.block,
            samples: self.samples,
            pixels: self
                .pixels
                .iter()
                .map(|&pixel| scene.to_u8_color(pixel))
                .collect(),
        }
    }
}

/// Display image of a block (gamma applied, 8 bits per channel), row-major,
/// top row first, with the sample count it was made from.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DisplayBlock {
    pub block: Block,
    pub samples: u32,
    pub pixels: Vec<[u8; 3]>,
}

impl DisplayBlock {
    /// Format tag and version; bump it on any layout change.
    const MAGIC: [u8; 4] = *b"PTD1";
    const PIXEL_LEN: usize = 3;

    /// The common header, then the pixels as r, g, b bytes.
    pub fn encode(&self) -> Vec<u8> {
        assert_eq!(
            self.pixels.len(),
            self.block.len(),
            "pixel count does not match the block size"
        );
        let mut bytes = encode_header(Self::MAGIC, &self.block, self.samples, Self::PIXEL_LEN);
        bytes.extend(self.pixels.iter().flatten());
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, WireError> {
        let (block, samples, data) = decode_header(bytes, Self::MAGIC, Self::PIXEL_LEN)?;
        Ok(Self {
            block,
            samples,
            pixels: data
                .chunks_exact(Self::PIXEL_LEN)
                .map(|pixel| pixel.try_into().unwrap())
                .collect(),
        })
    }
}

/// Estimated mean squared error of `merged`'s display image (continuous
/// 8-bit levels, before rounding), from the change the last pass of
/// `pass_samples` samples made to it, starting from `previous`.
///
/// With `previous` the mean of n' samples and `merged` the mean of
/// n = n' + m, the expected squared change is `m * var / (n * n')` while the
/// error left in `merged` is `var / n`: the change underestimates the error
/// by `m / n'`, a factor that keeps shrinking, so a fixed threshold on the
/// raw change would stop at an error depending on the pass size and the
/// noise level. Hence the `n' / m` rescaling. Display levels are a
/// nonlinear function of the radiance; for the small changes of a
/// converging block the relation holds to first order.
pub fn estimated_display_error(
    scene: &Scene,
    previous: &BlockResult,
    merged: &BlockResult,
    pass_samples: u32,
) -> f32 {
    let channels = |color: Color| [color.r, color.g, color.b];
    let mut sum = 0f64;
    for (&before, &after) in previous.pixels.iter().zip(&merged.pixels) {
        for (before, after) in channels(before).into_iter().zip(channels(after)) {
            let change = scene.display_level(after) - scene.display_level(before);
            sum += (change as f64).powi(2);
        }
    }
    let values = (3 * merged.pixels.len()).max(1) as f64;
    (sum / values * previous.samples as f64 / pass_samples as f64) as f32
}

/// The accumulation of a block after a pass, and whether it is final.
#[derive(Debug, Clone, PartialEq)]
pub struct Accumulation {
    pub result: BlockResult,
    /// [`estimated_display_error`], when there was a previous accumulation
    /// to estimate it from.
    pub error: Option<f32>,
    /// Converged, black, or out of samples.
    pub done: bool,
}

/// Merges a pass into the previous accumulation of its block (if any) and
/// decides whether the block is done. A first pass cannot estimate its own
/// error, so it is only final when black (or already at the sample cap).
pub fn accumulate(
    scene: &Scene,
    previous: Option<BlockResult>,
    pass: BlockResult,
    convergence: &Convergence,
) -> Result<Accumulation, WireError> {
    let (result, error) = match previous {
        None => (pass, None),
        Some(previous) => {
            if previous.block != pass.block {
                return Err(WireError::BlockMismatch {
                    expected: pass.block,
                    actual: previous.block,
                });
            }
            let merged = previous.merge(&pass);
            let error = estimated_display_error(scene, &previous, &merged, pass.samples);
            (merged, Some(error))
        }
    };
    let done = result.samples >= convergence.max_samples
        || match error {
            None => result.is_black(),
            Some(error) => error <= convergence.error_threshold,
        };
    Ok(Accumulation {
        result,
        error,
        done,
    })
}

/// Encodes the shared scene of a render session.
pub fn encode_scene(scene: &SceneDefinition) -> Vec<u8> {
    serde_json::to_vec(scene).expect("a scene definition always serializes")
}

pub fn decode_scene(bytes: &[u8]) -> Result<SceneDefinition, WireError> {
    Ok(serde_json::from_slice(bytes)?)
}

/// Renders row `row` of the payload's block (`0` = the block's top row).
///
/// Each row owns a random stream seeded from the payload seed and the row's
/// image position, so rows can be rendered in any order, on any thread, and
/// still produce the same pixels. The payload must have been validated
/// against the scene's camera ([`BlockPayload::validate`]).
pub fn render_block_row(scene: &Scene, payload: &BlockPayload, row: u32) -> Vec<Color> {
    let Block { x, y, width, .. } = payload.block;
    let y = y + row;
    // Camera pixel rows are bottom-up.
    let camera_y = scene.camera.height - 1 - y;
    let mut rng = SmallRng::seed_from_u64(row_seed(payload.seed, x, y));
    (x..x + width)
        .map(|x| scene.pixel_radiance_with(x, camera_y, payload.samples, &mut rng))
        .collect()
}

/// Seed of the random stream of the row starting at `(x, y)`: distinct
/// positions under one seed never collide, and different seeds collide only
/// by a 64-bit hash accident.
fn row_seed(seed: u64, x: u32, y: u32) -> u64 {
    splitmix64(splitmix64(seed) ^ ((x as u64) << 32 | y as u64))
}

/// SplitMix64 finalizer (Steele, Lea, Flood 2014): a bijective mix of all
/// input bits.
fn splitmix64(z: u64) -> u64 {
    let z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Debug)]
pub enum WireError {
    Json(serde_json::Error),
    BadMagic,
    Length {
        expected: usize,
        actual: usize,
    },
    NoSamples,
    OutOfImage {
        block: Block,
        width: u32,
        height: u32,
    },
    BlockMismatch {
        expected: Block,
        actual: Block,
    },
    InvalidOption {
        name: &'static str,
        value: String,
    },
    /// The inputs of a denoising task do not cover its tile exactly once,
    /// or lack the block being denoised.
    IncompleteTile {
        tile: Block,
    },
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireError::Json(err) => write!(f, "invalid JSON: {err}"),
            WireError::BadMagic => write!(f, "not a result of the expected format (bad magic)"),
            WireError::Length { expected, actual } => {
                write!(f, "expected {expected} bytes, got {actual}")
            }
            WireError::NoSamples => write!(f, "a block needs at least one sample per pixel"),
            WireError::OutOfImage {
                block,
                width,
                height,
            } => write!(f, "block {block:?} exceeds the {width}x{height} image"),
            WireError::BlockMismatch { expected, actual } => {
                write!(
                    f,
                    "expected a result for block {expected:?}, got {actual:?}"
                )
            }
            WireError::InvalidOption { name, value } => {
                write!(f, "invalid task option {name}={value:?}")
            }
            WireError::IncompleteTile { tile } => write!(
                f,
                "the inputs do not cover the denoising tile {tile:?} exactly once, block included"
            ),
        }
    }
}

impl std::error::Error for WireError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WireError::Json(err) => Some(err),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for WireError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

#[cfg(test)]
mod tests;
