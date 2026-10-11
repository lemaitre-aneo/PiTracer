//! Wire format of the distributed (ArmoniK) rendering tasks, shared by the
//! worker and the client.
//!
//! A render session uploads one scene ([`encode_scene`]), shared by every
//! task as a data dependency. Each task renders one rectangular [`Block`] of
//! the image: its payload is a [`BlockPayload`] (block, samples per pixel,
//! seed) and its single output is a [`BlockResult`] holding the block's
//! linear radiance in f32 (gamma is left to the client) together with the
//! sample count it averages, so results of successive passes over the same
//! block can be merged by weight.
//!
//! Coordinates are in display space: `(0, 0)` is the top-left pixel and
//! `y` grows downwards (the camera's own pixel rows are bottom-up; the flip
//! happens in [`render_block_row`]).

use rand::SeedableRng;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};

use crate::{Camera, Color, Scene, SceneDefinition};

/// Task option naming the result that holds the session's scene; every
/// task lists that result among its data dependencies.
pub const SCENE_ID_OPTION: &str = "sceneId";

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

/// Payload of one rendering task.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockPayload {
    pub block: Block,
    /// Samples per pixel (overrides the scene's `samples`).
    pub samples: u32,
    /// Seed of the task's random streams. Two tasks rendering the same block
    /// must use different seeds, or they compute the exact same samples
    /// (different blocks get independent streams even with equal seeds).
    pub seed: u64,
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
}

/// Output of one rendering task: the mean linear radiance of each pixel of
/// `block` over `samples` samples, row-major, top row first.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct BlockResult {
    pub block: Block,
    pub samples: u32,
    pub pixels: Vec<Color>,
}

impl BlockResult {
    /// Format tag and version; bump it on any layout change.
    const MAGIC: [u8; 4] = *b"PTB1";
    /// Magic, then x, y, width, height and samples as u32.
    const HEADER_LEN: usize = 4 + 5 * 4;
    const PIXEL_LEN: usize = 3 * 4;

    /// Little-endian binary layout: the [`Self::MAGIC`] tag, the block
    /// (x, y, width, height) and the sample count as u32, then the pixels as
    /// r, g, b f32 triplets.
    pub fn encode(&self) -> Vec<u8> {
        assert_eq!(
            self.pixels.len(),
            self.block.len(),
            "pixel count does not match the block size"
        );
        let mut bytes = Vec::with_capacity(Self::HEADER_LEN + self.pixels.len() * Self::PIXEL_LEN);
        bytes.extend_from_slice(&Self::MAGIC);
        for value in [
            self.block.x,
            self.block.y,
            self.block.width,
            self.block.height,
            self.samples,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for pixel in &self.pixels {
            for value in [pixel.r, pixel.g, pixel.b] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, WireError> {
        let (header, data) = bytes
            .split_at_checked(Self::HEADER_LEN)
            .ok_or(WireError::Length {
                expected: Self::HEADER_LEN,
                actual: bytes.len(),
            })?;
        let (magic, fields) = header.split_at(Self::MAGIC.len());
        if magic != Self::MAGIC {
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

        let expected = block.len() * Self::PIXEL_LEN;
        if data.len() != expected {
            return Err(WireError::Length {
                expected: Self::HEADER_LEN + expected,
                actual: bytes.len(),
            });
        }
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
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireError::Json(err) => write!(f, "invalid JSON: {err}"),
            WireError::BadMagic => write!(f, "not a block result (bad magic)"),
            WireError::Length { expected, actual } => {
                write!(f, "expected {expected} bytes, got {actual}")
            }
            WireError::NoSamples => write!(f, "a block needs at least one sample per pixel"),
            WireError::OutOfImage {
                block,
                width,
                height,
            } => write!(f, "block {block:?} exceeds the {width}x{height} image"),
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
mod tests {
    use super::*;

    const BLOCK: Block = Block {
        x: 2,
        y: 1,
        width: 3,
        height: 2,
    };

    /// Camera at the origin looking down -Z, with an emissive quad covering
    /// the upper half of the view (y > 0) and nothing below.
    fn half_lit_scene(width: u32, height: u32) -> Scene {
        let json = format!(
            r#"{{
                "camera": {{
                    "origin": [0, 0, 0],
                    "direction": [0, 0, -1],
                    "field_of_view": 90,
                    "width": {width},
                    "height": {height}
                }},
                "objects": [{{
                    "type": "quad",
                    "position": [-100, 0, -10],
                    "u": [200, 0, 0],
                    "v": [0, 100, 0],
                    "emission": [1, 2, 3]
                }}]
            }}"#
        );
        decode_scene(json.as_bytes()).unwrap().into()
    }

    fn payload(block: Block, seed: u64) -> BlockPayload {
        BlockPayload {
            block,
            samples: 4,
            seed,
        }
    }

    #[test]
    fn payload_round_trip() {
        let payload = payload(BLOCK, u64::MAX);
        assert_eq!(BlockPayload::decode(&payload.encode()).unwrap(), payload);
    }

    #[test]
    fn payload_rejects_unknown_fields() {
        let json =
            br#"{"block":{"x":0,"y":0,"width":1,"height":1},"samples":1,"seed":0,"sample":2}"#;
        assert!(matches!(
            BlockPayload::decode(json),
            Err(WireError::Json(_))
        ));
    }

    #[test]
    fn payload_validation() {
        let camera = half_lit_scene(8, 4).camera;
        assert!(payload(BLOCK, 0).validate(&camera).is_ok());
        // Touching the far edges is fine.
        let corner = Block {
            x: 5,
            y: 2,
            width: 3,
            height: 2,
        };
        assert!(payload(corner, 0).validate(&camera).is_ok());

        let overflowing = [
            Block { x: 6, ..corner },
            Block { y: 3, ..corner },
            Block {
                x: u32::MAX,
                ..corner
            },
        ];
        for block in overflowing {
            assert!(
                matches!(
                    payload(block, 0).validate(&camera),
                    Err(WireError::OutOfImage { .. })
                ),
                "{block:?}"
            );
        }
        let no_samples = BlockPayload {
            samples: 0,
            ..payload(BLOCK, 0)
        };
        assert!(matches!(
            no_samples.validate(&camera),
            Err(WireError::NoSamples)
        ));
    }

    fn result() -> BlockResult {
        BlockResult {
            block: BLOCK,
            samples: 17,
            pixels: (0..BLOCK.len())
                .map(|i| Color {
                    r: i as f32,
                    g: -0.5,
                    b: f32::MAX,
                })
                .collect(),
        }
    }

    #[test]
    fn result_round_trip() {
        let result = result();
        let bytes = result.encode();
        assert_eq!(bytes.len(), 24 + 12 * BLOCK.len());
        assert_eq!(BlockResult::decode(&bytes).unwrap(), result);
    }

    #[test]
    fn empty_result_round_trip() {
        let result = BlockResult {
            block: Block { width: 0, ..BLOCK },
            samples: 1,
            pixels: Vec::new(),
        };
        assert_eq!(BlockResult::decode(&result.encode()).unwrap(), result);
    }

    #[test]
    fn result_rejects_malformed_bytes() {
        let bytes = result().encode();
        for len in [0, 23, 24, bytes.len() - 1] {
            assert!(
                matches!(
                    BlockResult::decode(&bytes[..len]),
                    Err(WireError::Length { .. })
                ),
                "truncated to {len}"
            );
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(matches!(
            BlockResult::decode(&longer),
            Err(WireError::Length { .. })
        ));
        let mut bad_magic = bytes;
        bad_magic[0] = b'X';
        assert!(matches!(
            BlockResult::decode(&bad_magic),
            Err(WireError::BadMagic)
        ));
    }

    #[test]
    fn scene_round_trip() {
        let scene = r#"{
            "camera": {"origin": [1, 2, 3], "direction": [0, 0, -1], "width": 64, "height": 48},
            "samples": 7,
            "objects": [
                {"type": "sphere", "radius": 2, "position": [0, 0, -5], "material": "Metal", "fuzziness": 0.1},
                {"type": "sphere", "radius": 1, "position": [3, 0, -5], "material": "Refraction"},
                {"type": "quad", "position": [0, 0, 0], "u": [1, 0, 0], "v": [0, 1, 0], "emission": [4, 4, 4]},
                {"type": "ellipse", "position": [0, 0, 0], "u": [1, 0, 0], "v": [0, 2, 0], "texture": "marble", "scale": 3}
            ]
        }"#;
        let definition = decode_scene(scene.as_bytes()).unwrap();
        let round_trip = decode_scene(&encode_scene(&definition)).unwrap();
        // Debug prints f32 losslessly, so equal dumps mean equal scenes.
        assert_eq!(format!("{round_trip:?}"), format!("{definition:?}"));
    }

    #[test]
    fn rows_are_in_display_order() {
        let scene = half_lit_scene(4, 4);
        let block = Block {
            x: 0,
            y: 0,
            width: 4,
            height: 4,
        };
        let payload = payload(block, 0);
        let top = render_block_row(&scene, &payload, 0);
        let bottom = render_block_row(&scene, &payload, 3);
        let lit = Color {
            r: 1.0,
            g: 2.0,
            b: 3.0,
        };
        assert_eq!(top, vec![lit; 4]);
        assert_eq!(bottom, vec![Color::default(); 4]);
    }

    #[test]
    fn rendering_is_reproducible_and_seeded() {
        let scene: Scene = decode_scene(
            br#"{
                "camera": {"origin": [0, 0, 0], "direction": [0, 0, -1], "width": 8, "height": 8},
                "objects": [
                    {"type": "sphere", "radius": 1, "position": [0, 3, -5], "emission": [5, 5, 5]},
                    {"type": "sphere", "radius": 100, "position": [0, -101, -5]}
                ]
            }"#,
        )
        .unwrap()
        .into();
        let render = |seed| render_block_row(&scene, &payload(BLOCK, seed), 0);

        assert_eq!(render(1), render(1));
        assert_ne!(render(1), render(2));
    }

    #[test]
    fn row_streams_are_distinct() {
        // Includes the seed/position pairs a naive `seed ^ y` or `seed + x`
        // scheme would map to the same stream.
        let mut seeds = std::collections::HashSet::new();
        for seed in 0..8 {
            for x in 0..8 {
                for y in 0..8 {
                    assert!(seeds.insert(row_seed(seed, x, y)), "{seed} {x} {y}");
                }
            }
        }
    }
}
