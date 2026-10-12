//! Denoising tasks: once a block and its neighbours are final, a denoising
//! task assembles a tile made of the block grown by [`DENOISE_HALO`] pixels
//! from their final linear results, denoises it (Open Image Denoise, in the
//! worker), and keeps the block's own pixels. The halo is what OIDN needs
//! for a tile to come out as if the whole image had been denoised (no
//! seams), so the client plans a neighbourhood of as many blocks as that
//! takes ([`neighborhood`]) rather than making blocks larger.
//!
//! The same worker image runs both kinds of tasks, told apart by the
//! [`TASK_TYPE_OPTION`] task option.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{Block, BlockOutputs, BlockResult, Camera, Color, WireError};

/// Task option selecting what a task does ([`TaskType`]).
pub const TASK_TYPE_OPTION: &str = "taskType";

/// Pixels of context a denoising tile needs on each side of its block:
/// Open Image Denoise's `tileOverlap`, half its 174-pixel receptive field
/// rounded up to [`DENOISE_ALIGNMENT`] (`core/unet_filter.cpp`, OIDN 2.5.1;
/// the worker's tests check it against the linked library).
pub const DENOISE_HALO: u32 = 96;
/// Tile offsets and sizes OIDN wants multiples of (`tileAlignment`): block
/// sizes must be multiples of it for tiles grown from blocks to be aligned.
pub const DENOISE_ALIGNMENT: u32 = 16;
/// Priority of the denoising tasks, above the path-tracing follow-ups: a
/// neighbourhood is denoised as soon as it is final.
pub const DENOISE_PRIORITY: i32 = 9;

/// Denoised results of a block, `<prefix>-x-y` like the others; the
/// display one shares the display images' `display-` prefix.
pub const DENOISED_LINEAR_NAME: &str = "linear-denoised";
pub const DENOISED_DISPLAY_NAME: &str = "display-denoised";
/// Payload of a denoising task.
pub const DENOISE_PAYLOAD_NAME: &str = "denoise";

/// What a task does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    /// One pass over a block ([`crate::BlockPayload`]), the default.
    PathTracing,
    /// Denoise a block from its neighbourhood ([`DenoisePayload`]).
    Denoising,
}

impl TaskType {
    pub fn name(self) -> &'static str {
        match self {
            TaskType::PathTracing => "pathTracing",
            TaskType::Denoising => "denoising",
        }
    }

    /// Reads [`TASK_TYPE_OPTION`]; path tracing when absent.
    pub fn from_options(options: &HashMap<String, String>) -> Result<Self, WireError> {
        match options.get(TASK_TYPE_OPTION).map(String::as_str) {
            None => Ok(TaskType::PathTracing),
            Some(name) => [TaskType::PathTracing, TaskType::Denoising]
                .into_iter()
                .find(|kind| kind.name() == name)
                .ok_or_else(|| WireError::InvalidOption {
                    name: TASK_TYPE_OPTION,
                    value: name.to_owned(),
                }),
        }
    }

    pub fn to_options(self, options: &mut HashMap<String, String>) {
        options.insert(TASK_TYPE_OPTION.into(), self.name().into());
    }
}

/// One block of a denoising neighbourhood and the id of its final linear
/// result (a data dependency of the task).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenoiseInput {
    pub block: Block,
    pub result: String,
}

/// Payload of a denoising task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenoisePayload {
    /// The block to denoise.
    pub block: Block,
    /// Its neighbourhood, itself included ([`neighborhood`]).
    pub inputs: Vec<DenoiseInput>,
    /// The denoised results (the task's expected outputs).
    pub outputs: BlockOutputs,
}

impl DenoisePayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("a denoise payload always serializes")
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, WireError> {
        Ok(serde_json::from_slice(bytes)?)
    }
}

/// The pixels the denoising task of `block` works on: the block grown by
/// [`DENOISE_HALO`] on each side, clipped to the `width` x `height` image.
pub fn denoise_tile(block: &Block, width: u32, height: u32) -> Block {
    let x = block.x.saturating_sub(DENOISE_HALO);
    let y = block.y.saturating_sub(DENOISE_HALO);
    let right = (block.x + block.width + DENOISE_HALO).min(width);
    let bottom = (block.y + block.height + DENOISE_HALO).min(height);
    Block {
        x,
        y,
        width: right.saturating_sub(x),
        height: bottom.saturating_sub(y),
    }
}

fn intersection(a: &Block, b: &Block) -> Option<Block> {
    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    let right = (a.x + a.width).min(b.x + b.width);
    let bottom = (a.y + a.height).min(b.y + b.height);
    (x < right && y < bottom).then(|| Block {
        x,
        y,
        width: right - x,
        height: bottom - y,
    })
}

/// The blocks among `blocks` whose results the denoising task of `block`
/// needs: those its tile overlaps (itself included).
pub fn neighborhood(blocks: &[Block], block: &Block, width: u32, height: u32) -> Vec<Block> {
    let tile = denoise_tile(block, width, height);
    blocks
        .iter()
        .filter(|candidate| intersection(candidate, &tile).is_some())
        .copied()
        .collect()
}

/// A denoising tile: linear RGB f32 triplets of the pixels of `area`,
/// row-major, as OIDN takes them.
#[derive(Debug, Clone, PartialEq)]
pub struct Tile {
    pub area: Block,
    pub color: Vec<f32>,
    /// Sample count of the block being denoised.
    samples: u32,
}

impl Tile {
    /// Assembles the tile of `block` in a `width` x `height` image from the
    /// final linear results of its neighbourhood, which must cover it
    /// exactly once and include `block` itself.
    pub fn assemble(
        block: &Block,
        width: u32,
        height: u32,
        inputs: &[BlockResult],
    ) -> Result<Tile, WireError> {
        let area = denoise_tile(block, width, height);
        let samples = inputs
            .iter()
            .find(|input| input.block == *block)
            .ok_or(WireError::IncompleteTile { tile: area })?
            .samples;
        let mut color = vec![0f32; area.len() * 3];
        let mut covered = vec![0u8; area.len()];
        for input in inputs {
            let Some(overlap) = intersection(&input.block, &area) else {
                continue;
            };
            for y in overlap.y..overlap.y + overlap.height {
                for x in overlap.x..overlap.x + overlap.width {
                    let source =
                        ((y - input.block.y) * input.block.width + x - input.block.x) as usize;
                    let target = ((y - area.y) * area.width + x - area.x) as usize;
                    let pixel = input.pixels[source];
                    color[3 * target..3 * target + 3].copy_from_slice(&[pixel.r, pixel.g, pixel.b]);
                    covered[target] += 1;
                }
            }
        }
        if covered.iter().all(|&count| count == 1) {
            Ok(Tile {
                area,
                color,
                samples,
            })
        } else {
            Err(WireError::IncompleteTile { tile: area })
        }
    }

    /// The pixels of `block` (within the tile), as a result of the block's
    /// sample count.
    pub fn crop(&self, block: &Block) -> BlockResult {
        let mut pixels = Vec::with_capacity(block.len());
        for y in block.y..block.y + block.height {
            for x in block.x..block.x + block.width {
                let source = 3 * ((y - self.area.y) * self.area.width + x - self.area.x) as usize;
                let rgb = &self.color[source..source + 3];
                pixels.push(Color {
                    r: rgb[0],
                    g: rgb[1],
                    b: rgb[2],
                });
            }
        }
        BlockResult {
            block: *block,
            samples: self.samples,
            pixels,
        }
    }
}

/// Checks that a denoising payload fits `camera`'s image.
pub fn validate_denoise(payload: &DenoisePayload, camera: &Camera) -> Result<(), WireError> {
    let fits = |block: &Block| {
        block
            .x
            .checked_add(block.width)
            .is_some_and(|end| end <= camera.width)
            && block
                .y
                .checked_add(block.height)
                .is_some_and(|end| end <= camera.height)
    };
    for block in std::iter::once(&payload.block).chain(payload.inputs.iter().map(|i| &i.block)) {
        if !fits(block) {
            return Err(WireError::OutOfImage {
                block: *block,
                width: camera.width,
                height: camera.height,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
