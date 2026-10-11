//! Splitting the image into task blocks and assembling the rendered blocks
//! back into a displayable image.

use raytracer::{Block, DisplayBlock};

/// Tiles a `width` x `height` image with blocks of at most
/// `block_width` x `block_height` pixels (the last row and column are
/// clipped to the image), ordered from the image center outwards so the
/// middle of the picture shows up first.
pub fn tile(width: u32, height: u32, block_width: u32, block_height: u32) -> Vec<Block> {
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

/// Copies a display image into a tightly packed RGB24 image of
/// `image_width` pixels per row, framing it with a one-pixel `outline` if
/// given (marks blocks still being refined).
pub fn blit(rgb: &mut [u8], image_width: u32, image: &DisplayBlock, outline: Option<[u8; 3]>) {
    let Block {
        x,
        y,
        width,
        height,
    } = image.block;
    let pitch = image_width as usize * 3;
    for (row, pixels) in image.pixels.chunks_exact(width as usize).enumerate() {
        let start = (y as usize + row) * pitch + x as usize * 3;
        for (column, (dst, pixel)) in rgb[start..start + width as usize * 3]
            .chunks_exact_mut(3)
            .zip(pixels)
            .enumerate()
        {
            let edge = row == 0
                || row + 1 == height as usize
                || column == 0
                || column + 1 == width as usize;
            dst.copy_from_slice(match &outline {
                Some(color) if edge => color,
                _ => pixel,
            });
        }
    }
}

/// Outline color of a block refined to `samples` samples per pixel: from
/// blue at `min` to red at `max`, on a log scale (passes add samples
/// linearly, but the noise only falls as their square root).
pub fn outline_color(samples: u32, min: u32, max: u32) -> [u8; 3] {
    let (samples, min, max) = (samples as f32, min.max(1) as f32, max.max(1) as f32);
    let progress = if max > min {
        ((samples.clamp(min, max) / min).ln() / (max / min).ln()).clamp(0.0, 1.0)
    } else {
        1.0
    };
    // Fully saturated hue, from 240 degrees (blue) down to 0 (red), in
    // 60-degree segments.
    let hue = (1.0 - progress) * 4.0;
    let level = |fraction: f32| (fraction * 255.0).round() as u8;
    match hue {
        h if h <= 1.0 => [255, level(h), 0],
        h if h <= 2.0 => [level(2.0 - h), 255, 0],
        h if h <= 3.0 => [0, 255, level(h - 2.0)],
        h => [0, level(4.0 - h), 255],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_cover_the_image_exactly_once() {
        for (width, height, bw, bh) in
            [(16, 12, 4, 4), (17, 13, 5, 3), (3, 2, 60, 60), (1, 1, 1, 1)]
        {
            let mut covered = vec![0; (width * height) as usize];
            for block in tile(width, height, bw, bh) {
                assert!(
                    block.width >= 1
                        && block.width <= bw
                        && block.height >= 1
                        && block.height <= bh
                );
                for y in block.y..block.y + block.height {
                    for x in block.x..block.x + block.width {
                        covered[(y * width + x) as usize] += 1;
                    }
                }
            }
            assert!(
                covered.iter().all(|&n| n == 1),
                "{width}x{height} by {bw}x{bh}"
            );
        }
    }

    #[test]
    fn tiles_start_from_the_center() {
        let blocks = tile(30, 30, 10, 10);
        assert_eq!(
            blocks[0],
            Block {
                x: 10,
                y: 10,
                width: 10,
                height: 10
            }
        );
        // Corners come last.
        assert!(blocks[5..].iter().all(|b| b.x != 10 && b.y != 10));
    }

    fn image() -> DisplayBlock {
        DisplayBlock {
            block: Block {
                x: 1,
                y: 1,
                width: 3,
                height: 3,
            },
            samples: 1,
            pixels: (0..9).map(|i| [i, 0, 0]).collect(),
        }
    }

    fn red(rgb: &[u8]) -> Vec<u8> {
        rgb.chunks_exact(3).map(|p| p[0]).collect()
    }

    #[test]
    fn blit_places_rows_top_down() {
        let mut rgb = vec![255; 5 * 4 * 3];
        blit(&mut rgb, 5, &image(), None);
        #[rustfmt::skip]
        assert_eq!(red(&rgb), [
            255, 255, 255, 255, 255,
            255, 0,   1,   2,   255,
            255, 3,   4,   5,   255,
            255, 6,   7,   8,   255,
        ]);
    }

    #[test]
    fn blit_frames_with_the_outline() {
        let mut rgb = vec![255; 5 * 4 * 3];
        blit(&mut rgb, 5, &image(), Some([9, 9, 9]));
        #[rustfmt::skip]
        assert_eq!(red(&rgb), [
            255, 255, 255, 255, 255,
            255, 9,   9,   9,   255,
            255, 9,   4,   9,   255,
            255, 9,   9,   9,   255,
        ]);
    }

    #[test]
    fn outline_goes_from_blue_to_red() {
        assert_eq!(outline_color(100, 100, 10_000), [0, 0, 255]);
        assert_eq!(outline_color(1_000, 100, 10_000), [0, 255, 0]);
        assert_eq!(outline_color(10_000, 100, 10_000), [255, 0, 0]);
        // Clamped outside the range, and defined when it is empty.
        assert_eq!(outline_color(1, 100, 10_000), [0, 0, 255]);
        assert_eq!(outline_color(99_999, 100, 10_000), [255, 0, 0]);
        assert_eq!(outline_color(5, 10, 10), [255, 0, 0]);
    }
}
