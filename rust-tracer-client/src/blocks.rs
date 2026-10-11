//! Splitting the image into task blocks and assembling the rendered blocks
//! back into a displayable image.

use raytracer::{Block, BlockResult, Color};

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

/// Copies a rendered block into a tightly packed RGB24 image of
/// `image_width` pixels per row, converting each pixel with `to_rgb`.
pub fn blit(
    rgb: &mut [u8],
    image_width: u32,
    result: &BlockResult,
    to_rgb: impl Fn(Color) -> [u8; 3],
) {
    let Block { x, y, width, .. } = result.block;
    let pitch = image_width as usize * 3;
    for (row, pixels) in result.pixels.chunks_exact(width as usize).enumerate() {
        let start = (y as usize + row) * pitch + x as usize * 3;
        for (dst, &pixel) in rgb[start..start + width as usize * 3]
            .chunks_exact_mut(3)
            .zip(pixels)
        {
            dst.copy_from_slice(&to_rgb(pixel));
        }
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

    #[test]
    fn blit_places_rows_top_down() {
        let result = BlockResult {
            block: Block {
                x: 1,
                y: 1,
                width: 2,
                height: 2,
            },
            samples: 1,
            pixels: (0..4)
                .map(|i| Color {
                    r: i as f32,
                    g: 0.0,
                    b: 0.0,
                })
                .collect(),
        };
        let mut rgb = vec![255; 4 * 3 * 3];
        blit(&mut rgb, 4, &result, |c| [c.r as u8, c.g as u8, c.b as u8]);
        let red: Vec<_> = rgb.chunks_exact(3).map(|p| p[0]).collect();
        #[rustfmt::skip]
        assert_eq!(red, [
            255, 255, 255, 255,
            255, 0,   1,   255,
            255, 2,   3,   255,
        ]);
    }
}
