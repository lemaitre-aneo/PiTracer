//! Assembling the rendered blocks into a displayable image (the image is
//! split into blocks by `raytracer::tile_grid`).

use raytracer::{Block, DisplayBlock};

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
