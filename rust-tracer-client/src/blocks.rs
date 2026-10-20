//! Assembling the rendered blocks into a displayable image (the image is
//! split into blocks by `raytracer::tile_grid`).

use raytracer::{Block, DisplayBlock};

/// Copies a display image into a tightly packed RGB24 image of
/// `image_width` pixels per row.
pub fn blit(rgb: &mut [u8], image_width: u32, image: &DisplayBlock) {
    let Block { x, y, width, .. } = image.block;
    let pitch = image_width as usize * 3;
    for (row, pixels) in image.pixels.chunks_exact(width as usize).enumerate() {
        let start = (y as usize + row) * pitch + x as usize * 3;
        for (dst, pixel) in rgb[start..start + width as usize * 3]
            .chunks_exact_mut(3)
            .zip(pixels)
        {
            dst.copy_from_slice(pixel);
        }
    }
}

/// Sample counts at which the outline is blue, and red.
const OUTLINE_SAMPLES: (u32, u32) = (100, 500);

/// Outline color of a block refined to `samples` samples per pixel, as the
/// Python client (`Client/tracer/display.py`, `color_from_samples`): a fully
/// saturated hue going linearly from 0.66 (blue) at 100 samples to 0 (red)
/// at 500, clamped outside.
pub fn outline_color(samples: u32) -> [u8; 3] {
    let (min, max) = OUTLINE_SAMPLES;
    let factor = f64::from(samples.clamp(min, max) - min) / f64::from(max - min);
    hsv_to_rgb(0.66 * (1.0 - factor))
}

/// Python's `colorsys.hsv_to_rgb(hue, 1.0, 1.0)`, truncated to bytes like
/// the Python client does.
fn hsv_to_rgb(hue: f64) -> [u8; 3] {
    let sector = (hue * 6.0).floor();
    let f = hue * 6.0 - sector;
    let byte = |fraction: f64| (fraction * 255.0) as u8;
    let (rising, falling) = (byte(f), byte(1.0 - f));
    match sector as i32 % 6 {
        0 => [255, rising, 0],
        1 => [falling, 255, 0],
        2 => [0, 255, rising],
        3 => [0, falling, 255],
        4 => [rising, 0, 255],
        _ => [255, 0, falling],
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
        blit(&mut rgb, 5, &image());
        #[rustfmt::skip]
        assert_eq!(red(&rgb), [
            255, 255, 255, 255, 255,
            255, 0,   1,   2,   255,
            255, 3,   4,   5,   255,
            255, 6,   7,   8,   255,
        ]);
    }

    #[test]
    fn outline_matches_the_python_client() {
        // Reference values from Python's `colorsys`, as the Python client.
        assert_eq!(outline_color(100), [0, 10, 255]);
        assert_eq!(outline_color(200), [0, 255, 247]);
        assert_eq!(outline_color(300), [5, 255, 0]);
        assert_eq!(outline_color(400), [255, 252, 0]);
        assert_eq!(outline_color(500), [255, 0, 0]);
        // Clamped outside the range.
        assert_eq!(outline_color(1), [0, 10, 255]);
        assert_eq!(outline_color(10_000), [255, 0, 0]);
    }
}
