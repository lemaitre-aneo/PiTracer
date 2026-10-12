use rand::Rng;

use super::*;
use crate::test_utils::{assert_close, assert_stat_close, png, rng, seeds};
use crate::{Geometry, Lambertian, Object, Quad, UnitVector, Vector};

fn hit_at(u: f32, v: f32) -> Hit {
    Hit {
        point: Vector::default(),
        normal: UnitVector::default(),
        front: true,
        u,
        v,
        distance: 1.0,
        offset: 0.0,
    }
}

fn image(width: u32, height: u32, texels: &[[u8; 4]]) -> Image {
    Image::from_bytes(png(width, height, texels)).unwrap()
}

fn attenuation(image: &Image, u: f32, v: f32) -> Color {
    Texture::<dyn Rng>::get(image, hit_at(u, v), &mut rng(seeds()[0])).attenuation
}

fn opacity(image: &Image, u: f32, v: f32) -> f32 {
    Texture::<dyn Rng>::opacity(image, &hit_at(u, v))
}

#[test]
fn srgb_decoding_matches_the_standard() {
    assert_eq!(srgb_to_linear(0.0), 0.0);
    assert_eq!(srgb_to_linear(1.0), 1.0);
    // Mid grey: sRGB 188 is about 0.5 linear.
    assert_close(srgb_to_linear(188.0 / 255.0), 0.5029, 1e-4);
    // Linear segment.
    assert_close(srgb_to_linear(10.0 / 255.0), 10.0 / 255.0 / 12.92, 1e-7);
}

#[test]
fn the_image_covers_the_unit_square_top_row_up() {
    // Top row red, bottom row blue.
    let image = image(1, 2, &[[255, 0, 0, 255], [0, 0, 255, 255]]);
    assert_eq!(image.size(), Some((1.0, 2.0)));
    assert_eq!(
        attenuation(&image, 0.5, 0.75),
        Color {
            r: 1.0,
            g: 0.0,
            b: 0.0
        }
    );
    assert_eq!(
        attenuation(&image, 0.5, 0.25),
        Color {
            r: 0.0,
            g: 0.0,
            b: 1.0
        }
    );
    // Clamped outside, so the edge texels extend.
    assert_eq!(
        attenuation(&image, 0.5, 7.0),
        Color {
            r: 1.0,
            g: 0.0,
            b: 0.0
        }
    );
    assert_eq!(
        attenuation(&image, -3.0, -1.0),
        Color {
            r: 0.0,
            g: 0.0,
            b: 1.0
        }
    );
    // Halfway between the texel centers: the average.
    assert_close(attenuation(&image, 0.5, 0.5).r, 0.5, 1e-6);
    assert_close(attenuation(&image, 0.5, 0.5).b, 0.5, 1e-6);
}

#[test]
fn transparent_texels_do_not_bleed_their_color() {
    // Opaque white next to a transparent black texel.
    let image = image(2, 1, &[[255, 255, 255, 255], [0, 0, 0, 0]]);
    assert!(Texture::<dyn Rng>::transparent(&image));
    // Halfway: half covered, and what covers it is white.
    assert_close(opacity(&image, 0.5, 0.5), 0.5, 1e-6);
    assert_eq!(
        attenuation(&image, 0.5, 0.5),
        Color {
            r: 1.0,
            g: 1.0,
            b: 1.0
        }
    );
    assert_eq!(opacity(&image, 1.0, 0.5), 0.0);
    assert_eq!(attenuation(&image, 1.0, 0.5), Color::default());
}

#[test]
fn opaque_images_are_not_transparent() {
    let image = image(1, 1, &[[10, 20, 30, 255]]);
    assert!(!Texture::<dyn Rng>::transparent(&image));
    // Unloaded: black and opaque.
    assert!(!Texture::<dyn Rng>::transparent(&Image::default()));
    assert_eq!(attenuation(&Image::default(), 0.5, 0.5), Color::default());
}

#[test]
fn a_loaded_image_travels_in_the_scene_json() {
    let folder = tempfile::tempdir().unwrap();
    std::fs::write(
        folder.path().join("logo.png"),
        png(1, 1, &[[255, 0, 0, 128]]),
    )
    .unwrap();
    let mut image: Image = serde_json::from_str(r#"{"file": "logo.png"}"#).unwrap();
    assert_eq!(image.size(), None);
    image.load(folder.path()).unwrap();
    assert_eq!(image.size(), Some((1.0, 1.0)));

    let json = serde_json::to_value(&image).unwrap();
    // The path would not resolve on a worker: only the data travels.
    assert!(json.get("file").is_none());
    assert!(json["data"].is_string());
    let received: Image = serde_json::from_value(json).unwrap();
    assert_eq!(received.size(), Some((1.0, 1.0)));
    assert_eq!(
        attenuation(&received, 0.5, 0.5),
        attenuation(&image, 0.5, 0.5)
    );
    assert_close(opacity(&received, 0.5, 0.5), 128.0 / 255.0, 1e-6);
}

#[test]
fn bad_images_are_errors() {
    let folder = tempfile::tempdir().unwrap();
    let mut missing: Image = serde_json::from_str(r#"{"file": "missing.png"}"#).unwrap();
    assert!(matches!(
        missing.load(folder.path()),
        Err(ImageError::Read { .. })
    ));
    assert!(matches!(
        Image::default().load(folder.path()),
        Err(ImageError::NoFile)
    ));
    std::fs::write(folder.path().join("bad.png"), b"not a png").unwrap();
    let mut bad: Image = serde_json::from_str(r#"{"file": "bad.png"}"#).unwrap();
    assert!(matches!(
        bad.load(folder.path()),
        Err(ImageError::DecodeFile { .. })
    ));
    assert!(serde_json::from_str::<Image>(r#"{"data": "%%%"}"#).is_err());
}

/// A unit quad in the z = 0 plane, facing a ray coming down the z axis.
fn quad(texture: Image) -> Object<Quad, Lambertian, Image> {
    Object::new(
        Quad::new(
            Vector {
                x: -0.5,
                y: -0.5,
                z: 0.0,
            },
            Vector {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            Vector {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
        ),
        Lambertian {},
        texture,
    )
}

fn hit_fraction(object: &Object<Quad, Lambertian, Image>, n: usize) -> (f64, f64) {
    let origin = Vector {
        x: 0.0,
        y: 0.0,
        z: 5.0,
    };
    let direction = UnitVector::try_from(Vector {
        x: 0.0,
        y: 0.0,
        z: -1.0,
    })
    .unwrap();
    let mut rng = rng(seeds()[1]);
    let hits = (0..n)
        .filter(|_| object.hit(origin, direction, &mut rng).is_some())
        .count();
    let p = hits as f64 / n as f64;
    (p, (p * (1.0 - p) / n as f64).sqrt())
}

#[test]
fn rays_cross_a_transparent_surface_with_probability_one_minus_alpha() {
    let alpha = 64u8;
    let (p, se) = hit_fraction(&quad(image(1, 1, &[[0, 0, 0, alpha]])), 100_000);
    assert_stat_close(p, se, f64::from(alpha) / 255.0, 4f64, 0f64);

    assert_eq!(
        hit_fraction(&quad(image(1, 1, &[[0, 0, 0, 0]])), 1000).0,
        0.0
    );
    assert_eq!(
        hit_fraction(&quad(image(1, 1, &[[0, 0, 0, 255]])), 1000).0,
        1.0
    );
}

/// A ray seeing through the near side of a sphere stops at its far side.
#[test]
fn rays_seeing_through_a_surface_find_its_next_crossing() {
    // The sphere's surface coordinates are its world x and y (clamped onto
    // the image): left texel (x <= 0) transparent, right one opaque.
    let object = Object::new(
        crate::Sphere {
            radius: 1.0,
            position: Vector::default(),
        },
        Lambertian {},
        image(2, 1, &[[0, 0, 0, 0], [0, 0, 0, 255]]),
    );
    let origin = Vector {
        x: -5.0,
        y: 0.0,
        z: 0.0,
    };
    let direction = UnitVector::try_from(Vector {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    })
    .unwrap();
    let (distance, _) = object.hit(origin, direction, &mut rng(seeds()[2])).unwrap();
    assert_close(distance, 6.0, 1e-5);
    // From the other side, the opaque side is hit first.
    let origin = Vector {
        x: 5.0,
        y: 0.0,
        z: 0.0,
    };
    let direction = UnitVector::try_from(Vector {
        x: -1.0,
        y: 0.0,
        z: 0.0,
    })
    .unwrap();
    let (distance, _) = object.hit(origin, direction, &mut rng(seeds()[2])).unwrap();
    assert_close(distance, 4.0, 1e-5);
}

/// Images an SVG refers to by path are read relative to the SVG file when
/// the scene is loaded, and embedded in what travels.
#[test]
fn svg_image_files_are_embedded_when_loaded() {
    let folder = tempfile::tempdir().unwrap();
    std::fs::create_dir(folder.path().join("art")).unwrap();
    std::fs::write(
        folder.path().join("art/pixel.png"),
        png(1, 1, &[[255, 0, 0, 255]]),
    )
    .unwrap();
    std::fs::write(
        folder.path().join("art/logo.svg"),
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><image width="10" height="10" href="pixel.png"/></svg>"#,
    )
    .unwrap();
    let mut image: Image = serde_json::from_str(r#"{"file": "art/logo.svg"}"#).unwrap();
    image.load(folder.path()).unwrap();
    assert_eq!(
        attenuation(&image, 0.5, 0.5),
        Color {
            r: 1.0,
            g: 0.0,
            b: 0.0
        }
    );

    let json = serde_json::to_value(&image).unwrap();
    let bytes = BASE64.decode(json["data"].as_str().unwrap()).unwrap();
    assert!(!svg::has_external_images(&bytes));
    // The file is not needed any more.
    std::fs::remove_file(folder.path().join("art/pixel.png")).unwrap();
    let received: Image = serde_json::from_value(json).unwrap();
    assert_eq!(
        attenuation(&received, 0.5, 0.5),
        Color {
            r: 1.0,
            g: 0.0,
            b: 0.0
        }
    );

    // Missing, now.
    let mut missing: Image = serde_json::from_str(r#"{"file": "art/logo.svg"}"#).unwrap();
    assert!(matches!(
        missing.load(folder.path()),
        Err(ImageError::DecodeFile {
            source: DecodeError::Svg(SvgError::Missing(_)),
            ..
        })
    ));
}

#[cfg(feature = "svg-text")]
mod text {
    use super::*;

    const FONT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test-data/fonts/cmunvt.otf");

    fn svg_text(family: &str, text: &str) -> String {
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><text x="5" y="80" font-family="{family}" font-size="90">{text}</text></svg>"#
        )
    }

    /// Loads the SVG with the test font only (no system fonts, so the
    /// result does not depend on the machine).
    fn load(svg: &str, fonts: &[&str]) -> Result<Image, ImageError> {
        let folder = tempfile::tempdir().unwrap();
        std::fs::write(folder.path().join("text.svg"), svg).unwrap();
        let mut image: Image = serde_json::from_value(serde_json::json!({
            "file": "text.svg",
            "fonts": fonts,
            "system_fonts": false,
        }))
        .unwrap();
        image.load(folder.path()).map(|()| image)
    }

    fn coverage(image: &Image) -> f32 {
        let n = 50;
        let mut covered = 0.0;
        for i in 0..n {
            for j in 0..n {
                let (u, v) = ((i as f32 + 0.5) / n as f32, (j as f32 + 0.5) / n as f32);
                covered += opacity(image, u, v);
            }
        }
        covered / (n * n) as f32
    }

    #[test]
    fn text_becomes_paths_drawn_with_the_listed_fonts() {
        let image = load(
            &svg_text("CMU Typewriter Text Variable Width", "Hi"),
            &[FONT],
        )
        .unwrap();
        // Glyphs cover part of the canvas, not all of it.
        let covered = coverage(&image);
        assert!(covered > 0.05 && covered < 0.8, "{covered}");

        // What travels is text-free, and decodes without fonts.
        let json = serde_json::to_value(&image).unwrap();
        let bytes = BASE64.decode(json["data"].as_str().unwrap()).unwrap();
        assert!(!svg::has_text(&bytes));
        assert!(json.get("fonts").is_none());
        let received: Image = serde_json::from_value(json).unwrap();
        assert_eq!(coverage(&received), covered);
    }

    #[test]
    fn generic_families_use_the_first_listed_font() {
        let generic = load(&svg_text("sans-serif", "Hi"), &[FONT]).unwrap();
        let named = load(
            &svg_text("CMU Typewriter Text Variable Width", "Hi"),
            &[FONT],
        )
        .unwrap();
        assert_eq!(coverage(&generic), coverage(&named));
    }

    #[test]
    fn missing_fonts_are_errors() {
        let missing = |result: Result<Image, ImageError>| match result {
            Err(ImageError::DecodeFile {
                source: DecodeError::Svg(SvgError::Missing(missing)),
                ..
            }) => missing,
            other => panic!("{other:?}"),
        };
        // No font at all.
        let fonts = missing(load(&svg_text("Nonexistent", "Hi"), &[]));
        assert!(fonts[0].contains("Nonexistent"), "{fonts:?}");
        // A character the font does not have.
        let characters = missing(load(&svg_text("sans-serif", "Hi \u{6f22}"), &[FONT]));
        assert!(
            characters.iter().any(|m| m.contains('\u{6f22}')),
            "{characters:?}"
        );
    }

    #[test]
    fn bad_font_files_are_errors() {
        let folder = tempfile::tempdir().unwrap();
        let bad = folder.path().join("bad.ttf");
        std::fs::write(&bad, b"not a font").unwrap();
        for font in [bad.to_str().unwrap(), "/nonexistent/font.ttf"] {
            assert!(
                matches!(
                    load(&svg_text("sans-serif", "Hi"), &[font]),
                    Err(ImageError::DecodeFile {
                        source: DecodeError::Svg(SvgError::FontFile { .. }),
                        ..
                    })
                ),
                "{font}"
            );
        }
    }
}

#[cfg(not(feature = "svg-text"))]
#[test]
fn text_needs_the_svg_text_feature() {
    let svg =
        br#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><text>Hi</text></svg>"#;
    assert!(matches!(
        Image::from_bytes(svg.to_vec()),
        Err(ImageError::Decode(DecodeError::Svg(SvgError::Unsupported(
            _
        ))))
    ));
}
