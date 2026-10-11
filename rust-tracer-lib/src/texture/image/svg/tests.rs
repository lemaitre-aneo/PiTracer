use super::*;
use crate::test_utils::{assert_close, assert_stat_close, rng, seeds};
use rand::Rng;

fn svg(body: &str) -> Svg {
    Svg::parse(
        format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">{body}</svg>"#)
            .as_bytes(),
    )
    .unwrap()
}

/// Sample at canvas coordinates (x right, y down, 100x100 canvas).
fn at(svg: &Svg, x: f64, y: f64) -> ([f32; 3], f32) {
    svg.sample((x / 100.0) as f32, (1.0 - y / 100.0) as f32)
}

fn alpha(svg: &Svg, x: f64, y: f64) -> f32 {
    at(svg, x, y).1
}

/// Linear value of an sRGB-encoded one.
fn linear(srgb: f32) -> f32 {
    crate::texture::image::srgb_to_linear(srgb)
}

/// Straight linear color at a canvas point.
fn color(svg: &Svg, x: f64, y: f64) -> [f32; 3] {
    let (color, alpha) = at(svg, x, y);
    color.map(|c| c / alpha)
}

#[test]
fn y_extrema_split_curves_into_monotonic_pieces() {
    // A symmetric arch peaks halfway.
    let arch = [[0.0, 50.0], [0.0, 0.0], [100.0, 0.0], [100.0, 50.0]];
    assert_eq!(y_extrema(&arch), [0.5]);
    let pieces = split(&arch, &[0.5]);
    assert_eq!(pieces.len(), 2);
    // Consecutive pieces share their end point exactly.
    assert_eq!(pieces[0][3], pieces[1][0]);
    assert_eq!(pieces[0][3], [50.0, 12.5]);
    // Monotonic already: nothing to split.
    assert!(y_extrema(&[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]).is_empty());
    assert_eq!(y_extrema(&[[0.0, 0.0], [5.0, 10.0], [10.0, 0.0]]), [0.5]);
}

/// The outline is the exact cubic curve, not an approximation of it: points
/// a hair inside or outside the arch are classified correctly.
#[test]
fn curves_are_evaluated_exactly() {
    let arch = svg(r#"<path d="M0 50 C 0 0, 100 0, 100 50 Z" fill="white"/>"#);
    // Apex at (50, 12.5).
    assert_eq!(alpha(&arch, 50.0, 12.5 + 1e-4), 1.0);
    assert_eq!(alpha(&arch, 50.0, 12.5 - 1e-4), 0.0);
    // At y = 26, the curve crosses at the roots of y(t) = 26 (t = 0.2 and
    // 0.8 for this arch, where x(t) = 100 (3t^2 - 2t^3)).
    let x = |t: f64| 100.0 * (3.0 * t * t - 2.0 * t * t * t);
    let y = |t: f64| 50.0 - 150.0 * t * (1.0 - t);
    for t in [0.2, 0.8] {
        assert_close(y(t), 26.0, 1e-12);
        let (inside, outside) = if t < 0.5 {
            (1e-4, -1e-4)
        } else {
            (-1e-4, 1e-4)
        };
        assert_eq!(alpha(&arch, x(t) + inside, 26.0), 1.0, "t = {t}");
        assert_eq!(alpha(&arch, x(t) + outside, 26.0), 0.0, "t = {t}");
    }
}

/// usvg turns a circle into four cubic arcs, within ~3e-4 r of the circle.
#[test]
fn circles_cover_their_disk() {
    let circle = svg(r#"<circle cx="50" cy="50" r="40" fill="red"/>"#);
    for i in 0..100 {
        let angle = f64::from(i) * 0.0628;
        let (cos, sin) = (angle.cos(), angle.sin());
        assert_eq!(alpha(&circle, 50.0 + 39.9 * cos, 50.0 + 39.9 * sin), 1.0);
        assert_eq!(alpha(&circle, 50.0 + 40.1 * cos, 50.0 + 40.1 * sin), 0.0);
    }
    assert_eq!(at(&circle, 50.0, 50.0).0, [1.0, 0.0, 0.0]);

    let mut rng = rng(seeds()[0]);
    let n = 100_000;
    let covered = (0..n)
        .filter(|_| {
            alpha(
                &circle,
                rng.random::<f64>() * 100.0,
                rng.random::<f64>() * 100.0,
            ) > 0.0
        })
        .count();
    let p = covered as f64 / f64::from(n);
    let expected = std::f64::consts::PI * 0.4 * 0.4;
    assert_stat_close(
        p,
        (p * (1.0 - p) / f64::from(n)).sqrt(),
        expected,
        4f64,
        0f64,
    );
}

#[test]
fn fill_rules_decide_holes() {
    // Two nested squares drawn in the same direction.
    let squares = "M10 10 H90 V90 H10 Z M30 30 H70 V70 H30 Z";
    let nonzero = svg(&format!(r#"<path d="{squares}" fill="white"/>"#));
    let evenodd = svg(&format!(
        r#"<path d="{squares}" fill="white" fill-rule="evenodd"/>"#
    ));
    assert_eq!(alpha(&nonzero, 50.0, 50.0), 1.0);
    assert_eq!(alpha(&evenodd, 50.0, 50.0), 0.0);
    for svg in [&nonzero, &evenodd] {
        assert_eq!(alpha(svg, 20.0, 50.0), 1.0);
        assert_eq!(alpha(svg, 5.0, 50.0), 0.0);
    }
}

#[test]
fn transforms_and_view_boxes_apply() {
    let moved = svg(r#"<g transform="translate(50 0)"><rect width="10" height="10"/></g>"#);
    assert_eq!(alpha(&moved, 55.0, 5.0), 1.0);
    assert_eq!(alpha(&moved, 5.0, 5.0), 0.0);

    // A 20x10 view box on a 200x100 canvas: user units are 10 px.
    let scaled = Svg::parse(
        br#"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100" viewBox="0 0 20 10"><rect width="10" height="10"/></svg>"#,
    )
    .unwrap();
    assert_eq!(scaled.size(), (200.0, 100.0));
    assert_eq!(scaled.sample(0.4, 0.5).1, 1.0);
    assert_eq!(scaled.sample(0.6, 0.5).1, 0.0);
}

#[test]
fn clip_paths_restrict_drawing() {
    let clipped = svg(
        r#"<clipPath id="c"><rect x="50" width="50" height="100"/></clipPath>
        <rect width="100" height="100" clip-path="url(#c)"/>"#,
    );
    assert_eq!(alpha(&clipped, 75.0, 50.0), 1.0);
    assert_eq!(alpha(&clipped, 25.0, 50.0), 0.0);
}

#[test]
fn opacities_composite() {
    let half = svg(r#"<g opacity="0.5"><rect width="100" height="100" fill-opacity="0.5"/></g>"#);
    assert_close(alpha(&half, 50.0, 50.0), 0.25, 1e-6);

    // Translucent red over opaque blue: composited in sRGB, as SVG does,
    // then decoded.
    let layered = svg(r#"<rect width="100" height="100" fill="blue"/>
        <rect width="100" height="100" fill="red" fill-opacity="0.25"/>"#);
    let (color, alpha) = at(&layered, 50.0, 50.0);
    assert_close(alpha, 1.0, 1e-6);
    assert_close(color[0], linear(0.25), 1e-6);
    assert_close(color[2], linear(0.75), 1e-6);
}

#[test]
fn outside_the_canvas_is_transparent() {
    let full = svg(r#"<rect x="-50" y="-50" width="200" height="200"/>"#);
    assert_eq!(full.sample(0.5, 0.5).1, 1.0);
    assert_eq!(full.sample(-0.1, 0.5).1, 0.0);
    assert_eq!(full.sample(0.5, 1.1).1, 0.0);
}

#[test]
fn unsupported_features_are_errors() {
    for body in [
        r#"<text x="10" y="10">ArmoniK</text>"#,
        r#"<filter id="f"><feGaussianBlur stdDeviation="2"/></filter><rect width="10" height="10" filter="url(#f)"/>"#,
    ] {
        let svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">{body}</svg>"#
        );
        assert!(
            matches!(Svg::parse(svg.as_bytes()), Err(SvgError::Unsupported(_))),
            "{body}"
        );
    }
    assert!(matches!(Svg::parse(b"<svg"), Err(SvgError::Parse(_))));
}

/// Group opacity applies to the group as a whole: overlapping children do
/// not show through each other.
#[test]
fn group_opacity_applies_to_the_composited_group() {
    let group = svg(r#"<g opacity="0.5">
        <rect width="60" height="100" fill="red"/>
        <rect x="40" width="60" height="100" fill="blue"/>
        </g>"#);
    // Where the rectangles overlap: blue at half opacity, no red under it.
    assert_close(alpha(&group, 50.0, 50.0), 0.5, 1e-6);
    let [r, _, b] = color(&group, 50.0, 50.0);
    assert_close(r, 0.0, 1e-6);
    assert_close(b, 1.0, 1e-6);
}

#[test]
fn linear_gradients_interpolate_in_srgb() {
    let gradient = |spread: &str| {
        svg(&format!(
            r#"<linearGradient id="g" gradientUnits="userSpaceOnUse" x1="25" y1="0" x2="75" y2="0" spreadMethod="{spread}">
            <stop offset="0" stop-color="black"/><stop offset="1" stop-color="white"/>
            </linearGradient>
            <rect width="100" height="100" fill="url(#g)"/>"#
        ))
    };
    let pad = gradient("pad");
    assert_close(color(&pad, 25.0, 50.0)[0], 0.0, 1e-6);
    // Halfway: 0.5 in sRGB, about 0.214 linear.
    assert_close(color(&pad, 50.0, 50.0)[1], linear(0.5), 1e-5);
    assert_close(color(&pad, 75.0, 50.0)[2], 1.0, 1e-6);
    // Before the start and after the end.
    assert_close(color(&pad, 10.0, 50.0)[0], 0.0, 1e-6);
    assert_close(color(&pad, 90.0, 50.0)[0], 1.0, 1e-6);
    // 15 past the end: t = 1.3.
    assert_close(color(&gradient("repeat"), 90.0, 50.0)[0], linear(0.3), 1e-5);
    assert_close(
        color(&gradient("reflect"), 90.0, 50.0)[0],
        linear(0.7),
        1e-5,
    );
}

#[test]
fn radial_gradients_follow_their_circles() {
    let centered = svg(
        r#"<radialGradient id="g" gradientUnits="userSpaceOnUse" cx="50" cy="50" r="40">
        <stop offset="0" stop-color="white"/><stop offset="1" stop-color="black"/>
        </radialGradient>
        <rect width="100" height="100" fill="url(#g)"/>"#,
    );
    assert_close(color(&centered, 50.0, 50.0)[0], 1.0, 1e-6);
    assert_close(color(&centered, 70.0, 50.0)[0], linear(0.5), 1e-5);
    assert_close(color(&centered, 50.0, 90.0)[0], 0.0, 1e-6);

    // Focal point off center: t = 0 there, and still 1 on the end circle.
    let focal = svg(
        r#"<radialGradient id="g" gradientUnits="userSpaceOnUse" cx="50" cy="50" r="40" fx="30" fy="50">
        <stop offset="0" stop-color="white"/><stop offset="1" stop-color="black"/>
        </radialGradient>
        <rect width="100" height="100" fill="url(#g)"/>"#,
    );
    assert_close(color(&focal, 30.0, 50.0)[0], 1.0, 1e-6);
    assert_close(color(&focal, 89.999, 50.0)[0], 0.0, 1e-4);
    // Halfway from the focal point to the circle, toward +x: (30 + 90) / 2.
    assert_close(color(&focal, 60.0, 50.0)[0], linear(0.5), 1e-5);
    assert_eq!(
        radial_t([50.0, 50.0], [50.0, 50.0], 40.0, [50.0, 50.0], 0.0),
        Some(0.0)
    );
}

#[test]
fn masks_scale_what_they_cover() {
    let luminance = svg(
        r#"<mask id="m" maskUnits="userSpaceOnUse" x="0" y="0" width="100" height="100">
        <rect width="50" height="100" fill="white"/>
        <rect x="50" width="50" height="100" fill="rgb(128,128,128)"/>
        </mask>
        <rect width="100" height="100" fill="red" mask="url(#m)"/>"#,
    );
    assert_close(alpha(&luminance, 25.0, 50.0), 1.0, 1e-5);
    // Grey 128: luminance of the sRGB value.
    assert_close(alpha(&luminance, 75.0, 50.0), 128.0 / 255.0, 1e-5);

    let alpha_mask = svg(
        r#"<mask id="m" mask-type="alpha" maskUnits="userSpaceOnUse" x="0" y="0" width="50" height="100">
        <rect width="100" height="100" fill="black" fill-opacity="0.5"/>
        </mask>
        <rect width="100" height="100" fill="red" mask="url(#m)"/>"#,
    );
    assert_close(alpha(&alpha_mask, 25.0, 50.0), 0.5, 1e-6);
    // Outside the mask's rectangle.
    assert_eq!(alpha(&alpha_mask, 75.0, 50.0), 0.0);
}

#[test]
fn patterns_tile() {
    // 20x20 tiles, each with a 10x10 square in its top-left corner.
    let checker = svg(
        r#"<pattern id="p" patternUnits="userSpaceOnUse" x="0" y="0" width="20" height="20">
        <rect width="10" height="10" fill="red"/>
        </pattern>
        <rect width="100" height="100" fill="url(#p)"/>"#,
    );
    for (x, y, covered) in [
        (5.0, 5.0, true),
        (15.0, 5.0, false),
        (45.0, 65.0, true),
        (55.0, 65.0, false),
        (85.0, 85.0, true),
        (85.0, 95.0, false),
    ] {
        assert_eq!(alpha(&checker, x, y) > 0.0, covered, "({x}, {y})");
    }
    assert_close(color(&checker, 45.0, 65.0)[0], 1.0, 1e-6);
}

#[test]
fn blend_modes_combine_with_the_backdrop() {
    let blended = |mode: &str, top: &str| {
        svg(&format!(
            r#"<rect width="100" height="100" fill="rgb(255,128,0)"/>
            <g style="mix-blend-mode:{mode}"><rect width="100" height="100" fill="{top}"/></g>"#
        ))
    };
    // Multiply by white keeps the backdrop; by black gives black.
    assert_close(
        color(&blended("multiply", "white"), 50.0, 50.0)[1],
        linear(128.0 / 255.0),
        1e-5,
    );
    assert_close(
        color(&blended("multiply", "black"), 50.0, 50.0)[0],
        0.0,
        1e-6,
    );
    // Screen with black keeps the backdrop.
    assert_close(color(&blended("screen", "black"), 50.0, 50.0)[0], 1.0, 1e-6);
    assert_close(
        color(&blended("difference", "white"), 50.0, 50.0)[2],
        1.0,
        1e-6,
    );
    // Luminosity of grey 128 on the backdrop's hue: back in sRGB, the
    // luminance is the grey's.
    let srgb = color(&blended("luminosity", "rgb(128,128,128)"), 50.0, 50.0).map(|c| {
        if c <= 0.0031308 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        }
    });
    assert_close(lum(srgb), 128.0 / 255.0, 1e-3);
}

#[test]
fn strokes_cover_their_outline() {
    let square = svg(
        r#"<rect x="20" y="20" width="60" height="60" fill="none" stroke="black" stroke-width="10"/>"#,
    );
    // The stroke straddles the edge: 15 to 25 on the left side.
    assert_eq!(alpha(&square, 16.0, 50.0), 1.0);
    assert_eq!(alpha(&square, 24.0, 50.0), 1.0);
    assert_eq!(alpha(&square, 14.0, 50.0), 0.0);
    assert_eq!(alpha(&square, 26.0, 50.0), 0.0);
    assert_eq!(alpha(&square, 50.0, 50.0), 0.0);

    let dashed =
        svg(r#"<path d="M0 50 H100" stroke="black" stroke-width="4" stroke-dasharray="10 10"/>"#);
    assert_eq!(alpha(&dashed, 5.0, 50.0), 1.0);
    assert_eq!(alpha(&dashed, 15.0, 50.0), 0.0);
    assert_eq!(alpha(&dashed, 25.0, 50.0), 1.0);
    assert_eq!(alpha(&dashed, 5.0, 53.0), 0.0);

    // Round caps: a half disk past the end of the line.
    let capped =
        svg(r#"<path d="M20 50 H80" stroke="black" stroke-width="20" stroke-linecap="round"/>"#);
    assert_eq!(alpha(&capped, 80.0 + 9.9 * 0.6, 50.0 + 9.9 * 0.8), 1.0);
    assert_eq!(alpha(&capped, 80.0 + 10.1 * 0.6, 50.0 + 10.1 * 0.8), 0.0);
}

/// An `<image>` whose data is a 2x1 PNG of the given texels.
fn embedded(texels: &[[u8; 4]; 2], attributes: &str) -> Svg {
    use base64::Engine;
    let png =
        base64::engine::general_purpose::STANDARD.encode(crate::test_utils::png(2, 1, texels));
    svg(&format!(
        r#"<image width="100" height="100" {attributes} href="data:image/png;base64,{png}"/>"#
    ))
}

#[test]
fn embedded_images_are_placed_and_sampled() {
    // Red next to transparent; fitted (aspect ratio kept) into the square:
    // 100x50, centered vertically.
    let image = embedded(&[[255, 0, 0, 255], [0, 0, 255, 0]], "");
    assert_close(alpha(&image, 10.0, 50.0), 1.0, 1e-6);
    assert_close(color(&image, 10.0, 50.0)[0], 1.0, 1e-6);
    assert_eq!(alpha(&image, 90.0, 50.0), 0.0);
    // Above and below the fitted image.
    assert_eq!(alpha(&image, 10.0, 20.0), 0.0);
    assert_eq!(alpha(&image, 10.0, 80.0), 0.0);

    // Between the texel centers: bilinear, in sRGB like browsers.
    let smooth = embedded(&[[255, 0, 0, 255], [0, 0, 255, 255]], "");
    let [r, _, b] = color(&smooth, 50.0, 50.0);
    assert_close(r, linear(0.5), 1e-5);
    assert_close(b, linear(0.5), 1e-5);
    // Pixelated: the nearest texel.
    let pixelated = embedded(
        &[[255, 0, 0, 255], [0, 0, 255, 255]],
        r#"style="image-rendering:pixelated""#,
    );
    assert_eq!(color(&pixelated, 49.0, 50.0), [1.0, 0.0, 0.0]);
    assert_eq!(color(&pixelated, 51.0, 50.0), [0.0, 0.0, 1.0]);
}

#[test]
fn nested_svgs_render_within_their_viewport() {
    use base64::Engine;
    // A 10x10 SVG whose circle overflows its viewport.
    let inner = r#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><circle cx="10" cy="5" r="5" fill="red"/></svg>"#;
    let inner = base64::engine::general_purpose::STANDARD.encode(inner);
    let outer = svg(&format!(
        r#"<image x="20" y="20" width="60" height="60" href="data:image/svg+xml;base64,{inner}"/>"#
    ));
    // Scaled 6x: the circle is centered at (80, 50), radius 30.
    assert_close(color(&outer, 70.0, 50.0)[0], 1.0, 1e-6);
    assert_eq!(alpha(&outer, 55.0, 50.0), 1.0);
    assert_eq!(alpha(&outer, 45.0, 50.0), 0.0);
    // Clipped to the viewport (x < 80), though the circle extends past it.
    assert_eq!(alpha(&outer, 85.0, 50.0), 0.0);
}

#[test]
fn image_files_are_refused_once_the_scene_travels() {
    let error = Svg::parse(
        br#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><image width="10" height="10" href="pixel.png"/></svg>"#,
    );
    assert!(matches!(error, Err(SvgError::Missing(_))));
    // Undecodable embedded data.
    let error = Svg::parse(
        br#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><image width="10" height="10" href="data:image/png;base64,iVBORw0KGgo="/></svg>"#,
    );
    assert!(
        matches!(
            error,
            Err(SvgError::Missing(_) | SvgError::EmbeddedImage(_))
        ),
        "{:?}",
        error.err()
    );
}
