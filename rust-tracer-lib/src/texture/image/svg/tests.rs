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

    // Translucent red over opaque blue.
    let layered = svg(r#"<rect width="100" height="100" fill="blue"/>
        <rect width="100" height="100" fill="red" fill-opacity="0.25"/>"#);
    let (color, alpha) = at(&layered, 50.0, 50.0);
    assert_close(alpha, 1.0, 1e-6);
    assert_close(color[0], 0.25, 1e-6);
    assert_close(color[2], 0.75, 1e-6);
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
        r#"<rect width="10" height="10" stroke="black"/>"#,
        r#"<linearGradient id="g"><stop offset="0" stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient><rect width="10" height="10" fill="url(#g)"/>"#,
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
