//! SVG images evaluated exactly, without rasterizing: whether a point is
//! inside a filled path is decided from the path's Bezier curves themselves
//! (no flattening tolerance), so the image is sharp at any magnification.
//! The path tracer's sub-pixel jitter integrates the coverage of each pixel,
//! which anti-aliases the edges.
//!
//! The image is evaluated point by point, like a renderer would per pixel:
//! the group tree is painted back to front at the point, compositing in
//! sRGB-encoded premultiplied color as SVG does by default
//! (`color-interpolation: sRGB`), and only the result is decoded to linear.
//!
//! Supported: paths with solid, gradient (linear, radial) and pattern
//! paints, fill rules, strokes (with dashes), opacity, group isolation,
//! blend modes, transforms, clip paths, masks, and embedded images (PNG,
//! JPEG, GIF, WebP, filtered in sRGB like browsers do, and nested SVGs).
//! Filters are an error rather than a silently wrong image.
//!
//! Text and images referring to files only resolve on the machine loading
//! the scene: [`self_contained`] turns text into paths and embeds the
//! images there, so that what travels to the workers needs neither fonts
//! nor files; [`Svg::parse`] then refuses both.
//!
//! Strokes are the one approximation: `tiny-skia` turns them into outlines
//! (the offset of a Bezier curve is not a Bezier curve), to within about
//! 1/100 of a canvas unit; the outline is then evaluated exactly like a
//! fill.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use usvg::tiny_skia_path::{self, PathSegment, Point};
use usvg::{
    BlendMode, FillRule, Group, ImageKind, ImageRendering, MaskType, Node, Paint, SpreadMethod,
    Transform,
};

use super::Texels;

/// Premultiplied sRGB-encoded color and alpha.
type Rgba = [f32; 4];

/// A parsed SVG image: its canvas size, and its group tree in canvas
/// coordinates (y down).
pub(super) struct Svg {
    width: f64,
    height: f64,
    root: GroupItem,
}

enum Item {
    Group(GroupItem),
    Fill(FillItem),
    Image(ImageItem),
}

/// An embedded raster image.
struct ImageItem {
    /// Canvas to texel coordinates (texel i covers [i, i + 1)).
    inverse: Affine,
    texels: Texels,
    /// Bilinear filtering, rather than nearest texel.
    smooth: bool,
}

struct GroupItem {
    /// Painted on its own then composited, rather than straight onto the
    /// backdrop (needed for opacity, clips, masks and blend modes).
    isolate: bool,
    opacity: f32,
    blend: BlendMode,
    clip: Option<Arc<Clip>>,
    mask: Option<Arc<Mask>>,
    children: Vec<Item>,
}

/// A filled path, or the outline of a stroked one.
struct FillItem {
    shape: Shape,
    paint: PaintSource,
    opacity: f32,
}

enum PaintSource {
    /// Straight sRGB.
    Color([f32; 3]),
    Linear {
        /// Canvas to gradient coordinates.
        inverse: Affine,
        start: [f64; 2],
        end: [f64; 2],
        stops: Stops,
    },
    Radial {
        inverse: Affine,
        center: [f64; 2],
        radius: f64,
        focal: [f64; 2],
        focal_radius: f64,
        stops: Stops,
    },
    Pattern {
        /// Canvas to pattern coordinates.
        inverse: Affine,
        /// Tile: x, y, width, height.
        tile: [f64; 4],
        /// In tile coordinates (origin at the tile's corner).
        content: Box<GroupItem>,
    },
}

struct Stops {
    /// Offsets in [0, 1], increasing, with premultiplied colors.
    stops: Vec<(f32, Rgba)>,
    spread: SpreadMethod,
}

/// A clip path: the union of its shapes, within its own clip path if any.
struct Clip {
    shapes: Vec<Shape>,
    clip: Option<Arc<Clip>>,
}

struct Mask {
    kind: MaskType,
    /// The mask's rectangle: nothing shows outside it.
    region: Shape,
    content: GroupItem,
    mask: Option<Arc<Mask>>,
}

/// A 2D affine map, in f64: x' = a x + c y + e, y' = b x + d y + f.
#[derive(Clone, Copy)]
struct Affine([f64; 6]);

impl Affine {
    fn new(transform: Transform) -> Self {
        let Transform {
            sx,
            ky,
            kx,
            sy,
            tx,
            ty,
        } = transform;
        Self([sx, ky, kx, sy, tx, ty].map(f64::from))
    }

    fn apply(&self, x: f64, y: f64) -> [f64; 2] {
        let [a, b, c, d, e, f] = self.0;
        [a * x + c * y + e, b * x + d * y + f]
    }

    fn inverse(&self) -> Option<Self> {
        let [a, b, c, d, e, f] = self.0;
        let det = a * d - b * c;
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let (ia, ib, ic, id) = (d / det, -b / det, -c / det, a / det);
        Some(Self([
            ia,
            ib,
            ic,
            id,
            -(ia * e + ic * f),
            -(ib * e + id * f),
        ]))
    }
}

impl Clip {
    fn contains(&self, x: f64, y: f64) -> bool {
        self.shapes.iter().any(|shape| shape.contains(x, y))
            && self.clip.as_ref().is_none_or(|clip| clip.contains(x, y))
    }
}

impl Mask {
    /// How much of the masked content shows at the point, in [0, 1].
    fn value(&self, x: f64, y: f64) -> f32 {
        if !self.region.contains(x, y) {
            return 0.0;
        }
        let mut layer = [0f32; 4];
        self.content.paint(x, y, &mut layer);
        let value = match self.kind {
            // Premultiplied, so this is the luminance times the alpha.
            MaskType::Luminance => 0.2125 * layer[0] + 0.7154 * layer[1] + 0.0721 * layer[2],
            MaskType::Alpha => layer[3],
        };
        value * self.mask.as_ref().map_or(1.0, |mask| mask.value(x, y))
    }
}

impl GroupItem {
    fn paint(&self, x: f64, y: f64, backdrop: &mut Rgba) {
        if !self.isolate {
            for child in &self.children {
                child.paint(x, y, backdrop);
            }
            return;
        }
        if self.clip.as_ref().is_some_and(|clip| !clip.contains(x, y)) {
            return;
        }
        let mut layer = [0f32; 4];
        for child in &self.children {
            child.paint(x, y, &mut layer);
        }
        if layer[3] <= 0.0 {
            return;
        }
        let factor = self.opacity * self.mask.as_ref().map_or(1.0, |mask| mask.value(x, y));
        if factor <= 0.0 {
            return;
        }
        composite(backdrop, layer.map(|channel| channel * factor), self.blend);
    }
}

impl Item {
    fn paint(&self, x: f64, y: f64, backdrop: &mut Rgba) {
        match self {
            Item::Group(group) => group.paint(x, y, backdrop),
            Item::Fill(fill) => {
                if fill.shape.contains(x, y) {
                    let source = fill.paint.at(x, y).map(|channel| channel * fill.opacity);
                    composite(backdrop, source, BlendMode::Normal);
                }
            }
            Item::Image(image) => {
                let [tx, ty] = image.inverse.apply(x, y);
                let (width, height) = (image.texels.width as f64, image.texels.height as f64);
                if (0.0..width).contains(&tx) && (0.0..height).contains(&ty) {
                    let ([r, g, b], alpha) =
                        image
                            .texels
                            .filter(tx as f32, ty as f32, image.smooth, |byte| {
                                f32::from(byte) / 255.0
                            });
                    composite(backdrop, [r, g, b, alpha], BlendMode::Normal);
                }
            }
        }
    }
}

impl PaintSource {
    /// Premultiplied paint at the canvas point.
    fn at(&self, x: f64, y: f64) -> Rgba {
        match self {
            PaintSource::Color([r, g, b]) => [*r, *g, *b, 1.0],
            PaintSource::Linear {
                inverse,
                start,
                end,
                stops,
            } => {
                let [qx, qy] = inverse.apply(x, y);
                let (dx, dy) = (end[0] - start[0], end[1] - start[1]);
                let length2 = dx * dx + dy * dy;
                let t = if length2 > 0.0 {
                    ((qx - start[0]) * dx + (qy - start[1]) * dy) / length2
                } else {
                    // Degenerate: SVG paints the last stop.
                    1.0
                };
                stops.at(t)
            }
            PaintSource::Radial {
                inverse,
                center,
                radius,
                focal,
                focal_radius,
                stops,
            } => {
                let [qx, qy] = inverse.apply(x, y);
                match radial_t([qx, qy], *center, *radius, *focal, *focal_radius) {
                    Some(t) => stops.at(t),
                    None => [0.0; 4],
                }
            }
            PaintSource::Pattern {
                inverse,
                tile,
                content,
            } => {
                let [qx, qy] = inverse.apply(x, y);
                let [tx, ty, width, height] = *tile;
                let mut layer = [0f32; 4];
                content.paint(
                    (qx - tx).rem_euclid(width),
                    (qy - ty).rem_euclid(height),
                    &mut layer,
                );
                layer
            }
        }
    }
}

/// Parameter of a two-point conical gradient (SVG 2 radial gradient) at
/// `q`: the largest t such that q is on the circle interpolated between the
/// focal circle (t = 0) and the end circle (t = 1), with a non-negative
/// radius. None where no such circle passes (nothing is painted there).
fn radial_t(
    q: [f64; 2],
    center: [f64; 2],
    radius: f64,
    focal: [f64; 2],
    focal_radius: f64,
) -> Option<f64> {
    // |q - f - t d| = fr + t dr.
    let d = [center[0] - focal[0], center[1] - focal[1]];
    let dr = radius - focal_radius;
    let e = [q[0] - focal[0], q[1] - focal[1]];
    let a = d[0] * d[0] + d[1] * d[1] - dr * dr;
    let b = e[0] * d[0] + e[1] * d[1] + focal_radius * dr;
    let c = e[0] * e[0] + e[1] * e[1] - focal_radius * focal_radius;
    let valid = |t: f64| focal_radius + t * dr >= 0.0;
    if a.abs() < 1e-12 * (d[0] * d[0] + d[1] * d[1] + dr * dr) {
        // Linear: -2 b t + c = 0.
        let t = c / (2.0 * b);
        return (b != 0.0 && valid(t)).then_some(t);
    }
    let discriminant = b * b - a * c;
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let (t0, t1) = ((b + root) / a, (b - root) / a);
    let (high, low) = if t0 > t1 { (t0, t1) } else { (t1, t0) };
    [high, low].into_iter().find(|t| valid(*t))
}

impl Stops {
    fn at(&self, t: f64) -> Rgba {
        let t = match self.spread {
            SpreadMethod::Pad => t.clamp(0.0, 1.0),
            SpreadMethod::Repeat => t.rem_euclid(1.0),
            SpreadMethod::Reflect => {
                let period = t.rem_euclid(2.0);
                if period > 1.0 { 2.0 - period } else { period }
            }
        } as f32;
        let stops = &self.stops;
        let Some(next) = stops.iter().position(|(offset, _)| *offset > t) else {
            return stops[stops.len() - 1].1;
        };
        if next == 0 {
            return stops[0].1;
        }
        let ((o0, c0), (o1, c1)) = (stops[next - 1], stops[next]);
        let w = if o1 > o0 { (t - o0) / (o1 - o0) } else { 1.0 };
        std::array::from_fn(|i| c0[i] + w * (c1[i] - c0[i]))
    }
}

/// Composites premultiplied `source` over `backdrop` with a blend mode
/// (W3C Compositing and Blending Level 1, source-over with blending).
fn composite(backdrop: &mut Rgba, source: Rgba, blend: BlendMode) {
    let (sa, ba) = (source[3], backdrop[3]);
    if blend == BlendMode::Normal || ba <= 0.0 || sa <= 0.0 {
        for i in 0..4 {
            backdrop[i] = source[i] + backdrop[i] * (1.0 - sa);
        }
        return;
    }
    let straight = |color: &Rgba, alpha: f32| [0, 1, 2].map(|i| (color[i] / alpha).clamp(0.0, 1.0));
    let blended = blend_colors(blend, straight(backdrop, ba), straight(&source, sa));
    for i in 0..3 {
        backdrop[i] = source[i] * (1.0 - ba) + backdrop[i] * (1.0 - sa) + sa * ba * blended[i];
    }
    backdrop[3] = sa + ba * (1.0 - sa);
}

/// B(Cb, Cs) on straight colors.
fn blend_colors(blend: BlendMode, b: [f32; 3], s: [f32; 3]) -> [f32; 3] {
    let separable = |f: fn(f32, f32) -> f32| [0, 1, 2].map(|i| f(b[i], s[i]));
    match blend {
        BlendMode::Normal => s,
        BlendMode::Multiply => separable(|b, s| b * s),
        BlendMode::Screen => separable(screen),
        BlendMode::Overlay => separable(|b, s| hard_light(s, b)),
        BlendMode::Darken => separable(f32::min),
        BlendMode::Lighten => separable(f32::max),
        BlendMode::ColorDodge => separable(|b, s| {
            if b == 0.0 {
                0.0
            } else if s >= 1.0 {
                1.0
            } else {
                (b / (1.0 - s)).min(1.0)
            }
        }),
        BlendMode::ColorBurn => separable(|b, s| {
            if b >= 1.0 {
                1.0
            } else if s <= 0.0 {
                0.0
            } else {
                1.0 - ((1.0 - b) / s).min(1.0)
            }
        }),
        BlendMode::HardLight => separable(hard_light),
        BlendMode::SoftLight => separable(|b, s| {
            if s <= 0.5 {
                b - (1.0 - 2.0 * s) * b * (1.0 - b)
            } else {
                let d = if b <= 0.25 {
                    ((16.0 * b - 12.0) * b + 4.0) * b
                } else {
                    b.sqrt()
                };
                b + (2.0 * s - 1.0) * (d - b)
            }
        }),
        BlendMode::Difference => separable(|b, s| (b - s).abs()),
        BlendMode::Exclusion => separable(|b, s| b + s - 2.0 * b * s),
        BlendMode::Hue => set_lum(set_sat(s, sat(b)), lum(b)),
        BlendMode::Saturation => set_lum(set_sat(b, sat(s)), lum(b)),
        BlendMode::Color => set_lum(s, lum(b)),
        BlendMode::Luminosity => set_lum(b, lum(s)),
    }
}

fn screen(b: f32, s: f32) -> f32 {
    b + s - b * s
}

fn hard_light(b: f32, s: f32) -> f32 {
    if s <= 0.5 {
        b * 2.0 * s
    } else {
        screen(b, 2.0 * s - 1.0)
    }
}

fn lum(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

fn clip_color(c: [f32; 3]) -> [f32; 3] {
    let l = lum(c);
    let (n, x) = (c[0].min(c[1]).min(c[2]), c[0].max(c[1]).max(c[2]));
    let mut c = c;
    if n < 0.0 {
        c = c.map(|v| l + (v - l) * l / (l - n));
    }
    if x > 1.0 {
        c = c.map(|v| l + (v - l) * (1.0 - l) / (x - l));
    }
    c
}

fn set_lum(c: [f32; 3], l: f32) -> [f32; 3] {
    let d = l - lum(c);
    clip_color(c.map(|v| v + d))
}

fn sat(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

fn set_sat(c: [f32; 3], s: f32) -> [f32; 3] {
    let (min, max) = (c[0].min(c[1]).min(c[2]), c[0].max(c[1]).max(c[2]));
    if max > min {
        c.map(|v| (v - min) * s / (max - min))
    } else {
        [0.0; 3]
    }
}

/// The area a closed path encloses, under its fill rule: its outline as
/// pieces of curves monotonic in y, binned in horizontal bands.
struct Shape {
    rule: FillRule,
    /// Bounding box: x_min, y_min, x_max, y_max.
    bounds: [f64; 4],
    pieces: Vec<Piece>,
    /// Indices of the pieces overlapping each band, the bands splitting
    /// `bounds` vertically into equal parts.
    bands: Vec<Vec<u32>>,
}

/// A line or a quadratic or cubic Bezier curve along which y is monotonic,
/// so that a horizontal line crosses it at most once.
struct Piece {
    /// Control points (2, 3 or 4 used).
    points: [[f64; 2]; 4],
    degree: usize,
    /// Half-open y range [y_min, y_max): where pieces meet at a vertex, the
    /// vertex belongs to exactly one of them unless it is a y extremum of
    /// the outline, which correctly counts twice (cancelling) or never.
    y_min: f64,
    y_max: f64,
    /// x range of the control points, which bounds the curve's.
    x_min: f64,
    x_max: f64,
    /// +1 when y increases along the curve, -1 otherwise.
    winding: i32,
}

impl Piece {
    fn new(points: &[[f64; 2]]) -> Option<Self> {
        let (first, last) = (points[0], points[points.len() - 1]);
        // Horizontal: a horizontal ray never crosses it.
        if first[1] == last[1] {
            return None;
        }
        let mut control = [[0f64; 2]; 4];
        control[..points.len()].copy_from_slice(points);
        let xs = points.iter().map(|point| point[0]);
        Some(Self {
            points: control,
            degree: points.len() - 1,
            y_min: first[1].min(last[1]),
            y_max: first[1].max(last[1]),
            x_min: xs.clone().fold(f64::INFINITY, f64::min),
            x_max: xs.fold(f64::NEG_INFINITY, f64::max),
            winding: if last[1] > first[1] { 1 } else { -1 },
        })
    }

    fn at(&self, t: f64) -> [f64; 2] {
        let p = &self.points;
        let s = 1.0 - t;
        let blend = |i: usize| match self.degree {
            1 => s * p[0][i] + t * p[1][i],
            2 => s * s * p[0][i] + 2.0 * s * t * p[1][i] + t * t * p[2][i],
            _ => {
                s * s * s * p[0][i]
                    + 3.0 * s * s * t * p[1][i]
                    + 3.0 * s * t * t * p[2][i]
                    + t * t * t * p[3][i]
            }
        };
        [blend(0), blend(1)]
    }

    /// x where the piece crosses the line at height `y`, within its y range.
    fn x_at(&self, y: f64) -> f64 {
        let p = &self.points;
        if self.degree == 1 {
            let t = (y - p[0][1]) / (p[1][1] - p[0][1]);
            return p[0][0] + t * (p[1][0] - p[0][0]);
        }
        // y is monotonic along the piece: bisect on t, to f64 precision.
        let increasing = self.winding > 0;
        let (mut low, mut high) = (0f64, 1f64);
        for _ in 0..60 {
            let middle = 0.5 * (low + high);
            if (self.at(middle)[1] < y) == increasing {
                low = middle;
            } else {
                high = middle;
            }
        }
        self.at(0.5 * (low + high))[0]
    }

    /// Signed crossing of the ray from `(x, y)` toward +x.
    fn crossing(&self, x: f64, y: f64) -> i32 {
        if !(self.y_min <= y && y < self.y_max) || x >= self.x_max {
            0
        } else if x < self.x_min || self.x_at(y) > x {
            self.winding
        } else {
            0
        }
    }
}

/// Splits a Bezier curve at the parameters `ts` (increasing, in (0, 1)) with
/// de Casteljau's algorithm. Consecutive pieces share their end points
/// exactly, which the crossing count relies on.
fn split(points: &[[f64; 2]], ts: &[f64]) -> Vec<Vec<[f64; 2]>> {
    let lerp =
        |a: [f64; 2], b: [f64; 2], t: f64| [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
    let mut pieces = Vec::new();
    let mut rest = points.to_vec();
    let mut done = 0f64;
    for &t in ts {
        // Rescaled into what is left of the curve.
        let t = (t - done) / (1.0 - done);
        done += t * (1.0 - done);
        let mut left = vec![rest[0]];
        let mut right = vec![rest[rest.len() - 1]];
        let mut level = rest;
        while level.len() > 1 {
            level = level
                .windows(2)
                .map(|pair| lerp(pair[0], pair[1], t))
                .collect();
            left.push(level[0]);
            right.push(level[level.len() - 1]);
        }
        right.reverse();
        pieces.push(left);
        rest = right;
    }
    pieces.push(rest);
    pieces
}

/// Parameters in (0, 1) where the curve's y is extremal (y'(t) = 0).
fn y_extrema(points: &[[f64; 2]]) -> Vec<f64> {
    let y: Vec<f64> = points.iter().map(|point| point[1]).collect();
    let mut roots = match y.len() {
        3 => {
            let denominator = y[0] - 2.0 * y[1] + y[2];
            if denominator == 0.0 {
                vec![]
            } else {
                vec![(y[0] - y[1]) / denominator]
            }
        }
        4 => {
            // y'(t) / 3 = a t^2 + b t + c.
            let (d0, d1, d2) = (y[1] - y[0], y[2] - y[1], y[3] - y[2]);
            let (a, b, c) = (d0 - 2.0 * d1 + d2, 2.0 * (d1 - d0), d0);
            if a.abs() <= 1e-12 * (b.abs() + c.abs()) {
                if b == 0.0 { vec![] } else { vec![-c / b] }
            } else {
                let discriminant = b * b - 4.0 * a * c;
                if discriminant < 0.0 {
                    vec![]
                } else {
                    // Stable pairing: no cancellation in b +/- sqrt.
                    let q = -0.5 * (b + discriminant.sqrt().copysign(b));
                    if q == 0.0 {
                        vec![0.0]
                    } else {
                        vec![q / a, c / q]
                    }
                }
            }
        }
        _ => vec![],
    };
    roots.retain(|t| *t > 0.0 && *t < 1.0);
    roots.sort_by(f64::total_cmp);
    roots.dedup();
    roots
}

impl Shape {
    /// The shape enclosed by `path` mapped through `transform`, each subpath
    /// closed (as filling does).
    fn new(path: &tiny_skia_path::Path, transform: Transform, rule: FillRule) -> Self {
        let map = |mut point: Point| {
            transform.map_point(&mut point);
            [f64::from(point.x), f64::from(point.y)]
        };
        let mut pieces = Vec::new();
        let mut add = |points: &[[f64; 2]]| {
            for part in split(points, &y_extrema(points)) {
                pieces.extend(Piece::new(&part));
            }
        };
        let (mut start, mut current) = ([0f64; 2], [0f64; 2]);
        for segment in path.segments() {
            match segment {
                PathSegment::MoveTo(point) => {
                    add(&[current, start]);
                    start = map(point);
                    current = start;
                }
                PathSegment::LineTo(point) => {
                    let point = map(point);
                    add(&[current, point]);
                    current = point;
                }
                PathSegment::QuadTo(control, point) => {
                    let (control, point) = (map(control), map(point));
                    add(&[current, control, point]);
                    current = point;
                }
                PathSegment::CubicTo(first, second, point) => {
                    let (first, second, point) = (map(first), map(second), map(point));
                    add(&[current, first, second, point]);
                    current = point;
                }
                PathSegment::Close => {
                    add(&[current, start]);
                    current = start;
                }
            }
        }
        add(&[current, start]);

        let mut bounds = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for piece in &pieces {
            bounds[0] = bounds[0].min(piece.x_min);
            bounds[1] = bounds[1].min(piece.y_min);
            bounds[2] = bounds[2].max(piece.x_max);
            bounds[3] = bounds[3].max(piece.y_max);
        }
        let count = (pieces.len() / 4).clamp(1, 256);
        let mut bands = vec![Vec::new(); count];
        if !pieces.is_empty() {
            let height = (bounds[3] - bounds[1]).max(f64::MIN_POSITIVE);
            let band = |y: f64| (((y - bounds[1]) / height * count as f64) as usize).min(count - 1);
            for (i, piece) in pieces.iter().enumerate() {
                for band in &mut bands[band(piece.y_min)..=band(piece.y_max)] {
                    band.push(i as u32);
                }
            }
        }
        Self {
            rule,
            bounds,
            pieces,
            bands,
        }
    }

    fn contains(&self, x: f64, y: f64) -> bool {
        let [x_min, y_min, x_max, y_max] = self.bounds;
        if !(x_min <= x && x < x_max && y_min <= y && y < y_max) {
            return false;
        }
        let count = self.bands.len();
        let band = (((y - y_min) / (y_max - y_min) * count as f64) as usize).min(count - 1);
        let winding: i32 = self.bands[band]
            .iter()
            .map(|&i| self.pieces[i as usize].crossing(x, y))
            .sum();
        match self.rule {
            FillRule::NonZero => winding != 0,
            FillRule::EvenOdd => winding % 2 != 0,
        }
    }
}

/// What makes an SVG unusable as a texture.
#[derive(Debug)]
pub enum SvgError {
    Parse(usvg::Error),
    Unsupported(&'static str),
    /// A font file listed for the text that could not be loaded.
    FontFile {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    /// An embedded raster image that could not be decoded.
    EmbeddedImage(::image::ImageError),
    /// Fonts, characters or images the SVG needs and could not be found.
    Missing(Vec<String>),
}

impl std::fmt::Display for SvgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SvgError::Parse(error) => write!(f, "invalid SVG: {error}"),
            SvgError::Unsupported(feature) => write!(f, "unsupported SVG feature: {feature}"),
            SvgError::FontFile { path, source } => {
                write!(f, "could not load the font {}: {source}", path.display())
            }
            SvgError::EmbeddedImage(source) => write!(f, "invalid embedded image: {source}"),
            SvgError::Missing(missing) => write!(f, "not found: {}", missing.join("; ")),
        }
    }
}

impl std::error::Error for SvgError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SvgError::Parse(error) => Some(error),
            SvgError::FontFile { source, .. } => Some(source),
            SvgError::EmbeddedImage(source) => Some(source),
            SvgError::Unsupported(_) | SvgError::Missing(_) => None,
        }
    }
}

/// Whether some element of the SVG's XML matches.
fn any_element(bytes: &[u8], matches: impl Fn(usvg::roxmltree::Node) -> bool) -> bool {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|text| usvg::roxmltree::Document::parse(text).ok())
        .is_some_and(|document| document.descendants().any(matches))
}

/// Whether the SVG has text elements (usvg without its `text` feature
/// silently drops them).
pub(super) fn has_text(bytes: &[u8]) -> bool {
    any_element(bytes, |node| node.has_tag_name("text"))
}

/// Whether the SVG has images referring to files rather than embedding
/// their data.
pub(super) fn has_external_images(bytes: &[u8]) -> bool {
    any_element(bytes, |node| {
        node.has_tag_name("image")
            && node
                .attribute(("http://www.w3.org/1999/xlink", "href"))
                .or_else(|| node.attribute("href"))
                .is_some_and(|href| !href.trim_start().starts_with("data:"))
    })
}

/// What a parse needed and could not find: usvg only logs those (and draws
/// nothing), so they are recorded to fail instead.
#[derive(Clone, Default)]
pub(super) struct Missing(Arc<Mutex<Vec<String>>>);

impl Missing {
    pub(super) fn push(&self, what: String) {
        self.0.lock().unwrap().push(what);
    }

    fn check(&self) -> Result<(), SvgError> {
        let mut missing = std::mem::take(&mut *self.0.lock().unwrap());
        if missing.is_empty() {
            return Ok(());
        }
        missing.sort();
        missing.dedup();
        Err(SvgError::Missing(missing))
    }
}

/// usvg's image resolver, recording the images it cannot resolve; images
/// referring to files are only read if `files`.
fn image_resolver(missing: &Missing, files: bool) -> usvg::ImageHrefResolver<'static> {
    let data = usvg::ImageHrefResolver::default_data_resolver();
    let string = usvg::ImageHrefResolver::default_string_resolver();
    usvg::ImageHrefResolver {
        resolve_data: Box::new({
            let missing = missing.clone();
            move |mime, bytes, options| {
                let kind = data(mime, bytes, options).filter(readable);
                if kind.is_none() {
                    missing.push(format!("embedded image of type {mime:?}"));
                }
                kind
            }
        }),
        resolve_string: Box::new({
            let missing = missing.clone();
            move |href, options| {
                if !files {
                    missing.push(format!(
                        "image {href} (files are read when the scene is loaded)"
                    ));
                    return None;
                }
                let kind = string(href, options).filter(readable);
                if kind.is_none() {
                    missing.push(format!("image {href}"));
                }
                kind
            }
        }),
    }
}

/// Whether usvg will keep the image: it drops (and only logs) those whose
/// size it cannot read.
fn readable(kind: &ImageKind) -> bool {
    match kind {
        ImageKind::JPEG(data)
        | ImageKind::PNG(data)
        | ImageKind::GIF(data)
        | ImageKind::WEBP(data) => ::image::ImageReader::new(std::io::Cursor::new(data.as_slice()))
            .with_guessed_format()
            .is_ok_and(|reader| reader.into_dimensions().is_ok()),
        ImageKind::SVG(_) => true,
    }
}

/// Where an SVG's resources are found when the scene is loaded.
pub(super) struct Sources<'a> {
    /// The SVG file's folder, which image paths are relative to.
    pub folder: Option<&'a Path>,
    /// Font files for the text (resolved paths).
    #[cfg_attr(not(feature = "svg-text"), allow(dead_code))]
    pub fonts: &'a [PathBuf],
    /// Also the fonts installed on the system.
    #[cfg_attr(not(feature = "svg-text"), allow(dead_code))]
    pub system_fonts: bool,
}

/// The SVG with its text turned into paths and the image files it refers
/// to embedded: what can travel to the workers.
pub(super) fn self_contained(bytes: &[u8], sources: &Sources) -> Result<Vec<u8>, SvgError> {
    let missing = Missing::default();
    let mut options = usvg::Options {
        resources_dir: sources.folder.map(Path::to_owned),
        image_href_resolver: image_resolver(&missing, true),
        ..Default::default()
    };
    if has_text(bytes) {
        configure_text(&mut options, sources, &missing)?;
    }
    let tree = usvg::Tree::from_data(bytes, &options).map_err(SvgError::Parse)?;
    missing.check()?;
    // The writer turns text into its paths unless told to preserve it, and
    // embeds images as data URLs.
    Ok(tree.to_string(&usvg::WriteOptions::default()).into_bytes())
}

#[cfg(feature = "svg-text")]
fn configure_text(
    options: &mut usvg::Options,
    sources: &Sources,
    missing: &Missing,
) -> Result<(), SvgError> {
    super::text::configure(options, sources.fonts, sources.system_fonts, missing)
}

#[cfg(not(feature = "svg-text"))]
fn configure_text(
    _options: &mut usvg::Options,
    _sources: &Sources,
    _missing: &Missing,
) -> Result<(), SvgError> {
    Err(SvgError::Unsupported("text (needs the svg-text feature)"))
}

impl Svg {
    /// Parses a self-contained SVG (see [`self_contained`]).
    pub(super) fn parse(bytes: &[u8]) -> Result<Self, SvgError> {
        if has_text(bytes) {
            return Err(SvgError::Unsupported("text (convert it to paths)"));
        }
        let missing = Missing::default();
        let options = usvg::Options {
            image_href_resolver: image_resolver(&missing, false),
            ..Default::default()
        };
        let tree = usvg::Tree::from_data(bytes, &options).map_err(SvgError::Parse)?;
        missing.check()?;
        Ok(Self {
            width: f64::from(tree.size().width()),
            height: f64::from(tree.size().height()),
            root: compile_group(tree.root(), Transform::identity())?,
        })
    }

    /// Canvas size, in SVG user units.
    pub(super) fn size(&self) -> (f64, f64) {
        (self.width, self.height)
    }

    /// Linear color premultiplied by alpha, and alpha, at `(u, v)`: the
    /// canvas covers [0, 1]^2, `v` upward; outside it is transparent.
    pub(super) fn sample(&self, u: f32, v: f32) -> ([f32; 3], f32) {
        let x = f64::from(u) * self.width;
        let y = (1.0 - f64::from(v)) * self.height;
        if !(0.0..self.width).contains(&x) || !(0.0..self.height).contains(&y) {
            return ([0.0; 3], 0.0);
        }
        let mut color = [0f32; 4];
        self.root.paint(x, y, &mut color);
        let alpha = color[3].min(1.0);
        if alpha <= 0.0 {
            return ([0.0; 3], 0.0);
        }
        // Composited in sRGB: decode the straight color, then premultiply.
        let linear =
            [0, 1, 2].map(|i| super::srgb_to_linear((color[i] / alpha).clamp(0.0, 1.0)) * alpha);
        (linear, alpha)
    }
}

/// Compiles `group`, whose descendants' absolute transforms map to canvas
/// coordinates through `base` (the identity for the main tree; for a clip
/// path, mask or pattern, the transform of what uses it, since their
/// content's transforms are relative to them).
fn compile_group(group: &Group, base: Transform) -> Result<GroupItem, SvgError> {
    if !group.filters().is_empty() {
        return Err(SvgError::Unsupported("filters"));
    }
    let transform = base.pre_concat(group.abs_transform());
    let clip = group
        .clip_path()
        .map(|clip| clip_region(clip, transform).map(Arc::new))
        .transpose()?;
    let mask = group
        .mask()
        .map(|mask| compile_mask(mask, transform).map(Arc::new))
        .transpose()?;
    let mut children = Vec::new();
    for node in group.children() {
        match node {
            Node::Group(child) => children.push(Item::Group(compile_group(child, base)?)),
            Node::Path(path) => {
                if !path.is_visible() {
                    continue;
                }
                let transform = base.pre_concat(path.abs_transform());
                let fill = path
                    .fill()
                    .map(|fill| -> Result<_, SvgError> {
                        Ok(FillItem {
                            shape: Shape::new(path.data(), transform, fill.rule()),
                            paint: compile_paint(fill.paint(), transform)?,
                            opacity: fill.opacity().get(),
                        })
                    })
                    .transpose()?;
                let stroke = path
                    .stroke()
                    .map(|stroke| compile_stroke(path.data(), stroke, transform))
                    .transpose()?
                    .flatten();
                let (first, second) = match path.paint_order() {
                    usvg::PaintOrder::FillAndStroke => (fill, stroke),
                    usvg::PaintOrder::StrokeAndFill => (stroke, fill),
                };
                children.extend(first.into_iter().chain(second).map(Item::Fill));
            }
            Node::Image(image) => {
                if image.is_visible() {
                    children.push(compile_image(
                        image,
                        base.pre_concat(image.abs_transform()),
                    )?);
                }
            }
            Node::Text(_) => return Err(SvgError::Unsupported("text (convert it to paths)")),
        }
    }
    Ok(GroupItem {
        isolate: group.should_isolate(),
        opacity: group.opacity().get(),
        blend: group.blend_mode(),
        clip,
        mask,
        children,
    })
}

/// An embedded image whose pixel grid maps to canvas coordinates through
/// `transform` (usvg has resolved its placement and aspect ratio).
fn compile_image(image: &usvg::Image, transform: Transform) -> Result<Item, SvgError> {
    let data = match image.kind() {
        ImageKind::JPEG(data)
        | ImageKind::PNG(data)
        | ImageKind::GIF(data)
        | ImageKind::WEBP(data) => data,
        ImageKind::SVG(tree) => {
            // A nested SVG: its content, clipped to its viewport.
            let size = tree.size();
            let viewport = tiny_skia_path::Rect::from_xywh(0.0, 0.0, size.width(), size.height())
                .ok_or(SvgError::Unsupported("empty nested SVG"))?;
            let viewport = Shape::new(
                &tiny_skia_path::PathBuilder::from_rect(viewport),
                transform,
                FillRule::NonZero,
            );
            return Ok(Item::Group(GroupItem {
                isolate: true,
                opacity: 1.0,
                blend: BlendMode::Normal,
                clip: Some(Arc::new(Clip {
                    shapes: vec![viewport],
                    clip: None,
                })),
                mask: None,
                children: vec![Item::Group(compile_group(tree.root(), transform)?)],
            }));
        }
    };
    let texels = Texels::decode(data).map_err(SvgError::EmbeddedImage)?;
    Ok(Item::Image(ImageItem {
        inverse: Affine::new(transform)
            .inverse()
            .ok_or(SvgError::Unsupported("degenerate image transforms"))?,
        texels,
        smooth: !matches!(
            image.rendering_mode(),
            ImageRendering::OptimizeSpeed | ImageRendering::CrispEdges | ImageRendering::Pixelated
        ),
    }))
}

/// How finely strokes are turned into outlines: `tiny-skia`'s tolerance is
/// a fraction of a device pixel, made 100 times finer here.
const STROKE_PRECISION: f32 = 100.0;

/// The outline of a stroke, as a fill (None for an empty stroke).
fn compile_stroke(
    path: &tiny_skia_path::Path,
    stroke: &usvg::Stroke,
    transform: Transform,
) -> Result<Option<FillItem>, SvgError> {
    let resolution =
        tiny_skia_path::PathStroker::compute_resolution_scale(&transform) * STROKE_PRECISION;
    let style = stroke.to_tiny_skia();
    // The stroker ignores dashes: dash first.
    let dashed;
    let path = match &style.dash {
        Some(dash) => match path.dash(dash, resolution) {
            Some(path) => {
                dashed = path;
                &dashed
            }
            None => return Ok(None),
        },
        None => path,
    };
    let Some(outline) = path.stroke(&style, resolution) else {
        return Ok(None);
    };
    Ok(Some(FillItem {
        // Stroke outlines are drawn with the nonzero rule.
        shape: Shape::new(&outline, transform, FillRule::NonZero),
        paint: compile_paint(stroke.paint(), transform)?,
        opacity: stroke.opacity().get(),
    }))
}

/// A paint used by a path whose transform to canvas coordinates is
/// `transform`.
fn compile_paint(paint: &Paint, transform: Transform) -> Result<PaintSource, SvgError> {
    let inverse = |own: Transform| {
        Affine::new(transform.pre_concat(own))
            .inverse()
            .ok_or(SvgError::Unsupported("degenerate paint transforms"))
    };
    let stops = |stops: &[usvg::Stop], spread: SpreadMethod| Stops {
        stops: stops
            .iter()
            .map(|stop| {
                let color = stop.color();
                let alpha = stop.opacity().get();
                let rgb =
                    [color.red, color.green, color.blue].map(|c| f32::from(c) / 255.0 * alpha);
                (stop.offset().get(), [rgb[0], rgb[1], rgb[2], alpha])
            })
            .collect(),
        spread,
    };
    Ok(match paint {
        Paint::Color(color) => {
            PaintSource::Color([color.red, color.green, color.blue].map(|c| f32::from(c) / 255.0))
        }
        Paint::LinearGradient(gradient) => PaintSource::Linear {
            inverse: inverse(gradient.transform())?,
            start: [gradient.x1(), gradient.y1()].map(f64::from),
            end: [gradient.x2(), gradient.y2()].map(f64::from),
            stops: stops(gradient.stops(), gradient.spread_method()),
        },
        Paint::RadialGradient(gradient) => PaintSource::Radial {
            inverse: inverse(gradient.transform())?,
            center: [gradient.cx(), gradient.cy()].map(f64::from),
            radius: f64::from(gradient.r().get()),
            focal: [gradient.fx(), gradient.fy()].map(f64::from),
            focal_radius: f64::from(gradient.fr().get()),
            stops: stops(gradient.stops(), gradient.spread_method()),
        },
        Paint::Pattern(pattern) => {
            let rect = pattern.rect();
            PaintSource::Pattern {
                inverse: inverse(pattern.transform())?,
                tile: [rect.x(), rect.y(), rect.width(), rect.height()].map(f64::from),
                content: Box::new(compile_group(pattern.root(), Transform::identity())?),
            }
        }
    })
}

/// A mask applied to a group whose transform to canvas coordinates is
/// `transform` (the mask's rectangle and content are in its user space).
fn compile_mask(mask: &usvg::Mask, transform: Transform) -> Result<Mask, SvgError> {
    let region = tiny_skia_path::PathBuilder::from_rect(mask.rect().to_rect());
    Ok(Mask {
        kind: mask.kind(),
        region: Shape::new(&region, transform, FillRule::NonZero),
        content: compile_group(mask.root(), transform)?,
        mask: mask
            .mask()
            .map(|inner| compile_mask(inner, transform).map(Arc::new))
            .transpose()?,
    })
}

/// The region of a clip path applied to an element whose transform to
/// canvas coordinates is `transform`.
fn clip_region(clip: &usvg::ClipPath, transform: Transform) -> Result<Clip, SvgError> {
    let transform = transform.pre_concat(clip.transform());
    let mut shapes = Vec::new();
    clip_shapes(clip.root(), transform, &mut shapes)?;
    let inner = clip
        .clip_path()
        .map(|inner| clip_region(inner, transform).map(Arc::new))
        .transpose()?;
    Ok(Clip {
        shapes,
        clip: inner,
    })
}

fn clip_shapes(
    group: &Group,
    transform: Transform,
    shapes: &mut Vec<Shape>,
) -> Result<(), SvgError> {
    if group.clip_path().is_some() || group.mask().is_some() {
        return Err(SvgError::Unsupported(
            "clip paths or masks inside clip paths",
        ));
    }
    for node in group.children() {
        match node {
            Node::Group(child) => clip_shapes(child, transform, shapes)?,
            Node::Path(path) => {
                if path.is_visible() {
                    // In a clip path, the fill rule is the clip rule.
                    let rule = path.fill().map_or(FillRule::NonZero, |fill| fill.rule());
                    shapes.push(Shape::new(
                        path.data(),
                        transform.pre_concat(path.abs_transform()),
                        rule,
                    ));
                }
            }
            Node::Image(_) | Node::Text(_) => {
                return Err(SvgError::Unsupported("images or text inside clip paths"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
