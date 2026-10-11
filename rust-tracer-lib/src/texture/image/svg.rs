//! SVG images evaluated exactly, without rasterizing: whether a point is
//! inside a filled path is decided from the path's Bezier curves themselves
//! (no flattening tolerance), so the image is sharp at any magnification.
//! The path tracer's sub-pixel jitter integrates the coverage of each pixel,
//! which anti-aliases the edges.
//!
//! Supported: paths (and shapes, which `usvg` turns into paths) with solid
//! fills, fill rules, fill and group opacity, transforms and clip paths.
//! Anything else (strokes, gradients, patterns, masks, filters, blend modes,
//! embedded images, text) is an error rather than a silently wrong image.
//! Group opacity is applied to each child separately, which differs from SVG
//! only where children of a translucent group overlap.

use std::sync::Arc;

use usvg::tiny_skia_path::{PathSegment, Point};
use usvg::{FillRule, Group, Node, Paint, Transform};

/// A parsed SVG image: its canvas size, and its filled paths in painting
/// order (bottom first), in canvas coordinates (y down).
pub(super) struct Svg {
    width: f64,
    height: f64,
    layers: Vec<Layer>,
}

/// One filled path.
struct Layer {
    /// Linear RGB.
    color: [f32; 3],
    opacity: f32,
    shape: Shape,
    /// Clip regions the path is drawn within (all of them).
    clips: Vec<Arc<Clip>>,
}

/// A clip path: the union of its shapes, within its own clip path if any.
struct Clip {
    shapes: Vec<Shape>,
    clip: Option<Arc<Clip>>,
}

impl Clip {
    fn contains(&self, x: f64, y: f64) -> bool {
        self.shapes.iter().any(|shape| shape.contains(x, y))
            && self.clip.as_ref().is_none_or(|clip| clip.contains(x, y))
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
    fn new(path: &usvg::tiny_skia_path::Path, transform: Transform, rule: FillRule) -> Self {
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

impl Layer {
    fn covers(&self, x: f64, y: f64) -> bool {
        self.shape.contains(x, y) && self.clips.iter().all(|clip| clip.contains(x, y))
    }
}

/// What makes an SVG unusable as a texture.
#[derive(Debug)]
pub enum SvgError {
    Parse(usvg::Error),
    Unsupported(&'static str),
}

impl std::fmt::Display for SvgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SvgError::Parse(error) => write!(f, "invalid SVG: {error}"),
            SvgError::Unsupported(feature) => write!(f, "unsupported SVG feature: {feature}"),
        }
    }
}

impl std::error::Error for SvgError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SvgError::Parse(error) => Some(error),
            SvgError::Unsupported(_) => None,
        }
    }
}

impl Svg {
    pub(super) fn parse(bytes: &[u8]) -> Result<Self, SvgError> {
        // Without its `text` feature, usvg silently drops text: refuse it.
        if let Ok(text) = std::str::from_utf8(bytes)
            && let Ok(document) = usvg::roxmltree::Document::parse(text)
            && document.descendants().any(|node| node.has_tag_name("text"))
        {
            return Err(SvgError::Unsupported("text (convert it to paths)"));
        }
        let tree =
            usvg::Tree::from_data(bytes, &usvg::Options::default()).map_err(SvgError::Parse)?;
        let mut layers = Vec::new();
        collect(tree.root(), 1.0, &[], &mut layers)?;
        Ok(Self {
            width: f64::from(tree.size().width()),
            height: f64::from(tree.size().height()),
            layers,
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
        let mut color = [0f32; 3];
        let mut alpha = 0f32;
        if !(0.0..self.width).contains(&x) || !(0.0..self.height).contains(&y) {
            return (color, alpha);
        }
        // Front to back, until opaque.
        for layer in self.layers.iter().rev() {
            if alpha >= 1.0 {
                break;
            }
            if layer.covers(x, y) {
                let a = layer.opacity * (1.0 - alpha);
                for (channel, value) in color.iter_mut().zip(layer.color) {
                    *channel += a * value;
                }
                alpha += a;
            }
        }
        (color, alpha)
    }
}

/// Appends the filled paths of `group` to `layers`, in painting order.
fn collect(
    group: &Group,
    opacity: f32,
    clips: &[Arc<Clip>],
    layers: &mut Vec<Layer>,
) -> Result<(), SvgError> {
    if group.mask().is_some() {
        return Err(SvgError::Unsupported("masks"));
    }
    if !group.filters().is_empty() {
        return Err(SvgError::Unsupported("filters"));
    }
    if group.blend_mode() != usvg::BlendMode::Normal {
        return Err(SvgError::Unsupported("blend modes"));
    }
    let opacity = opacity * group.opacity().get();
    let mut clips = clips.to_vec();
    if let Some(clip) = group.clip_path() {
        clips.push(Arc::new(clip_region(clip, group.abs_transform())?));
    }
    for node in group.children() {
        match node {
            Node::Group(child) => collect(child, opacity, &clips, layers)?,
            Node::Path(path) => {
                if !path.is_visible() {
                    continue;
                }
                if path.stroke().is_some() {
                    return Err(SvgError::Unsupported("strokes"));
                }
                let Some(fill) = path.fill() else { continue };
                let Paint::Color(color) = fill.paint() else {
                    return Err(SvgError::Unsupported("gradients and patterns"));
                };
                layers.push(Layer {
                    color: [color.red, color.green, color.blue].map(super::srgb_to_linear),
                    opacity: opacity * fill.opacity().get(),
                    shape: Shape::new(path.data(), path.abs_transform(), fill.rule()),
                    clips: clips.clone(),
                });
            }
            Node::Image(_) => return Err(SvgError::Unsupported("embedded images")),
            Node::Text(_) => return Err(SvgError::Unsupported("text (convert it to paths)")),
        }
    }
    Ok(())
}

/// The region of a clip path applied to an element whose absolute transform
/// is `transform` (clip paths are in the user space of the element using
/// them, and their content's transforms are relative to the clip path).
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
