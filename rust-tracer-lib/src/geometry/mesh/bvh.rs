//! Bounding volume hierarchy over a mesh's triangles: binned surface area
//! heuristic build, flat node array, stack traversal.

use std::ops::Range;

use crate::{UnitVector, Vector};

use super::Triangle;

/// A ray, with what both the box test and the watertight triangle test
/// precompute once per traversal.
pub(super) struct Ray {
    pub(super) origin: Vector,
    pub(super) direction: UnitVector,
    /// 1 / direction, componentwise (+/-inf on a zero component).
    inverse: [f32; 3],
    /// The dominant axis of the direction (kz) and the other two, ordered so
    /// that the shear keeps the winding.
    axes: [usize; 3],
    /// Shear taking the direction to the kz axis: (-dx/dz, -dy/dz) applied
    /// as subtractions, and 1/dz.
    shear: [f32; 3],
}

impl Ray {
    pub(super) fn new(origin: Vector, direction: UnitVector) -> Self {
        let d = [direction.x, direction.y, direction.z];
        let kz = (0..3)
            .max_by(|&i, &j| d[i].abs().total_cmp(&d[j].abs()))
            .unwrap_or(2);
        let mut kx = (kz + 1) % 3;
        let mut ky = (kx + 1) % 3;
        if d[kz] < 0f32 {
            std::mem::swap(&mut kx, &mut ky);
        }
        Self {
            origin,
            direction,
            inverse: d.map(|c| 1f32 / c),
            axes: [kx, ky, kz],
            shear: [d[kx] / d[kz], d[ky] / d[kz], 1f32 / d[kz]],
        }
    }

    /// `p` in the ray's sheared frame: the ray runs along +z from (0, 0, 0)
    /// and z is the ray parameter t of p's projection.
    pub(super) fn local(&self, p: Vector) -> [f32; 3] {
        let p = [
            p.x - self.origin.x,
            p.y - self.origin.y,
            p.z - self.origin.z,
        ];
        let [kx, ky, kz] = self.axes;
        let rel = |k: usize| p[k];
        let z = rel(kz);
        [
            rel(kx) - self.shear[0] * z,
            rel(ky) - self.shear[1] * z,
            self.shear[2] * z,
        ]
    }

    /// Entry distance into the box if the ray meets it before `t_max`.
    ///
    /// A direction component too small to invert (0, or so tiny that the
    /// slab is reached beyond any scene) makes the ray parallel to that
    /// slab: it is in it for every t if the origin is, never otherwise. The
    /// slab arithmetic cannot say so, since an origin on a slab plane gives
    /// 0 * inf = NaN there, which would cull the box (a ray along a face of
    /// an axis-aligned mesh). The far bound is widened by 2*gamma(3) (Ize,
    /// "Robust BVH Ray Traversal", JCGT 2013) so that rounding in the slab
    /// distances cannot cull a box the watertight triangle test would hit.
    fn enters(&self, node: &Node, t_max: f32) -> Option<f32> {
        const WIDEN: f32 = 1f32 + 2f32 * 3f32 * f32::EPSILON / (1f32 - 3f32 * f32::EPSILON);
        let origin = [self.origin.x, self.origin.y, self.origin.z];
        let mut near = 0f32;
        let mut far = t_max;
        for (k, origin) in origin.into_iter().enumerate() {
            let low = node.min[k] - origin;
            let high = node.max[k] - origin;
            if self.inverse[k].is_infinite() {
                if low > 0f32 || high < 0f32 {
                    return None;
                }
                continue;
            }
            let t0 = low * self.inverse[k];
            let t1 = high * self.inverse[k];
            near = near.max(t0.min(t1));
            far = far.min(t0.max(t1) * WIDEN);
        }
        (near <= far).then_some(near)
    }
}

/// 32 bytes. A leaf when `count > 0` (triangles `index..index + count`);
/// otherwise its children are nodes `index` and `index + 1`.
#[derive(Debug, Clone, Copy)]
struct Node {
    min: [f32; 3],
    max: [f32; 3],
    index: u32,
    count: u32,
}

#[derive(Debug, Default)]
pub(super) struct Bvh {
    nodes: Vec<Node>,
}

/// Bins of the SAH sweep along the split axis.
const BINS: usize = 16;
/// A node with more triangles is always split.
const MAX_LEAF: usize = 8;
/// SAH cost of a traversal step relative to a triangle test.
const TRAVERSAL_COST: f32 = 1f32;
/// Depth past which nodes split at the median: halving bounds the remaining
/// depth by log2(triangles) <= 32, so the traversal stack never overflows.
const SAH_DEPTH: usize = 32;
const STACK: usize = SAH_DEPTH + 33;

#[derive(Debug, Clone, Copy)]
struct Bounds {
    min: [f32; 3],
    max: [f32; 3],
}

impl Bounds {
    const EMPTY: Self = Self {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };

    fn grow(&mut self, p: [f32; 3]) {
        for ((min, max), c) in self.min.iter_mut().zip(&mut self.max).zip(p) {
            *min = min.min(c);
            *max = max.max(c);
        }
    }

    fn union(mut self, other: &Self) -> Self {
        self.grow(other.min);
        self.grow(other.max);
        self
    }

    /// Half the surface area (the SAH only compares ratios); 0 when empty.
    fn half_area(&self) -> f32 {
        let [x, y, z] = std::array::from_fn(|k| (self.max[k] - self.min[k]).max(0f32));
        x * y + y * z + z * x
    }
}

/// What the build needs of a triangle.
struct Item {
    bounds: Bounds,
    centroid: [f32; 3],
}

impl Bvh {
    /// The hierarchy, and the triangles reordered so that every leaf is a
    /// contiguous range.
    pub(super) fn build(triangles: Vec<Triangle>) -> (Self, Vec<Triangle>) {
        if triangles.is_empty() {
            return (Self::default(), triangles);
        }
        let items: Vec<Item> = triangles
            .iter()
            .map(|triangle| {
                let mut bounds = Bounds::EMPTY;
                for p in triangle.vertices {
                    bounds.grow([p.x, p.y, p.z]);
                }
                let centroid = std::array::from_fn(|k| 0.5f32 * (bounds.min[k] + bounds.max[k]));
                Item { bounds, centroid }
            })
            .collect();
        let mut order: Vec<usize> = (0..triangles.len()).collect();
        let mut nodes = Vec::with_capacity(2 * triangles.len());
        nodes.push(Node {
            min: [0f32; 3],
            max: [0f32; 3],
            index: 0,
            count: 0,
        });
        Self::split(&items, &mut order, 0, 0, 0, &mut nodes);

        let mut slots: Vec<Option<Triangle>> = triangles.into_iter().map(Some).collect();
        let triangles = order
            .iter()
            .map(|&i| slots[i].take().expect("each triangle is placed once"))
            .collect();
        (Self { nodes }, triangles)
    }

    /// Fills node `node` with `order[start..start + len]` (len = slice
    /// length), splitting it recursively.
    fn split(
        items: &[Item],
        order: &mut [usize],
        start: usize,
        node: usize,
        depth: usize,
        nodes: &mut Vec<Node>,
    ) {
        let bounds = order
            .iter()
            .fold(Bounds::EMPTY, |b, &i| b.union(&items[i].bounds));
        let leaf = Node {
            min: bounds.min,
            max: bounds.max,
            index: start as u32,
            count: order.len() as u32,
        };
        if order.len() <= 1 {
            nodes[node] = leaf;
            return;
        }

        let mut centroids = Bounds::EMPTY;
        for &i in order.iter() {
            centroids.grow(items[i].centroid);
        }
        let extent: [f32; 3] = std::array::from_fn(|k| centroids.max[k] - centroids.min[k]);
        let axis = (0..3)
            .max_by(|&i, &j| extent[i].total_cmp(&extent[j]))
            .unwrap_or(0);

        let mid = match (depth < SAH_DEPTH && extent[axis] > 0f32)
            .then(|| Self::sah(items, order, &bounds, &centroids, axis))
            .flatten()
        {
            Some(mid) => mid,
            None if order.len() <= MAX_LEAF => {
                nodes[node] = leaf;
                return;
            }
            // Past the SAH depth, or all centroids coincide, or no split
            // beats a leaf yet the node is too big: halve it.
            None => {
                let mid = order.len() / 2;
                order.select_nth_unstable_by(mid, |&i, &j| {
                    items[i].centroid[axis].total_cmp(&items[j].centroid[axis])
                });
                mid
            }
        };

        let children = nodes.len();
        let placeholder = nodes[node];
        nodes.push(placeholder);
        nodes.push(placeholder);
        nodes[node] = Node {
            min: bounds.min,
            max: bounds.max,
            index: children as u32,
            count: 0,
        };
        let (left, right) = order.split_at_mut(mid);
        Self::split(items, left, start, children, depth + 1, nodes);
        Self::split(items, right, start + mid, children + 1, depth + 1, nodes);
    }

    /// Partitions `order` along `axis` at the cheapest bin boundary, and
    /// returns the split point; `None` when a leaf is cheaper (and allowed).
    // `!(<)`: a NaN cost (flat node, 0/0) takes the leaf.
    #[expect(clippy::neg_cmp_op_on_partial_ord)]
    fn sah(
        items: &[Item],
        order: &mut [usize],
        bounds: &Bounds,
        centroids: &Bounds,
        axis: usize,
    ) -> Option<usize> {
        let low = centroids.min[axis];
        let scale = BINS as f32 / (centroids.max[axis] - low);
        let bin = |i: usize| (((items[i].centroid[axis] - low) * scale) as usize).min(BINS - 1);

        let mut counts = [0usize; BINS];
        let mut boxes = [Bounds::EMPTY; BINS];
        for &i in order.iter() {
            let b = bin(i);
            counts[b] += 1;
            boxes[b] = boxes[b].union(&items[i].bounds);
        }

        // Areas and counts left of each boundary, then sweep from the right.
        let mut left_area = [0f32; BINS - 1];
        let mut left_count = [0usize; BINS - 1];
        let (mut acc, mut n) = (Bounds::EMPTY, 0usize);
        for b in 0..BINS - 1 {
            acc = acc.union(&boxes[b]);
            n += counts[b];
            left_area[b] = acc.half_area();
            left_count[b] = n;
        }
        let (mut acc, mut n) = (Bounds::EMPTY, 0usize);
        let mut best: Option<(f32, usize)> = None;
        for b in (0..BINS - 1).rev() {
            acc = acc.union(&boxes[b + 1]);
            n += counts[b + 1];
            if left_count[b] == 0 || n == 0 {
                continue;
            }
            let cost = left_area[b] * left_count[b] as f32 + acc.half_area() * n as f32;
            if best.is_none_or(|(c, _)| cost < c) {
                best = Some((cost, b));
            }
        }
        let (cost, boundary) = best?;
        let parent = bounds.half_area();
        let split_cost = TRAVERSAL_COST + cost / parent;
        if order.len() <= MAX_LEAF && !(split_cost < order.len() as f32) {
            return None;
        }

        // In-place partition: bins <= boundary to the left.
        let mut mid = 0;
        for k in 0..order.len() {
            if bin(order[k]) <= boundary {
                order.swap(k, mid);
                mid += 1;
            }
        }
        Some(mid)
    }

    /// Calls `visit(range, t_max)` on every leaf the ray enters before
    /// `t_max`, nearer children first; `visit` may lower `t_max` (closest
    /// hit) or leave it at infinity (every crossing).
    pub(super) fn traverse(&self, ray: &Ray, mut visit: impl FnMut(Range<usize>, &mut f32)) {
        let Some(root) = self.nodes.first() else {
            return;
        };
        let mut t_max = f32::INFINITY;
        if ray.enters(root, t_max).is_none() {
            return;
        }
        let mut stack = [0u32; STACK];
        let mut top = 0;
        let mut node = root;
        loop {
            if node.count > 0 {
                let start = node.index as usize;
                visit(start..start + node.count as usize, &mut t_max);
            } else {
                let index = node.index as usize;
                let (left, right) = (&self.nodes[index], &self.nodes[index + 1]);
                match (ray.enters(left, t_max), ray.enters(right, t_max)) {
                    (Some(l), Some(r)) => {
                        let (near, far) = if l <= r {
                            (index, index + 1)
                        } else {
                            (index + 1, index)
                        };
                        stack[top] = far as u32;
                        top += 1;
                        node = &self.nodes[near];
                        continue;
                    }
                    (Some(_), None) => {
                        node = left;
                        continue;
                    }
                    (None, Some(_)) => {
                        node = right;
                        continue;
                    }
                    (None, None) => {}
                }
            }
            // Pop the next pending node still in reach.
            loop {
                if top == 0 {
                    return;
                }
                top -= 1;
                let candidate = &self.nodes[stack[top] as usize];
                if ray.enters(candidate, t_max).is_some() {
                    node = candidate;
                    break;
                }
            }
        }
    }

    #[cfg(test)]
    pub(super) fn depth(&self) -> usize {
        fn depth(nodes: &[Node], node: usize) -> usize {
            let n = &nodes[node];
            if n.count > 0 {
                1
            } else {
                1 + depth(nodes, n.index as usize).max(depth(nodes, n.index as usize + 1))
            }
        }
        if self.nodes.is_empty() {
            0
        } else {
            depth(&self.nodes, 0)
        }
    }
}
