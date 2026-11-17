use std::path::{Path, PathBuf};
use std::sync::Arc;

use rand::RngExt;
use serde::{Deserialize, Serialize};

use crate::{Geometry, GeometryPrimitive, Hit, HitCaster, UnitVector, Vector, select};

mod bvh;
mod load;
#[cfg(test)]
mod tests;

use bvh::{Bvh, Ray};

mod definition {
    use std::path::PathBuf;

    use serde::{Deserialize, Serialize};

    use crate::Vector;

    /// No `deny_unknown_fields`: a mesh is flattened into an `Object` next to
    /// material and texture keys, which serde would otherwise reject. For the
    /// same reason the keys avoid the texture ones (`file`, `scale`, ...),
    /// which every flattened part of the object sees.
    #[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(default)]
    pub(crate) struct Mesh {
        /// An OBJ or STL file, relative to the scene file; replaced by the
        /// inline data below when loaded.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model: Option<PathBuf>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        pub vertices: Vec<Vector>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        pub triangles: Vec<[u32; 3]>,
        /// Per-vertex normals (used when `smooth`), empty or one per vertex.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        pub normals: Vec<Vector>,
        /// Per-vertex surface coordinates, empty or one per vertex.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        pub uvs: Vec<[f32; 2]>,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        pub smooth: bool,
        pub position: Vector,
        /// Degrees about x, then y, then z (world axes).
        pub rotation: Vector,
        pub scaling: Scaling,
    }

    /// One factor for every axis, or one per axis.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[serde(untagged)]
    pub(crate) enum Scaling {
        Uniform(f32),
        Axes(Vector),
    }

    impl Default for Scaling {
        fn default() -> Self {
            Self::Uniform(1f32)
        }
    }

    impl Scaling {
        pub fn axes(self) -> [f64; 3] {
            match self {
                Scaling::Uniform(s) => [f64::from(s); 3],
                Scaling::Axes(v) => [v.x, v.y, v.z].map(f64::from),
            }
        }
    }
}

/// Triangle mesh: `vertices` indexed by `triangles`, placed in the scene by
/// `scaling`, then `rotation` (degrees about x, then y, then z), then
/// `position`. Read from an OBJ or STL `model` file (see [`Mesh::load`]), or
/// given inline:
///
/// ```yaml
/// - type: mesh
///   model: bunny.obj          # or inline:
///   # vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
///   # triangles: [[0, 1, 2]]
///   smooth: true              # interpolate vertex normals
///   position: [0, -1, 5]
///   rotation: [0, 90, 0]
///   scaling: 10               # or [sx, sy, sz]
/// ```
///
/// Faces are flat-shaded by default; `smooth: true` interpolates the
/// per-vertex `normals` (the OBJ `vn`), or area-weighted averages of the
/// face normals when there are none. Rays always spawn off the true faces
/// (see [`Hit::geometric`]): smoothing changes how light scatters, never
/// which side of the surface a ray is on. `front` is the side the
/// counter-clockwise winding faces (outward for well-formed OBJ and STL
/// files), which is what dielectric meshes rely on.
///
/// The surface coordinates are the per-vertex `uvs` (the OBJ `vt`)
/// interpolated, or else the barycentric coordinates of the hit in its
/// triangle.
///
/// Built at deserialization: a bounding volume hierarchy (binned SAH) over
/// the triangles, Woop et al.'s watertight intersection (rays cannot slip
/// through shared edges, which a closed glass mesh would leak through), and
/// an area table for light sampling. Zero-area triangles are dropped; a mesh
/// without area (including `Mesh::default()` and a `model` not loaded yet)
/// is inert: never hit, never sampled.
#[derive(Clone, Serialize, Deserialize)]
#[serde(try_from = "definition::Mesh", into = "definition::Mesh")]
pub struct Mesh {
    definition: Arc<definition::Mesh>,
    built: Arc<Built>,
}

struct Built {
    /// In BVH leaf order.
    triangles: Vec<Triangle>,
    bvh: Bvh,
    /// Cumulative triangle areas over the total, last = 1 (f64: an f32 sum
    /// over a million triangles quantizes the small ones' probabilities).
    cdf: Vec<f64>,
    area: f32,
}

impl Mesh {
    /// A flat-shaded mesh from world-space vertices; fails on an index out of
    /// range or a non-finite coordinate.
    pub fn new(vertices: Vec<Vector>, triangles: Vec<[u32; 3]>) -> Result<Self, MeshError> {
        definition::Mesh {
            vertices,
            triangles,
            ..Default::default()
        }
        .try_into()
    }

    /// Like [`Mesh::new`], with per-vertex normals interpolated across the
    /// faces.
    pub fn smooth(
        vertices: Vec<Vector>,
        triangles: Vec<[u32; 3]>,
        normals: Vec<Vector>,
    ) -> Result<Self, MeshError> {
        definition::Mesh {
            vertices,
            triangles,
            normals,
            smooth: true,
            ..Default::default()
        }
        .try_into()
    }

    /// Reads the `model` file, relative to `base` (the scene file's folder),
    /// as OBJ or STL (told apart by the extension, else by the content). From
    /// then on the mesh carries the data inline, also when serialized. A mesh
    /// without `model` is left as is.
    pub fn load(&mut self, base: &Path) -> Result<(), MeshError> {
        let Some(model) = &self.definition.model else {
            return Ok(());
        };
        let path = base.join(model);
        let bytes = std::fs::read(&path).map_err(|source| MeshError::Read {
            path: path.clone(),
            source,
        })?;
        let data = load::parse(&path, &bytes).map_err(|message| MeshError::Parse {
            path: path.clone(),
            message,
        })?;
        let definition = definition::Mesh {
            model: None,
            vertices: data.vertices,
            triangles: data.triangles,
            normals: data.normals,
            uvs: data.uvs,
            ..(*self.definition).clone()
        };
        *self = definition.try_into().map_err(|error| match error {
            MeshError::Invalid(message) => MeshError::Parse { path, message },
            error => error,
        })?;
        Ok(())
    }

    /// Number of (non-degenerate) triangles.
    pub fn len(&self) -> usize {
        self.built.triangles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.built.triangles.is_empty()
    }

    /// Total surface area.
    pub fn area(&self) -> f32 {
        self.built.area
    }

    fn closest(&self, origin: Vector, direction: UnitVector) -> Option<(f32, &Triangle)> {
        let ray = Ray::new(origin, direction);
        let triangles = &self.built.triangles;
        let mut best: Option<(f32, usize)> = None;
        self.built.bvh.traverse(&ray, |range, t_max| {
            for i in range {
                if let Some(t) = triangles[i].intersect(&ray)
                    && t < *t_max
                {
                    *t_max = t;
                    best = Some((t, i));
                }
            }
        });
        best.map(|(t, i)| (t, &triangles[i]))
    }

    // Deliberate `!(>)`: NaN-safe degenerate check, as in `Quad::hit`.
    #[expect(clippy::neg_cmp_op_on_partial_ord)]
    fn target_pdf(&self, origin: Vector, scattered: UnitVector) -> f32 {
        if !(self.built.area > 0f32) {
            return 0f32;
        }
        // Uniform-area sampling seen in solid angle. A direction is produced
        // by every surface point along its ray, so its density sums
        // t^2 / (A * |cos|) over all the crossings, not just the first: a
        // non-convex or closed mesh is crossed several times (counting only
        // the nearest would integrate to the visible fraction of the area,
        // 1/6 for a box seen face-on). A ray exactly through a shared edge
        // counts it once per face (the watertight test reports it to both):
        // measure-zero directions, which the sampler and the BRDFs never
        // produce, so no bias.
        let ray = Ray::new(origin, scattered);
        let triangles = &self.built.triangles;
        let mut sum = 0f32;
        self.built.bvh.traverse(&ray, |range, _| {
            for triangle in &triangles[range] {
                // A crossing found in f32 at an f32-zero cosine (the ray
                // grazing the face's plane) has no finite density.
                let cos = scattered.dot(triangle.normal).abs();
                if let Some(t) = triangle.intersect(&ray)
                    && cos > 0f32
                {
                    sum += t * t / cos;
                }
            }
        });
        sum / self.built.area
    }

    fn build(definition: definition::Mesh) -> Result<Self, MeshError> {
        let count = definition.vertices.len();
        if definition.model.is_some() && count > 0 {
            return Err(MeshError::Invalid(
                "a mesh takes either a `model` file or inline `vertices`, not both".into(),
            ));
        }
        if !definition.normals.is_empty() && definition.normals.len() != count {
            return Err(MeshError::Invalid(format!(
                "{} normals for {count} vertices",
                definition.normals.len()
            )));
        }
        if !definition.uvs.is_empty() && definition.uvs.len() != count {
            return Err(MeshError::Invalid(format!(
                "{} uvs for {count} vertices",
                definition.uvs.len()
            )));
        }
        if let Some(index) = definition
            .triangles
            .iter()
            .flatten()
            .find(|&&i| i as usize >= count)
        {
            return Err(MeshError::Invalid(format!(
                "vertex index {index} out of range ({count} vertices)"
            )));
        }
        let finite = |v: &Vector| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
        if !definition.vertices.iter().all(finite) {
            return Err(MeshError::Invalid("non-finite vertex coordinate".into()));
        }
        let transform = Transform::new(&definition)?;

        let vertices: Vec<Vector> = definition
            .vertices
            .iter()
            .map(|&v| transform.point(v))
            .collect();
        // A mirroring transform reverses the winding: swap two corners so
        // the geometric normal keeps facing the side the model's did.
        let corners = |[a, b, c]: [u32; 3]| {
            let [a, b, c] = [a, b, c].map(|i| i as usize);
            select(transform.mirrors, [a, c, b], [a, b, c])
        };
        let normals: Option<Vec<Vector>> = match (definition.smooth, definition.normals.is_empty())
        {
            (false, _) => None,
            (true, false) => Some(
                definition
                    .normals
                    .iter()
                    .map(|&n| transform.normal(n))
                    .collect(),
            ),
            (true, true) => {
                // Area-weighted: the unnormalized face normal is twice the
                // face's area along its direction.
                let mut normals = vec![Vector::default(); vertices.len()];
                for &triangle in &definition.triangles {
                    let [a, b, c] = corners(triangle);
                    let n = (vertices[b] - vertices[a]).cross(vertices[c] - vertices[a]);
                    if n.norm2().is_finite() {
                        for i in [a, b, c] {
                            normals[i] += n;
                        }
                    }
                }
                Some(normals)
            }
        };
        let uvs = (!definition.uvs.is_empty()).then_some(&definition.uvs);

        let triangles: Vec<Triangle> = definition
            .triangles
            .iter()
            .filter_map(|&triangle| {
                let [a, b, c] = corners(triangle);
                Triangle::new(
                    [vertices[a], vertices[b], vertices[c]],
                    normals.as_ref().map(|n| [n[a], n[b], n[c]]),
                    uvs.map(|uv| [uv[a], uv[b], uv[c]]),
                )
            })
            .collect();

        let (bvh, triangles) = Bvh::build(triangles);

        let mut sum = 0f64;
        let mut cdf: Vec<f64> = triangles
            .iter()
            .map(|t| {
                sum += f64::from(t.area);
                sum
            })
            .collect();
        cdf.iter_mut().for_each(|c| *c /= sum);
        if let Some(last) = cdf.last_mut() {
            *last = 1f64;
        }

        Ok(Self {
            definition: Arc::new(definition),
            built: Arc::new(Built {
                triangles,
                bvh,
                cdf,
                area: sum as f32,
            }),
        })
    }
}

impl TryFrom<definition::Mesh> for Mesh {
    type Error = MeshError;

    fn try_from(value: definition::Mesh) -> Result<Self, Self::Error> {
        Self::build(value)
    }
}

impl From<Mesh> for definition::Mesh {
    fn from(value: Mesh) -> Self {
        Arc::unwrap_or_clone(value.definition)
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self::build(definition::Mesh::default()).expect("the empty mesh is valid")
    }
}

/// Two meshes are equal when they are defined alike.
impl PartialEq for Mesh {
    fn eq(&self, other: &Self) -> bool {
        self.definition == other.definition
    }
}

/// Counts instead of the (possibly millions of) triangles.
impl std::fmt::Debug for Mesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mesh")
            .field("model", &self.definition.model)
            .field("vertices", &self.definition.vertices.len())
            .field("triangles", &self.built.triangles.len())
            .field("smooth", &self.definition.smooth)
            .field("area", &self.built.area)
            .finish()
    }
}

/// For the scene's RNG only (`dyn Rng`): a generic `R` would overlap, for
/// the coherence checker, with the blanket impl lifting leaf primitives.
impl Geometry for Mesh {
    fn hit<'a>(
        &'a self,
        origin: Vector,
        direction: UnitVector,
        _rng: &mut (dyn rand::Rng + 'static),
    ) -> Option<(f32, &'a dyn HitCaster)> {
        self.closest(origin, direction)
            .map(|(t, triangle)| (t, triangle as &dyn HitCaster))
    }

    /// Only for callers holding the mesh rather than the struck triangle
    /// (the scene resolves hits on the [`HitCaster`]): traces the ray again
    /// to find which face `distance` lies on.
    fn to_hit(&self, origin: Vector, direction: UnitVector, distance: f32) -> Hit {
        match self.closest(origin, direction) {
            Some((_, triangle)) => triangle.surface(origin, direction, distance),
            None => Hit {
                point: origin + distance * direction,
                distance,
                ..Default::default()
            },
        }
    }

    // Deliberate `!(>)`: NaN-safe degenerate check, as in `Quad::hit`.
    #[expect(clippy::neg_cmp_op_on_partial_ord)]
    fn target_random(&self, origin: Vector, rng: &mut (dyn rand::Rng + 'static)) -> UnitVector {
        if !(self.built.area > 0f32) {
            return UnitVector::default();
        }
        // A triangle by area, then a uniform point on it: uniform over the
        // whole surface. `partition_point` skips past equal cumulative values,
        // so no zero-width step can be drawn.
        let u = rng.random::<f64>();
        let cdf = &self.built.cdf;
        let index = cdf.partition_point(|&c| c <= u).min(cdf.len() - 1);
        let triangle = &self.built.triangles[index];

        // Square-root warp of the unit square onto the triangle
        // (Shirley/Osres): area-uniform barycentrics.
        let s = rng.random::<f32>().sqrt();
        let r = rng.random::<f32>();
        let [p0, p1, p2] = triangle.vertices;
        let point = p0 + (s * (1f32 - r)) * (p1 - p0) + (s * r) * (p2 - p0);
        (point - origin).normalized()
    }

    fn target_pdf(&self, origin: Vector, scattered: UnitVector) -> f32 {
        self.target_pdf(origin, scattered)
    }
}

/// One face, in world space, with what its hit needs: the mesh hands it out
/// as the [`HitCaster`] of a hit.
#[derive(Debug, Clone)]
struct Triangle {
    vertices: [Vector; 3],
    /// normalize((p1 - p0) x (p2 - p0)): the counter-clockwise side.
    normal: UnitVector,
    /// n / |n|^2 for that unnormalized n: dotted with p x e2 and e1 x p it
    /// yields the barycentric coordinates (b1, b2) of p (relative to p0).
    w: Vector,
    area: f32,
    /// |p0| + |p1 - p0| + |p2 - p0| bounds the coordinate magnitude of every
    /// point of the face: the scale the hit guard and spawn offset are
    /// relative to (as for the quad).
    scale: f32,
    /// Per-corner shading normals (unnormalized: interpolation normalizes).
    normals: Option<[Vector; 3]>,
    uvs: Option<[[f32; 2]; 3]>,
}

impl Triangle {
    /// `None` for a face without area (or with a non-finite one).
    fn new(
        vertices: [Vector; 3],
        normals: Option<[Vector; 3]>,
        uvs: Option<[[f32; 2]; 3]>,
    ) -> Option<Self> {
        let [p0, p1, p2] = vertices;
        let (e1, e2) = (p1 - p0, p2 - p0);
        let n = e1.cross(e2);
        let n2 = n.norm2();
        if !(n2 > 0f32 && n2.is_finite()) {
            return None;
        }
        Some(Self {
            vertices,
            normal: n.normalized(),
            w: n / n2,
            area: 0.5f32 * n2.sqrt(),
            scale: p0.norm() + e1.norm() + e2.norm(),
            normals,
            uvs,
        })
    }

    /// Watertight ray-triangle intersection (Woop, Benthin, Wald, JCGT
    /// 2013): the triangle is sheared into the ray's frame, where the edge
    /// functions of adjacent faces share their exact operands, so a ray
    /// through an edge or vertex hits at least one of the faces around it.
    /// Two-sided.
    ///
    /// The edge functions decide whether the face is hit; the distance is
    /// the plane's, as for the quad. Woop's own t interpolates the corners'
    /// sheared depths with edge functions that cancel when the corners are
    /// far from the hit relative to t (~20 ulps of the corner distance, 4e-4
    /// on a wall-sized face), where n.(p0 - o) / n.d keeps the ulps of the
    /// plane distance.
    fn intersect(&self, ray: &Ray) -> Option<f32> {
        let [a, b, c] = self.vertices.map(|p| ray.local(p));
        let mut u = c[0] * b[1] - c[1] * b[0];
        let mut v = a[0] * c[1] - a[1] * c[0];
        let mut w = b[0] * a[1] - b[1] * a[0];
        // An edge function rounding to exactly 0 is where f32 can no longer
        // tell the sides apart: decide in f64 (as the paper does).
        if u == 0f32 || v == 0f32 || w == 0f32 {
            let f = |x: f32| f64::from(x);
            u = (f(c[0]) * f(b[1]) - f(c[1]) * f(b[0])) as f32;
            v = (f(a[0]) * f(c[1]) - f(a[1]) * f(c[0])) as f32;
            w = (f(b[0]) * f(a[1]) - f(b[1]) * f(a[0])) as f32;
        }
        if (u < 0f32 || v < 0f32 || w < 0f32) && (u > 0f32 || v > 0f32 || w > 0f32) {
            return None;
        }
        if u + v + w == 0f32 {
            return None;
        }
        let t = self.normal.dot(self.vertices[0] - ray.origin) / ray.direction.dot(self.normal);
        // Relative guard against re-hitting the face an origin numerically
        // sits on (see `Quad::hit`); also rejects faces behind the ray, and a
        // NaN or infinite t (a hit decided in the sheared frame while the
        // plane's cosine rounds to 0).
        if !(t > crate::HIT_EPSILON_REL * self.scale && t.is_finite()) {
            return None;
        }
        Some(t)
    }
}

impl<R: crate::Rng + ?Sized> GeometryPrimitive<R> for Triangle {
    fn hit(&self, origin: Vector, direction: UnitVector, _rng: &mut R) -> Option<f32> {
        self.intersect(&Ray::new(origin, direction))
    }

    fn to_hit(&self, origin: Vector, direction: UnitVector, distance: f32) -> Hit {
        self.surface(origin, direction, distance)
    }
}

impl Triangle {
    /// The hit `distance` along the ray (see [`GeometryPrimitive::to_hit`]).
    fn surface(&self, origin: Vector, direction: UnitVector, distance: f32) -> Hit {
        let [p0, p1, p2] = self.vertices;
        let (e1, e2) = (p1 - p0, p2 - p0);

        // Barycentrics of the ray point, clamped into the face (a hit through
        // an edge may round just outside it).
        let p = origin + distance * direction - p0;
        let b1 = self.w.dot(p.cross(e2)).clamp(0f32, 1f32);
        let b2 = self.w.dot(e1.cross(p)).clamp(0f32, 1f32 - b1);
        let b0 = 1f32 - b1 - b2;

        // Rebuilt from the barycentrics: the rounding of p0 + b1*e1 + b2*e2
        // is a few ulps of `scale` off the plane whatever the ray, whereas
        // origin + t*d carries the ulps of a far origin and of t.
        let point = p0 + b1 * e1 + b2 * e2;

        let front = direction.dot(self.normal) < 0f32;
        let geometric = select(front, self.normal, -self.normal);

        // The interpolated normal, turned to the side the ray came from. It
        // must still oppose the ray (materials rely on it: refraction and
        // reflection about a normal facing away are meaningless); where it
        // does not, near silhouettes, the face's own normal takes over.
        let normal = self
            .normals
            .map(|[n0, n1, n2]| b0 * n0 + b1 * n1 + b2 * n2)
            .filter(|n| n.norm2() > 0f32 && n.norm2().is_finite())
            .map(|n| {
                let n = n.normalized();
                select(n.dot(geometric) < 0f32, -n, n)
            })
            .filter(|n| n.dot(direction) < 0f32)
            .unwrap_or(geometric);

        let (u, v) = match self.uvs {
            Some([t0, t1, t2]) => (
                b0 * t0[0] + b1 * t1[0] + b2 * t2[0],
                b0 * t0[1] + b1 * t1[1] + b2 * t2[1],
            ),
            None => (b1, b2),
        };

        Hit {
            point,
            normal,
            geometric,
            front,
            u,
            v,
            distance,
            offset: crate::SURFACE_OFFSET_REL * self.scale,
        }
    }
}

/// Placement of the model in the scene: x' = R * S * x + position.
struct Transform {
    /// R * S, row-major, in f64 so the model coordinates are rounded once.
    linear: [[f64; 3]; 3],
    /// R * S^-1: the inverse transpose of R * S, which maps normals.
    normal: [[f64; 3]; 3],
    position: [f64; 3],
    /// An odd number of negative scaling factors.
    mirrors: bool,
}

impl Transform {
    fn new(definition: &definition::Mesh) -> Result<Self, MeshError> {
        let scaling = definition.scaling.axes();
        if !scaling.iter().all(|s| *s != 0f64 && s.is_finite()) {
            return Err(MeshError::Invalid(
                "`scaling` factors must be finite and non-zero".into(),
            ));
        }
        let rotation = definition.rotation;
        let [x, y, z] = [rotation.x, rotation.y, rotation.z].map(|a| f64::from(a).to_radians());
        let rx = [
            [1f64, 0f64, 0f64],
            [0f64, x.cos(), -x.sin()],
            [0f64, x.sin(), x.cos()],
        ];
        let ry = [
            [y.cos(), 0f64, y.sin()],
            [0f64, 1f64, 0f64],
            [-y.sin(), 0f64, y.cos()],
        ];
        let rz = [
            [z.cos(), -z.sin(), 0f64],
            [z.sin(), z.cos(), 0f64],
            [0f64, 0f64, 1f64],
        ];
        let r = mul(rz, mul(ry, rx));
        let scaled = |factor: &dyn Fn(f64) -> f64| {
            let mut m = r;
            for row in &mut m {
                for (value, s) in row.iter_mut().zip(scaling) {
                    *value *= factor(s);
                }
            }
            m
        };
        let position = definition.position;
        Ok(Self {
            linear: scaled(&|s| s),
            normal: scaled(&|s| 1f64 / s),
            position: [position.x, position.y, position.z].map(f64::from),
            mirrors: scaling.iter().filter(|s| **s < 0f64).count() % 2 == 1,
        })
    }

    fn point(&self, v: Vector) -> Vector {
        let [x, y, z] = apply(self.linear, v);
        Vector {
            x: (x + self.position[0]) as f32,
            y: (y + self.position[1]) as f32,
            z: (z + self.position[2]) as f32,
        }
    }

    fn normal(&self, n: Vector) -> Vector {
        let [x, y, z] = apply(self.normal, n);
        Vector {
            x: x as f32,
            y: y as f32,
            z: z as f32,
        }
    }
}

fn mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn apply(m: [[f64; 3]; 3], v: Vector) -> [f64; 3] {
    let v = [v.x, v.y, v.z].map(f64::from);
    m.map(|row| row[0] * v[0] + row[1] * v[1] + row[2] * v[2])
}

/// Why a mesh could not be built or loaded.
#[derive(Debug)]
pub enum MeshError {
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    /// A model file that is not valid OBJ or STL, or whose data does not
    /// make a valid mesh.
    Parse { path: PathBuf, message: String },
    /// Inconsistent mesh data (index out of range, mismatched normals, ...).
    Invalid(String),
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshError::Read { path, source } => {
                write!(f, "could not read the model {}: {source}", path.display())
            }
            MeshError::Parse { path, message } => {
                write!(f, "invalid model {}: {message}", path.display())
            }
            MeshError::Invalid(message) => write!(f, "invalid mesh: {message}"),
        }
    }
}

impl std::error::Error for MeshError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MeshError::Read { source, .. } => Some(source),
            MeshError::Parse { .. } | MeshError::Invalid(_) => None,
        }
    }
}
