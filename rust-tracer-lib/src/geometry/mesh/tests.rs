use rand::RngExt;
use rand::rngs::SmallRng;

use super::bvh::Ray;
use super::*;
use crate::test_utils::{
    Stats, assert_close, assert_sampler_matches_pdf, assert_stat_close, assert_vec_close, at_angle,
    rng, seeds, sphere_integral,
};
use crate::{Quad, select};

fn vector((x, y, z): (f32, f32, f32)) -> Vector {
    Vector { x, y, z }
}

fn definition(vertices: Vec<Vector>, triangles: Vec<[u32; 3]>) -> definition::Mesh {
    definition::Mesh {
        vertices,
        triangles,
        ..Default::default()
    }
}

/// One triangle, counter-clockwise as listed.
fn triangle(p0: (f32, f32, f32), p1: (f32, f32, f32), p2: (f32, f32, f32)) -> Mesh {
    Mesh::new(vec![vector(p0), vector(p1), vector(p2)], vec![[0, 1, 2]]).unwrap()
}

/// The parallelogram `position + [0, 1]u + [0, 1]v` as two triangles, wound
/// like the quad (u x v side in front), with uvs equal to the quad's
/// (alpha, beta).
fn square(position: (f32, f32, f32), u: (f32, f32, f32), v: (f32, f32, f32)) -> Mesh {
    let (q, u, v) = (vector(position), vector(u), vector(v));
    definition::Mesh {
        vertices: vec![q, q + u, q + u + v, q + v],
        triangles: vec![[0, 1, 2], [0, 2, 3]],
        uvs: vec![[0f32, 0f32], [1f32, 0f32], [1f32, 1f32], [0f32, 1f32]],
        ..Default::default()
    }
    .try_into()
    .unwrap()
}

/// Outward-wound triangles of a closed polyhedron given by its faces
/// (cyclic corner lists, fanned): each triangle is flipped as needed so its
/// normal points away from `center`, which suits star-shaped solids.
fn closed(vertices: &[Vector], faces: &[&[u32]], center: Vector) -> Vec<[u32; 3]> {
    let mut triangles = Vec::new();
    for face in faces {
        for k in 1..face.len() - 1 {
            let [a, b, c] = [face[0], face[k], face[k + 1]];
            let [pa, pb, pc] = [a, b, c].map(|i| vertices[i as usize]);
            let n = (pb - pa).cross(pc - pa);
            let outward = n.dot((pa + pb + pc) / 3f32 - center) > 0f32;
            triangles.push(select(outward, [a, b, c], [a, c, b]));
        }
    }
    triangles
}

/// Axis-aligned box [min, min + size]^3 as 12 outward triangles.
fn cube_definition(min: f32, size: f32) -> definition::Mesh {
    let vertices: Vec<Vector> = (0..8)
        .map(|i| Vector {
            x: min + size * (i & 1) as f32,
            y: min + size * ((i >> 1) & 1) as f32,
            z: min + size * ((i >> 2) & 1) as f32,
        })
        .collect();
    let faces: [&[u32]; 6] = [
        &[0, 2, 6, 4],
        &[1, 3, 7, 5],
        &[0, 1, 5, 4],
        &[2, 3, 7, 6],
        &[0, 1, 3, 2],
        &[4, 5, 7, 6],
    ];
    let center = Vector::splat(min + 0.5f32 * size);
    let triangles = closed(&vertices, &faces, center);
    definition(vertices, triangles)
}

fn cube(min: f32, size: f32) -> Mesh {
    cube_definition(min, size).try_into().unwrap()
}

/// Latitude-longitude sphere of radius `radius` around `center`, with the
/// exact sphere normals as vertex normals when `smooth`.
fn globe(center: Vector, radius: f32, slices: u32, stacks: u32, smooth: bool) -> Mesh {
    let mut vertices = vec![center + radius * Vector::from(UnitVector::Y)];
    let mut normals = vec![Vector::from(UnitVector::Y)];
    for stack in 1..stacks {
        let theta = std::f32::consts::PI * stack as f32 / stacks as f32;
        for slice in 0..slices {
            let phi = 2f32 * std::f32::consts::PI * slice as f32 / slices as f32;
            let n = Vector {
                x: theta.sin() * phi.cos(),
                y: theta.cos(),
                z: theta.sin() * phi.sin(),
            };
            vertices.push(center + radius * n);
            normals.push(n);
        }
    }
    vertices.push(center - radius * Vector::from(UnitVector::Y));
    normals.push(-Vector::from(UnitVector::Y));
    let bottom = vertices.len() as u32 - 1;
    let ring = |stack: u32, slice: u32| 1 + (stack - 1) * slices + slice % slices;

    let mut faces: Vec<Vec<u32>> = Vec::new();
    for slice in 0..slices {
        faces.push(vec![0, ring(1, slice), ring(1, slice + 1)]);
        faces.push(vec![
            bottom,
            ring(stacks - 1, slice),
            ring(stacks - 1, slice + 1),
        ]);
        for stack in 1..stacks - 1 {
            faces.push(vec![
                ring(stack, slice),
                ring(stack + 1, slice),
                ring(stack + 1, slice + 1),
                ring(stack, slice + 1),
            ]);
        }
    }
    let faces: Vec<&[u32]> = faces.iter().map(Vec::as_slice).collect();
    let triangles = closed(&vertices, &faces, center);
    definition::Mesh {
        vertices,
        triangles,
        normals: if smooth { normals } else { Vec::new() },
        smooth,
        ..Default::default()
    }
    .try_into()
    .unwrap()
}

/// A random soup of `count` triangles in [-10, 10]^3, of sizes spread over
/// two decades.
fn soup(count: usize, rng: &mut SmallRng) -> Mesh {
    let point = |rng: &mut SmallRng, spread: f32| {
        spread
            * Vector {
                x: rng.random::<f32>() - 0.5f32,
                y: rng.random::<f32>() - 0.5f32,
                z: rng.random::<f32>() - 0.5f32,
            }
    };
    let mut vertices = Vec::new();
    for _ in 0..count {
        let center = point(rng, 20f32);
        let size = 10f32.powf(rng.random::<f32>() * 2f32 - 1.5f32);
        for _ in 0..3 {
            vertices.push(center + point(rng, size));
        }
    }
    let triangles = (0..count as u32)
        .map(|i| [3 * i, 3 * i + 1, 3 * i + 2])
        .collect();
    Mesh::new(vertices, triangles).unwrap()
}

/// The mesh's own first hit, as the scene sees it.
fn struck(mesh: &Mesh, origin: Vector, direction: UnitVector) -> Option<Hit> {
    let mut rng = rng(seeds()[0]);
    let (t, caster) = Geometry::hit(mesh, origin, direction, &mut rng)?;
    Some(caster.as_geometry().to_hit(origin, direction, t))
}

fn distance(mesh: &Mesh, origin: Vector, direction: UnitVector) -> Option<f32> {
    mesh.closest(origin, direction).map(|(t, _)| t)
}

/// Brute force over every triangle: the nearest crossing, and the sum of
/// t^2 / |cos| over all of them.
fn brute_force(mesh: &Mesh, origin: Vector, direction: UnitVector) -> (Option<f32>, f32) {
    let ray = Ray::new(origin, direction);
    let mut nearest: Option<f32> = None;
    let mut sum = 0f32;
    for triangle in &mesh.built.triangles {
        if let Some(t) = triangle.intersect(&ray) {
            nearest = Some(nearest.map_or(t, |n| n.min(t)));
            sum += t * t / direction.dot(triangle.normal).abs();
        }
    }
    (nearest, sum)
}

fn f64_distance(a: Vector, b: Vector) -> f64 {
    [a.x - b.x, a.y - b.y, a.z - b.z]
        .map(f64::from)
        .iter()
        .map(|d| d * d)
        .sum::<f64>()
        .sqrt()
}

/// Ray-triangle ground truth: aiming at a known point of the face makes the
/// expected distance |aim - origin| (f64); points outside the face, faces
/// behind the ray, and rays parallel to the plane miss.
#[test]
fn hit_distances_analytic() {
    // in the plane z = 5
    let mesh = triangle(
        (-1f32, -1f32, 5f32),
        (2f32, -1f32, 5f32),
        (-1f32, 3f32, 5f32),
    );
    let origin = Vector::default();

    for aim in [
        (0f32, 0f32, 5f32),
        (-1f32, -1f32, 5f32), // corners are inclusive
        (2f32, -1f32, 5f32),
        (0.5f32, 1f32, 5f32), // on the hypotenuse
    ] {
        let aim = vector(aim);
        let t = distance(&mesh, origin, (aim - origin).normalized()).expect("must hit");
        assert_close(t, f64_distance(aim, origin), 1e-6);
    }

    // past the hypotenuse, past each leg
    for aim in [
        (0.6f32, 1.1f32, 5f32),
        (-1.01f32, 0f32, 5f32),
        (0f32, -1.01f32, 5f32),
    ] {
        let aim = vector(aim);
        assert_eq!(distance(&mesh, origin, (aim - origin).normalized()), None);
    }

    // behind the ray, parallel to the plane, in the plane
    assert_eq!(distance(&mesh, origin, -UnitVector::Z), None);
    assert_eq!(
        distance(&mesh, vector((0f32, 0f32, 9f32)), UnitVector::Z),
        None
    );
    assert_eq!(distance(&mesh, origin, UnitVector::X), None);
    assert_eq!(
        distance(&mesh, vector((-3f32, 0f32, 5f32)), UnitVector::X),
        None
    );

    // the empty mesh is inert
    assert_eq!(distance(&Mesh::default(), origin, UnitVector::Z), None);
}

/// A two-triangle square is the quad: same hits, distances, normals, sides
/// and (through its uvs) surface coordinates, from both sides.
#[test]
fn square_matches_quad() {
    let (position, u, v) = (
        (-4f32, -5f32, 300f32),
        (108f32, 0f32, 0f32),
        (0f32, 91.6f32, 37f32),
    );
    let mesh = square(position, u, v);
    let quad = Quad::new(vector(position), vector(u), vector(v));
    let mut rng = rng(seeds()[0]);
    let mut hits = 0;
    for _ in 0..2000 {
        let origin =
            400f32 * Vector::from(UnitVector::random(&mut rng)) + vector((50f32, 40f32, 300f32));
        let aim = vector(position)
            + (1.4f32 * rng.random::<f32>() - 0.2f32) * vector(u)
            + (1.4f32 * rng.random::<f32>() - 0.2f32) * vector(v);
        let direction = (aim - origin).normalized();
        let expected = Geometry::<dyn rand::Rng>::hit(&quad, origin, direction, &mut rng);
        let actual = struck(&mesh, origin, direction);
        let (Some((t, caster)), Some(actual)) = (expected, actual) else {
            // the two may only disagree within rounding of the boundary
            let alpha = quad_coordinates(&quad, aim);
            assert!(
                expected.is_none() == actual.is_none()
                    || alpha
                        .iter()
                        .any(|a| a.abs() < 1e-4 || (a - 1f32).abs() < 1e-4),
                "quad {expected:?} vs mesh {actual:?} at {alpha:?}"
            );
            continue;
        };
        hits += 1;
        let expected = caster.as_geometry().to_hit(origin, direction, t);
        // both are plane distances: a few ulps of the origin's distance to
        // the plane's corner, over the cosine
        let cos = direction.dot(expected.normal).abs();
        let reach = (origin - vector(position)).norm() + expected.distance;
        assert!(
            (actual.distance - expected.distance).abs() < 4e-7 * reach / cos,
            "t: mesh {} vs quad {}",
            actual.distance,
            expected.distance
        );
        assert_vec_close(actual.point, expected.point, 4e-7 * reach / cos);
        assert_vec_close(actual.normal.into(), expected.normal.into(), 1e-6);
        assert_eq!(actual.geometric, actual.normal);
        assert_eq!(actual.front, expected.front);
        assert_close(actual.u, expected.u, 1e-5);
        assert_close(actual.v, expected.v, 1e-5);
    }
    assert!(hits > 500, "only {hits} hits");
}

fn quad_coordinates(quad: &Quad, point: Vector) -> [f32; 2] {
    let hit = GeometryPrimitive::<dyn rand::Rng>::to_hit(
        quad,
        point + Vector::from(UnitVector::Z),
        -UnitVector::Z,
        1f32,
    );
    [hit.u, hit.v]
}

/// The hit conventions materials rely on: both normals oppose the ray,
/// `front` = came from the counter-clockwise side; (u, v) = barycentric
/// (b1, b2) without uvs, interpolated uvs with them.
#[test]
fn to_hit_orientation_and_coordinates() {
    // counter-clockwise seen from +z: front faces +z
    let mesh = triangle((0f32, 0f32, 0f32), (4f32, 0f32, 0f32), (0f32, 2f32, 0f32));
    let aim = vector((1f32, 0.5f32, 0f32)); // b1 = 1/4, b2 = 1/4

    let from_above = struck(
        &mesh,
        aim + 3f32 * Vector::from(UnitVector::Z),
        -UnitVector::Z,
    )
    .unwrap();
    assert!(from_above.front);
    assert_vec_close(from_above.normal.into(), UnitVector::Z.into(), 1e-6);
    assert_eq!(from_above.geometric, from_above.normal);
    assert_vec_close(from_above.point, aim, 1e-6);
    assert_close(from_above.distance, 3f32, 1e-6);
    assert_close(from_above.u, 0.25f32, 1e-6);
    assert_close(from_above.v, 0.25f32, 1e-6);

    let from_below = struck(
        &mesh,
        aim - 3f32 * Vector::from(UnitVector::Z),
        UnitVector::Z,
    )
    .unwrap();
    assert!(!from_below.front);
    assert_vec_close(from_below.normal.into(), (-UnitVector::Z).into(), 1e-6);

    // interpolated uvs: corners at (0.2, 0.1), (0.6, 0.1), (0.2, 0.9)
    let mesh: Mesh = definition::Mesh {
        uvs: vec![[0.2f32, 0.1f32], [0.6f32, 0.1f32], [0.2f32, 0.9f32]],
        ..definition(
            vec![
                vector((0f32, 0f32, 0f32)),
                vector((4f32, 0f32, 0f32)),
                vector((0f32, 2f32, 0f32)),
            ],
            vec![[0, 1, 2]],
        )
    }
    .try_into()
    .unwrap();
    let hit = struck(
        &mesh,
        aim + 3f32 * Vector::from(UnitVector::Z),
        -UnitVector::Z,
    )
    .unwrap();
    assert_close(
        hit.u,
        0.5f32 * 0.2f32 + 0.25f32 * 0.6f32 + 0.25f32 * 0.2f32,
        1e-6,
    );
    assert_close(
        hit.v,
        0.5f32 * 0.1f32 + 0.25f32 * 0.1f32 + 0.25f32 * 0.9f32,
        1e-6,
    );
}

/// Watertightness: rays aimed exactly at the shared edges and vertices of a
/// closed mesh, from inside and from outside, never slip between the faces.
/// (Moller-Trumbore leaks a fraction of such rays; a glass mesh would leak
/// light through them.)
#[test]
fn closed_mesh_has_no_cracks() {
    let mut rng = rng(seeds()[1]);
    for mesh in [
        globe(vector((50f32, 30f32, 80f32)), 16.5f32, 37, 23, false),
        cube(-7.3f32, 13.1f32),
    ] {
        let [mut lo, mut hi] = [
            Vector::splat(f32::INFINITY),
            Vector::splat(f32::NEG_INFINITY),
        ];
        let mut edges = Vec::new();
        for triangle in &mesh.built.triangles {
            for (k, p) in triangle.vertices.iter().enumerate() {
                lo = Vector {
                    x: lo.x.min(p.x),
                    y: lo.y.min(p.y),
                    z: lo.z.min(p.z),
                };
                hi = Vector {
                    x: hi.x.max(p.x),
                    y: hi.y.max(p.y),
                    z: hi.z.max(p.z),
                };
                edges.push((*p, triangle.vertices[(k + 1) % 3]));
            }
        }
        let center = 0.5f32 * (lo + hi);
        let mut rays = 0;
        for (a, b) in edges {
            for s in [0f32, 0.5f32, rng.random::<f32>(), rng.random::<f32>()] {
                let aim = a + s * (b - a);
                // from inside: every direction must find the shell
                for origin in [center, center + 0.3f32 * (aim - center)] {
                    let direction = (aim - origin).normalized();
                    assert!(
                        distance(&mesh, origin, direction).is_some(),
                        "crack at {aim:?} (edge {a:?} - {b:?}, s = {s}) from inside"
                    );
                    rays += 1;
                }
                // from outside, radially (the faces around the aim face
                // the origin: no silhouette, where rounding may honestly
                // pass the ray outside)
                let origin = center + 200f32 * Vector::from((aim - center).normalized());
                let direction = (aim - origin).normalized();
                assert!(
                    distance(&mesh, origin, direction).is_some(),
                    "crack at {aim:?} from outside"
                );
                rays += 1;
            }
        }
        assert_eq!(rays, 36 * mesh.len());
    }
}

/// The BVH is an acceleration only: on a random soup (wide size range,
/// overlaps) its nearest hit and its sum over every crossing equal the brute
/// force over all triangles, for rays from inside, outside, and along the
/// axes (zero direction components).
#[test]
fn bvh_matches_brute_force() {
    let mut rng = rng(seeds()[0]);
    let mesh = soup(3000, &mut rng);
    assert!(
        mesh.built.bvh.depth() > 5,
        "the soup must actually be split"
    );
    let mut hits = 0;
    for k in 0..5000 {
        let origin = match k % 3 {
            0 => 5f32 * Vector::from(UnitVector::random(&mut rng)),
            _ => 40f32 * Vector::from(UnitVector::random(&mut rng)),
        };
        let direction = match k % 7 {
            0 => UnitVector::X,
            1 => -UnitVector::Z,
            _ => (5f32 * Vector::from(UnitVector::random(&mut rng)) - origin).normalized(),
        };
        let (nearest, sum) = brute_force(&mesh, origin, direction);
        assert_eq!(distance(&mesh, origin, direction), nearest, "ray {k}");
        let pdf = mesh.target_pdf(origin, direction);
        assert_close(pdf, sum / mesh.area(), 1e-5 * f64::from(pdf));
        hits += usize::from(nearest.is_some());
    }
    assert!(hits > 1000, "only {hits} hits");
}

/// Degenerate inputs for the builder: thousands of identical triangles (all
/// centroids equal: no SAH split exists) and a long thin row (deep SAH
/// splits) still build within the traversal stack and are found.
#[test]
fn bvh_degenerate_layouts() {
    let tri = [
        vector((0f32, 0f32, 1f32)),
        vector((1f32, 0f32, 1f32)),
        vector((0f32, 1f32, 1f32)),
    ];
    let count = 5000u32;
    let vertices = (0..count).flat_map(|_| tri).collect();
    let triangles = (0..count).map(|i| [3 * i, 3 * i + 1, 3 * i + 2]).collect();
    let stack = Mesh::new(vertices, triangles).unwrap();
    assert!(stack.built.bvh.depth() <= 65);
    let t = distance(&stack, vector((0.2f32, 0.2f32, 0f32)), UnitVector::Z);
    assert_close(t.expect("must hit"), 1f32, 1e-6);

    // geometric row: x = 1.3^k -> wildly unbalanced SAH splits
    let mut vertices = Vec::new();
    for k in 0..60 {
        let x = 1.3f32.powi(k);
        vertices.extend([
            vector((x, 0f32, 0f32)),
            vector((x, 1f32, 0f32)),
            vector((x, 0f32, 1f32)),
        ]);
    }
    let triangles = (0..60u32).map(|i| [3 * i, 3 * i + 1, 3 * i + 2]).collect();
    let row = Mesh::new(vertices, triangles).unwrap();
    assert!(row.built.bvh.depth() <= 65);
    for k in 0..60 {
        // between the previous face (x / 1.3) and this one
        let x = 1.3f32.powi(k);
        let origin = vector((0.9f32 * x, 0.2f32, 0.2f32));
        assert_close(
            distance(&row, origin, UnitVector::X).expect("must hit"),
            f64::from(x) - f64::from(origin.x),
            1e-5,
        );
    }
}

/// A ray spawned via Hit::spawn cannot re-intersect the face it leaves:
/// outward rays from a convex mesh escape it entirely (grazing included),
/// and inward ones cross to the far side instead of re-hitting at t ~ 0.
#[test]
fn spawn_clears_own_surface() {
    let mut rng = rng(seeds()[1]);
    for (mesh, chord) in [
        (cube(-4f32, 108f32), 1f32),
        (cube(1f32, 0.05f32), 1e-3f32),
        (
            globe(vector((40f32, 16.5f32, 47f32)), 16.5f32, 24, 12, true),
            1e-2f32,
        ),
    ] {
        for _ in 0..20 {
            let origin = 500f32 * Vector::from(UnitVector::random(&mut rng));
            let target = mesh.built.triangles[rng.random_range(0..mesh.len())].vertices;
            let aim = (target[0] + target[1] + target[2]) / 3f32;
            let Some(hit) = struck(&mesh, origin, (aim - origin).normalized()) else {
                continue;
            };
            let mut directions = vec![hit.geometric];
            for angle in [30f64, 60f64, 85f64, 89.9f64, 89.99f64] {
                directions.push(at_angle(hit.geometric, angle.to_radians()));
            }
            for _ in 0..50 {
                directions.push(UnitVector::random(&mut rng));
            }
            for scattered in directions {
                let again = distance(&mesh, hit.spawn(scattered), scattered);
                if scattered.dot(hit.geometric) > 0f32 {
                    assert_eq!(again, None, "outward re-hit");
                } else {
                    assert!(
                        again.is_some_and(|t| t > chord),
                        "inward ray re-hit its own face: {again:?}"
                    );
                }
            }
        }
    }
}

/// The offset reported by to_hit bounds the hit point's true (f64) distance
/// from the face's plane, even from far origins (the point is rebuilt from
/// barycentrics), while staying far below the face's extent.
#[test]
fn to_hit_offset_bounds_placement_noise() {
    let mut rng = rng(seeds()[2]);
    let mesh = globe(vector((73f32, 46.5f32, 88f32)), 3f32, 16, 8, false);
    for _ in 0..500 {
        let triangle = &mesh.built.triangles[rng.random_range(0..mesh.len())];
        let [p0, p1, p2] = triangle.vertices;
        let (b1, b2) = (rng.random::<f32>(), rng.random::<f32>());
        let (b1, b2) = select(b1 + b2 > 1f32, (1f32 - b1, 1f32 - b2), (b1, b2));
        let aim = p0 + b1 * (p1 - p0) + b2 * (p2 - p0);
        let origin =
            aim + rng.random_range(5f32..5000f32) * Vector::from(UnitVector::random(&mut rng));
        let direction = (aim - origin).normalized();
        let Some((t, caster)) = Geometry::<dyn rand::Rng>::hit(&mesh, origin, direction, &mut rng)
        else {
            continue;
        };
        let hit = caster.as_geometry().to_hit(origin, direction, t);
        let extent = (p1 - p0).norm().min((p2 - p0).norm());
        // relative to the coordinates, so a small face far from the origin
        // gets a larger share of its size; still far below it
        assert!(hit.offset > 0f32 && hit.offset < 1e-3f32 * extent);

        // f64 plane of the struck face (the caster may be a neighbor)
        let face = mesh
            .built
            .triangles
            .iter()
            .find(|f| {
                std::ptr::eq(
                    *f as *const Triangle as *const u8,
                    caster as *const dyn HitCaster as *const u8,
                )
            })
            .expect("the caster is one of the mesh's faces");
        let [q0, q1, q2] = face.vertices.map(|p| [p.x, p.y, p.z].map(f64::from));
        let e1: [f64; 3] = std::array::from_fn(|k| q1[k] - q0[k]);
        let e2: [f64; 3] = std::array::from_fn(|k| q2[k] - q0[k]);
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let norm = n.iter().map(|c| c * c).sum::<f64>().sqrt();
        let point = [hit.point.x, hit.point.y, hit.point.z].map(f64::from);
        let error = (0..3).map(|k| (point[k] - q0[k]) * n[k]).sum::<f64>() / norm;
        assert!(
            error.abs() < f64::from(hit.offset),
            "placement noise {error:e} exceeds offset {:e} (origin {} away)",
            hit.offset,
            t
        );
    }
}

/// Smooth shading: the interpolated normal tracks the true sphere normal far
/// better than the faces do, always opposes the ray, and leaves the
/// geometric normal (spawning) on the face.
#[test]
fn smooth_normals_interpolate() {
    let center = vector((1f32, 2f32, 3f32));
    let flat = globe(center, 2f32, 12, 6, false);
    let smooth = globe(center, 2f32, 12, 6, true);
    let mut rng = rng(seeds()[0]);
    let (mut flat_error, mut smooth_error) = (Stats::new(), Stats::new());
    for _ in 0..2000 {
        let origin = center + 10f32 * Vector::from(UnitVector::random(&mut rng));
        let aim = center + 1.5f32 * Vector::from(UnitVector::random(&mut rng));
        let direction = (aim - origin).normalized();
        let (Some(f), Some(s)) = (
            struck(&flat, origin, direction),
            struck(&smooth, origin, direction),
        ) else {
            continue;
        };
        let truth = (s.point - center).normalized();
        assert!(s.normal.dot(direction) < 0f32);
        assert!(s.geometric.dot(direction) < 0f32);
        assert!(s.front);
        assert_eq!(s.geometric, f.normal);
        flat_error.push((Vector::from(f.normal) - truth).norm());
        smooth_error.push((Vector::from(s.normal) - truth).norm());
    }
    eprintln!(
        "normal error: flat {} vs smooth {}",
        flat_error.mean(),
        smooth_error.mean()
    );
    assert!(smooth_error.mean() < 0.2 * flat_error.mean());
}

/// The shading normal is turned to the side the ray comes from (vertex
/// normals may disagree with the winding), and gives way to the face's own
/// normal where it would face away from the ray (near silhouettes).
#[test]
fn smooth_normal_orientation_and_fallback() {
    let vertices = vec![
        vector((0f32, 0f32, 0f32)),
        vector((1f32, 0f32, 0f32)),
        vector((0f32, 1f32, 0f32)),
    ];
    let aim = vector((0.25f32, 0.25f32, 0f32));

    // normals against the winding (-z): flipped to the ray's side
    let tilted = vector((0.3f32, 0f32, -1f32));
    let mesh = Mesh::smooth(vertices.clone(), vec![[0, 1, 2]], vec![tilted; 3]).unwrap();
    let hit_above = struck(&mesh, aim + Vector::from(UnitVector::Z), -UnitVector::Z).unwrap();
    assert!(hit_above.front);
    assert_vec_close(hit_above.normal.into(), (-tilted).normalized().into(), 1e-6);
    assert_vec_close(hit_above.geometric.into(), UnitVector::Z.into(), 1e-6);

    // a normal leaning away from a grazing ray: the face's normal instead
    let leaning = vector((1f32, 0f32, 0.1f32));
    let mesh = Mesh::smooth(vertices, vec![[0, 1, 2]], vec![leaning; 3]).unwrap();
    let direction = vector((0.9f32, 0f32, -0.3f32)).normalized();
    let hit = struck(&mesh, aim - 3f32 * Vector::from(direction), direction).unwrap();
    assert_eq!(hit.normal, hit.geometric);
    assert_vec_close(hit.normal.into(), UnitVector::Z.into(), 1e-6);
}

/// Without given normals, a smooth mesh averages the face normals around
/// each vertex by area: on a symmetric roof, the ridge normal is straight up.
#[test]
fn smooth_normals_computed() {
    let mesh: Mesh = definition::Mesh {
        smooth: true,
        ..definition(
            vec![
                vector((-1f32, 0f32, -1f32)),
                vector((-1f32, 0f32, 1f32)),
                vector((0f32, 1f32, 1f32)),
                vector((0f32, 1f32, -1f32)),
                vector((1f32, 0f32, -1f32)),
                vector((1f32, 0f32, 1f32)),
            ],
            vec![[0, 1, 2], [0, 2, 3], [3, 2, 5], [3, 5, 4]],
        )
    }
    .try_into()
    .unwrap();
    // just left of the ridge (x = 0): flat would be (-1, 1, 0)/sqrt(2)
    let hit = struck(&mesh, vector((-1e-4f32, 5f32, 0f32)), -UnitVector::Y).unwrap();
    assert_vec_close(hit.normal.into(), UnitVector::Y.into(), 1e-3);
    assert_vec_close(
        hit.geometric.into(),
        vector((-1f32, 1f32, 0f32)).normalized().into(),
        1e-6,
    );
}

/// Placement: scaling, then rotation about x, y, z, then position; normals
/// by the inverse transpose; mirroring keeps the outside in front.
#[test]
fn transform_places_the_model() {
    let mesh: Mesh = definition::Mesh {
        scaling: definition::Scaling::Axes(vector((2f32, 3f32, 4f32))),
        rotation: vector((0f32, 0f32, 90f32)),
        position: vector((10f32, 0f32, 0f32)),
        ..definition(
            vec![
                vector((1f32, 1f32, 1f32)),
                vector((0f32, 0f32, 0f32)),
                vector((1f32, 0f32, 0f32)),
            ],
            vec![[0, 1, 2]],
        )
    }
    .try_into()
    .unwrap();
    // (1, 1, 1) -> scaled (2, 3, 4) -> z-quarter-turn (-3, 2, 4) -> (7, 2, 4)
    let [p0, p1, p2] = mesh.built.triangles[0].vertices;
    assert_vec_close(p0, vector((7f32, 2f32, 4f32)), 1e-6);
    assert_vec_close(p1, vector((10f32, 0f32, 0f32)), 1e-6);
    assert_vec_close(p2, vector((10f32, 2f32, 0f32)), 1e-6);

    // rotation order: x first, then y: (0, 1, 0) -x90-> (0, 0, 1) -y90-> (1, 0, 0)
    let transform = Transform::new(&definition::Mesh {
        rotation: vector((90f32, 90f32, 0f32)),
        ..Default::default()
    })
    .unwrap();
    assert_vec_close(
        transform.point(vector((0f32, 1f32, 0f32))),
        vector((1f32, 0f32, 0f32)),
        1e-6,
    );

    // non-uniform scaling: vertex normals equal to the face normal stay
    // equal to it (inverse transpose), where scaling them like points would
    // tilt them
    let vertices = vec![
        vector((0f32, 0f32, 0f32)),
        vector((1f32, 0f32, 0f32)),
        vector((0f32, 1f32, 1f32)),
    ];
    let face = vector((0f32, -1f32, 1f32));
    let mesh: Mesh = definition::Mesh {
        normals: vec![face; 3],
        smooth: true,
        scaling: definition::Scaling::Axes(vector((1f32, 1f32, 3f32))),
        ..definition(vertices, vec![[0, 1, 2]])
    }
    .try_into()
    .unwrap();
    let aim = vector((0.25f32, 0.25f32, 0.75f32));
    let hit = struck(&mesh, aim + 2f32 * face, -face.normalized()).unwrap();
    assert_vec_close(hit.normal.into(), hit.geometric.into(), 1e-6);

    // mirrored cube: still front from outside, back from inside
    for scaling in [vector((-1f32, 1f32, 1f32)), vector((-1f32, -1f32, -2f32))] {
        let mesh: Mesh = definition::Mesh {
            scaling: definition::Scaling::Axes(scaling),
            ..cube_definition(-1f32, 2f32)
        }
        .try_into()
        .unwrap();
        let mut rng = rng(seeds()[0]);
        for _ in 0..50 {
            let direction = UnitVector::random(&mut rng);
            assert!(!struck(&mesh, Vector::default(), direction).unwrap().front);
            let outside = -10f32 * Vector::from(direction);
            assert!(struck(&mesh, outside, direction).unwrap().front);
        }
    }
}

/// Inconsistent definitions fail loudly instead of rendering garbage.
#[test]
fn invalid_definitions_rejected() {
    let three = || {
        vec![
            vector((0f32, 0f32, 0f32)),
            vector((1f32, 0f32, 0f32)),
            vector((0f32, 1f32, 0f32)),
        ]
    };
    assert!(Mesh::new(three(), vec![[0, 1, 3]]).is_err());
    assert!(Mesh::new(vec![vector((f32::NAN, 0f32, 0f32))], vec![]).is_err());
    assert!(Mesh::smooth(three(), vec![[0, 1, 2]], vec![Vector::default(); 2]).is_err());
    let invalid = [
        definition::Mesh {
            uvs: vec![[0f32; 2]; 4],
            ..definition(three(), vec![[0, 1, 2]])
        },
        definition::Mesh {
            model: Some("model.obj".into()),
            ..definition(three(), vec![[0, 1, 2]])
        },
        definition::Mesh {
            scaling: definition::Scaling::Axes(vector((1f32, 0f32, 1f32))),
            ..definition(three(), vec![[0, 1, 2]])
        },
        definition::Mesh {
            scaling: definition::Scaling::Uniform(f32::INFINITY),
            ..definition(three(), vec![[0, 1, 2]])
        },
    ];
    for definition in invalid {
        assert!(
            Mesh::try_from(definition.clone()).is_err(),
            "{definition:?}"
        );
    }
}

/// Meshes without area (empty, a model not loaded yet, only degenerate
/// faces) are inert: no hits, zero pdf, a finite placeholder sample.
#[test]
fn degenerate_mesh_inert() {
    let mut rng = rng(seeds()[0]);
    let unloaded: Mesh = definition::Mesh {
        model: Some("missing.obj".into()),
        ..Default::default()
    }
    .try_into()
    .unwrap();
    let collinear = triangle((0f32, 0f32, 0f32), (1f32, 1f32, 1f32), (2f32, 2f32, 2f32));
    let point = triangle((1f32, 1f32, 1f32), (1f32, 1f32, 1f32), (1f32, 1f32, 1f32));
    for mesh in [Mesh::default(), unloaded, collinear, point] {
        assert!(mesh.is_empty());
        assert_eq!(mesh.area(), 0f32);
        for _ in 0..10 {
            let origin = 5f32 * Vector::from(UnitVector::random(&mut rng));
            let direction = UnitVector::random(&mut rng);
            assert_eq!(distance(&mesh, origin, direction), None);
            assert_eq!(mesh.target_pdf(origin, direction), 0f32);
            let sample = Geometry::target_random(&mesh, origin, &mut rng);
            assert!(sample.x.is_finite() && sample.y.is_finite() && sample.z.is_finite());
        }
    }
}

/// Every sampled direction reaches the mesh with a positive pdf.
#[test]
fn target_random_hits_mesh() {
    let mesh = globe(vector((0f32, 0f32, 5f32)), 1f32, 10, 5, false);
    for seed in seeds() {
        let mut rng = rng(seed);
        for origin in [
            Vector::default(),
            vector((3f32, 1f32, 9f32)),
            vector((0.2f32, 0f32, 5f32)),
        ] {
            for _ in 0..1000 {
                let direction = Geometry::target_random(&mesh, origin, &mut rng);
                assert!(distance(&mesh, origin, direction).is_some());
                assert!(mesh.target_pdf(origin, direction) > 0f32);
            }
        }
    }
}

/// On a planar mesh the density is the quad's closed form
/// t^2 / (A * |cos|), from either side.
#[test]
fn target_pdf_matches_quad() {
    let position = (-1f32, -1f32, 1f32);
    let (u, v) = ((2f32, 0f32, 0f32), (0f32, 2f32, 0f32));
    let mesh = square(position, u, v);
    let quad = Quad::new(vector(position), vector(u), vector(v));
    let mut rng = rng(seeds()[0]);
    for origin in [Vector::default(), vector((0.3f32, -0.2f32, 2.5f32))] {
        for _ in 0..1000 {
            let direction = UnitVector::random(&mut rng);
            let expected = Geometry::<dyn rand::Rng>::target_pdf(&quad, origin, direction);
            let actual = mesh.target_pdf(origin, direction);
            assert!(
                (actual - expected).abs() <= 1e-5 * expected
                    || (actual == 0f32) != (expected == 0f32) && near_boundary(origin, direction),
                "{direction:?}: mesh {actual} vs quad {expected}"
            );
        }
    }
    // closed form off the diagonal: toward (0.5, 0.2, 1), t^2 = 1.29 and
    // cos = 1/sqrt(1.29) (straight up would run along the shared diagonal,
    // see `Mesh::target_pdf`)
    let toward = vector((0.5f32, 0.2f32, 1f32)).normalized();
    assert_close(
        mesh.target_pdf(Vector::default(), toward),
        1.29f64.powf(1.5f64) / 4f64,
        1e-6,
    );

    fn near_boundary(origin: Vector, direction: UnitVector) -> bool {
        let t = (1f32 - origin.z) / direction.z;
        let p = origin + t * direction;
        (p.x.abs() - 1f32).abs() < 1e-4 || (p.y.abs() - 1f32).abs() < 1e-4
    }
}

/// target_pdf integrates to 1 over the sphere of directions, including for
/// closed and non-convex meshes, where a direction crosses the surface
/// several times: counting only the nearest crossing would integrate to the
/// visible fraction of the area (1/6 for the box seen face-on).
#[test]
fn target_pdf_normalizes() {
    let mut rng = rng(seeds()[1]);
    let l_shape: Mesh = {
        // an L-shaped prism: non-convex, closed
        let outline = [
            (0f32, 0f32),
            (2f32, 0f32),
            (2f32, 1f32),
            (1f32, 1f32),
            (1f32, 2f32),
            (0f32, 2f32),
        ];
        let mut vertices: Vec<Vector> =
            outline.iter().map(|&(x, y)| vector((x, y, 0f32))).collect();
        vertices.extend(outline.iter().map(|&(x, y)| vector((x, y, 1f32))));
        let mut triangles = vec![[0, 1, 2], [0, 2, 3], [0, 3, 5], [3, 4, 5]];
        triangles.extend([[6, 8, 7], [6, 9, 8], [6, 11, 9], [9, 11, 10]]);
        for i in 0..6u32 {
            let j = (i + 1) % 6;
            triangles.extend([[i, j + 6, j], [i, i + 6, j + 6]]);
        }
        Mesh::new(vertices, triangles).unwrap()
    };
    for (mesh, origin) in [
        (
            square((-1f32, -1f32, 1f32), (2f32, 0f32, 0f32), (0f32, 2f32, 0f32)),
            Vector::default(),
        ),
        (cube(-1f32, 2f32), vector((0.3f32, -0.2f32, 2.5f32))),
        (cube(-1f32, 2f32), vector((0.3f32, -0.2f32, 0.1f32))), // inside
        (l_shape, vector((1.6f32, 1.7f32, 0.4f32))),            // in the notch
    ] {
        let (integral, se) = sphere_integral(1_000_000, &mut rng, |w| mesh.target_pdf(origin, w));
        assert_stat_close(integral, se, 1f32, 4f32, 0f32);
    }
}

/// target_random is distributed according to target_pdf (probe-based: no
/// mesh is phi-symmetric), on a closed box seen from outside, where every
/// direction crosses twice.
#[test]
fn target_sampler_matches_target_pdf() {
    let mesh = cube(-1f32, 2f32);
    let origin = vector((0.5f32, 0.8f32, 3f32));
    let mut rng = rng(seeds()[0]);
    assert_sampler_matches_pdf(
        100_000,
        1_000_000,
        &mut rng,
        -UnitVector::Z,
        |rng| Geometry::target_random(&mesh, origin, rng),
        |w| mesh.target_pdf(origin, w),
    );
}

/// Faces are drawn in proportion to their area (f64 table): a 1:9 pair.
#[test]
fn target_random_weights_by_area() {
    let mesh = Mesh::new(
        vec![
            vector((0f32, 0f32, 5f32)),
            vector((1f32, 0f32, 5f32)),
            vector((0f32, 1f32, 5f32)),
            vector((10f32, 0f32, 5f32)),
            vector((13f32, 0f32, 5f32)),
            vector((10f32, 3f32, 5f32)),
        ],
        vec![[0, 1, 2], [3, 4, 5]],
    )
    .unwrap();
    let mut rng = rng(seeds()[2]);
    let mut small = Stats::new();
    for _ in 0..100_000 {
        let w = Geometry::target_random(&mesh, Vector::default(), &mut rng);
        small.push(if w.x / w.z < 1f32 / 5f32 { 1f64 } else { 0f64 });
    }
    assert_stat_close(small.mean(), small.se(), 0.1f32, 4f32, 0f32);
}

/// OBJ: every corner syntax, negative indices, polygons fanned, comments
/// and unknown records ignored; corners sharing all indices are merged.
#[test]
fn obj_parses() {
    let data = load::obj(
        b"# a square, then a triangle using relative indices
mtllib none.mtl
o thing
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0   # trailing comment
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
usemtl red
s 1
f 1/1/1 2/2/1 3/3/1 4/4/1
v 0 0 2
f -2/-1/-1 -4/-4/-1 -1/-2/-1
",
    )
    .unwrap();
    // -2/-1/-1 is corner 4/4/1 again (merged); the others are new
    assert_eq!(data.vertices.len(), 6);
    assert_eq!(data.triangles, vec![[0, 1, 2], [0, 2, 3], [3, 4, 5]]);
    assert_eq!(data.vertices[4], vector((1f32, 0f32, 0f32)));
    assert_eq!(data.uvs[4], [0f32, 0f32]);
    assert_eq!(data.vertices[5], vector((0f32, 0f32, 2f32)));
    assert_eq!(data.uvs[5], [1f32, 1f32]);
    assert!(
        data.normals
            .iter()
            .all(|n| *n == vector((0f32, 0f32, 1f32)))
    );

    // corners without vt: uvs dropped, vertices merged by position + normal
    let data = load::obj(
        b"v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0.5 0.5\nvn 0 0 1\nf 1//1 2//1 3//1\nf 1/1/1 3/1/1 2/1/1\n",
    )
    .unwrap();
    assert_eq!(data.vertices.len(), 3);
    assert!(data.uvs.is_empty());
    assert_eq!(data.normals.len(), 3);
    assert_eq!(data.triangles, vec![[0, 1, 2], [0, 2, 1]]);

    // plain positions
    let data = load::obj(b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
    assert_eq!(data.triangles, vec![[0, 1, 2]]);
    assert!(data.uvs.is_empty() && data.normals.is_empty());

    for (text, message) in [
        ("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n", "line 4"),
        ("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n", "index 0"),
        ("v 0 0\n", "line 1"),
        ("v 0 0 x\n", "`x`"),
        ("v 0 0 0\nv 1 0 0\nf 1 2\n", "3 corners"),
    ] {
        let error = load::obj(text.as_bytes()).unwrap_err();
        assert!(error.contains(message), "{error:?} lacks {message:?}");
    }
}

/// STL: binary (even with a header starting with `solid`) and ASCII read to
/// the same indexed mesh, shared corners merged.
#[test]
fn stl_parses() {
    let facets = [
        [(0f32, 0f32, 0f32), (1f32, 0f32, 0f32), (0f32, 1f32, 0f32)],
        [(0f32, 0f32, 0f32), (0f32, 1f32, 0f32), (0f32, 0f32, 1f32)],
        [(0f32, 0f32, 0f32), (0f32, 0f32, 1f32), (1f32, 0f32, 0f32)],
        [(1f32, 0f32, 0f32), (0f32, 0f32, 1f32), (0f32, 1f32, 0f32)],
    ];
    let mut binary = b"solid but actually binary".to_vec();
    binary.resize(80, 0);
    binary.extend((facets.len() as u32).to_le_bytes());
    let mut ascii = String::from("solid tetra\n");
    for facet in facets {
        binary.extend([0f32; 3].iter().flat_map(|c| c.to_le_bytes()));
        ascii += "  facet normal 0 0 0\n    outer loop\n";
        for (x, y, z) in facet {
            binary.extend([x, y, z].iter().flat_map(|c| c.to_le_bytes()));
            ascii += &format!("      vertex {x:e} {y} {z}\n");
        }
        binary.extend([0u8; 2]);
        ascii += "    endloop\n  endfacet\n";
    }
    ascii += "endsolid tetra\n";

    let from_binary = load::stl(&binary).unwrap();
    let from_ascii = load::stl(ascii.as_bytes()).unwrap();
    assert_eq!(from_binary, from_ascii);
    assert_eq!(from_binary.vertices.len(), 4);
    assert_eq!(from_binary.triangles.len(), 4);
    // by content when the extension says nothing
    assert_eq!(
        load::parse(Path::new("model"), &binary).unwrap(),
        from_binary
    );
    assert_eq!(
        load::parse(Path::new("model.STL"), ascii.as_bytes()).unwrap(),
        from_binary
    );

    assert!(load::stl(&binary[..binary.len() - 1]).is_err());
    let short = "solid x\nfacet\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nendloop\nendfacet\n";
    assert!(
        load::stl(short.as_bytes())
            .unwrap_err()
            .contains("3 vertices")
    );
}

/// A `model` file is read relative to the scene folder, then carried
/// inline: the serialized mesh has the data and no file name, and builds
/// the same mesh back (what workers receive).
#[test]
fn model_loads_and_travels_inline() {
    let folder = tempfile::tempdir().unwrap();
    std::fs::write(
        folder.path().join("tri.obj"),
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 4 3\n",
    )
    .unwrap();
    let json = r#"{"model": "tri.obj", "smooth": true, "scaling": 2, "position": [0, 0, 3]}"#;
    let mut mesh: Mesh = serde_json::from_str(json).unwrap();
    assert!(mesh.is_empty());
    mesh.load(folder.path()).unwrap();
    assert_eq!(mesh.len(), 2);
    assert_close(mesh.area(), 4f32, 1e-6);
    assert!(distance(&mesh, vector((1f32, 1f32, 0f32)), UnitVector::Z).is_some());

    let wire = serde_json::to_string(&mesh).unwrap();
    assert!(
        !wire.contains("model") && wire.contains("vertices"),
        "{wire}"
    );
    let received: Mesh = serde_json::from_str(&wire).unwrap();
    assert_eq!(received, mesh);
    assert_eq!(received.len(), 2);
    // loading again is a no-op
    let mut again = received.clone();
    again.load(Path::new("/nonexistent")).unwrap();
    assert_eq!(again, received);

    let mut missing: Mesh = serde_json::from_str(r#"{"model": "nope.obj"}"#).unwrap();
    assert!(matches!(
        missing.load(folder.path()),
        Err(MeshError::Read { .. })
    ));
    std::fs::write(folder.path().join("bad.obj"), "f 1 2 3\n").unwrap();
    let mut bad: Mesh = serde_json::from_str(r#"{"model": "bad.obj"}"#).unwrap();
    let error = bad.load(folder.path()).unwrap_err().to_string();
    assert!(
        error.contains("bad.obj") && error.contains("line 1"),
        "{error}"
    );
}

/// In a scene object the mesh keys sit next to the texture ones without
/// clashing: a marble mesh keeps the marble's `scale` and its own
/// `scaling`.
#[test]
fn object_keys_do_not_clash() {
    let json = r#"{
        "type": "mesh",
        "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        "triangles": [[0, 1, 2]],
        "scaling": [2, 2, 1],
        "material": "Metal",
        "texture": "marble",
        "scale": 3
    }"#;
    let object: crate::Object<crate::GeometryEnum, crate::MaterialEnum, crate::TextureEnum> =
        serde_json::from_str(json).unwrap();
    let crate::GeometryEnum::Mesh(mesh) = &object.geometry else {
        panic!("not a mesh: {:?}", object.geometry);
    };
    assert_close(mesh.area(), 2f32, 1e-6);
    let crate::TextureEnum::Marble(marble) = &object.texture else {
        panic!("not marble: {:?}", object.texture);
    };
    assert_eq!(marble.scale, 3f32);
    assert!(matches!(object.material, crate::MaterialEnum::Metal(_)));
}
