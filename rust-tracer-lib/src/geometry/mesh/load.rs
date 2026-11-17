//! Model file readers: Wavefront OBJ and STL (binary and ASCII), both turned
//! into one indexed vertex list.

use std::collections::HashMap;
use std::path::Path;

use crate::Vector;

/// Indexed mesh data, as the inline mesh definition carries it.
#[derive(Debug, Default, PartialEq)]
pub(super) struct Data {
    pub vertices: Vec<Vector>,
    pub triangles: Vec<[u32; 3]>,
    /// Empty, or one per vertex.
    pub normals: Vec<Vector>,
    /// Empty, or one per vertex.
    pub uvs: Vec<[f32; 2]>,
}

/// By the extension (`.obj`, `.stl`, any case), else by the content.
pub(super) fn parse(path: &Path, bytes: &[u8]) -> Result<Data, String> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    match extension.as_deref() {
        Some("obj") => obj(bytes),
        Some("stl") => stl(bytes),
        _ if binary_stl_len(bytes).is_some() || ascii_stl(bytes) => stl(bytes),
        _ => obj(bytes),
    }
}

/// Wavefront OBJ: `v`, `vt` and `vn` records, and `f` faces (polygons are
/// fanned into triangles; `v`, `v/vt`, `v//vn` and `v/vt/vn` corners;
/// negative indices count back from the latest record). Everything else
/// (groups, materials, lines, ...) is ignored.
///
/// Texture coordinates and normals are kept when every face corner has
/// them, and dropped otherwise (a smooth mesh then computes its normals).
/// Corners are merged into one vertex when they share all the kept indices.
pub(super) fn obj(bytes: &[u8]) -> Result<Data, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("not UTF-8 text: {e}"))?;
    let mut positions: Vec<Vector> = Vec::new();
    let mut texcoords: Vec<[f32; 2]> = Vec::new();
    let mut normals: Vec<Vector> = Vec::new();
    /// Resolved 0-based (position, texcoord, normal) of a face corner.
    type Corner = (u32, Option<u32>, Option<u32>);
    let mut faces: Vec<[Corner; 3]> = Vec::new();

    for (number, line) in text.lines().enumerate() {
        let error = |message: String| format!("line {}: {message}", number + 1);
        let line = line.split('#').next().unwrap_or_default();
        let mut fields = line.split_whitespace();
        let Some(keyword) = fields.next() else {
            continue;
        };
        match keyword {
            "v" | "vn" => {
                let [x, y, z] = floats(&mut fields).map_err(error)?;
                let v = Vector { x, y, z };
                if keyword == "v" {
                    positions.push(v);
                } else {
                    normals.push(v);
                }
            }
            "vt" => {
                let [u] = floats(&mut fields).map_err(error)?;
                // `v` is optional (1D textures); `w` is ignored.
                let v = match fields.next() {
                    Some(v) => float(v).map_err(error)?,
                    None => 0f32,
                };
                texcoords.push([u, v]);
            }
            "f" => {
                let corners = fields
                    .map(|corner| {
                        let mut parts = corner.split('/');
                        let mut index = |count: usize, required: bool| match parts
                            .next()
                            .filter(|p| !p.is_empty())
                        {
                            Some(p) => resolve(p, count).map(Some),
                            None if required => Err(format!("bad face corner `{corner}`")),
                            None => Ok(None),
                        };
                        Ok((
                            index(positions.len(), true)?.unwrap_or_default(),
                            index(texcoords.len(), false)?,
                            index(normals.len(), false)?,
                        ))
                    })
                    .collect::<Result<Vec<Corner>, String>>()
                    .map_err(error)?;
                if corners.len() < 3 {
                    return Err(error(format!(
                        "a face needs 3 corners, not {}",
                        corners.len()
                    )));
                }
                for k in 1..corners.len() - 1 {
                    faces.push([corners[0], corners[k], corners[k + 1]]);
                }
            }
            _ => {}
        }
    }

    let with_uvs = !faces.is_empty() && faces.iter().flatten().all(|c| c.1.is_some());
    let with_normals = !faces.is_empty() && faces.iter().flatten().all(|c| c.2.is_some());
    let mut data = Data::default();
    let mut merged: HashMap<Corner, u32> = HashMap::new();
    for face in faces {
        let triangle = face.map(|(p, t, n)| {
            let key = (p, t.filter(|_| with_uvs), n.filter(|_| with_normals));
            *merged.entry(key).or_insert_with(|| {
                data.vertices.push(positions[p as usize]);
                if let Some(t) = key.1 {
                    data.uvs.push(texcoords[t as usize]);
                }
                if let Some(n) = key.2 {
                    data.normals.push(normals[n as usize]);
                }
                (data.vertices.len() - 1) as u32
            })
        });
        data.triangles.push(triangle);
    }
    Ok(data)
}

fn float(field: &str) -> Result<f32, String> {
    field
        .parse::<f32>()
        .map_err(|_| format!("`{field}` is not a number"))
}

/// The next `N` fields as numbers.
fn floats<'a, const N: usize>(
    fields: &mut impl Iterator<Item = &'a str>,
) -> Result<[f32; N], String> {
    let mut values = [0f32; N];
    for value in &mut values {
        *value = float(fields.next().ok_or("missing coordinate")?)?;
    }
    Ok(values)
}

/// An OBJ index (1-based, or negative from the end) into `count` records.
fn resolve(field: &str, count: usize) -> Result<u32, String> {
    let index: i64 = field
        .parse()
        .map_err(|_| format!("`{field}` is not an index"))?;
    let resolved = match index {
        1.. => index - 1,
        ..0 => count as i64 + index,
        0 => return Err("index 0 (OBJ indices start at 1)".into()),
    };
    if !(0..count as i64).contains(&resolved) {
        return Err(format!("index {index} out of range ({count} records)"));
    }
    Ok(resolved as u32)
}

/// The facet count of a binary STL, if the size matches it: 80-byte
/// header, u32 count, then 50 bytes per facet. Size is the reliable test:
/// some binary files start with `solid` too.
fn binary_stl_len(bytes: &[u8]) -> Option<usize> {
    let count = u32::from_le_bytes(bytes.get(80..84)?.try_into().ok()?) as usize;
    (bytes.len() == 84 + 50 * count).then_some(count)
}

fn ascii_stl(bytes: &[u8]) -> bool {
    bytes.trim_ascii_start().starts_with(b"solid")
}

/// STL, binary or ASCII. Facets carry their own corner copies: identical
/// positions are merged (so smooth shading can average across faces). The
/// stored facet normals are ignored; the winding gives the orientation.
pub(super) fn stl(bytes: &[u8]) -> Result<Data, String> {
    let corners: Vec<Vector> = match binary_stl_len(bytes) {
        Some(count) => (0..count)
            .flat_map(|facet| {
                // Skip the facet normal (12 bytes); the attribute count (2
                // bytes) trails the corners.
                let base = 84 + 50 * facet + 12;
                (0..3).map(move |corner| {
                    let at = |k: usize| {
                        let offset = base + 12 * corner + 4 * k;
                        f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
                    };
                    Vector {
                        x: at(0),
                        y: at(1),
                        z: at(2),
                    }
                })
            })
            .collect(),
        None if ascii_stl(bytes) => ascii_stl_corners(bytes)?,
        None => return Err("neither a binary STL (size mismatch) nor an ASCII one".into()),
    };

    let mut data = Data::default();
    let mut merged: HashMap<[u32; 3], u32> = HashMap::new();
    let mut corner_index = |v: Vector| {
        // +0.0 and -0.0 are one position.
        let key = [v.x, v.y, v.z].map(|c| (c + 0f32).to_bits());
        *merged.entry(key).or_insert_with(|| {
            data.vertices.push(v);
            (data.vertices.len() - 1) as u32
        })
    };
    let triangles: Vec<[u32; 3]> = corners
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&c| c.map(&mut corner_index))
        .collect();
    data.triangles = triangles;
    Ok(data)
}

fn ascii_stl_corners(bytes: &[u8]) -> Result<Vec<Vector>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("not UTF-8 text: {e}"))?;
    let mut corners = Vec::new();
    let mut in_loop = 0usize;
    for (number, line) in text.lines().enumerate() {
        let error = |message: String| format!("line {}: {message}", number + 1);
        let mut fields = line.split_whitespace();
        match fields.next() {
            Some("outer") => in_loop = 0,
            Some("vertex") => {
                let [x, y, z] = floats(&mut fields).map_err(error)?;
                corners.push(Vector { x, y, z });
                in_loop += 1;
            }
            Some("endloop") if in_loop != 3 => {
                return Err(error(format!("a facet needs 3 vertices, not {in_loop}")));
            }
            _ => {}
        }
    }
    if corners.len() % 3 != 0 {
        return Err("truncated facet".into());
    }
    Ok(corners)
}
