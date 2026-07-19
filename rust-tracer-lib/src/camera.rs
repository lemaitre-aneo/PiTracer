use serde::{Deserialize, Serialize};

use crate::{UnitVector, Vector};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CameraDefinition {
    pub origin: Vector,
    pub direction: Vector,
    pub field_of_view: f32,
    pub width: u32,
    pub height: u32,
    pub world_height: f32,
}

impl Default for CameraDefinition {
    fn default() -> Self {
        Self {
            origin: Default::default(),
            direction: Default::default(),
            field_of_view: 90f32,
            width: 320,
            height: 240,
            world_height: 0.1,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Deserialize)]
#[serde(from = "CameraDefinition")]
pub struct Camera {
    pub origin: Vector,
    reference: Vector,
    u: Vector,
    v: Vector,
    pub width: u32,
    pub height: u32,
}

impl From<CameraDefinition> for Camera {
    fn from(definition: CameraDefinition) -> Self {
        let pixel_size = definition.world_height / definition.height as f32;
        let reference_distance =
            definition.world_height * 0.5f32 / (definition.field_of_view.to_radians() / 2f32).tan();
        let reference = definition.origin - reference_distance * definition.direction.normalized();
        const VUP: Vector = Vector {
            x: 0f32,
            y: 1f32,
            z: 0f32,
        };

        // Right-handed basis: u = d × VUP points viewer-right (world +X for a
        // −Z camera), v = u × d points up. Pixel y is bottom-up; presenters
        // flip rows.
        let u = pixel_size * definition.direction.cross(VUP).normalized();
        let v = pixel_size * u.cross(definition.direction).normalized();

        let origin = definition.origin
            - u * (definition.width / 2) as f32
            - v * (definition.height / 2) as f32;

        Self {
            origin,
            reference,
            u,
            v,
            width: definition.width,
            height: definition.height,
        }
    }
}

impl Camera {
    pub fn cast(&self, x: f32, y: f32) -> (Vector, UnitVector) {
        let origin = self.origin + x * self.u + y * self.v;
        let direction = (origin - self.reference).normalized();

        (origin, direction)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_utils::{assert_close, assert_vec_close, rng, seeds};

    // Pinhole spec: `field_of_view` is the full vertical angle subtended by
    // the image; square pixels, so the horizontal angle follows from the
    // aspect ratio. Even width/height only (center pixel must be exact).
    fn camera(field_of_view: f32) -> Camera {
        Camera::from(CameraDefinition {
            origin: Vector::from((1f32, 2f32, 3f32)),
            direction: Vector::from((0.2f32, -0.4f32, 1f32)),
            field_of_view,
            width: 320,
            height: 240,
            world_height: 0.1,
        })
    }

    #[test]
    fn center_ray_is_camera_direction() {
        let camera = camera(90f32);
        let (origin, direction) = camera.cast(160f32, 120f32);

        assert_vec_close(origin, Vector::from((1f32, 2f32, 3f32)), 1e-6);
        // The center-pixel origin is recovered through ±x·u ±y·v cancellation
        // whose f32 residue is amplified by 1/reference_distance (~20× here),
        // so this comparison is looser than plain f32 tolerance.
        assert_vec_close(
            Vector::from(direction),
            Vector::from(Vector::from((0.2f32, -0.4f32, 1f32)).normalized()),
            1e-5,
        );
    }

    /// The rays through the two vertical image edges (at the horizontal
    /// center) must subtend exactly `field_of_view`; horizontally, square
    /// pixels give 2·atan(tan(fov/2)·width/height).
    #[test]
    fn vertical_fov_exact() {
        for fov in [60f32, 90f32, 120f32] {
            let camera = camera(fov);

            let top = camera.cast(160f32, 0f32).1;
            let bottom = camera.cast(160f32, 240f32).1;
            assert_close(top.dot(bottom).acos().to_degrees(), fov, 1e-5);

            let left = camera.cast(0f32, 120f32).1;
            let right = camera.cast(320f32, 120f32).1;
            let expected = 2f32
                * ((fov / 2f32).to_radians().tan() * 320f32 / 240f32)
                    .atan()
                    .to_degrees();
            assert_close(left.dot(right).acos().to_degrees(), expected, 1e-5);
        }
    }

    /// Chirality: for a −Z camera with VUP = +Y, increasing pixel x must tilt
    /// rays toward world +X (viewer-right) and increasing pixel y toward
    /// world +Y (pixel rows are bottom-up; presenters flip). A left-handed
    /// basis passes every angle/normalization test but mirrors the image.
    #[test]
    fn basis_is_right_handed() {
        let camera = Camera::from(CameraDefinition {
            direction: Vector::from((0f32, 0f32, -1f32)),
            ..Default::default()
        });
        let center_x = camera.width as f32 / 2f32;
        let center_y = camera.height as f32 / 2f32;

        let right = Vector::from(camera.cast(camera.width as f32, center_y).1);
        let left = Vector::from(camera.cast(0f32, center_y).1);
        assert!(
            right.x > 0f32 && left.x < 0f32,
            "screen-right must be world +X, got right.x = {}, left.x = {}",
            right.x,
            left.x
        );

        let top = Vector::from(camera.cast(center_x, camera.height as f32).1);
        let bottom = Vector::from(camera.cast(center_x, 0f32).1);
        assert!(
            top.y > 0f32 && bottom.y < 0f32,
            "high pixel y must be world +Y, got top.y = {}, bottom.y = {}",
            top.y,
            bottom.y
        );
    }

    #[test]
    fn rays_are_normalized_and_continuous() {
        use rand::Rng as _;

        let camera = camera(90f32);
        let mut rng = rng(seeds()[0]);
        for _ in 0..100 {
            let x = rng.random::<f32>() * 320f32;
            let y = rng.random::<f32>() * 240f32;
            let (_, direction) = camera.cast(x, y);
            assert_close(Vector::from(direction).norm2(), 1f32, 1e-6);
        }

        // a vertical sweep of directions is a planar fan of < 180°: the
        // component along the fan axis must be strictly monotone in y
        let first = camera.cast(160f32, 0f32).1;
        let last = camera.cast(160f32, 240f32).1;
        let axis = (last - first).normalized();
        let mut previous = f32::NEG_INFINITY;
        for i in 0..=48 {
            let direction = camera.cast(160f32, 240f32 * i as f32 / 48f32).1;
            let along = direction.dot(axis);
            assert!(along > previous, "direction sweep not monotone at step {i}");
            previous = along;
        }
    }
}
