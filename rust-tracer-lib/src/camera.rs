use serde::{Deserialize, Serialize};

use crate::vector::*;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CameraDefinition {
    pub origin: Vector,
    pub direction: Vector,
    pub field_of_view: f32,
    pub width: u32,
    pub height: u32,
    pub world_width: f32,
}

impl Default for CameraDefinition {
    fn default() -> Self {
        Self {
            origin: Default::default(),
            direction: Default::default(),
            field_of_view: 90f32,
            width: 320,
            height: 240,
            world_width: 0.1,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Camera {
    origin: Vector,
    reference: Vector,
    u: Vector,
    v: Vector,
}

impl From<CameraDefinition> for Camera {
    fn from(definition: CameraDefinition) -> Self {
        let pixel_size = definition.world_width / definition.width as f32;
        let reference_distance =
            definition.world_width * 0.5f32 / (definition.field_of_view.to_radians() / 2f32).tan();
        let reference = definition.origin - reference_distance * definition.direction.normalized();
        const VUP: Vector = Vector {
            x: 0f32,
            y: 1f32,
            z: 0f32,
        };

        let u = pixel_size * VUP.cross(definition.direction).normalized();
        let v = pixel_size * definition.direction.cross(u).normalized();

        let origin = definition.origin
            - u * (definition.width / 2) as f32
            - v * (definition.height / 2) as f32;

        Self {
            origin,
            reference,
            u,
            v,
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
