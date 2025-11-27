use serde::{Deserialize, Serialize};

use crate::{GeometryPrimitive, Hit, UnitVector, Vector, select};

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sphere {
    pub radius: f32,
    pub position: Vector,
}

impl<R: crate::Rng + ?Sized> GeometryPrimitive<R> for Sphere {
    fn hit(&self, origin: Vector, direction: UnitVector, _rng: &mut R) -> Option<f32> {
        let oc = self.position - origin;
        let a = direction.norm2();
        let h = direction.dot(oc);
        let c = oc.norm2() - self.radius * self.radius;

        let delta = h * h - a * c;
        let sqrtd = delta.sqrt();

        if delta < 0f32 {
            None
        } else if h > sqrtd {
            Some((h - sqrtd) / a)
        } else {
            Some((h + sqrtd) / a)
        }
    }

    fn to_hit(&self, origin: Vector, direction: UnitVector, distance: f32) -> Hit {
        let radius2 = self.radius * self.radius;
        let mut point = origin + distance * direction;
        let outward_normal = (point - self.position).normalized();
        let front = (origin - self.position).norm2() >= radius2;

        // Due to numeric imprecisions, the intersection point might be within the sphere
        // In that case, force the intersection to be on the sphere
        let distance_to_center = (point - self.position).norm2();
        let diff = distance_to_center - radius2;
        if (front && diff < 0f32) || (!front && diff > 0f32) {
            point = self.position + outward_normal * self.radius;
        }
        let normal = select(front, outward_normal, -outward_normal);

        Hit {
            point,
            normal,
            front,
            u: point.x,
            v: point.y,
            distance,
        }
    }
}
