use crate::select;

use serde::{Deserialize, Serialize, de::Visitor};
use soa_derive::StructOfArray;

use crate::{color::*, vector::*};

#[derive(StructOfArray, Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[soa_derive(Debug, Default, Clone, PartialEq, Serialize)]
#[serde(default)]
pub struct Sphere {
    pub radius: f32,
    #[nested_soa]
    pub position: Vector,
    #[nested_soa]
    pub emission: Color,
    #[nested_soa]
    pub color: Color,
    pub reflexivity: Reflexivity,
    pub max_reflexivity: f32,
}

impl<'de> Deserialize<'de> for SphereVec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct FieldVisitor;

        impl<'de> Visitor<'de> for FieldVisitor {
            type Value = SphereVec;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("either [{radius: f32, position: Vector, emission: Vector, color: Color, reflexivity: Reflexivity, max_reflexivity: f32}] or {radius: [f32], position: [Vector], emission: [Vector], color: [Color], reflexivity: [Reflexivity], max_reflexivity: [f32]}")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut spheres = Vec::new();

                while let Some(vec) = seq.next_element()? {
                    spheres.push(vec);
                }

                Ok(spheres.into_iter().collect())
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut spheres = SphereVec::default();

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "radius" => spheres.radius = map.next_value()?,
                        "position" => spheres.position = map.next_value()?,
                        "emission" => spheres.emission = map.next_value()?,
                        "color" => spheres.color = map.next_value()?,
                        "reflexivity" => spheres.reflexivity = map.next_value()?,
                        "max_reflexivity" => spheres.max_reflexivity = map.next_value()?,
                        _ => Err(serde::de::Error::custom(
                            "Expected either `radius`, `position`, `emission`, `color`, `reflexivity`, `max_reflexivity`",
                        ))?,
                    }
                }

                Ok(spheres)
            }
        }

        deserializer.deserialize_any(FieldVisitor)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reflexivity {
    #[default]
    Diffuse,
    Specular,
    Refraction,
}

pub struct Radiance {
    pub emission: Color,
    pub absorption: Color,
    pub origin: Vector,
    pub direction: UnitVector,
}

const EPS: f32 = 7e-2f32;

impl SphereRef<'_> {
    pub fn intersect(self, origin: Vector, direction: UnitVector) -> f32 {
        let oc = self.position - origin;
        let a = direction.norm2();
        let h = direction.dot(oc);
        let c = oc.norm2() - self.radius * self.radius;

        let delta = h * h - a * c;
        let sqrtd = delta.sqrt();

        if delta < 0f32 {
            0f32
        } else if h > sqrtd {
            (h - sqrtd) / a
        } else {
            (h + sqrtd) / a
        }
    }

    pub fn radiance(
        self,
        origin: Vector,
        direction: UnitVector,
        distance: f32,
        rng: &mut impl rand::Rng,
    ) -> Radiance {
        let radius2 = self.radius * self.radius;
        let mut intersection_point = origin + distance * direction;
        let outward_normal = (intersection_point - self.position).normalized();
        let front_face = (origin - self.position).norm2() >= radius2;

        // Due to numeric imprecisions, the intersection point might be within the sphere
        // In that case, force the intersection to be on the sphere
        let distance_to_center = (intersection_point - self.position).norm2();
        let diff = distance_to_center - radius2;
        if (front_face && diff < 0f32) || (!front_face && diff > 0f32) {
            intersection_point = self.position + outward_normal * *self.radius;
        }
        let normal = select(front_face, outward_normal, -outward_normal);

        let emission = self.emission.to_owned();
        let absorption = self.color.to_owned();

        const AIR_REFRACTION_INDEX: f32 = 1.0f32;
        const GLASS_REFRACTION_INDEX: f32 = 1.5f32;
        const AIR_TO_GLASS: f32 = AIR_REFRACTION_INDEX / GLASS_REFRACTION_INDEX;
        const GLASS_TO_AIR: f32 = GLASS_REFRACTION_INDEX / AIR_REFRACTION_INDEX;
        const BASE_REFLECTANCE: f32 = (GLASS_REFRACTION_INDEX - AIR_REFRACTION_INDEX)
            * (GLASS_REFRACTION_INDEX - AIR_REFRACTION_INDEX)
            / ((GLASS_REFRACTION_INDEX + AIR_REFRACTION_INDEX)
                * (GLASS_REFRACTION_INDEX + AIR_REFRACTION_INDEX));

        let reflected = direction.reflect(normal);

        let direction = match self.reflexivity {
            Reflexivity::Diffuse => (normal + UnitVector::random(rng)).normalized(),
            Reflexivity::Specular => reflected,
            Reflexivity::Refraction => {
                let refraction_factor =
                    select(front_face, AIR_TO_GLASS, GLASS_TO_AIR);

                let cost = direction.dot(-normal).min(1f32);
                let sint = (1f32 - cost * cost).sqrt();

                let cannot_refract = refraction_factor * sint > 1f32;

                let reflectance =
                    BASE_REFLECTANCE + (1f32 - BASE_REFLECTANCE) * (1f32 - cost).powi(5);

                if cannot_refract || reflectance > rng.random::<f32>() {
                    reflected
                } else {
                    let perp = refraction_factor * (direction + cost * normal);
                    let parallel = -(1f32 - perp.norm2()).abs().sqrt() * normal;
                    (perp + parallel).normalized()
                }
            }
        };

        Radiance {
            emission,
            absorption,
            origin: intersection_point,
            direction,
        }
    }
}

impl<'a> SphereSlice<'a> {
    pub fn intersect(self, origin: Vector, direction: UnitVector) -> Option<(SphereRef<'a>, f32)> {
        let mut closest = f32::INFINITY;
        let mut found = usize::MAX;

        for (i, sphere) in self.iter().enumerate() {
            let t = sphere.intersect(origin, direction);
            let better = t > EPS && t < closest;
            closest = select(better, t, closest);
            found = select(better, i, found);
        }

        self.split_at(found.min(self.len()))
            .1
            .split_first()
            .map(|(sphere, _)| (sphere, closest))
    }
}
