use std::hint::select_unpredictable;

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
        let f = origin - self.position;
        let b = -f.dot(direction);
        let r2 = self.radius * self.radius;
        let z = f + b * direction;

        let delta = r2 - z.norm2();
        let q = b + b.signum() * delta.sqrt();

        let t0 = (f.norm2() - r2) / q;
        let t1 = q;

        let t = select_unpredictable(t0 > EPS, t0, t1);
        select_unpredictable(delta > 0f32, t, 0f32)
    }

    pub fn radiance(
        self,
        origin: Vector,
        direction: UnitVector,
        distance: f32,
        rng: &mut impl rand::Rng,
    ) -> Radiance {
        let intersection_point = origin + distance * direction;
        let normal = (intersection_point - self.position).normalized();
        let into = normal.dot(direction) < 0f32;
        let normal_opposite_to_array = select_unpredictable(into, normal, -normal);

        let emission = self.emission.to_owned();
        let mut absorption = self.color.to_owned();

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
            Reflexivity::Diffuse => {
                (normal_opposite_to_array + UnitVector::random(rng)).normalized()
            }
            Reflexivity::Specular => reflected,
            Reflexivity::Refraction => {
                let refraction_factor = select_unpredictable(into, AIR_TO_GLASS, GLASS_TO_AIR);
                let angle_of_attack = direction.dot(normal_opposite_to_array);

                let cos2t = 1f32
                    - refraction_factor
                        * refraction_factor
                        * (1f32 - angle_of_attack * angle_of_attack);

                if cos2t < 0f32 {
                    reflected
                } else {
                    let sign = select_unpredictable(into, 1f32, -1f32);
                    let refracted = refraction_factor * direction
                        - sign * (angle_of_attack * refraction_factor * cos2t.sqrt()) * normal;

                    let reflectance_factor =
                        1f32 - select_unpredictable(into, -angle_of_attack, refracted.dot(normal));
                    let reflectance =
                        BASE_REFLECTANCE + (1f32 - BASE_REFLECTANCE) * reflectance_factor.powi(5);
                    let transmittance = 1f32 - reflectance;

                    let reflexion_probability = 0.25f32 + 0.5f32 * reflectance;

                    if rng.random::<f32>() < reflexion_probability {
                        absorption *= reflectance / reflexion_probability;
                        reflected
                    } else {
                        absorption *= transmittance / (1f32 - reflexion_probability);
                        refracted.normalized()
                    }
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
            closest = select_unpredictable(better, t, closest);
            found = select_unpredictable(better, i, found);
        }

        self.split_at(found.min(self.len()))
            .1
            .split_first()
            .map(|(sphere, _)| (sphere, closest))
    }
}
