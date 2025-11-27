use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{Camera, CameraDefinition, Color, Flat, Hittable, MaterialEnum, Object, Sphere};

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SceneDefinition {
    pub camera: CameraDefinition,
    pub spheres: Vec<Object<Sphere, MaterialEnum, Flat>>,
    pub recursion_depth: u32,
    pub samples: u32,
    pub gamma: f32,
}

impl Default for SceneDefinition {
    fn default() -> Self {
        Self {
            camera: Default::default(),
            spheres: Default::default(),
            recursion_depth: 10,
            samples: 100,
            gamma: 2.2,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(from = "SceneDefinition")]
pub struct Scene {
    pub camera: Camera,
    pub hittables: Vec<Box<dyn Hittable>>,
    pub recursion_depth: u32,
    pub samples: u32,
    pub gamma: f32,
}

impl Default for Scene {
    fn default() -> Self {
        SceneDefinition::default().into()
    }
}

impl From<SceneDefinition> for Scene {
    fn from(value: SceneDefinition) -> Self {
        Self {
            camera: value.camera.into(),
            hittables: value
                .spheres
                .into_iter()
                .map(|s| Box::new(s) as Box<dyn Hittable>)
                .collect(),
            recursion_depth: value.recursion_depth,
            samples: value.samples,
            gamma: value.gamma,
        }
    }
}

impl Scene {
    pub fn radiance(&self, x: f32, y: f32, rng: &mut (dyn crate::Rng + 'static)) -> Color {
        let (mut origin, mut direction) = self.camera.cast(x, y);

        let mut colors = Vec::with_capacity(self.recursion_depth as usize);

        loop {
            let mut closest = f32::INFINITY;
            let mut best = None;
            for (i, hittable) in self.hittables.iter().enumerate() {
                if let Some((distance, hit_caster)) = hittable.hit(origin, direction, rng) {
                    if distance < closest && distance > 7e-2f32 {
                        closest = distance;
                        best = Some((hit_caster, i));
                    }
                }
            }
            let Some((best, i)) = best else {
                break;
            };
            let hittable = self.hittables[i].as_ref();
            let hit = best.as_geometry().to_hit(origin, direction, closest);

            let color = best.as_texture(hittable).get(hit, rng);
            colors.push(color);

            if colors.len() > self.recursion_depth as usize || color.attenuation == Color::default()
            {
                break;
            }

            if let Some(new_direction) = best.as_material(hittable).scatter(direction, hit, rng) {
                origin = hit.point;
                direction = new_direction;
            } else {
                break;
            }
        }

        let mut color = Color::default();
        for texture in colors.into_iter().rev() {
            color = texture.emission + color * texture.attenuation;
        }

        color
    }

    pub fn pixel_radiance(&self, x: u32, y: u32, rng: &mut (impl crate::Rng + 'static)) -> Color {
        let mut color = Color::default();
        for _ in 0..self.samples {
            let x = x as f32 + rng.random::<f32>();
            let y = y as f32 + rng.random::<f32>();

            color += self.radiance(x, y, rng);
        }

        color / self.samples as f32
    }

    pub fn to_u8_color(&self, color: Color) -> [u8; 3] {
        [
            color.r.clamp(0f32, 1f32),
            color.g.clamp(0f32, 1f32),
            color.b.clamp(0f32, 1f32),
        ]
        .map(|l| self.to_u8(l))
    }

    pub fn to_u8(&self, value: f32) -> u8 {
        (value.powf(1f32 / self.gamma) * 255f32 + 0.5f32) as u8
    }
}
