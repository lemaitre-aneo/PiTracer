use serde::{Deserialize, Serialize};

use crate::camera::*;
use crate::color::*;
use crate::sphere::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Scene<C = Camera> {
    pub camera: C,
    pub spheres: SphereVec,
    pub recursion_depth: u32,
    pub samples: u32,
    pub gamma: f32,
}

impl<C: Default> Default for Scene<C> {
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

impl<C> Scene<C> {
    pub fn map_camera<T>(self, f: impl FnOnce(C) -> T) -> Scene<T> {
        Scene {
            camera: f(self.camera),
            spheres: self.spheres,
            recursion_depth: self.recursion_depth,
            samples: self.samples,
            gamma: self.gamma,
        }
    }
}

impl Scene {
    pub fn radiance(&self, x: f32, y: f32, rng: &mut impl rand::Rng) -> Color {
        let (mut origin, mut direction) = self.camera.cast(x, y);

        // eprintln!("origin: {origin:?}\tdirection: {direction:?}");
        let mut colors = Vec::with_capacity(self.recursion_depth as usize);

        loop {
            let Some((sphere, distance)) = Sphere::intersect_many(&self.spheres, origin, direction)
            else {
                break;
            };

            if colors.len() >= self.recursion_depth as usize {
                colors.push((sphere.emission.to_owned(), sphere.color.to_owned()));
                break;
            }

            let radiance = sphere.radiance(origin, direction, distance, rng);

            origin = radiance.origin;
            direction = radiance.direction;
            colors.push((radiance.emission, radiance.absorption));

            if radiance.absorption == Color::default() {
                break;
            }
        }

        let mut color = Color::default();
        for (emission, absorption) in colors.into_iter().rev() {
            color = emission + color * absorption;
        }

        color
    }

    pub fn pixel_radiance(&self, x: u32, y: u32, rng: &mut impl rand::Rng) -> Color {
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
