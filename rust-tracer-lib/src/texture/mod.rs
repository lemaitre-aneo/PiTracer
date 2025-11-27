use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::{Color, Hit};

mod flat;

pub use flat::*;

pub trait Texture<R: RngCore + ?Sized = dyn RngCore>: std::fmt::Debug {
    fn get(&self, hit: Hit, rng: &mut R) -> TextureProperty;
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TextureProperty {
    pub emission: Color,
    pub attenuation: Color,
}
