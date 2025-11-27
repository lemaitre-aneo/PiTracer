use serde::{Deserialize, Serialize};

use crate::{Hit, Material, UnitVector};

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Metal {
    pub fuziness: f32,
}

impl<R: crate::Rng + ?Sized> Material<R> for Metal {
    fn scatter(&self, direction: UnitVector, hit: Hit, rng: &mut R) -> Option<UnitVector> {
        let reflected = direction.reflect(hit.normal);
        if self.fuziness > 0f32 {
            Some((reflected + self.fuziness * UnitVector::random(rng)).normalized())
        } else {
            Some(reflected)
        }
    }
}
