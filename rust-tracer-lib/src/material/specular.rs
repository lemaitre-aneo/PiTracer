use serde::{Deserialize, Serialize};

use crate::{Hit, Material, UnitVector};

#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Specular {}

impl<R: crate::Rng + ?Sized> Material<R> for Specular {
    fn scatter(&self, direction: UnitVector, hit: Hit, _rng: &mut R) -> Option<UnitVector> {
        Some(direction.reflect(hit.normal))
    }
}
