use serde::{Deserialize, Serialize};

use crate::{Hit, Material, UnitVector};

#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Lambertian {}

impl<R: crate::Rng + ?Sized> Material<R> for Lambertian {
    fn scatter(&self, _direction: UnitVector, hit: Hit, rng: &mut R) -> Option<UnitVector> {
        Some((hit.normal + UnitVector::random(rng)).normalized())
    }
}
