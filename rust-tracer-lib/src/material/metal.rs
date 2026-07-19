use serde::{Deserialize, Serialize};

use crate::{Hit, Material, UnitVector};

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Metal {
    /// The historical `fuziness` spelling stays accepted; unknown keys are
    /// dropped by the flatten composition, so without the alias old files
    /// would quietly become perfect mirrors.
    #[serde(alias = "fuziness")]
    pub fuzziness: f32,
}

impl<R: crate::Rng + ?Sized> Material<R> for Metal {
    fn scatter(&self, direction: UnitVector, hit: Hit, rng: &mut R) -> Option<UnitVector> {
        let reflected = direction.reflect(hit.normal);
        if self.fuzziness > 0f32 {
            Some((reflected + self.fuzziness * UnitVector::random(rng)).normalized())
        } else {
            Some(reflected)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_utils::{assert_vec_close, hit_at, rng, seeds};
    use crate::{Dielectric, Vector};

    /// Random (incoming, normal) pair respecting the hit convention
    /// (normal opposes the incoming direction).
    fn fixture(rng: &mut rand::rngs::SmallRng) -> (UnitVector, Hit) {
        let incoming = UnitVector::random(rng);
        let normal = UnitVector::random(rng);
        let normal = crate::select(incoming.dot(normal) < 0f32, normal, -normal);
        (incoming, hit_at(Vector::default(), normal, true))
    }

    #[test]
    fn mirror_when_fuzziness_zero() {
        let material: &dyn Material = &Metal { fuzziness: 0f32 };
        let mut rng = rng(seeds()[0]);
        for _ in 0..100 {
            let (incoming, hit) = fixture(&mut rng);
            let mirror = incoming.reflect(hit.normal);

            let scattered = material
                .scatter(incoming, hit, &mut rng)
                .expect("must scatter");
            assert_vec_close(Vector::from(scattered), Vector::from(mirror), 1e-7);

            // deterministic: independent of the rng state
            let again = material
                .scatter(incoming, hit, &mut rng)
                .expect("must scatter");
            assert_eq!(Vector::from(scattered), Vector::from(again));
        }
    }

    /// Fuzz model: scattered = normalize(mirror + fuzziness·u) with |u| = 1,
    /// so the angle to the mirror direction is at most asin(fuzziness):
    /// scattered·mirror ≥ √(1 − fuzziness²). The distribution inside the cone
    /// is a modeling choice; only the bound is spec.
    #[test]
    fn fuzz_cone_bound() {
        let mut rng = rng(seeds()[1]);
        for fuzziness in [0.1f32, 0.5f32, 0.9f32] {
            let material: &dyn Material = &Metal { fuzziness };
            let bound = (1f32 - fuzziness * fuzziness).sqrt();
            let (incoming, hit) = fixture(&mut rng);
            let mirror = incoming.reflect(hit.normal);
            for _ in 0..10_000 {
                let scattered = material
                    .scatter(incoming, hit, &mut rng)
                    .expect("must scatter");
                assert!(
                    scattered.dot(mirror) >= bound - 1e-5,
                    "fuzz {fuzziness}: {scattered:?} is {} > asin(fuzziness) away from {mirror:?}",
                    scattered.dot(mirror).acos(),
                );
            }
        }
    }

    /// MIS-critical spec: a specular/delta BSDF has no density with respect
    /// to solid angle, so mixing it with light sampling via pdf ratios is
    /// ill-defined. `pdf == 0` is the sentinel "exclude me from the mixture";
    /// the scene-level consequence is tested in scene.rs.
    #[test]
    fn pdf_is_zero_delta() {
        let mut rng = rng(seeds()[2]);
        let materials: [&dyn Material; 3] = [
            &Metal { fuzziness: 0f32 },
            &Metal { fuzziness: 0.5f32 },
            &Dielectric::new(1.5f32),
        ];
        for material in materials {
            for _ in 0..10 {
                let (incoming, hit) = fixture(&mut rng);
                let mirror = incoming.reflect(hit.normal);
                assert_eq!(material.pdf(incoming, hit, mirror), 0f32);
                assert_eq!(material.pdf(incoming, hit, hit.normal), 0f32);
                assert_eq!(
                    material.pdf(incoming, hit, UnitVector::random(&mut rng)),
                    0f32
                );
            }
        }
    }
}
