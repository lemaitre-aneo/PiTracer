use std::sync::Arc;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    Camera, CameraDefinition, Color, Geometry, GeometryEnum, HitCaster, Hittable, MaterialEnum,
    Object, Texture, TextureEnum, TextureProperty, UnitVector, Vector,
};

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SceneDefinition {
    pub camera: CameraDefinition,
    pub objects: Vec<Object<GeometryEnum, MaterialEnum, TextureEnum>>,
    pub recursion_depth: u32,
    pub samples: u32,
    pub gamma: f32,
    pub light_sampling: bool,
    /// Mixture continuation sampling (a distinct estimator, overriding the
    /// NEE `light_sampling` path when `> 0`). Instead of casting a separate
    /// shadow ray, the single continuation direction is drawn from a mixture
    /// of the light-target and BRDF densities (with this probability the
    /// bounce is aimed at a light, otherwise it follows the BRDF), so each
    /// diffuse bounce casts exactly one ray. `0.0` disables it and falls back
    /// to the NEE/path-tracing path. Meaningful range is `[0, 1)`: at `1.0`
    /// the BRDF term vanishes and directions the lights do not subtend are
    /// missed (biased). Default `0.25` (mixture is the default estimator).
    pub mixture_sampling: f32,
    /// Geometric decay of the mixture light probability with path depth:
    /// `c_l(d) = mixture_sampling * mixture_decay^d`. `1.0` keeps it constant
    /// at every bounce (the naive mixture, whose worst-case throughput grows
    /// like `(rho/c_b)^depth`: deep fireflies). A value `< 1` concentrates
    /// light aiming on the early bounces and relaxes to pure BRDF at depth,
    /// which keeps the worst-case path throughput a bounded constant (since
    /// `sum c_l(d)` converges): mild, sample-reducible fireflies even as
    /// rho -> 1. Default `0.99`: near the `1.0` performance optimum but
    /// strictly `< 1`, so the default estimator provably cannot diverge.
    pub mixture_decay: f32,
}

impl Default for SceneDefinition {
    fn default() -> Self {
        Self {
            camera: Default::default(),
            objects: Default::default(),
            recursion_depth: 10,
            samples: 100,
            gamma: 2.2,
            light_sampling: true,
            // Mixture continuation sampling by default (overrides NEE): ~2.5-3.4x
            // faster to converge on pitracer. decay 0.99 (< 1) keeps the
            // worst-case path throughput a bounded constant so the default can
            // never diverge into fireflies, while staying ~at the d=1.0 optimum.
            mixture_sampling: 0.25,
            mixture_decay: 0.99,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(from = "SceneDefinition")]
pub struct Scene {
    pub camera: Camera,
    pub hittables: Vec<Arc<dyn Hittable + Send + Sync>>,
    pub points_of_interest: Vec<(Arc<dyn Geometry + Send + Sync>, f32, f32)>,
    pub recursion_depth: u32,
    pub samples: u32,
    pub gamma: f32,
    pub light_sampling: bool,
    pub mixture_sampling: f32,
    pub mixture_decay: f32,
}

impl Default for Scene {
    fn default() -> Self {
        SceneDefinition::default().into()
    }
}

/// Monomorphizes one scene object (geometry and texture already concrete,
/// the material still a definition-side enum) into a hittable trait object.
fn build_hittable<G, T>(object: Object<G, MaterialEnum, T>) -> Arc<dyn Hittable + Send + Sync>
where
    G: Geometry + Send + Sync + 'static,
    T: Texture + Send + Sync + 'static,
{
    match object.material {
        MaterialEnum::Lambertian(lambertian) => {
            Arc::new(Object::new(object.geometry, lambertian, object.texture))
                as Arc<dyn Hittable + Send + Sync>
        }
        MaterialEnum::Metal(metal) => Arc::new(Object::new(object.geometry, metal, object.texture))
            as Arc<dyn Hittable + Send + Sync>,
        MaterialEnum::Dielectric(dielectric) => {
            Arc::new(Object::new(object.geometry, dielectric, object.texture))
                as Arc<dyn Hittable + Send + Sync>
        }
    }
}

/// Peels the texture tag off so `build_hittable` sees a concrete texture.
fn build_textured<G: Geometry + Send + Sync + 'static>(
    object: Object<G, MaterialEnum, TextureEnum>,
) -> Arc<dyn Hittable + Send + Sync> {
    match object.texture {
        TextureEnum::Flat(flat) => {
            build_hittable(Object::new(object.geometry, object.material, flat))
        }
        TextureEnum::Marble(marble) => {
            build_hittable(Object::new(object.geometry, object.material, marble))
        }
    }
}

/// Peels the geometry tag off a definition object so the chain below can
/// monomorphize the full geometry x texture x material product, and pairs the
/// result with its light-sampling weight |emission|^2.
fn build_object(
    object: Object<GeometryEnum, MaterialEnum, TextureEnum>,
) -> (Arc<dyn Hittable + Send + Sync>, f32) {
    let w = object.texture.emission().norm2();
    let h = match object.geometry {
        GeometryEnum::Sphere(sphere) => {
            build_textured(Object::new(sphere, object.material, object.texture))
        }
        GeometryEnum::Quad(quad) => {
            build_textured(Object::new(quad, object.material, object.texture))
        }
        GeometryEnum::Ellipse(ellipse) => {
            build_textured(Object::new(ellipse, object.material, object.texture))
        }
    };
    (h, w)
}

impl From<SceneDefinition> for Scene {
    fn from(value: SceneDefinition) -> Self {
        let hittables = value
            .objects
            .into_iter()
            .map(build_object)
            .collect::<Vec<_>>();

        let mut points_of_interest = hittables
            .iter()
            .filter_map(|(h, w)| {
                if *w > 0.0 {
                    Some((h.clone() as Arc<dyn Geometry + Send + Sync>, *w, 0.0))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        points_of_interest.sort_by(|(_, w0, _), (_, w1, _)| {
            w0.partial_cmp(w1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut total_weight = 0f32;
        for (_, w, _) in &points_of_interest {
            total_weight += *w;
        }
        let mut s = 0f32;
        for (_, w, c) in &mut points_of_interest {
            s += *w;
            *c = s / total_weight;
            *w /= total_weight;
        }
        Self {
            camera: value.camera.into(),
            hittables: hittables.into_iter().map(|(h, ..)| h).collect(),
            points_of_interest,
            recursion_depth: value.recursion_depth,
            samples: value.samples,
            gamma: value.gamma,
            light_sampling: value.light_sampling,
            mixture_sampling: value.mixture_sampling,
            mixture_decay: value.mixture_decay,
        }
    }
}

impl Scene {
    /// Nearest hit along a ray (each geometry applies its own relative
    /// self-intersection guard; see `HIT_EPSILON_REL` and `Hit::spawn`).
    fn closest_hit(
        &self,
        origin: Vector,
        direction: UnitVector,
        rng: &mut (dyn crate::Rng + 'static),
    ) -> Option<(&dyn HitCaster, &dyn Hittable, f32)> {
        let mut closest = f32::INFINITY;
        let mut best = None;
        for (i, hittable) in self.hittables.iter().enumerate() {
            if let Some((distance, hit_caster)) = hittable.hit(origin, direction, rng)
                && distance < closest
                && distance > 0f32
            {
                closest = distance;
                best = Some((hit_caster, i));
            }
        }
        best.map(|(caster, i)| (caster, self.hittables[i].as_ref() as &dyn Hittable, closest))
    }

    /// Density of the one-sample light-target strategy at `point` toward
    /// `direction` (mixture of the emissive objects' solid-angle pdfs, by
    /// their normalized weights).
    fn light_pdf(&self, point: Vector, direction: UnitVector) -> f32 {
        let mut pdf = 0f32;
        for (geometry, weight, _) in &self.points_of_interest {
            pdf += geometry.target_pdf(point, direction) * weight;
        }
        pdf
    }

    pub fn radiance(&self, x: f32, y: f32, rng: &mut (dyn crate::Rng + 'static)) -> Color {
        let (mut origin, mut direction) = self.camera.cast(x, y);

        let mut colors = Vec::with_capacity(self.recursion_depth as usize);

        // Light transport with next-event estimation: at each diffuse bounce
        // the light is sampled explicitly (direct term), while the path keeps
        // following pure BRDF samples with unweighted throughput. Emitters
        // can therefore be found by two strategies, so each pickup is
        // weighted by the balance heuristic p_used / (p_material + p_light):
        // direct samples via p_light, continuation hits via p_material
        // (carried in `emission_weight`). All weights are <= 1: unlike a
        // throughput-reweighting mixture (whose off-light weight is 2 per
        // bounce, i.e. 2^k fireflies on deep paths), no path can exceed the
        // physical bound.
        let mut emission_weight = 1f32;
        // Running mixture light probability
        // c_l = mixture_sampling * decay^depth, advanced by one multiply per
        // bounce (below) rather than a powi per intersection.
        let mut mixture_prob = self.mixture_sampling;

        while let Some((best, hittable, closest)) = self.closest_hit(origin, direction, rng) {
            let hit = best.as_geometry().to_hit(origin, direction, closest);

            let texture = best.as_texture(hittable).get(hit, rng);
            let mut color = TextureProperty {
                emission: texture.emission * emission_weight,
                attenuation: texture.attenuation,
            };

            if colors.len() + 1 > self.recursion_depth as usize
                || texture.attenuation == Color::default()
            {
                colors.push(color);
                break;
            }

            let material = best.as_material(hittable);
            let Some(scattered) = material.scatter(direction, hit, rng) else {
                colors.push(color);
                break;
            };

            let material_pdf = material.pdf(direction, hit, scattered);
            let diffuse = material_pdf > 0f32 && !self.points_of_interest.is_empty();

            // Continuation for the next bounce: the BRDF sample by default,
            // overridden by the mixture branch.
            let mut next_direction = scattered;

            if self.mixture_sampling > 0f32 && diffuse {
                // Mixture continuation sampling: NO separate shadow ray. The
                // single continuation direction is drawn from a mixture of
                // the light-target and BRDF densities,
                // p_mix = c_l*pl + c_b*pm, and the bounce throughput is
                // f*cos/p_mix. Emission is then picked up by ordinary path
                // tracing (full weight): one ray per bounce, steered at a
                // light with probability c_l. Unbiased for c_b > 0 (the BRDF
                // term keeps p_mix covering the whole lobe).
                //
                // c_l decays geometrically with depth (see `mixture_prob`).
                // The per-bounce throughput is bounded by rho/c_b, so the
                // worst-case path throughput is the product of rho/c_b(d)
                // over the bounces; a summable c_l schedule keeps
                // prod 1/(1 - c_l(d)) finite even at rho = 1 and unbounded
                // depth, so the firefly magnitude is a bounded constant
                // instead of (rho/c_b)^depth. Strong aiming lands on the
                // first bounce (where it matters and cannot compound); depth
                // relaxes to pure BRDF.
                let c_l = mixture_prob.min(1f32);
                let c_b = 1f32 - c_l;

                // With probability c_l aim the continuation at a light;
                // otherwise keep the BRDF sample already drawn in `scattered`.
                let cont_dir = if rng.random::<f32>() < c_l {
                    let random = rng.random::<f32>();
                    let candidate = match self.points_of_interest.binary_search_by(|(_, _, c)| {
                        c.partial_cmp(&random).unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                        Ok(x) => self.points_of_interest.get(x),
                        Err(x) => self.points_of_interest.get(x),
                    };
                    match candidate {
                        Some((geometry, _, _)) => geometry.target_random(hit.point, rng),
                        None => scattered,
                    }
                } else {
                    scattered
                };

                let pm = material.pdf(direction, hit, cont_dir);
                let pl = self.light_pdf(hit.point, cont_dir);
                let p_mix = c_l * pl + c_b * pm;

                // Below the horizon (pm == 0) or unsupported: zero throughput,
                // so the path contributes nothing further; terminate here.
                if pm <= 0f32 || p_mix <= 0f32 {
                    colors.push(color);
                    break;
                }

                // `attenuation` already carries rho and f*cos = rho*pm, so
                // scaling it by pm/p_mix gives the bounce throughput
                // rho*pm/p_mix.
                color.attenuation *= pm / p_mix;
                emission_weight = 1f32;
                next_direction = cont_dir;
            } else if self.light_sampling && diffuse {
                // Next-event estimation with balance-heuristic MIS: pick one
                // light by weight, sample a direction toward it, and collect
                // whatever emission that ray reaches (an occluder contributes
                // its own emission: 0 for dark objects, and other lights are
                // covered by the mixture density `light_pdf`).
                let random = rng.random::<f32>();
                let candidate = match self.points_of_interest.binary_search_by(|(_, _, c)| {
                    c.partial_cmp(&random).unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    Ok(x) => self.points_of_interest.get(x),
                    Err(x) => self.points_of_interest.get(x),
                };

                if let Some((geometry, _, _)) = candidate {
                    let light_direction = geometry.target_random(hit.point, rng);
                    // For cosine-sampled Lambertian, f*cos = albedo*pdf, so
                    // the balance-weighted direct term is
                    // rho * pm/(pl + pm) * L_e.
                    let pm = material.pdf(direction, hit, light_direction);
                    let pl = self.light_pdf(hit.point, light_direction);
                    let shadow_origin = hit.spawn(light_direction);
                    if pm > 0f32
                        && pl > 0f32
                        && let Some((direct, direct_hittable, direct_distance)) =
                            self.closest_hit(shadow_origin, light_direction, rng)
                    {
                        let direct_hit = direct.as_geometry().to_hit(
                            shadow_origin,
                            light_direction,
                            direct_distance,
                        );
                        let direct_emission = direct
                            .as_texture(direct_hittable)
                            .get(direct_hit, rng)
                            .emission;
                        color.emission += texture.attenuation * direct_emission * (pm / (pl + pm));
                    }
                }

                // Balance weight for emission the continuation ray may find.
                emission_weight =
                    material_pdf / (material_pdf + self.light_pdf(hit.point, scattered));
            } else {
                // No light sampling at this bounce (specular or no lights):
                // the continuation ray is the only strategy; full pickup.
                emission_weight = 1f32;
            }

            colors.push(color);
            mixture_prob *= self.mixture_decay;

            origin = hit.spawn(next_direction);
            direction = next_direction;
        }

        let mut color = Color::default();
        for texture in colors.into_iter().rev() {
            color = texture.emission + color * texture.attenuation;
        }

        color
    }

    pub fn pixel_radiance(&self, x: u32, y: u32, rng: &mut (impl crate::Rng + 'static)) -> Color {
        self.pixel_radiance_with(x, y, self.samples, rng)
    }

    /// [`Scene::pixel_radiance`] with an explicit sample count instead of the
    /// scene's own, so a shared scene can serve tasks asking for different
    /// sample counts.
    pub fn pixel_radiance_with(
        &self,
        x: u32,
        y: u32,
        samples: u32,
        rng: &mut (impl crate::Rng + 'static),
    ) -> Color {
        let mut color = Color::default();
        for _ in 0..samples {
            let x = x as f32 + rng.random::<f32>();
            let y = y as f32 + rng.random::<f32>();

            color += self.radiance(x, y, rng);
        }

        color / samples as f32
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

// Estimator-level ground truths (derived in the test plan, not from the code):
//
// - Furnace: a closed Lambertian enclosure with albedo rho and emission E has
//   L_inf = E/(1 - rho); truncated to m bounce terms,
//   S_m = E*(1 - rho^m)/(1 - rho). Cosine BRDF sampling makes the per-bounce
//   weight exactly rho, so every path yields the identical truncated sum:
//   zero variance with light_sampling off.
// - Direct light: a Lambertian surface (albedo rho) lit by a spherical light
//   (radiance E, radius r, distance d) exactly along the normal:
//   L = integral over the cone of (rho/pi)*E*cos dw
//     = rho*E*sin^2(theta_max) = rho*E*r^2/d^2.
// - Any correct light-sampling estimator only changes variance, never the
//   expectation: ON and OFF must converge to the same values.
#[cfg(test)]
mod test {
    use super::*;
    use crate::test_utils::{Stats, assert_close, assert_stat_close, rng, seeds};
    use crate::{Ellipse, Flat, Lambertian, Marble, Metal, Quad, Sphere, Vector};

    const WIDTH: u32 = 4;
    const HEIGHT: u32 = 4;

    fn lambertian(
        position: (f32, f32, f32),
        radius: f32,
        color: (f32, f32, f32),
        emission: (f32, f32, f32),
    ) -> Object<GeometryEnum, MaterialEnum, TextureEnum> {
        Object::new(
            GeometryEnum::Sphere(Sphere {
                radius,
                position: position.into(),
            }),
            MaterialEnum::Lambertian(Lambertian {}),
            TextureEnum::Flat(Flat {
                color: color.into(),
                emission: emission.into(),
            }),
        )
    }

    fn quad_lambertian(
        position: (f32, f32, f32),
        u: (f32, f32, f32),
        v: (f32, f32, f32),
        color: (f32, f32, f32),
        emission: (f32, f32, f32),
    ) -> Object<GeometryEnum, MaterialEnum, TextureEnum> {
        Object::new(
            GeometryEnum::Quad(Quad::new(position.into(), u.into(), v.into())),
            MaterialEnum::Lambertian(Lambertian {}),
            TextureEnum::Flat(Flat {
                color: color.into(),
                emission: emission.into(),
            }),
        )
    }

    fn ellipse_lambertian(
        position: (f32, f32, f32),
        u: (f32, f32, f32),
        v: (f32, f32, f32),
        color: (f32, f32, f32),
        emission: (f32, f32, f32),
    ) -> Object<GeometryEnum, MaterialEnum, TextureEnum> {
        Object::new(
            GeometryEnum::Ellipse(Ellipse::new(position.into(), u.into(), v.into())),
            MaterialEnum::Lambertian(Lambertian {}),
            TextureEnum::Flat(Flat {
                color: color.into(),
                emission: emission.into(),
            }),
        )
    }

    fn scene(
        camera_origin: (f32, f32, f32),
        camera_direction: (f32, f32, f32),
        objects: Vec<Object<GeometryEnum, MaterialEnum, TextureEnum>>,
        recursion_depth: u32,
        light_sampling: bool,
    ) -> Scene {
        Scene::from(SceneDefinition {
            camera: crate::CameraDefinition {
                origin: camera_origin.into(),
                direction: camera_direction.into(),
                field_of_view: 90f32,
                width: WIDTH,
                height: HEIGHT,
                world_height: 0.1,
            },
            objects,
            recursion_depth,
            samples: 1,
            gamma: 2.2,
            light_sampling,
            mixture_sampling: 0.0,
            mixture_decay: 1.0,
        })
    }

    /// n independent radiance samples through the exact image center
    /// (radiance, not pixel_radiance: no jitter, the hit point is pinned).
    /// Every sample must be finite (guards 0/0 in mixture weights).
    fn sample_center(scene: &Scene, n: usize, seed: u64) -> [Stats; 3] {
        let mut rng = rng(seed);
        let mut stats = [Stats::new(); 3];
        for _ in 0..n {
            let color = scene.radiance((WIDTH / 2) as f32, (HEIGHT / 2) as f32, &mut rng);
            assert!(
                color.r.is_finite() && color.g.is_finite() && color.b.is_finite(),
                "non-finite radiance sample: {color:?}",
            );
            stats[0].push(color.r);
            stats[1].push(color.g);
            stats[2].push(color.b);
        }
        stats
    }

    /// Camera enclosed in a uniformly emissive sphere: albedo rho, emission E.
    fn furnace(albedo: f32, emission: f32, depth: u32, light_sampling: bool) -> Scene {
        scene(
            (0f32, 0f32, 0f32),
            (0f32, 0f32, 1f32),
            vec![lambertian(
                (0f32, 0f32, 0f32),
                10f32,
                (albedo, albedo, albedo),
                (emission, emission, emission),
            )],
            depth,
            light_sampling,
        )
    }

    /// [S_depth, S_depth+2]: brackets the truncated furnace series without
    /// pinning an exact depth-counting convention (+/-1 bounce tolerated).
    fn furnace_bracket(albedo: f64, emission: f64, depth: u32) -> (f64, f64) {
        let series = |m: u32| emission * (1f64 - albedo.powi(m as i32)) / (1f64 - albedo);
        (series(depth), series(depth + 2))
    }

    /// Lambertian floor (rho = 0.75) whose surface point P = (0,0,0) has
    /// normal +Y, lit by a spherical light (E = 5, r = 0.2) at distance d = 1
    /// exactly along the normal; the camera center ray hits the floor exactly
    /// at P (closest approach to the light: 0.707 > 0.2). Expected radiance:
    /// L = rho*E*r^2/d^2 = 0.75 * 5 * 0.04 = 0.15 per channel, exact for
    /// depth >= 2.
    fn direct_light_scene(light_sampling: bool) -> Scene {
        scene(
            (1.2f32, 1.2f32, 0f32),
            (-1.2f32, -1.2f32, 0f32),
            vec![
                lambertian(
                    (0f32, -100f32, 0f32),
                    100f32,
                    (0.75f32, 0.75f32, 0.75f32),
                    (0f32, 0f32, 0f32),
                ),
                lambertian(
                    (0f32, 1f32, 0f32),
                    0.2f32,
                    (0f32, 0f32, 0f32),
                    (5f32, 5f32, 5f32),
                ),
            ],
            5,
            light_sampling,
        )
    }

    /// Same floor and camera as `direct_light_scene`, but lit by a 1x1
    /// horizontal quad (E = 5) centered at height 1 above P = (0,0,0). OFF
    /// uses the quad only as passive geometry (BRDF rays must find it); ON
    /// additionally routes NEE through target_random/target_pdf.
    fn quad_light_scene(light_sampling: bool) -> Scene {
        scene(
            (1.2f32, 1.2f32, 0f32),
            (-1.2f32, -1.2f32, 0f32),
            vec![
                lambertian(
                    (0f32, -100f32, 0f32),
                    100f32,
                    (0.75f32, 0.75f32, 0.75f32),
                    (0f32, 0f32, 0f32),
                ),
                quad_lambertian(
                    (-0.5f32, 1f32, -0.5f32),
                    (1f32, 0f32, 0f32),
                    (0f32, 0f32, 1f32),
                    (0f32, 0f32, 0f32),
                    (5f32, 5f32, 5f32),
                ),
            ],
            5,
            light_sampling,
        )
    }

    /// Same floor and camera as `quad_light_scene`, but lit by a disk light
    /// (semi-axes 0.5) centered at height 1 above P = (0,0,0). OFF uses the
    /// ellipse only as passive geometry; ON additionally routes NEE through
    /// its target_random/target_pdf.
    fn ellipse_light_scene(light_sampling: bool) -> Scene {
        scene(
            (1.2f32, 1.2f32, 0f32),
            (-1.2f32, -1.2f32, 0f32),
            vec![
                lambertian(
                    (0f32, -100f32, 0f32),
                    100f32,
                    (0.75f32, 0.75f32, 0.75f32),
                    (0f32, 0f32, 0f32),
                ),
                ellipse_lambertian(
                    (0f32, 1f32, 0f32),
                    (0.5f32, 0f32, 0f32),
                    (0f32, 0f32, 0.5f32),
                    (0f32, 0f32, 0f32),
                    (5f32, 5f32, 5f32),
                ),
            ],
            5,
            light_sampling,
        )
    }

    /// Estimator equality for the disk light: ON and OFF must converge to
    /// the same mean. The load-bearing bias detector for the ellipse's
    /// target_random/target_pdf pair inside the full NEE+MIS estimator.
    #[test]
    fn ellipse_light_sampling_invariance() {
        let off = sample_center(&ellipse_light_scene(false), 300_000, seeds()[0]);
        let on = sample_center(&ellipse_light_scene(true), 100_000, seeds()[1]);
        for (off, on) in off.iter().zip(&on) {
            let se = (off.se().powi(2) + on.se().powi(2)).sqrt();
            assert!(
                (off.mean() - on.mean()).abs() <= 4f64 * se,
                "ellipse light: ON {} vs OFF {} ({:.1}σ)",
                on.mean(),
                off.mean(),
                (off.mean() - on.mean()).abs() / se,
            );
        }
    }

    #[test]
    fn gamma_endpoints_monotone_clamp() {
        let scene = scene((0f32, 0f32, 0f32), (0f32, 0f32, 1f32), vec![], 10, false);
        assert_eq!(scene.to_u8(0f32), 0);
        assert_eq!(scene.to_u8(1f32), 255);
        let mut previous = 0u8;
        for i in 0..=1000 {
            let value = scene.to_u8(i as f32 / 1000f32);
            assert!(value >= previous, "to_u8 not monotone at {i}");
            previous = value;
        }

        let linear = Scene::from(SceneDefinition {
            gamma: 1f32,
            camera: CameraDefinition {
                direction: (0f32, 0f32, 1f32).into(),
                ..Default::default()
            },
            ..Default::default()
        });

        assert_eq!(linear.to_u8(0.5f32), 128);
        assert_eq!(scene.to_u8_color(Color::splat(2f32)), [255; 3]);
        assert_eq!(scene.to_u8_color(Color::splat(-1f32)), [0; 3]);
    }

    /// Only normalization is spec (relative weighting between lights is
    /// implementation freedom): weights > 0 summing to 1, CDF non-decreasing
    /// and ending at 1. Emissive quads must enter the pool exactly like
    /// emissive spheres; dark ones must not.
    #[test]
    fn points_of_interest_normalized() {
        let scene6 = scene(
            (0f32, 0f32, 0f32),
            (0f32, 0f32, 1f32),
            vec![
                lambertian(
                    (0f32, 0f32, 5f32),
                    1f32,
                    (0.5f32, 0.5f32, 0.5f32),
                    (0f32, 0f32, 0f32),
                ),
                lambertian(
                    (3f32, 0f32, 5f32),
                    1f32,
                    (0f32, 0f32, 0f32),
                    (2f32, 1f32, 0.5f32),
                ),
                lambertian(
                    (-3f32, 0f32, 5f32),
                    1f32,
                    (0.2f32, 0.2f32, 0.2f32),
                    (0f32, 0f32, 0f32),
                ),
                lambertian(
                    (0f32, 3f32, 5f32),
                    1f32,
                    (0f32, 0f32, 0f32),
                    (0.5f32, 0.5f32, 0.5f32),
                ),
                quad_lambertian(
                    (-1f32, 4f32, 4f32),
                    (2f32, 0f32, 0f32),
                    (0f32, 0f32, 2f32),
                    (0f32, 0f32, 0f32),
                    (1f32, 1f32, 1f32),
                ),
                quad_lambertian(
                    (-1f32, -4f32, 4f32),
                    (2f32, 0f32, 0f32),
                    (0f32, 0f32, 2f32),
                    (0.5f32, 0.5f32, 0.5f32),
                    (0f32, 0f32, 0f32),
                ),
            ],
            10,
            true,
        );
        assert_eq!(scene6.points_of_interest.len(), 3);
        let mut total = 0f32;
        let mut cdf = 0f32;
        for (_, weight, cumulative) in &scene6.points_of_interest {
            assert!(*weight > 0f32);
            total += weight;
            assert!(*cumulative >= cdf, "CDF must be non-decreasing");
            cdf = *cumulative;
        }
        assert_close(total, 1f32, 1e-6);
        assert_close(cdf, 1f32, 1e-6);

        let scene1 = furnace(0.5f32, 0.5f32, 10, true);
        assert_eq!(scene1.points_of_interest.len(), 1);
        assert_close(scene1.points_of_interest[0].1, 1f32, 1e-6);
        assert_close(scene1.points_of_interest[0].2, 1f32, 1e-6);
    }

    /// Purely emissive enclosure: every path terminates on the first hit with
    /// exactly E, with and without light sampling.
    #[test]
    fn emissive_enclosure_exact() {
        for light_sampling in [false, true] {
            let scene = scene(
                (0f32, 0f32, 0f32),
                (0f32, 0f32, 1f32),
                vec![lambertian(
                    (0f32, 0f32, 0f32),
                    10f32,
                    (0f32, 0f32, 0f32),
                    (0.25f32, 0.5f32, 0.75f32),
                )],
                10,
                light_sampling,
            );
            let mut rng = rng(seeds()[0]);
            for (x, y) in [
                (2f32, 2f32),
                (0.5f32, 0.5f32),
                (3.5f32, 1f32),
                (1f32, 3f32),
                (2.5f32, 2.5f32),
            ] {
                for _ in 0..20 {
                    let color = scene.radiance(x, y, &mut rng);
                    assert_eq!(
                        (color.r, color.g, color.b),
                        (0.25f32, 0.5f32, 0.75f32),
                        "light_sampling = {light_sampling}",
                    );
                }
            }
        }
    }

    /// A closed box of purely emissive quads around the camera: every path
    /// terminates on its first hit with exactly E. Pins the full loop
    /// (camera -> quad hit -> to_hit -> emission pickup) with zero MC noise,
    /// with and without light sampling.
    #[test]
    fn emissive_quad_box_exact() {
        let black = (0f32, 0f32, 0f32);
        let emission = (0.25f32, 0.5f32, 0.75f32);
        let walls = [
            (
                (-5f32, -5f32, -5f32),
                (0f32, 10f32, 0f32),
                (0f32, 0f32, 10f32),
            ),
            (
                (5f32, -5f32, -5f32),
                (0f32, 10f32, 0f32),
                (0f32, 0f32, 10f32),
            ),
            (
                (-5f32, -5f32, -5f32),
                (10f32, 0f32, 0f32),
                (0f32, 0f32, 10f32),
            ),
            (
                (-5f32, 5f32, -5f32),
                (10f32, 0f32, 0f32),
                (0f32, 0f32, 10f32),
            ),
            (
                (-5f32, -5f32, -5f32),
                (10f32, 0f32, 0f32),
                (0f32, 10f32, 0f32),
            ),
            (
                (-5f32, -5f32, 5f32),
                (10f32, 0f32, 0f32),
                (0f32, 10f32, 0f32),
            ),
        ];
        for light_sampling in [false, true] {
            let scene = scene(
                (0f32, 0f32, 0f32),
                (0f32, 0f32, 1f32),
                walls
                    .map(|(position, u, v)| quad_lambertian(position, u, v, black, emission))
                    .to_vec(),
                10,
                light_sampling,
            );
            let mut rng = rng(seeds()[0]);
            for (x, y) in [
                (2f32, 2f32),
                (0.5f32, 0.5f32),
                (3.5f32, 1f32),
                (1f32, 3f32),
                (2.5f32, 2.5f32),
            ] {
                for _ in 0..20 {
                    let color = scene.radiance(x, y, &mut rng);
                    assert_eq!(
                        (color.r, color.g, color.b),
                        (0.25f32, 0.5f32, 0.75f32),
                        "light_sampling = {light_sampling}",
                    );
                }
            }
        }
    }

    #[test]
    fn empty_scene_black() {
        for light_sampling in [false, true] {
            let scene = scene(
                (0f32, 0f32, 0f32),
                (0f32, 0f32, 1f32),
                vec![],
                10,
                light_sampling,
            );
            let mut rng = rng(seeds()[0]);
            for _ in 0..10 {
                assert_eq!(scene.radiance(2f32, 2f32, &mut rng), Color::default());
            }
        }
    }

    /// With cosine BRDF sampling the Lambertian per-bounce weight is exactly
    /// the albedo, so every furnace path yields the identical truncated
    /// series: zero variance, value within the +/-1-bounce bracket.
    #[test]
    fn white_furnace_off_deterministic() {
        let scene = furnace(0.5f32, 0.5f32, 10, false);
        let stats = sample_center(&scene, 100, seeds()[0]);
        let (lo, hi) = furnace_bracket(0.5f64, 0.5f64, 10);
        for stats in stats {
            assert!(
                stats.var() < 1e-20,
                "BRDF-sampled furnace paths must all carry weight ρ exactly (var = {})",
                stats.var(),
            );
            assert!(
                stats.mean() >= lo - 1e-12 && stats.mean() <= hi + 1e-12,
                "L = {} outside [{lo}, {hi}]",
                stats.mean(),
            );
        }
    }

    /// PRIMARY BIAS DETECTOR. A correct one-sample mixture estimator
    /// p_mix = (p_brdf + sum_i w_i * p_light_i) / 2 has per-bounce expected
    /// weight E[(cos/pi)/p_mix] = 1 under p_mix, so the mean must match the
    /// OFF value. Light samples falling below the horizon contribute 0 and
    /// must NOT be rejected-and-retried (that renormalizes the mixture and
    /// biases it); this furnace exercises the inside-light branch where half
    /// the light samples fall below the horizon.
    #[test]
    fn white_furnace_on_unbiased() {
        let scene = furnace(0.5f32, 0.5f32, 10, true);
        let (lo, hi) = furnace_bracket(0.5f64, 0.5f64, 10);
        let expected = (lo + hi) / 2f64;
        let bracket = (hi - lo) / 2f64;
        for seed in seeds() {
            let stats = sample_center(&scene, 30_000, seed);
            for stats in stats {
                assert_stat_close(stats.mean(), stats.se(), expected, 4f32, bracket);
            }
        }
    }

    /// Closed-form anchor, BRDF sampling only: L = rho*E*r^2/d^2 = 0.15.
    /// Per-sample value is rho*E*Bernoulli(sin^2(theta_max)): sigma ~ 0.735.
    #[test]
    fn direct_light_analytic_anchor_off() {
        let scene = direct_light_scene(false);
        let stats = sample_center(&scene, 1_000_000, seeds()[0]);
        for stats in stats {
            assert_stat_close(stats.mean(), stats.se(), 0.15f32, 4f32, 0f32);
        }
    }

    /// Same anchor with light sampling: the mean must not move.
    #[test]
    fn direct_light_analytic_anchor_on() {
        let scene = direct_light_scene(true);
        let stats = sample_center(&scene, 100_000, seeds()[1]);
        for stats in stats {
            assert_stat_close(stats.mean(), stats.se(), 0.15f32, 4f32, 0f32);
        }
    }

    /// Estimator equality, independent of any closed form: ON and OFF must
    /// converge to the same mean on the same scene.
    #[test]
    fn light_sampling_invariance() {
        let off = sample_center(&direct_light_scene(false), 1_000_000, seeds()[0]);
        let on = sample_center(&direct_light_scene(true), 100_000, seeds()[1]);
        for (off, on) in off.iter().zip(&on) {
            let se = (off.se().powi(2) + on.se().powi(2)).sqrt();
            assert!(
                (off.mean() - on.mean()).abs() <= 4f64 * se,
                "direct light: ON {} vs OFF {} ({:.1}σ)",
                on.mean(),
                off.mean(),
                (off.mean() - on.mean()).abs() / se,
            );
        }

        let off = sample_center(&furnace(0.5f32, 0.5f32, 10, false), 1_000, seeds()[0]);
        let on = sample_center(&furnace(0.5f32, 0.5f32, 10, true), 30_000, seeds()[2]);
        for (off, on) in off.iter().zip(&on) {
            let se = (off.se().powi(2) + on.se().powi(2)).sqrt();
            assert!(
                (off.mean() - on.mean()).abs() <= 4f64 * se,
                "furnace: ON {} vs OFF {} ({:.1}σ)",
                on.mean(),
                off.mean(),
                (off.mean() - on.mean()).abs() / se,
            );
        }
    }

    /// Estimator equality for the quad light: ON and OFF must converge to the
    /// same mean. This is the load-bearing bias detector for the quad's
    /// target_random/target_pdf pair inside the full NEE+MIS estimator (OFF
    /// never touches that code, so a wrong pdf or sampler moves ON only).
    #[test]
    fn quad_light_sampling_invariance() {
        let off = sample_center(&quad_light_scene(false), 300_000, seeds()[0]);
        let on = sample_center(&quad_light_scene(true), 100_000, seeds()[1]);
        for (off, on) in off.iter().zip(&on) {
            let se = (off.se().powi(2) + on.se().powi(2)).sqrt();
            assert!(
                (off.mean() - on.mean()).abs() <= 4f64 * se,
                "quad light: ON {} vs OFF {} ({:.1}σ)",
                on.mean(),
                off.mean(),
                (off.mean() - on.mean()).abs() / se,
            );
        }
    }

    /// The original complaint: on a small bright light (4% BRDF hit rate),
    /// light sampling must REDUCE variance, conservatively by 2x; a correct
    /// mixture gains about an order of magnitude here.
    #[test]
    fn light_sampling_reduces_variance() {
        let n = 100_000;
        let off = sample_center(&direct_light_scene(false), n, seeds()[0]);
        let on = sample_center(&direct_light_scene(true), n, seeds()[1]);
        for (off, on) in off.iter().zip(&on) {
            eprintln!(
                "variance: OFF {:.4} vs ON {:.4} ({:.1}x reduction)",
                off.var(),
                on.var(),
                off.var() / on.var(),
            );
            assert!(
                on.var() < 0.5f64 * off.var(),
                "light sampling must reduce variance: ON {} vs OFF {}",
                on.var(),
                off.var(),
            );
        }
    }

    /// Specular (delta) materials have pdf 0 and must be excluded from the
    /// light-sampling mixture entirely: the mirror path is deterministic, so
    /// ON samples must equal the OFF value exactly.
    #[test]
    fn specular_scene_ignores_light_sampling() {
        let build = |light_sampling| {
            scene(
                (0f32, 0f32, -5f32),
                (0f32, 0f32, 1f32),
                vec![
                    Object::new(
                        GeometryEnum::Sphere(Sphere {
                            radius: 1f32,
                            position: Vector::default(),
                        }),
                        MaterialEnum::Metal(Metal { fuzziness: 0f32 }),
                        TextureEnum::Flat(Flat {
                            color: (0.9f32, 0.9f32, 0.9f32).into(),
                            emission: Color::default(),
                        }),
                    ),
                    // emissive dome enclosing the whole scene
                    lambertian(
                        (0f32, 0f32, 0f32),
                        100f32,
                        (0f32, 0f32, 0f32),
                        (1f32, 2f32, 3f32),
                    ),
                ],
                10,
                light_sampling,
            )
        };
        // mirror path: metal at z = -1, reflected straight back into the dome:
        // L = 0.9*E exactly
        let expected = [0.9f32, 1.8f32, 2.7f32];
        for light_sampling in [false, true] {
            let stats = sample_center(&build(light_sampling), 100, seeds()[0]);
            for (stats, expected) in stats.iter().zip(expected) {
                assert!(
                    stats.var() < 1e-20,
                    "specular path must stay deterministic (light_sampling = {light_sampling}, var = {})",
                    stats.var(),
                );
                assert_close(stats.mean(), expected, 1e-6);
            }
        }
    }

    /// Unit-albedo furnace: every collected term adds E exactly, so the value
    /// counts the bounce budget: E <= L <= (depth+2)*E, strictly increasing
    /// with depth.
    #[test]
    fn recursion_depth_bounds_energy() {
        let mut previous = 0f64;
        for depth in [0u32, 5, 10] {
            let stats = sample_center(&furnace(1f32, 1f32, depth, false), 100, seeds()[0]);
            let value = stats[0].mean();
            assert!(stats[0].var() < 1e-20);
            assert!(
                value >= 1f64 - 1e-12 && value <= f64::from(depth + 2) + 1e-12,
                "depth {depth}: L = {value}",
            );
            assert!(value > previous, "L must increase with depth");
            previous = value;
        }

        for depth in [0u32, 5, 10] {
            let stats = sample_center(&furnace(1f32, 1f32, depth, true), 10_000, seeds()[1]);
            let mean = stats[0].mean();
            assert!(
                mean >= 1f64 - 4f64 * stats[0].se()
                    && mean <= f64::from(depth + 2) + 4f64 * stats[0].se(),
                "depth {depth} with light sampling: L = {mean}",
            );
        }

        // rho = 0.5: the truncation depth actually varies the value
        for depth in [5u32, 10] {
            let (lo, hi) = furnace_bracket(0.5f64, 0.5f64, depth);
            let stats = sample_center(&furnace(0.5f32, 0.5f32, depth, false), 100, seeds()[2]);
            assert!(
                stats[0].mean() >= lo - 1e-12 && stats[0].mean() <= hi + 1e-12,
                "depth {depth}: L = {} outside [{lo}, {hi}]",
                stats[0].mean(),
            );
        }
    }

    /// Fireflies (speck noise): with next-event estimation every collected
    /// term is at most (1 + rho)*E_max <= 2*E_max and there are at most
    /// depth+1 of them, so no path sample can exceed 2*E_max*(depth+1), a
    /// deterministic bound, not a statistical one. A throughput-reweighting
    /// mixture estimator violates it exponentially: its off-light weight is
    /// 2 per bounce, so deep paths carry ~(2*rho)^k and show up as saturated
    /// specks on bright-albedo scenes.
    #[test]
    fn light_sampling_no_fireflies() {
        let depth = 30;
        let scene = scene(
            (0f32, 0f32, 0f32),
            (0f32, 0f32, 1f32),
            vec![
                // near-white enclosure: deep paths keep almost full throughput
                lambertian(
                    (0f32, 0f32, 0f32),
                    10f32,
                    (0.999f32, 0.999f32, 0.999f32),
                    (0f32, 0f32, 0f32),
                ),
                // small bright light inside
                lambertian(
                    (0f32, 8f32, 0f32),
                    1f32,
                    (0f32, 0f32, 0f32),
                    (10f32, 10f32, 10f32),
                ),
            ],
            depth,
            true,
        );
        let bound = 2f64 * 10f64 * f64::from(depth + 1);
        let mut rng = rng(seeds()[0]);
        let mut max = 0f64;
        for _ in 0..100_000 {
            let color = scene.radiance(2f32, 2f32, &mut rng);
            for value in [color.r, color.g, color.b] {
                assert!(value.is_finite(), "non-finite radiance {color:?}");
                max = max.max(f64::from(value));
            }
        }
        assert!(
            max <= bound,
            "firefly: max sample {max} exceeds the physical bound {bound}"
        );
    }

    /// Firefly bound with a quad light: same 2*E_max*(depth+1) invariant
    /// through the quad pdf, including shading points nearly coplanar with
    /// the light (where target_pdf is 0 and the direct term must just skip).
    #[test]
    fn quad_light_no_fireflies() {
        let depth = 30;
        let scene = scene(
            (0f32, 0f32, 0f32),
            (0f32, 0f32, 1f32),
            vec![
                // near-white enclosure: deep paths keep almost full throughput
                lambertian(
                    (0f32, 0f32, 0f32),
                    10f32,
                    (0.999f32, 0.999f32, 0.999f32),
                    (0f32, 0f32, 0f32),
                ),
                // small bright quad light inside
                quad_lambertian(
                    (-0.5f32, 8f32, -0.5f32),
                    (1f32, 0f32, 0f32),
                    (0f32, 0f32, 1f32),
                    (0f32, 0f32, 0f32),
                    (10f32, 10f32, 10f32),
                ),
            ],
            depth,
            true,
        );
        let bound = 2f64 * 10f64 * f64::from(depth + 1);
        let mut rng = rng(seeds()[0]);
        let mut max = 0f64;
        for _ in 0..100_000 {
            let color = scene.radiance(2f32, 2f32, &mut rng);
            for value in [color.r, color.g, color.b] {
                assert!(value.is_finite(), "non-finite radiance {color:?}");
                max = max.max(f64::from(value));
            }
        }
        assert!(
            max <= bound,
            "firefly: max sample {max} exceeds the physical bound {bound}"
        );
    }

    /// Mixture sampling has no 2*E_max*(depth+1) guarantee; its firefly bound
    /// is the throughput product prod_d rho/(1 - c_l(d)) with
    /// c_l(d) = mixture_sampling * decay^d. A decay < 1 makes sum c_l(d)
    /// converge, so that product (hence the max sample) stays a bounded
    /// constant even on a deep near-white enclosure, where a constant c_l
    /// (decay = 1) would grow like (rho/c_b)^depth. Both the default
    /// (0.25/0.99) and an aggressive schedule are checked; the analytic
    /// per-channel bound sum_k E_max*P_k (P_k = prod_{j<k} rho/(1 - c_l(j)))
    /// is a hard, deterministic ceiling, not a statistical one. The
    /// aggressive (0.5, 0.5) case is the load-bearing one: its bound is
    /// O(1e3) while a decay that silently stayed at 1 would carry
    /// ~2^30 = 1e9, so the test fails loudly if the decay schedule is ever
    /// dropped.
    #[test]
    fn mixture_no_fireflies() {
        let depth = 30u32;
        let em = 10f64;
        let rho = 0.999f64;
        let build = |ms: f32, decay: f32| {
            let mut scene = scene(
                (0f32, 0f32, 0f32),
                (0f32, 0f32, 1f32),
                vec![
                    // near-white enclosure: deep paths keep almost full throughput
                    lambertian(
                        (0f32, 0f32, 0f32),
                        10f32,
                        (0.999f32, 0.999f32, 0.999f32),
                        (0f32, 0f32, 0f32),
                    ),
                    // small bright light inside
                    lambertian(
                        (0f32, 8f32, 0f32),
                        1f32,
                        (0f32, 0f32, 0f32),
                        (10f32, 10f32, 10f32),
                    ),
                ],
                depth,
                false, // light_sampling irrelevant: mixture overrides it
            );
            scene.mixture_sampling = ms;
            scene.mixture_decay = decay;
            scene
        };

        for (ms, decay) in [(0.25f32, 0.99f32), (0.5f32, 0.5f32)] {
            // Analytic ceiling: sum_k E_max*P_k with the per-bounce
            // throughput capped at rho/(1 - c_l), c_l = ms*decay^k
            // (<= ms < 1, so never 1/0).
            let mut bound = 0f64;
            let mut throughput = 1f64;
            for k in 0..=depth {
                bound += em * throughput;
                let c_l = ms as f64 * (decay as f64).powi(k as i32);
                throughput *= rho / (1f64 - c_l);
            }

            let scene = build(ms, decay);
            let mut rng = rng(seeds()[0]);
            let mut max = 0f64;
            for _ in 0..100_000 {
                let color = scene.radiance(2f32, 2f32, &mut rng);
                for value in [color.r, color.g, color.b] {
                    assert!(
                        value.is_finite(),
                        "non-finite radiance {color:?} (ms={ms}, decay={decay})"
                    );
                    max = max.max(f64::from(value));
                }
            }
            assert!(
                max <= bound,
                "mixture firefly (ms={ms}, decay={decay}): max sample {max} exceeds throughput bound {bound}"
            );
        }
    }

    /// Mixture sampling (the default estimator) is unbiased: same mean as NEE
    /// on the direct-light scene (analytic 0.15). Guards the correctness of
    /// what ships by default, the way `light_sampling_invariance` guards NEE.
    #[test]
    fn mixture_light_sampling_invariance() {
        let nee = sample_center(&direct_light_scene(true), 1_000_000, seeds()[0]);
        let mut scene = direct_light_scene(true);
        scene.mixture_sampling = 0.25;
        scene.mixture_decay = 0.99;
        let mix = sample_center(&scene, 1_000_000, seeds()[1]);
        for (nee, mix) in nee.iter().zip(&mix) {
            let se = (nee.se().powi(2) + mix.se().powi(2)).sqrt();
            assert!(
                (nee.mean() - mix.mean()).abs() <= 4f64 * se,
                "mixture {} vs NEE {} ({:.1}σ)",
                mix.mean(),
                nee.mean(),
                (nee.mean() - mix.mean()).abs() / se,
            );
        }
    }

    /// Heavy tier: tighter convergence bounds (~1% relative on the anchor).
    ///
    /// `extra` absorbs the deterministic receiver displacement of ray
    /// spawning: bounce origins sit SURFACE_OFFSET_REL*(r + |C|) = 2e-4 off
    /// the floor, toward the light at distance ~1, biasing its solid angle by
    /// up to 2*2e-4 = +6e-5 on the 0.15 anchor. Resolvable at 1e7 samples
    /// (1.5 sigma) though invisible in the fast tier.
    #[test]
    #[ignore = "heavy convergence tier"]
    fn heavy_direct_light_convergence() {
        let stats = sample_center(&direct_light_scene(false), 10_000_000, seeds()[2]);
        for stats in stats {
            assert_stat_close(stats.mean(), stats.se(), 0.15f32, 4f32, 6e-5f32);
        }
        let stats = sample_center(&direct_light_scene(true), 1_000_000, seeds()[0]);
        for stats in stats {
            assert_stat_close(stats.mean(), stats.se(), 0.15f32, 4f32, 6e-5f32);
        }
    }

    /// End-to-end smoke test of the texture monomorphization: a marble
    /// Lambertian sphere inside an emissive dome renders finite radiance,
    /// bounded by the dome emission (marble attenuation < 1 everywhere).
    #[test]
    fn marble_sphere_renders_finite() {
        for light_sampling in [false, true] {
            let scene = scene(
                (0f32, 0f32, -5f32),
                (0f32, 0f32, 1f32),
                vec![
                    Object::new(
                        GeometryEnum::Sphere(Sphere {
                            radius: 1f32,
                            position: Vector::default(),
                        }),
                        MaterialEnum::Lambertian(Lambertian {}),
                        TextureEnum::Marble(Marble::default()),
                    ),
                    // emissive dome enclosing the whole scene
                    lambertian(
                        (0f32, 0f32, 0f32),
                        100f32,
                        (0f32, 0f32, 0f32),
                        (1f32, 1f32, 1f32),
                    ),
                ],
                10,
                light_sampling,
            );
            let stats = sample_center(&scene, 1_000, seeds()[0]);
            for stats in stats {
                assert!(
                    stats.mean() > 0f64 && stats.mean() < 1f64,
                    "marble-lit mean {} escapes (0, E)",
                    stats.mean(),
                );
            }
        }
    }

    /// Heavy tier: furnace unbiasedness across 5 fresh seeds.
    #[test]
    #[ignore = "heavy convergence tier"]
    fn heavy_furnace_unbiased_seeds() {
        let scene = furnace(0.5f32, 0.5f32, 10, true);
        let (lo, hi) = furnace_bracket(0.5f64, 0.5f64, 10);
        let expected = (lo + hi) / 2f64;
        let bracket = (hi - lo) / 2f64;
        for seed in [10u64, 11, 12, 13, 14] {
            let stats = sample_center(&scene, 1_000_000, seed);
            for stats in stats {
                assert_stat_close(stats.mean(), stats.se(), expected, 4f32, bracket);
            }
        }
    }
}
