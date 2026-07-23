//! Statistical helpers shared by the in-crate test modules.
//!
//! Conventions (see CLAUDE.md / test plan):
//! - all randomness is seeded (`seeds`/`rng`), tests are deterministic;
//! - statistical assertions are k-sigma bounds with k = 4: if a marginal
//!   failure shows up on a correct implementation, increase N, never k > 5.
//!
//! The renderer computes in f32, but all test-side statistics accumulate in
//! f64 (f32 Welford at 10⁶ samples would drown in rounding error). Helper
//! parameters are `impl Into<f64>` so call sites can pass either width.

use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::{Hit, Onb, UnitVector, Vector};

pub(crate) fn seeds() -> [u64; 3] {
    [1, 2, 3]
}

/// `seed_from_u64` (not `from_seed`): it SplitMix-expands the seed into a
/// well-mixed state. Feeding constant bytes directly as xoshiro state makes
/// the first outputs structurally degenerate (observed: a 1-6e-10 tail draw
/// within the first few outputs).
pub(crate) fn rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

/// Welford online mean/variance accumulator (f64).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Stats {
    n: u64,
    mean: f64,
    m2: f64,
}

impl Stats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, x: impl Into<f64>) {
        let x = x.into();
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (x - self.mean);
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Unbiased sample variance.
    pub fn var(&self) -> f64 {
        self.m2 / (self.n - 1) as f64
    }

    /// Standard error of the mean.
    pub fn se(&self) -> f64 {
        (self.var() / self.n as f64).sqrt()
    }
}

/// Deterministic closeness with a scale-relative tolerance.
#[track_caller]
pub(crate) fn assert_close(a: impl Into<f64>, b: impl Into<f64>, tol: impl Into<f64>) {
    let (a, b, tol) = (a.into(), b.into(), tol.into());
    let scale = 1f64.max(a.abs()).max(b.abs());
    assert!(
        (a - b).abs() <= tol * scale,
        "expected {b}, got {a} (diff {}, tol {})",
        (a - b).abs(),
        tol * scale,
    );
}

/// Componentwise `assert_close`.
#[track_caller]
pub(crate) fn assert_vec_close(a: Vector, b: Vector, tol: impl Into<f64>) {
    let tol = tol.into();
    assert_close(a.x, b.x, tol);
    assert_close(a.y, b.y, tol);
    assert_close(a.z, b.z, tol);
}

/// |observed - expected| <= k*se + extra, with a diagnostic in sigmas.
/// `extra` absorbs a deterministic tolerance (e.g. an analytic bracket width).
#[track_caller]
pub(crate) fn assert_stat_close(
    observed: impl Into<f64>,
    se: impl Into<f64>,
    expected: impl Into<f64>,
    k: impl Into<f64>,
    extra: impl Into<f64>,
) {
    let (observed, se, expected, k, extra) = (
        observed.into(),
        se.into(),
        expected.into(),
        k.into(),
        extra.into(),
    );
    let dist = (observed - expected).abs();
    let sigmas = (dist - extra).max(0f64) / se;
    assert!(
        dist <= k * se + extra,
        "observed {observed} vs expected {expected}: off by {dist} ≈ {sigmas:.1}σ (se {se}, extra {extra})",
    );
}

/// MC estimate of ∫_{S²} f(ω) dω = 4π·E_{ω~uniform}[f(ω)]. Returns (estimate, se).
///
/// Relies on `UnitVector::random` being uniform, which is itself under test
/// (vector.rs); run that phase first when interpreting failures.
pub(crate) fn sphere_integral<F: Into<f64>>(
    n: usize,
    rng: &mut SmallRng,
    f: impl Fn(UnitVector) -> F,
) -> (f64, f64) {
    let mut stats = Stats::new();
    for _ in 0..n {
        let w = UnitVector::random(rng);
        stats.push(4f64 * std::f64::consts::PI * f(w).into());
    }
    (stats.mean(), stats.se())
}

/// Mean of f over draws from an arbitrary direction sampler. Returns (mean, se).
pub(crate) fn sampler_mean<F: Into<f64>>(
    n: usize,
    rng: &mut SmallRng,
    mut sample: impl FnMut(&mut SmallRng) -> UnitVector,
    f: impl Fn(UnitVector) -> F,
) -> (f64, f64) {
    let mut stats = Stats::new();
    for _ in 0..n {
        stats.push(f(sample(rng)).into());
    }
    (stats.mean(), stats.se())
}

/// Hit fixture: the normal must oppose whatever incident direction the test uses.
pub(crate) fn hit_at(point: Vector, normal: UnitVector, front: bool) -> Hit {
    Hit {
        point,
        normal,
        front,
        u: 0f32,
        v: 0f32,
        distance: 1f32,
        offset: 0f32,
    }
}

/// Some unit vector orthogonal to `axis`.
pub(crate) fn tangent(axis: UnitVector) -> UnitVector {
    let other = crate::select(axis.x.abs() < 0.9, UnitVector::X, UnitVector::Y);
    axis.cross(other).normalized()
}

/// Unit vector at angle `theta` (radians, f64 for test-side trigonometry)
/// from `axis`, in the plane spanned with `tangent(axis)`.
pub(crate) fn at_angle(axis: UnitVector, theta: f64) -> UnitVector {
    (theta.cos() as f32 * axis + theta.sin() as f32 * tangent(axis)).normalized()
}

/// Wilson–Hilferty approximation of the 0.999 chi-squared quantile.
/// Accurate to <1% for df ≥ 50 (callers add a 1.02 safety factor).
pub(crate) fn chi2_crit_999(df: usize) -> f64 {
    let df = df as f64;
    let a = 2f64 / (9f64 * df);
    df * (1f64 - a + 3.0902 * a.sqrt()).powi(3)
}

/// Chi-squared goodness-of-fit of a direction sampler against a φ-symmetric
/// target distribution, over an (z = ω·axis, φ) grid in the frame of `axis`.
///
/// `bin_prob(z0, z1)` must return the analytic probability of the z-slice
/// [z0, z1] under the target distribution; φ-symmetry makes each of the
/// `n_phi` azimuthal sub-bins carry `bin_prob / n_phi`.
///
/// Also asserts the support: no sample may fall below `z_min` (within f32
/// rounding slack).
#[track_caller]
#[expect(clippy::too_many_arguments)]
pub(crate) fn chi2_directions(
    n: usize,
    rng: &mut SmallRng,
    axis: UnitVector,
    z_min: f64,
    n_z: usize,
    n_phi: usize,
    mut sample: impl FnMut(&mut SmallRng) -> UnitVector,
    bin_prob: impl Fn(f64, f64) -> f64,
) {
    let base = Onb::new(axis);
    let u = base.transform(Vector::from((1f32, 0f32, 0f32)));
    let v = base.transform(Vector::from((0f32, 1f32, 0f32)));

    let mut counts = vec![0u64; n_z * n_phi];
    let mut outside = 0u64;
    for _ in 0..n {
        let w = sample(rng);
        let z = f64::from(w.dot(axis));
        if z < z_min - 1e-6 {
            outside += 1;
            continue;
        }
        let zi = ((z - z_min) / (1f64 - z_min) * n_z as f64) as usize;
        let phi = f64::from(w.dot(v)).atan2(f64::from(w.dot(u)));
        let phi = if phi < 0f64 {
            phi + std::f64::consts::TAU
        } else {
            phi
        };
        let pi = (phi / std::f64::consts::TAU * n_phi as f64) as usize;
        counts[zi.min(n_z - 1) * n_phi + pi.min(n_phi - 1)] += 1;
    }
    assert_eq!(
        outside, 0,
        "{outside}/{n} samples below the support z_min = {z_min}"
    );

    let mut x2 = 0f64;
    for zi in 0..n_z {
        let z0 = z_min + (1f64 - z_min) * zi as f64 / n_z as f64;
        let z1 = z_min + (1f64 - z_min) * (zi + 1) as f64 / n_z as f64;
        let expected = bin_prob(z0, z1) / n_phi as f64 * n as f64;
        assert!(
            expected >= 20f64,
            "z-slice {zi} expects {expected:.1} < 20 samples per bin — increase n or coarsen bins",
        );
        for pi in 0..n_phi {
            let observed = counts[zi * n_phi + pi] as f64;
            x2 += (observed - expected) * (observed - expected) / expected;
        }
    }
    let df = n_z * n_phi - 1;
    let crit = 1.02 * chi2_crit_999(df);
    assert!(x2 < crit, "chi² = {x2:.1} ≥ {crit:.1} (df {df})");
}

/// Sampler↔pdf consistency: E_{ω~sampler}[f(ω)] must equal ∫ f(ω)·pdf(ω) dω
/// for a probe set with power against wrong normalization/exponent/axis.
#[track_caller]
pub(crate) fn assert_sampler_matches_pdf(
    n_sample: usize,
    n_integral: usize,
    rng: &mut SmallRng,
    axis: UnitVector,
    mut sample: impl FnMut(&mut SmallRng) -> UnitVector,
    pdf: impl Fn(UnitVector) -> f32,
) {
    let t = tangent(axis);
    let probes = [
        ("1", Box::new(|_| 1f64) as Box<dyn Fn(UnitVector) -> f64>),
        ("z", Box::new(move |w| f64::from(w.dot(axis)))),
        ("z²", Box::new(move |w| f64::from(w.dot(axis)).powi(2))),
        ("exp(z)", Box::new(move |w| f64::from(w.dot(axis)).exp())),
        ("t", Box::new(move |w| f64::from(w.dot(t)))),
    ];
    for (name, f) in &probes {
        let (lhs, se_l) = sampler_mean(n_sample, rng, &mut sample, f);
        let (rhs, se_r) = sphere_integral(n_integral, rng, |w| f(w) * f64::from(pdf(w)));
        let se = (se_l * se_l + se_r * se_r).sqrt();
        assert!(
            (lhs - rhs).abs() <= 4f64 * se,
            "probe {name}: E_sampler[f] = {lhs} vs ∫f·pdf = {rhs} (se {se}, {:.1}σ)",
            (lhs - rhs).abs() / se,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn chi2_crit_anchors() {
        // Reference values of the 0.999 chi-squared quantile.
        assert_close(chi2_crit_999(99), 148.2, 0.01);
        assert_close(chi2_crit_999(399), 492.0, 0.01);
    }

    #[test]
    fn welford_stats() {
        let mut stats = Stats::new();
        for x in [1f64, 2f64, 3f64, 4f64] {
            stats.push(x);
        }
        assert_close(stats.mean(), 2.5, 1e-15);
        assert_close(stats.var(), 5f64 / 3f64, 1e-15);
        assert_close(stats.se(), (5f64 / 12f64).sqrt(), 1e-15);
    }
}
