//! Headless timing/quality benchmark for the light-sampling estimators.
//! Throwaway (not wired into the app). Run from the repo root:
//!
//!   cargo run --release -p rust-tracer-standalone --example bench -- samples 256
//!   cargo run --release -p rust-tracer-standalone --example bench -- time 6
//!   ... append `calib` for a 4-config subset.
//!
//! `samples` mode: render `spp` independent paths of every pixel; report the
//! per-sample variance V (mean over pixels/channels) and derive the
//! equal-time quality MSE@1s = V*tau (tau = time per full-image sample).
//!
//! `time` mode (the direct test): render each estimator for a fixed
//! wall-clock budget and measure the RMSE of the resulting image against a
//! converged NEE reference; the reference's own variance is subtracted so it
//! does not bias the comparison. This measures convergence-per-second without
//! assuming MSE = V/n, and catches any estimator bias.

use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use raytracer::{Scene, SceneDefinition};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Clone, Copy)]
struct Config {
    label: &'static str,
    light_sampling: bool,
    mixture_sampling: f32,
    mixture_decay: f32,
}

fn load() -> SceneDefinition {
    config::Config::builder()
        .add_source(config::File::with_name(
            "rust-tracer-standalone/pitracer.yml",
        ))
        .build()
        .unwrap()
        .try_deserialize::<SceneDefinition>()
        .unwrap()
}

fn scene_for(cfg: Config) -> (Scene, u32, u32) {
    let mut def = load();
    def.light_sampling = cfg.light_sampling;
    def.mixture_sampling = cfg.mixture_sampling;
    def.mixture_decay = cfg.mixture_decay;
    def.samples = 1;
    let (w, h) = (def.camera.width, def.camera.height);
    (Scene::from(def), w, h)
}

const NEE: Config = Config {
    label: "NEE",
    light_sampling: true,
    mixture_sampling: 0.0,
    mixture_decay: 1.0,
};

fn configs(calib: bool) -> Vec<Config> {
    let mut v = vec![
        Config {
            label: "no-light-sampling",
            light_sampling: false,
            mixture_sampling: 0.0,
            mixture_decay: 1.0,
        },
        NEE,
    ];
    if calib {
        for (label, ms, dc) in [
            ("mix s=0.30 d=0.90", 0.3, 0.9),
            ("mix s=0.30 d=1.00", 0.3, 1.0),
        ] {
            v.push(Config {
                label,
                light_sampling: false,
                mixture_sampling: ms,
                mixture_decay: dc,
            });
        }
    } else {
        for ms in [0.1f32, 0.2, 0.3, 0.4, 0.5] {
            for dc in [0.5f32, 0.7, 0.9, 1.0] {
                let label: &'static str =
                    Box::leak(format!("mix s={ms:.2} d={dc:.2}").into_boxed_str());
                v.push(Config {
                    label,
                    light_sampling: false,
                    mixture_sampling: ms,
                    mixture_decay: dc,
                });
            }
        }
    }
    v
}

// ---- samples mode ---------------------------------------------------------

/// (wall seconds for `spp`, mean per-sample variance V, max channel value).
fn measure_variance(cfg: Config, spp: usize) -> (f64, f64, f32) {
    let (scene, w, h) = scene_for(cfg);
    let start = Instant::now();
    let rows: Vec<(f64, f32)> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut rng = SmallRng::seed_from_u64(0x9E37_79B9_7F4A_7C15 ^ y as u64);
            let mut var_sum = 0f64;
            let mut max = 0f32;
            for x in 0..w {
                let mut mean = [0f64; 3];
                let mut m2 = [0f64; 3];
                for i in 1..=spp {
                    let c = scene.pixel_radiance(x, y, &mut rng);
                    let s = [c.r, c.g, c.b];
                    max = max.max(c.r).max(c.g).max(c.b);
                    for ch in 0..3 {
                        let d = s[ch] as f64 - mean[ch];
                        mean[ch] += d / i as f64;
                        m2[ch] += d * (s[ch] as f64 - mean[ch]);
                    }
                }
                var_sum += (m2[0] + m2[1] + m2[2]) / (3.0 * (spp - 1).max(1) as f64);
            }
            (var_sum, max)
        })
        .collect();
    let elapsed = start.elapsed().as_secs_f64();
    let pixels = (w as u64 * h as u64) as f64;
    let v = rows.iter().map(|r| r.0).sum::<f64>() / pixels;
    let max = rows.iter().map(|r| r.1).fold(0f32, f32::max);
    (elapsed, v, max)
}

fn run_samples(spp: usize, calib: bool) {
    let def = load();
    println!(
        "[samples] {}x{}, depth {}, spp={spp}\n",
        def.camera.width, def.camera.height, def.recursion_depth
    );
    println!(
        "{:<20} {:>8} {:>10} {:>12} {:>10}",
        "estimator", "time(s)", "V", "MSE@1s", "max"
    );
    let mut rows = Vec::new();
    for cfg in configs(calib) {
        let (t, v, max) = measure_variance(cfg, spp);
        let mse_1s = v * (t / spp as f64);
        rows.push((cfg.label, mse_1s));
        println!(
            "{:<20} {:>8.2} {:>10.3e} {:>12.3e} {:>10.0}",
            cfg.label, t, v, mse_1s, max
        );
    }
    report_efficiency(&rows);
}

// ---- time mode ------------------------------------------------------------

/// Converged reference image (NEE): per-pixel mean RGB and per-pixel MSE of
/// that mean (= sample variance / spp, channel-averaged), row-major.
fn reference(spp: usize) -> (Vec<[f64; 3]>, Vec<f64>, u32, u32) {
    let (scene, w, h) = scene_for(NEE);
    let rows: Vec<(Vec<[f64; 3]>, Vec<f64>)> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut rng = SmallRng::seed_from_u64(0xFEED_0000_0000_0000u64 ^ y as u64);
            let mut means = Vec::with_capacity(w as usize);
            let mut mses = Vec::with_capacity(w as usize);
            for x in 0..w {
                let mut mean = [0f64; 3];
                let mut m2 = [0f64; 3];
                for i in 1..=spp {
                    let c = scene.pixel_radiance(x, y, &mut rng);
                    let s = [c.r, c.g, c.b];
                    for ch in 0..3 {
                        let d = s[ch] as f64 - mean[ch];
                        mean[ch] += d / i as f64;
                        m2[ch] += d * (s[ch] as f64 - mean[ch]);
                    }
                }
                let var = (m2[0] + m2[1] + m2[2]) / (3.0 * (spp - 1) as f64);
                means.push(mean);
                mses.push(var / spp as f64);
            }
            (means, mses)
        })
        .collect();
    let mut ref_mean = Vec::new();
    let mut ref_mse = Vec::new();
    for (m, s) in rows {
        ref_mean.extend(m);
        ref_mse.extend(s);
    }
    (ref_mean, ref_mse, w, h)
}

/// Render for `budget` seconds; return (samples, elapsed, RMSE vs reference).
fn render_for_time(
    cfg: Config,
    budget: f64,
    ref_mean: &[[f64; 3]],
    ref_mse: &[f64],
) -> (usize, f64, f64) {
    let (scene, w, h) = scene_for(cfg);
    let npix = (w as usize) * (h as usize);
    let mut sum = vec![[0f64; 3]; npix];
    let mut n = 0usize;
    let start = Instant::now();
    while start.elapsed().as_secs_f64() < budget {
        let b = n as u64;
        let rows: Vec<Vec<[f64; 3]>> = (0..h)
            .into_par_iter()
            .map(|y| {
                let mut rng =
                    SmallRng::seed_from_u64(0x7115_0000_0000_0000u64 ^ (b << 20) ^ y as u64);
                (0..w)
                    .map(|x| {
                        let c = scene.pixel_radiance(x, y, &mut rng);
                        [c.r as f64, c.g as f64, c.b as f64]
                    })
                    .collect()
            })
            .collect();
        for (y, row) in rows.iter().enumerate() {
            let off = y * w as usize;
            for (x, px) in row.iter().enumerate() {
                for ch in 0..3 {
                    sum[off + x][ch] += px[ch];
                }
            }
        }
        n += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();

    // Bias-corrected MSE: E[(mean - ref)^2] = var(mean) + var(ref); subtract
    // the known reference variance to recover var(mean), the estimator's own
    // MSE.
    let mut mse = 0f64;
    for i in 0..npix {
        let mut e = 0f64;
        for ch in 0..3 {
            let d = sum[i][ch] / n as f64 - ref_mean[i][ch];
            e += d * d;
        }
        mse += e / 3.0 - ref_mse[i];
    }
    (n, elapsed, (mse / npix as f64).max(0.0).sqrt())
}

fn run_time(budget: f64, calib: bool) {
    const REF_SPP: usize = 1024;
    let def = load();
    println!(
        "[time] {}x{}, depth {}, budget={budget}s, reference=NEE {REF_SPP}spp\n",
        def.camera.width, def.camera.height, def.recursion_depth
    );
    let t0 = Instant::now();
    let (ref_mean, ref_mse, _, _) = reference(REF_SPP);
    println!("(reference built in {:.1}s)\n", t0.elapsed().as_secs_f64());

    println!(
        "{:<20} {:>9} {:>8} {:>12}",
        "estimator", "samples", "time(s)", "RMSE"
    );
    let mut rows = Vec::new();
    for cfg in configs(calib) {
        let (n, t, rmse) = render_for_time(cfg, budget, &ref_mean, &ref_mse);
        rows.push((cfg.label, rmse * rmse)); // store MSE for efficiency ratio
        println!("{:<20} {:>9} {:>8.2} {:>12.5}", cfg.label, n, t, rmse);
    }
    report_efficiency(&rows);
}

// ---- shared ---------------------------------------------------------------

/// `rows` holds (label, MSE-like) where lower is better; report ratio vs NEE.
fn report_efficiency(rows: &[(&str, f64)]) {
    if let Some(&(_, nee)) = rows.iter().find(|(l, _)| *l == "NEE") {
        println!("\nefficiency vs NEE (equal-time MSE ratio, >1 = better than NEE):");
        for (label, mse) in rows {
            println!("{:<20} {:>6.2}x", label, nee / mse);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("samples");
    let calib = args.iter().any(|a| a == "calib");
    match mode {
        "time" => run_time(
            args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6.0),
            calib,
        ),
        "samples" => run_samples(
            args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256),
            calib,
        ),
        other => eprintln!("unknown mode {other:?}; use `samples <spp>` or `time <seconds>`"),
    }
}
