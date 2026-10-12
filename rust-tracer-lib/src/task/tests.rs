use rand::RngExt as _;

use super::*;
use crate::test_utils::{Stats, assert_stat_close, rng};

const BLOCK: Block = Block {
    x: 2,
    y: 1,
    width: 3,
    height: 2,
};

/// Camera at the origin looking down -Z, with an emissive quad covering
/// the upper half of the view (y > 0) and nothing below.
fn half_lit_scene(width: u32, height: u32) -> Scene {
    let json = format!(
        r#"{{
            "camera": {{
                "origin": [0, 0, 0],
                "direction": [0, 0, -1],
                "field_of_view": 90,
                "width": {width},
                "height": {height}
            }},
            "objects": [{{
                "type": "quad",
                "position": [-100, 0, -10],
                "u": [200, 0, 0],
                "v": [0, 100, 0],
                "emission": [1, 2, 3]
            }}]
        }}"#
    );
    decode_scene(json.as_bytes()).unwrap().into()
}

/// Empty scene with the default gamma (2.2).
fn gamma_scene() -> Scene {
    decode_scene(b"{}").unwrap().into()
}

fn outputs() -> BlockOutputs {
    BlockOutputs {
        linear: "linear-final".into(),
        display: "display-final".into(),
    }
}

fn payload(block: Block, seed: u64) -> BlockPayload {
    BlockPayload {
        block,
        samples: 4,
        seed,
        outputs: outputs(),
        previous: None,
    }
}

#[test]
fn payload_round_trip() {
    let first = payload(BLOCK, u64::MAX);
    assert_eq!(BlockPayload::decode(&first.encode()).unwrap(), first);
    let next = first.next("accumulation".into());
    assert_eq!(BlockPayload::decode(&next.encode()).unwrap(), next);
}

#[test]
fn payload_rejects_unknown_or_missing_fields() {
    let outputs = r#""outputs":{"linear":"a","display":"b"}"#;
    let block = r#""block":{"x":0,"y":0,"width":1,"height":1}"#;
    let ok = format!(r#"{{{block},"samples":1,"seed":0,{outputs}}}"#);
    assert!(BlockPayload::decode(ok.as_bytes()).is_ok());
    for bad in [
        format!(r#"{{{block},"samples":1,"seed":0,{outputs},"sample":2}}"#),
        format!(r#"{{{block},"samples":1,"seed":0}}"#),
    ] {
        assert!(
            matches!(
                BlockPayload::decode(bad.as_bytes()),
                Err(WireError::Json(_))
            ),
            "{bad}"
        );
    }
}

#[test]
fn next_payload_continues_the_block_with_a_new_seed() {
    let first = payload(BLOCK, 3);
    let second = first.next("linear-1".into());
    let third = second.next("linear-2".into());
    assert_eq!(second.previous.as_deref(), Some("linear-1"));
    assert_eq!(third.previous.as_deref(), Some("linear-2"));
    for next in [&second, &third] {
        assert_eq!(next.block, first.block);
        assert_eq!(next.samples, first.samples);
        assert_eq!(next.outputs, first.outputs);
    }
    let seeds = [first.seed, second.seed, third.seed];
    assert!(seeds[0] != seeds[1] && seeds[1] != seeds[2] && seeds[0] != seeds[2]);
}

#[test]
fn payload_validation() {
    let camera = half_lit_scene(8, 4).camera;
    assert!(payload(BLOCK, 0).validate(&camera).is_ok());
    // Touching the far edges is fine.
    let corner = Block {
        x: 5,
        y: 2,
        width: 3,
        height: 2,
    };
    assert!(payload(corner, 0).validate(&camera).is_ok());

    let overflowing = [
        Block { x: 6, ..corner },
        Block { y: 3, ..corner },
        Block {
            x: u32::MAX,
            ..corner
        },
    ];
    for block in overflowing {
        assert!(
            matches!(
                payload(block, 0).validate(&camera),
                Err(WireError::OutOfImage { .. })
            ),
            "{block:?}"
        );
    }
    let no_samples = BlockPayload {
        samples: 0,
        ..payload(BLOCK, 0)
    };
    assert!(matches!(
        no_samples.validate(&camera),
        Err(WireError::NoSamples)
    ));
}

#[test]
fn convergence_options() {
    let options = |pairs: &[(&str, &str)]| -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    };
    assert_eq!(
        Convergence::from_options(&options(&[])).unwrap(),
        Convergence::default()
    );
    let set = Convergence {
        error_threshold: 0.25,
        max_samples: 640,
    };
    let mut written = options(&[("other", "kept")]);
    set.to_options(&mut written);
    assert_eq!(Convergence::from_options(&written).unwrap(), set);
    assert_eq!(written["other"], "kept");

    for (name, value) in [
        (ERROR_THRESHOLD_OPTION, "abc"),
        (ERROR_THRESHOLD_OPTION, "-1"),
        (ERROR_THRESHOLD_OPTION, "NaN"),
        (MAX_SAMPLES_OPTION, "-1"),
        (MAX_SAMPLES_OPTION, "1.5"),
    ] {
        assert!(
            matches!(
                Convergence::from_options(&options(&[(name, value)])),
                Err(WireError::InvalidOption { .. })
            ),
            "{name}={value}"
        );
    }
}

fn uniform(block: Block, samples: u32, value: f32) -> BlockResult {
    BlockResult {
        block,
        samples,
        pixels: vec![
            Color {
                r: value,
                g: value,
                b: value,
            };
            block.len()
        ],
    }
}

fn result() -> BlockResult {
    BlockResult {
        block: BLOCK,
        samples: 17,
        pixels: (0..BLOCK.len())
            .map(|i| Color {
                r: i as f32,
                g: -0.5,
                b: f32::MAX,
            })
            .collect(),
    }
}

#[test]
fn result_round_trip() {
    let result = result();
    let bytes = result.encode();
    assert_eq!(bytes.len(), 24 + 12 * BLOCK.len());
    assert_eq!(BlockResult::decode(&bytes).unwrap(), result);
}

#[test]
fn empty_result_round_trip() {
    let result = BlockResult {
        block: Block { width: 0, ..BLOCK },
        samples: 1,
        pixels: Vec::new(),
    };
    assert_eq!(BlockResult::decode(&result.encode()).unwrap(), result);
}

#[test]
fn display_round_trip() {
    let display = DisplayBlock {
        block: BLOCK,
        samples: 9,
        pixels: (0..BLOCK.len() as u8).map(|i| [i, 255 - i, 128]).collect(),
    };
    let bytes = display.encode();
    assert_eq!(bytes.len(), 24 + 3 * BLOCK.len());
    assert_eq!(DisplayBlock::decode(&bytes).unwrap(), display);
}

#[test]
fn results_reject_malformed_bytes() {
    let linear = result().encode();
    let display = result().to_display(&gamma_scene()).encode();
    // Each format refuses the other.
    assert!(matches!(
        BlockResult::decode(&display),
        Err(WireError::BadMagic)
    ));
    assert!(matches!(
        DisplayBlock::decode(&linear),
        Err(WireError::BadMagic)
    ));
    let check = |bytes: Vec<u8>, decode: &dyn Fn(&[u8]) -> Result<(), WireError>| {
        for len in [0, 23, 24, bytes.len() - 1] {
            assert!(
                matches!(decode(&bytes[..len]), Err(WireError::Length { .. })),
                "truncated to {len}"
            );
        }
        let mut longer = bytes;
        longer.push(0);
        assert!(matches!(decode(&longer), Err(WireError::Length { .. })));
    };
    check(linear, &|bytes| BlockResult::decode(bytes).map(drop));
    check(display, &|bytes| DisplayBlock::decode(bytes).map(drop));
}

#[test]
fn display_applies_the_scene_gamma() {
    let scene = gamma_scene();
    let display = uniform(BLOCK, 3, 0.5).to_display(&scene);
    assert_eq!(display.block, BLOCK);
    assert_eq!(display.samples, 3);
    let level = scene.to_u8(0.5);
    assert_eq!(level, 186); // 255 * 0.5^(1/2.2), rounded
    assert!(display.pixels.iter().all(|&p| p == [level; 3]));
    // Out-of-range radiance saturates.
    let bright = uniform(BLOCK, 1, 7.0).to_display(&scene);
    assert!(bright.pixels.iter().all(|&p| p == [255; 3]));
}

#[test]
fn merge_weights_by_sample_count() {
    let merged = uniform(BLOCK, 3, 1.0).merge(&uniform(BLOCK, 1, 5.0));
    assert_eq!(merged.samples, 4);
    assert_eq!(merged, uniform(BLOCK, 4, 2.0));
}

fn convergence(error_threshold: f32, max_samples: u32) -> Convergence {
    Convergence {
        error_threshold,
        max_samples,
    }
}

#[test]
fn a_first_pass_is_final_only_when_black_or_capped() {
    let scene = gamma_scene();
    let black = accumulate(&scene, None, uniform(BLOCK, 4, 0.0), &convergence(0.0, 100)).unwrap();
    assert!(black.done);
    assert_eq!(black.error, None);

    let lit = uniform(BLOCK, 4, 1e-6);
    let first = accumulate(&scene, None, lit.clone(), &convergence(f32::MAX, 100)).unwrap();
    assert!(!first.done);
    assert_eq!(first.result, lit);

    let capped = accumulate(&scene, None, lit, &convergence(0.0, 4)).unwrap();
    assert!(capped.done);
}

#[test]
fn later_passes_stop_on_the_estimated_error_or_the_cap() {
    let scene = gamma_scene();
    let previous = uniform(BLOCK, 8, 0.5);
    let pass = uniform(BLOCK, 4, 0.6);
    let run = |convergence| {
        accumulate(&scene, Some(previous.clone()), pass.clone(), &convergence).unwrap()
    };

    let loose = run(convergence(f32::MAX, 100));
    assert!(loose.done);
    assert_eq!(loose.result, previous.merge(&pass));
    let error = loose.error.unwrap();
    assert!(error > 0.0);

    assert!(!run(convergence(error * 0.99, 100)).done);
    assert!(run(convergence(error, 100)).done);
    assert!(run(convergence(0.0, 12)).done);

    // A pass that changes nothing has no measurable error.
    let still = accumulate(
        &scene,
        Some(previous.clone()),
        uniform(BLOCK, 4, 0.5),
        &convergence(0.0, 100),
    )
    .unwrap();
    assert_eq!(still.error, Some(0.0));
    assert!(still.done);

    let elsewhere = uniform(Block { x: 0, ..BLOCK }, 4, 0.5);
    assert!(matches!(
        accumulate(&scene, Some(previous), elsewhere, &convergence(0.0, 100)),
        Err(WireError::BlockMismatch { .. })
    ));
}

/// The estimate must match the actual display-space error of the merged
/// image (against the exact radiance), not the raw change between passes,
/// which is `m / n'` = 1/3 of it here.
#[test]
fn estimated_error_is_unbiased() {
    let scene = gamma_scene();
    let block = Block {
        x: 0,
        y: 0,
        width: 20,
        height: 20,
    };
    let (previous_samples, pass_samples) = (30u32, 10u32);
    // Uniform noise of standard deviation 0.05 around each pixel's radiance.
    let half_width = 0.05 * 3f32.sqrt();
    let mut rng = rng(11);
    let truth: Vec<Color> = (0..block.len())
        .map(|_| Color {
            r: rng.random_range(0.2..0.8),
            g: rng.random_range(0.2..0.8),
            b: rng.random_range(0.2..0.8),
        })
        .collect();
    let mut mean_of = |samples: u32| BlockResult {
        block,
        samples,
        pixels: truth
            .iter()
            .map(|&exact| {
                let mut noisy = |value: f32| {
                    let sum: f32 = (0..samples)
                        .map(|_| value + rng.random_range(-half_width..half_width))
                        .sum();
                    sum / samples as f32
                };
                Color {
                    r: noisy(exact.r),
                    g: noisy(exact.g),
                    b: noisy(exact.b),
                }
            })
            .collect(),
    };

    let mut difference = Stats::new();
    let mut actual = Stats::new();
    for _ in 0..200 {
        let previous = mean_of(previous_samples);
        let merged = previous.merge(&mean_of(pass_samples));
        let estimate = estimated_display_error(&scene, &previous, &merged, pass_samples);
        let channels = |color: Color| [color.r, color.g, color.b];
        let mut error = 0f64;
        for (&exact, &got) in truth.iter().zip(&merged.pixels) {
            for (exact, got) in channels(exact).into_iter().zip(channels(got)) {
                error += ((scene.display_level(got) - scene.display_level(exact)) as f64).powi(2);
            }
        }
        let error = error / (3 * block.len()) as f64;
        difference.push(estimate as f64 - error);
        actual.push(error);
    }
    assert_stat_close(difference.mean(), difference.se(), 0, 4, 0);
    // The comparison has teeth: the error is far from zero.
    assert!(actual.mean() > 10.0 * difference.se(), "{}", actual.mean());
}

#[test]
fn display_names_share_a_prefix() {
    let prefix = format!("{DISPLAY_NAME}-");
    assert!(name(DISPLAY_NAME, &BLOCK).starts_with(&prefix));
    assert!(name(FINAL_DISPLAY_NAME, &BLOCK).starts_with(&prefix));
    assert!(!name(LINEAR_NAME, &BLOCK).starts_with(&prefix));
    assert!(!name(PAYLOAD_NAME, &BLOCK).starts_with(&prefix));
    assert_eq!(name(FINAL_LINEAR_NAME, &BLOCK), "linear-final-2-1");
}

#[test]
fn scene_round_trip() {
    let scene = r#"{
        "camera": {"origin": [1, 2, 3], "direction": [0, 0, -1], "width": 64, "height": 48},
        "samples": 7,
        "objects": [
            {"type": "sphere", "radius": 2, "position": [0, 0, -5], "material": "Metal", "fuzziness": 0.1},
            {"type": "sphere", "radius": 1, "position": [3, 0, -5], "material": "Refraction"},
            {"type": "quad", "position": [0, 0, 0], "u": [1, 0, 0], "v": [0, 1, 0], "emission": [4, 4, 4]},
            {"type": "ellipse", "position": [0, 0, 0], "u": [1, 0, 0], "v": [0, 2, 0], "texture": "marble", "scale": 3}
        ]
    }"#;
    let definition = decode_scene(scene.as_bytes()).unwrap();
    let round_trip = decode_scene(&encode_scene(&definition)).unwrap();
    // Debug prints f32 losslessly, so equal dumps mean equal scenes.
    assert_eq!(format!("{round_trip:?}"), format!("{definition:?}"));
}

#[test]
fn rows_are_in_display_order() {
    let scene = half_lit_scene(4, 4);
    let block = Block {
        x: 0,
        y: 0,
        width: 4,
        height: 4,
    };
    let payload = payload(block, 0);
    let top = render_block_row(&scene, &payload, 0);
    let bottom = render_block_row(&scene, &payload, 3);
    let lit = Color {
        r: 1.0,
        g: 2.0,
        b: 3.0,
    };
    assert_eq!(top, vec![lit; 4]);
    assert_eq!(bottom, vec![Color::default(); 4]);
}

#[test]
fn rendering_is_reproducible_and_seeded() {
    let scene: Scene = decode_scene(
        br#"{
            "camera": {"origin": [0, 0, 0], "direction": [0, 0, -1], "width": 8, "height": 8},
            "objects": [
                {"type": "sphere", "radius": 1, "position": [0, 3, -5], "emission": [5, 5, 5]},
                {"type": "sphere", "radius": 100, "position": [0, -101, -5]}
            ]
        }"#,
    )
    .unwrap()
    .into();
    let render = |seed| render_block_row(&scene, &payload(BLOCK, seed), 0);

    assert_eq!(render(1), render(1));
    assert_ne!(render(1), render(2));
}

#[test]
fn row_streams_are_distinct() {
    // Includes the seed/position pairs a naive `seed ^ y` or `seed + x`
    // scheme would map to the same stream.
    let mut seeds = std::collections::HashSet::new();
    for seed in 0..8 {
        for x in 0..8 {
            for y in 0..8 {
                assert!(seeds.insert(row_seed(seed, x, y)), "{seed} {x} {y}");
            }
        }
    }
}

#[test]
fn tiles_cover_the_image_exactly_once() {
    for (width, height, bw, bh) in [(16, 12, 4, 4), (17, 13, 5, 3), (3, 2, 60, 60), (1, 1, 1, 1)] {
        let mut covered = vec![0; (width * height) as usize];
        for block in tile_grid(width, height, bw, bh) {
            assert!(
                block.width >= 1 && block.width <= bw && block.height >= 1 && block.height <= bh
            );
            for y in block.y..block.y + block.height {
                for x in block.x..block.x + block.width {
                    covered[(y * width + x) as usize] += 1;
                }
            }
        }
        assert!(
            covered.iter().all(|&n| n == 1),
            "{width}x{height} by {bw}x{bh}"
        );
    }
}

#[test]
fn tiles_start_from_the_center() {
    let blocks = tile_grid(30, 30, 10, 10);
    assert_eq!(
        blocks[0],
        Block {
            x: 10,
            y: 10,
            width: 10,
            height: 10
        }
    );
    // Corners come last.
    assert!(blocks[5..].iter().all(|b| b.x != 10 && b.y != 10));
}
