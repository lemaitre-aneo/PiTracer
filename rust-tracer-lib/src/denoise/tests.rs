use super::*;
use crate::tile_grid;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

fn grid(size: u32) -> Vec<Block> {
    tile_grid(WIDTH, HEIGHT, size, size)
}

fn block(x: u32, y: u32, size: u32) -> Block {
    Block {
        x,
        y,
        width: size,
        height: size,
    }
}

#[test]
fn task_type_options() {
    let mut options = HashMap::new();
    assert_eq!(
        TaskType::from_options(&options).unwrap(),
        TaskType::PathTracing
    );
    for kind in [TaskType::PathTracing, TaskType::Denoising] {
        kind.to_options(&mut options);
        assert_eq!(TaskType::from_options(&options).unwrap(), kind);
    }
    options.insert(TASK_TYPE_OPTION.into(), "rendering".into());
    assert!(matches!(
        TaskType::from_options(&options),
        Err(WireError::InvalidOption { .. })
    ));
}

#[test]
fn tiles_grow_by_the_halo_within_the_image() {
    assert_eq!(
        denoise_tile(&block(256, 192, 64), WIDTH, HEIGHT),
        Block {
            x: 160,
            y: 96,
            width: 256,
            height: 256
        }
    );
    // Clipped at the image borders.
    assert_eq!(
        denoise_tile(&block(0, 0, 64), WIDTH, HEIGHT),
        Block {
            x: 0,
            y: 0,
            width: 160,
            height: 160
        }
    );
    assert_eq!(
        denoise_tile(&block(576, 416, 64), WIDTH, HEIGHT),
        Block {
            x: 480,
            y: 320,
            width: 160,
            height: 160
        }
    );
}

#[test]
fn neighbourhoods_grow_with_the_halo() {
    // 96 px of halo over 64-px blocks: 2 blocks on each side, 5x5.
    let blocks = grid(64);
    assert_eq!(
        neighborhood(&blocks, &block(256, 192, 64), WIDTH, HEIGHT).len(),
        25
    );
    assert_eq!(
        neighborhood(&blocks, &block(0, 0, 64), WIDTH, HEIGHT).len(),
        9
    );
    // 128-px blocks need only one: 3x3.
    let blocks = grid(128);
    assert_eq!(
        neighborhood(&blocks, &block(256, 128, 128), WIDTH, HEIGHT).len(),
        9
    );
}

/// A grid of results whose pixels encode their own coordinates.
fn results(blocks: &[Block]) -> Vec<BlockResult> {
    blocks
        .iter()
        .map(|block| BlockResult {
            block: *block,
            samples: block.x + 1,
            pixels: (block.y..block.y + block.height)
                .flat_map(|y| {
                    (block.x..block.x + block.width).map(move |x| Color {
                        r: x as f32,
                        g: y as f32,
                        b: 1.0,
                    })
                })
                .collect(),
        })
        .collect()
}

#[test]
fn tiles_are_assembled_from_the_neighbourhood_and_cropped_back() {
    let blocks = grid(64);
    // Interior, top-left corner, and the clipped bottom-right corner block.
    let corner = Block {
        x: 576,
        y: 448,
        width: 64,
        height: 32,
    };
    assert!(blocks.contains(&corner));
    for center in [block(256, 192, 64), block(0, 0, 64), corner] {
        let inputs: Vec<_> = results(&neighborhood(&blocks, &center, WIDTH, HEIGHT));
        let tile = Tile::assemble(&center, WIDTH, HEIGHT, &inputs).unwrap();
        assert_eq!(tile.area, denoise_tile(&center, WIDTH, HEIGHT));
        // Every tile pixel comes from the right place of the right block.
        for (i, rgb) in tile.color.chunks_exact(3).enumerate() {
            let x = tile.area.x + i as u32 % tile.area.width;
            let y = tile.area.y + i as u32 / tile.area.width;
            assert_eq!(rgb, [x as f32, y as f32, 1.0]);
        }
        // Cropping the block gives it back, with its sample count.
        let own = inputs.iter().find(|input| input.block == center).unwrap();
        assert_eq!(&tile.crop(&center), own);
    }
}

#[test]
fn incomplete_or_overlapping_inputs_are_errors() {
    let blocks = grid(64);
    let center = block(256, 192, 64);
    let inputs = results(&neighborhood(&blocks, &center, WIDTH, HEIGHT));
    let without = |skip: Block| -> Vec<BlockResult> {
        inputs
            .iter()
            .filter(|input| input.block != skip)
            .cloned()
            .collect()
    };
    for broken in [
        without(block(128, 64, 64)),
        without(center),
        inputs.iter().chain(&inputs[..1]).cloned().collect(),
    ] {
        assert!(matches!(
            Tile::assemble(&center, WIDTH, HEIGHT, &broken),
            Err(WireError::IncompleteTile { .. })
        ));
    }
}

fn payload() -> DenoisePayload {
    DenoisePayload {
        block: block(64, 0, 64),
        inputs: vec![DenoiseInput {
            block: block(0, 0, 64),
            result: "linear-final-0-0".into(),
        }],
        outputs: BlockOutputs {
            linear: "linear-denoised-64-0".into(),
            display: "display-denoised-64-0".into(),
        },
    }
}

#[test]
fn payload_round_trip_and_validation() {
    let payload = payload();
    assert_eq!(DenoisePayload::decode(&payload.encode()).unwrap(), payload);
    let mut json: serde_json::Value = serde_json::from_slice(&payload.encode()).unwrap();
    json["extra"] = 1.into();
    assert!(DenoisePayload::decode(json.to_string().as_bytes()).is_err());

    let camera = |width, height| {
        let mut definition = crate::decode_scene(b"{}").unwrap();
        definition.camera.width = width;
        definition.camera.height = height;
        crate::Scene::from(definition).camera
    };
    assert!(validate_denoise(&payload, &camera(128, 64)).is_ok());
    assert!(matches!(
        validate_denoise(&payload, &camera(100, 64)),
        Err(WireError::OutOfImage { .. })
    ));
}
