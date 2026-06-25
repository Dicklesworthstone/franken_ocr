use std::hint::black_box;
use std::time::Instant;

use franken_ocr::native_engine::tensor::Mat;
use franken_ocr::native_engine::vision_bridge::{project, PROJ_IN, PROJ_OUT, TOKENS_PER_VIEW};

fn deterministic_vec(len: usize, mut state: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mantissa = ((state >> 41) as u32) & 0x7f_ffff;
        let unit = mantissa as f32 / 8_388_608.0;
        out.push((unit - 0.5) * 0.125);
    }
    out
}

fn parse_iters() -> usize {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--iters" {
            if let Some(value) = args.next() {
                return value.parse().expect("--iters must be a positive integer");
            }
        }
    }
    12
}

fn main() {
    let iters = parse_iters();
    assert!(iters > 0, "--iters must be > 0");

    let x = Mat::from_vec(
        TOKENS_PER_VIEW,
        PROJ_IN,
        deterministic_vec(TOKENS_PER_VIEW * PROJ_IN, 0x1234_5678_9abc_def0),
    );
    let w = Mat::from_vec(
        PROJ_OUT,
        PROJ_IN,
        deterministic_vec(PROJ_OUT * PROJ_IN, 0x0f1e_2d3c_4b5a_6978),
    );
    let bias = deterministic_vec(PROJ_OUT, 0x55aa_33cc_77ee_11dd);

    let warm = project(&x, &w, Some(&bias)).expect("warmup projector run");
    black_box(&warm);

    let start = Instant::now();
    let mut checksum = 0.0f64;
    for iter in 0..iters {
        let y = project(black_box(&x), black_box(&w), Some(black_box(&bias)))
            .expect("projector run");
        checksum += y.data[(iter * 7_919) % y.data.len()] as f64;
        black_box(&y);
    }
    let elapsed = start.elapsed();
    black_box(checksum);

    let elapsed_ms = elapsed.as_secs_f64() * 1_000.0;
    println!(
        "{{\"rows\":{},\"in\":{},\"out\":{},\"iters\":{},\"elapsed_ms\":{:.6},\"per_iter_ms\":{:.6},\"checksum\":{:.9}}}",
        TOKENS_PER_VIEW,
        PROJ_IN,
        PROJ_OUT,
        iters,
        elapsed_ms,
        elapsed_ms / iters as f64,
        checksum,
    );
}
