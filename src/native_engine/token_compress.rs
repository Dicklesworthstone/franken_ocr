//! Shared vision→LLM token-compression connectors (A9, bd-3jo6.1.9).
//!
//! The zoo's connectors compress the vision tower's token grid before the
//! decoder splice. Three families exist (`docs/zoo/*.md` censuses):
//!
//! * **pixel-shuffle (space-to-depth)** — SmolVLM2 (`scale_factor` 4): pure
//!   data movement, implemented here as [`pixel_shuffle`]. The exact permute
//!   sequence is `docs/zoo/smolvlm2-spec.md` §3 ("§3 IS the A9 spec").
//! * **conv token-compressor** — Baidu/GOT's 16× SAM neck (`neck→net_2→net_3`)
//!   — already certified in place (`vision_sam.rs` `forward_with`); shared by
//!   import per the B3 precedent, never relocated.
//! * **linear projector** — the plain `Linear` bridge (Baidu 2048→1280
//!   `vision_bridge`, GOT `mm_projector_vary`, SmolVLM2 `modality_projection`
//!   12288→960) — `vision_sam::Linear::apply` / `nn::matmul` serve these.
//!
//! Every connector is exact-parity gated: pixel-shuffle is bit-exact (data
//! movement only, no arithmetic), the conv/linear paths carry their towers'
//! measured budgets.

use crate::error::{FocrError, FocrResult};

use super::tensor::Mat;

/// SmolVLM2 pixel-shuffle (space-to-depth) over one frame's token grid
/// (`SmolVLMConnector.pixel_shuffle`, spec §3): fold each `s×s` block of
/// tokens into the channel dim.
///
/// `x` is `[seq, d]` row-major with `seq = g*g` a square token grid (SigLIP
/// row-major patch order); the result is `[seq/s², d*s²]` where output token
/// `(r1, c1)` concatenates the block's channels in `(dr, dc)` row-major order:
///
/// ```text
/// out[r1*(g/s) + c1][(dr*s + dc)*d + d0] = x[(r1*s + dr)*g + (c1*s + dc)][d0]
/// ```
///
/// which is exactly the reference `view/permute/reshape` chain flattened to a
/// gather (derived index-by-index from the §3 sequence; the torch cross-check
/// lives in `scripts/gen_reference_fixtures_smolvlm2_vision.py` and the parity
/// test below is bit-exact against its committed fixture).
///
/// # Errors
/// [`FocrError::Other`] if the grid is not square, not divisible by `s`, or
/// `s == 0`.
pub fn pixel_shuffle(x: &Mat, s: usize) -> FocrResult<Mat> {
    let (seq, d) = (x.rows, x.cols);
    if s == 0 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "pixel_shuffle: scale factor must be non-zero"
        )));
    }
    let g = (seq as f64).sqrt().round() as usize;
    if g * g != seq {
        return Err(FocrError::Other(anyhow::anyhow!(
            "pixel_shuffle: token count {seq} is not a square grid"
        )));
    }
    if g == 0 || !g.is_multiple_of(s) {
        return Err(FocrError::Other(anyhow::anyhow!(
            "pixel_shuffle: grid side {g} not divisible by scale {s}"
        )));
    }
    let go = g / s; // output grid side
    let out_rows = go * go;
    let out_cols = d * s * s;
    let mut out = vec![0.0f32; out_rows * out_cols];
    for r1 in 0..go {
        for c1 in 0..go {
            let dst_row = &mut out[(r1 * go + c1) * out_cols..(r1 * go + c1 + 1) * out_cols];
            for dr in 0..s {
                for dc in 0..s {
                    let src = ((r1 * s + dr) * g + (c1 * s + dc)) * d;
                    let dst = (dr * s + dc) * d;
                    dst_row[dst..dst + d].copy_from_slice(&x.data[src..src + d]);
                }
            }
        }
    }
    Ok(Mat::from_vec(out_rows, out_cols, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-derived 4×4 grid, d=1, s=2: token t holds value t. Output token
    /// (r1,c1) must hold the 2×2 block [(2r1)(2c1), (2r1)(2c1+1),
    /// (2r1+1)(2c1), (2r1+1)(2c1+1)] in row-major block order.
    #[test]
    fn pixel_shuffle_hand_case() {
        let x = Mat::from_vec(16, 1, (0..16).map(|v| v as f32).collect());
        let y = pixel_shuffle(&x, 2).unwrap();
        assert_eq!((y.rows, y.cols), (4, 4));
        assert_eq!(y.row(0), &[0.0, 1.0, 4.0, 5.0]);
        assert_eq!(y.row(1), &[2.0, 3.0, 6.0, 7.0]);
        assert_eq!(y.row(2), &[8.0, 9.0, 12.0, 13.0]);
        assert_eq!(y.row(3), &[10.0, 11.0, 14.0, 15.0]);
    }

    /// s=1 is the identity (shape and bytes).
    #[test]
    fn pixel_shuffle_scale_one_is_identity() {
        let x = Mat::from_vec(9, 3, (0..27).map(|v| v as f32 * 0.5).collect());
        let y = pixel_shuffle(&x, 1).unwrap();
        assert_eq!(y, x);
    }

    /// Channel blocks stay contiguous: d>1 keeps each source token's channel
    /// run intact at offset (dr*s+dc)*d.
    #[test]
    fn pixel_shuffle_keeps_channel_runs() {
        // 2×2 grid, d=3, s=2 → one output token holding all 4 tokens' channels.
        let x = Mat::from_vec(4, 3, (0..12).map(|v| v as f32).collect());
        let y = pixel_shuffle(&x, 2).unwrap();
        assert_eq!((y.rows, y.cols), (1, 12));
        // Token order (0,0),(0,1),(1,0),(1,1) == source rows 0,1,2,3.
        assert_eq!(
            y.row(0),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        );
    }

    #[test]
    fn pixel_shuffle_error_handling() {
        // Non-square token count.
        let x = Mat::from_vec(6, 2, vec![0.0; 12]);
        assert!(pixel_shuffle(&x, 2).is_err());
        // Grid not divisible by scale.
        let x = Mat::from_vec(9, 2, vec![0.0; 18]);
        assert!(pixel_shuffle(&x, 2).is_err());
        // Zero scale.
        let x = Mat::from_vec(4, 2, vec![0.0; 8]);
        assert!(pixel_shuffle(&x, 0).is_err());
    }

    // ── oracle parity (bit-exact vs the committed torch fixture) ────────────
    // Runtime-read with skip-with-SUCCESS when the fixture is absent (the C5
    // pattern) — the fixture is committed, so CI exercises this.

    fn load_vision_fixture() -> Option<serde_json::Value> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/smolvlm2/vision_oracle_fixtures.json"
        );
        let Ok(text) = std::fs::read_to_string(path) else {
            eprintln!(
                "skip-with-SUCCESS: {path} absent (gen_reference_fixtures_smolvlm2_vision.py)"
            );
            return None;
        };
        Some(serde_json::from_str(&text).expect("vision_oracle_fixtures.json parses"))
    }

    fn mat_from_json(v: &serde_json::Value) -> Mat {
        let rows: Vec<Vec<f32>> = v
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
                    .collect()
            })
            .collect();
        let (r, c) = (rows.len(), rows[0].len());
        Mat::from_vec(r, c, rows.into_iter().flatten().collect())
    }

    /// **A9/C4 L1 — pixel-shuffle bit-exact vs the torch reference** (small
    /// inline case: 8×8 grid, d=3, s=4, values transcription-cross-checked
    /// against `SmolVLMConnector.pixel_shuffle` at fixture-generation time).
    #[test]
    fn pixel_shuffle_matches_torch_small_case() {
        let Some(fx) = load_vision_fixture() else {
            return;
        };
        let small = &fx["l1_pixel_shuffle"]["small"];
        assert_eq!(small["s"].as_u64(), Some(4));
        let x = mat_from_json(&small["input"]);
        let want = mat_from_json(&small["output"]);
        let got = pixel_shuffle(&x, 4).unwrap();
        assert_eq!(
            (got.rows, got.cols),
            (want.rows, want.cols),
            "shape mismatch"
        );
        assert_eq!(got.data, want.data, "pixel_shuffle must be BIT-exact");
    }

    /// **A9/C4 L1 — the real SmolVLM2 shape** ([1024,768] s=4 → [64,12288]):
    /// rebuild the deterministic input from the fixture's `input_spec`
    /// formula, shuffle, and compare the sha256 of the f32-LE output bytes.
    #[test]
    fn pixel_shuffle_matches_torch_real_shape() {
        use sha2::{Digest, Sha256};
        let Some(fx) = load_vision_fixture() else {
            return;
        };
        let real = &fx["l1_pixel_shuffle"]["real_shape"];
        // input_spec: ((arange(1024*768) % 17) - 8) * 0.125 as f32 [1024,768]
        let data: Vec<f32> = (0..1024i64 * 768)
            .map(|i| ((i % 17) - 8) as f32 * 0.125)
            .collect();
        let x = Mat::from_vec(1024, 768, data);
        // Guard the rebuild against spec drift via the input sha.
        let mut h = Sha256::new();
        for v in &x.data {
            h.update(v.to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", h.finalize()),
            real["input_sha256_f32"].as_str().unwrap(),
            "rebuilt input drifted from the fixture's input_spec"
        );
        let y = pixel_shuffle(&x, 4).unwrap();
        assert_eq!((y.rows, y.cols), (64, 12288));
        let mut h = Sha256::new();
        for v in &y.data {
            h.update(v.to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", h.finalize()),
            real["output"]["sha256_f32"].as_str().unwrap(),
            "pixel_shuffle output bytes diverged from the torch reference"
        );
    }

    /// **C4 — connector parity on the REAL weights** (env-gated): oracle
    /// post-LN seam → our [`pixel_shuffle`] must be bit-exact vs the oracle's
    /// dump, and the `modality_projection` GEMM (`Linear(12288→960,
    /// bias=False)`, kept high-precision by C2) must match the oracle's
    /// connector output to vision-floor tolerance.
    #[test]
    fn smolvlm2_connector_matches_torch_oracle() {
        let Ok(dir) = std::env::var("FOCR_SMOLVLM2_DIR") else {
            return;
        };
        let Some(fx) = load_vision_fixture() else {
            return;
        };
        let post_ln_path = format!("{dir}/smolvlm2_vision_post_ln.bin");
        let ps_path = format!("{dir}/smolvlm2_pixel_shuffle_out.bin");
        let conn_path = format!("{dir}/smolvlm2_connector_out.bin");
        let model_path = format!("{dir}/model.safetensors");
        if !std::path::Path::new(&post_ln_path).is_file() {
            eprintln!("skip-with-SUCCESS: {post_ln_path} absent");
            return;
        }
        let read_f32 = |p: &str| -> Vec<f32> {
            let bytes = std::fs::read(p).expect("oracle blob reads");
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let post_ln = read_f32(&post_ln_path);
        let n_frames = post_ln.len() / (1024 * 768);
        assert_eq!(
            n_frames * 1024 * 768,
            post_ln.len(),
            "post_ln not [F,1024,768]"
        );
        let ps_want = read_f32(&ps_path);
        let conn_want = read_f32(&conn_path);
        assert_eq!(ps_want.len(), n_frames * 64 * 12288);
        assert_eq!(conn_want.len(), n_frames * 64 * 960);

        // Pixel-shuffle per frame: BIT-exact (pure data movement).
        let mut ps_ours = Vec::with_capacity(ps_want.len());
        for f in 0..n_frames {
            let x = Mat::from_vec(
                1024,
                768,
                post_ln[f * 1024 * 768..(f + 1) * 1024 * 768].to_vec(),
            );
            ps_ours.extend_from_slice(&pixel_shuffle(&x, 4).unwrap().data);
        }
        assert_eq!(
            ps_ours, ps_want,
            "pixel_shuffle on the real post-LN seam must be BIT-exact"
        );

        // Connector GEMM: [F*64, 12288] @ proj^T [12288, 960], no bias.
        let weights = super::super::weights::Weights::load(std::path::Path::new(&model_path))
            .expect("smolvlm2 safetensors loads");
        let proj = weights
            .mat("model.connector.modality_projection.proj.weight")
            .expect("connector proj tensor");
        assert_eq!((proj.rows, proj.cols), (960, 12288));
        let lin =
            super::super::vision_sam::Linear::from_row_major(&proj.data, Vec::new(), 960, 12288)
                .expect("connector linear");
        let x = Mat::from_vec(n_frames * 64, 12288, ps_ours);
        let ours = lin.apply(&x).expect("connector GEMM");
        let floor = fx["nondeterminism_floor"]["vision_maxabs_cross_thread"]
            .as_f64()
            .unwrap();
        // MEASURED budget: the oracle floor is 0.0 (torch vision is
        // thread-deterministic here), so the drift is entirely our GEMM's
        // f32 summation order vs torch's over K=12288 — measured 2.59e-4
        // maxabs on values of O(1..10) (~1e-5 relative) at cos 1.00000000
        // (2026-07-02, 13 frames). Budget = 4x the measured drift; the floor
        // term keeps the gate honest if the fixtures are ever regenerated on
        // a nondeterministic stack.
        let tol = (floor * 16.0).max(1.1e-3);
        let mut max_abs = 0.0f64;
        let mut dot = 0.0f64;
        let (mut na, mut nb) = (0.0f64, 0.0f64);
        for (a, b) in ours.data.iter().zip(&conn_want) {
            let (a, b) = (f64::from(*a), f64::from(*b));
            max_abs = max_abs.max((a - b).abs());
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        eprintln!("[C4 parity] connector maxabs={max_abs:.3e} cos={cos:.8} tol={tol:.3e}");
        assert!(cos >= 0.9999, "connector cosine {cos} < 0.9999");
        assert!(
            max_abs <= tol,
            "connector maxabs {max_abs:.3e} > tol {tol:.3e} (floor {floor:.3e})"
        );
    }
}
