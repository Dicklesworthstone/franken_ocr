//! Scratch wasm micro-benchmark harness for franken_ocr's int8/int4 SIMD core.
//!
//! Raw wasm exports (no wasm-bindgen) so Node can instantiate the module with
//! an empty import object and time the kernels with `performance.now()` —
//! wasm has no clock, so timing is the caller's job.
//!
//! Every bench returns a checksum of the output so the optimizer cannot delete
//! the loop, and fills operands from a deterministic xorshift over the FULL
//! byte range (a sparse matrix would flatter the kernel).

use franken_ocr::simd;
use ft_core::{DType, Device, TensorMeta};

fn xs32(state: &mut u32) -> u32 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    *state = s;
    s
}

fn fill_i8(len: usize, seed: u32) -> Vec<i8> {
    let mut s = seed | 1;
    (0..len).map(|_| (xs32(&mut s) & 0xff) as u8 as i8).collect()
}

fn fill_u8(len: usize, seed: u32) -> Vec<u8> {
    let mut s = seed | 1;
    (0..len).map(|_| (xs32(&mut s) & 0xff) as u8).collect()
}

fn fill_f32(len: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| ((xs32(&mut s) & 0xffff) as f32 / 65536.0) * 0.02 + 0.001)
        .collect()
}

fn checksum_i32(v: &[i32]) -> f64 {
    let mut acc: i64 = 0;
    for (i, x) in v.iter().enumerate() {
        acc = acc.wrapping_add((*x as i64).wrapping_mul((i as i64 & 7) + 1));
    }
    acc as f64
}

fn checksum_f32(v: &[f32]) -> f64 {
    let mut acc = 0f64;
    for x in v {
        acc += *x as f64;
    }
    acc
}

/// mode 0 = scalar oracle, 1 = dispatched entrypoint.
#[no_mangle]
pub extern "C" fn bench_s8s8(mode: u32, m: usize, k: usize, n: usize, rounds: u32) -> f64 {
    let a = fill_i8(m * k, 0x1234_5678);
    let b = fill_i8(n * k, 0x9abc_def0);
    let mut out = vec![0i32; m * n];
    let mut sum = 0f64;
    for _ in 0..rounds {
        out.iter_mut().for_each(|x| *x = 0);
        match mode {
            0 => simd::scalar::igemm_s8s8(&a, &b, m, k, n, &mut out),
            _ => simd::igemm_s8s8(&a, &b, m, k, n, &mut out),
        }
        sum += checksum_i32(&out);
    }
    sum
}

#[no_mangle]
pub extern "C" fn bench_u8s8(mode: u32, m: usize, k: usize, n: usize, rounds: u32) -> f64 {
    let a = fill_u8(m * k, 0x0f0f_1234);
    let b = fill_i8(n * k, 0xfeed_face);
    let mut out = vec![0i32; m * n];
    let mut sum = 0f64;
    for _ in 0..rounds {
        out.iter_mut().for_each(|x| *x = 0);
        match mode {
            0 => simd::scalar::igemm_u8s8(&a, &b, m, k, n, &mut out),
            _ => simd::igemm_u8s8(&a, &b, m, k, n, &mut out),
        }
        sum += checksum_i32(&out);
    }
    sum
}

#[no_mangle]
pub extern "C" fn bench_s4s8(
    mode: u32,
    m: usize,
    k: usize,
    n: usize,
    group: usize,
    rounds: u32,
) -> f64 {
    let a = fill_i8(m * k, 0x2244_6688);
    let packed = fill_u8(n * (k / 2), 0x1357_9bdf);
    let scales = fill_f32(n * (k / group), 0x0246_8ace);
    let mut out = vec![0f32; m * n];
    let mut sum = 0f64;
    for _ in 0..rounds {
        match mode {
            0 => simd::int4::igemm_s4s8_packed_scalar(
                &a, &packed, &scales, group, m, k, n, &mut out,
            ),
            _ => simd::int4::igemm_s4s8_packed(&a, &packed, &scales, group, m, k, n, &mut out),
        }
        sum += checksum_f32(&out);
    }
    sum
}

/// Bit-identity proof, on the BUILT wasm artifact: dispatched int8 kernel vs
/// the scalar oracle across the shipped shape battery. Returns the number of
/// FAILING cases (0 = all bit-identical).
#[no_mangle]
pub extern "C" fn selftest_failures() -> u32 {
    let r = simd::selftest();
    r.cases.iter().filter(|c| !c.ok).count() as u32
}

#[no_mangle]
pub extern "C" fn selftest_cases() -> u32 {
    simd::selftest().cases.len() as u32
}

/// int4 packed dispatch vs its scalar oracle, bit-identical f32 check, over a
/// shape battery incl. both group sizes, k tails and the K=6848 dense shape.
/// Returns the number of failing (shape, lane) cases.
#[no_mangle]
pub extern "C" fn int4_parity_failures() -> u32 {
    const SHAPES: &[(usize, usize, usize, usize)] = &[
        // (m, k, n, group)
        (1, 16, 1, 16),
        (1, 32, 3, 16),
        (1, 32, 3, 32),
        (1, 64, 7, 16),
        (1, 96, 5, 32),
        (1, 1280, 896, 16),
        (1, 896, 1280, 16),
        (1, 1280, 896, 32),
        (1, 6848, 64, 16),
        (4, 1280, 64, 16),
        (3, 1280, 33, 32),
    ];
    let mut bad = 0u32;
    for (idx, &(m, k, n, group)) in SHAPES.iter().enumerate() {
        let seed = 0x5eed_0000 ^ (idx as u32 * 0x9e37_79b9);
        let a = fill_i8(m * k, seed);
        let packed = fill_u8(n * (k / 2), seed ^ 0xaaaa_5555);
        let scales = fill_f32(n * (k / group), seed ^ 0x1234_abcd);
        let mut got = vec![0f32; m * n];
        let mut want = vec![0f32; m * n];
        simd::int4::igemm_s4s8_packed(&a, &packed, &scales, group, m, k, n, &mut got);
        simd::int4::igemm_s4s8_packed_scalar(&a, &packed, &scales, group, m, k, n, &mut want);
        for (g, w) in got.iter().zip(want.iter()) {
            if g.to_bits() != w.to_bits() {
                bad += 1;
            }
        }
    }
    bad
}

/// Route tag length + bytes are awkward over the raw ABI; return a small id.
/// 0 = scalar, 1 = wasm-simd128, 9 = other.
#[no_mangle]
pub extern "C" fn route_id() -> u32 {
    match simd::tier_string() {
        "scalar" => 0,
        "wasm32+simd128" => 1,
        _ => 9,
    }
}

/// Pure streaming-read bandwidth probe: how fast can this engine simply pull
/// `bytes` of linear memory through a v128 accumulator? Establishes whether a
/// GEMV that plateaus is bandwidth-bound or issue-bound.
#[no_mangle]
pub extern "C" fn bench_stream(bytes: usize, rounds: u32) -> f64 {
    let buf = fill_i8(bytes, 0x7777_7777);
    let mut total = 0i64;
    for _ in 0..rounds {
        let mut acc: i64 = 0;
        // Chunked i8->i32 sum; LLVM autovectorizes this trivially on +simd128.
        for c in buf.chunks_exact(64) {
            let mut s = 0i32;
            for x in c {
                s += *x as i32;
            }
            acc += s as i64;
        }
        total = total.wrapping_add(acc);
    }
    total as f64
}

// ── f32 GEMM probe (LEVER 1 precondition) ───────────────────────────────────
// Does the wasm build's f32 matmul run matrixmultiply's simd128 microkernel, or
// its generic fallback? Build this same crate with and without +simd128 and
// compare: the answer is a ratio, not a claim.

/// `C[m,n] = A[m,k] @ B[k,n]` through ft-kernel-cpu's public sgemm chokepoint.
#[no_mangle]
pub extern "C" fn bench_sgemm(m: usize, k: usize, n: usize, rounds: u32) -> f64 {
    let a = fill_f32(m * k, 0x2468_ace0);
    let b = fill_f32(k * n, 0x1357_bdf9);
    let mut c = vec![0f32; m * n];
    let am = TensorMeta::from_shape(vec![m, k], DType::F32, Device::Cpu);
    let bm = TensorMeta::from_shape(vec![k, n], DType::F32, Device::Cpu);
    let mut sum = 0f64;
    for _ in 0..rounds {
        ft_kernel_cpu::matmul_tensor_contiguous_f32_into(&mut c, &a, &b, &am, &bm).unwrap();
        sum += checksum_f32(&c);
    }
    sum
}

/// Strictly k-sequential scalar reference for the same product — the bit-exact
/// oracle a hand kernel would have to match.
#[no_mangle]
pub extern "C" fn bench_sgemm_naive(m: usize, k: usize, n: usize, rounds: u32) -> f64 {
    let a = fill_f32(m * k, 0x2468_ace0);
    let b = fill_f32(k * n, 0x1357_bdf9);
    let mut c = vec![0f32; m * n];
    let mut sum = 0f64;
    for _ in 0..rounds {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        sum += checksum_f32(&c);
    }
    sum
}

// ── LEVER 3 probe: does LLVM autovectorize the f32 glue under +simd128? ─────
// Doctrine #3 says never hand-SIMD what LLVM already vectorizes. The way to
// know is to build this same crate twice (+simd128 / no simd128) and compare:
// a ~1.0x ratio means scalar code either way (an opening); 2-4x means LLVM
// already did the work and a hand kernel would be the founding prior's mistake.

fn fill_bytes(len: usize, seed: u32) -> Vec<u8> {
    let mut s = seed | 1;
    (0..len).map(|_| (xs32(&mut s) & 0xff) as u8).collect()
}

/// bf16 -> f32 widening, byte-for-byte the shape of `weights::decode_f32`'s
/// BF16 arm (LEVER 2's inner loop).
#[no_mangle]
pub extern "C" fn bench_bf16_widen(elems: usize, rounds: u32) -> f64 {
    let data = fill_bytes(elems * 2, 0x5a5a_1234);
    let mut sum = 0f64;
    for _ in 0..rounds {
        let out: Vec<f32> = data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::bf16::from_le_bytes(*c).to_f32())
            .collect();
        sum += out[0] as f64 + out[out.len() - 1] as f64;
    }
    sum
}

#[no_mangle]
pub extern "C" fn bench_layer_norm(rows: usize, cols: usize, rounds: u32) -> f64 {
    let x = fill_f32(rows * cols, 0x1111_2222);
    let w = fill_f32(cols, 0x3333_4444);
    let b = fill_f32(cols, 0x5555_6666);
    let mut sum = 0f64;
    for _ in 0..rounds {
        let y = ft_kernel_cpu::layer_norm_forward_f32(&x, Some(&w), Some(&b), rows, cols, 1e-5);
        sum += checksum_f32(&y);
    }
    sum
}

#[no_mangle]
pub extern "C" fn bench_softmax(rows: usize, cols: usize, rounds: u32) -> f64 {
    let x = fill_f32(rows * cols, 0x7777_8888);
    let meta = TensorMeta::from_shape(vec![rows, cols], DType::F32, Device::Cpu);
    let mut sum = 0f64;
    for _ in 0..rounds {
        let y = ft_kernel_cpu::softmax_dim_tensor_contiguous_f32(&x, &meta, 1).unwrap();
        sum += checksum_f32(&y);
    }
    sum
}

#[no_mangle]
pub extern "C" fn bench_sdpa(nbh: usize, sq: usize, sk: usize, d: usize, rounds: u32) -> f64 {
    let q = fill_f32(nbh * sq * d, 0x9999_aaaa);
    let k = fill_f32(nbh * sk * d, 0xbbbb_cccc);
    let v = fill_f32(nbh * sk * d, 0xdddd_eeee);
    let scale = 1.0 / (d as f32).sqrt();
    let mut sum = 0f64;
    for _ in 0..rounds {
        let y =
            ft_kernel_cpu::sdpa_forward_f32(&q, &k, &v, nbh, sq, sk, d, d, scale, false);
        sum += checksum_f32(&y);
    }
    sum
}
