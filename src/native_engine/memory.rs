//! What a run holds in memory (U2-D3, plan §6.3): the process's real peak, read
//! from the kernel; an estimate of that peak before a document starts; and the
//! largest page window a memory budget allows.
//!
//! The weights a model hydrates are counted from its tensor directory. The
//! scratch of a forward is not in any table, so it is measured
//! (`FOCR_TIMING`'s `memory:` rows) and named here. It is the larger part:
//! Unlimited-OCR-2's KV peaks near 128 MB on 3 pages with block retirement,
//! where SAM's forward alone adds 2 GB. A page window therefore moves the
//! estimate far less than the weights do.

use super::decoder::config::NUM_HIDDEN_LAYERS;
use super::srswa::SrSwaLayerCache;
use super::timing_log;
use super::weights::Weights;
use super::window::{PER_PAGE_EST_BASE, TAIL_TOKENS};
use crate::error::{FocrError, FocrResult};

/// What overrides the default memory budget: bytes, or a whole number of
/// `KiB`, `MiB`, `GiB` or `TiB`.
pub const BUDGET_ENV: &str = "FOCR_MEMORY_BUDGET";

/// Output tokens a page is assumed to reach, until U2-F1 measures the p90
/// of the distribution.
pub const DEFAULT_TOKENS_PER_PAGE: usize = 1000;

/// The binary, runtime and tokenizer, before any weight loads (82 MB measured).
const PROCESS_BYTES: u64 = 90_000_000;

/// The high-water scratch of a page's vision forward: SAM's forward (1.99 GB
/// measured) and what hydrating CLIP holds beyond its finished weights
/// (0.6 GB, the rest of its 1.81 GB row).
const VISION_SCRATCH_BYTES: u64 = 2_600_000_000;

/// The high-water scratch of a decode, past its KV (0.58 GB measured).
const DECODE_SCRATCH_BYTES: u64 = 600_000_000;

/// A page raster in flight: a 300 dpi letter page (plan §6.3).
const RASTER_BYTES: u64 = 25_000_000;

/// A preprocessed page: 1024 × 1024 × 3 f32.
const PREPROCESSED_BYTES: u64 = 1024 * 1024 * 3 * 4;

/// The share of physical memory the default budget allows, in quarters.
///
/// Half would refuse every run on a 16 GB machine, the plan's target host,
/// whose weights and scratch alone need about 10 GB. Three quarters leaves that
/// machine 4 GB for everything else.
const DEFAULT_BUDGET_QUARTERS: u64 = 3;

const GIB: f64 = (1u64 << 30) as f64;

/// What a model's weights take in memory, read from its tensor directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelMemory {
    /// The artifact's bytes.
    pub artifact: u64,
    /// The artifact is mapped, so the kernel can take its pages back.
    pub mapped: bool,
    /// SAM, CLIP and the projector, widened to f32.
    pub vision_caches: u64,
    /// The decoder's attention, norms and `lm_head`, widened to f32.
    pub widened_weights: u64,
    /// The router, widened and held twice: as stored and transposed.
    pub router: u64,
}

impl ModelMemory {
    /// `weights`' memory, for an artifact of `artifact` bytes.
    #[must_use]
    pub fn of(weights: &Weights, artifact: u64) -> Self {
        let mut memory = Self {
            artifact,
            mapped: weights.is_mapped(),
            vision_caches: 0,
            widened_weights: 0,
            router: 0,
        };
        for name in weights.names() {
            let Some(record) = weights.record(name) else {
                continue;
            };
            let elements = record.shape.iter().product::<usize>();
            let widened = u64::try_from(elements)
                .unwrap_or(u64::MAX)
                .saturating_mul(4);
            if [
                "model.sam_model.",
                "model.vision_model.",
                "model.projector.",
            ]
            .iter()
            .any(|prefix| name.starts_with(prefix))
            {
                memory.vision_caches += widened;
            } else if name.contains(".dash_router.") || name.contains(".indexer.") {
                memory.router += widened * 2;
            } else if name == "lm_head.weight"
                || name == "model.norm.weight"
                || (name.starts_with("model.layers.") && !name.contains(".mlp."))
            {
                memory.widened_weights += widened;
            }
        }
        memory
    }
}

/// What an [`Estimate`] takes as given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EstimateAssumptions {
    /// Own pages per window.
    pub page_window: usize,
    /// Output tokens per page.
    pub tokens_per_page: usize,
    /// KV blocks retire as pages pass.
    pub retirement: bool,
}

/// An [`Estimate`]'s parts, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EstimateBreakdown {
    /// The binary, runtime and tokenizer.
    pub process: u64,
    /// The artifact.
    pub artifact: u64,
    /// SAM, CLIP and the projector, widened.
    pub vision_caches: u64,
    /// A vision forward's scratch.
    pub vision_scratch: u64,
    /// The decoder's widened weights.
    pub widened_weights: u64,
    /// The router.
    pub router: u64,
    /// A decode's scratch.
    pub decode_scratch: u64,
    /// The rasters and preprocessed pages of a window in flight.
    pub rasters: u64,
    /// KV at its peak.
    pub kv: u64,
}

/// A run's estimated peak resident set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Estimate {
    /// The peak, in bytes.
    pub total: u64,
    /// The part the kernel can take back under pressure: a mapped artifact.
    pub reclaimable: u64,
    /// Its parts.
    pub breakdown: EstimateBreakdown,
    /// What it took as given.
    pub assumptions: EstimateAssumptions,
}

impl Estimate {
    /// What the machine must really find: the peak less what it can reclaim.
    #[must_use]
    pub fn committed(&self) -> u64 {
        self.total - self.reclaimable
    }
}

/// The peak resident set of a windowed document read under `assumptions`.
///
/// The parts add because their peaks stack: a forward's freed scratch stays
/// resident, so measured peaks grow by each part in turn.
#[must_use]
pub fn estimate_peak_bytes(model: &ModelMemory, assumptions: EstimateAssumptions) -> Estimate {
    let EstimateAssumptions {
        page_window,
        tokens_per_page,
        retirement,
    } = assumptions;
    // A window's own pages and its lookahead.
    let window_pages = page_window + 1;
    let image_rows = window_pages * PER_PAGE_EST_BASE;
    // Retirement frees the rows of pages the decode has passed, so only the
    // images, one page's text and the copied tail are resident together.
    let rows = if retirement {
        image_rows + tokens_per_page + TAIL_TOKENS
    } else {
        image_rows + page_window * tokens_per_page
    };
    let kv = NUM_HIDDEN_LAYERS * SrSwaLayerCache::estimate_bytes(rows);
    let breakdown = EstimateBreakdown {
        process: PROCESS_BYTES,
        artifact: model.artifact,
        vision_caches: model.vision_caches,
        vision_scratch: VISION_SCRATCH_BYTES,
        widened_weights: model.widened_weights,
        router: model.router,
        decode_scratch: DECODE_SCRATCH_BYTES,
        rasters: u64::try_from(window_pages)
            .unwrap_or(u64::MAX)
            .saturating_mul(RASTER_BYTES + PREPROCESSED_BYTES),
        kv: u64::try_from(kv).unwrap_or(u64::MAX),
    };
    let EstimateBreakdown {
        process,
        artifact,
        vision_caches,
        vision_scratch,
        widened_weights,
        router,
        decode_scratch,
        rasters,
        kv,
    } = breakdown;
    let total = [
        process,
        artifact,
        vision_caches,
        vision_scratch,
        widened_weights,
        router,
        decode_scratch,
        rasters,
        kv,
    ]
    .iter()
    .fold(0, |sum: u64, part| sum.saturating_add(*part));
    Estimate {
        total,
        reclaimable: if model.mapped { model.artifact } else { 0 },
        breakdown,
        assumptions,
    }
}

/// The largest window up to `most` whose estimate fits `budget`, or the
/// smallest window's estimate when none does.
///
/// # Errors
/// The one-page [`Estimate`], for the refusal to explain.
pub fn fitting_window(
    most: usize,
    budget: u64,
    estimate_at: impl Fn(usize) -> Estimate,
) -> Result<usize, Estimate> {
    (1..=most)
        .rev()
        .find(|window| estimate_at(*window).committed() <= budget)
        .ok_or_else(|| estimate_at(1))
}

/// Why a machine cannot take a document at any window, parts included.
#[must_use]
pub fn refusal(estimate: &Estimate, budget: u64) -> FocrError {
    let gib = |bytes: u64| bytes as f64 / GIB;
    let part = estimate.breakdown;
    let scratch = part.vision_scratch + part.decode_scratch;
    let weights = part.vision_caches + part.widened_weights + part.router;
    FocrError::Usage(format!(
        "this document needs about {:.1} GiB at one page per window, past the {:.1} GiB memory \
         budget (artifact {:.1} GiB, hydrated weights {:.1} GiB, forward scratch {:.1} GiB, \
         KV and pages {:.1} GiB, of which {:.1} GiB can be reclaimed); set {BUDGET_ENV} to \
         allow more, or FOCR_MMAP=1 to map the artifact",
        gib(estimate.total),
        gib(budget),
        gib(part.artifact),
        gib(weights),
        gib(scratch),
        gib(part.kv + part.rasters),
        gib(estimate.reclaimable),
    ))
}

/// The memory budget: [`BUDGET_ENV`], or a share of physical memory.
///
/// # Errors
/// [`FocrError::Usage`] for a value that does not parse, or when there is no
/// value and the platform does not report its memory.
pub fn memory_budget() -> FocrResult<u64> {
    match std::env::var(BUDGET_ENV) {
        Ok(value) => parse_budget(&value),
        Err(_) => physical_memory_bytes()
            .map(|bytes| bytes / 4 * DEFAULT_BUDGET_QUARTERS)
            .ok_or_else(|| {
                FocrError::Usage(format!(
                    "this platform does not report its physical memory; set {BUDGET_ENV}"
                ))
            }),
    }
}

/// A [`BUDGET_ENV`] value in bytes.
fn parse_budget(value: &str) -> FocrResult<u64> {
    let trimmed = value.trim();
    let (digits, unit) = [
        ("KiB", 1u64 << 10),
        ("MiB", 1 << 20),
        ("GiB", 1 << 30),
        ("TiB", 1 << 40),
    ]
    .iter()
    .find_map(|(suffix, unit)| Some((trimmed.strip_suffix(suffix)?.trim_end(), *unit)))
    .unwrap_or((trimmed, 1));
    digits
        .parse::<u64>()
        .ok()
        .filter(|count| *count > 0)
        .and_then(|count| count.checked_mul(unit))
        .ok_or_else(|| {
            FocrError::Usage(format!(
                "{BUDGET_ENV}={value:?} is not a memory size; give bytes, or a whole number of \
                 KiB, MiB, GiB or TiB, such as 12GiB"
            ))
        })
}

/// The machine's physical memory, where a safe read of it exists.
fn physical_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        String::from_utf8(output.stdout).ok()?.trim().parse().ok()
    }
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        meminfo.lines().find_map(|line| {
            let kib = line.strip_prefix("MemTotal:")?.trim();
            kib.strip_suffix(" kB")?
                .trim()
                .parse::<u64>()
                .ok()?
                .checked_mul(1024)
        })
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// The process's peak resident set so far, in bytes.
///
/// The kernel keeps this as a high-water mark, so it never falls: a window's
/// reading is the peak of the run up to that window's end, not of that window
/// alone. `None` where the platform has no safe way to read it.
#[must_use]
pub fn peak_rss_bytes() -> Option<u64> {
    #[cfg(all(unix, feature = "native"))]
    {
        use nix::sys::resource::{UsageWho, getrusage};
        let peak = u64::try_from(getrusage(UsageWho::RUSAGE_SELF).ok()?.max_rss()).ok()?;
        // Bytes on Apple platforms, KiB everywhere else.
        Some(if cfg!(target_vendor = "apple") {
            peak
        } else {
            peak.saturating_mul(1024)
        })
    }
    #[cfg(not(all(unix, feature = "native")))]
    {
        None
    }
}

/// A `FOCR_TIMING` row with the peak so far, at the end of `stage`.
pub(super) fn log_peak_rss(stage: &str) {
    if let Some(peak) = peak_rss_bytes() {
        timing_log(&format!("memory: {stage}: peak RSS {peak} bytes"));
    }
}

/// The run summary's `FOCR_TIMING` row: `estimate` against the real peak.
pub(super) fn log_against_estimate(estimate: &Estimate) {
    if let Some(peak) = peak_rss_bytes() {
        timing_log(&format!(
            "memory: estimate {} bytes at window {}, {} tokens per page; peak RSS {peak} bytes, \
             {:.3} of the estimate",
            estimate.total,
            estimate.assumptions.page_window,
            estimate.assumptions.tokens_per_page,
            peak as f64 / estimate.total as f64,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: ModelMemory = ModelMemory {
        artifact: 4_173_000_000,
        mapped: false,
        vision_caches: 1_606_000_000,
        widened_weights: 978_000_000,
        router: 64_000_000,
    };

    fn assumptions(page_window: usize, retirement: bool) -> EstimateAssumptions {
        EstimateAssumptions {
            page_window,
            tokens_per_page: DEFAULT_TOKENS_PER_PAGE,
            retirement,
        }
    }

    /// The plan's KV arithmetic by hand: 12 layers of (2 × 1280 + 128) × 4 B
    /// per row, 126 KiB, in blocks of 32 rows, each with a small header.
    #[test]
    fn kv_is_the_plans_row_arithmetic() {
        const ROW: u64 = 12 * (2 * 1280 + 128) * 4;
        assert_eq!(ROW, 129_024);
        let kv = |window, retirement| {
            estimate_peak_bytes(&MODEL, assumptions(window, retirement))
                .breakdown
                .kv
        };

        // W=10 without retirement: 11 × 274 prompt rows and 10,000 of output,
        // the plan's "~1.5 GiB" row. Whole blocks round it up a little.
        let rows = 11 * 274 + 10_000;
        let bytes = kv(10, false);
        assert!(bytes >= rows * ROW, "{bytes}");
        assert!(bytes < (rows + 64) * (ROW + 1024), "{bytes}");

        // With retirement: every image, one page of text and the copied tail,
        // the plan's "≤ ~0.35 GiB" row with the page it left out.
        let rows = 11 * 274 + 1000 + 128;
        let bytes = kv(10, true);
        assert!(bytes >= rows * ROW, "{bytes}");
        assert!(bytes < (rows + 64) * (ROW + 1024), "{bytes}");
    }

    #[test]
    fn the_total_is_the_sum_of_its_parts() {
        let estimate = estimate_peak_bytes(&MODEL, assumptions(10, true));
        let part = estimate.breakdown;
        assert_eq!(
            estimate.total,
            part.process
                + part.artifact
                + part.vision_caches
                + part.vision_scratch
                + part.widened_weights
                + part.router
                + part.decode_scratch
                + part.rasters
                + part.kv
        );
        assert_eq!(part.rasters, 11 * (25_000_000 + 12_582_912));
        assert_eq!(estimate.committed(), estimate.total, "owned bytes stay");
    }

    #[test]
    fn a_mapped_artifact_counts_toward_the_peak_but_not_against_the_budget() {
        let owned = estimate_peak_bytes(&MODEL, assumptions(10, true));
        let mapped = estimate_peak_bytes(
            &ModelMemory {
                mapped: true,
                ..MODEL
            },
            assumptions(10, true),
        );
        assert_eq!(mapped.total, owned.total);
        assert_eq!(mapped.committed(), owned.total - MODEL.artifact);
    }

    #[test]
    fn the_estimate_grows_with_the_window_and_the_text() {
        let total = |window, tokens_per_page, retirement| {
            estimate_peak_bytes(
                &MODEL,
                EstimateAssumptions {
                    page_window: window,
                    tokens_per_page,
                    retirement,
                },
            )
            .total
        };
        for retirement in [true, false] {
            for window in 1..10 {
                assert!(total(window, 1000, retirement) < total(window + 1, 1000, retirement));
                assert!(total(window, 1000, retirement) < total(window, 2000, retirement));
            }
        }
        assert!(total(10, 1000, true) < total(10, 1000, false));
    }

    #[test]
    fn auto_takes_the_largest_window_that_fits_and_grows_with_the_budget() {
        let at = |window| estimate_peak_bytes(&MODEL, assumptions(window, false));
        let need = |window| at(window).committed();
        assert_eq!(fitting_window(10, need(10), at), Ok(10));
        assert_eq!(fitting_window(10, u64::MAX, at), Ok(10));
        assert_eq!(fitting_window(10, need(7), at), Ok(7));
        assert_eq!(fitting_window(10, need(7) - 1, at), Ok(6));
        assert_eq!(fitting_window(10, need(1), at), Ok(1));
        assert_eq!(fitting_window(4, u64::MAX, at), Ok(4), "the cap holds");

        let mut last = 0;
        for budget in (need(1)..=need(10)).step_by(50_000_000) {
            let window = fitting_window(10, budget, at).expect("it fits");
            assert!(window >= last, "monotone in the budget");
            last = window;
        }
    }

    #[test]
    fn below_one_page_auto_refuses_with_the_parts_and_the_ways_out() {
        let at = |window| estimate_peak_bytes(&MODEL, assumptions(window, true));
        let budget = at(1).committed() - 1;
        let estimate = fitting_window(10, budget, at).expect_err("nothing fits");
        assert_eq!(estimate.assumptions.page_window, 1);

        let error = refusal(&estimate, budget);
        assert!(matches!(error, FocrError::Usage(_)), "exit 2: {error:?}");
        let text = error.to_string();
        for says in ["memory budget", "artifact", "hydrated weights", "scratch"] {
            assert!(text.contains(says), "{says}: {text}");
        }
        assert!(text.contains(BUDGET_ENV), "{text}");
        assert!(text.contains("FOCR_MMAP=1"), "{text}");
    }

    #[test]
    fn budgets_parse_in_bytes_and_binary_units_and_nothing_else() {
        assert_eq!(parse_budget("1048576").expect("bytes"), 1 << 20);
        assert_eq!(parse_budget("12GiB").expect("GiB"), 12 << 30);
        assert_eq!(parse_budget(" 512 MiB ").expect("MiB"), 512 << 20);
        assert_eq!(parse_budget("2TiB").expect("TiB"), 2 << 40);
        assert_eq!(parse_budget("64KiB").expect("KiB"), 64 << 10);
        for bad in [
            "",
            "0",
            "0GiB",
            "12GB",
            "12G",
            "twelve",
            "-1",
            "1.5GiB",
            "99999999999TiB",
        ] {
            let error = parse_budget(bad).expect_err(bad);
            assert!(matches!(error, FocrError::Usage(_)), "{bad}: {error:?}");
            assert!(error.to_string().contains(BUDGET_ENV), "{error}");
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn physical_memory_is_read_on_this_platform() {
        let bytes = physical_memory_bytes().expect("a reading");
        assert!((1u64 << 30..1 << 50).contains(&bytes), "{bytes}");
    }

    #[cfg(all(unix, feature = "native"))]
    #[test]
    fn peak_rss_is_a_plausible_high_water_mark() {
        const MIB: u64 = 1 << 20;
        let before = peak_rss_bytes().expect("a reading on this platform");
        // A running test binary is at least a megabyte, and well under 1 TiB:
        // a wrong unit would miss one of these by three orders of magnitude.
        assert!((MIB..1 << 40).contains(&before), "{before}");

        // Touch 64 MiB more than the process has ever held.
        let size = usize::try_from(before + 64 * MIB).expect("fits");
        let mut held = vec![0u8; size];
        for page in held.chunks_mut(4096) {
            page[0] = 1;
        }
        let after = peak_rss_bytes().expect("a reading on this platform");
        assert!(after >= before, "a high-water mark never falls");
        assert!(after >= u64::try_from(held.len()).expect("fits"), "{after}");
        std::hint::black_box(&held);
    }
}
