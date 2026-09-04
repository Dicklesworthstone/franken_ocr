//! focr-ios — the C ABI the FrankenOCR iPhone/iPad app links against.
//!
//! Design rules, inherited from `focr-wasm` (the other boundary crate) and from
//! the frankentts `ftts-ffi` precedent:
//!
//! * **Errors are values, never panics.** Every fallible edge returns a status
//!   or NULL and leaves a human-readable reason in the calling thread's error
//!   slot. The release profile is `panic = "abort"`, so an escaping panic would
//!   kill the app; every entry point catches unwinds anyway so debug builds
//!   stay sound.
//! * **This crate adds zero numerics.** It decodes an image, calls the same
//!   `OcrModel` entry points the CLI and the browser call, and serializes the
//!   result. There is exactly one inference implementation.
//! * **No environment.** Everything the library would read from `FOCR_*` on a
//!   desktop is a setter here, because an app has no environment to set.
//! * **The engine loads from a PATH, not from bytes.** That is the difference
//!   that makes a 3 GB artifact viable on a phone: `Weights::load` reaches the
//!   mmap island, so the blob is clean file-backed pages instead of the dirty
//!   anonymous heap the browser is forced into.
//!
//! Ownership and threading are specified in `include/focr_ios.h`, which is the
//! document the Swift side actually reads. Keep the two in lockstep.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use franken_ocr::error::{FocrError, FocrResult};
use franken_ocr::native_engine::{OcrModel, RecognizedDocument};

// ── Error slot ─────────────────────────────────────────────────────────────

thread_local! {
    /// The calling thread's last error. Thread-local so two threads reporting
    /// failures cannot overwrite each other's message, and so no lock is taken
    /// on the error path.
    static LAST_ERROR: RefCell<CString> =
        RefCell::new(CString::new("").expect("empty string has no interior NUL"));
}

fn set_error(message: impl Into<String>) {
    let message = message.into();
    // A NUL inside the message would truncate it silently at the boundary;
    // replace rather than drop the diagnostic entirely.
    let sanitized = message.replace('\0', "?");
    let value = CString::new(sanitized)
        .unwrap_or_else(|_| CString::new("error message contained a NUL").expect("literal"));
    LAST_ERROR.with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            *slot = value;
        }
    });
}

fn clear_error() {
    LAST_ERROR.with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            *slot = CString::new("").expect("literal");
        }
    });
}

/// Record `err` and return its stable exit code, so the Swift side can branch
/// on "model not found" vs "cancelled" vs "unsupported PDF codec" without
/// string matching.
fn fail(stage: &str, err: &FocrError) -> i32 {
    set_error(format!("{stage}: {err}"));
    err.exit_code()
}

/// Generic-failure code for the non-`FocrError` edges (image decode, bad UTF-8,
/// a null pointer from the caller). Matches the CLI's exit-code table.
const EXIT_GENERIC: i32 = 1;
const EXIT_USAGE: i32 = 2;
const EXIT_INPUT_DECODE: i32 = 4;

/// Run `body`, converting an unwind into a recorded error plus `fallback`.
fn guarded<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".to_string());
            set_error(format!("internal panic: {message}"));
            fallback
        }
    }
}

// ── Pointer helpers (the audited island) ───────────────────────────────────
//
// Everything that dereferences a caller-supplied pointer goes through here, so
// the `unsafe` surface is four small functions with one SAFETY note each rather
// than a scattering across every entry point.
#[allow(unsafe_code)]
mod ptr_island {
    use super::{CStr, c_char};

    /// Borrow a caller-supplied NUL-terminated UTF-8 string.
    ///
    /// # Safety
    /// `s` must be NULL or a valid pointer to a NUL-terminated string that stays
    /// alive and unmodified for the duration of the call.
    pub(super) unsafe fn opt_str<'a>(s: *const c_char) -> Option<&'a str> {
        if s.is_null() {
            return None;
        }
        // SAFETY: non-null by the check above; the caller's contract (documented
        // in focr_ios.h) guarantees NUL termination and validity for the call.
        unsafe { CStr::from_ptr(s) }.to_str().ok()
    }

    /// Borrow a caller-supplied byte buffer.
    ///
    /// # Safety
    /// `ptr` must be NULL, or point to `len` initialized bytes that stay alive
    /// and unmodified for the duration of the call. A zero `len` yields an empty
    /// slice without dereferencing `ptr`.
    pub(super) unsafe fn opt_bytes<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
        if ptr.is_null() {
            return None;
        }
        if len == 0 {
            return Some(&[]);
        }
        // SAFETY: non-null and len > 0 by the checks above; the caller's
        // contract guarantees `len` initialized bytes valid for the call.
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Borrow a caller-supplied array of 1-based PDF page numbers.
    ///
    /// # Safety
    /// `ptr` must be NULL, or point to `len` initialized `u32` values that stay
    /// alive and unmodified for the duration of the call. A zero `len` yields
    /// an empty slice without dereferencing `ptr`.
    pub(super) unsafe fn opt_u32s<'a>(ptr: *const u32, len: usize) -> Option<&'a [u32]> {
        if ptr.is_null() {
            return None;
        }
        if len == 0 {
            return Some(&[]);
        }
        // SAFETY: non-null and len > 0 by the checks above; the caller's
        // contract guarantees `len` initialized u32 values valid for the call.
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Write `value` through a caller-supplied out-parameter, if it is non-NULL.
    ///
    /// # Safety
    /// `out` must be NULL or a valid, aligned, writable pointer to a `T`.
    pub(super) unsafe fn write_out<T>(out: *mut T, value: T) -> bool {
        if out.is_null() {
            return false;
        }
        // SAFETY: non-null by the check above; the caller's contract guarantees
        // it is aligned and writable.
        unsafe { out.write(value) };
        true
    }
}

/// Move a Rust `String` across the boundary as a caller-owned `char *`.
/// Returns NULL (and records an error) if the string contains a NUL.
fn into_owned_c_string(value: String) -> *mut c_char {
    match CString::new(value) {
        Ok(c) => c.into_raw(),
        Err(_) => {
            set_error("result contained an interior NUL and could not be returned");
            std::ptr::null_mut()
        }
    }
}

// ── Diagnostics ────────────────────────────────────────────────────────────

/// See `focr_ios.h`.
#[unsafe(no_mangle)]
#[allow(unsafe_code)] // audited export, part of the C ABI surface
pub extern "C" fn focr_last_error_message() -> *const c_char {
    // Returning a pointer into thread-local storage is sound for the documented
    // lifetime ("valid until the next call that replaces it on this thread"):
    // the CString is owned by this thread's slot and is only replaced by another
    // call on this same thread.
    LAST_ERROR.with(|slot| match slot.try_borrow() {
        Ok(slot) => slot.as_ptr(),
        Err(_) => c"".as_ptr(),
    })
}

/// See `focr_ios.h`.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub extern "C" fn focr_engine_info_json() -> *const c_char {
    static INFO: OnceLock<CString> = OnceLock::new();
    INFO.get_or_init(|| {
        let json = serde_json::json!({
            "crate_version": env!("CARGO_PKG_VERSION"),
            "detected_tier": franken_ocr::simd::tier_string(),
            "dense_route": format!("{:?}", franken_ocr::simd::effective_dense_route()),
            "threads": franken_ocr::kernel_pool_width(),
            "project_license": franken_ocr::FOCR_PROJECT_LICENSE_NOTICE,
        })
        .to_string();
        CString::new(json).unwrap_or_else(|_| CString::new("{}").expect("literal"))
    })
    .as_ptr()
}

/// See `focr_ios.h`.
///
/// # Safety
/// `out_json` must be NULL or a valid, aligned, writable `char *` slot. On
/// success it receives a caller-owned string that must be released with
/// [`focr_string_free`].
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn focr_selftest_json(out_json: *mut *mut c_char) -> i32 {
    guarded(EXIT_GENERIC, || {
        clear_error();
        let report = franken_ocr::simd::selftest();
        let json = serde_json::json!({
            "hardware_selected": format!("{:?}", report.hardware_selected),
            "effective_route": format!("{:?}", report.effective_route),
            "executed_routes": report
                .executed_routes
                .iter()
                .map(|r| format!("{r:?}"))
                .collect::<Vec<_>>(),
            "route_consistent": report.route_consistent,
            "available": report.available.iter().map(|t| format!("{t:?}")).collect::<Vec<_>>(),
            "models": report
                .models
                .iter()
                .map(|(name, ok)| serde_json::json!({"model": name, "ok": ok}))
                .collect::<Vec<_>>(),
            "all_ok": report.all_ok,
        })
        .to_string();
        let ok = report.all_ok && report.route_consistent;
        let ptr = into_owned_c_string(json);
        if ptr.is_null() {
            return EXIT_GENERIC;
        }
        // SAFETY: `out_json` is the caller's out-parameter, documented as NULL
        // or a valid writable `char *` slot.
        if !unsafe { ptr_island::write_out(out_json, ptr) } {
            // Nobody will free it if we cannot hand it back.
            drop(unsafe { CString::from_raw(ptr) });
            set_error("focr_selftest_json: out_json was NULL");
            return EXIT_USAGE;
        }
        if ok {
            0
        } else {
            set_error("selftest: int8 GEMM diverged from the scalar oracle on this device");
            EXIT_GENERIC
        }
    })
}

/// See `focr_ios.h`.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub extern "C" fn focr_kernel_pool_width() -> usize {
    guarded(1, franken_ocr::kernel_pool_width)
}

// ── Engine ─────────────────────────────────────────────────────────────────

/// Opaque engine handle. Holds the model plus the C strings whose pointers were
/// handed out for it, so those pointers stay valid exactly as long as the
/// engine does.
pub struct FocrEngine {
    model: Arc<OcrModel>,
    model_id: CString,
    license: CString,
}

fn cstring_or_empty(value: &str) -> CString {
    CString::new(value).unwrap_or_else(|_| CString::new("").expect("literal"))
}

/// See `focr_ios.h`.
///
/// # Safety
/// `artifact_path` must be NULL or point to a NUL-terminated UTF-8 string that
/// stays valid for the duration of the call. A non-NULL return is a handle the
/// caller owns and must release exactly once with [`focr_engine_close`].
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn focr_engine_open(artifact_path: *const c_char) -> *mut FocrEngine {
    guarded(std::ptr::null_mut(), || {
        clear_error();
        // SAFETY: documented as NULL or a valid NUL-terminated string.
        let Some(path) = (unsafe { ptr_island::opt_str(artifact_path) }) else {
            set_error("focr_engine_open: artifact_path was NULL or not valid UTF-8");
            return std::ptr::null_mut();
        };
        // Build the kernel pool before the first forward rather than during it,
        // so the pool's threads get their QoS class while the app is still
        // responsive rather than mid-page.
        let _ = franken_ocr::init_kernel_pool();
        let model = match OcrModel::load(Path::new(path)) {
            Ok(model) => model,
            Err(err) => {
                fail("focr_engine_open", &err);
                return std::ptr::null_mut();
            }
        };
        let model_id = cstring_or_empty(model.arch().id());
        let license = cstring_or_empty(model.arch().license_notice());
        Box::into_raw(Box::new(FocrEngine {
            model,
            model_id,
            license,
        }))
    })
}

/// See `focr_ios.h`.
///
/// # Safety
/// `engine` must be NULL, or a handle returned by [`focr_engine_open`] that has
/// not already been closed. Closing the same handle twice is undefined; closing
/// NULL is an explicit no-op.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn focr_engine_close(engine: *mut FocrEngine) {
    if engine.is_null() {
        return;
    }
    guarded((), || {
        // SAFETY: non-null by the check above, and documented as a pointer
        // previously returned by `focr_engine_open` and not yet closed. Taking
        // ownership back into a Box drops the model and its caches.
        drop(unsafe { Box::from_raw(engine) });
    });
}

/// Borrow an engine handle for a read-only call.
///
/// # Safety
/// `engine` must be NULL or a live handle from `focr_engine_open`.
#[allow(unsafe_code)]
unsafe fn engine_ref<'a>(engine: *const FocrEngine) -> Option<&'a FocrEngine> {
    if engine.is_null() {
        return None;
    }
    // SAFETY: non-null by the check above; the caller's contract guarantees the
    // handle is live and that access to it is serialized (see focr_ios.h).
    Some(unsafe { &*engine })
}

/// See `focr_ios.h`.
///
/// # Safety
/// `engine` must be NULL or a live handle from [`focr_engine_open`]. The
/// returned pointer is owned by the engine and is valid only until it is closed.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn focr_engine_model_id(engine: *const FocrEngine) -> *const c_char {
    // SAFETY: documented as NULL or a live engine handle.
    match unsafe { engine_ref(engine) } {
        Some(engine) => engine.model_id.as_ptr(),
        None => c"".as_ptr(),
    }
}

/// See `focr_ios.h`.
///
/// # Safety
/// `engine` must be NULL or a live handle from [`focr_engine_open`]. The
/// returned pointer is owned by the engine and is valid only until it is closed.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn focr_engine_license(engine: *const FocrEngine) -> *const c_char {
    // SAFETY: documented as NULL or a live engine handle.
    match unsafe { engine_ref(engine) } {
        Some(engine) => engine.license.as_ptr(),
        None => c"".as_ptr(),
    }
}

// ── Recognition ────────────────────────────────────────────────────────────

fn should_route_tall(model_id: &str, width: u32, height: u32) -> bool {
    model_id == "unlimited-ocr" && franken_ocr::tall::is_tall(width, height)
}

/// The CLI's tall-capture router applies only to ordinary document OCR. The
/// iOS picker exposes specialty models as distinct product modes, so only the
/// default Unlimited-OCR model is eligible here; a tall photo-description,
/// chart, structured-GOT, or music input is not safely line-concatenable.
fn recognize_document(
    engine: &FocrEngine,
    image: image::DynamicImage,
) -> FocrResult<(RecognizedDocument, Option<usize>)> {
    let (width, height) = (image.width(), image.height());
    let route_tall = should_route_tall(engine.model.arch().id(), width, height);
    if !route_tall {
        return engine
            .model
            .recognize_dynamic_with_layout(image)
            .map(|doc| (doc, None));
    }

    let profile = franken_ocr::tall::ink_profile(&image);
    let plan = franken_ocr::tall::plan_strips(width, height, &profile);
    let strips = franken_ocr::tall::cut_strips(&image, &plan);
    let mut parts = Vec::with_capacity(strips.len());
    for (strip, bounds) in strips.into_iter().zip(&plan) {
        let doc = engine.model.recognize_dynamic_with_layout(strip)?;
        parts.push((doc, bounds.top));
    }
    let strip_count = plan.len();
    Ok((franken_ocr::tall::merge_documents(parts), Some(strip_count)))
}

/// Build the same JSON envelope `focr-wasm`'s `recognize_json` returns, with
/// additive iOS metadata for the CLI's tall-capture and low-yield behavior.
fn recognize_envelope(engine: &FocrEngine, image_bytes: &[u8]) -> FocrResult<String> {
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| FocrError::InputDecode(format!("could not decode image: {e}")))?;
    let (width, height) = (img.width(), img.height());
    let (doc, tall_strip_count) = recognize_document(engine, img)?;
    let low_yield = (engine.model.arch().id() == "unlimited-ocr")
        .then(|| franken_ocr::tall::low_yield_assessment(&doc.markdown, width, height))
        .flatten()
        .map(|assessment| {
            serde_json::json!({
                "yield_chars": assessment.yield_chars,
                "input_megapixels": assessment.input_megapixels,
            })
        });
    let layout: Vec<serde_json::Value> = doc
        .layout
        .iter()
        .map(|span| serde_json::json!({"label": span.label, "boxes": span.boxes}))
        .collect();
    let music = engine.model.take_music_meta().map(|meta| {
        serde_json::json!({
            "staves": meta
                .staves
                .iter()
                .map(|(index, bbox)| serde_json::json!({
                    "index": index,
                    "bbox": [bbox.0, bbox.1, bbox.2, bbox.3],
                }))
                .collect::<Vec<_>>(),
            "skips": meta
                .skips
                .iter()
                .map(|skip| serde_json::json!({
                    "index": skip.index,
                    "bbox": [skip.bbox.0, skip.bbox.1, skip.bbox.2, skip.bbox.3],
                    "reason": skip.reason,
                }))
                .collect::<Vec<_>>(),
            "warnings": meta
                .warnings
                .iter()
                .map(|w| serde_json::json!({
                    "kind": w.kind,
                    "part": w.part,
                    "measure": w.measure,
                    "detail": w.detail,
                }))
                .collect::<Vec<_>>(),
        })
    });
    Ok(serde_json::json!({
        "model_id": engine.model.arch().id(),
        "output": doc.markdown,
        "layout": layout,
        "music": music,
        "tall_strip_count": tall_strip_count,
        "low_yield": low_yield,
    })
    .to_string())
}

/// See `focr_ios.h`.
///
/// # Safety
/// `engine` must be NULL or a live handle from [`focr_engine_open`], and access
/// to that handle must be serialized — it is not thread-safe. `image_bytes` must
/// be NULL or point to `image_len` initialized bytes valid for the call.
/// `out_json` must be NULL or a valid writable `char *` slot, which on success
/// receives a caller-owned string to release with [`focr_string_free`].
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn focr_recognize_json(
    engine: *mut FocrEngine,
    image_bytes: *const u8,
    image_len: usize,
    out_json: *mut *mut c_char,
) -> i32 {
    guarded(EXIT_GENERIC, || {
        clear_error();
        // SAFETY: documented as NULL or a live engine handle.
        let Some(engine) = (unsafe { engine_ref(engine) }) else {
            set_error("focr_recognize_json: engine was NULL");
            return EXIT_USAGE;
        };
        // SAFETY: documented as NULL or `image_len` initialized bytes.
        let Some(bytes) = (unsafe { ptr_island::opt_bytes(image_bytes, image_len) }) else {
            set_error("focr_recognize_json: image_bytes was NULL");
            return EXIT_USAGE;
        };
        if bytes.is_empty() {
            set_error("focr_recognize_json: image_bytes was empty");
            return EXIT_INPUT_DECODE;
        }
        let json = match recognize_envelope(engine, bytes) {
            Ok(json) => json,
            Err(err) => return fail("focr_recognize_json", &err),
        };
        let ptr = into_owned_c_string(json);
        if ptr.is_null() {
            return EXIT_GENERIC;
        }
        // SAFETY: the caller's out-parameter.
        if !unsafe { ptr_island::write_out(out_json, ptr) } {
            drop(unsafe { CString::from_raw(ptr) });
            set_error("focr_recognize_json: out_json was NULL");
            return EXIT_USAGE;
        }
        0
    })
}

// ── Progress and cancellation ──────────────────────────────────────────────

/// A C callback plus its opaque context, parked in a global so the engine's
/// progress sink can reach it.
///
/// The engine invokes the sink from whichever thread is running the forward, so
/// this must be `Send + Sync`. A raw `*mut c_void` is neither, which is exactly
/// the point of naming the invariant here instead of leaving it implicit.
#[derive(Clone, Copy)]
struct ProgressTarget {
    func: FocrProgressFn,
    ctx: *mut c_void,
}

// SAFETY: `ctx` is never dereferenced by Rust — it is an opaque token handed
// straight back to `func`. The header makes thread-safety the callback's
// obligation ("invoked from the forward's thread; must be thread-safe") and
// makes lifetime the installer's obligation ("clear it before releasing
// whatever ctx points at"). Under those two documented conditions, moving the
// pair between threads is sound.
#[allow(unsafe_code)]
unsafe impl Send for ProgressTarget {}
// SAFETY: as above; `ProgressTarget` is immutable once installed, so shared
// references carry no additional obligation beyond `Send`.
#[allow(unsafe_code)]
unsafe impl Sync for ProgressTarget {}

/// See `focr_ios.h`.
pub type FocrProgressFn =
    extern "C" fn(ctx: *mut c_void, stage: *const c_char, current: u64, total: u64);

struct ProgressRegistryState {
    target: Option<ProgressTarget>,
    in_flight: usize,
}

struct ProgressRegistry {
    state: Mutex<ProgressRegistryState>,
    quiesced: Condvar,
    /// Only one external callback may run at a time. This makes reentrant
    /// self-clear compatible with quiescence: two simultaneous callbacks can
    /// never each wait for the other to retire.
    invocation_gate: Mutex<()>,
}

fn progress_registry() -> &'static ProgressRegistry {
    static REGISTRY: OnceLock<ProgressRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| ProgressRegistry {
        state: Mutex::new(ProgressRegistryState {
            target: None,
            in_flight: 0,
        }),
        quiesced: Condvar::new(),
        invocation_gate: Mutex::new(()),
    })
}

thread_local! {
    /// Distinguishes a callback clearing itself from an unrelated thread that
    /// must wait for all borrowed callback contexts to quiesce before returning.
    static IN_PROGRESS_CALLBACK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct ProgressInvocation {
    target: ProgressTarget,
    _gate: std::sync::MutexGuard<'static, ()>,
}

impl ProgressInvocation {
    fn begin() -> Option<Self> {
        let registry = progress_registry();
        let gate = registry
            .invocation_gate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut state = registry
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let target = state.target?;
        state.in_flight = state
            .in_flight
            .checked_add(1)
            .expect("progress callback in-flight count overflow");
        IN_PROGRESS_CALLBACK.with(|active| active.set(true));
        Some(Self {
            target,
            _gate: gate,
        })
    }
}

impl Drop for ProgressInvocation {
    fn drop(&mut self) {
        IN_PROGRESS_CALLBACK.with(|active| active.set(false));
        let registry = progress_registry();
        let mut state = registry
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(state.in_flight > 0);
        state.in_flight -= 1;
        if state.in_flight == 0 {
            registry.quiesced.notify_all();
        }
    }
}

/// See `focr_ios.h`.
///
/// # Safety
/// `ctx` is never dereferenced here — it is an opaque token handed back to
/// `func` — but the caller must keep whatever it points at alive for as long as
/// the callback is installed, and must clear the callback (`func = None`) before
/// releasing it. `func` may be invoked from any thread, so it must be
/// thread-safe and non-blocking.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn focr_set_progress_callback(
    func: Option<FocrProgressFn>,
    ctx: *mut c_void,
) {
    guarded((), || {
        let target = func.map(|func| ProgressTarget { func, ctx });
        let called_from_callback = IN_PROGRESS_CALLBACK.with(std::cell::Cell::get);
        let registry = progress_registry();
        {
            let mut state = registry
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Stop new invocations before waiting for existing ones. A caller on
            // another thread may release its old `ctx` as soon as this function
            // returns, so replacement has to be a quiescing operation.
            state.target = None;
            while !called_from_callback && state.in_flight != 0 {
                state = registry
                    .quiesced
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            state.target = target;
            // Keep the registry state and the engine fast-path arm/disarm in one
            // linearized critical section. Otherwise concurrent install/clear
            // calls can publish their two halves in opposite orders and leave a
            // live target permanently disarmed (or an empty target armed).
            if target.is_none() {
                franken_ocr::native_engine::progress::set_progress_sink(None);
            } else {
                franken_ocr::native_engine::progress::set_progress_sink(Some(Arc::new(|event| {
                    // SAFETY: acquire an in-flight lease, release the registry mutex,
                    // then invoke caller code. Clearing from another thread waits for
                    // this lease; clearing reentrantly marks the target absent and lets
                    // the current invocation retire naturally on return.
                    let Some(invocation) = ProgressInvocation::begin() else {
                        return;
                    };
                    // `event.stage` is a `&'static str` from a fixed set, but it is not
                    // NUL-terminated, so it needs one allocation to cross as a C string.
                    // The hooks sit on outer sequential loops (per vision block, per
                    // token), never inside a rayon body, so this is not a hot path.
                    let Ok(stage) = CString::new(event.stage) else {
                        return;
                    };
                    (invocation.target.func)(
                        invocation.target.ctx,
                        stage.as_ptr(),
                        event.current,
                        event.total,
                    );
                })));
            }
        }
    });
}

/// See `focr_ios.h`.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub extern "C" fn focr_request_cancel() {
    guarded((), franken_ocr::request_shutdown);
}

/// See `focr_ios.h`.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub extern "C" fn focr_reset_cancel() {
    guarded((), franken_ocr::reset_shutdown);
}

// ── PDF ────────────────────────────────────────────────────────────────────

/// An opened PDF. Parsing a document is not free — a scanned book is tens of
/// megabytes of object graph — so the parse happens ONCE here and every page
/// render borrows it. A stateless render-by-bytes API re-parses per page, which
/// turns a 300-page document into 300 full parses.
pub struct FocrPdf {
    pages: franken_ocr::pdf::PdfPages,
}

/// See `focr_ios.h`.
///
/// # Safety
/// `pdf_bytes` must be NULL or point to `pdf_len` initialized bytes valid for
/// the duration of the call. The bytes are parsed into an owned document, so the
/// caller may release them as soon as this returns. A non-NULL return must be
/// released exactly once with [`focr_pdf_close`].
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn focr_pdf_open(pdf_bytes: *const u8, pdf_len: usize) -> *mut FocrPdf {
    guarded(std::ptr::null_mut(), || {
        clear_error();
        // SAFETY: documented as NULL or `pdf_len` initialized bytes.
        let Some(bytes) = (unsafe { ptr_island::opt_bytes(pdf_bytes, pdf_len) }) else {
            set_error("focr_pdf_open: pdf_bytes was NULL");
            return std::ptr::null_mut();
        };
        match franken_ocr::pdf::PdfPages::from_bytes(bytes) {
            Ok(pages) => Box::into_raw(Box::new(FocrPdf { pages })),
            Err(err) => {
                fail("focr_pdf_open", &err);
                std::ptr::null_mut()
            }
        }
    })
}

/// See `focr_ios.h`.
///
/// # Safety
/// `pdf` must be NULL or a handle from [`focr_pdf_open`] that has not been
/// closed. Closing NULL is a no-op; closing twice is undefined.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn focr_pdf_close(pdf: *mut FocrPdf) {
    if pdf.is_null() {
        return;
    }
    guarded((), || {
        // SAFETY: non-null by the check above, and documented as a live handle.
        drop(unsafe { Box::from_raw(pdf) });
    });
}

/// See `focr_ios.h`.
///
/// # Safety
/// `pdf` must be NULL or a live handle from [`focr_pdf_open`].
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn focr_pdf_page_count(pdf: *const FocrPdf) -> u32 {
    guarded(0, || {
        if pdf.is_null() {
            set_error("focr_pdf_page_count: pdf was NULL");
            return 0;
        }
        // SAFETY: non-null by the check above, and documented as a live handle.
        let pdf = unsafe { &*pdf };
        u32::try_from(pdf.pages.len()).unwrap_or(u32::MAX)
    })
}

/// See `focr_ios.h`.
///
/// # Safety
/// `pdf` must be NULL or a live handle from [`focr_pdf_open`]. `out_png` and
/// `out_len` must be NULL or valid writable slots; on success they receive a
/// caller-owned buffer that must be released with [`focr_bytes_free`], passing
/// back exactly the length returned.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn focr_pdf_render_page(
    pdf: *const FocrPdf,
    page: u32,
    out_png: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guarded(EXIT_GENERIC, || {
        clear_error();
        if pdf.is_null() {
            set_error("focr_pdf_render_page: pdf was NULL");
            return EXIT_USAGE;
        }
        // SAFETY: non-null by the check above, and documented as a live handle.
        let pdf = unsafe { &*pdf };
        let png = match render_pdf_page_png(&pdf.pages, page) {
            Ok(png) => png,
            Err(err) => return fail("focr_pdf_render_page", &err),
        };
        let len = png.len();
        let ptr = Box::into_raw(png.into_boxed_slice()).cast::<u8>();
        // SAFETY: the caller's out-parameters.
        let wrote_ptr = unsafe { ptr_island::write_out(out_png, ptr) };
        let wrote_len = unsafe { ptr_island::write_out(out_len, len) };
        if !wrote_ptr || !wrote_len {
            // Reclaim rather than leak when we cannot hand the buffer back.
            // SAFETY: `ptr`/`len` are exactly what we just produced above.
            drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) });
            set_error("focr_pdf_render_page: out_png or out_len was NULL");
            return EXIT_USAGE;
        }
        0
    })
}

/// Rasterize one page and re-encode it as PNG for the Swift side.
///
/// `page` is 1-based at this boundary (it is what a person sees in a page
/// picker); `PdfPages::render` is 0-based. Converting here rather than in Swift
/// keeps the off-by-one in one place, next to the range check that produces the
/// error message naming the real page count.
fn render_pdf_page_png(pages: &franken_ocr::pdf::PdfPages, page: u32) -> FocrResult<Vec<u8>> {
    if page == 0 {
        return Err(FocrError::Usage(format!(
            "PDF pages are 1-based; this document has {} page(s)",
            pages.len()
        )));
    }
    let image = pages.render(page as usize - 1)?;
    let mut png = Vec::new();
    image
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| FocrError::InputDecode(format!("could not encode page {page} as PNG: {e}")))?;
    Ok(png)
}

/// Cross-page recognition has a deliberately tighter app limit than the
/// engine's 32K token ceiling. The boundary temporarily owns every rendered
/// page before the core squashes them to Base-640; bounding that collection is
/// what keeps an unusually high-resolution scan from turning an otherwise valid
/// context into iOS memory pressure. Longer books remain available in explicit
/// page ranges, or through the independent per-page workflow.
const MAX_APPLE_CROSS_PAGE_PAGES: usize = 32;

fn validate_cross_page_selection(selected: &[u32], page_count: usize) -> FocrResult<()> {
    if selected.len() < 2 {
        return Err(FocrError::Usage(
            "cross-page recognition needs at least two selected pages".to_string(),
        ));
    }
    if selected.len() > MAX_APPLE_CROSS_PAGE_PAGES {
        return Err(FocrError::Usage(format!(
            "cross-page recognition is bounded to {MAX_APPLE_CROSS_PAGE_PAGES} pages on Apple \
             devices; select a shorter range or turn off shared context"
        )));
    }

    let mut previous = 0;
    for &page in selected {
        let page_index = usize::try_from(page).unwrap_or(usize::MAX);
        if page == 0 || page_index > page_count {
            return Err(FocrError::Usage(format!(
                "cross-page source page {page} is out of range; this document has \
                 {page_count} page(s)"
            )));
        }
        if page <= previous {
            return Err(FocrError::Usage(
                "cross-page source pages must be unique and strictly increasing".to_string(),
            ));
        }
        previous = page;
    }
    Ok(())
}

fn split_cross_page_output(output: &str, expected: usize) -> FocrResult<Vec<String>> {
    let pages: Vec<String> = output
        .split(franken_ocr::native_engine::postprocess::PAGE_MARKER)
        .skip(1)
        .map(str::trim)
        .map(ToOwned::to_owned)
        .collect();
    if pages.len() != expected {
        return Err(FocrError::Other(
            std::io::Error::other(format!(
                "cross-page result contained {} page bodies for {expected} selected pages",
                pages.len()
            ))
            .into(),
        ));
    }
    Ok(pages)
}

fn recognize_pdf_cross_page_envelope(
    engine: &FocrEngine,
    pdf: &FocrPdf,
    selected: &[u32],
) -> FocrResult<String> {
    validate_cross_page_selection(selected, pdf.pages.len())?;
    let mut images = Vec::with_capacity(selected.len());
    for &page in selected {
        images.push(pdf.pages.render(page as usize - 1)?);
    }

    let output = engine.model.recognize_multi_page_dynamic(images)?;
    let page_bodies = split_cross_page_output(&output, selected.len())?;
    let pages = selected
        .iter()
        .zip(page_bodies)
        .map(|(&source_page, output)| {
            serde_json::json!({"source_page": source_page, "output": output})
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "model_id": engine.model.arch().id(),
        "output": output,
        "pages": pages,
    })
    .to_string())
}

/// See `focr_ios.h`.
///
/// # Safety
/// `engine` and `pdf` must be NULL or live handles from this library, and all
/// access to each handle must be serialized. `source_pages` must be NULL or
/// point to `page_count` initialized values valid for the call. `out_json` must
/// be NULL or a writable `char *` slot; on success it receives a caller-owned
/// string released with [`focr_string_free`].
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn focr_recognize_pdf_cross_page_json(
    engine: *mut FocrEngine,
    pdf: *const FocrPdf,
    source_pages: *const u32,
    page_count: usize,
    out_json: *mut *mut c_char,
) -> i32 {
    guarded(EXIT_GENERIC, || {
        clear_error();
        // SAFETY: documented as NULL or a live engine handle.
        let Some(engine) = (unsafe { engine_ref(engine) }) else {
            set_error("focr_recognize_pdf_cross_page_json: engine was NULL");
            return EXIT_USAGE;
        };
        if pdf.is_null() {
            set_error("focr_recognize_pdf_cross_page_json: pdf was NULL");
            return EXIT_USAGE;
        }
        // SAFETY: non-null by the check above and documented as a live handle.
        let pdf = unsafe { &*pdf };
        // SAFETY: documented as NULL or `page_count` initialized values.
        let Some(selected) = (unsafe { ptr_island::opt_u32s(source_pages, page_count) }) else {
            set_error("focr_recognize_pdf_cross_page_json: source_pages was NULL");
            return EXIT_USAGE;
        };
        let json = match recognize_pdf_cross_page_envelope(engine, pdf, selected) {
            Ok(json) => json,
            Err(err) => return fail("focr_recognize_pdf_cross_page_json", &err),
        };
        let ptr = into_owned_c_string(json);
        if ptr.is_null() {
            return EXIT_GENERIC;
        }
        // SAFETY: the caller's out-parameter.
        if !unsafe { ptr_island::write_out(out_json, ptr) } {
            drop(unsafe { CString::from_raw(ptr) });
            set_error("focr_recognize_pdf_cross_page_json: out_json was NULL");
            return EXIT_USAGE;
        }
        0
    })
}

// ── Decode options ─────────────────────────────────────────────────────────

/// See `focr_ios.h`.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub extern "C" fn focr_set_no_repeat_ngram(n: u32) {
    guarded((), || {
        franken_ocr::native_engine::set_decode_overrides(
            franken_ocr::native_engine::DecodeOverrides {
                no_repeat_ngram: if n == 0 { None } else { Some(n as usize) },
                ..Default::default()
            },
        );
    });
}

/// See `focr_ios.h`.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub extern "C" fn focr_set_got_format(on: bool) {
    guarded((), || franken_ocr::native_engine::force_got_format(on));
}

/// See `focr_ios.h`.
///
/// # Safety
/// `question` must be NULL or point to a NUL-terminated UTF-8 string valid for
/// the duration of the call. The contents are copied; the caller keeps ownership.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn focr_set_smolvlm2_question(question: *const c_char) {
    guarded((), || {
        // SAFETY: documented as NULL or a valid NUL-terminated string.
        let question = unsafe { ptr_island::opt_str(question) }.unwrap_or("");
        franken_ocr::native_engine::set_smolvlm2_question(if question.is_empty() {
            None
        } else {
            Some(question.to_string())
        });
    });
}

// ── Memory ─────────────────────────────────────────────────────────────────

/// See `focr_ios.h`.
///
/// # Safety
/// `s` must be NULL, or a pointer this library returned through a `char **`
/// out-parameter and that has not already been freed.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn focr_string_free(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    // SAFETY: documented as a pointer previously returned by this library
    // through a `char **` out-parameter and not yet freed.
    drop(unsafe { CString::from_raw(s) });
}

/// See `focr_ios.h`.
///
/// # Safety
/// `ptr` must be NULL, or a buffer this library returned with EXACTLY this
/// `len`, not already freed. The length is part of the contract: it is used to
/// reconstitute the owning box, so a wrong length is undefined behavior.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn focr_bytes_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    // SAFETY: documented as a pointer previously returned by this library with
    // EXACTLY this length, and not yet freed.
    drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) });
}

#[cfg(test)]
// The tests exercise the C ABI as a C caller would, so they read back the
// `const char *` returns. Same island discipline as the exports above: the
// pointers under test are ones this crate just produced and still owns.
#[allow(unsafe_code)]
mod tests {
    use super::*;

    static PROGRESS_TEST_LOCK: Mutex<()> = Mutex::new(());
    static REENTRANT_PROGRESS_COUNT: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    extern "C" fn clear_progress_from_callback(
        _ctx: *mut c_void,
        _stage: *const c_char,
        _current: u64,
        _total: u64,
    ) {
        REENTRANT_PROGRESS_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: clearing uses no context and is explicitly supported by the C ABI.
        unsafe { focr_set_progress_callback(None, std::ptr::null_mut()) };
    }

    #[test]
    fn error_slot_starts_empty_and_records() {
        // A fresh thread so the assertion is about the initial state, not about
        // whatever an earlier test in this thread left behind.
        std::thread::spawn(|| {
            let empty = unsafe { CStr::from_ptr(focr_last_error_message()) };
            assert_eq!(empty.to_bytes(), b"");
            set_error("boom");
            let recorded = unsafe { CStr::from_ptr(focr_last_error_message()) };
            assert_eq!(recorded.to_str().expect("utf8"), "boom");
        })
        .join()
        .expect("thread");
    }

    #[test]
    fn interior_nul_in_error_is_replaced_not_truncated() {
        std::thread::spawn(|| {
            set_error("before\0after");
            let recorded = unsafe { CStr::from_ptr(focr_last_error_message()) };
            assert_eq!(recorded.to_str().expect("utf8"), "before?after");
        })
        .join()
        .expect("thread");
    }

    #[test]
    fn null_arguments_are_usage_errors_not_crashes() {
        // SAFETY: every pointer here is NULL, which each entry point documents
        // as a usage error it must detect and report rather than dereference.
        // Proving exactly that is the point of this test.
        unsafe {
            assert_eq!(
                focr_recognize_json(
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut()
                ),
                EXIT_USAGE
            );
            assert_eq!(focr_pdf_page_count(std::ptr::null()), 0);
            assert!(focr_pdf_open(std::ptr::null(), 0).is_null());
            focr_pdf_close(std::ptr::null_mut());
            assert_eq!(
                focr_pdf_render_page(
                    std::ptr::null(),
                    1,
                    std::ptr::null_mut(),
                    std::ptr::null_mut()
                ),
                EXIT_USAGE
            );
            assert_eq!(
                focr_recognize_pdf_cross_page_json(
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut()
                ),
                EXIT_USAGE
            );
            // Closing NULL is explicitly a no-op.
            focr_engine_close(std::ptr::null_mut());
            focr_string_free(std::ptr::null_mut());
            focr_bytes_free(std::ptr::null_mut(), 0);
        }
    }

    #[test]
    fn opening_a_missing_artifact_reports_model_not_found() {
        let path = CString::new("/nonexistent/focr-ios/model.focrq").expect("literal");
        // SAFETY: `path` is a live CString for the duration of the call.
        let engine = unsafe { focr_engine_open(path.as_ptr()) };
        assert!(engine.is_null());
        let message = unsafe { CStr::from_ptr(focr_last_error_message()) }
            .to_str()
            .expect("utf8");
        assert!(
            message.contains("focr_engine_open"),
            "message should name the stage, got {message:?}"
        );
    }

    #[test]
    fn engine_info_is_valid_json_with_the_route_split() {
        let info = unsafe { CStr::from_ptr(focr_engine_info_json()) }
            .to_str()
            .expect("utf8");
        let parsed: serde_json::Value = serde_json::from_str(info).expect("valid json");
        // Hardware capability and the effective route are reported separately on
        // purpose; collapsing them would overclaim on Apple silicon.
        assert!(parsed.get("detected_tier").is_some());
        assert!(parsed.get("dense_route").is_some());
        assert!(parsed.get("threads").is_some());
    }

    #[test]
    fn tall_router_is_limited_to_the_default_document_model() {
        assert!(should_route_tall("unlimited-ocr", 500, 1_500));
        assert!(!should_route_tall("unlimited-ocr", 500, 1_499));
        assert!(!should_route_tall("smol-vlm", 500, 2_000));
        assert!(!should_route_tall("got-ocr", 500, 2_000));
        assert!(!should_route_tall("tromr", 500, 2_000));
    }

    #[test]
    fn cross_page_selection_is_bounded_ordered_and_in_range() {
        assert!(validate_cross_page_selection(&[1, 3, 7], 7).is_ok());
        assert!(validate_cross_page_selection(&[1], 7).is_err());
        assert!(validate_cross_page_selection(&[0, 1], 7).is_err());
        assert!(validate_cross_page_selection(&[1, 8], 7).is_err());
        assert!(validate_cross_page_selection(&[2, 2], 7).is_err());
        assert!(validate_cross_page_selection(&[3, 2], 7).is_err());
        assert!(
            validate_cross_page_selection(
                &(1..=u32::try_from(MAX_APPLE_CROSS_PAGE_PAGES + 1).expect("small bound"))
                    .collect::<Vec<_>>(),
                MAX_APPLE_CROSS_PAGE_PAGES + 1,
            )
            .is_err()
        );
    }

    #[test]
    fn cross_page_output_keeps_empty_pages_and_refuses_count_drift() {
        let output = "discarded preface<PAGE> first <PAGE>   <PAGE>third\n";
        assert_eq!(
            split_cross_page_output(output, 3).expect("three page bodies"),
            ["first", "", "third"]
        );
        assert!(split_cross_page_output(output, 2).is_err());
        assert!(split_cross_page_output("no marker", 1).is_err());
    }

    #[test]
    fn progress_callback_can_clear_itself_without_deadlocking() {
        let _guard = PROGRESS_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        REENTRANT_PROGRESS_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
        for _ in 0..10 {
            // SAFETY: the callback has no context and clears itself before returning.
            unsafe {
                focr_set_progress_callback(Some(clear_progress_from_callback), std::ptr::null_mut())
            };
            franken_ocr::native_engine::progress::emit("decode", 1, 1);
            franken_ocr::native_engine::progress::emit("decode", 1, 1);
        }
        assert_eq!(
            REENTRANT_PROGRESS_COUNT.load(std::sync::atomic::Ordering::Relaxed),
            10
        );
    }

    #[test]
    fn selftest_proves_the_int8_kernels_on_this_host() {
        let mut out: *mut c_char = std::ptr::null_mut();
        // SAFETY: `out` is a live, writable slot; the returned string is owned
        // by this caller and is freed exactly once below.
        let code = unsafe { focr_selftest_json(&raw mut out) };
        assert!(!out.is_null(), "selftest must always return its report");
        let json = unsafe { CStr::from_ptr(out) }.to_str().expect("utf8");
        let parsed: serde_json::Value = serde_json::from_str(json).expect("valid json");
        assert_eq!(parsed["all_ok"], serde_json::Value::Bool(true));
        assert_eq!(code, 0, "int8 GEMM must match the scalar oracle: {json}");
        // SAFETY: `out` came from this library and has not been freed.
        unsafe { focr_string_free(out) };
    }
}
