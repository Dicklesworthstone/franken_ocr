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
use std::sync::{Arc, Mutex, OnceLock};

use franken_ocr::error::{FocrError, FocrResult};
use franken_ocr::native_engine::OcrModel;

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

/// Build the same JSON envelope `focr-wasm`'s `recognize_json` returns, so the
/// app and the playground consume ONE result shape.
fn recognize_envelope(engine: &FocrEngine, image_bytes: &[u8]) -> FocrResult<String> {
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| FocrError::InputDecode(format!("could not decode image: {e}")))?;
    let doc = engine.model.recognize_dynamic_with_layout(img)?;
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

fn progress_slot() -> &'static Mutex<Option<ProgressTarget>> {
    static SLOT: OnceLock<Mutex<Option<ProgressTarget>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
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
        if let Ok(mut slot) = progress_slot().lock() {
            *slot = target;
        }
        if target.is_none() {
            // Disarm the engine's relaxed-atomic fast path so an un-observed run
            // pays nothing at all.
            franken_ocr::native_engine::progress::set_progress_sink(None);
            return;
        }
        franken_ocr::native_engine::progress::set_progress_sink(Some(Box::new(|event| {
            // `try_lock`, not `lock`: a progress hook must never be able to
            // block — or deadlock — the forward it is reporting on.
            let Ok(slot) = progress_slot().try_lock() else {
                return;
            };
            let Some(target) = *slot else { return };
            // `event.stage` is a `&'static str` from a fixed set, but it is not
            // NUL-terminated, so it needs one allocation to cross as a C string.
            // The hooks sit on outer sequential loops (per vision block, per
            // token), never inside a rayon body, so this is not a hot path.
            let Ok(stage) = CString::new(event.stage) else {
                return;
            };
            (target.func)(target.ctx, stage.as_ptr(), event.current, event.total);
        })));
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
