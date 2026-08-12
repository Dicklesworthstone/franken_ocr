/* @ts-self-types="./focr_wasm.d.ts" */
import { startWorkers } from './snippets/wasm-bindgen-rayon-38edf6e439f6d70d/src/workerHelpers.no-bundler.js';


/**
 * Staging area for a model arriving over the network as chunks: exactly one
 * copy of the weight blob lives here, reserved up front, plus the (small)
 * tokenizer sidecars keyed by their canonical zoo filenames.
 */
export class ModelStaging {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelStagingFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelstaging_free(ptr, 0);
    }
    /**
     * Bytes staged so far (for progress display).
     * @returns {number}
     */
    filled() {
        const ret = wasm.modelstaging_filled(this.__wbg_ptr);
        return ret;
    }
    /**
     * Free the staged bytes explicitly (a superseded or failed load should
     * not wait for the JS GC's FinalizationRegistry).
     */
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.modelstaging_free(ptr);
    }
    /**
     * An empty staging area. Call [`Self::reserve`] before the first
     * [`Self::push`].
     */
    constructor() {
        const ret = wasm.modelstaging_new();
        this.__wbg_ptr = ret;
        ModelStagingFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * PHASE 1 of staging (bd-syf2): plan the blob's segmentation from a
     * downloaded PREFIX of the artifact, and reserve every planned segment.
     *
     * `header_prefix` is the artifact's leading bytes (4 MiB is ample: the
     * 3.0 GB Unlimited artifact's header is ~0.5 MB). `total_bytes` is the
     * artifact's full pinned size. Returns a JSON object:
     *
     * * `{"status":"planned","segments":[len,…],"payload_base":N}` — staging is
     *   armed; start pushing bytes from offset 0.
     * * `{"status":"need_prefix","need_bytes":N}` — the prefix did not contain
     *   the whole header; refetch at least `N` leading bytes and call again
     *   (this converges in at most two probes).
     *
     * A small model (TrOMR) plans to exactly one segment and behaves like the
     * old single-buffer `reserve` did.
     *
     * # Errors
     * A malformed header, an artifact whose layout offers no clean cut, or a
     * failed reservation (named with its byte count, never an opaque trap).
     * @param {Uint8Array} header_prefix
     * @param {number} total_bytes
     * @returns {string}
     */
    plan(header_prefix, total_bytes) {
        let deferred3_0;
        let deferred3_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray8ToWasm0(header_prefix, wasm.__wbindgen_export3);
            const len0 = WASM_VECTOR_LEN;
            wasm.modelstaging_plan(retptr, this.__wbg_ptr, ptr0, len0, total_bytes);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            var ptr2 = r0;
            var len2 = r1;
            if (r3) {
                ptr2 = 0; len2 = 0;
                throw takeObject(r2);
            }
            deferred3_0 = ptr2;
            deferred3_1 = len2;
            return getStringFromWasm0(ptr2, len2);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export2(deferred3_0, deferred3_1, 1);
        }
    }
    /**
     * PHASE 2: append one downloaded chunk. The caller streams the artifact
     * start-to-end and never needs to know where the segment edges are — a
     * chunk that spans a boundary is split across the two segments here.
     * Refuses bytes past the planned total: a mismatched manifest must fail
     * loudly, not grow silently.
     * @param {Uint8Array} chunk
     */
    push(chunk) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray8ToWasm0(chunk, wasm.__wbindgen_export3);
            const len0 = WASM_VECTOR_LEN;
            wasm.modelstaging_push(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * The planned segment count (1 for every model that fits one buffer).
     * @returns {number}
     */
    segment_count() {
        const ret = wasm.modelstaging_segment_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Attach one tokenizer sidecar by its canonical zoo filename:
     * `tokenizer.json`, `qwen.tiktoken`, the four TrOMR tables
     * `tokenizer_{rhythm,pitch,lift,note}.json`, or OneChart's OPT
     * slow-tokenizer triple `vocab.json` / `merges.txt` /
     * `added_tokens.json`.
     * @param {string} name
     * @param {Uint8Array} bytes
     */
    set_sidecar(name, bytes) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(name, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArray8ToWasm0(bytes, wasm.__wbindgen_export3);
            const len1 = WASM_VECTOR_LEN;
            wasm.modelstaging_set_sidecar(retptr, this.__wbg_ptr, ptr0, len0, ptr1, len1);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) ModelStaging.prototype[Symbol.dispose] = ModelStaging.prototype.free;

/**
 * A loaded model plus the recognize entrypoints the playground calls.
 */
export class WasmEngine {
    static __wrap(ptr) {
        const obj = Object.create(WasmEngine.prototype);
        obj.__wbg_ptr = ptr;
        WasmEngineFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmEngineFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmengine_free(ptr, 0);
    }
    /**
     * Free the engine (and, if this was the last handle, the weight bytes)
     * explicitly rather than waiting for the JS GC.
     */
    free_engine() {
        const ptr = this.__destroy_into_raw();
        wasm.wasmengine_free_engine(ptr);
    }
    /**
     * Build the engine from a completed staging area (consumes it — the
     * weight bytes move, they are not copied).
     *
     * Fails if the staging is incomplete, the blob does not parse as a
     * `.focrq`/safetensors container, or the Unlimited-OCR recipe validation
     * rejects the tensor dtypes.
     * @param {ModelStaging} staging
     * @returns {WasmEngine}
     */
    static from_staging(staging) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(staging, ModelStaging);
            var ptr0 = staging.__destroy_into_raw();
            wasm.wasmengine_from_staging(retptr, ptr0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return WasmEngine.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * The model-weights license notice that must travel with the artifact.
     * @returns {string}
     */
    license_notice() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmengine_license_notice(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export2(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * The loaded model's registry id (`unlimited-ocr`, `tromr`, …).
     * @returns {string}
     */
    model_id() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmengine_model_id(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export2(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Recognize one encoded image (PNG/JPEG bytes) and return the model's
     * primary text output: markdown for the OCR models, MusicXML for TrOMR.
     * @param {Uint8Array} image_bytes
     * @returns {string}
     */
    recognize(image_bytes) {
        let deferred3_0;
        let deferred3_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray8ToWasm0(image_bytes, wasm.__wbindgen_export3);
            const len0 = WASM_VECTOR_LEN;
            wasm.wasmengine_recognize(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            var ptr2 = r0;
            var len2 = r1;
            if (r3) {
                ptr2 = 0; len2 = 0;
                throw takeObject(r2);
            }
            deferred3_0 = ptr2;
            deferred3_1 = len2;
            return getStringFromWasm0(ptr2, len2);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export2(deferred3_0, deferred3_1, 1);
        }
    }
    /**
     * Recognize one encoded image and return a JSON envelope:
     * `{"model_id", "output", "layout": [{label, boxes}], "music": {...}?}`.
     * `layout` mirrors `focr ocr --json`; `music` carries the TrOMR staff
     * metadata (recognized bboxes, per-staff skips with reasons, annotate-only
     * warnings) when the run produced any.
     * @param {Uint8Array} image_bytes
     * @returns {string}
     */
    recognize_json(image_bytes) {
        let deferred3_0;
        let deferred3_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray8ToWasm0(image_bytes, wasm.__wbindgen_export3);
            const len0 = WASM_VECTOR_LEN;
            wasm.wasmengine_recognize_json(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            var ptr2 = r0;
            var len2 = r1;
            if (r3) {
                ptr2 = 0; len2 = 0;
                throw takeObject(r2);
            }
            deferred3_0 = ptr2;
            deferred3_1 = len2;
            return getStringFromWasm0(ptr2, len2);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export2(deferred3_0, deferred3_1, 1);
        }
    }
}
if (Symbol.dispose) WasmEngine.prototype[Symbol.dispose] = WasmEngine.prototype.free;

/**
 * One JSON object describing this module: crate version, detected ISA tier,
 * and the licenses that must travel with the model weights.
 * @returns {string}
 */
export function engine_info() {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.engine_info(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export2(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Install the panic hook. `#[wasm_bindgen(start)]` runs this once at module
 * instantiation, before any exported call.
 */
export function focr_wasm_start() {
    wasm.focr_wasm_start();
}

/**
 * @param {number} num_threads
 * @returns {Promise<any>}
 */
export function initThreadPool(num_threads) {
    const ret = wasm.initThreadPool(num_threads);
    return takeObject(ret);
}

/**
 * The effective ordinary dense-int8 route on this host. On wasm32 today this
 * reports the scalar tier; when a simd128 kernel island lands it will report
 * that route, and the site asserts whichever it expects — a silent scalar
 * fallback is invisible any other way.
 * @returns {string}
 */
export function int8_route() {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.int8_route(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export2(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Sniff + parse a PDF and report its shape as JSON: `{"pages": N}`.
 *
 * The playground calls this once per uploaded PDF to drive its page picker;
 * the bytes cross the boundary per call (no session object — a scanned PDF is
 * a few MB, and statelessness keeps the worker protocol trivial).
 *
 * # Errors
 * A non-PDF payload (no `%PDF-` magic) or a parse failure, each named.
 * @param {Uint8Array} bytes
 * @returns {string}
 */
export function pdf_info(bytes) {
    let deferred3_0;
    let deferred3_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        wasm.pdf_info(retptr, ptr0, len0);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
        var ptr2 = r0;
        var len2 = r1;
        if (r3) {
            ptr2 = 0; len2 = 0;
            throw takeObject(r2);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export2(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Rasterize one PDF page (1-based, matching the CLI's `--pages` convention)
 * to PNG bytes.
 *
 * The raster rides the exact native pipeline — largest image XObject, codec
 * dispatch, page `/Rotate` + content-matrix rotation normalization — so the
 * PNG this returns is pixel-identical to what `focr ocr file.pdf` would feed
 * the model for that page. JS previews it in an `<img>` and passes it back
 * through the existing [`WasmEngine::recognize`] path unchanged.
 *
 * # Errors
 * Parse failures, an out-of-range page, and unsupported codecs — a
 * `JPXDecode`/`JBIG2Decode` page surfaces the library's precise existing
 * error text (…"no pure-Rust decoder; rasterize this PDF out of band and
 * retry"), never a panic.
 * @param {Uint8Array} bytes
 * @param {number} page
 * @returns {Uint8Array}
 */
export function pdf_render_page(bytes, page) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        wasm.pdf_render_page(retptr, ptr0, len0, page);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
        if (r3) {
            throw takeObject(r2);
        }
        var v2 = getArrayU8FromWasm0(r0, r1).slice();
        wasm.__wbindgen_export2(r0, r1 * 1, 1);
        return v2;
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * Request cooperative cancellation of the in-flight recognition: the decode
 * loop observes the flag at its next checkpoint and returns a `Cancelled`
 * error. Call [`reset_cancel`] before the next run.
 */
export function request_cancel() {
    wasm.request_cancel();
}

/**
 * Clear the cancellation flag so the next recognition can run.
 */
export function reset_cancel() {
    wasm.reset_cancel();
}

/**
 * Select GOT-OCR2's `OCR with format:` mode for the NEXT recognition — the
 * browser analog of the CLI's `--format` / `--task formula|tables|chart|…`
 * (which imply it) and of `FOCR_GOT_FORMAT`.
 *
 * This is not a convenience: GOT's whole reason to exist in the zoo is the
 * structured output plain mode cannot produce (LaTeX formulas, HTML tables,
 * molecular SMILES, geometry, `**kern` music). wasm has no environment, so
 * without this export the browser could only ever reach the plain `OCR: `
 * mode. Default `false` ⇒ byte-identical to plain-mode behavior.
 * @param {boolean} on
 */
export function set_got_format(on) {
    wasm.set_got_format(on);
}

/**
 * Set (or clear, with `0`) the sliding no-repeat n-gram decode guard for the
 * NEXT engine built by [`WasmEngine::from_staging`] — the browser analog of
 * `--no-repeat-ngram` / `FOCR_NO_REPEAT_NGRAM`. The README documents a
 * tighter guard (20 vs the default 35) as the mitigation for degenerate
 * repetition loops on hard dense pages; the measured in-browser failure on
 * such a page (an f32-drift-tipped repeat attractor) is the same class. This
 * is a mitigation the native CLI user can apply identically — not a silent
 * numerics change: the site labels the lane's guard setting.
 * @param {number} n
 */
export function set_no_repeat_ngram(n) {
    wasm.set_no_repeat_ngram(n);
}

/**
 * Install (or, with `None`/`undefined`, remove) the live progress callback for
 * the SYNCHRONOUS `recognize` call — the seam that lets the playground show
 * "vision block 12/36" instead of a frozen tab for minutes.
 *
 * The callback is invoked as `f(stage: string, current: number, total: number)`
 * (`total === 0` means indeterminate) from INSIDE the forward's call stack, on
 * the thread that entered `recognize_json`. That is the only way progress can
 * escape at all: the worker is blocked in one synchronous wasm call, so nothing
 * asynchronous can run until it returns.
 *
 * Three rules make this safe and cheap:
 *
 * * **The engine hooks sit on outer, sequential loops only** (per vision block,
 *   per decoded token, never inside a rayon body), so the `js_sys::Function` —
 *   which is emphatically NOT `Send`, whatever the sink signature says — is only
 *   ever called from the JS-owning worker thread. The `Send`/`Sync` promise
 *   below is the wasm single-threaded-JS invariant written down, not a claim
 *   that a `Function` can cross a thread.
 * * **A throwing callback cannot poison a run.** The `Result` is discarded: a
 *   broken progress handler must never be able to fail a recognition.
 * * **Zero cost when unset.** `set_progress_sink(None)` disarms the engine's
 *   relaxed-atomic fast path, and the native build never installs anything.
 * @param {Function | null} [f]
 */
export function set_progress_callback(f) {
    wasm.set_progress_callback(isLikeNone(f) ? 0 : addHeapObject(f));
}

/**
 * Set (or clear, with an empty string) the SmolVLM2 describe/VQA question —
 * the browser analog of `--question` / `FOCR_SMOLVLM2_QUESTION`. Cleared (the
 * default) ⇒ the model-card caption prompt, i.e. plain image description.
 * @param {string} question
 */
export function set_smolvlm2_question(question) {
    const ptr0 = passStringToWasm0(question, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
    const len0 = WASM_VECTOR_LEN;
    wasm.set_smolvlm2_question(ptr0, len0);
}

/**
 * Rayon's *actual* worker count in this module right now.
 *
 * This is the honest test of the threaded lane: the build flags, the presence
 * of `initThreadPool`, and even a `SharedArrayBuffer`-backed memory can all be
 * right while the pool silently fell back to one thread. Serial builds report
 * `1` here by construction.
 *
 * Calling this initializes rayon's global pool if it is not built yet, so JS
 * must call `initThreadPool` FIRST in the threaded module.
 * @returns {number}
 */
export function thread_count() {
    const ret = wasm.thread_count();
    return ret >>> 0;
}

/**
 * The module's `WebAssembly.Memory`, so JS can assert
 * `wasm_memory().buffer instanceof SharedArrayBuffer` after instantiation.
 * Link flags are a claim; this is the receipt.
 * @returns {any}
 */
export function wasm_memory() {
    const ret = wasm.wasm_memory();
    return takeObject(ret);
}

export class wbg_rayon_PoolBuilder {
    static __wrap(ptr) {
        const obj = Object.create(wbg_rayon_PoolBuilder.prototype);
        obj.__wbg_ptr = ptr;
        wbg_rayon_PoolBuilderFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        wbg_rayon_PoolBuilderFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wbg_rayon_poolbuilder_free(ptr, 0);
    }
    build() {
        wasm.wbg_rayon_poolbuilder_build(this.__wbg_ptr);
    }
    /**
     * @returns {string}
     */
    mainJS() {
        const ret = wasm.wbg_rayon_poolbuilder_mainJS(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @returns {number}
     */
    numThreads() {
        const ret = wasm.wbg_rayon_poolbuilder_numThreads(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    receiver() {
        const ret = wasm.wbg_rayon_poolbuilder_receiver(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) wbg_rayon_PoolBuilder.prototype[Symbol.dispose] = wbg_rayon_PoolBuilder.prototype.free;

/**
 * @param {number} receiver
 */
export function wbg_rayon_start_worker(receiver) {
    wasm.wbg_rayon_start_worker(receiver);
}
function __wbg_get_imports(memory) {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_is_undefined_c05833b95a3cf397: function(arg0) {
            const ret = getObject(arg0) === undefined;
            return ret;
        },
        __wbg___wbindgen_memory_de265df8aadd6273: function() {
            const ret = wasm.memory;
            return addHeapObject(ret);
        },
        __wbg___wbindgen_module_a22faa8909381977: function() {
            const ret = wasmModule;
            return addHeapObject(ret);
        },
        __wbg___wbindgen_throw_344f42d3211c4765: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_call_44b7209e1e252e6a: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
            const ret = getObject(arg0).call(getObject(arg1), getObject(arg2), getObject(arg3), getObject(arg4));
            return addHeapObject(ret);
        }, arguments); },
        __wbg_error_5d345173be594081: function(arg0, arg1) {
            console.error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_getRandomValues_127d43fea0fcc894: function() { return handleError(function (arg0) {
            globalThis.crypto.getRandomValues(getObject(arg0));
        }, arguments); },
        __wbg_instanceof_Window_05ba1ee4f6781663: function(arg0) {
            let result;
            try {
                result = getObject(arg0) instanceof Window;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_length_1f0964f4a5e2c6d8: function(arg0) {
            const ret = getObject(arg0).length;
            return ret;
        },
        __wbg_new_with_length_e6785c33c8e4cce8: function(arg0) {
            const ret = new Uint8Array(arg0 >>> 0);
            return addHeapObject(ret);
        },
        __wbg_now_e7c6795a7f81e10f: function(arg0) {
            const ret = getObject(arg0).now();
            return ret;
        },
        __wbg_performance_3fcf6e32a7e1ed0a: function(arg0) {
            const ret = getObject(arg0).performance;
            return addHeapObject(ret);
        },
        __wbg_prototypesetcall_4770620bbe4688a0: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), getObject(arg2));
        },
        __wbg_startWorkers_622cedd0d351664e: function(arg0, arg1, arg2) {
            const ret = startWorkers(takeObject(arg0), takeObject(arg1), wbg_rayon_PoolBuilder.__wrap(arg2));
            return addHeapObject(ret);
        },
        __wbg_static_accessor_GLOBAL_4ef717fb391d88b7: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_8d1badc68b5a74f4: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_static_accessor_SELF_146583524fe1469b: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_static_accessor_URL_151cb8815849ce83: function() {
            const ret = import.meta.url;
            return addHeapObject(ret);
        },
        __wbg_static_accessor_WINDOW_f2829a2234d7819e: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_subarray_3ed232c8a6baee09: function(arg0, arg1, arg2) {
            const ret = getObject(arg0).subarray(arg1 >>> 0, arg2 >>> 0);
            return addHeapObject(ret);
        },
        __wbg_timeOrigin_f3d5cb4f4a06c2b7: function(arg0) {
            const ret = getObject(arg0).timeOrigin;
            return ret;
        },
        __wbindgen_cast_0000000000000001: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return addHeapObject(ret);
        },
        __wbindgen_object_clone_ref: function(arg0) {
            const ret = getObject(arg0);
            return addHeapObject(ret);
        },
        __wbindgen_object_drop_ref: function(arg0) {
            takeObject(arg0);
        },
        memory: memory || new WebAssembly.Memory({initial:30,maximum:65536,shared:true}),
    };
    return {
        __proto__: null,
        "./focr_wasm_bg.js": import0,
    };
}

const ModelStagingFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelstaging_free(ptr, 1));
const WasmEngineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmengine_free(ptr, 1));
const wbg_rayon_PoolBuilderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wbg_rayon_poolbuilder_free(ptr, 1));

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function dropObject(idx) {
    if (idx < 1028) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer !== wasm.memory.buffer) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.buffer !== wasm.memory.buffer) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getObject(idx) { return heap[idx]; }

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        wasm.__wbindgen_export(addHeapObject(e));
    }
}

let heap = new Array(1024).fill(undefined);
heap.push(undefined, null, true, false);

let heap_next = heap.length;

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

let cachedTextDecoder = (typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true }) : undefined);
if (cachedTextDecoder) cachedTextDecoder.decode();

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().slice(ptr, ptr + len));
}

const cachedTextEncoder = (typeof TextEncoder !== 'undefined' ? new TextEncoder() : undefined);

if (cachedTextEncoder) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module, thread_stack_size) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    if (typeof thread_stack_size !== 'undefined' && (typeof thread_stack_size !== 'number' || thread_stack_size === 0 || thread_stack_size % 65536 !== 0)) {
        throw new Error('invalid stack size');
    }

    wasm.__wbindgen_start(thread_stack_size);
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module, memory) {
    if (wasm !== undefined) return wasm;

    let thread_stack_size
    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module, memory, thread_stack_size} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports(memory);
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module, thread_stack_size);
}

async function __wbg_init(module_or_path, memory) {
    if (wasm !== undefined) return wasm;

    let thread_stack_size
    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path, memory, thread_stack_size} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('focr_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports(memory);

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module, thread_stack_size);
}

export { initSync, __wbg_init as default };
