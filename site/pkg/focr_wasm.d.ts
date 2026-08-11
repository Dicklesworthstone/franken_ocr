/* tslint:disable */
/* eslint-disable */

/**
 * Staging area for a model arriving over the network as chunks: exactly one
 * copy of the weight blob lives here, reserved up front, plus the (small)
 * tokenizer sidecars keyed by their canonical zoo filenames.
 */
export class ModelStaging {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Bytes staged so far (for progress display).
     */
    filled(): number;
    /**
     * Free the staged bytes explicitly (a superseded or failed load should
     * not wait for the JS GC's FinalizationRegistry).
     */
    free(): void;
    /**
     * An empty staging area. Call [`Self::reserve`] before the first
     * [`Self::push`].
     */
    constructor();
    /**
     * Append one downloaded chunk. Refuses bytes past the reserved size —
     * a mismatched manifest must fail loudly, not grow silently.
     */
    push(chunk: Uint8Array): void;
    /**
     * Reserve the full weight-blob size up front (`try_reserve_exact`, so a
     * failed reservation names the byte count instead of trapping on an
     * overcommitted grow).
     */
    reserve(total_bytes: number): void;
    /**
     * Attach one tokenizer sidecar by its canonical zoo filename:
     * `tokenizer.json`, `qwen.tiktoken`, or the four TrOMR tables
     * `tokenizer_{rhythm,pitch,lift,note}.json`.
     */
    set_sidecar(name: string, bytes: Uint8Array): void;
}

/**
 * A loaded model plus the recognize entrypoints the playground calls.
 */
export class WasmEngine {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Free the engine (and, if this was the last handle, the weight bytes)
     * explicitly rather than waiting for the JS GC.
     */
    free_engine(): void;
    /**
     * Build the engine from a completed staging area (consumes it — the
     * weight bytes move, they are not copied).
     *
     * Fails if the staging is incomplete, the blob does not parse as a
     * `.focrq`/safetensors container, or the Unlimited-OCR recipe validation
     * rejects the tensor dtypes.
     */
    static from_staging(staging: ModelStaging): WasmEngine;
    /**
     * The model-weights license notice that must travel with the artifact.
     */
    license_notice(): string;
    /**
     * The loaded model's registry id (`unlimited-ocr`, `tromr`, …).
     */
    model_id(): string;
    /**
     * Recognize one encoded image (PNG/JPEG bytes) and return the model's
     * primary text output: markdown for the OCR models, MusicXML for TrOMR.
     */
    recognize(image_bytes: Uint8Array): string;
    /**
     * Recognize one encoded image and return a JSON envelope:
     * `{"model_id", "output", "layout": [{label, boxes}], "music": {...}?}`.
     * `layout` mirrors `focr ocr --json`; `music` carries the TrOMR staff
     * metadata (recognized bboxes, per-staff skips with reasons, annotate-only
     * warnings) when the run produced any.
     */
    recognize_json(image_bytes: Uint8Array): string;
}

/**
 * One JSON object describing this module: crate version, detected ISA tier,
 * and the licenses that must travel with the model weights.
 */
export function engine_info(): string;

/**
 * Install the panic hook. `#[wasm_bindgen(start)]` runs this once at module
 * instantiation, before any exported call.
 */
export function focr_wasm_start(): void;

/**
 * The effective ordinary dense-int8 route on this host. On wasm32 today this
 * reports the scalar tier; when a simd128 kernel island lands it will report
 * that route, and the site asserts whichever it expects — a silent scalar
 * fallback is invisible any other way.
 */
export function int8_route(): string;

/**
 * Request cooperative cancellation of the in-flight recognition: the decode
 * loop observes the flag at its next checkpoint and returns a `Cancelled`
 * error. Call [`reset_cancel`] before the next run.
 */
export function request_cancel(): void;

/**
 * Clear the cancellation flag so the next recognition can run.
 */
export function reset_cancel(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_modelstaging_free: (a: number, b: number) => void;
    readonly __wbg_wasmengine_free: (a: number, b: number) => void;
    readonly engine_info: (a: number) => void;
    readonly int8_route: (a: number) => void;
    readonly modelstaging_filled: (a: number) => number;
    readonly modelstaging_free: (a: number) => void;
    readonly modelstaging_new: () => number;
    readonly modelstaging_push: (a: number, b: number, c: number, d: number) => void;
    readonly modelstaging_reserve: (a: number, b: number, c: number) => void;
    readonly modelstaging_set_sidecar: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly wasmengine_free_engine: (a: number) => void;
    readonly wasmengine_from_staging: (a: number, b: number) => void;
    readonly wasmengine_license_notice: (a: number, b: number) => void;
    readonly wasmengine_model_id: (a: number, b: number) => void;
    readonly wasmengine_recognize: (a: number, b: number, c: number, d: number) => void;
    readonly wasmengine_recognize_json: (a: number, b: number, c: number, d: number) => void;
    readonly focr_wasm_start: () => void;
    readonly request_cancel: () => void;
    readonly reset_cancel: () => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export: (a: number, b: number, c: number) => void;
    readonly __wbindgen_export2: (a: number, b: number) => number;
    readonly __wbindgen_export3: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
