import Foundation
import FocrCore

/// A stage the engine reports progress for. The raw values are exactly the
/// strings the Rust progress hooks emit.
enum RecognitionStage: String {
    case preprocess
    case vision
    case prefill
    case decode
    case postprocess
    case staff

    /// The label a person reads. Matches the site's `STAGE_LABELS`.
    var label: String {
        switch self {
        case .preprocess: "Preparing the image"
        case .vision: "Vision encoder"
        case .prefill: "Reading the prompt"
        case .decode: "Reading the page"
        case .postprocess: "Assembling the result"
        case .staff: "Staff"
        }
    }

    /// Share of the whole run this stage accounts for. Same weights the site's
    /// worker uses, so the phone's bar and the browser's bar move alike.
    var weight: Double {
        switch self {
        case .preprocess: 0.03
        case .vision: 0.42
        case .prefill: 0.05
        case .decode: 0.50
        case .postprocess: 0.0
        case .staff: 0.0
        }
    }

    /// Weight of every stage that completes before this one.
    var precedingWeight: Double {
        switch self {
        case .preprocess: 0.0
        case .vision: 0.03
        case .prefill: 0.45
        case .decode: 0.50
        case .postprocess: 1.0
        case .staff: 0.0
        }
    }
}

struct ProgressUpdate: Sendable {
    let stage: RecognitionStage
    let current: UInt64
    /// 0 means indeterminate — the engine does not know the denominator.
    let total: UInt64
}

/// A recognized page: the model's text output plus its grounded layout.
struct Recognition: Sendable {
    let modelID: String
    /// Markdown for the OCR models, MusicXML for TrOMR.
    let output: String
    let layout: [LayoutSpan]
    let music: MusicMeta?
    /// Present when the default document model routed an extreme-aspect image
    /// through the CLI's same smart-cut horizontal-strip path.
    let tallStripCount: Int?
    /// A successful but suspiciously sparse page result. This is surfaced as
    /// an actionable warning instead of quietly presenting a false success.
    let lowYield: LowYield?

    struct LowYield: Sendable {
        let characters: Int
        let megapixels: Double
    }

    struct LayoutSpan: Sendable, Identifiable {
        let id = UUID()
        let label: String
        /// `[x1, y1, x2, y2]` in source-image pixels.
        let boxes: [[Int]]
    }

    struct MusicMeta: Sendable {
        let staves: [Staff]
        let skips: [Skip]
        let warnings: [Warning]

        struct Staff: Sendable, Identifiable { let id = UUID(); let index: Int; let bbox: [Int] }
        struct Skip: Sendable, Identifiable {
            let id = UUID(); let index: Int; let bbox: [Int]; let reason: String
        }
        struct Warning: Sendable, Identifiable {
            let id = UUID(); let kind: String; let part: String; let measure: Int; let detail: String
        }
    }
}

/// An error that crossed the C boundary, carrying the engine's own exit code so
/// the UI can distinguish "you need to download the model" from "that PDF uses
/// a codec we refuse to guess at" without matching on strings.
struct EngineError: LocalizedError {
    enum Kind: Int32 {
        case generic = 1
        case usage = 2
        case modelNotFound = 3
        case inputDecode = 4
        case timeout = 5
        case cancelled = 6
        case formatMismatch = 7
    }

    let kind: Kind
    let message: String

    var errorDescription: String? { message }
    var isCancellation: Bool { kind == .cancelled }

    /// Whether this ends a whole document walk or just the page that hit it.
    ///
    /// Mirrors `pdf::is_fatal_to_document` in the engine, and must stay in step
    /// with it: a page whose codec has no pure-Rust decoder is survivable, but a
    /// missing model, a cancelled run, or a format mismatch means the run itself
    /// is over. `timeout` is deliberately survivable — the stage budget is
    /// per-forward, so one slow page is a skip, not a verdict on the whole book.
    var isFatalToDocument: Bool {
        switch kind {
        case .modelNotFound, .cancelled, .formatMismatch: true
        case .generic, .usage, .inputDecode, .timeout: false
        }
    }

    /// Read the calling thread's error slot. Must be called on the SAME thread
    /// that made the failing call — inside the actor, that is guaranteed.
    static func fromNative(code: Int32) -> EngineError {
        let message = String(cString: focr_last_error_message())
        return EngineError(
            kind: Kind(rawValue: code) ?? .generic,
            message: message.isEmpty ? "the engine failed without a message" : message
        )
    }
}

/// Build/device facts the engine reports about itself.
struct EngineInfo: Sendable, Decodable {
    let crate_version: String
    /// The ISA tier the hardware advertises.
    let detected_tier: String
    /// What ordinary dense int8 GEMM will ACTUALLY run. Differs from
    /// `detected_tier` on Apple silicon by design; showing only one would
    /// overclaim.
    let dense_route: String
    let threads: Int
    let project_license: String
}

/// All engine access lives here.
///
/// The Rust handle is not thread-safe, and an actor's serialization is the
/// whole safety argument — so no engine call may leave this type. The progress
/// callback is the one thing that legitimately runs off-actor: Rust invokes it
/// from whichever thread is inside the forward, so it hops to the main actor
/// through a `@Sendable` continuation rather than touching actor state.
actor Engine {
    private var handle: OpaquePointer?

    /// Set while a recognition is in flight so the progress trampoline can find
    /// somewhere to deliver. A class box, because the C callback receives an
    /// opaque pointer, not a Swift closure.
    private static let observer = ProgressObserver()

    // ── Static diagnostics (no engine required) ─────────────────────────────

    static func info() -> EngineInfo? {
        let json = String(cString: focr_engine_info_json())
        return try? JSONDecoder().decode(EngineInfo.self, from: Data(json.utf8))
    }

    /// Width of the kernel pool, installing it if needed. Calling this at launch
    /// means the pool's threads get built — and get their QoS class — while the
    /// app is idle rather than in the middle of the first page.
    @discardableResult
    static func warmKernelPool() -> Int { Int(focr_kernel_pool_width()) }

    /// Re-run the dispatched int8 GEMM against the bit-identical scalar oracle
    /// on THIS device. Returns the raw JSON verdict; throws if anything
    /// diverged. This is the proof that the kernels are correct on a phone the
    /// binary was never built on.
    static func selftest() throws -> String {
        var out: UnsafeMutablePointer<CChar>?
        let code = focr_selftest_json(&out)
        defer { if let out { focr_string_free(out) } }
        let json = out.map { String(cString: $0) } ?? ""
        guard code == 0 else { throw EngineError.fromNative(code: code) }
        return json
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────

    var isLoaded: Bool { handle != nil }

    /// Open the artifact at `url`. Idempotent.
    ///
    /// Cheap on iOS: the artifact is memory-mapped, so this validates the
    /// container and returns, and the gigabytes page in lazily during the
    /// first forward instead of being read up front.
    func load(artifact url: URL) throws {
        guard handle == nil else { return }
        let opened = url.path.withCString { focr_engine_open($0) }
        guard let opened else { throw EngineError.fromNative(code: 3) }
        handle = opened
    }

    /// Drop the model and its caches. The next recognition reloads lazily.
    func unload() {
        guard let handle else { return }
        focr_engine_close(handle)
        self.handle = nil
    }

    var modelID: String? {
        guard let handle else { return nil }
        return String(cString: focr_engine_model_id(handle))
    }

    var licenseNotice: String? {
        guard let handle else { return nil }
        return String(cString: focr_engine_license(handle))
    }

    // ── Options ────────────────────────────────────────────────────────────

    func setNoRepeatNgram(_ n: UInt32) { focr_set_no_repeat_ngram(n) }
    func setGotFormat(_ on: Bool) { focr_set_got_format(on) }
    func setSmolVLM2Question(_ question: String) {
        question.withCString { focr_set_smolvlm2_question($0) }
    }

    // ── Recognition ────────────────────────────────────────────────────────

    /// Recognize one encoded image. Long and blocking — minutes for a document
    /// page — which is why it lives behind an actor and why the caller drives a
    /// progress UI from `onProgress`.
    ///
    /// `onProgress` is invoked from the engine's own thread; it is marked
    /// `@Sendable` and should hop to wherever it needs to be.
    func recognize(
        imageData: Data,
        onProgress: (@Sendable (ProgressUpdate) -> Void)? = nil
    ) throws -> Recognition {
        guard let handle else {
            throw EngineError(kind: .modelNotFound, message: "no model is loaded")
        }
        focr_reset_cancel()

        if let onProgress {
            Engine.observer.install(onProgress)
        }
        defer {
            Engine.observer.clear()
        }

        var out: UnsafeMutablePointer<CChar>?
        let code = imageData.withUnsafeBytes { raw -> Int32 in
            let base = raw.bindMemory(to: UInt8.self).baseAddress
            return focr_recognize_json(handle, base, raw.count, &out)
        }
        defer { if let out { focr_string_free(out) } }
        guard code == 0 else { throw EngineError.fromNative(code: code) }
        guard let out else {
            throw EngineError(kind: .generic, message: "the engine returned no result")
        }
        return try Recognition(json: String(cString: out))
    }

    /// Ask the in-flight recognition to stop. It observes the request at its
    /// next checkpoint and throws `.cancelled`.
    nonisolated func requestCancel() { focr_request_cancel() }

    // ── PDF ────────────────────────────────────────────────────────────────

    // PDF work lives on `PdfDocument` below — it is independent of the engine
    // and must stay callable while a recognition is in flight.

    deinit {
        if let handle { focr_engine_close(handle) }
    }
}

/// An opened PDF.
///
/// Parsed once; every page render borrows that parse. The alternative — handing
/// the raw bytes across the boundary per page — re-parses the whole document
/// each time, which is the wrong shape for "OCR this entire scan" and gets
/// quadratically worse as the book gets longer.
///
/// Not an actor: rendering is pure with respect to the document, and the
/// document is immutable once opened. It is a `final class` so `deinit` can
/// close the handle exactly once.
final class PdfDocument: @unchecked Sendable {
    private let handle: OpaquePointer
    let pageCount: Int

    /// Parse off the main actor.
    ///
    /// Parsing is real work — a scanned book is tens of megabytes of object
    /// graph — and every caller here is main-actor isolated, so doing it inline
    /// freezes the UI for as long as it takes.
    static func open(data: Data) async -> PdfDocument? {
        await Task.detached(priority: .userInitiated) { PdfDocument(data: data) }.value
    }

    /// Rasterize off the main actor. Same reason as `open`, per page — and a
    /// document walk does this once for every page in the book.
    func page(_ page: Int) async throws -> Data {
        try await Task.detached(priority: .userInitiated) { try self.renderPage(page) }.value
    }

    /// Returns nil if the bytes are not a PDF this build can parse.
    init?(data: Data) {
        let opened = data.withUnsafeBytes { raw -> OpaquePointer? in
            let base = raw.bindMemory(to: UInt8.self).baseAddress
            return focr_pdf_open(base, raw.count)
        }
        guard let opened else { return nil }
        handle = opened
        pageCount = Int(focr_pdf_page_count(opened))
    }

    /// Rasterize one 1-based page to PNG data.
    ///
    /// Throws rather than returning nil so the caller can record WHY a page was
    /// skipped — "JPXDecode: no pure-Rust decoder" is a materially different
    /// outcome from a corrupt file, and a document walk should say which.
    func renderPage(_ page: Int) throws -> Data {
        var ptr: UnsafeMutablePointer<UInt8>?
        var len = 0
        let code = focr_pdf_render_page(handle, UInt32(page), &ptr, &len)
        guard code == 0, let ptr else { throw EngineError.fromNative(code: code) }
        // Copy out, then hand the buffer back with the EXACT length we were
        // given — the free function reconstitutes a Box from (ptr, len).
        let copied = Data(bytes: ptr, count: len)
        focr_bytes_free(ptr, len)
        return copied
    }

    deinit { focr_pdf_close(handle) }
}

/// Holds the active progress closure and owns the C trampoline.
///
/// This exists because the C callback takes an opaque `void *`, and a Swift
/// closure is not one. Passing an unretained pointer to this box, whose
/// lifetime is the process, avoids every retain/release question at the
/// boundary: the box always outlives the callback.
private final class ProgressObserver: @unchecked Sendable {
    private let lock = NSLock()
    private var sink: (@Sendable (ProgressUpdate) -> Void)?

    func install(_ sink: @escaping @Sendable (ProgressUpdate) -> Void) {
        lock.lock()
        self.sink = sink
        lock.unlock()
        let ctx = Unmanaged.passUnretained(self).toOpaque()
        focr_set_progress_callback(progressTrampoline, ctx)
    }

    func clear() {
        // Clear the native side FIRST, so no in-flight call can find a sink
        // we are about to drop.
        focr_set_progress_callback(nil, nil)
        lock.lock()
        sink = nil
        lock.unlock()
    }

    fileprivate func deliver(stage: String, current: UInt64, total: UInt64) {
        guard let stage = RecognitionStage(rawValue: stage) else { return }
        lock.lock()
        let sink = self.sink
        lock.unlock()
        sink?(ProgressUpdate(stage: stage, current: current, total: total))
    }
}

/// The C function pointer the engine calls. Runs on the forward's thread.
private func progressTrampoline(
    ctx: UnsafeMutableRawPointer?,
    stage: UnsafePointer<CChar>?,
    current: UInt64,
    total: UInt64
) {
    guard let ctx, let stage else { return }
    let observer = Unmanaged<ProgressObserver>.fromOpaque(ctx).takeUnretainedValue()
    observer.deliver(stage: String(cString: stage), current: current, total: total)
}

// ── Envelope decoding ──────────────────────────────────────────────────────

private extension Recognition {
    init(json: String) throws {
        guard let data = json.data(using: .utf8),
              let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw EngineError(kind: .generic, message: "the engine returned malformed JSON")
        }
        modelID = root["model_id"] as? String ?? "unknown"
        output = root["output"] as? String ?? ""
        layout = (root["layout"] as? [[String: Any]] ?? []).map { span in
            LayoutSpan(
                label: span["label"] as? String ?? "",
                boxes: (span["boxes"] as? [[Int]]) ?? []
            )
        }
        tallStripCount = root["tall_strip_count"] as? Int
        if let warning = root["low_yield"] as? [String: Any] {
            lowYield = LowYield(
                characters: warning["yield_chars"] as? Int ?? 0,
                megapixels: warning["input_megapixels"] as? Double ?? 0
            )
        } else {
            lowYield = nil
        }
        if let m = root["music"] as? [String: Any] {
            music = MusicMeta(
                staves: (m["staves"] as? [[String: Any]] ?? []).map {
                    MusicMeta.Staff(index: $0["index"] as? Int ?? 0,
                                    bbox: $0["bbox"] as? [Int] ?? [])
                },
                skips: (m["skips"] as? [[String: Any]] ?? []).map {
                    MusicMeta.Skip(index: $0["index"] as? Int ?? 0,
                                   bbox: $0["bbox"] as? [Int] ?? [],
                                   reason: $0["reason"] as? String ?? "unknown")
                },
                warnings: (m["warnings"] as? [[String: Any]] ?? []).map {
                    MusicMeta.Warning(kind: $0["kind"] as? String ?? "",
                                      part: $0["part"] as? String ?? "",
                                      measure: $0["measure"] as? Int ?? 0,
                                      detail: $0["detail"] as? String ?? "")
                }
            )
        } else {
            music = nil
        }
    }
}
