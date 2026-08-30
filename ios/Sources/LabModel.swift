import Foundation
import SwiftUI
import UIKit

/// Adaptive recognition forecast keyed by model and stored in this device's app
/// sandbox. Broad model-class lookup values seed a first run; actual megapixel and
/// page throughput steadily replaces them. It intentionally publishes nothing until
/// the engine has completed a measurable piece of its first stage.
private struct OCRAdaptiveETA {
    private static let version = "v1"

    private var modelID = "unlimited-ocr"
    private var beganWarm = true
    private var pageCount = 1
    private var inputMegapixels = 1.0
    private var predictedFinishElapsed: TimeInterval?
    private var hasMeasuredWork = false
    private var completedPageSeconds: [Double] = []
    private var completedMegapixels = 0.0

    mutating func reset(modelID: String, pageCount: Int, inputMegapixels: Double) {
        self.modelID = modelID
        self.pageCount = max(1, pageCount)
        self.inputMegapixels = max(0.1, inputMegapixels)
        beganWarm = true
        predictedFinishElapsed = nil
        hasMeasuredWork = false
        completedPageSeconds.removeAll(keepingCapacity: true)
        completedMegapixels = 0
    }

    mutating func setBeganWarm(_ warm: Bool) {
        beganWarm = warm
    }

    mutating func observe(
        progress: ProgressUpdate?,
        overallFraction: Double,
        elapsed: TimeInterval
    ) {
        guard let progress,
              progress.current > 0,
              overallFraction >= 0.02,
              elapsed > 0
        else { return }

        hasMeasuredWork = true
        let fraction = min(0.985, max(0.02, overallFraction))
        let observedTotal = elapsed / fraction
        let priorTotal = coldPriorTotal

        // Once a document has completed a page, its own page cadence is a better
        // predictor than either a global lookup or an early token callback.
        let completedProjection: Double?
        if !completedPageSeconds.isEmpty {
            let average = completedPageSeconds.reduce(0, +) / Double(completedPageSeconds.count)
            completedProjection = elapsed
                + average * Double(max(0, pageCount - completedPageSeconds.count))
        } else {
            completedProjection = nil
        }

        let observedWeight = min(0.90, 0.32 + fraction * 0.58)
        var candidate = priorTotal * (1 - observedWeight) + observedTotal * observedWeight
        if let completedProjection {
            candidate = candidate * 0.35 + completedProjection * 0.65
        }
        candidate = max(elapsed + 0.75, candidate)

        if let old = predictedFinishElapsed {
            let smoothed = old * 0.70 + candidate * 0.30
            predictedFinishElapsed = min(smoothed, old + 4.0)
        } else {
            predictedFinishElapsed = candidate
        }
    }

    mutating func recordCompletedPage(seconds: Double, megapixels: Double) {
        guard seconds.isFinite, seconds > 0 else { return }
        completedPageSeconds.append(seconds)
        let mp = max(0.1, megapixels)
        completedMegapixels += mp
        Self.update(key: secondsPerPageKey, sample: seconds)
        Self.update(key: secondsPerMegapixelKey, sample: seconds / mp)
    }

    func remainingSeconds(at elapsed: TimeInterval) -> Int? {
        guard hasMeasuredWork, let predictedFinishElapsed else { return nil }
        return max(1, Int(ceil(predictedFinishElapsed - elapsed)))
    }

    mutating func finish(totalElapsed: TimeInterval) {
        guard totalElapsed.isFinite, totalElapsed > 0 else { return }
        if pageCount == 1 {
            Self.update(key: secondsPerPageKey, sample: totalElapsed)
            Self.update(
                key: secondsPerMegapixelKey,
                sample: totalElapsed / inputMegapixels
            )
        } else if !completedPageSeconds.isEmpty {
            Self.update(
                key: secondsPerPageKey,
                sample: totalElapsed / Double(completedPageSeconds.count)
            )
            if completedMegapixels > 0 {
                Self.update(
                    key: secondsPerMegapixelKey,
                    sample: totalElapsed / completedMegapixels
                )
            }
        }
        predictedFinishElapsed = totalElapsed
    }

    private var coldPriorTotal: Double {
        let loadTail = beganWarm ? 0.0 : 3.0
        if pageCount > 1 {
            return loadTail + learnedSecondsPerPage * Double(pageCount)
        }
        return loadTail + learnedSecondsPerMegapixel * inputMegapixels
    }

    private var secondsPerPageKey: String {
        "ocr.eta.\(Self.version).\(modelID).secondsPerPage.\(beganWarm ? "warm" : "cold")"
    }

    private var secondsPerMegapixelKey: String {
        "ocr.eta.\(Self.version).\(modelID).secondsPerMegapixel.\(beganWarm ? "warm" : "cold")"
    }

    private var learnedSecondsPerPage: Double {
        let learned = UserDefaults.standard.double(forKey: secondsPerPageKey)
        if learned > 0 { return learned }
        switch modelID {
        case "unlimited-ocr": return 42
        case "got-ocr2": return 30
        case "tromr": return 24
        case "onechart": return 20
        case "smolvlm2": return 14
        default: return 32
        }
    }

    private var learnedSecondsPerMegapixel: Double {
        let learned = UserDefaults.standard.double(forKey: secondsPerMegapixelKey)
        if learned > 0 { return learned }
        switch modelID {
        case "unlimited-ocr": return 13
        case "got-ocr2": return 9
        case "tromr": return 8
        case "onechart": return 7
        case "smolvlm2": return 4.5
        default: return 10
        }
    }

    private static func update(key: String, sample: Double) {
        guard sample.isFinite, sample > 0 else { return }
        let defaults = UserDefaults.standard
        let old = defaults.double(forKey: key)
        defaults.set(old > 0 ? old * 0.72 + sample * 0.28 : sample, forKey: key)
    }
}

/// What happened to one page of a document.
///
/// A page that cannot be rasterized (JPEG 2000, JBIG2, a born-digital vector
/// page) must not take the whole document down with it — the CLI skips it with a
/// named reason and keeps going, and so does this.
struct PageOutcome: Identifiable {
    enum State {
        case queued
        case running
        case done(characters: Int, seconds: Double)
        case skipped(reason: String)
    }

    /// 1-based source page number, which is also the identity a person sees.
    let id: Int
    var state: State = .queued
    var text: String = ""

    var isTerminal: Bool {
        switch state {
        case .done, .skipped: true
        case .queued, .running: false
        }
    }
}

/// The one place app state lives.
///
/// Everything UI-facing is main-actor and `@Observable`; the single thread
/// transition in the whole app is the hop into the `Engine` actor, which is
/// where the long blocking forward runs.
@MainActor
@Observable
final class LabModel {
    // ── Engine + model ─────────────────────────────────────────────────────
    let engine = Engine()
    let store = ModelStore()

    private(set) var spec: ModelSpec = ModelCatalog.all[0] {
        didSet {
            guard spec.id != oldValue.id else { return }
            // A different model means a different engine. Drop the old one
            // before anything touches the new artifact.
            let lifecycleToken = nextEngineLifecycleToken()
            Task { await engine.unload(lifecycleToken: lifecycleToken) }
            store.refreshInstalledState(spec)
            recognition = nil
            statusKind = .neutral
            status = defaultStatus
            UserDefaults.standard.set(spec.id, forKey: "selectedModel")
        }
    }

    var info: EngineInfo?
    var licenseNotice: String?

    // ── Input ──────────────────────────────────────────────────────────────
    /// The image bytes actually handed to the engine (PNG/JPEG).
    var imageData: Data?
    var previewImage: UIImage?
    var imageName: String?

    /// Loaded PDF, if the input was a document. Parsed once and reused for
    /// every page render.
    var pdf: PdfDocument?
    var pdfPageCount: Int { pdf?.pageCount ?? 0 }
    /// The page shown in the preview. Recognition covers the whole selection,
    /// not just this page — this only drives what you are looking at.
    var previewPage: Int = 1
    /// CLI-style page selection (`3,5-9`). Empty means every page.
    var pageSelection: String = ""

    /// One row per page in the current run.
    var pageOutcomes: [PageOutcome] = []

    /// SmolVLM2's question. Kept even when another model is selected so it
    /// survives a round trip through the picker.
    var question: String = ""

    /// GOT-OCR2's `OCR with format:` mode. This is the whole reason GOT is in
    /// the picker — in plain mode it produces roughly what the default model
    /// does, only slower. On for that lane by default.
    var gotFormat: Bool = true

    // ── Run state ──────────────────────────────────────────────────────────
    var isRecognizing = false
    var isLoadingModel = false
    var isClearingModel = false
    var elapsed: TimeInterval = 0
    var estimatedRemainingSeconds: Int?
    var progress: ProgressUpdate?
    var recognition: Recognition?
    var lastRunSeconds: Double?

    var status: String = ""
    var statusKind: StatusLine.Kind = .neutral

    var showConsent = false
    var showSelftest = false
    var selftestJSON: String?
    var showLayoutBoxes = false
    var viewSource = false

    /// Index into `pageOutcomes` of the page being read, if any.
    var currentPageIndex: Int?

    private var timer: Timer?
    private var runStartedAt: Date?
    private var recognizeTask: Task<Void, Never>?
    private let runActivity = OCRActivityController.shared
    /// Guards against a stale run publishing over a newer one.
    private var generation = 0
    private var engineLifecycleToken: UInt64 = 0
    private var inputGeneration = 0
    private var previewGeneration = 0
    private var eta = OCRAdaptiveETA()

    init() {
        if let saved = UserDefaults.standard.string(forKey: "selectedModel"),
           let found = ModelCatalog.spec(id: saved) {
            spec = found
        }
        status = defaultStatus
        info = Engine.info()
        store.refreshInstalledState(spec)
    }

    private var defaultStatus: String {
        spec.isSupportedOnThisDevice
            ? "Ready to load \(spec.shortName)."
            : "\(spec.shortName) needs a device with more memory than this one reports."
    }

    // ── Derived UI state ───────────────────────────────────────────────────

    var isInstalled: Bool { store.phase == .ready }

    var canRecognize: Bool {
        guard isInstalled, !isRecognizing, !isClearingModel,
              spec.isSupportedOnThisDevice else { return false }
        // A document is runnable even if its FIRST page failed to preview — the
        // other pages may be perfectly readable. Deliberately does NOT resolve
        // the page selection: this is read on every SwiftUI body evaluation, and
        // an invalid spec is reported when Recognize is pressed, not by silently
        // disabling the button with no explanation.
        return imageData != nil || pdf != nil
    }

    /// A monotonic, honest progress fraction.
    ///
    /// Capped below 1 on purpose: only a returned result means done. Where the
    /// engine reports a real denominator (vision blocks, staves) this is
    /// measured; the decode stage has no knowable denominator — EOS is not
    /// predictable — so it is estimated against a per-model token count and the
    /// UI says "estimated" rather than pretending otherwise.
    var progressFraction: Double {
        guard let progress else { return 0 }
        let stage = progress.stage
        let within = min(1, Double(progress.current) / Double(denominator(for: progress)))
        return min(0.99, stage.precedingWeight + within * stage.weight)
    }

    /// How many units this stage is expected to take.
    ///
    /// The vision and staff stages report a real denominator (blocks, staves)
    /// and it is used as-is. Decode does not: the engine reports the max-token
    /// CAP (32768), not an expectation, because EOS is not predictable. Dividing
    /// by the cap pins decode near zero — and decode is half the bar — so a page
    /// would look frozen for its entire second half. Fall back to the per-model
    /// token estimate, the same one the browser lane uses, and label the result
    /// estimated rather than pretending it is measured.
    private func denominator(for progress: ProgressUpdate) -> Int {
        let reported = Int(progress.total)
        guard progress.stage == .decode else {
            return reported > 0 ? reported : estimatedTokens
        }
        // A cap-sized "total" is not a prediction. Only trust it when it is
        // tighter than our own estimate (a caller-set --max-length below it).
        if reported > 0, reported < estimatedTokens { return reported }
        return estimatedTokens
    }

    /// Progress across the WHOLE document: pages already finished, plus how far
    /// into the current page we are. Without the second term a 40-page run would
    /// sit frozen for minutes at a time, which reads as a hang.
    var documentProgressFraction: Double {
        let total = pageOutcomes.count
        guard total > 0 else { return progressFraction }
        let finished = pageOutcomes.filter(\.isTerminal).count
        let current = currentPageIndex != nil ? progressFraction : 0
        return min(0.999, (Double(finished) + current) / Double(total))
    }

    /// True when this run is a document walk rather than a single image.
    var isDocumentRun: Bool { !pageOutcomes.isEmpty }

    /// True when the bar is extrapolating rather than counting — decode always
    /// is, because the number of tokens a page will produce is not knowable
    /// until EOS arrives.
    var progressIsEstimated: Bool {
        guard let progress else { return false }
        return progress.total == 0 || progress.stage == .decode
    }

    private var estimatedTokens: Int {
        switch spec.id {
        case "unlimited-ocr": 1600
        case "tromr": 300
        default: 700
        }
    }

    var progressDetail: String {
        guard let progress else {
            return isDocumentRun ? "Rendering page…" : ""
        }
        let qualifier = progressIsEstimated ? " (estimated)" : ""
        // On a document run the leading percentage is the DOCUMENT's, because
        // that is the number a person actually wants; the page's own stage
        // detail follows it.
        let headline = isDocumentRun
            ? "\(Int(documentProgressFraction * 100))%\(qualifier)"
            : "\(Int(progressFraction * 100))%\(qualifier)"
        var detail = "\(headline) · \(progress.stage.label)"
        if progress.stage == .decode {
            // Show the honest count, not a fraction of a cap nobody will reach.
            if progress.current > 0 { detail += " · \(progress.current) tokens" }
        } else if progress.total > 0 {
            detail += " \(progress.current)/\(progress.total)"
        } else if progress.current > 0 {
            detail += " · \(progress.current)"
        }
        return detail
    }

    /// "Page 7 of 40 · 6 done · 1 skipped" — the document-level line.
    var documentDetail: String {
        guard isDocumentRun else { return "" }
        let total = pageOutcomes.count
        let done = completedPageCount
        let skipped = pageOutcomes.filter { if case .skipped = $0.state { true } else { false } }.count
        var parts: [String] = []
        if let index = currentPageIndex {
            let sourcePage = pageOutcomes[index].id
            if total == pdfPageCount, sourcePage == index + 1 {
                parts.append("Page \(sourcePage) of \(total)")
            } else {
                parts.append("Source page \(sourcePage) of \(pdfPageCount)")
                parts.append("\(index + 1) of \(total) selected")
            }
        } else {
            parts.append("\(total) page\(total == 1 ? "" : "s")")
        }
        parts.append("\(done) done")
        if skipped > 0 { parts.append("\(skipped) skipped") }
        return parts.joined(separator: " · ")
    }

    // ── Model lifecycle ────────────────────────────────────────────────────

    func requestDownload() { showConsent = true }

    func selectModel(_ candidate: ModelSpec) {
        guard !isRecognizing, !isClearingModel, !store.isBusy else {
            status = "Finish or cancel the current model operation before switching models."
            statusKind = .warn
            return
        }
        spec = candidate
    }

    func confirmDownload() {
        showConsent = false
        store.download(spec)
    }

    func clearModel() {
        guard !isRecognizing, !isClearingModel else { return }
        isClearingModel = true
        let clearingSpec = spec
        let lifecycleToken = nextEngineLifecycleToken()
        Task {
            await engine.unload(lifecycleToken: lifecycleToken)
            await store.clear(clearingSpec)
            if case .failed(let message) = store.phase {
                status = message
                statusKind = .err
            } else {
                recognition = nil
                status = "Model removed. \(clearingSpec.weights.bytes.humanBytes) freed."
                statusKind = .neutral
            }
            isClearingModel = false
        }
    }

    /// Drop the engine under memory pressure or on backgrounding. A no-op mid
    /// recognition — tearing the model out from under a running forward would
    /// turn a slow page into a crash.
    func releaseEngineIfIdle() {
        guard !isRecognizing, !isClearingModel else { return }
        let lifecycleToken = nextEngineLifecycleToken()
        Task { await engine.unload(lifecycleToken: lifecycleToken) }
    }

    // ── Input ──────────────────────────────────────────────────────────────

    /// Take a dropped/picked file.
    ///
    /// `async` because a PDF has to be parsed off the main actor, and callers
    /// need a completion signal: returning before `pdf` is set would leave a
    /// fast "pick then Recognize" tap doing nothing at all.
    func accept(data: Data, name: String) async {
        guard !isRecognizing else {
            status = "Stop the current recognition before choosing another input."
            statusKind = .warn
            return
        }
        inputGeneration &+= 1
        let importGeneration = inputGeneration
        if data.starts(with: Array("%PDF".utf8)) {
            await acceptPDF(data, name: name, generation: importGeneration)
        } else {
            guard importGeneration == inputGeneration else { return }
            acceptImage(data, name: name)
        }
    }

    /// Bring a Live Camera capture back into the ordinary page/transcription
    /// workspace. Live mode can accumulate lines across several viewpoints, so
    /// its text intentionally carries no layout boxes tied to the final frame.
    func adoptLiveCameraCapture(text: String, snapshot: Data?) {
        let output = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !output.isEmpty else { return }
        inputGeneration &+= 1
        previewGeneration &+= 1
        pdf = nil
        pageOutcomes = []
        imageData = snapshot
        previewImage = snapshot.flatMap(UIImage.init(data:))
        imageName = "live-camera.jpg"
        recognition = Recognition(
            modelID: "apple-vision-live",
            output: output,
            layout: [],
            music: nil,
            tallStripCount: nil,
            lowYield: nil
        )
        lastRunSeconds = nil
        statusKind = .ok
        status = "Captured \(output.count) characters live, entirely on this device."
    }

    private func acceptImage(_ data: Data, name: String) {
        guard let image = UIImage(data: data) else {
            status = "That file is not a PNG or JPEG this app can read."
            statusKind = .warn
            return
        }
        pdf = nil
        pageOutcomes = []
        imageData = data
        previewImage = image
        imageName = name
        recognition = nil
        status = "\(name) · \(Int(image.size.width))×\(Int(image.size.height))"
        statusKind = .neutral
    }

    private func acceptPDF(_ data: Data, name: String, generation: Int) async {
        // Parsing and rasterizing are off the main actor: this type is
        // main-actor isolated, and a scanned book is enough object graph that
        // doing either inline visibly freezes the UI.
        status = "Opening \(name)…"
        statusKind = .neutral
        guard let document = await PdfDocument.open(data: data) else {
            guard generation == inputGeneration else { return }
            status = "That PDF could not be parsed."
            statusKind = .err
            return
        }
        guard generation == inputGeneration, !isRecognizing else { return }
        pdf = document
        previewPage = 1
        pageSelection = ""
        imageName = name
        recognition = nil
        pageOutcomes = []
        let pages = document.pageCount
        let previewLoaded = await loadPreviewPage()
        guard generation == inputGeneration else { return }
        if previewLoaded {
            status = "\(name) · \(pages) page\(pages == 1 ? "" : "s"). "
                + "Recognize reads the whole document."
            statusKind = .neutral
        }
    }

    /// Render the page the user is looking at. Purely a preview — it does not
    /// constrain what `recognize()` covers.
    @discardableResult
    func loadPreviewPage() async -> Bool {
        guard let pdf else { return false }
        previewGeneration &+= 1
        let requestGeneration = previewGeneration
        let requestedPage = previewPage
        do {
            let png = try await pdf.page(requestedPage)
            guard requestGeneration == previewGeneration,
                  self.pdf === pdf,
                  previewPage == requestedPage,
                  !isRecognizing
            else { return false }
            imageData = png
            previewImage = UIImage(data: png)
            return true
        } catch {
            guard requestGeneration == previewGeneration,
                  self.pdf === pdf,
                  previewPage == requestedPage,
                  !isRecognizing
            else { return false }
            // The engine names exactly what was unsupported (JPEG 2000, JBIG2,
            // a born-digital vector page) rather than returning a wrong result.
            imageData = nil
            previewImage = nil
            status = error.localizedDescription
            statusKind = .warn
            return false
        }
    }

    /// A malformed or out-of-range page selection.
    struct PageSpecError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    private struct RunInputError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// Resolve `pageSelection` into 1-based page numbers in source order with
    /// duplicates removed. Empty selection means the whole document.
    ///
    /// Mirrors `pdf::select_pages` in the engine, including its refusals:
    /// out-of-range and malformed pages **throw, naming the real page count**,
    /// rather than being quietly dropped. Silently ignoring "9" on a 3-page
    /// document leaves the reader believing page 9 was read.
    func selectedPages() throws -> [Int] {
        let total = pdfPageCount
        guard total > 0 else { return [] }
        let spec = pageSelection.trimmingCharacters(in: .whitespaces)
        guard !spec.isEmpty else { return Array(1...total) }

        func bad(_ what: String) -> PageSpecError {
            PageSpecError(
                message: "Pages \"\(spec)\": \(what) (expected 1-based pages/ranges "
                    + "like \"1,5-9\"; this document has \(total) page(s))"
            )
        }
        func one(_ token: Substring) throws -> Int {
            let text = token.trimmingCharacters(in: .whitespaces)
            guard let n = Int(text), String(n) == text else {
                throw bad("unparseable page \"\(text)\"")
            }
            if n == 0 { throw bad("page 0 (pages are 1-based)") }
            if n > total { throw bad("page \(n) is out of range") }
            return n
        }

        var wanted: Set<Int> = []
        for part in spec.split(separator: ",", omittingEmptySubsequences: false) {
            let piece = part.trimmingCharacters(in: .whitespaces)
            guard !piece.isEmpty else { throw bad("empty element") }
            if let dash = piece.firstIndex(of: "-"), dash != piece.startIndex {
                let lo = try one(piece[piece.startIndex..<dash])
                let hi = try one(piece[piece.index(after: dash)...])
                guard lo <= hi else { throw bad("reversed range \"\(piece)\"") }
                for p in lo...hi { wanted.insert(p) }
            } else {
                wanted.insert(try one(piece[...]))
            }
        }
        return wanted.sorted()
    }

    // ── Recognition ────────────────────────────────────────────────────────

    func recognize() {
        guard canRecognize else { return }
        previewGeneration &+= 1
        generation += 1
        let runGeneration = generation
        let lifecycleToken = nextEngineLifecycleToken()
        isRecognizing = true
        recognition = nil
        progress = nil
        elapsed = 0
        estimatedRemainingSeconds = nil
        // Cleared up front: it is only assigned on successful completion, so
        // a cancelled or mid-flight run would otherwise show — and export —
        // the PREVIOUS run's timing as if it belonged to this result.
        lastRunSeconds = nil
        statusKind = .neutral
        status = "Working…"

        // Seed the per-page rows up front so the whole plan is visible from the
        // first second: you can see it is going to do 40 pages before it starts.
        // A bad page spec is a usage error reported here, before any work.
        let pages: [Int]
        if pdf != nil {
            do {
                pages = try selectedPages()
            } catch {
                isRecognizing = false
                statusKind = .warn
                status = error.localizedDescription
                return
            }
            if pages.isEmpty {
                isRecognizing = false
                statusKind = .warn
                status = "That page selection matched no pages."
                return
            }
        } else {
            pages = []
        }
        pageOutcomes = pages.map { PageOutcome(id: $0) }
        eta.reset(
            modelID: spec.id,
            pageCount: max(1, pages.count),
            inputMegapixels: pixelMegapixels(previewImage)
        )
        runActivity.begin()

        // A minutes-long run must not be interrupted by the screen sleeping,
        // which suspends the app. A whole document is much longer still.
        UIApplication.shared.isIdleTimerDisabled = true
        let wallClockStarted = Date()
        runStartedAt = wallClockStarted
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.elapsed = Date().timeIntervalSince(wallClockStarted)
                self.refreshETA()
            }
        }

        recognizeTask = Task { [weak self] in
            guard let self else { return }
            defer {
                UIApplication.shared.isIdleTimerDisabled = false
                self.timer?.invalidate()
                self.timer = nil
                self.isRecognizing = false
                self.recognizeTask = nil
                self.progress = nil
                self.estimatedRemainingSeconds = nil
                self.runStartedAt = nil
            }
            do {
                let beganWarm = await self.engine.isLoaded
                self.eta.setBeganWarm(beganWarm)
                try await self.ensureEngineLoaded(lifecycleToken: lifecycleToken)
                // Per-RUN options. These are process-global setters that apply
                // to the next recognition, so applying them only at engine load
                // would silently ignore a toggle flipped afterwards.
                await self.applyPerRunOptions()
                if self.pdf != nil {
                    try await self.recognizeDocument(generation: runGeneration)
                } else {
                    try await self.recognizeSingleImage(generation: runGeneration)
                }
            } catch let error as EngineError where error.isCancellation {
                guard runGeneration == self.generation else { return }
                self.statusKind = .warn
                self.status = self.pageOutcomes.isEmpty
                    ? "Cancelled."
                    : "Cancelled after \(self.completedPageCount) of \(self.pageOutcomes.count) pages."
                self.runActivity.finish(
                    status: .cancelled,
                    headline: "Recognition stopped",
                    detail: self.status
                )
            } catch {
                guard runGeneration == self.generation else { return }
                self.statusKind = .err
                self.status = error.localizedDescription
                self.runActivity.finish(
                    status: .failed,
                    headline: "Vision reactor interrupted",
                    detail: "Open FrankenOCR to retry"
                )
            }
        }
    }

    /// The plain single-image path.
    private func recognizeSingleImage(generation runGeneration: Int) async throws {
        guard let imageData else {
            throw RunInputError(message: "The selected image is no longer available. Choose it again and retry.")
        }
        let started = Date()
        let result = try await engine.recognize(imageData: imageData) { [weak self] update in
            Task { @MainActor [weak self] in
                guard let self, runGeneration == self.generation else { return }
                self.progress = update
                self.refreshETA()
                self.publishActivity(update)
            }
        }
        guard runGeneration == generation else { return }
        let engineSeconds = Date().timeIntervalSince(started)
        let seconds = currentRunElapsed(fallback: engineSeconds)
        elapsed = seconds
        eta.finish(totalElapsed: seconds)
        estimatedRemainingSeconds = nil
        lastRunSeconds = seconds
        recognition = result
        if let warning = result.lowYield {
            statusKind = .warn
            status = String(
                format: "Low-yield result: %.2f MP produced only %d text characters. Try a sharper or higher-resolution capture.",
                warning.megapixels, warning.characters
            )
        } else {
            statusKind = .ok
            let route = result.tallStripCount.map { " · \($0) smart strips" } ?? ""
            status = String(
                format: "Done in %.1fs · %d characters%@, entirely on this device.",
                seconds, result.output.count, route
            )
        }
        runActivity.finish(
            status: .complete,
            headline: "Text assembled",
            detail: "\(result.output.count) characters ready to read and export"
        )
    }

    /// Walk every selected page of the document.
    ///
    /// Sequential on purpose: the engine admits ONE forward at a time (each page
    /// already fans out across every core internally), so running pages
    /// concurrently would not be faster and would multiply peak memory by the
    /// number of pages in flight — on a phone, the fastest way to get killed.
    ///
    /// A page that fails to rasterize is recorded with its reason and the walk
    /// continues, matching the CLI: one JPEG-2000 page in a 300-page scan must
    /// not throw away the other 299.
    private func recognizeDocument(generation runGeneration: Int) async throws {
        guard let pdf else {
            throw RunInputError(message: "The selected document is no longer available. Choose it again and retry.")
        }
        let started = Date()

        for (index, outcome) in pageOutcomes.enumerated() {
            guard runGeneration == generation else { return }
            try Task.checkCancellation()
            let page = outcome.id

            pageOutcomes[index].state = .running
            currentPageIndex = index
            status = "Page \(page) of \(pdfPageCount) — \(completedPageCount) done"

            // Render, off the main actor. A failure here is this page's
            // problem, not the run's.
            let png: Data
            do {
                png = try await pdf.page(page)
            } catch {
                pageOutcomes[index].state = .skipped(reason: error.localizedDescription)
                continue
            }

            // Show the page currently being read.
            previewPage = page
            previewImage = UIImage(data: png)
            let pageMegapixels = pixelMegapixels(previewImage)

            let pageStarted = Date()
            do {
                let result = try await engine.recognize(imageData: png) { [weak self] update in
                    Task { @MainActor [weak self] in
                        guard let self, runGeneration == self.generation else { return }
                        self.progress = update
                        self.refreshETA()
                        self.publishActivity(update)
                    }
                }
                guard runGeneration == generation else { return }
                let pageSeconds = Date().timeIntervalSince(pageStarted)
                pageOutcomes[index].text = result.output
                pageOutcomes[index].state = .done(
                    characters: result.output.count,
                    seconds: pageSeconds
                )
                eta.recordCompletedPage(seconds: pageSeconds, megapixels: pageMegapixels)
                // Keep the most recent page's layout for the box overlay.
                recognition = result
            } catch let error as EngineError where error.isFatalToDocument {
                // Mirrors `pdf::is_fatal_to_document`. A missing model or a
                // format mismatch is the DOCUMENT's problem: skipping 300 pages
                // one at a time would be a slow, confusing way to report that
                // there was never a model loaded.
                throw error
            } catch {
                // Anything else — an undecodable page codec, a bad raster — is
                // this page's problem, and the other pages still deserve to run.
                pageOutcomes[index].state = .skipped(reason: error.localizedDescription)
            }
            progress = nil
        }

        guard runGeneration == generation else { return }
        currentPageIndex = nil
        let engineSeconds = Date().timeIntervalSince(started)
        let seconds = currentRunElapsed(fallback: engineSeconds)
        elapsed = seconds
        eta.finish(totalElapsed: seconds)
        estimatedRemainingSeconds = nil
        lastRunSeconds = seconds
        let done = completedPageCount
        let skipped = pageOutcomes.filter { if case .skipped = $0.state { true } else { false } }.count
        if done == 0 {
            statusKind = .err
            status = String(
                format: "No pages could be recognized in %.0fs · %d skipped. Review the page errors and try another model or source file.",
                seconds, skipped
            )
            runActivity.finish(
                status: .failed,
                headline: "No pages recognized",
                detail: "Open FrankenOCR to review \(skipped) page error\(skipped == 1 ? "" : "s")"
            )
            return
        }

        statusKind = skipped == 0 ? .ok : .warn
        status = String(
            format: "%d of %d pages in %.0fs%@, entirely on this device.",
            done, pageOutcomes.count, seconds,
            skipped == 0 ? "" : " · \(skipped) skipped"
        )
        runActivity.finish(
            status: .complete,
            headline: "Document assembled",
            detail: "\(done) pages ready to read and export"
        )
    }

    private func publishActivity(_ update: ProgressUpdate) {
        // Activity progress is the position within this run, not the source PDF
        // page number. A selection such as 5-9 must read 1/5 through 5/5 rather
        // than producing impossible labels such as "Page 6 of 5".
        let page = currentPageIndex.map { $0 + 1 } ?? (isDocumentRun ? 1 : 0)
        runActivity.update(
            progress: update,
            page: page,
            pageCount: pageOutcomes.count,
            elapsed: elapsed,
            totalIsEstimated: update.total == 0 || update.stage == .decode
        )
    }

    private func refreshETA() {
        let fraction = isDocumentRun ? documentProgressFraction : progressFraction
        eta.observe(progress: progress, overallFraction: fraction, elapsed: elapsed)
        estimatedRemainingSeconds = eta.remainingSeconds(at: elapsed)
    }

    private func currentRunElapsed(fallback: TimeInterval) -> TimeInterval {
        runStartedAt.map { Date().timeIntervalSince($0) } ?? fallback
    }

    private func pixelMegapixels(_ image: UIImage?) -> Double {
        guard let image else { return 1 }
        if let cgImage = image.cgImage {
            return max(0.1, Double(cgImage.width * cgImage.height) / 1_000_000)
        }
        return max(
            0.1,
            Double(image.size.width * image.scale * image.size.height * image.scale)
                / 1_000_000
        )
    }

    var completedPageCount: Int {
        pageOutcomes.filter { if case .done = $0.state { true } else { false } }.count
    }

    /// The whole document's combined text, with page markers — the same shape
    /// the CLI writes for a multi-page run.
    ///
    /// Markdown pages concatenate into one valid document. MusicXML pages do
    /// NOT: each page is a complete XML document with its own declaration and
    /// root element, so gluing several together produces something that is not
    /// valid XML and that no score reader will open. For that lane the pages are
    /// bundled as plain text with explicit separators, and
    /// [`exportFilename`] names the file `.txt` so it never claims to be
    /// MusicXML.
    var documentText: String {
        guard !pageOutcomes.isEmpty else { return recognition?.output ?? "" }
        let music = spec.producesMusicXML
        return pageOutcomes.compactMap { outcome -> String? in
            switch outcome.state {
            case .done:
                let header = music
                    ? "===== page \(outcome.id) ====="
                    : "<!-- page \(outcome.id) -->"
                return "\(header)\n\n\(outcome.text)"
            case .skipped(let reason):
                return music
                    ? "===== page \(outcome.id) skipped: \(reason) ====="
                    : "<!-- page \(outcome.id) skipped: \(reason) -->"
            case .queued, .running:
                return nil
            }
        }
        .joined(separator: "\n\n")
    }

    /// How many pages actually produced output in this run.
    private var producedPageCount: Int { completedPageCount }

    private func ensureEngineLoaded(lifecycleToken: UInt64) async throws {
        isLoadingModel = await !engine.isLoaded
        defer { isLoadingModel = false }
        if isLoadingModel { status = "Waking the model…" }
        let url = try ModelStore.artifactURL(for: spec)
        try await engine.load(artifact: url, lifecycleToken: lifecycleToken)
        if spec.decodeGuard > 0 {
            await engine.setNoRepeatNgram(spec.decodeGuard)
        }
        licenseNotice = await engine.licenseNotice
    }

    /// Options that belong to the RUN rather than to the loaded engine. The
    /// engine exposes these as process-global setters that take effect on the
    /// next recognition, so they are re-applied every run — otherwise a toggle
    /// flipped after the model loaded would be silently ignored.
    private func applyPerRunOptions() async {
        switch spec.id {
        case "smolvlm2":
            await engine.setSmolVLM2Question(question)
        case "got-ocr2":
            await engine.setGotFormat(gotFormat)
        default:
            break
        }
    }

    func cancel() {
        engine.requestCancel()
        status = "Stopping at the next checkpoint…"
        statusKind = .warn
    }

    private func nextEngineLifecycleToken() -> UInt64 {
        engineLifecycleToken &+= 1
        return engineLifecycleToken
    }

    // ── Diagnostics ────────────────────────────────────────────────────────

    func runSelftest() {
        Task.detached(priority: .userInitiated) {
            let result: String
            do {
                result = try Engine.selftest()
            } catch {
                result = "{\"error\": \"\(error.localizedDescription)\"}"
            }
            await MainActor.run {
                self.selftestJSON = result
                self.showSelftest = true
            }
        }
    }

    /// What to write when the user exports. A document walk exports the whole
    /// document, not the page that happens to be on screen.
    ///
    /// A multi-page music run is a BUNDLE of separate scores, not one MusicXML
    /// file, so it is named `.txt`. Handing someone a `.musicxml` that no reader
    /// can open would be worse than an honest extension.
    var exportFilename: String {
        guard spec.producesMusicXML else { return "\(exportStem).md" }
        return producedPageCount > 1 ? "\(exportStem)-pages.txt" : "\(exportStem).musicxml"
    }

    /// The text an export or the source view should show.
    var displayText: String {
        isDocumentRun ? documentText : (recognition?.output ?? "")
    }

    // ── HTML export ────────────────────────────────────────────────────────

    /// The styled-HTML lane exists only where the output IS Markdown. A
    /// MusicXML score belongs in a score reader, not a browser, and wrapping
    /// it in a web page would only obscure that.
    var canExportHtml: Bool {
        recognition != nil && !spec.producesMusicXML
    }

    var htmlExportFilename: String {
        "\(exportStem).html"
    }

    private var exportStem: String {
        (imageName as NSString?)?.deletingPathExtension ?? "page"
    }

    /// OneChart's output is a structured-data dict, not Markdown; the
    /// paragraph pass would collapse its newlines into one run-on line.
    /// Present it as the code block it actually is.
    private func htmlPageMarkdown(_ text: String) -> String {
        spec.id == "onechart" ? "```\n\(text)\n```" : text
    }

    /// Everything the styled HTML document needs, captured as plain values so
    /// the actual rendering can happen inside the share transfer instead of on
    /// every view update.
    func htmlExportPayload() -> (provenance: HtmlExport.Provenance, sections: [HtmlExport.Section]) {
        var sections: [HtmlExport.Section] = []
        var pageSummary: String?
        if isDocumentRun {
            for outcome in pageOutcomes {
                switch outcome.state {
                case .done:
                    sections.append(.page(number: outcome.id,
                                          markdown: htmlPageMarkdown(outcome.text)))
                case .skipped(let reason):
                    sections.append(.skipped(number: outcome.id, reason: reason))
                case .queued, .running:
                    break
                }
            }
            let skips = pageOutcomes.filter {
                if case .skipped = $0.state { true } else { false }
            }.count
            pageSummary = skips > 0
                ? "\(completedPageCount) pages recognized · \(skips) skipped"
                : "\(completedPageCount) pages"
        } else {
            sections.append(.page(number: nil,
                                  markdown: htmlPageMarkdown(recognition?.output ?? "")))
        }
        let provenance = HtmlExport.Provenance(
            title: exportStem,
            modelName: spec.shortName,
            characters: displayText.count,
            // Nil for a run that is still going or was cancelled — it is
            // cleared at run start and assigned only on completion, so a
            // partial export never inherits another run's timing.
            seconds: lastRunSeconds,
            pageSummary: pageSummary
        )
        return (provenance, sections)
    }
}
