import Foundation
import SwiftUI
import UIKit

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

    var spec: ModelSpec = ModelCatalog.all[0] {
        didSet {
            guard spec.id != oldValue.id else { return }
            // A different model means a different engine. Drop the old one
            // before anything touches the new artifact.
            Task { await engine.unload() }
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

    /// Loaded PDF, if the input was a document.
    var pdfData: Data?
    var pdfPageCount: Int = 0
    var pdfPage: Int = 1

    /// SmolVLM2's question. Kept even when another model is selected so it
    /// survives a round trip through the picker.
    var question: String = ""

    // ── Run state ──────────────────────────────────────────────────────────
    var isRecognizing = false
    var isLoadingModel = false
    var elapsed: TimeInterval = 0
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

    private var timer: Timer?
    private var recognizeTask: Task<Void, Never>?
    /// Guards against a stale run publishing over a newer one.
    private var generation = 0

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
        isInstalled && imageData != nil && !isRecognizing && spec.isSupportedOnThisDevice
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
        let denominator: Double = if progress.total > 0 {
            Double(progress.total)
        } else {
            Double(estimatedTokens)
        }
        let within = denominator > 0
            ? min(1, Double(progress.current) / denominator)
            : 0
        return min(0.99, stage.precedingWeight + within * stage.weight)
    }

    var progressIsEstimated: Bool { (progress?.total ?? 0) == 0 }

    private var estimatedTokens: Int {
        switch spec.id {
        case "unlimited-ocr": 1600
        case "tromr": 300
        default: 700
        }
    }

    var progressDetail: String {
        guard let progress else { return "" }
        let percent = Int(progressFraction * 100)
        let qualifier = progressIsEstimated ? " (estimated)" : ""
        var detail = "\(percent)%\(qualifier) · \(progress.stage.label)"
        if progress.total > 0 {
            detail += " · \(progress.current)/\(progress.total)"
        } else if progress.current > 0 {
            detail += " · \(progress.current) tokens"
        }
        return detail
    }

    // ── Model lifecycle ────────────────────────────────────────────────────

    func requestDownload() { showConsent = true }

    func confirmDownload() {
        showConsent = false
        store.download(spec)
    }

    func clearModel() {
        Task { await engine.unload() }
        store.clear(spec)
        recognition = nil
        status = "Model removed. \(spec.weights.bytes.humanBytes) freed."
        statusKind = .neutral
    }

    /// Drop the engine under memory pressure or on backgrounding. A no-op mid
    /// recognition — tearing the model out from under a running forward would
    /// turn a slow page into a crash.
    func releaseEngineIfIdle() {
        guard !isRecognizing else { return }
        Task { await engine.unload() }
    }

    // ── Input ──────────────────────────────────────────────────────────────

    func accept(data: Data, name: String) {
        if data.starts(with: Array("%PDF".utf8)) {
            acceptPDF(data, name: name)
        } else {
            acceptImage(data, name: name)
        }
    }

    private func acceptImage(_ data: Data, name: String) {
        guard let image = UIImage(data: data) else {
            status = "That file is not a PNG or JPEG this app can read."
            statusKind = .warn
            return
        }
        pdfData = nil
        pdfPageCount = 0
        imageData = data
        previewImage = image
        imageName = name
        recognition = nil
        status = "\(name) · \(Int(image.size.width))×\(Int(image.size.height))"
        statusKind = .neutral
    }

    private func acceptPDF(_ data: Data, name: String) {
        do {
            let pages = try Engine.pdfPageCount(data)
            pdfData = data
            pdfPageCount = pages
            pdfPage = 1
            imageName = name
            recognition = nil
            status = "\(name) · \(pages) page\(pages == 1 ? "" : "s"). Pick a page to render."
            statusKind = .neutral
            loadPDFPage()
        } catch {
            status = error.localizedDescription
            statusKind = .err
        }
    }

    func loadPDFPage() {
        guard let pdfData else { return }
        do {
            let png = try Engine.pdfRenderPage(pdfData, page: pdfPage)
            imageData = png
            previewImage = UIImage(data: png)
            status = "Page \(pdfPage) of \(pdfPageCount) rendered."
            statusKind = .ok
        } catch {
            // The engine names exactly what was unsupported (JPEG 2000, JBIG2,
            // a born-digital vector page) rather than returning a wrong result.
            imageData = nil
            previewImage = nil
            status = error.localizedDescription
            statusKind = .err
        }
    }

    // ── Recognition ────────────────────────────────────────────────────────

    func recognize() {
        guard canRecognize, let imageData else { return }
        generation += 1
        let runGeneration = generation
        isRecognizing = true
        recognition = nil
        progress = nil
        elapsed = 0
        statusKind = .neutral
        status = "Working…"

        // A minutes-long forward must not be interrupted by the screen
        // sleeping, which suspends the app.
        UIApplication.shared.isIdleTimerDisabled = true
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.elapsed += 0.5 }
        }

        recognizeTask = Task { [weak self] in
            guard let self else { return }
            defer {
                UIApplication.shared.isIdleTimerDisabled = false
                self.timer?.invalidate()
                self.timer = nil
                self.isRecognizing = false
                self.recognizeTask = nil
            }
            do {
                try await self.ensureEngineLoaded()
                let started = Date()
                let result = try await self.engine.recognize(imageData: imageData) { update in
                    Task { @MainActor [weak self] in
                        guard let self, runGeneration == self.generation else { return }
                        self.progress = update
                    }
                }
                guard runGeneration == self.generation else { return }
                let seconds = Date().timeIntervalSince(started)
                self.lastRunSeconds = seconds
                self.recognition = result
                self.progress = nil
                self.statusKind = .ok
                self.status = String(
                    format: "Done in %.1fs · %d characters, entirely on this device.",
                    seconds, result.output.count
                )
            } catch let error as EngineError where error.isCancellation {
                guard runGeneration == self.generation else { return }
                self.statusKind = .warn
                self.status = "Cancelled."
                self.progress = nil
            } catch {
                guard runGeneration == self.generation else { return }
                self.statusKind = .err
                self.status = error.localizedDescription
                self.progress = nil
            }
        }
    }

    private func ensureEngineLoaded() async throws {
        if await engine.isLoaded { return }
        isLoadingModel = true
        defer { isLoadingModel = false }
        status = "Waking the model…"
        let url = try ModelStore.artifactURL(for: spec)
        try await engine.load(artifact: url)
        if spec.decodeGuard > 0 {
            await engine.setNoRepeatNgram(spec.decodeGuard)
        }
        if spec.id == "smolvlm2" {
            await engine.setSmolVLM2Question(question)
        }
        licenseNotice = await engine.licenseNotice
    }

    func cancel() {
        engine.requestCancel()
        status = "Stopping at the next checkpoint…"
        statusKind = .warn
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

    /// What to write when the user exports.
    var exportFilename: String {
        let stem = (imageName as NSString?)?.deletingPathExtension ?? "page"
        return spec.producesMusicXML ? "\(stem).musicxml" : "\(stem).md"
    }
}
