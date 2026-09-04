import AVFoundation
import Foundation
import SwiftUI
import UIKit
import VisionKit

#if !targetEnvironment(macCatalyst)

/// One update from Apple's native Live Text camera scanner.
///
/// Boxes use normalized UIKit coordinates: the origin is at the top-left of
/// the live camera surface. `animatedLineIDs` contains only newly recognized or
/// materially changed lines, so a stationary paragraph does not continuously
/// replay the same animation.
struct LiveCameraBatch: Identifiable {
    struct Line: Identifiable {
        let id: UUID
        let text: String
        let confidence: Float
        let box: CGRect
    }

    let id = UUID()
    let lines: [Line]
    let animatedLineIDs: Set<UUID>
}

/// Main-actor bridge around VisionKit's dedicated live-scanning controller.
///
/// DataScannerViewController is the system Live Text camera surface. It owns
/// camera scheduling, high-frame-rate item tracking, and Apple's on-device text
/// recognizer. FrankenOCR deliberately asks for `.accurate` rather than `.fast`:
/// DataScanner's tracking keeps the surface live without accumulating a queue.
@MainActor
private final class LiveCameraController: NSObject, ObservableObject,
                                          DataScannerViewControllerDelegate {
    @Published private(set) var latestBatch: LiveCameraBatch?
    @Published private(set) var isRunning = false
    @Published private(set) var isPaused = false
    @Published private(set) var torchOn = false
    @Published private(set) var errorMessage: String?

    fileprivate weak var scanner: DataScannerViewController?
    private var previousTranscripts: [UUID: String] = [:]

    func attach(_ scanner: DataScannerViewController) {
        self.scanner = scanner
        scanner.delegate = self
    }

    func start() {
        guard DataScannerViewController.isSupported else {
            errorMessage = "Apple Live Text camera scanning is not supported on this device."
            return
        }
        guard DataScannerViewController.isAvailable else {
            errorMessage = "Live Camera is unavailable. Check Camera access in Settings and try again."
            return
        }
        do {
            try scanner?.startScanning()
            errorMessage = nil
            isRunning = true
            isPaused = false
        } catch {
            errorMessage = "Live Camera could not start: \(error.localizedDescription)"
            isRunning = false
            isPaused = false
        }
    }

    func togglePause() {
        if isRunning {
            scanner?.stopScanning()
            isRunning = false
            isPaused = true
            turnTorchOff()
        } else if isPaused {
            start()
        }
    }

    func stop() {
        scanner?.stopScanning()
        isRunning = false
        isPaused = false
        turnTorchOff()
    }

    func toggleTorch() {
        guard let camera = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .back
        ), camera.hasTorch else { return }
        do {
            try camera.lockForConfiguration()
            let turnOn = camera.torchMode != .on
            camera.torchMode = turnOn ? .on : .off
            camera.unlockForConfiguration()
            torchOn = turnOn
        } catch {
            errorMessage = "The flashlight could not be changed."
        }
    }

    func capturePhotoJPEG() async -> Data? {
        guard let scanner else { return nil }
        do {
            let image = try await scanner.capturePhoto()
            return image.jpegData(compressionQuality: 0.86)
        } catch {
            // Captured text is still useful if a final camera snapshot races
            // with dismissal or camera interruption.
            return nil
        }
    }

    func dataScanner(
        _ dataScanner: DataScannerViewController,
        didAdd addedItems: [RecognizedItem],
        allItems: [RecognizedItem]
    ) {
        publish(allItems, changedItems: addedItems, scanner: dataScanner)
    }

    func dataScanner(
        _ dataScanner: DataScannerViewController,
        didUpdate updatedItems: [RecognizedItem],
        allItems: [RecognizedItem]
    ) {
        publish(allItems, changedItems: updatedItems, scanner: dataScanner)
    }

    func dataScanner(
        _ dataScanner: DataScannerViewController,
        didRemove _: [RecognizedItem],
        allItems: [RecognizedItem]
    ) {
        publish(allItems, changedItems: [], scanner: dataScanner)
    }

    func dataScanner(
        _: DataScannerViewController,
        becameUnavailableWithError error: DataScannerViewController.ScanningUnavailable
    ) {
        switch error {
        case .cameraRestricted:
            errorMessage = "Camera access is restricted. Enable it in Settings to use Live Camera."
        case .unsupported:
            errorMessage = "Apple Live Text camera scanning is not supported on this device."
        @unknown default:
            errorMessage = "Apple Live Text camera scanning became unavailable."
        }
        isRunning = false
        isPaused = false
    }

    private func publish(
        _ items: [RecognizedItem],
        changedItems: [RecognizedItem],
        scanner: DataScannerViewController
    ) {
        guard isRunning else { return }
        let surface = scanner.view.bounds.size
        guard surface.width > 0, surface.height > 0 else { return }

        let changedIDs = Set(changedItems.map(\.id))
        var lines: [LiveCameraBatch.Line] = []
        var animatedIDs: Set<UUID> = []
        var currentTranscripts: [UUID: String] = [:]

        for item in items {
            guard case let .text(recognized) = item,
                  let candidate = recognized.observation.topCandidates(1).first
            else { continue }

            let text = recognized.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            // DataScanner's own stabilized transcript is the source of truth
            // for Live Camera. Its highlight can be valid while Vision's
            // candidate confidence remains below a fixed threshold, so a hard
            // confidence gate made visible words silently disappear before
            // they ever reached the tray. Deduplication below absorbs the
            // focus-settling repeats without dropping Apple's accepted text.
            guard text.count >= 2 else { continue }

            let bounds = recognized.bounds
            let points = [bounds.topLeft, bounds.topRight, bounds.bottomRight, bounds.bottomLeft]
            let minX = points.map(\.x).min() ?? 0
            let maxX = points.map(\.x).max() ?? minX
            let minY = points.map(\.y).min() ?? 0
            let maxY = points.map(\.y).max() ?? minY
            let box = CGRect(
                x: min(max(minX / surface.width, 0), 1),
                y: min(max(minY / surface.height, 0), 1),
                width: min(max((maxX - minX) / surface.width, 0), 1),
                height: min(max((maxY - minY) / surface.height, 0), 1)
            )

            currentTranscripts[recognized.id] = text
            lines.append(LiveCameraBatch.Line(
                id: recognized.id,
                text: text,
                confidence: candidate.confidence,
                box: box
            ))

            if changedIDs.contains(recognized.id), previousTranscripts[recognized.id] != text {
                animatedIDs.insert(recognized.id)
            }
        }

        lines.sort {
            if abs($0.box.midY - $1.box.midY) > 0.025 {
                return $0.box.midY < $1.box.midY
            }
            return $0.box.minX < $1.box.minX
        }
        previousTranscripts = currentTranscripts
        latestBatch = LiveCameraBatch(lines: lines, animatedLineIDs: animatedIDs)
    }

    private func turnTorchOff() {
        guard torchOn,
              let camera = AVCaptureDevice.default(
                .builtInWideAngleCamera,
                for: .video,
                position: .back
              ), camera.hasTorch
        else { return }
        do {
            try camera.lockForConfiguration()
            camera.torchMode = .off
            camera.unlockForConfiguration()
        } catch {
            // Dismissal must not fail because the camera was already released.
        }
        torchOn = false
    }
}

private struct NativeLiveTextScanner: UIViewControllerRepresentable {
    @ObservedObject var controller: LiveCameraController

    func makeUIViewController(context _: Context) -> DataScannerViewController {
        let scanner = DataScannerViewController(
            recognizedDataTypes: [.text()],
            qualityLevel: .accurate,
            recognizesMultipleItems: true,
            isHighFrameRateTrackingEnabled: true,
            isPinchToZoomEnabled: true,
            isGuidanceEnabled: true,
            isHighlightingEnabled: true
        )
        controller.attach(scanner)
        // Construction already runs on the main actor. Starting synchronously
        // prevents a queued start from racing dismissal and resurrecting the
        // camera after dismantle has stopped it.
        controller.start()
        return scanner
    }

    func updateUIViewController(_: DataScannerViewController, context _: Context) {}

    static func dismantleUIViewController(
        _ scanner: DataScannerViewController,
        coordinator _: Void
    ) {
        scanner.stopScanning()
    }
}

struct LiveTextAccumulator {
private struct CapturedLine {
        var sourceID: UUID
        var text: String
        var confidence: Float
        var normalized: String
    }

    private var lines: [CapturedLine] = []
    var text: String { lines.map(\.text).joined(separator: "\n") }

    mutating func clear() { lines.removeAll(keepingCapacity: true) }

    /// Accumulate useful lines without copying the same stationary paragraph on
    /// every tracking update. Only lines accepted into the tray are animated.
    mutating func ingest(_ batch: LiveCameraBatch) -> [LiveCameraBatch.Line] {
        var accepted: [LiveCameraBatch.Line] = []
        for incoming in batch.lines {
            let normalized = Self.normalize(incoming.text)
            guard normalized.count >= 2 else { continue }

            // VisionKit keeps a stable ID while one tracked text observation
            // settles. Preserve that identity so a partial transcript is
            // replaced by its corrected form instead of being appended as a
            // second line merely because their word-overlap score is low.
            if let exactIndex = lines.firstIndex(where: { $0.sourceID == incoming.id }) {
                let existing = lines[exactIndex]
                guard existing.normalized != normalized else {
                    if incoming.confidence > existing.confidence {
                        lines[exactIndex].confidence = incoming.confidence
                    }
                    continue
                }
                lines[exactIndex] = CapturedLine(
                    sourceID: incoming.id,
                    text: incoming.text,
                    confidence: incoming.confidence,
                    normalized: normalized
                )
                accepted.append(incoming)
                continue
            }

            let match = lines.indices
                .map { ($0, Self.similarity(normalized, lines[$0].normalized)) }
                .max { $0.1 < $1.1 }

            if let match, match.1 >= 0.78 {
                if incoming.text.count > lines[match.0].text.count
                    || incoming.confidence > lines[match.0].confidence + 0.08 {
                    lines[match.0] = CapturedLine(
                        sourceID: incoming.id,
                        text: incoming.text,
                        confidence: incoming.confidence,
                        normalized: normalized
                    )
                    accepted.append(incoming)
                }
                continue
            }

            lines.append(CapturedLine(
                sourceID: incoming.id,
                text: incoming.text,
                confidence: incoming.confidence,
                normalized: normalized
            ))
            // Animation follows actual tray admission. Depending on focus and
            // tracking timing, VisionKit can introduce a valid line in an
            // all-items callback without retaining its ID in changedItems.
            accepted.append(incoming)
        }

        if lines.count > 160 { lines.removeFirst(lines.count - 160) }
        return Array(accepted.prefix(12))
    }

    private static func normalize(_ text: String) -> String {
        text.lowercased()
            .unicodeScalars
            .map { CharacterSet.alphanumerics.contains($0) ? Character(String($0)) : " " }
            .reduce(into: "") { $0.append($1) }
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
    }

    private static func similarity(_ a: String, _ b: String) -> Double {
        if a == b { return 1 }
        if a.count >= 5, (a.contains(b) || b.contains(a)) {
            return Double(min(a.count, b.count)) / Double(max(a.count, b.count))
        }
        let left = Set(a.split(separator: " "))
        let right = Set(b.split(separator: " "))
        guard !left.isEmpty || !right.isEmpty else { return 0 }
        return Double(left.intersection(right).count) / Double(left.union(right).count)
    }
}

/// A whole recognized line travels as one strip. The characters are never
/// invented, shuffled, or scattered into arbitrary lanes: the displayed text
/// is exactly the native scanner transcript and begins at its real screen box.
private struct FlyingTextStrip: Identifiable {
    let id = UUID()
    let text: String
    let origin: CGPoint
    let destinationX: CGFloat
    let curve: CGFloat
    let born: TimeInterval
    let duration: TimeInterval
}

private struct CaptureTrayTopPreference: PreferenceKey {
    static let defaultValue: CGFloat? = nil

    static func reduce(value: inout CGFloat?, nextValue: () -> CGFloat?) {
        guard let next = nextValue() else { return }
        value = value.map { min($0, next) } ?? next
    }
}

private struct TextWarpLayer: View {
    let strips: [FlyingTextStrip]
    let landingY: CGFloat?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        if !reduceMotion {
            TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { timeline in
                Canvas { context, size in
                    let now = timeline.date.timeIntervalSinceReferenceDate
                    for strip in strips {
                        let progress = (now - strip.born) / strip.duration
                        guard progress >= 0, progress <= 1 else { continue }
                        drawStrip(strip, progress: CGFloat(progress), context: &context, size: size)
                    }
                }
            }
            // Canvas has no intrinsic size. Without this expansion SwiftUI can
            // legally collapse the animation layer to zero even though the
            // camera and overlay fill the screen.
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .ignoresSafeArea()
            .allowsHitTesting(false)
        }
    }

    private func drawStrip(
        _ strip: FlyingTextStrip,
        progress: CGFloat,
        context: inout GraphicsContext,
        size: CGSize
    ) {
        let start = CGPoint(x: strip.origin.x * size.width, y: strip.origin.y * size.height)
        let trayEdge = landingY ?? size.height * 0.75
        let end = CGPoint(
            x: strip.destinationX * size.width,
            y: min(size.height - 24, max(size.height * 0.48, trayEdge + 18))
        )
        let control = CGPoint(
            x: (start.x + end.x) * 0.5 + strip.curve * size.width,
            y: min(start.y, end.y) - 62
        )

        // Draw the actual recognized glyphs along the curve. Each character's
        // position and rotation follow the Bézier tangent, so the line visibly
        // bends into the tray instead of translating as one rigid label.
        let glyphs = Array(strip.text.prefix(34))
        guard !glyphs.isEmpty else { return }
        let eased = progress * progress * (3 - 2 * progress)
        let curveSpan = min(0.24, CGFloat(glyphs.count) * 0.010)
        let fade = min(1, max(0, (1 - progress) / 0.18))

        for (index, glyph) in glyphs.enumerated() {
            let fraction = glyphs.count == 1
                ? 1
                : CGFloat(index) / CGFloat(glyphs.count - 1)
            let t = min(1, max(0, eased - (1 - fraction) * curveSpan))
            let point = bezierPoint(t, start: start, control: control, end: end)
            let tangent = bezierTangent(t, start: start, control: control, end: end)
            let angle = atan2(tangent.y, tangent.x)
            let resolved = context.resolve(
                Text(String(glyph))
                    .font(.system(size: Lab.typeSize(13), weight: .black, design: .monospaced))
                    .foregroundStyle(Lab.accent)
            )
            var copy = context
            copy.opacity = Double(fade * (0.72 + 0.28 * fraction))
            copy.translateBy(x: point.x, y: point.y)
            copy.rotate(by: .radians(Double(angle)))
            copy.scaleBy(x: 1 - 0.28 * progress, y: 1 - 0.18 * progress)
            copy.draw(resolved, at: .zero, anchor: .center)
        }
    }

    private func bezierPoint(
        _ t: CGFloat,
        start: CGPoint,
        control: CGPoint,
        end: CGPoint
    ) -> CGPoint {
        let oneMinus = 1 - t
        return CGPoint(
            x: oneMinus * oneMinus * start.x
                + 2 * oneMinus * t * control.x
                + t * t * end.x,
            y: oneMinus * oneMinus * start.y
                + 2 * oneMinus * t * control.y
                + t * t * end.y
        )
    }

    private func bezierTangent(
        _ t: CGFloat,
        start: CGPoint,
        control: CGPoint,
        end: CGPoint
    ) -> CGPoint {
        CGPoint(
            x: 2 * (1 - t) * (control.x - start.x) + 2 * t * (end.x - control.x),
            y: 2 * (1 - t) * (control.y - start.y) + 2 * t * (end.y - control.y)
        )
    }
}

private struct ScannerReticle: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GeometryReader { geometry in
            let rect = CGRect(
                x: geometry.size.width * 0.07,
                y: geometry.size.height * 0.16,
                width: geometry.size.width * 0.86,
                height: geometry.size.height * 0.43
            )
            TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: reduceMotion)) { timeline in
                let phase = timeline.date.timeIntervalSinceReferenceDate
                    .truncatingRemainder(dividingBy: 2.8) / 2.8
                Canvas { context, _ in
                    let border = Path(roundedRect: rect, cornerRadius: 18)
                    context.stroke(border, with: .color(Lab.accent.opacity(0.34)), lineWidth: 1)
                    guard !reduceMotion else { return }
                    let y = rect.minY + rect.height * CGFloat(phase)
                    var beam = Path()
                    beam.move(to: CGPoint(x: rect.minX + 10, y: y))
                    beam.addLine(to: CGPoint(x: rect.maxX - 10, y: y))
                    context.stroke(
                        beam,
                        with: .linearGradient(
                            Gradient(colors: [.clear, Lab.accent.opacity(0.9), .clear]),
                            startPoint: CGPoint(x: rect.minX, y: y),
                            endPoint: CGPoint(x: rect.maxX, y: y)
                        ),
                        lineWidth: 1.5
                    )
                }
            }
        }
        .allowsHitTesting(false)
    }
}

struct LiveCameraView: View {
    let model: LabModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @StateObject private var camera = LiveCameraController()
    @State private var accumulator = LiveTextAccumulator()
    @State private var strips: [FlyingTextStrip] = []
    @State private var captureTrayTop: CGFloat?
    @State private var copied = false
    @State private var copyResetTask: Task<Void, Never>?
    @State private var isFinishing = false

    private var capturedText: String { accumulator.text }

    var body: some View {
        ZStack {
            NativeLiveTextScanner(controller: camera)
                .ignoresSafeArea()

            LinearGradient(
                colors: [.black.opacity(0.55), .clear, .black.opacity(0.84)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            ScannerReticle()
            TextWarpLayer(strips: strips, landingY: captureTrayTop)

            VStack(spacing: 0) {
                topBar
                Spacer()
                accuracyNotice
                capturePanel
            }
            .padding(.horizontal, 16)
            .padding(.top, 8)
            .padding(.bottom, 10)

            if let error = camera.errorMessage { errorOverlay(error) }
        }
        .coordinateSpace(name: "liveCameraSurface")
        .tint(Lab.accent)
        .onDisappear {
            copyResetTask?.cancel()
            camera.stop()
        }
        .onChange(of: camera.latestBatch?.id) { _, _ in
            guard let batch = camera.latestBatch else { return }
            ingest(batch)
        }
        .onPreferenceChange(CaptureTrayTopPreference.self) { captureTrayTop = $0 }
    }

    private var topBar: some View {
        HStack(spacing: 12) {
            Button { dismiss() } label: {
                Image(systemName: "xmark")
                    .font(.system(size: Lab.typeSize(15), weight: .bold))
                    .frame(width: 42, height: 42)
                    .background(.black.opacity(0.50), in: Circle())
            }
            .accessibilityLabel("Close Live Camera")

            VStack(alignment: .leading, spacing: 2) {
                Text("LIVE CAMERA")
                    .font(.system(size: Lab.typeSize(15), weight: .black, design: .monospaced))
                    .foregroundStyle(.white)
                HStack(spacing: 6) {
                    Circle()
                        .fill(camera.isRunning ? Lab.accent : Lab.amber)
                        .frame(width: 6, height: 6)
                    Text(camera.isRunning
                         ? "APPLE LIVE TEXT · ON DEVICE"
                         : (camera.isPaused ? "PAUSED · CAPTURED TEXT HELD" : "STARTING CAMERA"))
                        .font(.system(size: Lab.typeSize(9), weight: .bold, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.68))
                }
            }
            Spacer()
            Button { camera.togglePause() } label: {
                Image(systemName: camera.isPaused ? "play.fill" : "pause.fill")
                    .font(.system(size: Lab.typeSize(15), weight: .semibold))
                    .frame(width: 42, height: 42)
                    .background(.black.opacity(0.50), in: Circle())
            }
            .disabled(!camera.isRunning && !camera.isPaused)
            .accessibilityLabel(camera.isPaused ? "Resume Live Camera" : "Pause Live Camera")
            .accessibilityHint("Keeps the text already captured in the tray")
            Button { camera.toggleTorch() } label: {
                Image(systemName: camera.torchOn ? "flashlight.on.fill" : "flashlight.off.fill")
                    .font(.system(size: Lab.typeSize(16), weight: .semibold))
                    .frame(width: 42, height: 42)
                    .background(.black.opacity(0.50), in: Circle())
            }
            .accessibilityLabel(camera.torchOn ? "Turn flashlight off" : "Turn flashlight on")
        }
        .foregroundStyle(.white)
    }

    private var accuracyNotice: some View {
        HStack(alignment: .top, spacing: 9) {
            Image(systemName: "bolt.fill")
                .foregroundStyle(Lab.amber)
                .padding(.top, 1)
            Text("FAST LIVE MODE: This uses Apple's on-device Live Text recognizer—not the Baidu/FrankenOCR model. It is optimized for real-time video and can be less accurate. For maximum accuracy, use Camera to take a still photo.")
                .font(.system(size: Lab.typeSize(9), weight: .semibold, design: .monospaced))
                .foregroundStyle(.white.opacity(0.82))
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(10)
        .background(.black.opacity(0.70), in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(Lab.amber.opacity(0.48), lineWidth: 1)
        )
        .padding(.bottom, 8)
    }

    private var capturePanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("CAPTURED TEXT")
                        .font(.system(size: Lab.typeSize(11), weight: .black, design: .monospaced))
                        .kerning(1.5)
                        .foregroundStyle(Lab.accent)
                    Text("\(camera.latestBatch?.lines.count ?? 0) live lines · \(capturedText.count) captured characters")
                        .font(.system(size: Lab.typeSize(9), design: .monospaced))
                        .foregroundStyle(.white.opacity(0.52))
                }
                Spacer()
                ShareLink(item: capturedText) {
                    Image(systemName: "square.and.arrow.up")
                        .frame(width: 34, height: 34)
                }
                .disabled(capturedText.isEmpty)
                .accessibilityLabel("Share captured text")
                Button(copied ? "Copied" : "Copy") {
                    UIPasteboard.general.string = capturedText
                    copied = true
                    copyResetTask?.cancel()
                    copyResetTask = Task {
                        try? await Task.sleep(for: .seconds(1.2))
                        guard !Task.isCancelled else { return }
                        copied = false
                    }
                }
                .disabled(capturedText.isEmpty)
                .font(.system(size: Lab.typeSize(11), weight: .bold, design: .monospaced))

                Button("Clear") {
                    accumulator.clear()
                    strips.removeAll(keepingCapacity: true)
                }
                .disabled(capturedText.isEmpty)
                .font(.system(size: Lab.typeSize(11), weight: .bold, design: .monospaced))
            }

            ScrollView {
                Text(capturedText.isEmpty
                     ? "Point the camera at text. Actual recognized lines will bend into this tray."
                     : capturedText)
                    .font(.system(size: Lab.contentTypeSize(13), design: .monospaced))
                    .foregroundStyle(capturedText.isEmpty ? .white.opacity(0.42) : .white.opacity(0.92))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minHeight: 72, maxHeight: 142)

            Button { finishCapture() } label: {
                Label(
                    isFinishing ? "Capturing frame…" : "Use captured text",
                    systemImage: "arrow.down.doc.fill"
                )
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(PrimaryButtonStyle())
            .disabled(capturedText.isEmpty || isFinishing)
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
        .overlay(
            RoundedRectangle(cornerRadius: 18)
                .strokeBorder(Lab.accent.opacity(0.34), lineWidth: 1)
        )
        .background {
            GeometryReader { geometry in
                Color.clear.preference(
                    key: CaptureTrayTopPreference.self,
                    value: geometry.frame(in: .named("liveCameraSurface")).minY
                )
            }
        }
    }

    private func errorOverlay(_ message: String) -> some View {
        VStack(spacing: 14) {
            Image(systemName: "camera.fill")
                .font(.system(size: Lab.typeSize(28)))
                .foregroundStyle(Lab.amber)
            Text(message)
                .font(.system(size: Lab.typeSize(13), weight: .semibold))
                .foregroundStyle(.white)
                .multilineTextAlignment(.center)
            Button("Open Settings") {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            }
            .buttonStyle(PrimaryButtonStyle())
            Button("Close") { dismiss() }
                .buttonStyle(GhostButtonStyle())
        }
        .padding(24)
        .frame(maxWidth: 320)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
    }

    private func ingest(_ batch: LiveCameraBatch) {
        // Assign a new value back to @State. Mutating a nested value in place
        // is not a reliable invalidation boundary for every SwiftUI release,
        // and the old path could recognize text while leaving this tray stale.
        var nextAccumulator = accumulator
        let accepted = nextAccumulator.ingest(batch)
        accumulator = nextAccumulator
        guard !reduceMotion, !accepted.isEmpty else { return }
        let now = Date.timeIntervalSinceReferenceDate
        strips.removeAll { now - $0.born > $0.duration }
        strips.append(contentsOf: accepted.enumerated().map { index, line in
            FlyingTextStrip(
                text: line.text,
                origin: CGPoint(x: line.box.midX, y: line.box.midY),
                destinationX: 0.30 + CGFloat(index % 5) * 0.10,
                curve: CGFloat((index % 3) - 1) * 0.09,
                born: now + Double(index) * 0.055,
                duration: 1.0 + Double(index % 4) * 0.08
            )
        })
        if strips.count > 48 { strips.removeFirst(strips.count - 48) }
    }

    private func finishCapture() {
        let text = capturedText
        isFinishing = true
        Task { @MainActor in
            let snapshot = await camera.capturePhotoJPEG()
            model.adoptLiveCameraCapture(text: text, snapshot: snapshot)
            dismiss()
        }
    }
}

#else

/// VisionKit's realtime DataScanner is deliberately unavailable in Mac
/// Catalyst. Keep the universal app honest and useful: imported images and PDFs
/// still use the full FrankenOCR engine, while this route explains where the
/// realtime Apple scanner is supported instead of presenting a dead camera.
struct LiveCameraView: View {
    let model: LabModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            LabBackground()
            VStack(alignment: .leading, spacing: 24) {
                HStack {
                    LabLabel(text: "Live camera")
                    Spacer()
                    Button { dismiss() } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: Lab.typeSize(14), weight: .bold))
                            .frame(width: 38, height: 38)
                    }
                    .buttonStyle(GhostButtonStyle())
                    .accessibilityLabel("Close")
                }

                Spacer()

                LabPanel {
                    VStack(alignment: .leading, spacing: 16) {
                        Image(systemName: "macbook.and.iphone")
                            .font(.system(size: Lab.typeSize(38), weight: .bold))
                            .foregroundStyle(Lab.amber)
                        Text("Live Camera continues on iPhone and iPad")
                            .font(.system(size: Lab.typeSize(22), weight: .bold))
                            .foregroundStyle(Lab.textPrimary)
                        Text(
                            "Apple's realtime Live Text camera is not available to Mac Catalyst apps. "
                                + "On this Mac, drop or choose an image or PDF in the Vision Table to "
                                + "run the full, more accurate FrankenOCR model entirely on-device."
                        )
                        .font(.system(size: Lab.typeSize(17)))
                        .foregroundStyle(Lab.textDim)
                        .fixedSize(horizontal: false, vertical: true)
                        Button { dismiss() } label: {
                            Label("Return to the Vision Table", systemImage: "doc.viewfinder")
                        }
                        .buttonStyle(PrimaryButtonStyle())
                    }
                }

                Spacer()
            }
            .padding(24)
            .frame(maxWidth: 720)
        }
        .tint(Lab.accent)
    }
}

#endif
