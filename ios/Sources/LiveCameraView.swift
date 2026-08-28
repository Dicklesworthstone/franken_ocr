import AVFoundation
import CoreImage
import Foundation
import ImageIO
import SwiftUI
import UIKit
import Vision

/// One low-latency OCR pass over the latest camera frame.
///
/// The video output discards late frames and the controller runs at most one
/// Vision request at a time. There is deliberately no frame queue: when the
/// recognizer is busy, newer frames replace older ones instead of building a
/// seconds-long backlog behind the viewfinder.
private struct LiveCameraBatch: Identifiable {
    struct Line: Identifiable {
        let id = UUID()
        let text: String
        let confidence: Float
        /// Vision coordinates: normalized, with the origin at bottom-left.
        let box: CGRect
    }

    let id = UUID()
    let lines: [Line]
    let snapshot: Data?
    let latency: TimeInterval
}

/// Owns the capture session and the fast on-device text request.
///
/// Capture/session state is confined to `captureQueue`; published UI state is
/// always delivered on the main queue. This keeps camera work and Vision off
/// SwiftUI's render thread without introducing another inference queue.
private final class LiveCameraController: NSObject, ObservableObject,
                                          AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()

    @Published private(set) var latestBatch: LiveCameraBatch?
    @Published private(set) var isRunning = false
    @Published private(set) var torchOn = false
    @Published private(set) var errorMessage: String?

    private let captureQueue = DispatchQueue(
        label: "com.frankenocr.live-camera",
        qos: .userInitiated
    )
    private let output = AVCaptureVideoDataOutput()
    private let context = CIContext(options: [.cacheIntermediates: false])
    private var camera: AVCaptureDevice?
    private var configured = false
    private var lastScan = ContinuousClock.now
    private let minimumScanInterval = Duration.milliseconds(430)

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndStart()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] allowed in
                if allowed {
                    self?.configureAndStart()
                } else {
                    self?.publishError("Camera access is off. Enable it in Settings to use Live Camera.")
                }
            }
        case .denied, .restricted:
            publishError("Camera access is off. Enable it in Settings to use Live Camera.")
        @unknown default:
            publishError("The camera is unavailable on this device.")
        }
    }

    func stop() {
        captureQueue.async { [weak self] in
            guard let self else { return }
            if self.session.isRunning { self.session.stopRunning() }
            DispatchQueue.main.async { self.isRunning = false }
        }
    }

    func toggleTorch() {
        captureQueue.async { [weak self] in
            guard let self, let camera = self.camera, camera.hasTorch else { return }
            do {
                try camera.lockForConfiguration()
                let turnOn = camera.torchMode != .on
                camera.torchMode = turnOn ? .on : .off
                camera.unlockForConfiguration()
                DispatchQueue.main.async { self.torchOn = turnOn }
            } catch {
                self.publishError("The flashlight could not be changed.")
            }
        }
    }

    private func configureAndStart() {
        captureQueue.async { [weak self] in
            guard let self else { return }
            if !self.configured {
                guard self.configure() else { return }
                self.configured = true
            }
            guard !self.session.isRunning else { return }
            self.session.startRunning()
            DispatchQueue.main.async {
                self.errorMessage = nil
                self.isRunning = true
            }
        }
    }

    private func configure() -> Bool {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .high

        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .back
        ) else {
            publishError("This device does not have an available rear camera.")
            return false
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            guard session.canAddInput(input) else {
                publishError("The rear camera could not be attached to the capture session.")
                return false
            }
            session.addInput(input)
            camera = device
        } catch {
            publishError("The rear camera could not be opened: \(error.localizedDescription)")
            return false
        }

        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: captureQueue)
        guard session.canAddOutput(output) else {
            publishError("The live camera output could not be created.")
            return false
        }
        session.addOutput(output)
        return true
    }

    private func publishError(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.errorMessage = message
            self?.isRunning = false
        }
    }

    func captureOutput(
        _: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from _: AVCaptureConnection
    ) {
        let now = ContinuousClock.now
        guard now - lastScan >= minimumScanInterval,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        else { return }
        lastScan = now

        let started = ContinuousClock.now
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .fast
        request.usesLanguageCorrection = true
        request.minimumTextHeight = 0.012

        do {
            // The iPhone's back-camera sensor buffer is landscape-left while
            // the app's primary live surface is portrait.
            let handler = VNImageRequestHandler(
                cvPixelBuffer: pixelBuffer,
                orientation: .right,
                options: [:]
            )
            try handler.perform([request])
            let lines = (request.results ?? [])
                .compactMap { observation -> LiveCameraBatch.Line? in
                    guard let candidate = observation.topCandidates(1).first,
                          !candidate.string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    else { return nil }
                    return LiveCameraBatch.Line(
                        text: candidate.string,
                        confidence: candidate.confidence,
                        box: observation.boundingBox
                    )
                }
                .sorted {
                    if abs($0.box.midY - $1.box.midY) > 0.025 {
                        return $0.box.midY > $1.box.midY
                    }
                    return $0.box.minX < $1.box.minX
                }

            let snapshot = makeSnapshot(pixelBuffer)
            let latency = started.duration(to: .now).timeInterval
            let batch = LiveCameraBatch(lines: lines, snapshot: snapshot, latency: latency)
            DispatchQueue.main.async { [weak self] in self?.latestBatch = batch }
        } catch {
            publishError("Live text recognition failed: \(error.localizedDescription)")
        }
    }

    /// A compact upright frame for handing the final live capture back to the
    /// ordinary page UI. The OCR request itself reads the pixel buffer directly.
    private func makeSnapshot(_ pixelBuffer: CVPixelBuffer) -> Data? {
        let upright = CIImage(cvPixelBuffer: pixelBuffer).oriented(.right)
        let longest = max(upright.extent.width, upright.extent.height)
        let scale = min(1, 1_600 / max(longest, 1))
        let image = upright.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        return context.jpegRepresentation(
            of: image,
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            options: [
                kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.84
            ]
        )
    }
}

private extension Duration {
    var timeInterval: TimeInterval {
        let parts = components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}

private struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context _: Context) -> PreviewSurface {
        let view = PreviewSurface()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ view: PreviewSurface, context _: Context) {
        view.previewLayer.session = session
    }

    final class PreviewSurface: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }
    }
}

private struct LiveGlyphSeed {
    let character: Character
    let origin: CGPoint
}

private struct LiveTextAccumulator {
    private struct CapturedLine {
        var text: String
        var confidence: Float
        var normalized: String
    }

    private var lines: [CapturedLine] = []

    var text: String { lines.map(\.text).joined(separator: "\n") }

    mutating func clear() { lines.removeAll(keepingCapacity: true) }

    /// Merge a batch without duplicating the same sign or paragraph on every
    /// 430 ms pass. Longer/higher-confidence readings replace near-duplicates;
    /// genuinely new lines append in the order the camera encountered them.
    mutating func ingest(_ batch: LiveCameraBatch) -> [LiveGlyphSeed] {
        var seeds: [LiveGlyphSeed] = []
        for incoming in batch.lines where incoming.confidence >= 0.22 {
            let text = incoming.text.trimmingCharacters(in: .whitespacesAndNewlines)
            let normalized = Self.normalize(text)
            guard normalized.count >= 2 else { continue }

            let match = lines.indices
                .map { ($0, Self.similarity(normalized, lines[$0].normalized)) }
                .max { $0.1 < $1.1 }

            if let match, match.1 >= 0.76 {
                if text.count > lines[match.0].text.count
                    || incoming.confidence > lines[match.0].confidence + 0.08 {
                    lines[match.0] = CapturedLine(
                        text: text,
                        confidence: incoming.confidence,
                        normalized: normalized
                    )
                }
                continue
            }

            lines.append(CapturedLine(
                text: text,
                confidence: incoming.confidence,
                normalized: normalized
            ))
            seeds.append(contentsOf: Self.glyphSeeds(text: text, box: incoming.box))
        }

        // A live camera can be left open indefinitely. Keep the newest useful
        // lines instead of allowing view state to grow without a bound.
        if lines.count > 180 { lines.removeFirst(lines.count - 180) }
        return Array(seeds.prefix(72))
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

    private static func glyphSeeds(text: String, box: CGRect) -> [LiveGlyphSeed] {
        let visible = text.filter { !$0.isWhitespace }.prefix(24)
        let count = max(visible.count, 1)
        return visible.enumerated().map { index, character in
            let fraction = (Double(index) + 0.5) / Double(count)
            return LiveGlyphSeed(
                character: character,
                origin: CGPoint(
                    x: box.minX + box.width * fraction,
                    y: 1 - box.midY
                )
            )
        }
    }
}

private struct FlyingGlyph: Identifiable {
    let id = UUID()
    let character: Character
    let origin: CGPoint
    let destinationX: CGFloat
    let curve: CGFloat
    let born: TimeInterval
    let duration: TimeInterval
}

private struct GlyphWarpLayer: View {
    let glyphs: [FlyingGlyph]
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        if !reduceMotion {
            TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { timeline in
                Canvas { context, size in
                    let now = timeline.date.timeIntervalSinceReferenceDate
                    for glyph in glyphs {
                        let raw = (now - glyph.born) / glyph.duration
                        guard raw >= 0, raw <= 1 else { continue }
                        let t = CGFloat(raw)
                        let eased = t * t * (3 - 2 * t)
                        let start = CGPoint(
                            x: glyph.origin.x * size.width,
                            y: glyph.origin.y * size.height * 0.70
                        )
                        let end = CGPoint(
                            x: glyph.destinationX * size.width,
                            y: size.height * 0.79
                        )
                        let x = start.x + (end.x - start.x) * eased
                            + sin(t * .pi) * glyph.curve * size.width
                        let y = start.y + (end.y - start.y) * eased
                            - sin(t * .pi) * 42
                        let opacity = Double(sin(t * .pi))
                        context.opacity = max(0, opacity)
                        context.draw(
                            Text(String(glyph.character))
                                .font(.system(size: 16, weight: .black, design: .monospaced))
                                .foregroundStyle(Lab.accent),
                            at: CGPoint(x: x, y: y)
                        )
                    }
                }
            }
            .allowsHitTesting(false)
        }
    }
}

private struct ScannerOverlay: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GeometryReader { geometry in
            let rect = CGRect(
                x: geometry.size.width * 0.08,
                y: geometry.size.height * 0.17,
                width: geometry.size.width * 0.84,
                height: geometry.size.height * 0.42
            )
            TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: reduceMotion)) { timeline in
                let phase = timeline.date.timeIntervalSinceReferenceDate
                    .truncatingRemainder(dividingBy: 2.4) / 2.4
                let y = rect.minY + rect.height * CGFloat(phase)
                Canvas { context, _ in
                    var corners = Path()
                    let arm: CGFloat = 28
                    corners.move(to: CGPoint(x: rect.minX, y: rect.minY + arm))
                    corners.addLine(to: rect.origin)
                    corners.addLine(to: CGPoint(x: rect.minX + arm, y: rect.minY))
                    corners.move(to: CGPoint(x: rect.maxX - arm, y: rect.minY))
                    corners.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
                    corners.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + arm))
                    corners.move(to: CGPoint(x: rect.minX, y: rect.maxY - arm))
                    corners.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
                    corners.addLine(to: CGPoint(x: rect.minX + arm, y: rect.maxY))
                    corners.move(to: CGPoint(x: rect.maxX - arm, y: rect.maxY))
                    corners.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
                    corners.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - arm))
                    context.stroke(corners, with: .color(Lab.accent.opacity(0.9)), lineWidth: 2)

                    var beam = Path()
                    beam.move(to: CGPoint(x: rect.minX + 8, y: y))
                    beam.addLine(to: CGPoint(x: rect.maxX - 8, y: y))
                    context.stroke(
                        beam,
                        with: .linearGradient(
                            Gradient(colors: [.clear, Lab.accent, .clear]),
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
    @State private var glyphs: [FlyingGlyph] = []
    @State private var lastSnapshot: Data?
    @State private var copied = false

    private var capturedText: String { accumulator.text }

    var body: some View {
        ZStack {
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            LinearGradient(
                colors: [.black.opacity(0.58), .clear, .black.opacity(0.82)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            ScannerOverlay()
            GlyphWarpLayer(glyphs: glyphs)

            VStack(spacing: 0) {
                topBar
                Spacer()
                capturePanel
            }
            .padding(.horizontal, 16)
            .padding(.top, 8)
            .padding(.bottom, 10)

            if let error = camera.errorMessage {
                errorOverlay(error)
            }
        }
        .preferredColorScheme(.dark)
        .tint(Lab.accent)
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
        .onChange(of: camera.latestBatch?.id) { _, _ in
            guard let batch = camera.latestBatch else { return }
            ingest(batch)
        }
    }

    private var topBar: some View {
        HStack(spacing: 12) {
            Button { dismiss() } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 15, weight: .bold))
                    .frame(width: 42, height: 42)
                    .background(.black.opacity(0.50), in: Circle())
            }
            .accessibilityLabel("Close Live Camera")

            VStack(alignment: .leading, spacing: 2) {
                Text("LIVE CAMERA")
                    .font(.system(size: 15, weight: .black, design: .monospaced))
                    .foregroundStyle(.white)
                HStack(spacing: 6) {
                    Circle()
                        .fill(camera.isRunning ? Lab.accent : Lab.amber)
                        .frame(width: 6, height: 6)
                    Text(camera.isRunning ? "ON-DEVICE TEXT STREAM" : "STARTING CAMERA")
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.68))
                }
            }
            Spacer()
            Button { camera.toggleTorch() } label: {
                Image(systemName: camera.torchOn ? "flashlight.on.fill" : "flashlight.off.fill")
                    .font(.system(size: 16, weight: .semibold))
                    .frame(width: 42, height: 42)
                    .background(.black.opacity(0.50), in: Circle())
            }
            .accessibilityLabel(camera.torchOn ? "Turn flashlight off" : "Turn flashlight on")
        }
        .foregroundStyle(.white)
    }

    private var capturePanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("CAPTURED TEXT")
                        .font(.system(size: 11, weight: .black, design: .monospaced))
                        .kerning(1.5)
                        .foregroundStyle(Lab.accent)
                    if let batch = camera.latestBatch {
                        Text(String(format: "live pass %.0f ms · %d characters",
                                    batch.latency * 1_000, capturedText.count))
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.52))
                    }
                }
                Spacer()
                Button(copied ? "Copied" : "Copy") {
                    UIPasteboard.general.string = capturedText
                    copied = true
                    Task {
                        try? await Task.sleep(for: .seconds(1.2))
                        copied = false
                    }
                }
                .disabled(capturedText.isEmpty)
                .font(.system(size: 11, weight: .bold, design: .monospaced))

                Button("Clear") {
                    accumulator.clear()
                    glyphs.removeAll(keepingCapacity: true)
                }
                .disabled(capturedText.isEmpty)
                .font(.system(size: 11, weight: .bold, design: .monospaced))
            }

            ScrollView {
                Text(capturedText.isEmpty
                     ? "Point the camera at text. Recognized letters will stream into this tray."
                     : capturedText)
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(capturedText.isEmpty ? .white.opacity(0.42) : .white.opacity(0.92))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minHeight: 72, maxHeight: 152)

            Button {
                model.adoptLiveCameraCapture(text: capturedText, snapshot: lastSnapshot)
                dismiss()
            } label: {
                Label("Use captured text", systemImage: "arrow.down.doc.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(PrimaryButtonStyle())
            .disabled(capturedText.isEmpty)

            Text("Live Camera uses Apple's private, on-device Vision recognizer for immediate feedback. The still Camera button runs the full FrankenOCR model for deeper document parsing. Nothing is uploaded.")
                .font(.system(size: 9))
                .foregroundStyle(.white.opacity(0.42))
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
        .overlay(
            RoundedRectangle(cornerRadius: 18)
                .strokeBorder(Lab.accent.opacity(0.34), lineWidth: 1)
        )
    }

    private func errorOverlay(_ message: String) -> some View {
        VStack(spacing: 14) {
            Image(systemName: "camera.fill")
                .font(.system(size: 28))
                .foregroundStyle(Lab.amber)
            Text(message)
                .font(.system(size: 13, weight: .semibold))
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
        lastSnapshot = batch.snapshot ?? lastSnapshot
        let seeds = accumulator.ingest(batch)
        guard !reduceMotion, !seeds.isEmpty else { return }
        let now = Date.timeIntervalSinceReferenceDate
        glyphs.removeAll { now - $0.born > $0.duration }
        glyphs.append(contentsOf: seeds.enumerated().map { index, seed in
            let lane = CGFloat(index % 9) / 8
            return FlyingGlyph(
                character: seed.character,
                origin: seed.origin,
                destinationX: 0.18 + lane * 0.64,
                curve: CGFloat((index % 5) - 2) * 0.026,
                born: now + Double(index % 7) * 0.026,
                duration: 1.05 + Double(index % 6) * 0.07
            )
        })
        if glyphs.count > 220 { glyphs.removeFirst(glyphs.count - 220) }
    }
}
