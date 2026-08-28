import SwiftUI

/// The document machine made visible. Counts and stage transitions are native
/// engine facts; motion is presentation and never advances completion itself.
struct OcrReactorView: View {
    let progress: ProgressUpdate?
    let fraction: Double
    let isEstimated: Bool
    let elapsed: TimeInterval
    let estimatedRemainingSeconds: Int?
    let currentPage: Int?
    let pageCount: Int
    let completedPages: Int
    let emittedCharacters: Int
    let cancel: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.isLuminanceReduced) private var luminanceReduced

    private var stage: RecognitionStage? { progress?.stage }

    private var headline: String {
        stage?.label ?? (currentPage == nil ? "Waking the model" : "Rendering the page")
    }

    private var explanation: String {
        switch stage {
        case .preprocess:
            "Image pixels are normalized, resized, and arranged for the visual encoder."
        case .vision:
            "The high-precision vision tower is converting page structure into visual tokens."
        case .prefill:
            "Visual tokens and the recognition prompt are entering the language decoder."
        case .decode:
            "The local decoder is emitting document tokens until it reaches a real end marker."
        case .postprocess:
            "Model output is becoming grounded layout spans and exportable document text."
        case .staff:
            "Detected music staves are being processed into structured notation."
        case nil:
            "Verified weights are being mapped into the on-device inference engine."
        }
    }

    private var activeModule: Int {
        switch stage {
        case .preprocess: 0
        case .vision: 1
        case .prefill: 2
        case .decode: 3
        case .postprocess, .staff: 4
        case nil: 0
        }
    }

    var body: some View {
        reactorCard
            .accessibilityElement(children: .contain)
            .accessibilityLabel("OCR processing reactor")
            .accessibilityValue("\(headline). \(explanation)")
    }

    private var reactorCard: some View {
        reactorTimeline
            .padding(17)
            .background { reactorBackground }
            .overlay { reactorBorder }
            .shadow(color: Lab.accent.opacity(0.18), radius: 28, y: 14)
    }

    private var reactorTimeline: some View {
        TimelineView(
            .animation(
                minimumInterval: animationConstrained ? 1.0 / 12.0 : 1.0 / 24.0,
                paused: reduceMotion
            )
        ) { timeline in
            reactorContent(at: timeline.date)
        }
    }

    private var reactorBackground: some View {
        RoundedRectangle(cornerRadius: 22, style: .continuous)
            .fill(reduceTransparency ? Lab.backgroundDeep : Lab.inset.opacity(0.92))
            .overlay {
                LinearGradient(
                    colors: [Lab.accent.opacity(0.1), .clear, Lab.violet.opacity(0.07)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
            }
    }

    private var reactorBorder: some View {
        RoundedRectangle(cornerRadius: 22, style: .continuous)
            .strokeBorder(
                LinearGradient(
                    colors: [Lab.accent.opacity(0.55), Lab.line, Lab.violet.opacity(0.2)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                lineWidth: 1
            )
    }

    @ViewBuilder
    private func reactorContent(at date: Date) -> some View {
        VStack(alignment: .leading, spacing: 15) {
            reactorHeader

            reactorCanvas(time: date.timeIntervalSinceReferenceDate)
                .frame(height: 226)
                .accessibilityHidden(true)

            LabProgressBar(fraction: fraction)
            metricRow

            if isEstimated {
                Label(
                    "Decode completion is an estimate; token count is exact and EOS is unknowable in advance.",
                    systemImage: "waveform.path.ecg"
                )
                .font(.system(size: Lab.typeSize(10), design: .monospaced))
                .foregroundStyle(Lab.textFaint)
            }

            Button(role: .cancel, action: cancel) {
                Label("Cancel recognition", systemImage: "stop.fill")
            }
            .buttonStyle(GhostButtonStyle(tint: Lab.red))
        }
    }

    private var reactorHeader: some View {
        HStack(alignment: .top, spacing: 12) {
            ZStack {
                Circle().fill(Lab.accent.opacity(0.12))
                Circle().strokeBorder(Lab.accent.opacity(0.52), lineWidth: 1)
                Image(systemName: symbol)
                    .font(.system(size: Lab.typeSize(18), weight: .bold))
                    .foregroundStyle(Lab.accent)
                    .symbolEffect(.pulse, isActive: !reduceMotion)
            }
            .frame(width: 44, height: 44)

            VStack(alignment: .leading, spacing: 4) {
                Text(headline.uppercased())
                    .font(.system(size: Lab.typeSize(13), weight: .heavy, design: .monospaced))
                    .tracking(1.4)
                    .foregroundStyle(Lab.textPrimary)
                Text(explanation)
                    .font(.system(size: Lab.typeSize(12)))
                    .foregroundStyle(Lab.textDim)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)
            VStack(alignment: .trailing, spacing: 3) {
                Text(Self.clock(elapsed))
                    .foregroundStyle(Lab.textFaint)
                if let estimatedRemainingSeconds {
                    Text("≈ \(estimatedRemainingSeconds)s left")
                        .foregroundStyle(Lab.accent)
                        .contentTransition(.numericText())
                        .accessibilityLabel(
                            "About \(estimatedRemainingSeconds) seconds remaining"
                        )
                }
            }
            .font(.system(size: Lab.typeSize(12), weight: .semibold, design: .monospaced))
            .monospacedDigit()
        }
    }

    private var metricRow: some View {
        HStack(spacing: 8) {
            metric(
                "PAGE",
                pageCount > 0
                    ? "\(currentPage ?? min(pageCount, completedPages + 1))/\(pageCount)"
                    : "1/1"
            )
            metric("UNITS", unitValue)
            metric("TEXT", emittedCharacters > 0 ? emittedCharacters.formatted() : "—")
            metric("PROGRESS", "\(Int(fraction * 100))%\(isEstimated ? "≈" : "")")
        }
    }

    private var animationConstrained: Bool {
        reduceMotion
            || luminanceReduced
            || ProcessInfo.processInfo.isLowPowerModeEnabled
            || ProcessInfo.processInfo.thermalState.rawValue
                >= ProcessInfo.ThermalState.serious.rawValue
    }

    private var symbol: String {
        switch stage {
        case .preprocess: "photo.badge.arrow.down"
        case .vision: "eye.fill"
        case .prefill: "square.stack.3d.up.fill"
        case .decode: "text.word.spacing"
        case .postprocess: "doc.richtext.fill"
        case .staff: "music.note.list"
        case nil: "bolt.horizontal.circle.fill"
        }
    }

    private var unitValue: String {
        guard let progress else { return "—" }
        if progress.stage == .decode || progress.total == 0 {
            return progress.current.formatted()
        }
        return "\(progress.current)/\(progress.total)"
    }

    private func metric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.system(size: Lab.typeSize(8), weight: .bold, design: .monospaced))
                .tracking(1)
                .foregroundStyle(Lab.textFaint)
            Text(value)
                .font(.system(size: Lab.typeSize(11), weight: .semibold, design: .monospaced))
                .foregroundStyle(Lab.textMid)
                .monospacedDigit()
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.035), in: RoundedRectangle(cornerRadius: 9))
    }

    private func reactorCanvas(time: TimeInterval) -> some View {
        Canvas(opaque: false, colorMode: .linear, rendersAsynchronously: true) { context, size in
            let bounds = CGRect(origin: .zero, size: size).insetBy(dx: 1, dy: 1)
            context.fill(
                Path(roundedRect: bounds, cornerRadius: 17),
                with: .linearGradient(
                    Gradient(colors: [Color.black.opacity(0.58), Lab.violet.opacity(0.045)]),
                    startPoint: .zero,
                    endPoint: CGPoint(x: size.width, y: size.height)
                )
            )

            drawPage(context: &context, size: size)
            drawPipeline(context: &context, size: size, time: time)
            drawTokens(context: &context, size: size)
        }
        .overlay(alignment: .topLeading) { instrumentLabel("SOURCE PIXELS").padding(12) }
        .overlay(alignment: .topTrailing) { instrumentLabel("DOCUMENT OUTPUT").padding(12) }
    }

    private func drawPage(context: inout GraphicsContext, size: CGSize) {
        let page = CGRect(x: 20, y: 42, width: max(50, size.width * 0.18), height: size.height - 76)
        context.fill(Path(roundedRect: page, cornerRadius: 6), with: .color(Lab.textMid.opacity(0.08)))
        context.stroke(Path(roundedRect: page, cornerRadius: 6), with: .color(Lab.accent.opacity(0.28)), lineWidth: 1)
        for row in 0..<8 {
            let width = page.width * (row % 3 == 2 ? 0.58 : 0.78)
            let line = CGRect(x: page.minX + 9, y: page.minY + 14 + CGFloat(row) * 14, width: width, height: 3)
            context.fill(Path(roundedRect: line, cornerRadius: 1.5), with: .color(Lab.textFaint.opacity(0.2)))
        }
    }

    private func drawPipeline(context: inout GraphicsContext, size: CGSize, time: TimeInterval) {
        let startX = size.width * 0.28
        let endX = size.width * 0.77
        let y = size.height * 0.53
        let moduleCount = 5
        var centers: [CGPoint] = []
        for index in 0..<moduleCount {
            let x = startX + (endX - startX) * CGFloat(index) / CGFloat(moduleCount - 1)
            centers.append(CGPoint(x: x, y: y))
        }

        var rail = Path()
        rail.move(to: centers[0])
        for point in centers.dropFirst() { rail.addLine(to: point) }
        context.stroke(rail, with: .color(Lab.accent.opacity(0.2)), lineWidth: 1.5)

        for (index, center) in centers.enumerated() {
            let energized = index <= activeModule
            let radius: CGFloat = index == activeModule ? 17 : 12
            context.fill(
                Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2)),
                with: .color((energized ? Lab.accent : Lab.reference).opacity(energized ? 0.24 : 0.07))
            )
            context.stroke(
                Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2)),
                with: .color((energized ? Lab.accent : Lab.reference).opacity(energized ? 0.8 : 0.18)),
                lineWidth: index == activeModule ? 2 : 1
            )
        }

        if !reduceMotion {
            let travel = CGFloat(time.truncatingRemainder(dividingBy: 1.35) / 1.35)
            let x = startX + (endX - startX) * travel
            let particle = CGRect(x: x - 3, y: y - 3, width: 6, height: 6)
            context.fill(Path(ellipseIn: particle), with: .color(Lab.accent))
        }
    }

    private func drawTokens(context: inout GraphicsContext, size: CGSize) {
        // One visible cell per real reported unit, capped by the instrument's
        // viewport. The exact uncapped count remains in the metric row.
        let count = min(16, Int(progress?.current ?? 0))
        let originX = size.width * 0.82
        let width = max(5, (size.width - originX - 22) / 4 - 3)
        for index in 0..<count {
            let column = index % 4
            let row = index / 4
            let frame = CGRect(
                x: originX + CGFloat(column) * (width + 3),
                y: 54 + CGFloat(row) * 19,
                width: width,
                height: 12
            )
            context.fill(
                Path(roundedRect: frame, cornerRadius: 3),
                with: .color((index == count - 1 ? Lab.violet : Lab.accent).opacity(0.55))
            )
        }
    }

    private func instrumentLabel(_ text: String) -> some View {
        Text(text)
            .font(.system(size: Lab.typeSize(8), weight: .bold, design: .monospaced))
            .tracking(1.1)
            .foregroundStyle(Lab.textFaint)
    }

    private static func clock(_ seconds: TimeInterval) -> String {
        let whole = max(0, Int(seconds.rounded(.down)))
        return String(format: "%d:%02d", whole / 60, whole % 60)
    }
}
