import SwiftUI

enum MonsterMood: Equatable {
    case idle
    case waking
    case working
    case success
    case error

    var isEnergized: Bool { self == .waking || self == .working }
}

enum MonsterInstrument {
    case voice
    case hearing
    case vision

    var symbol: String {
        switch self {
        case .voice: "waveform"
        case .hearing: "ear"
        case .vision: "viewfinder"
        }
    }
}

/// The suite's cute monster, expressed as native vector shapes so it remains
/// sharp in a widget-sized mark or a large desktop workspace. Motion is a
/// semantic reaction to real state; it never communicates fabricated progress.
struct MonsterStatusMark: View {
    let mood: MonsterMood
    let instrument: MonsterInstrument
    var accent: Color = Lab.accent

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        TimelineView(
            .animation(
                minimumInterval: energyConstrained ? 1.0 / 6.0 : 1.0 / 15.0,
                paused: reduceMotion || !mood.isEnergized
            )
        ) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate
            let charge = mood.isEnergized ? (0.72 + 0.28 * sin(time * 3.2)) : 0.35
            ZStack {
                Circle()
                    .fill(accent.opacity(0.10 + 0.08 * charge))
                    .blur(radius: 5)
                    .padding(-6)

                HStack(spacing: 0) {
                    Capsule().fill(Color(white: 0.28)).frame(width: 9, height: 18)
                    Spacer()
                    Capsule().fill(Color(white: 0.28)).frame(width: 9, height: 18)
                }
                .padding(.horizontal, -3)

                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [Color(red: 0.44, green: 0.88, blue: 0.48), accent.opacity(0.72)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .overlay(alignment: .top) {
                        MonsterHair().fill(Color(red: 0.04, green: 0.09, blue: 0.065)).frame(height: 17)
                    }
                    .overlay {
                        VStack(spacing: 5) {
                            HStack(spacing: 9) {
                                MonsterEye(mood: mood, phase: time)
                                MonsterEye(mood: mood, phase: time + 0.13)
                            }
                            MonsterMouth(mood: mood)
                        }
                        .offset(y: 3)
                    }
                    .overlay(alignment: .leading) {
                        Path { path in
                            path.move(to: CGPoint(x: 7, y: 25))
                            path.addLine(to: CGPoint(x: 12, y: 28))
                            path.addLine(to: CGPoint(x: 8, y: 32))
                        }
                        .stroke(Color.black.opacity(0.42), style: StrokeStyle(lineWidth: 1.2, lineCap: .round))
                    }

                Circle()
                    .fill(Color.black.opacity(0.86))
                    .overlay {
                        Image(systemName: instrument.symbol)
                            .font(.system(size: 9, weight: .black))
                            .foregroundStyle(accent)
                    }
                    .frame(width: 22, height: 22)
                    .overlay(Circle().stroke(accent.opacity(0.55), lineWidth: 1))
                    .offset(x: 18, y: 19)
            }
            .shadow(color: statusColor.opacity(0.30 + 0.20 * charge), radius: 9)
        }
        .aspectRatio(1, contentMode: .fit)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(accessibilityLabel)
    }

    private var energyConstrained: Bool {
        ProcessInfo.processInfo.isLowPowerModeEnabled
            || ProcessInfo.processInfo.thermalState.rawValue
                >= ProcessInfo.ThermalState.serious.rawValue
    }

    private var statusColor: Color {
        switch mood {
        case .idle: accent
        case .waking, .working: Lab.amber
        case .success: Lab.accent
        case .error: Lab.red
        }
    }

    private var accessibilityLabel: String {
        switch mood {
        case .idle: "Monster is ready"
        case .waking: "Monster is waking the model"
        case .working: "Monster is working"
        case .success: "Monster finished successfully"
        case .error: "Monster needs attention"
        }
    }
}

private struct MonsterEye: View {
    let mood: MonsterMood
    let phase: TimeInterval

    var body: some View {
        let blink = mood.isEnergized && phase.truncatingRemainder(dividingBy: 4.6) > 4.45
        ZStack {
            Capsule().fill(Color.white.opacity(0.94))
            Circle()
                .fill(Color.black.opacity(0.86))
                .padding(3)
                .offset(x: mood == .working ? 1.5 : 0, y: mood == .error ? 1.5 : 0)
        }
        .frame(width: 12, height: blink ? 2 : 13)
        .animation(.easeOut(duration: 0.08), value: blink)
    }
}

private struct MonsterMouth: View {
    let mood: MonsterMood
    var body: some View {
        Canvas { context, size in
            var mouth = Path()
            switch mood {
            case .success:
                mouth.move(to: CGPoint(x: 1, y: 2))
                mouth.addQuadCurve(
                    to: CGPoint(x: size.width - 1, y: 2),
                    control: CGPoint(x: size.width / 2, y: size.height)
                )
            case .error:
                mouth.move(to: CGPoint(x: 1, y: size.height - 1))
                mouth.addQuadCurve(
                    to: CGPoint(x: size.width - 1, y: size.height - 1),
                    control: CGPoint(x: size.width / 2, y: 0)
                )
            default:
                mouth.move(to: CGPoint(x: 1, y: size.height / 2))
                mouth.addLine(to: CGPoint(x: size.width - 1, y: size.height / 2))
            }
            context.stroke(mouth, with: .color(Color.black.opacity(0.64)), style: StrokeStyle(lineWidth: 1.6, lineCap: .round))
        }
        .frame(width: 16, height: 7)
    }
}

private struct MonsterHair: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: rect.origin)
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.midY))
        path.addLine(to: CGPoint(x: rect.width * 0.78, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.width * 0.61, y: rect.midY))
        path.addLine(to: CGPoint(x: rect.width * 0.43, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.width * 0.24, y: rect.midY))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
        path.closeSubpath()
        return path
    }
}
