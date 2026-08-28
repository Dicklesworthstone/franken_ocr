import SwiftUI

/// The franken-ocr.com design system, ported.
///
/// The site is dark-only by construction — every surface is authored against
/// `#060b09` and there is no light palette to map onto. Rather than invent one,
/// the app forces dark too, which also keeps the one visual identity across
/// web and phone.
///
/// Colors carry FIXED semantics here exactly as they do on the site and in
/// `viz.js`: emerald is "ours / active / exact", amber is "warning /
/// threshold", red is "failed / skipped / refused", slate is "the reference we
/// are measured against", violet is "structure / shape / metadata". A color
/// used off-meaning is a bug, not a style choice.
enum Lab {
    // Surfaces
    static let background = Color(hex: 0x060B09)
    static let backgroundDeep = Color(hex: 0x030706)
    static let panel = Color.white.opacity(0.022)
    static let panelStrong = Color(hex: 0x030706).opacity(0.72)
    static let inset = Color(hex: 0x020605).opacity(0.66)

    // Text
    static let textPrimary = Color(hex: 0xE8EEF2)
    static let textMid = Color(hex: 0xCBD5E1)
    static let textDim = Color(hex: 0x94A3B8)
    /// The site picked this over `#64748b` deliberately: 5.18:1 on the page
    /// background, where the darker slate measured 4.16:1 and failed AA.
    static let textFaint = Color(hex: 0x748496)

    // Semantics
    static let accent = Color(hex: 0x34D399)
    static let accentDeep = Color(hex: 0x059669)
    /// Eyebrows, section labels, table headers — a step deeper than `accent`.
    static let accentInk = Color(hex: 0x10B981)
    /// Text drawn ON an accent fill.
    static let onAccent = Color(hex: 0x04140D)
    static let amber = Color(hex: 0xFBBF24)
    static let red = Color(hex: 0xF87171)
    static let reference = Color(hex: 0x7E8E9F)
    static let violet = Color(hex: 0xA78BFA)

    static let line = Color.white.opacity(0.07)
    static let lineStrong = Color(hex: 0x34D399).opacity(0.26)

    static let radius: CGFloat = 12
    static let radiusLarge: CGFloat = 18
}

extension Color {
    init(hex: UInt32) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255,
            opacity: 1
        )
    }
}

/// The page's fixed background wash: three radial gradients that never scroll,
/// matching `body::before` on the site.
struct LabBackground: View {
    var body: some View {
        ZStack {
            Lab.background
            GeometryReader { geo in
                let w = geo.size.width
                let h = geo.size.height
                ZStack {
                    wash(Color(hex: 0x10B981).opacity(0.13), at: .init(x: 0.12, y: -0.06),
                         size: CGSize(width: w * 1.5, height: h * 0.8))
                    wash(Color(hex: 0x044225).opacity(0.34), at: .init(x: 0.96, y: 0.22),
                         size: CGSize(width: w * 1.2, height: h * 0.9))
                    wash(Color(hex: 0xFBBF24).opacity(0.05), at: .init(x: 0.40, y: 1.08),
                         size: CGSize(width: w, height: h * 0.7))
                }
            }
        }
        .ignoresSafeArea()
    }

    private func wash(_ color: Color, at unit: UnitPoint, size: CGSize) -> some View {
        RadialGradient(colors: [color, .clear], center: unit, startRadius: 0,
                       endRadius: max(size.width, size.height) * 0.7)
    }
}

/// Uppercase, wide-tracked, monospaced emerald — the site's `.label` /
/// `.eyebrow`. Used for the "01 · The specimen" step captions.
struct LabLabel: View {
    let text: String
    var body: some View {
        Text(text.uppercased())
            .font(.system(size: 11, weight: .heavy, design: .monospaced))
            .kerning(2.4)
            .foregroundStyle(Lab.accentInk)
    }
}

/// A Frankenstein bolt stud. The site draws these at the top-left and
/// bottom-right of every panel via `::before`/`::after`, and it is the single
/// most recognizable piece of the FrankenSuite identity — the same mark appears
/// on frankentts and franken-markdown.
struct Bolt: View {
    var size: CGFloat = 13
    var body: some View {
        ZStack {
            Circle()
                .fill(
                    RadialGradient(
                        colors: [Color(hex: 0x64748B), Color(hex: 0x020617)],
                        center: .init(x: 0.35, y: 0.3),
                        startRadius: 0,
                        endRadius: size * 0.7
                    )
                )
                .overlay(Circle().strokeBorder(.white.opacity(0.18), lineWidth: 1))
            // The crossed slot, drawn as two rotated bars.
            Group {
                Capsule().frame(width: size * 0.62, height: 1.6).rotationEffect(.degrees(45))
                Capsule().frame(width: size * 0.62, height: 1.6).rotationEffect(.degrees(-45))
            }
            .foregroundStyle(Color(hex: 0x1E293B))
        }
        .frame(width: size, height: size)
        .shadow(color: Lab.accent.opacity(0.22), radius: 5)
    }
}

/// The site's `.panel`: an inset dark surface, a hairline emerald-tinted
/// border, and the two bolts.
struct LabPanel<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(18)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: Lab.radiusLarge))
            .overlay(
                RoundedRectangle(cornerRadius: Lab.radiusLarge)
                    .strokeBorder(Lab.line, lineWidth: 1)
            )
            .overlay(alignment: .topLeading) { Bolt().offset(x: -5, y: -5) }
            .overlay(alignment: .bottomTrailing) { Bolt().offset(x: 5, y: 5) }
            .shadow(color: .black.opacity(0.45), radius: 22, y: 12)
    }
}

/// Uppercase mono on an emerald capsule — the site's primary button.
struct PrimaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .heavy, design: .monospaced))
            .kerning(1.2)
            .textCase(.uppercase)
            .foregroundStyle(Lab.onAccent)
            .padding(.vertical, 13)
            .padding(.horizontal, 20)
            // 44pt minimum touch target, as the site enforces under
            // `@media (pointer: coarse)`.
            .frame(minHeight: 44)
            .background(
                LinearGradient(colors: [Lab.accent, Lab.accentDeep],
                               startPoint: .topLeading, endPoint: .bottomTrailing),
                in: Capsule()
            )
            .opacity(isEnabled ? (configuration.isPressed ? 0.75 : 1) : 0.35)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
            .hoverEffect(.highlight)
    }
}

/// Outlined capsule; `tint` carries the semantic (emerald normal, red for
/// destructive, amber for a caution).
struct GhostButtonStyle: ButtonStyle {
    var tint: Color = Lab.accent
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .heavy, design: .monospaced))
            .kerning(1.2)
            .textCase(.uppercase)
            .foregroundStyle(tint)
            .padding(.vertical, 13)
            .padding(.horizontal, 18)
            .frame(minHeight: 44)
            .background(tint.opacity(configuration.isPressed ? 0.16 : 0.06), in: Capsule())
            .overlay(Capsule().strokeBorder(tint.opacity(0.45), lineWidth: 1))
            .opacity(isEnabled ? 1 : 0.35)
            .hoverEffect(.highlight)
    }
}

/// A monospaced key/value line, the site's `.kv`.
struct KeyValueLine: View {
    let key: String
    let value: String
    var valueColor: Color = Lab.textMid

    var body: some View {
        HStack(spacing: 8) {
            Text(key)
                .foregroundStyle(Lab.textFaint)
            Text(value)
                .foregroundStyle(valueColor)
                .textSelection(.enabled)
            Spacer(minLength: 0)
        }
        .font(.system(size: 12, design: .monospaced))
    }
}

/// The status line: mono text behind a 2pt colored left border whose color is
/// the state. Straight from the site's `#status`.
struct StatusLine: View {
    enum Kind { case neutral, ok, warn, err

        var color: Color {
            switch self {
            case .neutral: Lab.textFaint
            case .ok: Lab.accent
            case .warn: Lab.amber
            case .err: Lab.red
            }
        }
    }

    let kind: Kind
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Rectangle().frame(width: 2).foregroundStyle(kind.color)
            Text(text)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(kind == .neutral ? Lab.textDim : kind.color)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }
}

/// A determinate bar in the site's emerald, over the inset well.
struct LabProgressBar: View {
    /// 0...1. Clamped, and never drawn at a full 1 by the caller — see
    /// `LabModel.progressFraction`.
    let fraction: Double

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Lab.inset)
                Capsule()
                    .fill(LinearGradient(colors: [Lab.accentDeep, Lab.accent],
                                         startPoint: .leading, endPoint: .trailing))
                    .frame(width: max(0, min(1, fraction)) * geo.size.width)
            }
        }
        .frame(height: 8)
        .overlay(Capsule().strokeBorder(Lab.line, lineWidth: 1))
        .animation(.easeOut(duration: 0.25), value: fraction)
    }
}
