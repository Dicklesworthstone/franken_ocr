import SwiftUI
import UIKit

/// The franken-ocr.com design system, extended with a warm laboratory-paper
/// light palette for native surfaces while preserving the site's semantics.
///
/// Colors carry FIXED semantics here exactly as they do on the site and in
/// `viz.js`: emerald is "ours / active / exact", amber is "warning /
/// threshold", red is "failed / skipped / refused", slate is "the reference we
/// are measured against", violet is "structure / shape / metadata". A color
/// used off-meaning is a bug, not a style choice.
enum LabAppearance: String {
    static let storageKey = "frankenocr.appearance"
    case dark
    case light
    var colorScheme: ColorScheme { self == .dark ? .dark : .light }
}

enum LabTextScale {
    static let storageKey = "frankenocr.uiTextScale"
    static let defaultValue = 1.0
    static let minimum = 0.8
    static let maximum = 1.5
    static let increment = 0.1

    static var current: CGFloat {
        let stored = UserDefaults.standard.object(forKey: storageKey) as? NSNumber
        return CGFloat(clamped(stored?.doubleValue ?? defaultValue))
    }

    static func adjusted(_ value: Double, by steps: Int) -> Double {
        clamped(clamped(value) + Double(steps) * increment)
    }

    static func clamped(_ value: Double) -> Double {
        min(max(value, minimum), maximum)
    }
}

enum Lab {
    // Surfaces
    static let background = adaptive(dark: 0x060B09, light: 0xF1F7F2)
    static let backgroundDeep = adaptive(dark: 0x030706, light: 0xE3EFE6)
    static let panel = adaptive(dark: 0xFFFFFF, light: 0xFFFFFF, darkAlpha: 0.022, lightAlpha: 0.94)
    static let panelStrong = adaptive(dark: 0x030706, light: 0xF9FCF9, darkAlpha: 0.72, lightAlpha: 0.98)
    static let inset = adaptive(dark: 0x020605, light: 0xDFECE2, darkAlpha: 0.66, lightAlpha: 0.92)

    // Text
    static let textPrimary = adaptive(dark: 0xE8EEF2, light: 0x12231A)
    static let textMid = adaptive(dark: 0xCBD5E1, light: 0x263B30)
    static let textDim = adaptive(dark: 0x94A3B8, light: 0x40584B)
    /// The site picked this over `#64748b` deliberately: 5.18:1 on the page
    /// background, where the darker slate measured 4.16:1 and failed AA.
    static let textFaint = adaptive(dark: 0x748496, light: 0x53695D)

    // Semantics
    static let accent = adaptive(dark: 0x34D399, light: 0x067A50)
    static let accentDeep = adaptive(dark: 0x059669, light: 0x05633F)
    /// Eyebrows, section labels, table headers — a step deeper than `accent`.
    static let accentInk = adaptive(dark: 0x10B981, light: 0x066E48)
    /// Text drawn ON an accent fill.
    static let onAccent = Color(hex: 0x04140D)
    static let amber = adaptive(dark: 0xFBBF24, light: 0x9A5A00)
    static let red = adaptive(dark: 0xF87171, light: 0xB4232C)
    static let reference = adaptive(dark: 0x7E8E9F, light: 0x4A6172)
    static let violet = adaptive(dark: 0xA78BFA, light: 0x6540A8)

    static let line = adaptive(dark: 0xFFFFFF, light: 0x16452D, darkAlpha: 0.07, lightAlpha: 0.15)
    static let lineStrong = adaptive(dark: 0x34D399, light: 0x067A50, darkAlpha: 0.26, lightAlpha: 0.30)

    private static func adaptive(
        dark: UInt32,
        light: UInt32,
        darkAlpha: CGFloat = 1,
        lightAlpha: CGFloat = 1
    ) -> Color {
        Color(uiColor: UIColor { traits in
            UIColor(hex: traits.userInterfaceStyle == .dark ? dark : light)
                .withAlphaComponent(traits.userInterfaceStyle == .dark ? darkAlpha : lightAlpha)
        })
    }

    static let radius: CGFloat = 12
    static let radiusLarge: CGFloat = 18

    static func typeSize(_ base: CGFloat) -> CGFloat {
        contentTypeSize(base * LabTextScale.current)
    }

    /// Sizing for recognized/source document content. Browser-style UI zoom
    /// must not silently rewrite the user's preferred reading/output scale.
    static func contentTypeSize(_ base: CGFloat) -> CGFloat {
#if targetEnvironment(macCatalyst)
        return base * 1.38
#else
        return UIFontMetrics(forTextStyle: .body).scaledValue(for: base)
#endif
    }
}

struct LabAppearanceButton: View {
    @Binding var selection: String
    private var appearance: LabAppearance { LabAppearance(rawValue: selection) ?? .dark }

    var body: some View {
        Button {
            selection = appearance == .dark ? LabAppearance.light.rawValue : LabAppearance.dark.rawValue
        } label: {
            Image(systemName: appearance == .dark ? "sun.max.fill" : "moon.stars.fill")
                .font(.system(size: Lab.typeSize(15), weight: .bold))
                .frame(width: 44, height: 44)
                .background(Lab.panelStrong, in: Circle())
                .overlay(Circle().stroke(Lab.line))
        }
        .buttonStyle(.plain)
        .foregroundStyle(appearance == .dark ? Lab.amber : Lab.accentDeep)
        .accessibilityIdentifier("appearance-toggle")
        .accessibilityLabel(appearance == .dark ? "Switch to light mode" : "Switch to dark mode")
        .accessibilityValue(appearance == .dark ? "Dark mode" : "Light mode")
        .accessibilityHint("Remembers this choice for future launches")
    }
}

/// The suite wordmark restores the name's natural F/OCR hierarchy without
/// giving up its uppercase laboratory typography.
struct FrankenWordmark: View {
    let productInitial: String
    let productRemainder: String
    let fullName: String
    var size: CGFloat = 20
    var accent: Color = Lab.amber

    var body: some View {
        (
            Text("F")
                .font(.system(size: Lab.typeSize(size), weight: .black, design: .monospaced))
                .foregroundColor(Lab.textPrimary.opacity(0.88))
            + Text("RANKEN")
                .font(.system(size: Lab.typeSize(size * 0.66), weight: .black, design: .monospaced))
                .foregroundColor(Lab.textPrimary.opacity(0.88))
            + Text(productInitial)
                .font(.system(size: Lab.typeSize(size), weight: .black, design: .monospaced))
                .foregroundColor(accent)
            + Text(productRemainder)
                .font(.system(size: Lab.typeSize(size * 0.66), weight: .black, design: .monospaced))
                .foregroundColor(accent)
        )
        .kerning(0.8)
        .lineLimit(1)
        .minimumScaleFactor(0.72)
        .allowsTightening(true)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(fullName)
    }
}

private struct CatalystReadableType: ViewModifier {
    func body(content: Content) -> some View {
#if targetEnvironment(macCatalyst)
        content.dynamicTypeSize(.xLarge)
#else
        content
#endif
    }
}

extension View {
    func catalystReadableType() -> some View {
        modifier(CatalystReadableType())
    }
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

private extension UIColor {
    convenience init(hex: UInt32) {
        self.init(
            red: CGFloat((hex >> 16) & 0xFF) / 255,
            green: CGFloat((hex >> 8) & 0xFF) / 255,
            blue: CGFloat(hex & 0xFF) / 255,
            alpha: 1
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
                    wash(Lab.accentInk.opacity(0.13), at: .init(x: 0.12, y: -0.06),
                         size: CGSize(width: w * 1.5, height: h * 0.8))
                    wash(Lab.accentDeep.opacity(0.20), at: .init(x: 0.96, y: 0.22),
                         size: CGSize(width: w * 1.2, height: h * 0.9))
                    wash(Lab.amber.opacity(0.05), at: .init(x: 0.40, y: 1.08),
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
            .font(.system(size: Lab.typeSize(11), weight: .heavy, design: .monospaced))
            .kerning(2.4)
            .foregroundStyle(Lab.accentInk)
    }
}

/// A Frankenstein bolt stud. The site draws these at the top-left and
/// bottom-right of every panel via `::before`/`::after`, and it is the single
/// most recognizable piece of the FrankenSuite identity — the same mark appears
/// on frankentts and franken-markdown.
struct Bolt: View {
    var size: CGFloat = 11
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
            Capsule()
                .frame(width: size * 0.56, height: 1.2)
                .rotationEffect(.degrees(-28))
                .foregroundStyle(Color.black.opacity(0.68))
        }
        .frame(width: size, height: size)
        .shadow(color: Color.black.opacity(0.55), radius: 2, y: 1)
        .accessibilityHidden(true)
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
            .font(.system(size: Lab.typeSize(12), weight: .heavy, design: .monospaced))
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
            .font(.system(size: Lab.typeSize(12), weight: .heavy, design: .monospaced))
            .kerning(1.2)
            .textCase(.uppercase)
            // Keep controls causal and scannable at accessibility sizes. Giving
            // the label an honest single-line intrinsic width also lets the
            // surrounding ViewThatFits choose its two-row layout instead of
            // accepting an ugly mid-word wrap such as "PHOTO" / "S".
            .lineLimit(1)
            .minimumScaleFactor(0.72)
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
        .font(.system(size: Lab.typeSize(12), design: .monospaced))
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
                .font(.system(size: Lab.typeSize(12), design: .monospaced))
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
