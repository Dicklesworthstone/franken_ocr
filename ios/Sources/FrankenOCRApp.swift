import SwiftUI

@main
struct FrankenOCRApp: App {
    var body: some Scene {
        WindowGroup {
            LabView()
                // The design system is dark-only by construction — every
                // surface is authored against #060b09 and there is no light
                // palette to fall back to.
                .preferredColorScheme(.dark)
        }
        .commands { OCRCommands() }
    }
}

struct OCRCommandActions {
    let importFile: () -> Void
    let recognize: () -> Void
    let liveCamera: () -> Void
    let stop: () -> Void
    let canRecognize: Bool
    let canStop: Bool
}

private struct OCRCommandKey: FocusedValueKey {
    typealias Value = OCRCommandActions
}

extension FocusedValues {
    var ocrCommands: OCRCommandActions? {
        get { self[OCRCommandKey.self] }
        set { self[OCRCommandKey.self] = newValue }
    }
}

private struct OCRCommands: Commands {
    @FocusedValue(\.ocrCommands) private var actions

    var body: some Commands {
        CommandMenu("Recognition") {
            Button("Open Image or PDF…") { actions?.importFile() }
                .keyboardShortcut("o", modifiers: [.command])

            Divider()

            Button("Recognize") { actions?.recognize() }
                .keyboardShortcut(.return, modifiers: [.command])
                .disabled(actions?.canRecognize != true)

            Button("Open Live Camera") { actions?.liveCamera() }
                .keyboardShortcut("l", modifiers: [.command, .shift])

            Divider()

            Button("Stop Recognition") { actions?.stop() }
                .keyboardShortcut(.escape, modifiers: [])
                .disabled(actions?.canStop != true)
        }
    }
}
