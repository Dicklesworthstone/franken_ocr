import SwiftUI
import UIKit

@main
struct FrankenOCRApp: App {
    var body: some Scene {
        WindowGroup {
            LabView()
                .background(CatalystWindowFreedom())
#if targetEnvironment(macCatalyst)
                .frame(minWidth: 480, minHeight: 420)
#endif
        }
#if targetEnvironment(macCatalyst)
        .defaultSize(width: 1180, height: 820)
        .windowResizability(.contentMinSize)
#endif
        .commands {
            OCRCommands()
            OCRTextSizeCommands()
        }
    }
}

private struct CatalystWindowFreedom: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> Controller { Controller() }
    func updateUIViewController(_ controller: Controller, context: Context) { controller.configure() }

    final class Controller: UIViewController {
        override func viewDidAppear(_ animated: Bool) {
            super.viewDidAppear(animated)
            configure()
        }

        override func viewDidLayoutSubviews() {
            super.viewDidLayoutSubviews()
            configure()
        }

        func configure() {
#if targetEnvironment(macCatalyst)
            guard let restrictions = view.window?.windowScene?.sizeRestrictions else { return }
            restrictions.minimumSize = CGSize(width: 480, height: 420)
            restrictions.maximumSize = CGSize(width: 10_000, height: 10_000)
#endif
        }
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
        // Replace Catalyst's built-in Open command instead of adding a second
        // Command-O key equivalent. Duplicate shortcuts produce undefined menu
        // dispatch and were reported by UIKit during the real test launch.
        CommandGroup(replacing: .newItem) {
            Button("Open Image or PDF…") { actions?.importFile() }
                .keyboardShortcut("o", modifiers: [.command])
        }

        CommandMenu("Recognition") {
            Button("Recognize") { actions?.recognize() }
                .keyboardShortcut(.return, modifiers: [.command])
                .disabled(actions?.canRecognize != true)

#if !targetEnvironment(macCatalyst)
            Button("Open Live Camera") { actions?.liveCamera() }
                .keyboardShortcut("l", modifiers: [.command, .shift])
#endif

            Divider()

            Button("Stop Recognition") { actions?.stop() }
                .keyboardShortcut(.escape, modifiers: [])
                .disabled(actions?.canStop != true)
        }
    }
}

private struct OCRTextSizeCommands: Commands {
    @AppStorage(LabTextScale.storageKey) private var textScale = LabTextScale.defaultValue

    var body: some Commands {
        CommandMenu("Text Size") {
            Button("Make Text Bigger") {
                textScale = LabTextScale.adjusted(textScale, by: 1)
            }
            .keyboardShortcut("+", modifiers: [.command])
            .disabled(LabTextScale.clamped(textScale) >= LabTextScale.maximum)

            Button("Make Text Smaller") {
                textScale = LabTextScale.adjusted(textScale, by: -1)
            }
            .keyboardShortcut("-", modifiers: [.command])
            .disabled(LabTextScale.clamped(textScale) <= LabTextScale.minimum)

            Divider()

            Button("Actual Size") {
                textScale = LabTextScale.defaultValue
            }
            .keyboardShortcut("0", modifiers: [.command])
            .disabled(abs(textScale - LabTextScale.defaultValue) < 0.001)
        }
    }
}
