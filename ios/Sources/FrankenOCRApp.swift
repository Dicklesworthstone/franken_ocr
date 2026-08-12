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
    }
}
