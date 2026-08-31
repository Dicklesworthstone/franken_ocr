import AppIntents
import Foundation
import WidgetKit

#if canImport(ActivityKit) && !targetEnvironment(macCatalyst)
import ActivityKit

@MainActor
final class OCRActivityController {
    static let shared = OCRActivityController()
    private var activity: Activity<FrankenOCRRunActivityAttributes>?
    private var startedAt: Date?

    private init() {}

    func begin() {
        guard activity == nil, startedAt == nil else { return }
        let start = Date.now
        startedAt = start
        let attributes = FrankenOCRRunActivityAttributes(runID: UUID(), startedAt: start)
        let state = FrankenOCRRunContentState(
            stage: "Preparing the specimen",
            detail: "Reading pixels into private on-device memory",
            page: 0,
            pageCount: 0,
            completedUnits: 0,
            totalUnits: 0,
            totalIsEstimated: false,
            elapsedSeconds: 0,
            status: .preparing
        )
        if ActivityAuthorizationInfo().areActivitiesEnabled {
            activity = try? Activity.request(
                attributes: attributes,
                content: ActivityContent(state: state, staleDate: nil),
                pushType: nil
            )
        }
        publish(.working, headline: state.stage, detail: state.detail)
    }

    func update(
        progress: ProgressUpdate,
        page: Int,
        pageCount: Int,
        elapsed: TimeInterval,
        totalIsEstimated: Bool
    ) {
        let status: FrankenOCRRunContentState.Status
        switch progress.stage {
        case .preprocess, .vision, .prefill: status = .vision
        case .decode, .staff: status = .reading
        case .postprocess: status = .assembling
        }
        let pageDetail = pageCount > 0 ? "Page \(page) of \(pageCount) · " : ""
        let units: String
        if progress.total > 0, !totalIsEstimated {
            units = "\(progress.current) of \(progress.total) measured units"
        } else {
            units = "\(progress.current) emitted units · total not known yet"
        }
        let state = FrankenOCRRunContentState(
            stage: progress.stage.label,
            detail: pageDetail + units,
            page: page,
            pageCount: pageCount,
            completedUnits: progress.current,
            totalUnits: progress.total,
            totalIsEstimated: totalIsEstimated,
            elapsedSeconds: max(0, Int(elapsed)),
            status: status
        )
        if let activity {
            Task { await activity.update(ActivityContent(state: state, staleDate: nil)) }
        }
        publish(.working, headline: state.stage, detail: state.detail)
    }

    func finish(status: FrankenOCRRunContentState.Status, headline: String, detail: String) {
        guard activity != nil || startedAt != nil else { return }
        let current = activity
        let start = current?.attributes.startedAt ?? startedAt ?? .now
        activity = nil
        startedAt = nil
        let state = FrankenOCRRunContentState(
            stage: headline,
            detail: detail,
            page: 0,
            pageCount: 0,
            completedUnits: 0,
            totalUnits: 0,
            totalIsEstimated: false,
            elapsedSeconds: max(0, Int(Date().timeIntervalSince(start))),
            status: status
        )
        let dismissal: ActivityUIDismissalPolicy = status == .complete ? .after(.now + 45) : .immediate
        if let current {
            Task { await current.end(ActivityContent(state: state, staleDate: nil), dismissalPolicy: dismissal) }
        }
        let readiness: FrankenOCRWidgetSnapshot.Readiness
        switch status {
        case .complete: readiness = .complete
        case .failed: readiness = .needsAttention
        default: readiness = .ready
        }
        publish(readiness, headline: headline, detail: detail)
    }

    private func publish(
        _ readiness: FrankenOCRWidgetSnapshot.Readiness,
        headline: String,
        detail: String
    ) {
        FrankenOCRSharedStore.save(
            FrankenOCRWidgetSnapshot(
                readiness: readiness,
                headline: headline,
                detail: detail,
                updatedAt: .now
            )
        )
        WidgetCenter.shared.reloadTimelines(ofKind: "FrankenOCRVisionWidget")
    }
}
#else
@MainActor
final class OCRActivityController {
    static let shared = OCRActivityController()
    private init() {}
    func begin() {}
    func update(
        progress: ProgressUpdate,
        page: Int,
        pageCount: Int,
        elapsed: TimeInterval,
        totalIsEstimated: Bool
    ) {}
    func finish(status: FrankenOCRRunContentState.Status, headline: String, detail: String) {}
}
#endif

struct OpenOCRIntent: AppIntent {
    static let title: LocalizedStringResource = "Recognize a Document"
    static let description = IntentDescription("Open FrankenOCR to recognize an image or PDF privately.")
    static let openAppWhenRun = true
    @MainActor func perform() async throws -> some IntentResult {
        FrankenOCRSharedStore.request(.recognize)
        return .result()
    }
}

struct OpenLiveCameraIntent: AppIntent {
    static let title: LocalizedStringResource = "Open Live Camera"
    static let description = IntentDescription(
        "Open FrankenOCR's fast on-device Vision camera. Camera capture begins only after you open the view."
    )
    static let openAppWhenRun = true
    @MainActor func perform() async throws -> some IntentResult {
        FrankenOCRSharedStore.request(.liveCamera)
        return .result()
    }
}

struct FrankenOCRShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: OpenOCRIntent(),
            phrases: ["Read a document with \(.applicationName)"],
            shortTitle: "Recognize Document",
            systemImageName: "doc.viewfinder"
        )
        AppShortcut(
            intent: OpenLiveCameraIntent(),
            phrases: ["Open Live Camera in \(.applicationName)"],
            shortTitle: "Live Camera",
            systemImageName: "viewfinder"
        )
    }
}
