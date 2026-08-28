import Foundation

#if !targetEnvironment(macCatalyst)
import ActivityKit
#endif

struct FrankenOCRRunContentState: Codable, Hashable {
    enum Status: String, Codable, Hashable {
        case preparing
        case vision
        case reading
        case assembling
        case complete
        case cancelled
        case failed
    }

    var stage: String
    var detail: String
    var page: Int
    var pageCount: Int
    var completedUnits: UInt64
    var totalUnits: UInt64
    var totalIsEstimated: Bool
    var elapsedSeconds: Int
    var status: Status
}

#if !targetEnvironment(macCatalyst)
struct FrankenOCRRunActivityAttributes: ActivityAttributes {
    typealias ContentState = FrankenOCRRunContentState
    var runID: UUID
    var startedAt: Date
}
#else
struct FrankenOCRRunActivityAttributes {
    typealias ContentState = FrankenOCRRunContentState
    var runID: UUID
    var startedAt: Date
}
#endif

struct FrankenOCRWidgetSnapshot: Codable, Hashable {
    enum Readiness: String, Codable {
        case modelRequired
        case ready
        case working
        case complete
        case needsAttention
    }
    var readiness: Readiness
    var headline: String
    var detail: String
    var updatedAt: Date

    static let placeholder = FrankenOCRWidgetSnapshot(
        readiness: .ready,
        headline: "Open the Vision Table",
        detail: "Recognize text privately on this device",
        updatedAt: .now
    )
}

enum FrankenOCRSharedStore {
    static let suiteName = "group.com.frankenocr.FrankenOCR"
    private static let snapshotKey = "widget.snapshot.v1"
    private static let requestedActionKey = "intent.requested-action.v1"
    private static let stagedDocumentKey = "share.staged-document.v1"

    enum RequestedAction: String {
        case liveCamera
        case recognize
    }

    static func loadSnapshot() -> FrankenOCRWidgetSnapshot {
        guard let defaults = UserDefaults(suiteName: suiteName),
              let data = defaults.data(forKey: snapshotKey),
              let snapshot = try? JSONDecoder().decode(FrankenOCRWidgetSnapshot.self, from: data)
        else { return .placeholder }
        return snapshot
    }

    static func save(_ snapshot: FrankenOCRWidgetSnapshot) {
        guard let defaults = UserDefaults(suiteName: suiteName),
              let data = try? JSONEncoder().encode(snapshot)
        else { return }
        defaults.set(data, forKey: snapshotKey)
    }

    static func request(_ action: RequestedAction) {
        UserDefaults(suiteName: suiteName)?.set(action.rawValue, forKey: requestedActionKey)
    }

    static func consumeRequestedAction() -> RequestedAction? {
        guard let defaults = UserDefaults(suiteName: suiteName),
              let rawValue = defaults.string(forKey: requestedActionKey)
        else { return nil }
        defaults.removeObject(forKey: requestedActionKey)
        return RequestedAction(rawValue: rawValue)
    }

    static func stageDocument(from source: URL, preferredExtension: String? = nil) throws -> URL {
        guard let container = FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: suiteName)
        else { throw CocoaError(.fileNoSuchFile) }
        let directory = container.appendingPathComponent("Incoming", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let ext = preferredExtension ?? (source.pathExtension.isEmpty ? "image" : source.pathExtension)
        let destination = directory.appendingPathComponent("\(UUID().uuidString).\(ext)")
        try FileManager.default.copyItem(at: source, to: destination)
        UserDefaults(suiteName: suiteName)?.set(destination.lastPathComponent, forKey: stagedDocumentKey)
        return destination
    }

    static func consumeStagedDocumentURL() -> URL? {
        guard let defaults = UserDefaults(suiteName: suiteName),
              let name = defaults.string(forKey: stagedDocumentKey),
              let container = FileManager.default.containerURL(
                forSecurityApplicationGroupIdentifier: suiteName)
        else { return nil }
        defaults.removeObject(forKey: stagedDocumentKey)
        return container.appendingPathComponent("Incoming", isDirectory: true).appendingPathComponent(name)
    }
}
