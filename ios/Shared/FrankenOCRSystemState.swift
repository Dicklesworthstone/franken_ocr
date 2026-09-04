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
    private static let stagedDocumentsKey = "share.staged-documents.v2"
    static let maximumStagedDocuments = 32

    struct StagedDocument: Codable, Hashable, Sendable {
        let storedName: String
        let displayName: String
    }

    struct ConsumedDocument: Sendable {
        let url: URL
        let displayName: String
    }

    struct StagedSelectionError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

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

    /// Validate the shape before any provider starts copying temporary files.
    /// The app accepts either one PDF or the same 1...32 image batch as Files.
    static func validateStagedSelection(itemCount: Int, pdfCount: Int) throws {
        guard itemCount > 0 else {
            throw StagedSelectionError(message: "Share one PDF, or up to 32 images.")
        }
        guard itemCount <= maximumStagedDocuments else {
            throw StagedSelectionError(message: "Share no more than 32 images at once.")
        }
        guard pdfCount >= 0, pdfCount <= itemCount else {
            throw StagedSelectionError(message: "The shared selection was malformed.")
        }
        guard pdfCount == 0 || (pdfCount == 1 && itemCount == 1) else {
            throw StagedSelectionError(message: "Share one PDF by itself, or up to 32 images.")
        }
    }

    /// Copy one provider-owned temporary file before its callback returns, but
    /// do not publish a partial selection to the app. The extension publishes
    /// the ordered records only after every copy succeeds.
    static func stageDocument(
        from source: URL,
        preferredExtension: String? = nil,
        displayName: String? = nil
    ) throws -> StagedDocument {
        let directory = try incomingDirectory()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let ext = preferredExtension ?? (source.pathExtension.isEmpty ? "image" : source.pathExtension)
        let destination = directory.appendingPathComponent("\(UUID().uuidString).\(ext)")
        try FileManager.default.copyItem(at: source, to: destination)
        let visibleName = displayName?.trimmingCharacters(in: .whitespacesAndNewlines)
        return StagedDocument(
            storedName: destination.lastPathComponent,
            displayName: visibleName.flatMap { $0.isEmpty ? nil : $0 }
                ?? source.lastPathComponent
        )
    }

    static func publishStagedDocuments(_ documents: [StagedDocument]) throws {
        guard !documents.isEmpty, documents.count <= maximumStagedDocuments else {
            throw StagedSelectionError(message: "The staged selection was empty or too large.")
        }
        guard documents.allSatisfy({ safeStoredName($0.storedName) }) else {
            throw StagedSelectionError(message: "The staged selection contained an unsafe filename.")
        }
        guard let defaults = UserDefaults(suiteName: suiteName) else {
            throw CocoaError(.fileWriteUnknown)
        }
        defaults.set(try JSONEncoder().encode(documents), forKey: stagedDocumentsKey)
        defaults.removeObject(forKey: stagedDocumentKey)
    }

    static func discardStagedDocuments(_ documents: [StagedDocument]) {
        guard let directory = try? incomingDirectory() else { return }
        for document in documents where safeStoredName(document.storedName) {
            try? FileManager.default.removeItem(
                at: directory.appendingPathComponent(document.storedName, isDirectory: false)
            )
        }
    }

    static func consumeStagedDocuments() -> [ConsumedDocument] {
        guard let defaults = UserDefaults(suiteName: suiteName),
              let directory = try? incomingDirectory()
        else { return [] }

        if let data = defaults.data(forKey: stagedDocumentsKey) {
            defaults.removeObject(forKey: stagedDocumentsKey)
            defaults.removeObject(forKey: stagedDocumentKey)
            guard let records = try? JSONDecoder().decode([StagedDocument].self, from: data) else {
                return []
            }
            return records.compactMap { record in
                guard safeStoredName(record.storedName) else { return nil }
                return ConsumedDocument(
                    url: directory.appendingPathComponent(record.storedName, isDirectory: false),
                    displayName: record.displayName
                )
            }
        }

        // One-release compatibility for a document staged by the v1 extension.
        guard let name = defaults.string(forKey: stagedDocumentKey), safeStoredName(name) else {
            return []
        }
        defaults.removeObject(forKey: stagedDocumentKey)
        return [ConsumedDocument(
            url: directory.appendingPathComponent(name, isDirectory: false),
            displayName: name
        )]
    }

    private static func incomingDirectory() throws -> URL {
        guard let container = FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: suiteName)
        else { throw CocoaError(.fileNoSuchFile) }
        return container.appendingPathComponent("Incoming", isDirectory: true)
    }

    private static func safeStoredName(_ name: String) -> Bool {
        !name.isEmpty && name == (name as NSString).lastPathComponent
    }
}
