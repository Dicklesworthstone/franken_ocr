import Foundation
import Observation

struct RecognitionHistoryEntry: Codable, Equatable, Identifiable {
    let id: UUID
    let createdAt: Date
    let sourceName: String
    let modelName: String
    let characterCount: Int
    let pageCount: Int
    let seconds: Double?
    let byteCount: Int
    let fileName: String

    private enum CodingKeys: String, CodingKey {
        case id
        case createdAtMilliseconds
        case sourceName
        case modelName
        case characterCount
        case pageCount
        case seconds
        case byteCount
        case fileName
    }

    init(
        id: UUID,
        createdAt: Date,
        sourceName: String,
        modelName: String,
        characterCount: Int,
        pageCount: Int,
        seconds: Double?,
        byteCount: Int,
        fileName: String
    ) {
        self.id = id
        self.createdAt = createdAt
        self.sourceName = sourceName
        self.modelName = modelName
        self.characterCount = characterCount
        self.pageCount = pageCount
        self.seconds = seconds
        self.byteCount = byteCount
        self.fileName = fileName
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        let milliseconds = try container.decode(Int64.self, forKey: .createdAtMilliseconds)
        createdAt = Date(timeIntervalSince1970: Double(milliseconds) / 1_000)
        sourceName = try container.decode(String.self, forKey: .sourceName)
        modelName = try container.decode(String.self, forKey: .modelName)
        characterCount = try container.decode(Int.self, forKey: .characterCount)
        pageCount = try container.decode(Int.self, forKey: .pageCount)
        seconds = try container.decodeIfPresent(Double.self, forKey: .seconds)
        byteCount = try container.decode(Int.self, forKey: .byteCount)
        fileName = try container.decode(String.self, forKey: .fileName)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(Self.milliseconds(createdAt), forKey: .createdAtMilliseconds)
        try container.encode(sourceName, forKey: .sourceName)
        try container.encode(modelName, forKey: .modelName)
        try container.encode(characterCount, forKey: .characterCount)
        try container.encode(pageCount, forKey: .pageCount)
        try container.encodeIfPresent(seconds, forKey: .seconds)
        try container.encode(byteCount, forKey: .byteCount)
        try container.encode(fileName, forKey: .fileName)
    }

    static func normalized(_ date: Date) -> Date? {
        let scaled = date.timeIntervalSince1970 * 1_000
        guard scaled.isFinite, let value = Int64(exactly: scaled.rounded()) else { return nil }
        return Date(timeIntervalSince1970: Double(value) / 1_000)
    }

    private static func milliseconds(_ date: Date) throws -> Int64 {
        let scaled = date.timeIntervalSince1970 * 1_000
        guard scaled.isFinite, let value = Int64(exactly: scaled.rounded()) else {
            throw RecognitionHistoryError.invalidResult
        }
        return value
    }
}

struct RecognitionHistoryResult {
    let text: String
    let sourceName: String
    let modelName: String
    let pageCount: Int
    let seconds: Double?
    let fileExtension: String
}

@Observable
final class RecognitionHistoryStore {
    static let maximumEntries = 20
    static let maximumAge: TimeInterval = 14 * 24 * 60 * 60
    static let maximumStoredBytes = 16 * 1_024 * 1_024

    private static let manifestSchema = "frankenocr.recognition-history.v1"
    private static let manifestName = "history.json"
    private static let allowedExtensions = Set(["md", "musicxml", "txt"])

    private(set) var entries: [RecognitionHistoryEntry] = []
    private(set) var storageBytes = 0

    private let directory: URL
    private let fileManager: FileManager
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(
        directory requestedDirectory: URL? = nil,
        now: Date = .now,
        fileManager: FileManager = .default
    ) {
        self.fileManager = fileManager
        directory = requestedDirectory ?? Self.defaultDirectory(fileManager: fileManager)
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        prepareDirectory()
        reload(now: now)
    }

    @discardableResult
    func record(
        _ result: RecognitionHistoryResult,
        createdAt: Date = .now
    ) throws -> RecognitionHistoryEntry {
        let data = Data(result.text.utf8)
        let normalizedExtension = result.fileExtension.lowercased()
        guard let stableCreatedAt = RecognitionHistoryEntry.normalized(createdAt),
              !result.text.isEmpty,
              data.count <= Self.maximumStoredBytes,
              Self.allowedExtensions.contains(normalizedExtension),
              result.pageCount > 0,
              result.seconds?.isFinite != false,
              result.seconds.map({ $0 >= 0 }) != false else {
            throw RecognitionHistoryError.invalidResult
        }

        let id = UUID()
        let fileName = "\(id.uuidString.lowercased()).\(normalizedExtension)"
        let url = directory.appendingPathComponent(fileName, isDirectory: false)
        try data.write(to: url, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        excludeFromBackup(url)

        let entry = RecognitionHistoryEntry(
            id: id,
            createdAt: stableCreatedAt,
            sourceName: Self.boundedLabel(result.sourceName, fallback: "Untitled"),
            modelName: Self.boundedLabel(result.modelName, fallback: "FrankenOCR"),
            characterCount: result.text.count,
            pageCount: result.pageCount,
            seconds: result.seconds,
            byteCount: data.count,
            fileName: fileName
        )
        let previousEntries = entries
        entries.insert(entry, at: 0)
        let removed = prune(now: stableCreatedAt, deleteRemoved: false)
        do {
            try persistManifest()
            for removedEntry in removed { removeDocument(for: removedEntry) }
        } catch {
            entries = previousEntries
            try? fileManager.removeItem(at: url)
            recalculateStorage()
            throw error
        }
        return entry
    }

    func fileURL(for entry: RecognitionHistoryEntry) -> URL? {
        guard entries.contains(where: { $0.id == entry.id && $0.fileName == entry.fileName }),
              Self.isOwnedFileName(entry.fileName, id: entry.id) else { return nil }
        let url = directory.appendingPathComponent(entry.fileName, isDirectory: false)
        guard fileManager.fileExists(atPath: url.path) else { return nil }
        return url
    }

    func text(for entry: RecognitionHistoryEntry) -> String? {
        guard let url = fileURL(for: entry),
              let data = try? Data(contentsOf: url),
              data.count == entry.byteCount,
              data.count <= Self.maximumStoredBytes else { return nil }
        return String(data: data, encoding: .utf8)
    }

    func delete(_ entry: RecognitionHistoryEntry) {
        guard let index = entries.firstIndex(where: { $0.id == entry.id }) else { return }
        let removed = entries.remove(at: index)
        recalculateStorage()
        do {
            try persistManifest()
            removeDocument(for: removed)
        } catch {
            entries.insert(removed, at: index)
            recalculateStorage()
        }
    }

    func deleteAll() {
        let removed = entries
        entries.removeAll(keepingCapacity: false)
        recalculateStorage()
        do {
            try persistManifest()
            for entry in removed { removeDocument(for: entry) }
        } catch {
            entries = removed
            recalculateStorage()
        }
    }

    private func prepareDirectory() {
        try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        excludeFromBackup(directory)
    }

    private func excludeFromBackup(_ url: URL) {
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutableURL = url
        try? mutableURL.setResourceValues(values)
    }

    private func reload(now: Date) {
        let manifestURL = directory.appendingPathComponent(Self.manifestName)
        guard let data = try? Data(contentsOf: manifestURL), data.count <= 512_000,
              let manifest = try? decoder.decode(Manifest.self, from: data),
              manifest.schema == Self.manifestSchema else {
            entries = []
            storageBytes = 0
            return
        }
        var seenIDs = Set<UUID>()
        entries = manifest.entries.filter { entry in
            guard seenIDs.insert(entry.id).inserted,
                  Self.isValidMetadata(entry),
                  Self.isOwnedFileName(entry.fileName, id: entry.id) else { return false }
            let url = directory.appendingPathComponent(entry.fileName, isDirectory: false)
            guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
                  values.isRegularFile == true,
                  values.fileSize == entry.byteCount else { return false }
            return true
        }
        let removed = prune(now: now, deleteRemoved: false)
        do {
            try persistManifest()
            for entry in removed { removeDocument(for: entry) }
        } catch {
            // The old manifest and every referenced file remain intact.
        }
    }

    @discardableResult
    private func prune(now: Date, deleteRemoved: Bool = true) -> [RecognitionHistoryEntry] {
        entries.sort {
            if $0.createdAt == $1.createdAt { return $0.id.uuidString < $1.id.uuidString }
            return $0.createdAt > $1.createdAt
        }
        var kept: [RecognitionHistoryEntry] = []
        var removed: [RecognitionHistoryEntry] = []
        var bytes = 0
        for entry in entries {
            let fits = kept.count < Self.maximumEntries
                && now.timeIntervalSince(entry.createdAt) <= Self.maximumAge
                && entry.createdAt.timeIntervalSince(now) <= 60
                && bytes <= Self.maximumStoredBytes - entry.byteCount
            if fits {
                kept.append(entry)
                bytes += entry.byteCount
            } else {
                removed.append(entry)
            }
        }
        entries = kept
        storageBytes = bytes
        if deleteRemoved {
            for entry in removed { removeDocument(for: entry) }
        }
        return removed
    }

    private func removeDocument(for entry: RecognitionHistoryEntry) {
        guard Self.isOwnedFileName(entry.fileName, id: entry.id) else { return }
        try? fileManager.removeItem(
            at: directory.appendingPathComponent(entry.fileName, isDirectory: false)
        )
    }

    private func recalculateStorage() {
        storageBytes = entries.reduce(0) { $0 + $1.byteCount }
    }

    private func persistManifest() throws {
        let data = try encoder.encode(Manifest(schema: Self.manifestSchema, entries: entries))
        try data.write(
            to: directory.appendingPathComponent(Self.manifestName),
            options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication]
        )
    }

    private static func boundedLabel(_ value: String, fallback: String) -> String {
        let bounded = String(value.trimmingCharacters(in: .whitespacesAndNewlines).prefix(160))
        return bounded.isEmpty ? fallback : bounded
    }

    private static func isValidMetadata(_ entry: RecognitionHistoryEntry) -> Bool {
        !entry.sourceName.isEmpty && entry.sourceName.count <= 160
            && !entry.modelName.isEmpty && entry.modelName.count <= 160
            && entry.characterCount > 0
            && entry.pageCount > 0
            && entry.seconds?.isFinite != false
            && entry.seconds.map({ $0 >= 0 }) != false
            && entry.byteCount > 0 && entry.byteCount <= maximumStoredBytes
    }

    private static func isOwnedFileName(_ fileName: String, id: UUID) -> Bool {
        let url = URL(fileURLWithPath: fileName)
        return url.lastPathComponent == fileName
            && url.deletingPathExtension().lastPathComponent == id.uuidString.lowercased()
            && allowedExtensions.contains(url.pathExtension.lowercased())
    }

    private static func defaultDirectory(fileManager: FileManager) -> URL {
        let root = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )) ?? fileManager.temporaryDirectory
        return root
            .appendingPathComponent("FrankenOCR", isDirectory: true)
            .appendingPathComponent("Recognition History", isDirectory: true)
    }

    private struct Manifest: Codable {
        let schema: String
        let entries: [RecognitionHistoryEntry]
    }
}

enum RecognitionHistoryError: LocalizedError {
    case invalidResult

    var errorDescription: String? {
        "The recognized document is not valid for local history."
    }
}
