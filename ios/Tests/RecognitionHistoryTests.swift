import Foundation
import XCTest
@testable import FrankenOCR

final class RecognitionHistoryTests: XCTestCase {
    func testHistoryPersistsExportedTextWithoutSourceArtifactsInManifest() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = RecognitionHistoryStore(directory: directory)
        let secretText = "# Invoice\n\nPrivate recognized value 1842"

        let entry = try store.record(
            RecognitionHistoryResult(
                text: secretText,
                sourceName: " scan.pdf ",
                modelName: "Unlimited OCR",
                pageCount: 2,
                seconds: 3.25,
                fileExtension: "MD"
            )
        )

        XCTAssertEqual(entry.sourceName, "scan.pdf")
        XCTAssertEqual(entry.pageCount, 2)
        XCTAssertEqual(store.text(for: entry), secretText)
        XCTAssertEqual(store.storageBytes, Data(secretText.utf8).count)

        let manifest = try String(
            contentsOf: directory.appendingPathComponent("history.json"),
            encoding: .utf8
        )
        XCTAssertTrue(manifest.contains("frankenocr.recognition-history.v1"))
        XCTAssertFalse(manifest.contains(secretText))
        for forbidden in ["imageData", "pdfData", "layout", "question", "boundingBox"] {
            XCTAssertFalse(manifest.localizedCaseInsensitiveContains(forbidden), forbidden)
        }

        let restored = RecognitionHistoryStore(directory: directory)
        XCTAssertEqual(restored.entries, [entry])
        XCTAssertEqual(restored.text(for: entry), secretText)
    }

    func testHistoryRejectsEmptyTextAndUnownedExtension() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = RecognitionHistoryStore(directory: directory)

        XCTAssertThrowsError(
            try store.record(
                RecognitionHistoryResult(
                    text: "",
                    sourceName: "page",
                    modelName: "OCR",
                    pageCount: 1,
                    seconds: nil,
                    fileExtension: "md"
                )
            )
        )
        XCTAssertThrowsError(
            try store.record(
                RecognitionHistoryResult(
                    text: "result",
                    sourceName: "page",
                    modelName: "OCR",
                    pageCount: 1,
                    seconds: nil,
                    fileExtension: "html"
                )
            )
        )
        XCTAssertTrue(store.entries.isEmpty)
    }

    func testHistoryPrunesByCountAndAge() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let now = Date()
        let store = RecognitionHistoryStore(directory: directory, now: now)

        for index in 0..<(RecognitionHistoryStore.maximumEntries + 3) {
            try store.record(
                RecognitionHistoryResult(
                    text: "result \(index)",
                    sourceName: "page-\(index)",
                    modelName: "OCR",
                    pageCount: 1,
                    seconds: 1,
                    fileExtension: "md"
                ),
                createdAt: now
            )
        }
        XCTAssertEqual(store.entries.count, RecognitionHistoryStore.maximumEntries)
        let retainedURLs = try store.entries.map { try XCTUnwrap(store.fileURL(for: $0)) }

        let expired = RecognitionHistoryStore(
            directory: directory,
            now: now.addingTimeInterval(RecognitionHistoryStore.maximumAge + 1)
        )
        XCTAssertTrue(expired.entries.isEmpty)
        XCTAssertEqual(expired.storageBytes, 0)
        XCTAssertTrue(retainedURLs.allSatisfy { !FileManager.default.fileExists(atPath: $0.path) })
    }

    func testMalformedManifestDoesNotDeleteUnclaimedText() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = RecognitionHistoryStore(directory: directory)
        let entry = try store.record(
            RecognitionHistoryResult(
                text: "recognized",
                sourceName: "page",
                modelName: "OCR",
                pageCount: 1,
                seconds: 1,
                fileExtension: "md"
            )
        )
        let resultURL = try XCTUnwrap(store.fileURL(for: entry))
        try Data("{not-json".utf8).write(
            to: directory.appendingPathComponent("history.json"),
            options: .atomic
        )

        let recovered = RecognitionHistoryStore(directory: directory)

        XCTAssertTrue(recovered.entries.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: resultURL.path))
    }

    func testDeleteAndClearRemoveOnlyOwnedHistoryFiles() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let unrelated = directory.appendingPathComponent("keep-me.txt")
        try Data("owner data".utf8).write(to: unrelated)
        let store = RecognitionHistoryStore(directory: directory)
        let first = try record("first", in: store)
        let second = try record("second", in: store)
        let firstURL = try XCTUnwrap(store.fileURL(for: first))
        let secondURL = try XCTUnwrap(store.fileURL(for: second))

        store.delete(first)
        XCTAssertFalse(FileManager.default.fileExists(atPath: firstURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: secondURL.path))

        store.deleteAll()
        XCTAssertFalse(FileManager.default.fileExists(atPath: secondURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: unrelated.path))
        XCTAssertTrue(store.entries.isEmpty)
    }

    private func record(
        _ text: String,
        in store: RecognitionHistoryStore
    ) throws -> RecognitionHistoryEntry {
        try store.record(
            RecognitionHistoryResult(
                text: text,
                sourceName: "page",
                modelName: "OCR",
                pageCount: 1,
                seconds: 1,
                fileExtension: "md"
            )
        )
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("FrankenOCRHistoryTests-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
