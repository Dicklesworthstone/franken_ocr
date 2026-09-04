import XCTest
@testable import FrankenOCR

final class ConcurrencyTests: XCTestCase {
    func testLifecycleFenceRejectsAnUnloadThatArrivesAfterANewerLoad() {
        var fence = EngineLifecycleFence()

        XCTAssertTrue(fence.accept(41))
        XCTAssertTrue(fence.accept(42))
        XCTAssertFalse(fence.accept(41))
        XCTAssertEqual(fence.latestToken, 42)
    }

    func testCrossPageEnvelopePreservesRequestedSourcePageIdentity() throws {
        let json = #"{"model_id":"unlimited-ocr","output":"<PAGE>one<PAGE>five","pages":[{"source_page":1,"output":"one"},{"source_page":5,"output":"five"}]}"#

        let result = try CrossPageRecognition(json: json, expectedSourcePages: [1, 5])

        XCTAssertEqual(result.modelID, "unlimited-ocr")
        XCTAssertEqual(
            result.pages,
            [
                .init(sourcePage: 1, output: "one"),
                .init(sourcePage: 5, output: "five")
            ]
        )
    }

    func testCrossPageEnvelopeRefusesMissingOrReorderedPages() {
        let json = #"{"model_id":"unlimited-ocr","output":"<PAGE>five<PAGE>one","pages":[{"source_page":5,"output":"five"},{"source_page":1,"output":"one"}]}"#

        XCTAssertThrowsError(
            try CrossPageRecognition(json: json, expectedSourcePages: [1, 5])
        )
    }

    #if !targetEnvironment(macCatalyst)
    func testLiveTextAccumulatorAcceptsARealLineWithoutAChangedItemID() {
        let lineID = UUID()
        let batch = LiveCameraBatch(
            lines: [
                .init(
                    id: lineID,
                    text: "Recognized by Apple Vision",
                    confidence: 0.41,
                    box: CGRect(x: 0.2, y: 0.3, width: 0.5, height: 0.08)
                )
            ],
            animatedLineIDs: []
        )
        var accumulator = LiveTextAccumulator()

        let accepted = accumulator.ingest(batch)

        XCTAssertEqual(accepted.map(\.id), [lineID])
        XCTAssertEqual(accumulator.text, "Recognized by Apple Vision")
    }

    func testLiveTextAccumulatorDeduplicatesStationaryScannerUpdates() {
        let lineID = UUID()
        let line = LiveCameraBatch.Line(
            id: lineID,
            text: "Stationary paragraph",
            confidence: 0.82,
            box: CGRect(x: 0.1, y: 0.2, width: 0.7, height: 0.1)
        )
        var accumulator = LiveTextAccumulator()

        XCTAssertEqual(accumulator.ingest(.init(lines: [line], animatedLineIDs: [lineID])).count, 1)
        XCTAssertTrue(accumulator.ingest(.init(lines: [line], animatedLineIDs: [])).isEmpty)
        XCTAssertEqual(accumulator.text, "Stationary paragraph")
    }

    func testLiveTextAccumulatorReplacesASettlingTranscriptAndCanClear() {
        let lineID = UUID()
        var accumulator = LiveTextAccumulator()
        let partial = LiveCameraBatch.Line(
            id: lineID,
            text: "Franken",
            confidence: 0.40,
            box: CGRect(x: 0.2, y: 0.4, width: 0.2, height: 0.08)
        )
        let settled = LiveCameraBatch.Line(
            id: lineID,
            text: "FrankenOCR is alive",
            confidence: 0.87,
            box: CGRect(x: 0.2, y: 0.4, width: 0.5, height: 0.08)
        )

        _ = accumulator.ingest(.init(lines: [partial], animatedLineIDs: [lineID]))
        let replacements = accumulator.ingest(.init(lines: [settled], animatedLineIDs: [lineID]))

        XCTAssertEqual(replacements.map(\.text), ["FrankenOCR is alive"])
        XCTAssertEqual(accumulator.text, "FrankenOCR is alive")
        accumulator.clear()
        XCTAssertTrue(accumulator.text.isEmpty)
    }
    #endif
}
