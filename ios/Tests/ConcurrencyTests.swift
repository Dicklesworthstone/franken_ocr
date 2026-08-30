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
}
