import XCTest

final class FrankenOCRAppearanceUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testAppearanceTogglePersistsLightModeAcrossLaunches() throws {
        let app = XCUIApplication()
        app.launch()

        let toggle = app.buttons["appearance-toggle"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 12))
        XCTAssertTrue(
            ["Switch to light mode", "Switch to dark mode"].contains(toggle.label),
            "Appearance control exposed an unexpected state: \(toggle.label)"
        )

        if toggle.label == "Switch to dark mode" {
            toggle.tap()
            XCTAssertEqual(toggle.label, "Switch to light mode")
        }

        toggle.tap()
        XCTAssertEqual(toggle.label, "Switch to dark mode")
        keepScreenshot(of: app, named: "Remembered light appearance")

        app.terminate()
        app.launch()

        let relaunchedToggle = app.buttons["appearance-toggle"]
        XCTAssertTrue(relaunchedToggle.waitForExistence(timeout: 12))
        XCTAssertEqual(relaunchedToggle.label, "Switch to dark mode")

        // Leave the shared simulator in the product's default dark appearance.
        relaunchedToggle.tap()
        XCTAssertEqual(relaunchedToggle.label, "Switch to light mode")
    }

    func testFilePickerMakesImageBatchImportDiscoverable() throws {
        let app = XCUIApplication()
        app.launch()

        let batchButton = app.buttons["Files / Batch"]
        XCTAssertTrue(batchButton.waitForExistence(timeout: 12))
        XCTAssertTrue(batchButton.isHittable)
        XCTAssertTrue(app.staticTexts["Choose pages or images"].exists)
        let figureToggle = app.switches["extract-figures-toggle"]
        XCTAssertTrue(figureToggle.waitForExistence(timeout: 5))
        for _ in 0..<3 where !figureToggle.isHittable {
            app.swipeUp()
        }
        XCTAssertTrue(figureToggle.isHittable)
        keepScreenshot(of: app, named: "Files and batch import")
    }

    private func keepScreenshot(of app: XCUIApplication, named name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
