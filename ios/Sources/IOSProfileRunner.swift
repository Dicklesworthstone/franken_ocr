import CryptoKit
import Foundation

/// Hidden, measurement-only physical-device lane.
///
/// Ordinary launches never enter this code. A profiler launch sets
/// `FOCR_IOS_PROFILE=1`, places the pinned fixture in Documents, and receives a
/// JSONL receipt beside it. Keeping the lane in the shipping host is
/// deliberate: the benchmark exercises the same Swift -> C ABI -> Rust model
/// path as the product, with the same mmap-backed artifact and kernel pool.
@MainActor
enum IOSProfileRunner {
    private static let environment = ProcessInfo.processInfo.environment

    static var isRequested: Bool { environment["FOCR_IOS_PROFILE"] == "1" }

    static func runIfRequested() async {
        guard isRequested else { return }
        do {
            try await run()
        } catch {
            let message = "FOCR_IOS_PROFILE_ERROR \(error.localizedDescription)"
            print(message)
        }
    }

    private static func run() async throws {
        let modelID = environment["FOCR_IOS_PROFILE_MODEL_ID"] ?? ModelCatalog.unlimitedOCR.id
        guard let spec = ModelCatalog.spec(id: modelID) else {
            throw ProfileError("unknown model id \"\(modelID)\"")
        }
        let fixtureURL = try profileFixtureURL()
        let fixture = try Data(contentsOf: fixtureURL, options: [.mappedIfSafe])
        guard !fixture.isEmpty else { throw ProfileError("profile fixture is empty") }

        let requestedRuns = Int(environment["FOCR_IOS_PROFILE_RUNS"] ?? "20") ?? 20
        let runs = min(200, max(1, requestedRuns))
        let artifactURL = try ModelStore.artifactURL(for: spec)
        guard FileManager.default.fileExists(atPath: artifactURL.path) else {
            throw ProfileError("\(spec.weights.name) is not installed")
        }

        let documents = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let stamp = ISO8601DateFormatter.profileTimestamp.string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let receiptURL = documents.appendingPathComponent("focr-ios-profile-\(stamp).jsonl")
        let receipt = try JSONLReceipt(url: receiptURL)
        let engine = Engine()
        let info = Engine.info()
        let bundle = Bundle.main

        try receipt.append([
            "event": "run_start",
            "schema_version": 1,
            "source_commit": environment["FOCR_IOS_PROFILE_SOURCE_SHA"] ?? "unreported",
            "model": [
                "id": spec.id,
                "bytes": spec.weights.bytes,
                "sha256": spec.weights.sha256,
                "relative_path": "\(spec.id)/\(spec.weights.name)",
            ],
            "fixture": fixtureURL.lastPathComponent,
            "fixture_bytes": fixture.count,
            "fixture_sha256": sha256(fixture),
            "runs": runs,
            "device_model": deviceModel,
            "system_name": ProcessInfo.processInfo.operatingSystemVersionString,
            "active_processors": ProcessInfo.processInfo.activeProcessorCount,
            "physical_memory_bytes": ProcessInfo.processInfo.physicalMemory,
            "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
            "engine_info": [
                "crate_version": info?.crate_version ?? "unknown",
                "detected_tier": info?.detected_tier ?? "unknown",
                "dense_route": info?.dense_route ?? "unknown",
                "threads": info?.threads ?? 0,
            ],
            "performance_switches": [
                "threads": environment["FOCR_THREADS"] ?? "unset",
                "int8_autovec": environment["FOCR_INT8_AUTOVEC"] ?? "unset",
                "force_arch": environment["FOCR_FORCE_ARCH"] ?? "unset",
                "mmap": environment["FOCR_MMAP"] ?? "unset",
                "timing": environment["FOCR_TIMING"] ?? "unset",
            ],
            "app_version": bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown",
            "app_build": bundle.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown",
            "receipt_path": receiptURL.path,
        ])

        let loadStarted = ContinuousClock.now
        try await engine.load(artifact: artifactURL, lifecycleToken: 1)
        let loadMS = milliseconds(since: loadStarted)
        if spec.decodeGuard > 0 { await engine.setNoRepeatNgram(spec.decodeGuard) }
        try receipt.append([
            "event": "engine_loaded",
            "load_ms": loadMS,
            "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
        ])

        var firstOutputHash: String?
        var allOutputsIdentical = true
        for index in 0..<runs {
            let clock = StageClock()
            let started = ContinuousClock.now
            let result = try await engine.recognize(imageData: fixture) { update in
                clock.observe(update)
            }
            let wallMS = milliseconds(since: started)
            let outputHash = sha256(Data(result.output.utf8))
            if let firstOutputHash {
                allOutputsIdentical = allOutputsIdentical && firstOutputHash == outputHash
            } else {
                firstOutputHash = outputHash
            }
            try receipt.append([
                "event": "sample",
                "index": index,
                "wall_ms": wallMS,
                "output_bytes": result.output.utf8.count,
                "output_sha256": outputHash,
                "matches_first_output": outputHash == firstOutputHash,
                "layout_spans": result.layout.count,
                "stage_first_ms": clock.firstMilliseconds,
                "stage_last_ms": clock.lastMilliseconds,
                "stage_updates": clock.updateCounts,
                "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
            ])
        }
        try receipt.append([
            "event": "run_complete",
            "completed_runs": runs,
            "all_outputs_identical": allOutputsIdentical,
            "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
        ])
        print("FOCR_IOS_PROFILE \(receiptURL.path)")
    }

    private static func profileFixtureURL() throws -> URL {
        let name = environment["FOCR_IOS_PROFILE_FIXTURE"] ?? "focr-ios-profile.png"
        if name.hasPrefix("/") { return URL(fileURLWithPath: name) }
        let documents = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return documents.appendingPathComponent(name)
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func milliseconds(since start: ContinuousClock.Instant) -> Double {
        let duration = ContinuousClock.now - start
        return Double(duration.components.seconds) * 1_000
            + Double(duration.components.attoseconds) / 1_000_000_000_000_000
    }

    private static var deviceModel: String {
        var system = utsname()
        uname(&system)
        return withUnsafePointer(to: &system.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
    }
}

private final class StageClock: @unchecked Sendable {
    private let lock = NSLock()
    private let started = ContinuousClock.now
    private var first: [String: Double] = [:]
    private var last: [String: Double] = [:]
    private var counts: [String: Int] = [:]

    func observe(_ update: ProgressUpdate) {
        let name = update.stage.rawValue
        let elapsed = IOSProfileRunnerMilliseconds.since(started)
        lock.lock()
        if first[name] == nil { first[name] = elapsed }
        last[name] = elapsed
        counts[name, default: 0] += 1
        lock.unlock()
    }

    var firstMilliseconds: [String: Double] { snapshot { first } }
    var lastMilliseconds: [String: Double] { snapshot { last } }
    var updateCounts: [String: Int] { snapshot { counts } }

    private func snapshot<T>(_ read: () -> [String: T]) -> [String: T] {
        lock.lock()
        defer { lock.unlock() }
        return read()
    }
}

private enum IOSProfileRunnerMilliseconds {
    static func since(_ start: ContinuousClock.Instant) -> Double {
        let duration = ContinuousClock.now - start
        return Double(duration.components.seconds) * 1_000
            + Double(duration.components.attoseconds) / 1_000_000_000_000_000
    }
}

private final class JSONLReceipt {
    private let handle: FileHandle

    init(url: URL) throws {
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw ProfileError("could not create \(url.lastPathComponent)")
        }
        handle = try FileHandle(forWritingTo: url)
    }

    deinit { try? handle.close() }

    func append(_ object: [String: Any]) throws {
        var data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        data.append(0x0A)
        try handle.write(contentsOf: data)
        try handle.synchronize()
    }
}

private struct ProfileError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

private extension ISO8601DateFormatter {
    static let profileTimestamp: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()
}
