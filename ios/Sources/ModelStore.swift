import CryptoKit
import Foundation

/// One downloadable file. `parts` is non-empty when the asset is split, because
/// GitHub caps a release asset at 2 GiB and the flagship artifact is 3.0 GB —
/// the parts are concatenated into one logical file, and BOTH each part and the
/// whole are verified against their pinned digests.
struct ModelAsset: Sendable {
    let name: String
    let bytes: Int
    let sha256: String
    let parts: [Part]

    struct Part: Sendable {
        let name: String
        let bytes: Int
        let sha256: String
    }

    init(name: String, bytes: Int, sha256: String, parts: [Part] = []) {
        self.name = name
        self.bytes = bytes
        self.sha256 = sha256
        self.parts = parts
    }
}

/// A model the app can run. Digests mirror `site/model-manifest.js` and
/// `models/manifest-v2.json`, which remain the source of truth.
struct ModelSpec: Sendable, Identifiable {
    let id: String
    let label: String
    let shortName: String
    let license: String
    let releaseTag: String
    /// `owner/repo[/subdir]` on HuggingFace, or nil while no public mirror
    /// exists for this model. Mirrors `models/manifest-v2.json`, which is the
    /// source of truth for what is actually published where.
    let huggingFaceRepo: String?
    let weights: ModelAsset
    let sidecars: [ModelAsset]
    /// Sliding no-repeat n-gram guard applied at load. 0 leaves the engine
    /// default. Hard dense scans can tip a decode into a repetition loop; the
    /// tighter guard is the README's documented mitigation, applied honestly
    /// rather than silently.
    let decodeGuard: UInt32
    /// Physical memory below which this model is refused rather than allowed to
    /// get the app killed. nil means it runs anywhere.
    let minimumDeviceMemory: UInt64?
    /// What the model is for, in one line.
    let blurb: String
    /// Output is MusicXML rather than Markdown.
    let producesMusicXML: Bool

    var totalBytes: Int { weights.bytes + sidecars.reduce(0) { $0 + $1.bytes } }

    /// Where to fetch this model's files, best host first.
    ///
    /// HuggingFace leads: it serves ranged requests, sits behind a CDN, has no
    /// per-file size cap, and does not 503 the way GitHub release assets do
    /// under load. GitHub stays as the fallback so a download still completes if
    /// one host is unreachable — every byte is digest-verified on arrival, so
    /// mixing hosts across a resumed transfer is safe by construction.
    ///
    /// The `huggingFaceRepo` path must be a PUBLIC repo; a private one answers
    /// 401 and the fallback silently becomes the only route.
    var baseURLs: [URL] {
        var urls: [URL] = []
        if let hf = huggingFaceRepo {
            urls.append(URL(string: "https://huggingface.co/\(hf)/resolve/main/")!)
        }
        urls.append(
            URL(string: "https://github.com/Dicklesworthstone/franken_ocr/releases/download/\(releaseTag)/")!
        )
        return urls
    }

    /// Whether this device has the memory to attempt the model.
    var isSupportedOnThisDevice: Bool {
        guard let minimumDeviceMemory else { return true }
        return ProcessInfo.processInfo.physicalMemory >= minimumDeviceMemory
    }
}

enum ModelCatalog {
    static let unlimitedOCR = ModelSpec(
        id: "unlimited-ocr",
        label: "Baidu Unlimited-OCR",
        shortName: "Unlimited-OCR",
        license: "Baidu Unlimited-OCR - Copyright (c) 2026 Baidu, MIT License",
        releaseTag: "models-unlimited-wasm-v1",
        // No HuggingFace mirror published for the wasm-int4 artifact yet, so
        // this lane is GitHub-only and carries that host's 503 risk. HF has no
        // 2 GiB per-file cap either, so a mirror could ship the 3.0 GB artifact
        // whole and retire the two-part split entirely (bd, see beads).
        huggingFaceRepo: nil,
        weights: ModelAsset(
            name: "unlimited-ocr.wasm-int4.focrq",
            bytes: 3_003_988_117,
            sha256: "2653831ccd7f481f898f80ae5c95fa1ec7ee2a5a18005d3c927ddf64ed75e187",
            parts: [
                .init(name: "unlimited-ocr.wasm-int4.focrq.part1",
                      bytes: 1_677_721_600,
                      sha256: "95e8bc996ef08dc9ff179dba522ee45e953823913dbf73ac710d799627a9b2c5"),
                .init(name: "unlimited-ocr.wasm-int4.focrq.part2",
                      bytes: 1_326_266_517,
                      sha256: "1b6673345d1223f6ad4443df3f9c0760b4e401549c731c1c0d0c9e392dffda93"),
            ]
        ),
        sidecars: [
            ModelAsset(name: "tokenizer.json",
                       bytes: 9_979_544,
                       sha256: "a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4"),
        ],
        decodeGuard: 20,
        // 8 GB class. The artifact is mapped, not heaped, so this is a bound on
        // how much page cache the device can keep hot rather than on allocation.
        minimumDeviceMemory: 7 * 1024 * 1024 * 1024,
        blurb: "Document pages into Markdown: text, tables, LaTeX, reading order.",
        producesMusicXML: false
    )

    static let tromr = ModelSpec(
        id: "tromr",
        label: "Polyphonic-TrOMR",
        shortName: "TrOMR",
        license: "Polyphonic-TrOMR (NetEase) - Apache-2.0",
        releaseTag: "models-tromr-v1",
        // Matches the mirror `models/manifest-v2.json` already lists. NOTE: at
        // the time of writing that repo answers 401, so this path is a no-op
        // until it is made public — the GitHub fallback carries the download.
        huggingFaceRepo: "Dicklesworthstone/franken_ocr-weights/tromr",
        weights: ModelAsset(
            name: "tromr.int8.focrq",
            bytes: 61_107_485,
            sha256: "cced11c0f05656dd54cc615a15939c472dc8f916f04ae154ea4a0364839f845a"
        ),
        sidecars: [
            ModelAsset(name: "tokenizer_rhythm.json", bytes: 10_743,
                       sha256: "603bfef760e8424f7808acba423532b4beb2d88dbf085f81add6a8e543a34035"),
            ModelAsset(name: "tokenizer_pitch.json", bytes: 2_682,
                       sha256: "2382e8b20c1473290e200789604656b3a06bdf4b55a0818a0f7d175e8cb64ade"),
            ModelAsset(name: "tokenizer_lift.json", bytes: 979,
                       sha256: "b61ba09cecd5bc343e6a038a2e26718b54cd3c08e8f9b72013ecf80c3cac86b2"),
            ModelAsset(name: "tokenizer_note.json", bytes: 830,
                       sha256: "504d886d11e3c1fe92893abd46edfc68dfbe7a8eb83e6b51646532dad8a485e1"),
        ],
        decodeGuard: 0,
        minimumDeviceMemory: nil,
        blurb: "Printed or scanned sheet music into MusicXML, a full page or one staff.",
        producesMusicXML: true
    )

    /// The models this build actually runs.
    ///
    /// GOT-OCR2, SmolVLM2, and OneChart are intentionally absent: each hydrates
    /// its vision tower to f32 whole (measured wasm peaks of 3.4 GB, 2.8 GB),
    /// because the streamed per-block residency mode is keyed to the
    /// Unlimited-OCR recipe. Extending that mode to their towers is the work
    /// that earns them a place here; shipping them before it would be shipping
    /// a model that gets the app killed.
    static let all: [ModelSpec] = [unlimitedOCR, tromr]

    static func spec(id: String) -> ModelSpec? { all.first { $0.id == id } }
}

/// Where a model download currently is.
enum DownloadPhase: Sendable, Equatable {
    case idle
    case downloading(asset: String, done: Int, total: Int, eta: String)
    case verifying(asset: String)
    case ready
    case failed(String)
}

/// Downloads, verifies, and stores model artifacts.
///
/// Design notes worth keeping:
///
/// * **Ranged and resumable.** Chunks of 32 MiB appended to the destination
///   file, so an interrupted 3 GB download resumes from its byte count rather
///   than starting over on a phone that just walked out of Wi-Fi range.
/// * **Digest-verified on arrival.** SHA-256 is computed incrementally as the
///   bytes land when the download started from zero; a resumed download
///   re-hashes the file at the end, because a streamed hash cannot reuse a
///   prefix it never saw. A mismatch deletes the file and refuses, naming it.
/// * **Never more than one chunk in memory.** The point of the whole exercise
///   is to not need gigabytes of RAM.
@MainActor
@Observable
final class ModelStore {
    private(set) var phase: DownloadPhase = .idle
    private(set) var installedBytes: Int = 0

    private var task: Task<Void, Never>?
    private static let chunkBytes = 32 * 1024 * 1024

    /// Root for all model data. Application Support, excluded from backup: these
    /// are multi-gigabyte files that can always be re-downloaded, and putting
    /// them in iCloud backup would be antisocial.
    static func rootDirectory() throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true
        )
        let root = base.appendingPathComponent("franken_ocr", isDirectory: true)
        if !FileManager.default.fileExists(atPath: root.path) {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            var mutable = root
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            try? mutable.setResourceValues(values)
        }
        return root
    }

    static func directory(for spec: ModelSpec) throws -> URL {
        let dir = try rootDirectory().appendingPathComponent(spec.id, isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    static func artifactURL(for spec: ModelSpec) throws -> URL {
        try directory(for: spec).appendingPathComponent(spec.weights.name)
    }

    /// Whether every required file is present at its pinned size.
    ///
    /// Size-only at launch, on purpose: every byte was digest-verified when it
    /// was downloaded, and the `.focrq` container re-checks its own structure at
    /// engine load. Re-hashing 3 GB on every launch would cost tens of seconds
    /// to defend against corruption this storage does not produce.
    func isInstalled(_ spec: ModelSpec) -> Bool {
        guard let dir = try? Self.directory(for: spec) else { return false }
        var total = 0
        for asset in [spec.weights] + spec.sidecars {
            let url = dir.appendingPathComponent(asset.name)
            guard let size = try? FileManager.default
                .attributesOfItem(atPath: url.path)[.size] as? Int, size == asset.bytes
            else { return false }
            total += size
        }
        installedBytes = total
        return true
    }

    func refreshInstalledState(_ spec: ModelSpec) {
        phase = isInstalled(spec) ? .ready : .idle
    }

    // ── Download ───────────────────────────────────────────────────────────

    func download(_ spec: ModelSpec) {
        guard task == nil else { return }
        task = Task { [weak self] in
            guard let self else { return }
            do {
                try await self.run(spec)
                self.phase = .ready
                _ = self.isInstalled(spec)
            } catch is CancellationError {
                self.phase = .idle
            } catch {
                self.phase = .failed(error.localizedDescription)
            }
            self.task = nil
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
        phase = .idle
    }

    func clear(_ spec: ModelSpec) {
        cancel()
        if let dir = try? Self.directory(for: spec) {
            try? FileManager.default.removeItem(at: dir)
        }
        installedBytes = 0
        phase = .idle
    }

    private func run(_ spec: ModelSpec) async throws {
        let dir = try Self.directory(for: spec)

        for sidecar in spec.sidecars {
            try Task.checkCancellation()
            try await fetch(asset: sidecar, from: spec.baseURLs, into: dir)
        }

        try Task.checkCancellation()
        if spec.weights.parts.isEmpty {
            try await fetch(asset: spec.weights, from: spec.baseURLs, into: dir)
        } else {
            try await fetchSplit(spec.weights, from: spec.baseURLs, into: dir)
        }
    }

    /// Download one whole asset, resuming if a partial file exists.
    private func fetch(asset: ModelAsset, from bases: [URL], into dir: URL) async throws {
        let destination = dir.appendingPathComponent(asset.name)
        if let size = try? FileManager.default
            .attributesOfItem(atPath: destination.path)[.size] as? Int, size == asset.bytes {
            return // already have it, verified when it landed
        }
        try await downloadWithFallback(
            bases: bases,
            file: asset.name,
            to: destination,
            expectedBytes: asset.bytes,
            expectedDigest: asset.sha256
        )
    }

    /// Try each host in turn. The digest check is what makes this safe: a
    /// partial file from a host that died mid-transfer is resumed against the
    /// next host, and the whole is verified before it counts as installed.
    ///
    /// Cancellation is never a reason to try the next host — that would turn a
    /// user's Cancel into a retry storm.
    private func downloadWithFallback(
        bases: [URL],
        file: String,
        to destination: URL,
        expectedBytes: Int,
        expectedDigest: String
    ) async throws {
        var lastError: Error?
        for base in bases {
            do {
                try await downloadRanged(
                    url: base.appendingPathComponent(file),
                    to: destination,
                    expectedBytes: expectedBytes,
                    expectedDigest: expectedDigest,
                    label: file
                )
                return
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                lastError = error
            }
        }
        throw lastError
            ?? ModelStoreError.badResponse(file)
    }

    /// Download an asset that ships as ordered byte-split parts, then
    /// concatenate. Each part is verified on arrival; the concatenated whole is
    /// verified before it is allowed to become the artifact.
    private func fetchSplit(_ asset: ModelAsset, from bases: [URL], into dir: URL) async throws {
        let destination = dir.appendingPathComponent(asset.name)
        if let size = try? FileManager.default
            .attributesOfItem(atPath: destination.path)[.size] as? Int, size == asset.bytes {
            return
        }

        var partURLs: [URL] = []
        for part in asset.parts {
            try Task.checkCancellation()
            let partURL = dir.appendingPathComponent(part.name)
            let have = (try? FileManager.default
                .attributesOfItem(atPath: partURL.path)[.size] as? Int) ?? 0
            if have != part.bytes {
                try await downloadWithFallback(
                    bases: bases,
                    file: part.name,
                    to: partURL,
                    expectedBytes: part.bytes,
                    expectedDigest: part.sha256
                )
            }
            partURLs.append(partURL)
        }

        // Concatenate into a temp file, verify the whole, then move into place —
        // so a half-written artifact can never be mistaken for a complete one.
        phase = .verifying(asset: asset.name)
        let staging = dir.appendingPathComponent(asset.name + ".assembling")
        try? FileManager.default.removeItem(at: staging)
        FileManager.default.createFile(atPath: staging.path, contents: nil)
        let out = try FileHandle(forWritingTo: staging)
        var whole = SHA256()
        for partURL in partURLs {
            let input = try FileHandle(forReadingFrom: partURL)
            while true {
                try Task.checkCancellation()
                guard let block = try input.read(upToCount: 8 * 1024 * 1024), !block.isEmpty
                else { break }
                whole.update(data: block)
                try out.write(contentsOf: block)
            }
            try input.close()
        }
        try out.close()

        let digest = whole.finalize().hexString
        guard digest == asset.sha256 else {
            try? FileManager.default.removeItem(at: staging)
            throw ModelStoreError.digestMismatch(asset.name, expected: asset.sha256, got: digest)
        }
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: staging, to: destination)
        // The parts are ~3 GB of pure duplication once the whole exists.
        for partURL in partURLs { try? FileManager.default.removeItem(at: partURL) }
    }

    /// The actual transfer: 32 MiB ranged requests appended to `destination`.
    private func downloadRanged(
        url: URL,
        to destination: URL,
        expectedBytes: Int,
        expectedDigest: String,
        label: String
    ) async throws {
        var offset = (try? FileManager.default
            .attributesOfItem(atPath: destination.path)[.size] as? Int) ?? 0
        if offset > expectedBytes { // a stale or corrupt partial
            try? FileManager.default.removeItem(at: destination)
            offset = 0
        }
        if !FileManager.default.fileExists(atPath: destination.path) {
            FileManager.default.createFile(atPath: destination.path, contents: nil)
        }

        let startedAt = Date()
        let startedOffset = offset
        // A streamed hash is only valid if we saw every byte.
        let canStreamHash = offset == 0
        var streaming = SHA256()

        let handle = try FileHandle(forWritingTo: destination)
        defer { try? handle.close() }
        try handle.seekToEnd()

        while offset < expectedBytes {
            try Task.checkCancellation()
            let upper = min(offset + Self.chunkBytes, expectedBytes) - 1
            var request = URLRequest(url: url)
            request.setValue("bytes=\(offset)-\(upper)", forHTTPHeaderField: "Range")
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse,
                  http.statusCode == 206 || (http.statusCode == 200 && offset == 0)
            else {
                throw ModelStoreError.badResponse(label)
            }
            guard !data.isEmpty else { throw ModelStoreError.truncated(label) }
            if canStreamHash { streaming.update(data: data) }
            try handle.write(contentsOf: data)
            offset += data.count

            phase = .downloading(
                asset: label, done: offset, total: expectedBytes,
                eta: Self.eta(done: offset - startedOffset,
                              total: expectedBytes - startedOffset,
                              since: startedAt)
            )
        }
        try handle.close()

        phase = .verifying(asset: label)
        let digest: String
        if canStreamHash {
            digest = streaming.finalize().hexString
        } else {
            digest = try await Self.hashFile(at: destination)
        }
        guard digest == expectedDigest else {
            try? FileManager.default.removeItem(at: destination)
            throw ModelStoreError.digestMismatch(label, expected: expectedDigest, got: digest)
        }
    }

    /// Hash a file 8 MiB at a time, off the main actor.
    ///
    /// `Task.detached` matters here: `ModelStore` is `@MainActor`, and hashing
    /// 3 GB on the main actor would freeze the UI for the whole verification.
    private static func hashFile(at url: URL) async throws -> String {
        try await Task.detached(priority: .utility) {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            var hasher = SHA256()
            while let block = try handle.read(upToCount: 8 * 1024 * 1024), !block.isEmpty {
                hasher.update(data: block)
            }
            return hasher.finalize().hexString
        }.value
    }

    private static func eta(done: Int, total: Int, since: Date) -> String {
        let elapsed = Date().timeIntervalSince(since)
        guard elapsed > 1, done > 0 else { return "estimating…" }
        let rate = Double(done) / elapsed
        guard rate > 0 else { return "estimating…" }
        let remaining = Double(total - done) / rate
        if remaining < 60 { return "~\(Int(remaining))s left" }
        let minutes = Int(remaining) / 60
        let seconds = Int(remaining) % 60
        return "~\(minutes)m \(seconds)s left"
    }
}

enum ModelStoreError: LocalizedError {
    case badResponse(String)
    case truncated(String)
    case digestMismatch(String, expected: String, got: String)

    var errorDescription: String? {
        switch self {
        case .badResponse(let name):
            "\(name): the server refused a ranged request"
        case .truncated(let name):
            "\(name): the server stopped sending before the file was complete"
        case .digestMismatch(let name, _, _):
            "\(name): digest mismatch; the file was cleared so you can retry"
        }
    }
}

extension SHA256.Digest {
    var hexString: String { map { String(format: "%02x", $0) }.joined() }
}

extension Int {
    /// "2.8 GB" / "61 MB" — the site's own rule (GB at or above 1024 MB).
    var humanBytes: String {
        let mb = Double(self) / 1_048_576
        if mb >= 1024 { return String(format: "%.1f GB", mb / 1024) }
        return String(format: "%.0f MB", mb)
    }
}
