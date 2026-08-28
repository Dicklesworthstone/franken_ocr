import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

private enum LabTextEntry: Hashable {
    case smolQuestion
    case pageSelection
}

private struct LabTextEntryFramePreferenceKey: PreferenceKey {
    static let defaultValue: [LabTextEntry: CGRect] = [:]

    static func reduce(
        value: inout [LabTextEntry: CGRect],
        nextValue: () -> [LabTextEntry: CGRect]
    ) {
        value.merge(nextValue(), uniquingKeysWith: { _, new in new })
    }
}

private extension View {
    func reportLabTextEntryFrame(_ entry: LabTextEntry) -> some View {
        background {
            GeometryReader { proxy in
                Color.clear.preference(
                    key: LabTextEntryFramePreferenceKey.self,
                    value: [entry: proxy.frame(in: .named("lab-text-entry-space"))]
                )
            }
        }
    }
}

/// The whole app: one laboratory screen mirroring the site's playground —
/// 01 The specimen, 02 The page, 03 The transcription.
///
/// On a phone the three steps stack. On an iPad in landscape, steps 01+02 sit
/// beside step 03, because a transcription is worth reading next to the page it
/// came from and the tablet finally has the width for it.
struct LabView: View {
    private enum Destination: String, CaseIterable, Identifiable {
        case capture = "Capture"
        case live = "Live"
        case result = "Result"
        case models = "Models"

        var id: Self { self }
    }

    @State private var model = LabModel()
    @State private var destination: Destination = .capture
    @Environment(\.scenePhase) private var scenePhase

    @State private var photoItem: PhotosPickerItem?
    @State private var showFileImporter = false
    @State private var showCamera = false
    @State private var showLiveCamera = false
    @State private var copied = false
    @State private var copyResetTask: Task<Void, Never>?
    @State private var textEntryFrames: [LabTextEntry: CGRect] = [:]
    @FocusState private var focusedTextEntry: LabTextEntry?

    var body: some View {
        ZStack {
            LabBackground()
            GeometryReader { geometry in
                if usesThreeColumnDesktop(width: geometry.size.width) {
                    VStack(spacing: 14) {
                        header
                        HStack(alignment: .top, spacing: 14) {
                            ScrollView(.vertical) {
                                specimenCard.padding(.vertical, 4)
                            }
                            .scrollIndicators(.hidden)
                            .defaultScrollAnchor(.top)
                            .frame(maxWidth: .infinity)

                            ScrollView(.vertical) {
                                pageCard.padding(.vertical, 4)
                            }
                            .scrollIndicators(.hidden)
                            .defaultScrollAnchor(.top)
                            .frame(maxWidth: .infinity)

                            ScrollView(.vertical) {
                                transcriptionCard.padding(.vertical, 4)
                            }
                            .scrollIndicators(.hidden)
                            .defaultScrollAnchor(.top)
                            .frame(maxWidth: .infinity)
                        }
                        .frame(maxHeight: .infinity)
                        footer
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 14)
                    .frame(maxWidth: 1440, maxHeight: .infinity)
                    .frame(maxWidth: .infinity)
                } else if usesWideWorkspace, geometry.size.width >= 700 {
                    VStack(spacing: 14) {
                        header
                        HStack(alignment: .top, spacing: 16) {
                            ScrollView(.vertical) {
                                VStack(spacing: 16) {
                                    specimenCard
                                    pageCard
                                }
                                .padding(.vertical, 4)
                            }
                            .scrollIndicators(.hidden)
                            .defaultScrollAnchor(.top)
                            .frame(maxWidth: .infinity)

                            ScrollView(.vertical) {
                                transcriptionCard
                                    .padding(.vertical, 4)
                            }
                            .scrollIndicators(.hidden)
                            .defaultScrollAnchor(.top)
                            .frame(maxWidth: .infinity)
                        }
                        .frame(maxHeight: .infinity)
                        footer
                    }
                    .padding(.horizontal, geometry.size.width >= 1000 ? 24 : 16)
                    .padding(.vertical, 14)
                    .frame(maxWidth: 1320, maxHeight: .infinity)
                    .frame(maxWidth: .infinity)
                } else {
                    ScrollView {
                        VStack(spacing: 22) {
                            header
                            destinationPicker
                            compactWorkspace
                            footer
                        }
                        .padding(.horizontal, 18)
                        .padding(.vertical, 22)
                        .frame(maxWidth: 760)
                        .frame(maxWidth: .infinity)
                    }
                    .scrollDismissesKeyboard(.interactively)
                }
            }
        }
        .catalystReadableType()
        .coordinateSpace(name: "lab-text-entry-space")
        .onPreferenceChange(LabTextEntryFramePreferenceKey.self) { frames in
            textEntryFrames = frames
        }
        .simultaneousGesture(
            SpatialTapGesture(coordinateSpace: .named("lab-text-entry-space"))
                .onEnded { tap in
                    guard focusedTextEntry != nil else { return }
                    let tappedAField = textEntryFrames.values.contains { frame in
                        frame.contains(tap.location)
                    }
                    if !tappedAField { focusedTextEntry = nil }
                }
        )
        .preferredColorScheme(.dark)
        .tint(Lab.accent)
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") {
                    focusedTextEntry = nil
                }
                .font(.system(size: Lab.typeSize(13), weight: .semibold))
            }
        }
        // Only the verifying → ready transition is a download finishing.
        // `.ready` alone also appears when the picker lands on an
        // already-installed model, which deserves no fanfare.
        .sensoryFeedback(.success, trigger: model.store.phase) { old, new in
            if case .verifying = old, case .ready = new { true } else { false }
        }
        .sensoryFeedback(.error, trigger: model.statusKind) { _, kind in kind == .err }
        .task {
            Engine.warmKernelPool()
            consumeRequestedAction()
            consumeStagedDocument()
            await loadDebugFixtureIfRequested()
        }
        .onReceive(NotificationCenter.default.publisher(
            for: UIApplication.didReceiveMemoryWarningNotification)
        ) { _ in
            model.releaseEngineIfIdle()
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background { model.releaseEngineIfIdle() }
            if phase == .active {
                consumeRequestedAction()
                consumeStagedDocument()
            }
        }
        .onChange(of: model.isRecognizing) { oldValue, newValue in
            if oldValue, !newValue,
               model.recognition != nil || model.completedPageCount > 0
            {
                withAnimation(.snappy) { destination = .result }
            }
        }
        .onChange(of: photoItem) { _, item in load(photoItem: item) }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [.png, .jpeg, .pdf],
            allowsMultipleSelection: false
        ) { result in load(fileResult: result) }
        .sheet(isPresented: $model.showSelftest) { selftestSheet }
        .fullScreenCover(isPresented: $showLiveCamera) {
            LiveCameraView(model: model)
        }
        .onOpenURL { url in
            guard url.scheme?.lowercased() == "frankenocr" else { return }
            if url.host?.lowercased() == "cancel" {
                model.cancel()
            } else if url.host?.lowercased() == "live-camera" {
                showLiveCamera = true
            } else if url.host?.lowercased() == "recognize" {
                consumeStagedDocument()
            }
        }
        .confirmationDialog(
            "Download \(model.spec.totalBytes.humanBytes)?",
            isPresented: $model.showConsent,
            titleVisibility: .visible
        ) {
            Button("Download") { model.confirmDownload() }
            Button("Not now", role: .cancel) {}
        } message: {
            Text(consentMessage)
        }
        .userActivity("com.frankenocr.workspace") { activity in
            activity.title = "FrankenOCR \(destination.rawValue)"
            activity.isEligibleForHandoff = true
            activity.userInfo = ["route": destination.rawValue]
        }
        .onContinueUserActivity("com.frankenocr.workspace") { activity in
            guard let rawValue = activity.userInfo?["route"] as? String,
                  let restored = Destination(rawValue: rawValue)
            else { return }
            destination = restored
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            load(fileResult: .success([url]))
            return true
        }
        .focusedSceneValue(
            \.ocrCommands,
            OCRCommandActions(
                importFile: { showFileImporter = true },
                recognize: {
                    focusedTextEntry = nil
                    model.recognize()
                },
                liveCamera: { showLiveCamera = true },
                stop: { model.cancel() },
                canRecognize: model.canRecognize,
                canStop: model.isRecognizing
            )
        )
    }

    private var usesWideWorkspace: Bool {
#if targetEnvironment(macCatalyst)
        true
#else
        UIDevice.current.userInterfaceIdiom == .pad
#endif
    }

    private func usesThreeColumnDesktop(width: CGFloat) -> Bool {
#if targetEnvironment(macCatalyst)
        width >= 860
#else
        false
#endif
    }

    private var destinationPicker: some View {
        Picker("Workspace", selection: $destination) {
            ForEach(Destination.allCases) { destination in
                Text(destination.rawValue).tag(destination)
            }
        }
        .pickerStyle(.segmented)
        .accessibilityLabel("FrankenOCR workspace")
    }

    @ViewBuilder
    private var compactWorkspace: some View {
        switch destination {
        case .capture:
            if model.store.phase != .ready { specimenCard }
            pageCard
        case .live:
            liveCameraLaunchCard
        case .result:
            transcriptionCard
        case .models:
            specimenCard
        }
    }

    private var liveCameraLaunchCard: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 14) {
                LabLabel(text: "Live camera")
                HStack(alignment: .top, spacing: 14) {
                    Image(systemName: liveCameraPlatformSymbol)
                        .font(.system(size: Lab.typeSize(34), weight: .bold))
                        .foregroundStyle(Lab.amber)
                    VStack(alignment: .leading, spacing: 5) {
                        Text(liveCameraPlatformTitle)
                            .font(.headline)
                            .foregroundStyle(Lab.textPrimary)
                        Text(liveCameraPlatformDetail)
                        .font(.subheadline)
                        .foregroundStyle(Lab.textDim)
                    }
                }
                Button {
#if targetEnvironment(macCatalyst)
                    destination = .capture
                    showFileImporter = true
#else
                    showLiveCamera = true
#endif
                } label: {
                    Label(liveCameraPlatformAction, systemImage: liveCameraPlatformActionSymbol)
                }
                .buttonStyle(PrimaryButtonStyle())
            }
        }
    }

    private var liveCameraPlatformTitle: String {
#if targetEnvironment(macCatalyst)
        "Live Camera continues on iPhone and iPad"
#else
        "See real words bend into the capture tray"
#endif
    }

    private var liveCameraPlatformDetail: String {
#if targetEnvironment(macCatalyst)
        "Apple's realtime Live Text scanner is unavailable to Mac Catalyst apps. "
            + "Choose an image or PDF here to use the full FrankenOCR model on this Mac."
#else
        "Live Camera uses Apple's fast on-device Vision recognizer, not the Baidu-derived "
            + "FrankenOCR model. It trades some accuracy for realtime responsiveness; still "
            + "images and PDFs keep the full model."
#endif
    }

    private var liveCameraPlatformSymbol: String {
#if targetEnvironment(macCatalyst)
        "macbook.and.iphone"
#else
        "viewfinder.circle.fill"
#endif
    }

    private var liveCameraPlatformAction: String {
#if targetEnvironment(macCatalyst)
        "Choose an image or PDF"
#else
        "Open Live Camera"
#endif
    }

    private var liveCameraPlatformActionSymbol: String {
#if targetEnvironment(macCatalyst)
        "doc.badge.plus"
#else
        "camera.viewfinder"
#endif
    }

    private func consumeRequestedAction() {
        switch FrankenOCRSharedStore.consumeRequestedAction() {
        case .liveCamera:
            destination = .live
            showLiveCamera = true
        case .recognize:
            destination = .capture
        case .none:
            break
        }
    }

    private func consumeStagedDocument() {
        guard let staged = FrankenOCRSharedStore.consumeStagedDocumentURL() else { return }
        load(fileResult: .success([staged]))
    }

    private var consentMessage: String {
        var text = "This downloads the \(model.spec.totalBytes.humanBytes) "
            + "\(model.spec.label) model into this app. "
            + "After that it works completely offline, and nothing you recognize "
            + "ever leaves the device. The download resumes if it is interrupted."
        if model.spec.id == "unlimited-ocr" {
            text += " It is a large file and wants a recent, high-memory device."
        }
        return text
    }

    // ── Header ─────────────────────────────────────────────────────────────

    private var header: some View {
        HStack(spacing: 12) {
            MonsterStatusMark(mood: monsterMood, instrument: .vision, accent: Lab.amber)
                .frame(width: 52, height: 52)

            VStack(alignment: .leading, spacing: 1) {
                Text("franken_ocr")
                    .font(.system(size: Lab.typeSize(16), weight: .heavy, design: .monospaced))
                    .foregroundStyle(Lab.textPrimary)
                Text("READS_LOCALLY")
                    .font(.system(size: Lab.typeSize(8), weight: .heavy, design: .monospaced))
                    .kerning(2.2)
                    .foregroundStyle(Lab.accentInk)
            }
            Spacer()
            Button {
                model.runSelftest()
            } label: {
                Image(systemName: "checkmark.seal")
                    .font(.system(size: Lab.typeSize(16), weight: .semibold))
            }
            .buttonStyle(.plain)
            .foregroundStyle(Lab.accent)
            .frame(width: 44, height: 44)
            .accessibilityLabel("Verify the int8 kernels on this device")
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("FrankenOCR, reads locally")
    }

    private var monsterMood: MonsterMood {
        if model.statusKind == .err { return .error }
        if model.isRecognizing { return .working }
        if model.isLoadingModel { return .waking }
        if model.recognition != nil || model.completedPageCount > 0 { return .success }
        return .idle
    }

    // ── 01 The specimen ────────────────────────────────────────────────────

    private var specimenCard: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 14) {
                LabLabel(text: "01 · The specimen")

                // A `Picker` with `.menu` style centers its own label and
                // ignores the frame alignment around it, which breaks the left
                // rail every other element in this panel sits on. A `Menu` lets
                // the label be laid out explicitly.
                Menu {
                    ForEach(ModelCatalog.all) { spec in
                        Button {
                            model.spec = spec
                        } label: {
                            if spec.id == model.spec.id {
                                Label("\(spec.label) · \(spec.totalBytes.humanBytes)",
                                      systemImage: "checkmark")
                            } else {
                                Text("\(spec.label) · \(spec.totalBytes.humanBytes)")
                            }
                        }
                    }
                } label: {
                    HStack(spacing: 6) {
                        Text("\(model.spec.label) · \(model.spec.totalBytes.humanBytes)")
                            .font(.system(size: Lab.typeSize(17), weight: .semibold))
                            .lineLimit(1)
                            .minimumScaleFactor(0.76)
                            .allowsTightening(true)
                            .foregroundStyle(Lab.accent)
                        Image(systemName: "chevron.up.chevron.down")
                            .font(.system(size: Lab.typeSize(11), weight: .bold))
                            .foregroundStyle(Lab.accent.opacity(0.75))
                        Spacer(minLength: 0)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .frame(minHeight: 44)
                    .contentShape(Rectangle())
                }
                .accessibilityLabel("Model: \(model.spec.label)")

                Text(model.spec.blurb)
                    .font(.system(size: Lab.typeSize(13)))
                    .foregroundStyle(Lab.textDim)

                if let info = model.info {
                    // Hardware capability and the route actually executed are
                    // reported separately, exactly as `robot backends` does —
                    // collapsing them would overclaim on Apple silicon.
                    KeyValueLine(key: "int8 route:",
                                 value: "\(info.detected_tier) / \(info.dense_route)")
                    KeyValueLine(key: "threads:", value: "\(info.threads)")
                }

                if model.spec.id == "got-ocr2" {
                    Toggle("Structured output", isOn: $model.gotFormat)
                        .font(.system(size: Lab.typeSize(13), weight: .semibold))
                        .foregroundStyle(Lab.textMid)
                        .tint(Lab.accent)
                    Text("GOT's `OCR with format:` mode: LaTeX formulas, HTML tables, molecular SMILES, geometry. Off, it reads plain text, which the default model already does faster.")
                        .font(.system(size: Lab.typeSize(11)))
                        .foregroundStyle(Lab.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if model.spec.id == "smolvlm2" {
                    // SmolVLM2 has no instruction modes — the task IS the
                    // question, so this field is the whole control surface for
                    // the model. Empty restores the model-card caption prompt,
                    // which is why it is optional rather than validated.
                    Text("Question")
                        .font(.system(size: Lab.typeSize(13), weight: .semibold))
                        .foregroundStyle(Lab.textMid)
                    TextField("Can you describe this image?", text: $model.question, axis: .vertical)
                        .focused($focusedTextEntry, equals: .smolQuestion)
                        .font(.system(size: Lab.typeSize(13)))
                        .foregroundStyle(Lab.textPrimary)
                        .textFieldStyle(.plain)
                        .lineLimit(1 ... 3)
                        .textInputAutocapitalization(.sentences)
                        .submitLabel(.done)
                        .padding(8)
                        .background(Lab.inset, in: RoundedRectangle(cornerRadius: 8))
                        .reportLabTextEntryFrame(.smolQuestion)
                    Text("Ask anything about the photo. Left blank, it writes a plain description.")
                        .font(.system(size: Lab.typeSize(11)))
                        .foregroundStyle(Lab.textFaint)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if !model.spec.isSupportedOnThisDevice {
                    StatusLine(
                        kind: .warn,
                        text: "This device reports \(Int(ProcessInfo.processInfo.physicalMemory / 1_073_741_824)) GB "
                            + "of memory. \(model.spec.shortName) wants more; it may be terminated mid-page."
                    )
                }

                downloadControls
                StatusLine(kind: model.statusKind, text: model.status)

                if let license = model.licenseNotice ?? Optional(model.spec.license) {
                    Text(license)
                        .font(.system(size: Lab.typeSize(10), design: .monospaced))
                        .foregroundStyle(Lab.textFaint)
                }
            }
        }
    }

    @ViewBuilder
    private var downloadControls: some View {
        switch model.store.phase {
        case .downloading(let asset, let done, let total, let eta):
            VStack(alignment: .leading, spacing: 8) {
                LabProgressBar(fraction: Double(done) / Double(max(total, 1)))
                Text("\(asset) · \(done.humanBytes) / \(total.humanBytes) · \(eta)")
                    .font(.system(size: Lab.typeSize(11), design: .monospaced))
                    .foregroundStyle(Lab.textDim)
                Button("Cancel") { model.store.cancel() }
                    .buttonStyle(GhostButtonStyle(tint: Lab.red))
            }

        case .verifying(let asset):
            HStack(spacing: 10) {
                ProgressView().tint(Lab.accent)
                Text("Verifying \(asset)…")
                    .font(.system(size: Lab.typeSize(12), design: .monospaced))
                    .foregroundStyle(Lab.textDim)
            }

        case .ready:
            HStack(spacing: 10) {
                Label("Installed", systemImage: "checkmark.circle.fill")
                    .font(.system(size: Lab.typeSize(12), weight: .semibold, design: .monospaced))
                    .foregroundStyle(Lab.accent)
                Spacer()
                Button("Remove") { model.clearModel() }
                    .buttonStyle(GhostButtonStyle(tint: Lab.red))
            }

        case .failed(let message):
            VStack(alignment: .leading, spacing: 8) {
                StatusLine(kind: .err, text: message)
                Button("Retry") { model.confirmDownload() }
                    .buttonStyle(GhostButtonStyle())
            }

        case .idle:
            Button("Load model (\(model.spec.totalBytes.humanBytes))") {
                model.requestDownload()
            }
            .buttonStyle(PrimaryButtonStyle())
            .disabled(!model.spec.isSupportedOnThisDevice)
        }
    }

    // ── 02 The page ────────────────────────────────────────────────────────

    private var pageCard: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 14) {
                LabLabel(text: "02 · The page")

                if let preview = model.previewImage {
                    GeometryReader { geometry in
                        ZStack {
                            Image(uiImage: preview)
                                .resizable()
                                .scaledToFit()
                                .frame(
                                    width: geometry.size.width,
                                    height: geometry.size.height
                                )
                            if model.showLayoutBoxes, let recognition = model.recognition {
                                LayoutOverlay(
                                    spans: recognition.layout,
                                    imageSize: preview.size,
                                    music: recognition.music
                                )
                            }
                        }
                    }
                    // `maxHeight` alone does not clip an oversized SwiftUI
                    // child. Tall screenshots were therefore painting through
                    // the controls and transcription below this card. Give the
                    // preview a concrete viewport and clip every overlay to it.
                    .frame(height: 340)
                    .clipShape(RoundedRectangle(cornerRadius: Lab.radius))
                    .background(Lab.inset, in: RoundedRectangle(cornerRadius: Lab.radius))
                    .overlay(RoundedRectangle(cornerRadius: Lab.radius)
                        .strokeBorder(Lab.line, lineWidth: 1))
                } else {
                    dropZone
                }

                inputButtons

                if model.pdfPageCount > 1 {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(spacing: 10) {
                            Text("Preview")
                                .font(.system(size: Lab.typeSize(12), design: .monospaced))
                                .foregroundStyle(Lab.textDim)
                            Stepper(
                                value: $model.previewPage, in: 1...model.pdfPageCount,
                                onEditingChanged: { editing in
                                    if !editing { Task { await model.loadPreviewPage() } }
                                }
                            ) {
                                Text("\(model.previewPage) of \(model.pdfPageCount)")
                                    .font(.system(size: Lab.typeSize(12), design: .monospaced))
                                    .foregroundStyle(Lab.textMid)
                            }
                            .disabled(model.isRecognizing)
                        }

                        HStack(spacing: 8) {
                            Text("Pages")
                                .font(.system(size: Lab.typeSize(12), design: .monospaced))
                                .foregroundStyle(Lab.textDim)
                            TextField("all", text: $model.pageSelection)
                                .focused($focusedTextEntry, equals: .pageSelection)
                                .font(.system(size: Lab.typeSize(12), design: .monospaced))
                                .foregroundStyle(Lab.textMid)
                                .textFieldStyle(.plain)
                                .autocorrectionDisabled()
                                .keyboardType(.numbersAndPunctuation)
                                .padding(.vertical, 7).padding(.horizontal, 10)
                                .background(Lab.inset, in: RoundedRectangle(cornerRadius: 8))
                                .overlay(RoundedRectangle(cornerRadius: 8)
                                    .strokeBorder(Lab.line, lineWidth: 1))
                                .reportLabTextEntryFrame(.pageSelection)
                                .disabled(model.isRecognizing)
                        }

                        Text("Recognize reads the whole document. Leave Pages empty for all \(model.pdfPageCount), or give a range like 3,5-9. Pages using JPEG 2000 or JBIG2 are skipped with a named reason rather than returning a wrong result.")
                            .font(.system(size: Lab.typeSize(11)))
                            .foregroundStyle(Lab.textFaint)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                runControls

                Text("Nothing here is uploaded. The image is read into memory and recognized on this device's own cores.")
                    .font(.system(size: Lab.typeSize(11)))
                    .foregroundStyle(Lab.textFaint)
                    // Inside a constrained column this truncates to one line
                    // unless it is told it may grow vertically.
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var dropZone: some View {
        VStack(spacing: 8) {
            Image(systemName: "doc.viewfinder")
                .font(.system(size: Lab.typeSize(28)))
                .foregroundStyle(Lab.accent.opacity(0.8))
            Text("Choose a page")
                .font(.system(size: Lab.typeSize(14), weight: .semibold))
                .foregroundStyle(Lab.textMid)
            Text("PNG, JPEG, or a scanned PDF")
                .font(.system(size: Lab.typeSize(12)))
                .foregroundStyle(Lab.textFaint)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 34)
        .background(Lab.inset, in: RoundedRectangle(cornerRadius: Lab.radius))
        .overlay(
            RoundedRectangle(cornerRadius: Lab.radius)
                .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [6, 5]))
                .foregroundStyle(Lab.accent.opacity(0.35))
        )
        // iPad: dropping a file onto the zone is the natural gesture.
        .dropDestination(for: Data.self) { items, _ in
            guard let data = items.first else { return false }
            Task { await model.accept(data: data, name: "dropped") }
            return true
        }
    }

    private var inputButtons: some View {
        VStack(spacing: 10) {
#if targetEnvironment(macCatalyst)
            HStack(spacing: 10) { photoButton; filesButton }
#else
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) { photoButton; filesButton; cameraButton }
                VStack(spacing: 10) {
                    photoButton
                    HStack(spacing: 10) { filesButton; cameraButton }
                }
            }
            Button { showLiveCamera = true } label: {
                Label("Live Camera", systemImage: "viewfinder.circle.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(GhostButtonStyle(tint: Lab.accent))
            .disabled(model.isRecognizing)
#endif
        }
    }

    private var photoButton: some View {
        PhotosPicker(selection: $photoItem, matching: .images) {
            Label("Photos", systemImage: "photo.on.rectangle")
        }
        .buttonStyle(GhostButtonStyle())
    }

    private var filesButton: some View {
        Button { showFileImporter = true } label: {
            Label("Files", systemImage: "folder")
        }
        .buttonStyle(GhostButtonStyle())
    }

    private var cameraButton: some View {
        Button { showCamera = true } label: {
            Label("Camera", systemImage: "camera")
        }
        .buttonStyle(GhostButtonStyle())
        .fullScreenCover(isPresented: $showCamera) {
            CameraPicker { data in
                if let data { Task { await model.accept(data: data, name: "photo.jpg") } }
                showCamera = false
            }
            .ignoresSafeArea()
        }
    }

    @ViewBuilder
    private var runControls: some View {
        if model.isRecognizing {
            OcrReactorView(
                progress: model.progress,
                fraction: model.isDocumentRun
                    ? model.documentProgressFraction
                    : model.progressFraction,
                isEstimated: model.progressIsEstimated,
                elapsed: model.elapsed,
                currentPage: model.currentPageIndex.map { $0 + 1 },
                pageCount: model.pageOutcomes.count,
                completedPages: model.pageOutcomes.filter(\.isTerminal).count,
                emittedCharacters: model.pageOutcomes.reduce(0) { $0 + $1.text.count },
                cancel: model.cancel
            )
        } else {
            Button("Recognize") { model.recognize() }
                .buttonStyle(PrimaryButtonStyle())
                .disabled(!model.canRecognize)
                .sensoryFeedback(.success, trigger: model.recognition?.output)
        }
    }

    // ── 03 The transcription ───────────────────────────────────────────────

    private var transcriptionCard: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    LabLabel(text: "03 · The transcription")
                    Spacer()
                    if model.recognition != nil, !model.spec.producesMusicXML {
                        Picker("View", selection: $model.viewSource) {
                            Text("Rendered").tag(false)
                            Text("Source").tag(true)
                        }
                        .pickerStyle(.segmented)
                        .frame(width: 190)
                    }
                }

                if model.isDocumentRun {
                    pageLedger
                }

                if let recognition = model.recognition {
                    resultMeta(recognition)

                    if !recognition.layout.isEmpty {
                        Toggle("Show layout boxes", isOn: $model.showLayoutBoxes)
                            .font(.system(size: Lab.typeSize(12), design: .monospaced))
                            .foregroundStyle(Lab.textDim)
                            .tint(Lab.accent)
                    }

                    musicWarnings(recognition)

                    // On a document run this is every recognized page joined
                    // with page markers, not just the page last finished.
                    if model.viewSource || model.spec.producesMusicXML {
                        ScrollView([.horizontal, .vertical], showsIndicators: true) {
                            Text(model.displayText)
                                .font(.system(size: Lab.typeSize(12), design: .monospaced))
                                .foregroundStyle(Lab.textMid)
                                .textSelection(.enabled)
                                .padding(12)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .background(Lab.inset, in: RoundedRectangle(cornerRadius: Lab.radius))
                        .frame(maxHeight: 420)
                        .clipped()
                    } else {
                        // Rendered Markdown can be taller than the result card.
                        // A frame constraint does not itself prevent drawing
                        // outside that frame, so make the result its own scroll
                        // viewport rather than letting it overlap adjacent UI.
                        ScrollView(.vertical, showsIndicators: true) {
                            MarkdownView(markdown: model.displayText)
                                .padding(.vertical, 2)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(maxHeight: 420)
                        .clipped()
                    }

                    exportControls
                } else {
                    Text("Output appears here: Markdown for a document page, MusicXML for a staff. These are the same bytes the CLI writes.")
                        .font(.system(size: Lab.typeSize(13)))
                        .foregroundStyle(Lab.textFaint)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.vertical, 26)
                }
            }
        }
    }

    /// The three ways a result leaves the app: the raw bytes as a file, the
    /// same content as a styled self-contained web page (Markdown lanes only),
    /// and the clipboard for "paste this into Notes" — the lightest path,
    /// which a share sheet is a heavy way to spell.
    private var exportControls: some View {
        // Computed ONCE per render: `displayText` joins the whole document,
        // and `ViewThatFits` measures both of its children, so putting these
        // inside the button builder would redo a multi-megabyte join per
        // layout candidate on a long book.
        let text = model.displayText
        let html = model.canExportHtml ? model.htmlExportPayload() : nil
        return ViewThatFits(in: .horizontal) {
            HStack(spacing: 10) { exportButtons(text: text, html: html) }
            VStack(spacing: 10) { exportButtons(text: text, html: html) }
        }
    }

    @ViewBuilder
    private func exportButtons(
        text: String,
        html: (provenance: HtmlExport.Provenance, sections: [HtmlExport.Section])?
    ) -> some View {
        ShareLink(
            item: TranscriptionFile(text: text, filename: model.exportFilename),
            preview: SharePreview(model.exportFilename)
        ) {
            Label("Export \(model.exportFilename)", systemImage: "square.and.arrow.up")
        }
        .buttonStyle(GhostButtonStyle())

        if let html {
            ShareLink(
                item: HtmlDocumentFile(
                    provenance: html.provenance,
                    sections: html.sections,
                    filename: model.htmlExportFilename
                ),
                preview: SharePreview(model.htmlExportFilename)
            ) {
                Label("Export .html", systemImage: "doc.richtext")
            }
            .buttonStyle(GhostButtonStyle())
        }

        Button {
            UIPasteboard.general.string = text
            copied = true
            copyResetTask?.cancel()
            copyResetTask = Task {
                try? await Task.sleep(for: .seconds(1.6))
                copied = false
            }
        } label: {
            Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
        }
        .buttonStyle(GhostButtonStyle())
        .sensoryFeedback(.success, trigger: copied) { _, isCopied in isCopied }
    }

    /// One row per page: what happened, and why if it did not.
    ///
    /// Scrolls on its own for a long book. A plain stack under a `maxHeight`
    /// would CLIP the overflow rather than scroll it, leaving the tail of a
    /// 300-page ledger unreachable.
    private var pageLedger: some View {
        ScrollView(.vertical) {
            pageLedgerRows
        }
        .frame(maxHeight: model.pageOutcomes.count > 6 ? 220 : .infinity)
        .fixedSize(horizontal: false, vertical: model.pageOutcomes.count <= 6)
        .background(Lab.inset, in: RoundedRectangle(cornerRadius: Lab.radius))
        .overlay(RoundedRectangle(cornerRadius: Lab.radius).strokeBorder(Lab.line, lineWidth: 1))
    }

    private var pageLedgerRows: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(model.pageOutcomes) { outcome in
                HStack(alignment: .top, spacing: 10) {
                    Text(String(format: "%3d", outcome.id))
                        .font(.system(size: Lab.typeSize(11), design: .monospaced))
                        .foregroundStyle(Lab.textFaint)
                    icon(for: outcome.state)
                    Text(label(for: outcome.state))
                        .font(.system(size: Lab.typeSize(11), design: .monospaced))
                        .foregroundStyle(color(for: outcome.state))
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 0)
                }
                .padding(.vertical, 4)
                .accessibilityElement(children: .combine)
                .accessibilityLabel("Page \(outcome.id), \(label(for: outcome.state))")
            }
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func icon(for state: PageOutcome.State) -> some View {
        switch state {
        case .queued:
            Image(systemName: "circle.dotted")
                .font(.system(size: Lab.typeSize(11))).foregroundStyle(Lab.textFaint)
        case .running:
            ProgressView().controlSize(.mini).tint(Lab.accent)
        case .done:
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: Lab.typeSize(11))).foregroundStyle(Lab.accent)
        case .skipped:
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: Lab.typeSize(11))).foregroundStyle(Lab.red)
        }
    }

    private func label(for state: PageOutcome.State) -> String {
        switch state {
        case .queued: "queued"
        case .running: "reading…"
        case .done(let characters, let seconds):
            String(format: "%d characters · %.1fs", characters, seconds)
        case .skipped(let reason): "skipped: \(reason)"
        }
    }

    private func color(for state: PageOutcome.State) -> Color {
        switch state {
        case .queued: Lab.textFaint
        case .running: Lab.textMid
        case .done: Lab.textDim
        case .skipped: Lab.red
        }
    }

    private func resultMeta(_ recognition: Recognition) -> some View {
        var text = "\(model.spec.shortName) · \(model.displayText.count) characters"
        if let music = recognition.music {
            text = "\(music.staves.count) staff/staves recognized · "
                + "\(music.skips.count) skipped · \(music.warnings.count) sanity warning(s)"
        }
        if let seconds = model.lastRunSeconds {
            text += String(format: " · %.1fs on this device", seconds)
        }
        return KeyValueLine(key: "", value: text, valueColor: Lab.textDim)
    }

    @ViewBuilder
    private func musicWarnings(_ recognition: Recognition) -> some View {
        if let music = recognition.music, !music.warnings.isEmpty || !music.skips.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(music.skips) { skip in
                    StatusLine(kind: .err,
                               text: "staff \(skip.index + 1) skipped: \(skip.reason)")
                }
                ForEach(music.warnings) { warning in
                    StatusLine(
                        kind: .warn,
                        text: "\(warning.kind) (part \(warning.part), measure \(warning.measure)): \(warning.detail)"
                    )
                }
                Text("Sanity warnings are annotate-only observations the engine makes about its own transcription. Honest signal, not silent cleanup.")
                    .font(.system(size: Lab.typeSize(11)))
                    .foregroundStyle(Lab.textFaint)
            }
        }
    }

    // ── Selftest ───────────────────────────────────────────────────────────

    private var selftestSheet: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text("This re-runs the dispatched int8 GEMM against a bit-identical scalar oracle on THIS device, including the worst-case K=6848 accumulation row. It is the proof that the kernels are correct on hardware the binary was never built on.")
                        .font(.system(size: Lab.typeSize(13)))
                        .foregroundStyle(Lab.textDim)
                    Text(model.selftestJSON ?? "…")
                        .font(.system(size: Lab.typeSize(11), design: .monospaced))
                        .foregroundStyle(Lab.textMid)
                        .textSelection(.enabled)
                }
                .padding(18)
            }
            .background(LabBackground())
            .navigationTitle("Kernel selftest")
            .navigationBarTitleDisplayMode(.inline)
        }
        .preferredColorScheme(.dark)
    }

    // ── Footer ─────────────────────────────────────────────────────────────

    private var footer: some View {
        VStack(spacing: 8) {
            Text("Recognized locally. Nothing is uploaded, and there is no analytics of any kind.")
                .font(.system(size: Lab.typeSize(11)))
                .foregroundStyle(Lab.textFaint)
                .multilineTextAlignment(.center)
            // A string LITERAL, so `Text` parses the Markdown link — the same
            // free tappable-link trick frankentts's footer uses.
            Text("© 2026 Jeffrey Emanuel · [franken-ocr.com](https://franken-ocr.com)")
                .font(.system(size: Lab.typeSize(10), design: .monospaced))
                .foregroundStyle(Lab.textFaint.opacity(0.7))
                .tint(Lab.accent.opacity(0.8))
            Text(
                "If you like this free app, please show your appreciation by trying out my paid skills site at [JeffreysSkills.md](https://jeffreys-skills.md)."
            )
            .font(.system(size: Lab.typeSize(10), design: .monospaced))
            .foregroundStyle(Lab.textFaint.opacity(0.72))
            .tint(Lab.accent.opacity(0.8))
            .multilineTextAlignment(.center)
            .frame(maxWidth: 320)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 8)
    }

    // ── Input loading ──────────────────────────────────────────────────────

    /// Debug-only: load a fixture named by `FOCR_DEBUG_INPUT` at launch, so a
    /// multi-page document run can be driven from the command line instead of
    /// through the file picker. Compiled out of release builds entirely; the
    /// same shape as frankentts's `FTTS_DEBUG_*` harnesses.
    private func loadDebugFixtureIfRequested() async {
        #if DEBUG
        guard let path = ProcessInfo.processInfo.environment["FOCR_DEBUG_INPUT"],
              let data = FileManager.default.contents(atPath: path)
        else { return }
        // `await`, not fire-and-forget: a PDF is parsed off the main actor, so
        // starting the run before acceptance completes would find nothing to do.
        await model.accept(data: data, name: (path as NSString).lastPathComponent)
        if ProcessInfo.processInfo.environment["FOCR_DEBUG_RUN"] != nil {
            model.recognize()
        }
        #endif
    }

    private func load(photoItem: PhotosPickerItem?) {
        guard let photoItem else { return }
        Task {
            // A failure here surfaces exactly as a file-importer failure does;
            // silently doing nothing after a picker tap reads as a dead button.
            if let data = try? await photoItem.loadTransferable(type: Data.self) {
                await model.accept(data: data, name: "photo")
            } else {
                model.status = "Could not read that photo."
                model.statusKind = .err
            }
            // Cleared so picking the SAME photo again re-fires `onChange`;
            // the nil round trip is absorbed by the guard above.
            self.photoItem = nil
        }
    }

    private func load(fileResult: Result<[URL], Error>) {
        switch fileResult {
        case .success(let urls):
            guard let url = urls.first else { return }
            // The security scope must be released only after the bytes are
            // read, which happens here — `accept` works on the copy.
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            guard let data = try? Data(contentsOf: url) else {
                model.status = "Could not read that file."
                model.statusKind = .err
                return
            }
            Task { await model.accept(data: data, name: url.lastPathComponent) }
        case .failure(let error):
            model.status = error.localizedDescription
            model.statusKind = .err
        }
    }
}

/// Draws the engine's grounded layout boxes over the page.
///
/// The browser playground receives this array and never renders it. It is one
/// of the clearest wins a native UI gets for free: you can see exactly what the
/// model thought each region was.
private struct LayoutOverlay: View {
    let spans: [Recognition.LayoutSpan]
    let imageSize: CGSize
    let music: Recognition.MusicMeta?

    var body: some View {
        GeometryReader { geo in
            // The image is drawn `.fit`, so derive the letterboxed content rect.
            let scale = min(geo.size.width / imageSize.width,
                            geo.size.height / imageSize.height)
            let drawn = CGSize(width: imageSize.width * scale,
                               height: imageSize.height * scale)
            let origin = CGPoint(x: (geo.size.width - drawn.width) / 2,
                                 y: (geo.size.height - drawn.height) / 2)

            ZStack(alignment: .topLeading) {
                ForEach(spans) { span in
                    ForEach(Array(span.boxes.enumerated()), id: \.offset) { _, box in
                        if box.count == 4 {
                            let rect = CGRect(
                                x: origin.x + CGFloat(box[0]) * scale,
                                y: origin.y + CGFloat(box[1]) * scale,
                                width: CGFloat(box[2] - box[0]) * scale,
                                height: CGFloat(box[3] - box[1]) * scale
                            )
                            Rectangle()
                                .strokeBorder(Lab.accent.opacity(0.85), lineWidth: 1)
                                .frame(width: rect.width, height: rect.height)
                                .offset(x: rect.minX, y: rect.minY)
                        }
                    }
                }
                // Skipped staves are drawn in the failure color, matching the
                // site's music legend.
                ForEach(music?.skips ?? []) { skip in
                    if skip.bbox.count == 4 {
                        let rect = CGRect(
                            x: origin.x + CGFloat(skip.bbox[0]) * scale,
                            y: origin.y + CGFloat(skip.bbox[1]) * scale,
                            width: CGFloat(skip.bbox[2] - skip.bbox[0]) * scale,
                            height: CGFloat(skip.bbox[3] - skip.bbox[1]) * scale
                        )
                        Rectangle()
                            .strokeBorder(Lab.red.opacity(0.9), lineWidth: 1.5)
                            .frame(width: rect.width, height: rect.height)
                            .offset(x: rect.minX, y: rect.minY)
                    }
                }
            }
        }
        .allowsHitTesting(false)
    }
}

/// A transcription packaged for `ShareLink` with the right filename and type.
struct TranscriptionFile: Transferable {
    let text: String
    let filename: String

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .plainText) { file in
            let url = try Self.uniqueExportURL(filename: file.filename)
            try file.text.write(to: url, atomically: true, encoding: .utf8)
            return SentTransferredFile(url)
        }
    }
}

extension TranscriptionFile {
    /// A fresh directory per share, keeping the human-readable filename.
    ///
    /// A fixed temp path would let a second share overwrite the file while a
    /// lazy consumer of the first (AirDrop reads on send) still holds it —
    /// the same in-flight-share hazard frankentts's exporter guards against.
    /// The system prunes its temp directory, so nothing accumulates.
    static func uniqueExportURL(filename: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent(filename)
    }
}

/// The same result as a styled, self-contained web page. The document is
/// rendered inside the transfer, not on every view update — the payload is
/// captured as plain values so a 300-page book costs nothing until the user
/// actually shares it.
struct HtmlDocumentFile: Transferable {
    let provenance: HtmlExport.Provenance
    let sections: [HtmlExport.Section]
    let filename: String

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .html) { file in
            let url = try TranscriptionFile.uniqueExportURL(filename: file.filename)
            let html = HtmlExport.document(provenance: file.provenance, sections: file.sections)
            try html.write(to: url, atomically: true, encoding: .utf8)
            return SentTransferredFile(url)
        }
    }
}

/// Minimal camera bridge — `UIImagePickerController` is still the shortest path
/// to a single still capture.
struct CameraPicker: UIViewControllerRepresentable {
    let onCapture: (Data?) -> Void

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = UIImagePickerController.isSourceTypeAvailable(.camera)
            ? .camera : .photoLibrary
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_: UIImagePickerController, context _: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator(onCapture: onCapture) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate,
                             UINavigationControllerDelegate {
        let onCapture: (Data?) -> Void
        init(onCapture: @escaping (Data?) -> Void) { self.onCapture = onCapture }

        func imagePickerController(
            _: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            let image = info[.originalImage] as? UIImage
            onCapture(image?.jpegData(compressionQuality: 0.92))
        }

        func imagePickerControllerDidCancel(_: UIImagePickerController) { onCapture(nil) }
    }
}
