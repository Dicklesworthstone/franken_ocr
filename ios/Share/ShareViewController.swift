import UIKit
import UniformTypeIdentifiers

final class ShareViewController: UIViewController {
    private struct ProviderSelection {
        let provider: NSItemProvider
        let type: UTType
        let isPDF: Bool
    }

    private let statusLabel = UILabel()
    private let openButton = UIButton(type: .system)

    override func viewDidLoad() {
        super.viewDidLoad()
        configureView()
        stageSharedDocument()
    }

    private func configureView() {
        view.backgroundColor = UIColor(red: 0.004, green: 0.028, blue: 0.018, alpha: 1)
        let mark = UIImageView(image: UIImage(systemName: "text.viewfinder"))
        mark.tintColor = UIColor(red: 0.98, green: 0.72, blue: 0.20, alpha: 1)
        mark.preferredSymbolConfiguration = UIImage.SymbolConfiguration(pointSize: 30, weight: .bold)

        let title = UILabel()
        title.text = "VISION TABLE"
        title.textColor = .white
        title.font = .monospacedSystemFont(ofSize: 18, weight: .black)

        statusLabel.text = "Securing the specimen locally…"
        statusLabel.textColor = UIColor.white.withAlphaComponent(0.66)
        statusLabel.font = .preferredFont(forTextStyle: .subheadline)
        statusLabel.numberOfLines = 0
        statusLabel.textAlignment = .center

        var configuration = UIButton.Configuration.filled()
        configuration.title = "Open FrankenOCR"
        configuration.image = UIImage(systemName: "viewfinder")
        configuration.imagePadding = 8
        configuration.baseBackgroundColor = UIColor(red: 0.04, green: 0.55, blue: 0.34, alpha: 1)
        configuration.cornerStyle = .capsule
        openButton.configuration = configuration
        openButton.isEnabled = false
        openButton.addTarget(self, action: #selector(openVisionTable), for: .touchUpInside)

        let cancel = UIButton(type: .system)
        cancel.setTitle("Cancel", for: .normal)
        cancel.tintColor = UIColor.white.withAlphaComponent(0.62)
        cancel.addTarget(self, action: #selector(cancelShare), for: .touchUpInside)

        let stack = UIStackView(arrangedSubviews: [mark, title, statusLabel, openButton, cancel])
        stack.axis = .vertical
        stack.alignment = .center
        stack.spacing = 16
        stack.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(greaterThanOrEqualTo: view.leadingAnchor, constant: 24),
            stack.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor, constant: -24),
            stack.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            stack.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            statusLabel.widthAnchor.constraint(lessThanOrEqualToConstant: 360),
            openButton.heightAnchor.constraint(greaterThanOrEqualToConstant: 48),
        ])
    }

    private func stageSharedDocument() {
        let providers = (extensionContext?.inputItems as? [NSExtensionItem])?
            .compactMap(\.attachments).flatMap { $0 } ?? []
        let supported = [UTType.pdf, .png, .jpeg, .image]
        let selections = providers.compactMap { provider -> ProviderSelection? in
            guard let type = supported.first(where: {
                provider.hasItemConformingToTypeIdentifier($0.identifier)
            }) else { return nil }
            return ProviderSelection(provider: provider, type: type, isPDF: type == .pdf)
        }
        guard selections.count == providers.count else {
            showFailure("Every shared item must be a PNG, JPEG, image, or PDF.")
            return
        }
        do {
            try FrankenOCRSharedStore.validateStagedSelection(
                itemCount: selections.count,
                pdfCount: selections.filter(\.isPDF).count
            )
        } catch {
            showFailure(error.localizedDescription)
            return
        }

        let noun = selections.count == 1
            ? (selections[0].isPDF ? "PDF" : "image")
            : "images"
        statusLabel.text = "Securing \(selections.count == 1 ? "the" : String(selections.count)) \(noun) locally…"
        stageSelections(selections, at: 0, staged: [])
    }

    /// Load serially so order is deterministic and each provider-owned
    /// temporary URL is copied before its completion callback returns.
    private func stageSelections(
        _ selections: [ProviderSelection],
        at index: Int,
        staged: [FrankenOCRSharedStore.StagedDocument]
    ) {
        guard index < selections.count else {
            do {
                try FrankenOCRSharedStore.publishStagedDocuments(staged)
                Task { @MainActor [weak self] in
                    let noun = staged.count == 1 ? "specimen" : "images"
                    self?.statusLabel.text = "\(staged.count) \(noun) secured locally in selection order."
                    self?.openButton.isEnabled = true
                }
            } catch {
                FrankenOCRSharedStore.discardStagedDocuments(staged)
                Task { @MainActor [weak self] in
                    self?.showFailure("Could not publish that selection: \(error.localizedDescription)")
                }
            }
            return
        }

        let selection = selections[index]
        selection.provider.loadFileRepresentation(
            forTypeIdentifier: selection.type.identifier
        ) { [weak self] url, error in
            guard let url else {
                FrankenOCRSharedStore.discardStagedDocuments(staged)
                let detail = error?.localizedDescription ?? "The shared item could not be opened."
                Task { @MainActor [weak self] in self?.showFailure(detail) }
                return
            }
            do {
                let visibleName = selection.provider.suggestedName ?? url.lastPathComponent
                let record = try FrankenOCRSharedStore.stageDocument(
                    from: url,
                    preferredExtension: selection.type.preferredFilenameExtension,
                    displayName: visibleName
                )
                guard let self else {
                    FrankenOCRSharedStore.discardStagedDocuments(staged + [record])
                    return
                }
                self.stageSelections(selections, at: index + 1, staged: staged + [record])
            } catch {
                FrankenOCRSharedStore.discardStagedDocuments(staged)
                Task { @MainActor [weak self] in
                    self?.showFailure("Could not stage item \(index + 1): \(error.localizedDescription)")
                }
            }
        }
    }

    private func showFailure(_ message: String) {
        statusLabel.text = message
        statusLabel.textColor = UIColor(red: 0.97, green: 0.44, blue: 0.44, alpha: 1)
    }

    @objc private func openVisionTable() {
        guard let url = URL(string: "frankenocr://recognize") else { return }
        extensionContext?.open(url) { [weak self] _ in
            self?.extensionContext?.completeRequest(returningItems: nil)
        }
    }

    @objc private func cancelShare() {
        extensionContext?.cancelRequest(withError: CocoaError(.userCancelled))
    }
}
