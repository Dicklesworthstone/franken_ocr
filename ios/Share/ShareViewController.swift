import UIKit
import UniformTypeIdentifiers

final class ShareViewController: UIViewController {
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
        guard let pair = providers.lazy.compactMap({ provider -> (NSItemProvider, UTType)? in
            supported.first(where: { provider.hasItemConformingToTypeIdentifier($0.identifier) })
                .map { (provider, $0) }
        }).first else {
            showFailure("Share one image or PDF to recognize it.")
            return
        }

        pair.0.loadFileRepresentation(forTypeIdentifier: pair.1.identifier) { [weak self] url, _ in
            guard let url else {
                Task { @MainActor in self?.showFailure("The shared document could not be opened.") }
                return
            }
            do {
                _ = try FrankenOCRSharedStore.stageDocument(
                    from: url,
                    preferredExtension: pair.1.preferredFilenameExtension
                )
                Task { @MainActor in
                    self?.statusLabel.text = "Specimen secured locally. Ready for private recognition."
                    self?.openButton.isEnabled = true
                }
            } catch {
                Task { @MainActor in self?.showFailure("Could not stage that file: \(error.localizedDescription)") }
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
