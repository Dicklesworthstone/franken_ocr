import Observation
import SwiftUI
import UIKit

struct RecognitionHistorySheet: View {
    @Bindable var history: RecognitionHistoryStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ZStack {
                LabBackground()
                ScrollView {
                    RecognitionHistoryLibrary(history: history)
                        .padding(16)
                }
                .scrollIndicators(.hidden)
            }
            .navigationTitle("Recent results")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .accessibilityIdentifier("recognition-history-sheet")
    }
}

struct RecognitionHistoryLibrary: View {
    @Bindable var history: RecognitionHistoryStore
    @State private var confirmClear = false
    @State private var copiedID: UUID?

    var body: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    LabLabel(text: "Private library")
                    Spacer()
                    if !history.entries.isEmpty {
                        Button("Clear All", role: .destructive) { confirmClear = true }
                            .buttonStyle(GhostButtonStyle(tint: Lab.red))
                    }
                }
                privacyNote
                if history.entries.isEmpty {
                    ContentUnavailableView(
                        "No recent results",
                        systemImage: "doc.text.magnifyingglass",
                        description: Text("Finished recognition appears here automatically.")
                    )
                    .foregroundStyle(Lab.textDim)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 24)
                } else {
                    LazyVStack(spacing: 12) {
                        ForEach(history.entries) { entry in
                            historyCard(entry)
                        }
                    }
                }
            }
        }
        .alert("Clear recent results?", isPresented: $confirmClear) {
            Button("Clear All", role: .destructive) { history.deleteAll() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(
                "This permanently removes every locally saved transcript. "
                    + "Your models and source documents stay intact."
            )
        }
        .accessibilityIdentifier("recognition-history-library")
    }

    private var privacyNote: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "lock.shield.fill")
                .foregroundStyle(Lab.accent)
            VStack(alignment: .leading, spacing: 5) {
                Text(
                    "Only exported text and minimal provenance are saved. Source images, PDFs, "
                        + "questions, and layout coordinates are never copied into history. "
                        + "The newest 20 results are kept for up to 14 days."
                )
                .font(.system(size: Lab.typeSize(11), weight: .medium))
                .foregroundStyle(Lab.textDim)
                Text(storageSummary)
                    .font(.system(size: Lab.typeSize(9), weight: .bold, design: .monospaced))
                    .foregroundStyle(Lab.violet)
            }
        }
        .padding(12)
        .background(Lab.inset, in: RoundedRectangle(cornerRadius: 12))
        .accessibilityElement(children: .combine)
    }

    private var storageSummary: String {
        let size = ByteCountFormatter.string(
            fromByteCount: Int64(history.storageBytes),
            countStyle: .file
        )
        let noun = history.entries.count == 1 ? "result" : "results"
        return "\(history.entries.count) \(noun) · \(size) on this device"
    }

    private func historyCard(_ entry: RecognitionHistoryEntry) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            historyHeader(entry)
            historyPreview(entry)
            historyActions(entry)
            Text(entrySummary(entry))
                .font(.system(size: Lab.typeSize(9), design: .monospaced))
                .foregroundStyle(Lab.textFaint)
        }
        .padding(12)
        .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Lab.line, lineWidth: 1))
        .accessibilityIdentifier("history-entry-\(entry.id.uuidString)")
    }

    private func historyHeader(_ entry: RecognitionHistoryEntry) -> some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 3) {
                Text(entry.sourceName)
                    .font(.system(size: Lab.typeSize(15), weight: .bold))
                    .foregroundStyle(Lab.textPrimary)
                    .lineLimit(1)
                Text(entry.createdAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.system(size: Lab.typeSize(10), design: .monospaced))
                    .foregroundStyle(Lab.textFaint)
            }
            Spacer()
            Text(entry.modelName)
                .font(.system(size: Lab.typeSize(9), weight: .bold, design: .monospaced))
                .foregroundStyle(Lab.accent)
                .lineLimit(1)
        }
    }

    @ViewBuilder
    private func historyPreview(_ entry: RecognitionHistoryEntry) -> some View {
        if let preview = history.text(for: entry) {
            Text(preview)
                .font(.system(size: Lab.typeSize(11), design: .monospaced))
                .foregroundStyle(Lab.textMid)
                .lineLimit(4)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            StatusLine(kind: .warn, text: "The saved text is unavailable.")
        }
    }

    private func historyActions(_ entry: RecognitionHistoryEntry) -> some View {
        HStack(spacing: 8) {
            if let url = history.fileURL(for: entry) {
                ShareLink(item: url) {
                    Label("Share", systemImage: "square.and.arrow.up")
                }
                .buttonStyle(GhostButtonStyle())
            }
            Button {
                UIPasteboard.general.string = history.text(for: entry)
                copiedID = entry.id
            } label: {
                Label(
                    copiedID == entry.id ? "Copied" : "Copy",
                    systemImage: copiedID == entry.id ? "checkmark" : "doc.on.doc"
                )
            }
            .buttonStyle(GhostButtonStyle(tint: Lab.accent))
            .disabled(history.text(for: entry) == nil)
            Spacer()
            Button(role: .destructive) { history.delete(entry) } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(GhostButtonStyle(tint: Lab.red))
            .accessibilityLabel("Delete \(entry.sourceName) result")
        }
    }

    private func entrySummary(_ entry: RecognitionHistoryEntry) -> String {
        var parts = [
            "\(entry.characterCount) characters",
            "\(entry.pageCount) page\(entry.pageCount == 1 ? "" : "s")"
        ]
        if let seconds = entry.seconds {
            parts.append(String(format: "%.1fs on device", seconds))
        }
        return parts.joined(separator: " · ")
    }
}
