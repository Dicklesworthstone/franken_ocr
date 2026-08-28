import SwiftUI

/// Renders the model's Markdown natively.
///
/// The site renders into a sandboxed iframe because it is handling untrusted
/// HTML in a browser. Native SwiftUI has no such hazard — there is no script to
/// execute — so this parses the Markdown into typed blocks and draws them, which
/// is both safer and far better looking than a WKWebView would be.
///
/// Styling mirrors the site's `RENDER_CSS` so a page looks the same in the app,
/// in the browser playground, and in the exported HTML.
struct MarkdownView: View {
    let markdown: String

    private var presentation: (blocks: [MarkdownBlock], omittedFigures: Int) {
        let parsed = MarkdownBlock.parse(markdown)
        let visible = parsed.filter { block in
            guard case .figure(let caption) = block else { return true }
            return !caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        return (visible, parsed.count - visible.count)
    }

    var body: some View {
        let content = presentation
        VStack(alignment: .leading, spacing: 14) {
            ForEach(Array(content.blocks.enumerated()), id: \.offset) { _, block in
                view(for: block)
            }
            if content.omittedFigures > 0 {
                Label(
                    "\(content.omittedFigures) non-text figure region\(content.omittedFigures == 1 ? "" : "s") omitted from this reading view",
                    systemImage: "photo.stack"
                )
                .font(.system(size: Lab.typeSize(11), design: .monospaced))
                .foregroundStyle(Lab.textFaint)
                .padding(.top, 4)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func view(for block: MarkdownBlock) -> some View {
        switch block {
        case .heading(let level, let text):
            Text(inline(text))
                .font(.system(size: headingSize(level), weight: .heavy))
                .kerning(-0.4)
                .foregroundStyle(Lab.textPrimary)
                .padding(.top, level <= 2 ? 6 : 2)

        case .paragraph(let text):
            Text(inline(text))
                .font(.system(size: Lab.typeSize(15)))
                .lineSpacing(4)
                .foregroundStyle(Lab.textMid)
                .textSelection(.enabled)

        case .bullet(let items):
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text("•").foregroundStyle(Lab.accent)
                        Text(inline(item))
                            .font(.system(size: Lab.typeSize(15)))
                            .foregroundStyle(Lab.textMid)
                    }
                }
            }

        case .ordered(let items):
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(items.enumerated()), id: \.offset) { index, item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text("\(index + 1).")
                            .font(.system(size: Lab.typeSize(14), design: .monospaced))
                            .foregroundStyle(Lab.accent)
                        Text(inline(item))
                            .font(.system(size: Lab.typeSize(15)))
                            .foregroundStyle(Lab.textMid)
                    }
                }
            }

        case .code(let text):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(text)
                    .font(.system(size: Lab.typeSize(12.5), design: .monospaced))
                    .foregroundStyle(Lab.textMid)
                    .padding(12)
            }
            .background(Lab.inset, in: RoundedRectangle(cornerRadius: Lab.radius))
            .overlay(RoundedRectangle(cornerRadius: Lab.radius)
                .strokeBorder(Lab.line, lineWidth: 1))

        case .table(let header, let rows):
            // Wide content scrolls inside its own container; the page never
            // scrolls sideways.
            ScrollView(.horizontal, showsIndicators: true) {
                MarkdownTable(header: header, rows: rows)
            }

        case .rule:
            Rectangle().frame(height: 1).foregroundStyle(Lab.line)

        case .figure(let caption):
            HStack(spacing: 8) {
                Image(systemName: "photo")
                Text(caption.isEmpty ? "figure (not extracted)" : caption)
            }
            .font(.system(size: Lab.typeSize(13)).italic())
            .foregroundStyle(Lab.textFaint)
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .overlay(RoundedRectangle(cornerRadius: Lab.radius)
                .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [4, 4]))
                .foregroundStyle(Lab.line))
        }
    }

    private func headingSize(_ level: Int) -> CGFloat {
        switch level {
        case 1: 26
        case 2: 22
        case 3: 19
        case 4: 17
        default: 15
        }
    }

    /// Inline emphasis, code, and links via `AttributedString`'s Markdown
    /// parser — which handles exactly the inline subset we care about and
    /// nothing dangerous.
    private func inline(_ text: String) -> AttributedString {
        (try? AttributedString(
            markdown: text,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        )) ?? AttributedString(text)
    }
}

private struct MarkdownTable: View {
    let header: [String]
    let rows: [[String]]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 0) {
                ForEach(Array(header.enumerated()), id: \.offset) { _, cell in
                    Text(cell)
                        .font(.system(size: Lab.typeSize(11), weight: .heavy, design: .monospaced))
                        .kerning(1.2)
                        .textCase(.uppercase)
                        .foregroundStyle(Lab.accent)
                        .frame(minWidth: 90, alignment: .leading)
                        .padding(.vertical, 9).padding(.horizontal, 11)
                }
            }
            .background(Lab.accent.opacity(0.05))

            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                Divider().overlay(Lab.line)
                HStack(spacing: 0) {
                    ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                        Text(cell)
                            .font(.system(size: Lab.typeSize(13)))
                            .foregroundStyle(Lab.textMid)
                            .frame(minWidth: 90, alignment: .leading)
                            .padding(.vertical, 9).padding(.horizontal, 11)
                    }
                }
            }
        }
        .overlay(RoundedRectangle(cornerRadius: Lab.radius).strokeBorder(Lab.line, lineWidth: 1))
    }
}

/// A block of rendered Markdown.
enum MarkdownBlock {
    case heading(level: Int, text: String)
    case paragraph(String)
    case bullet([String])
    case ordered([String])
    case code(String)
    case table(header: [String], rows: [[String]])
    case rule
    case figure(caption: String)

    /// A deliberately small parser covering what these models actually emit:
    /// ATX headings, fenced code, pipe tables, bullet and ordered lists, rules,
    /// image references, and paragraphs. Anything unrecognized stays a
    /// paragraph, so no content is ever silently dropped.
    static func parse(_ markdown: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        let lines = markdown.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n")
        var index = 0
        var paragraph: [String] = []

        func flushParagraph() {
            let joined = paragraph.joined(separator: " ").trimmingCharacters(in: .whitespaces)
            if !joined.isEmpty { blocks.append(.paragraph(joined)) }
            paragraph.removeAll()
        }

        while index < lines.count {
            let line = lines[index]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.isEmpty {
                flushParagraph()
                index += 1
                continue
            }

            // Fenced code
            if trimmed.hasPrefix("```") {
                flushParagraph()
                var body: [String] = []
                index += 1
                while index < lines.count,
                      !lines[index].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    body.append(lines[index])
                    index += 1
                }
                index += 1 // closing fence
                blocks.append(.code(body.joined(separator: "\n")))
                continue
            }

            // Horizontal rule
            if trimmed == "---" || trimmed == "***" || trimmed == "___" {
                flushParagraph()
                blocks.append(.rule)
                index += 1
                continue
            }

            // ATX heading
            if trimmed.hasPrefix("#") {
                let hashes = trimmed.prefix { $0 == "#" }.count
                if hashes <= 6, trimmed.dropFirst(hashes).hasPrefix(" ") {
                    flushParagraph()
                    blocks.append(.heading(
                        level: hashes,
                        text: String(trimmed.dropFirst(hashes + 1))
                    ))
                    index += 1
                    continue
                }
            }

            // Standalone image reference
            if trimmed.hasPrefix("!["), let close = trimmed.firstIndex(of: "]") {
                flushParagraph()
                let caption = String(trimmed[trimmed.index(trimmed.startIndex, offsetBy: 2)..<close])
                blocks.append(.figure(caption: caption))
                index += 1
                continue
            }

            // Pipe table: a header row followed by a separator row of dashes.
            if trimmed.hasPrefix("|"), index + 1 < lines.count,
               isTableSeparator(lines[index + 1]) {
                flushParagraph()
                let header = tableCells(trimmed)
                var rows: [[String]] = []
                index += 2
                while index < lines.count,
                      lines[index].trimmingCharacters(in: .whitespaces).hasPrefix("|") {
                    rows.append(tableCells(lines[index].trimmingCharacters(in: .whitespaces)))
                    index += 1
                }
                blocks.append(.table(header: header, rows: rows))
                continue
            }

            // Bullet list
            if trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") {
                flushParagraph()
                var items: [String] = []
                while index < lines.count {
                    let item = lines[index].trimmingCharacters(in: .whitespaces)
                    guard item.hasPrefix("- ") || item.hasPrefix("* ") else { break }
                    items.append(String(item.dropFirst(2)))
                    index += 1
                }
                blocks.append(.bullet(items))
                continue
            }

            // Ordered list
            if let match = orderedPrefixLength(trimmed) {
                flushParagraph()
                var items: [String] = []
                var length: Int? = match
                while index < lines.count, let drop = length {
                    let item = lines[index].trimmingCharacters(in: .whitespaces)
                    items.append(String(item.dropFirst(drop)))
                    index += 1
                    length = index < lines.count
                        ? orderedPrefixLength(lines[index].trimmingCharacters(in: .whitespaces))
                        : nil
                }
                blocks.append(.ordered(items))
                continue
            }

            paragraph.append(trimmed)
            index += 1
        }
        flushParagraph()
        return blocks
    }

    private static func isTableSeparator(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("|") else { return false }
        let stripped = trimmed.filter { !" |-:".contains($0) }
        return stripped.isEmpty && trimmed.contains("-")
    }

    private static func tableCells(_ line: String) -> [String] {
        var body = line
        if body.hasPrefix("|") { body.removeFirst() }
        if body.hasSuffix("|") { body.removeLast() }
        return body.components(separatedBy: "|").map {
            $0.trimmingCharacters(in: .whitespaces)
        }
    }

    /// Length of an ordered-list prefix like `12. `, or nil.
    private static func orderedPrefixLength(_ line: String) -> Int? {
        let digits = line.prefix { $0.isNumber }
        guard !digits.isEmpty else { return nil }
        let rest = line.dropFirst(digits.count)
        guard rest.hasPrefix(". ") else { return nil }
        return digits.count + 2
    }
}
