import Foundation

/// Renders a recognition into one self-contained, styled HTML file.
///
/// The saved page keeps the franken-ocr.com identity (the site's `RENDER_CSS`
/// dark-emerald laboratory) and goes further than the browser export does:
/// a provenance header, pipe tables rendered as real `<table>`s, working
/// links, labelled page separators for document runs, and a print stylesheet
/// so paper comes out black-on-white. Everything is inline — the file fetches
/// nothing, ever, which is the same promise the app itself makes.
enum HtmlExport {

    /// What the header and footer say about where the document came from.
    struct Provenance: Sendable {
        let title: String
        let modelName: String
        let characters: Int
        let seconds: Double?
        /// "5 pages · 1 skipped" on a document run, nil for a single page.
        let pageSummary: String?
    }

    /// One stretch of the document: a recognized page, or the honest record
    /// of one that was not.
    enum Section: Sendable {
        case page(number: Int?, markdown: String)
        case skipped(number: Int, reason: String)
    }

    // ── Document assembly ──────────────────────────────────────────────────

    static func document(provenance: Provenance, sections: [Section]) -> String {
        var body: [String] = []
        let multiPage = sections.count > 1
        for section in sections {
            switch section {
            case .page(let number, let markdown):
                if multiPage, let number {
                    body.append(#"<div class="page-marker"><span>Page \#(number)</span></div>"#)
                }
                body.append(renderMarkdown(markdown))
            case .skipped(let number, let reason):
                body.append(
                    #"<div class="page-marker skip"><span>Page \#(number) · skipped — \#(escape(reason))</span></div>"#
                )
            }
        }

        var meta = "\(escape(provenance.modelName)) · \(provenance.characters) characters"
        if let summary = provenance.pageSummary { meta += " · \(escape(summary))" }
        if let seconds = provenance.seconds {
            meta += String(format: " · %.1fs on device", seconds)
        }
        meta += " · \(Self.dateStamp())"

        return """
        <!doctype html>
        <html lang="en"><head><meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>\(escape(provenance.title))</title>
        <style>\(css)</style></head><body>
        <main>
        <header class="doc-head">
          <div class="brand">franken_ocr · reads_locally</div>
          <div class="doc-title">\(escape(provenance.title))</div>
          <div class="doc-meta">\(meta)</div>
        </header>
        \(body.joined(separator: "\n"))
        </main>
        <footer class="credit">
          Recognized entirely on-device by <a href="https://franken-ocr.com">franken_ocr</a> — no cloud, no upload.<br>
          If you like this free app, please show your appreciation by trying out my paid skills site at
          <a href="https://jeffreys-skills.md">JeffreysSkills.md</a>.
        </footer>
        </body></html>
        """
    }

    private static func dateStamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: Date())
    }

    // ── Markdown → HTML ────────────────────────────────────────────────────

    /// Markdown rendered through the same `MarkdownBlock` parser the on-screen
    /// view uses, with embedded HTML `<table>`s (GOT's structured mode emits
    /// them) split out first and passed through the allowlist — exactly the
    /// split the site's `renderMarkdown` does, so a table is never mangled by
    /// the paragraph pass and the Markdown is never trusted as HTML.
    static func renderMarkdown(_ source: String) -> String {
        var parts: [String] = []
        var cursor = source.startIndex
        let tableRe = /<table\b.*?<\/table\s*>/.ignoresCase().dotMatchesNewlines()
        for match in source.matches(of: tableRe) {
            parts.append(renderBlocks(String(source[cursor..<match.range.lowerBound])))
            parts.append(#"<div class="tbl">\#(sanitizeEmbeddedTable(String(match.0)))</div>"#)
            cursor = match.range.upperBound
        }
        parts.append(renderBlocks(String(source[cursor...])))
        return parts.filter { !$0.isEmpty }.joined(separator: "\n")
    }

    private static func renderBlocks(_ markdown: String) -> String {
        guard !markdown.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return "" }
        return MarkdownBlock.parse(markdown).map { block in
            switch block {
            case .heading(let level, let text):
                return "<h\(level)>\(renderInline(text))</h\(level)>"
            case .paragraph(let text):
                return "<p>\(renderInline(text))</p>"
            case .bullet(let items):
                return "<ul>" + items.map { "<li>\(renderInline($0))</li>" }.joined() + "</ul>"
            case .ordered(let items):
                return "<ol>" + items.map { "<li>\(renderInline($0))</li>" }.joined() + "</ol>"
            case .code(let text):
                return "<pre>\(escape(text))</pre>"
            case .table(let header, let rows):
                let head = header.map { "<th>\(renderInline($0))</th>" }.joined()
                let body = rows.map { row in
                    "<tr>" + row.map { "<td>\(renderInline($0))</td>" }.joined() + "</tr>"
                }.joined()
                return #"<div class="tbl"><table><thead><tr>\#(head)</tr></thead><tbody>\#(body)</tbody></table></div>"#
            case .rule:
                return "<hr>"
            case .figure(let caption):
                let label = caption.isEmpty ? "figure (not extracted)" : escape(caption)
                return #"<p><span class="figure">\#(label)</span></p>"#
            }
        }
        .joined(separator: "\n")
    }

    /// Inline emphasis over escaped text. Real links survive as `<a>` when the
    /// target is http(s) — an improvement over the browser export, which drops
    /// them — and image references become the same honest placeholder the
    /// on-screen renderer draws.
    private static func renderInline(_ text: String) -> String {
        var t = escape(text)
        t = t.replacing(
            /!\[[^\]]*\]\([^)]*\)/,
            with: #"<span class="figure">figure (not extracted)</span>"#
        )
        t = t.replacing(/\[([^\]]*)\]\(([^)\s]*)\)/) { match in
            let label = String(match.1)
            let url = String(match.2)
            guard url.hasPrefix("https://") || url.hasPrefix("http://") else { return label }
            return #"<a href="\#(url)">\#(label)</a>"#
        }
        t = t.replacing(/`([^`]+)`/) { "<code>\($0.1)</code>" }
        t = t.replacing(/\*\*([^*]+)\*\*/) { "<strong>\($0.1)</strong>" }
        t = t.replacing(/(^|[^*])\*([^*\n]+)\*/) { "\($0.1)<em>\($0.2)</em>" }
        return t
    }

    // ── Embedded-HTML table allowlist ──────────────────────────────────────

    private static let allowedTags: Set<String> = [
        "table", "thead", "tbody", "tfoot", "tr", "td", "th",
        "caption", "colgroup", "col",
    ]

    /// Keeps only table-structure tags with digit-only colspan/rowspan;
    /// everything else is unwrapped (the text stays, the tag goes) and every
    /// stretch of text between tags is escaped. This is stricter than the
    /// site's sanitizer, which leaves inter-tag text alone.
    private static func sanitizeEmbeddedTable(_ html: String) -> String {
        var out = ""
        var cursor = html.startIndex
        let tagRe = /<(\/?)([a-zA-Z][a-zA-Z0-9]*)((?:[^>"']|"[^"]*"|'[^']*')*)>/
        for match in html.matches(of: tagRe) {
            out += escape(String(html[cursor..<match.range.lowerBound]))
            cursor = match.range.upperBound
            let name = match.2.lowercased()
            guard allowedTags.contains(name) else { continue }
            if !match.1.isEmpty {
                out += "</\(name)>"
                continue
            }
            var attrs = ""
            let attrRe = /(colspan|rowspan)\s*=\s*"?(\d{1,3})/.ignoresCase()
            for attr in String(match.3).matches(of: attrRe) {
                attrs += #" \#(attr.1.lowercased())="\#(attr.2)""#
            }
            out += "<\(name)\(attrs)>"
        }
        out += escape(String(html[cursor...]))
        return out
    }

    static func escape(_ text: String) -> String {
        text
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
    }

    // ── Stylesheet ─────────────────────────────────────────────────────────

    /// One inline stylesheet. Screen keeps the site's dark-emerald identity;
    /// `@media print` re-inks the whole page black-on-white because a printed
    /// transcription is a reading document, not a screenshot of the lab.
    private static let css = """
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 2.4rem 1.3rem 3rem;
      background: #060b09; color: #cbd5e1;
      font: 400 16px/1.72 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      -webkit-font-smoothing: antialiased;
      overflow-wrap: break-word;
    }
    main { max-width: 46rem; margin: 0 auto; }
    .doc-head { border-bottom: 1px solid rgba(52,211,153,0.26); padding-bottom: 1rem; margin-bottom: 1.7rem; }
    .brand {
      font: 800 0.68rem/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      letter-spacing: 0.22em; text-transform: uppercase; color: #10b981;
    }
    .doc-title { margin: 0.5rem 0 0.35rem; font-size: 1.45rem; font-weight: 800; letter-spacing: -0.02em; color: #e8eef2; }
    .doc-meta { color: #748496; font: 400 0.78rem/1.6 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    h1, h2, h3, h4, h5, h6 { color: #e8eef2; font-weight: 800; letter-spacing: -0.02em; line-height: 1.22; margin: 1.6rem 0 0.6rem; }
    h1 { font-size: 1.5rem; } h2 { font-size: 1.25rem; } h3 { font-size: 1.08rem; }
    h4, h5, h6 { font-size: 0.98rem; }
    p { margin: 0 0 0.9rem; }
    ul, ol { margin: 0 0 0.95rem; padding-left: 1.3rem; }
    li { margin-bottom: 0.3rem; }
    strong { color: #e8eef2; font-weight: 700; }
    em { color: #cbd5e1; }
    a { color: #34d399; text-decoration-color: rgba(52,211,153,0.4); }
    hr { border: 0; border-top: 1px solid rgba(255,255,255,0.09); margin: 1.5rem 0; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    code { font-size: 0.86em; background: rgba(2,6,5,0.66); border: 1px solid rgba(255,255,255,0.07); border-radius: 5px; padding: 0.08em 0.34em; }
    pre { background: rgba(2,6,5,0.66); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 0.9rem 1rem; overflow-x: auto; font-size: 0.8rem; line-height: 1.6; }
    .tbl { overflow-x: auto; margin: 0 0 1.1rem; }
    table { border-collapse: collapse; width: 100%; min-width: 380px; font-size: 0.86rem; }
    th, td { border: 1px solid rgba(255,255,255,0.11); padding: 0.5rem 0.65rem; text-align: left; vertical-align: top; }
    th { color: #34d399; font-weight: 700; background: rgba(16,185,129,0.05); }
    .figure { display: inline-block; border: 1px dashed rgba(255,255,255,0.16); border-radius: 8px; padding: 0.7rem 1rem; color: #748496; font-size: 0.82rem; font-style: italic; }
    .page-marker { display: flex; align-items: center; gap: 0.75rem; margin: 2rem 0 1.2rem; }
    .page-marker::before, .page-marker::after { content: ""; flex: 1; border-top: 1px solid rgba(255,255,255,0.09); }
    .page-marker span {
      font: 800 0.66rem/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      letter-spacing: 0.18em; text-transform: uppercase; color: #10b981;
      border: 1px solid rgba(52,211,153,0.35); border-radius: 999px; padding: 0.32rem 0.7rem;
      white-space: nowrap; max-width: 80%; overflow: hidden; text-overflow: ellipsis;
    }
    .page-marker.skip span { color: #f87171; border-color: rgba(248,113,113,0.4); }
    .credit { max-width: 46rem; margin: 2.6rem auto 0; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.09); color: #748496; font-size: 0.78rem; line-height: 1.7; }
    .credit a { color: #34d399; }
    @media print {
      :root { color-scheme: light; }
      body { background: #fff; color: #111827; padding: 0; }
      .doc-head { border-color: #d1dbd5; }
      .brand, .page-marker span { color: #047857; }
      .doc-title, h1, h2, h3, h4, h5, h6, strong { color: #0f1a15; }
      .doc-meta, .credit, .figure { color: #5b6b63; }
      a { color: #047857; }
      code, pre { background: #f3f6f4; border-color: #d9e2dd; color: #1f2937; }
      th { color: #047857; background: #ecfdf5; }
      th, td { border-color: #cfdad3; }
      hr, .page-marker::before, .page-marker::after { border-color: #d1dbd5; }
      .page-marker span { border-color: #a7cbb9; }
      .page-marker.skip span { color: #b91c1c; border-color: #fca5a5; }
      .figure { border-color: #cfdad3; }
    }
    """
}
