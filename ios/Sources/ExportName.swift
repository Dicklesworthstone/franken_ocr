import Foundation
/// An export's file name can come from another app (a shared item's
/// suggested name or a picked file's name). Joined onto the export folder
/// unchecked, a "/" or ".." in it could point the write outside that
/// folder, so only one plain file name is kept.
func safeExportFilename(_ name: String) -> String {
    let flat = String(name.map { $0 == "/" || $0 == "\0" ? "-" : $0 })
        .trimmingCharacters(in: .whitespacesAndNewlines)
    return flat.isEmpty || flat == "." || flat == ".." ? "export" : flat
}
