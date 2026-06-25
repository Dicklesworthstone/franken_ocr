//! `franken_ocr` (long-name) CLI binary — a thin shim over the shared
//! entrypoint in the library.
//!
//! The real dispatch lives in `franken_ocr::cli_main` (`src/cli.rs`). This
//! binary and its short alias `src/bin/focr.rs` are byte-for-byte equivalent
//! one-line shims; keeping the logic in the lib means neither `src/main.rs` nor
//! `src/bin/focr.rs` is shared across build targets (no "present in multiple
//! build targets" warning). See AGENTS.md doctrine #9.
#![forbid(unsafe_code)]

fn main() -> std::process::ExitCode {
    franken_ocr::cli_main()
}
