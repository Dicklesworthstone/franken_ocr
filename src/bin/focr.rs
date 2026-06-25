//! `focr` (short alias) CLI binary — a thin shim over the shared entrypoint in
//! the library.
//!
//! `focr` is the name agents and humans actually type. Both this binary and the
//! long-name `src/main.rs` shim delegate to `franken_ocr::cli_main`
//! (`src/cli.rs`), so the dispatch lives in exactly one place and no source file
//! is shared across `[[bin]]` targets. See AGENTS.md doctrine #9.
#![forbid(unsafe_code)]

fn main() -> std::process::ExitCode {
    franken_ocr::cli_main()
}
