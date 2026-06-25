//! `.focrq` reader + safetensors fallback + the WeightsManifest census
//! (PROPOSED_ARCHITECTURE.md §6.12, §7). **Stub** — implemented by bd-1es.3.
//!
//! The `.focrq` container is a length-prefixed, self-describing blob (magic
//! `b"FOCRQ\0"`, a tensor directory indexing one mmap/payload by byte range).
//! On load it runs a census to catch wrong/stale weights at load time, not as
//! garbage output.

use std::path::Path;

use crate::error::{FocrError, FocrResult};

/// The loaded weight set for one model (every tensor indexed by internal name).
///
/// Stub: the dependency-free byte-range index + dtype directory land with the
/// reader bead.
#[derive(Debug, Default)]
pub struct Weights {
    _private: (),
}

impl Weights {
    /// Load a `.focrq` blob (or fall back to a safetensors shard) from `path`.
    ///
    /// # Errors
    /// Always [`FocrError::NotImplemented`] in the skeleton.
    pub fn load(_path: &Path) -> FocrResult<Self> {
        Err(FocrError::NotImplemented(
            "native_engine::weights::Weights::load — .focrq reader lands in Phase 2 (bd-1es.3)"
                .into(),
        ))
    }
}
