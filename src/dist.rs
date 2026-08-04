//! Model artifact distribution — `focr pull` + first-run auto-download.
//!
//! `focr` ships without the 6.67 GB weights; it fetches compatible model packages
//! on demand. The committed Unlimited-OCR entry uses the conservative exact
//! recipe required by the runtime. A small
//! JSON [`Manifest`] lists mirror URLs (GitHub Releases + Hugging Face) and the
//! sha256 of every byte; the downloader verifies each part AND the reassembled
//! whole, then installs into `~/.cache/franken_ocr/models/` (already a model
//! search dir), so once cached, INFERENCE is fully offline.
//!
//! HTTP is asupersync's native, capability-gated stack over rustls + Mozilla
//! webpki roots (feature `tls-webpki-roots`) — no `reqwest`/`ureq`/`hyper`.
//! Redirects (GitHub 302 → S3, HF → CDN) are followed automatically; the body
//! is streamed frame-by-frame so a 2 GB part never sits in RAM.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use asupersync::Cx;
use asupersync::bytes::Buf;
use asupersync::http::h1::HttpError;
use asupersync::http::{Body, Client, ClientError, Method};
use asupersync::runtime::RuntimeBuilder;

use crate::error::{FocrError, FocrResult};

/// Exact recipe required by this runtime for the default Unlimited-OCR artifact.
pub const UNLIMITED_OCR_REQUIRED_RECIPE: &str = crate::quant::convert::UNLIMITED_OCR_INT8_RECIPE_ID;

/// Release that owns the immutable, hash-pinned Unlimited-OCR model artifact.
///
/// This is deliberately independent of the `focr` binary version: patch
/// releases can keep consuming the same verified 4.16 GB model bytes without
/// renaming or republishing them.
pub(crate) const UNLIMITED_OCR_ARTIFACT_VERSION: &str = "0.7.0";

/// Recipe carried by the historical full-int8 artifact. It is retained for
/// provenance and fail-closed compatibility tests, but is not compatible with
/// the conservative default runtime.
pub const UNLIMITED_OCR_LEGACY_FULL_INT8_RECIPE: &str =
    "unlimited-ocr-full-int8-attn-int8-lmhead-int8-v1";
/// Exact recipe for the published GOT-OCR2 int8 artifact.
pub const GOT_OCR2_INT8_RECIPE: &str = "got-ocr2-decoder-int8-lmhead-omitted-tied-v1";
/// Exact recipe for the published SmolVLM2 int8 artifact.
pub const SMOLVLM2_INT8_RECIPE: &str = "smolvlm2-decoder-int8-lmhead-bf16-v1";
/// Exact recipe for the published OneChart int8 artifact.
pub const ONECHART_INT8_RECIPE: &str = "onechart-decoder-int8-lmhead-omitted-tied-v1";
/// Exact recipe for the published TrOMR int8 artifact.
pub const TROMR_INT8_RECIPE: &str = "tromr-decoder-int8-v1";
/// Exact recipe for the published TrOMR f32 reference artifact.
pub const TROMR_F32_RECIPE: &str = "tromr-f32-v1";

const MAX_MANIFEST_BYTES: usize = 1024 * 1024;
const MAX_MODELS: usize = 64;
const MAX_QUANTS_PER_MODEL: usize = 8;
const MAX_SIDECARS_PER_MODEL: usize = 32;
const MAX_PARTS_PER_FILE: usize = 64;
const MAX_URLS_PER_PART: usize = 8;
const MAX_ARTIFACT_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const MAX_PART_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_NAME_BYTES: usize = 255;
const MAX_RECIPE_BYTES: usize = 255;
const MAX_URL_BYTES: usize = 4096;
const HASH_BUFFER_BYTES: usize = 64 * 1024;

static STAGING_NONCE: AtomicU64 = AtomicU64::new(0);

/// The default quant tag requested by `focr pull`. A matching tag is not enough:
/// [`validate_quant_compatibility`] also requires the exact runtime recipe.
pub const DEFAULT_QUANT: &str = "int8";

/// Environment override for the manifest source (a local path or an HTTPS URL).
/// Takes precedence over [`DEFAULT_MANIFEST_SOURCE`].
pub const MANIFEST_URL_ENV: &str = "FOCR_MANIFEST_URL";

/// Reserved source identifier for the release-bound manifest embedded in this
/// binary. Artifact URLs and hashes therefore cannot change underneath an
/// installed `focr`; a custom local/HTTPS manifest remains an explicit opt-in.
pub const DEFAULT_MANIFEST_SOURCE: &str = "builtin:models/manifest-v2.json";

/// The supported manifest schema version. Version 2 makes per-quant `recipe`
/// metadata mandatory; both v1 and future layouts are rejected exactly.
pub const MANIFEST_SCHEMA_VERSION: u32 = 2;

/// The committed repo manifest, embedded at build time. Both `focr models` and
/// the default `focr pull` consume these exact release-bound bytes; only an
/// explicit `--manifest` or [`MANIFEST_URL_ENV`] override selects another
/// source.
pub const BUILTIN_MANIFEST_JSON: &str = include_str!("../models/manifest-v2.json");

/// Parse the embedded repo manifest (see [`BUILTIN_MANIFEST_JSON`]).
///
/// # Errors
/// [`FocrError::FormatMismatch`] if the committed manifest is malformed or its
/// schema differs from this binary — a build-time file, so a failure here is a
/// packaging bug, and the schema round-trip test catches it.
pub fn builtin_manifest() -> FocrResult<Manifest> {
    parse_manifest(BUILTIN_MANIFEST_JSON.as_bytes())
}

/// The download manifest: every artifact, its mirrors, and its sha256s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Manifest layout version (see [`MANIFEST_SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Model short name (e.g. `"unlimited-ocr"`).
    pub model: String,
    /// License notice that must travel with the redistributed weights.
    #[serde(default)]
    pub license_notice: String,
    /// Per-quant artifacts, keyed by quant tag (`"int8"`, …). Describes the
    /// primary [`model`](Self::model) (default `unlimited-ocr`).
    pub quants: BTreeMap<String, QuantEntry>,
    /// The tokenizer sidecar for the primary model, installed beside the `.focrq`.
    pub tokenizer: RemoteFile,
    /// Additional models, keyed by model id (e.g. `"got-ocr2"`), selectable via
    /// `focr pull <model>`. The primary model above is NOT duplicated here.
    #[serde(default)]
    pub models: BTreeMap<String, ModelEntry>,
}

/// The artifacts for one non-primary model (its quants + tokenizer sidecar).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// License notice that must travel with this model's redistributed weights.
    #[serde(default)]
    pub license_notice: String,
    /// Per-quant artifacts, keyed by quant tag (`"int8"`, …).
    pub quants: BTreeMap<String, QuantEntry>,
    /// The tokenizer sidecar (e.g. `qwen.tiktoken`), installed beside the `.focrq`.
    pub tokenizer: RemoteFile,
    /// Additional runtime-required sidecars beyond [`tokenizer`](Self::tokenizer)
    /// — e.g. OneChart's `merges.txt` + `added_tokens.json` beside its
    /// `vocab.json`, or TrOMR's remaining three tokenizer tables. Installed
    /// beside the `.focrq` like the tokenizer.
    #[serde(default)]
    pub sidecars: Vec<RemoteFile>,
}

/// The artifacts for one quant level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantEntry {
    /// Stable converter/runtime recipe identifier. For Unlimited-OCR this must
    /// exactly equal [`UNLIMITED_OCR_REQUIRED_RECIPE`] before any artifact part
    /// is requested.
    pub recipe: String,
    /// The quantized weights blob.
    pub focrq: RemoteFile,
}

/// A logical file split into one or more ordered, sha256-pinned parts. The
/// concatenation of the parts IS the file (a GitHub-friendly split that HF can
/// mirror part-for-part); a single-part file is the unsplit case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteFile {
    /// Install filename in the cache dir (e.g. `unlimited-ocr.int8.focrq`).
    pub filename: String,
    /// Total size of the reassembled file, in bytes.
    pub size: u64,
    /// Lowercase-hex sha256 of the reassembled file.
    pub sha256: String,
    /// Ordered parts; concatenation = the file.
    pub parts: Vec<RemotePart>,
}

/// One sha256-pinned part with its mirror URLs (tried in order).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemotePart {
    /// Size of this part, in bytes.
    pub size: u64,
    /// Lowercase-hex sha256 of this part.
    pub sha256: String,
    /// Mirror URLs for this part, tried in order until one verifies.
    pub urls: Vec<String>,
}

type ModelSelection<'a> = (
    &'a str,
    &'a BTreeMap<String, QuantEntry>,
    &'a RemoteFile,
    &'a [RemoteFile],
    Option<&'a str>,
    &'a str,
);

/// Lowercase-hex encode a 32-byte digest.
fn hex32(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap_or('0'));
        s.push(char::from_digit((b & 0xf) as u32, 16).unwrap_or('0'));
    }
    s
}

/// True iff `expected` (lowercase-hex sha256) equals the digest of `data`.
#[cfg(test)]
fn sha256_hex_matches(data: &[u8], expected: &str) -> bool {
    let mut h = Sha256::new();
    h.update(data);
    let got: [u8; 32] = h.finalize().into();
    hex32(&got) == expected.trim().to_ascii_lowercase()
}

/// The per-user cache root for franken_ocr artifacts, resolved per platform:
/// `%LOCALAPPDATA%\franken_ocr` on Windows (falling back to
/// `%USERPROFILE%\.cache\franken_ocr`), and `$HOME/.cache/franken_ocr`
/// everywhere else. Returns `None` only when the platform's home/appdata
/// environment is entirely unset.
#[must_use]
pub fn cache_root() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            return Some(PathBuf::from(local).join("franken_ocr"));
        }
        if let Some(profile) = std::env::var_os("USERPROFILE") {
            return Some(PathBuf::from(profile).join(".cache").join("franken_ocr"));
        }
        None
    }
    #[cfg(not(windows))]
    {
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache").join("franken_ocr"))
    }
}

/// The cache directory new artifacts install into: `%LOCALAPPDATA%\franken_ocr\models`
/// on Windows, `~/.cache/franken_ocr/models` elsewhere (the first user-cache entry
/// of the model search path).
pub fn cache_models_dir() -> FocrResult<PathBuf> {
    cache_root().map(|root| root.join("models")).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "cannot resolve a user cache directory (set HOME, or LOCALAPPDATA/USERPROFILE on Windows)"
        ))
    })
}

/// Is `source` an HTTP(S)-shaped URL (vs. a local filesystem path)? Plain HTTP
/// is recognized here so validation can reject it explicitly.
fn is_url(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://")
}

/// Resolve the manifest source string: explicit `arg`, else [`MANIFEST_URL_ENV`],
/// else [`DEFAULT_MANIFEST_SOURCE`].
pub fn resolve_manifest_source(arg: Option<&str>) -> String {
    if let Some(a) = arg {
        return a.to_string();
    }
    if let Ok(env) = std::env::var(MANIFEST_URL_ENV)
        && !env.trim().is_empty()
    {
        return env;
    }
    DEFAULT_MANIFEST_SOURCE.to_string()
}

/// Parse + validate a manifest from raw JSON bytes.
pub fn parse_manifest(bytes: &[u8]) -> FocrResult<Manifest> {
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err(FocrError::FormatMismatch(format!(
            "manifest is {} bytes; limit is {MAX_MANIFEST_BYTES} bytes",
            bytes.len()
        )));
    }
    let manifest: Manifest = serde_json::from_slice(bytes)
        .map_err(|e| FocrError::FormatMismatch(format!("manifest JSON parse: {e}")))?;
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        return Err(FocrError::FormatMismatch(format!(
            "manifest schema_version {} is unsupported; this binary requires exactly {}",
            manifest.schema_version, MANIFEST_SCHEMA_VERSION
        )));
    }
    validate_manifest(&manifest)?;
    Ok(manifest)
}

fn valid_manifest_atom(value: &str, max_bytes: usize) -> bool {
    !value.is_empty()
        && value.len() <= max_bytes
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

fn validate_manifest_atom(value: &str, what: &str, max_bytes: usize) -> FocrResult<()> {
    if valid_manifest_atom(value, max_bytes) {
        Ok(())
    } else {
        Err(FocrError::FormatMismatch(format!(
            "manifest {what} {value:?} must be 1..={max_bytes} bytes of ASCII letters, digits, '.', '_', or '-'"
        )))
    }
}

fn is_windows_device_name(filename: &str) -> bool {
    let stem = filename.split('.').next().unwrap_or(filename);
    let upper = stem.to_ascii_uppercase();
    matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (upper.len() == 4
            && (upper.starts_with("COM") || upper.starts_with("LPT"))
            && matches!(upper.as_bytes()[3], b'1'..=b'9'))
}

fn validate_filename(filename: &str, what: &str) -> FocrResult<()> {
    validate_manifest_atom(filename, what, MAX_NAME_BYTES)?;
    let mut components = Path::new(filename).components();
    if matches!(components.next(), Some(Component::Normal(_)))
        && components.next().is_none()
        && !filename.ends_with('.')
        && !filename.starts_with(".focr-")
        && !is_windows_device_name(filename)
    {
        Ok(())
    } else {
        Err(FocrError::FormatMismatch(format!(
            "manifest {what} {filename:?} must be one portable, non-reserved filename component"
        )))
    }
}

fn validate_hash(hash: &str, what: &str) -> FocrResult<()> {
    if hash.len() == 64
        && hash
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f'))
    {
        Ok(())
    } else {
        Err(FocrError::FormatMismatch(format!(
            "manifest {what} sha256 must be exactly 64 lowercase hex characters"
        )))
    }
}

fn validate_https_url(url: &str, what: &str) -> FocrResult<()> {
    let authority = url
        .strip_prefix("https://")
        .and_then(|rest| rest.split(['/', '?', '#']).next());
    if url.len() <= MAX_URL_BYTES
        && authority.is_some_and(|value| !value.is_empty())
        && !url
            .bytes()
            .any(|b| b.is_ascii_control() || b.is_ascii_whitespace())
    {
        Ok(())
    } else {
        Err(FocrError::FormatMismatch(format!(
            "manifest {what} URL must be a non-empty HTTPS URL of at most {MAX_URL_BYTES} bytes"
        )))
    }
}

fn validate_remote_file(file: &RemoteFile, what: &str) -> FocrResult<()> {
    validate_filename(&file.filename, &format!("{what} filename"))?;
    if file.size == 0 || file.size > MAX_ARTIFACT_BYTES {
        return Err(FocrError::FormatMismatch(format!(
            "manifest {what} size {} is outside 1..={MAX_ARTIFACT_BYTES}",
            file.size
        )));
    }
    validate_hash(&file.sha256, what)?;
    if file.parts.is_empty() || file.parts.len() > MAX_PARTS_PER_FILE {
        return Err(FocrError::FormatMismatch(format!(
            "manifest {what} must contain 1..={MAX_PARTS_PER_FILE} parts"
        )));
    }
    let mut total = 0u64;
    for (index, part) in file.parts.iter().enumerate() {
        let part_what = format!("{what} part {index}");
        if part.size == 0 || part.size > MAX_PART_BYTES {
            return Err(FocrError::FormatMismatch(format!(
                "manifest {part_what} size {} is outside 1..={MAX_PART_BYTES}",
                part.size
            )));
        }
        total = total.checked_add(part.size).ok_or_else(|| {
            FocrError::FormatMismatch(format!("manifest {what} part-size sum overflows u64"))
        })?;
        validate_hash(&part.sha256, &part_what)?;
        if part.urls.is_empty() || part.urls.len() > MAX_URLS_PER_PART {
            return Err(FocrError::FormatMismatch(format!(
                "manifest {part_what} must contain 1..={MAX_URLS_PER_PART} mirror URLs"
            )));
        }
        for (url_index, url) in part.urls.iter().enumerate() {
            validate_https_url(url, &format!("{part_what} mirror {url_index}"))?;
        }
    }
    if total != file.size {
        return Err(FocrError::FormatMismatch(format!(
            "manifest {what} part sizes sum to {total}, expected {}",
            file.size
        )));
    }
    Ok(())
}

fn validate_quant_entries(
    model_id: &str,
    quants: &BTreeMap<String, QuantEntry>,
    filenames: &mut BTreeSet<String>,
) -> FocrResult<()> {
    if quants.is_empty() || quants.len() > MAX_QUANTS_PER_MODEL {
        return Err(FocrError::FormatMismatch(format!(
            "manifest model {model_id:?} must contain 1..={MAX_QUANTS_PER_MODEL} quants"
        )));
    }
    for (tag, entry) in quants {
        validate_manifest_atom(tag, &format!("quant tag for {model_id}"), MAX_NAME_BYTES)?;
        validate_manifest_atom(
            &entry.recipe,
            &format!("recipe for {model_id}/{tag}"),
            MAX_RECIPE_BYTES,
        )?;
        validate_remote_file(&entry.focrq, &format!("{model_id}/{tag} artifact"))?;
        if !filenames.insert(entry.focrq.filename.to_ascii_lowercase()) {
            return Err(FocrError::FormatMismatch(format!(
                "manifest model {model_id:?} installs duplicate filename {:?}",
                entry.focrq.filename
            )));
        }
    }
    Ok(())
}

fn validate_manifest(manifest: &Manifest) -> FocrResult<()> {
    validate_filename(&manifest.model, "primary model id")?;
    if manifest.models.len() > MAX_MODELS {
        return Err(FocrError::FormatMismatch(format!(
            "manifest has {} secondary models; limit is {MAX_MODELS}",
            manifest.models.len()
        )));
    }

    let mut primary_names = BTreeSet::new();
    validate_remote_file(&manifest.tokenizer, "primary tokenizer")?;
    primary_names.insert(manifest.tokenizer.filename.to_ascii_lowercase());
    validate_quant_entries(&manifest.model, &manifest.quants, &mut primary_names)?;

    let mut model_ids = BTreeSet::from([manifest.model.to_ascii_lowercase()]);
    for (model_id, entry) in &manifest.models {
        validate_filename(model_id, "secondary model id")?;
        if !model_ids.insert(model_id.to_ascii_lowercase()) {
            return Err(FocrError::FormatMismatch(format!(
                "manifest model id {model_id:?} is duplicated under portable case-folding"
            )));
        }
        if entry.sidecars.len() > MAX_SIDECARS_PER_MODEL {
            return Err(FocrError::FormatMismatch(format!(
                "manifest model {model_id:?} has {} sidecars; limit is {MAX_SIDECARS_PER_MODEL}",
                entry.sidecars.len()
            )));
        }
        let mut names = BTreeSet::new();
        validate_remote_file(&entry.tokenizer, &format!("{model_id} tokenizer"))?;
        names.insert(entry.tokenizer.filename.to_ascii_lowercase());
        validate_quant_entries(model_id, &entry.quants, &mut names)?;
        for (index, sidecar) in entry.sidecars.iter().enumerate() {
            validate_remote_file(sidecar, &format!("{model_id} sidecar {index}"))?;
            if !names.insert(sidecar.filename.to_ascii_lowercase()) {
                return Err(FocrError::FormatMismatch(format!(
                    "manifest model {model_id:?} installs duplicate filename {:?}",
                    sidecar.filename
                )));
            }
        }
    }
    Ok(())
}

/// Exact recipe required for one published model/quant pair.
#[must_use]
pub fn required_quant_recipe(model_id: &str, quant: &str) -> Option<&'static str> {
    match (model_id, quant) {
        ("unlimited-ocr", "int8") => Some(UNLIMITED_OCR_REQUIRED_RECIPE),
        ("got-ocr2", "int8") => Some(GOT_OCR2_INT8_RECIPE),
        ("smolvlm2", "int8") => Some(SMOLVLM2_INT8_RECIPE),
        ("onechart", "int8") => Some(ONECHART_INT8_RECIPE),
        ("tromr", "int8") => Some(TROMR_INT8_RECIPE),
        ("tromr", "f32") => Some(TROMR_F32_RECIPE),
        _ => None,
    }
}

/// Whether a manifest quant exactly matches this runtime's model contract.
#[must_use]
pub fn quant_recipe_is_compatible(model_id: &str, quant: &str, recipe: &str) -> bool {
    required_quant_recipe(model_id, quant) == Some(recipe)
}

fn validate_quant_compatibility(model_id: &str, quant: &str, entry: &QuantEntry) -> FocrResult<()> {
    let Some(required) = required_quant_recipe(model_id, quant) else {
        return Err(FocrError::FormatMismatch(format!(
            "manifest artifact {model_id}/{quant} has no recipe contract in this runtime"
        )));
    };
    if quant_recipe_is_compatible(model_id, quant, &entry.recipe) {
        return Ok(());
    }
    Err(FocrError::FormatMismatch(format!(
        "manifest artifact {model_id}/{quant} declares recipe {:?}, but this runtime requires \
         {required:?}; the artifact is blocked before download; use the committed manifest or \
         regenerate it with the model's certified converter recipe",
        entry.recipe
    )))
}

// ── HTTP download (asupersync native stack) ─────────────────────────────────

/// Distinct download failures, surfaced with context up the `FocrError` chain.
#[derive(Debug)]
enum DownloadError {
    Request(ClientError),
    Body(HttpError),
    UnexpectedStatus(u16),
    Io(std::io::Error),
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Request(e) => write!(f, "request failed: {e}"),
            Self::Body(e) => write!(f, "body stream failed: {e}"),
            Self::UnexpectedStatus(s) => write!(f, "unexpected HTTP status {s}"),
            Self::Io(e) => write!(f, "io: {e}"),
        }
    }
}

/// Build the streaming client. The streaming body still enforces
/// `max_body_size`, and the runtime-default client leaves it at the 16 MiB codec
/// default — far below a 2 GB part — so we raise it explicitly to 8 GiB.
fn streaming_client() -> Client {
    Client::builder()
        .max_body_size(8 * 1024 * 1024 * 1024)
        .build()
}

/// Stream one URL's body into `sink`, invoking it with each chunk. Returns the
/// total bytes written. Redirects/TLS are handled inside asupersync.
async fn stream_url<S: FnMut(&[u8]) -> Result<(), std::io::Error>>(
    cx: &Cx,
    client: &Client,
    url: &str,
    mut sink: S,
) -> Result<u64, DownloadError> {
    let resp = client
        .request_streaming(cx, Method::Get, url, Vec::new(), Vec::new())
        .await
        .map_err(DownloadError::Request)?;
    if !(200..=299).contains(&resp.head.status) {
        return Err(DownloadError::UnexpectedStatus(resp.head.status));
    }
    let mut total: u64 = 0;
    let mut body = resp.body;
    while let Some(frame) =
        std::future::poll_fn(|task_cx| Pin::new(&mut body).poll_frame(task_cx)).await
    {
        let frame = frame.map_err(DownloadError::Body)?;
        if let Some(mut chunk) = frame.into_data() {
            while chunk.has_remaining() {
                let bytes = chunk.chunk();
                sink(bytes).map_err(DownloadError::Io)?;
                let n = bytes.len();
                total = total.checked_add(n as u64).ok_or_else(|| {
                    DownloadError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "response byte count overflowed u64",
                    ))
                })?;
                chunk.advance(n);
            }
        }
    }
    Ok(total)
}

fn extend_bounded(
    buffer: &mut Vec<u8>,
    chunk: &[u8],
    limit: usize,
    what: &str,
) -> std::io::Result<()> {
    let next = buffer.len().checked_add(chunk.len()).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{what} byte count overflowed usize"),
        )
    })?;
    if next > limit {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{what} exceeds {limit}-byte limit"),
        ));
    }
    buffer.extend_from_slice(chunk);
    Ok(())
}

fn checked_part_response_size(
    received: u64,
    chunk_len: usize,
    declared: u64,
) -> std::io::Result<u64> {
    let next = received.checked_add(chunk_len as u64).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "part response byte count overflowed u64",
        )
    })?;
    if next > declared {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("part response exceeded declared {declared}-byte size"),
        ));
    }
    Ok(next)
}

fn map_manifest_download_error(error: DownloadError) -> FocrError {
    match error {
        DownloadError::Io(error) if error.kind() == std::io::ErrorKind::InvalidData => {
            FocrError::FormatMismatch(format!("manifest fetch: {error}"))
        }
        other => FocrError::Other(anyhow::anyhow!("manifest fetch: {other}")),
    }
}

/// Download a manifest with an independent 1 MiB bound. Model parts use the
/// larger streaming client body allowance, but manifest JSON never does.
async fn fetch_manifest_bytes(cx: &Cx, url: &str) -> FocrResult<Vec<u8>> {
    let client = streaming_client();
    let mut buf = Vec::new();
    stream_url(cx, &client, url, |chunk| {
        extend_bounded(&mut buf, chunk, MAX_MANIFEST_BYTES, "manifest")
    })
    .await
    .map_err(map_manifest_download_error)?;
    Ok(buf)
}

fn read_local_manifest(path: &Path) -> FocrResult<Vec<u8>> {
    let file = std::fs::File::open(path)
        .map_err(|e| FocrError::ModelNotFound(format!("manifest {}: {e}", path.display())))?;
    if file
        .metadata()
        .map_err(|e| FocrError::Other(anyhow::anyhow!("stat manifest {}: {e}", path.display())))?
        .len()
        > MAX_MANIFEST_BYTES as u64
    {
        return Err(FocrError::FormatMismatch(format!(
            "manifest {} exceeds {MAX_MANIFEST_BYTES}-byte limit",
            path.display()
        )));
    }
    let mut bytes = Vec::new();
    file.take((MAX_MANIFEST_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| FocrError::Other(anyhow::anyhow!("read manifest {}: {e}", path.display())))?;
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err(FocrError::FormatMismatch(format!(
            "manifest {} exceeds {MAX_MANIFEST_BYTES}-byte limit",
            path.display()
        )));
    }
    Ok(bytes)
}

/// Download + verify one [`RemoteFile`] into `dest` (a `.tmp` path), then return.
/// Each part is streamed straight into `dest` while a per-part AND a whole-file
/// hasher run; a failed mirror rolls `dest` + the whole-file hash back to the
/// part boundary and tries the next URL. The whole-file sha256 is checked last.
/// Per-part progress goes to stderr (stdout stays clean for any JSON consumer);
/// `quiet` silences it.
async fn download_remote_file(
    cx: &Cx,
    file: &RemoteFile,
    dest: &Path,
    mut out: std::fs::File,
    quiet: bool,
) -> FocrResult<()> {
    let client = streaming_client();
    let mut full = Sha256::new();
    let mut committed: u64 = 0;

    for (i, part) in file.parts.iter().enumerate() {
        let full_checkpoint = full.clone();
        let mut part_ok = false;
        let mut last_err: Option<String> = None;

        for url in &part.urls {
            // Roll back to the part boundary before each attempt.
            out.set_len(committed)
                .and_then(|()| out.seek(SeekFrom::Start(committed)).map(|_| ()))
                .map_err(|e| FocrError::Other(anyhow::anyhow!("seek {}: {e}", dest.display())))?;
            full = full_checkpoint.clone();
            let mut part_hash = Sha256::new();

            if !quiet {
                eprintln!(
                    "  part {}/{} ({:.1} MB) <- {url}",
                    i + 1,
                    file.parts.len(),
                    part.size as f64 / 1.0e6
                );
            }
            let mut received = 0u64;
            let res = stream_url(cx, &client, url, |chunk| {
                let next = checked_part_response_size(received, chunk.len(), part.size)?;
                out.write_all(chunk)?;
                part_hash.update(chunk);
                full.update(chunk);
                received = next;
                Ok(())
            })
            .await;

            match res {
                Ok(n) if n == part.size => {
                    let got: [u8; 32] = part_hash.finalize().into();
                    if hex32(&got) == part.sha256.trim().to_ascii_lowercase() {
                        part_ok = true;
                        committed += part.size;
                        break;
                    }
                    last_err = Some(format!("part {} sha256 mismatch from {url}", i + 1));
                }
                Ok(n) => {
                    last_err = Some(format!("part {} size {n} != expected {}", i + 1, part.size))
                }
                Err(e) => last_err = Some(format!("{e} (from {url})")),
            }
        }

        if !part_ok {
            return Err(FocrError::Other(anyhow::anyhow!(
                "all mirrors failed for part {} of {}: {}",
                i + 1,
                file.filename,
                last_err.unwrap_or_else(|| "no urls".into())
            )));
        }
    }

    out.flush()
        .map_err(|e| FocrError::Other(anyhow::anyhow!("flush {}: {e}", dest.display())))?;
    out.sync_all()
        .map_err(|e| FocrError::Other(anyhow::anyhow!("sync {}: {e}", dest.display())))?;
    let got: [u8; 32] = full.finalize().into();
    if hex32(&got) != file.sha256.trim().to_ascii_lowercase() {
        return Err(FocrError::FormatMismatch(format!(
            "reassembled {} sha256 {} != manifest {}",
            file.filename,
            hex32(&got),
            file.sha256
        )));
    }
    if committed != file.size {
        return Err(FocrError::FormatMismatch(format!(
            "reassembled {} size {committed} != manifest {}",
            file.filename, file.size
        )));
    }
    Ok(())
}

/// Is `path` already a byte-perfect copy of `file` (so the download is skippable)?
fn already_cached(path: &Path, file: &RemoteFile) -> bool {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return false;
    };
    if meta.file_type().is_symlink() || !meta.is_file() || meta.len() != file.size {
        return false;
    }
    // Size matched; confirm with a fixed-size streaming buffer. A cache hit must
    // never allocate another 4+ GB merely to hash an mmap-friendly artifact.
    let Ok(opened) = std::fs::File::open(path) else {
        return false;
    };
    let Ok(got) = sha256_reader(opened) else {
        return false;
    };
    hex32(&got) == file.sha256
}

fn sha256_reader(mut reader: impl Read) -> std::io::Result<[u8; 32]> {
    let mut hash = Sha256::new();
    let mut buffer = [0u8; HASH_BUFFER_BYTES];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(hash.finalize().into())
}

struct InstallLock {
    _file: std::fs::File,
}

fn coordination_key(final_path: &Path) -> FocrResult<String> {
    let filename = final_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            FocrError::FormatMismatch(format!(
                "install path {} has a non-UTF-8 filename",
                final_path.display()
            ))
        })?;
    let mut hash = Sha256::new();
    hash.update(filename.as_bytes());
    let digest: [u8; 32] = hash.finalize().into();
    Ok(hex32(&digest)[..32].to_owned())
}

/// Map a downloader-owned staging pathname back to its advisory lock. Doctor
/// uses this to distinguish a live multi-gigabyte download from an orphan left
/// by a crashed process.
pub(crate) fn pull_lock_path_for_staging(staging_path: &Path) -> Option<PathBuf> {
    let name = staging_path.file_name()?.to_str()?;
    let remainder = name.strip_prefix(".focr-stage-")?;
    let (key, suffix) = remainder.split_once('-')?;
    if key.len() != 32
        || !key.bytes().all(|byte| byte.is_ascii_hexdigit())
        || !suffix.ends_with(".partial")
    {
        return None;
    }
    Some(
        staging_path
            .parent()?
            .join(format!(".focr-pull-{key}.lock")),
    )
}

impl InstallLock {
    fn acquire(final_path: &Path) -> FocrResult<Self> {
        let parent = final_path.parent().ok_or_else(|| {
            FocrError::Other(anyhow::anyhow!(
                "install path {} has no parent directory",
                final_path.display()
            ))
        })?;
        let path = parent.join(format!(".focr-pull-{}.lock", coordination_key(final_path)?));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|error| {
                FocrError::Other(anyhow::anyhow!(
                    "open install lock {}: {error}",
                    path.display()
                ))
            })?;
        let path_metadata = std::fs::symlink_metadata(&path).map_err(|error| {
            FocrError::Other(anyhow::anyhow!(
                "stat install lock {}: {error}",
                path.display()
            ))
        })?;
        let descriptor_metadata = file.metadata().map_err(|error| {
            FocrError::Other(anyhow::anyhow!(
                "stat open install lock {}: {error}",
                path.display()
            ))
        })?;
        if path_metadata.file_type().is_symlink()
            || !path_metadata.is_file()
            || !descriptor_metadata.is_file()
        {
            return Err(FocrError::FormatMismatch(format!(
                "install lock {} must be one regular non-symlink file",
                path.display()
            )));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            if path_metadata.dev() != descriptor_metadata.dev()
                || path_metadata.ino() != descriptor_metadata.ino()
            {
                return Err(FocrError::FormatMismatch(format!(
                    "install lock {} changed while it was opened",
                    path.display()
                )));
            }
        }
        file.try_lock().map_err(|error| match error {
            std::fs::TryLockError::WouldBlock => FocrError::Timeout(format!(
                "another pull is installing {} (lock {}; retry after it finishes)",
                final_path.display(),
                path.display()
            )),
            std::fs::TryLockError::Error(error) => FocrError::Other(anyhow::anyhow!(
                "acquire install lock {}: {error}",
                path.display()
            )),
        })?;
        Ok(Self { _file: file })
    }
}

fn create_staging_file(final_path: &Path) -> FocrResult<(PathBuf, std::fs::File)> {
    let parent = final_path.parent().ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "install path {} has no parent directory",
            final_path.display()
        ))
    })?;
    let key = coordination_key(final_path)?;
    for _ in 0..64 {
        let nonce = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".focr-stage-{key}-{}-{nonce}.partial",
            std::process::id(),
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "create staging file {}: {error}",
                    path.display()
                )));
            }
        }
    }
    Err(FocrError::Other(anyhow::anyhow!(
        "could not allocate a unique staging file for {}",
        final_path.display()
    )))
}

#[cfg(unix)]
fn sync_parent_dir(path: &Path) -> FocrResult<()> {
    let parent = path.parent().ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "installed path {} has no parent",
            path.display()
        ))
    })?;
    std::fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| FocrError::Other(anyhow::anyhow!("sync {}: {error}", parent.display())))
}

#[cfg(not(unix))]
fn sync_parent_dir(_path: &Path) -> FocrResult<()> {
    Ok(())
}

fn ensure_real_directory(path: &Path, what: &str) -> FocrResult<()> {
    std::fs::create_dir_all(path)
        .map_err(|e| FocrError::Other(anyhow::anyhow!("create {what} {}: {e}", path.display())))?;
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|e| FocrError::Other(anyhow::anyhow!("inspect {what} {}: {e}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(FocrError::FormatMismatch(format!(
            "{what} {} must be a real directory, not a symlink or other file type",
            path.display()
        )));
    }
    Ok(())
}

fn validate_install_target_type(path: &Path) -> FocrResult<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err(FocrError::FormatMismatch(format!(
                "install target {} is a symlink; refusing to follow or replace it",
                path.display()
            )))
        }
        Ok(metadata) if metadata.is_dir() => Err(FocrError::FormatMismatch(format!(
            "install target {} is a directory",
            path.display()
        ))),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(FocrError::Other(anyhow::anyhow!(
            "inspect install target {}: {error}",
            path.display()
        ))),
    }
}

fn commit_staging_file(staging_path: &Path, final_path: &Path) -> FocrResult<()> {
    validate_install_target_type(final_path)?;
    std::fs::rename(staging_path, final_path).map_err(|error| {
        FocrError::Other(anyhow::anyhow!(
            "install {} -> {}: {error}",
            staging_path.display(),
            final_path.display()
        ))
    })?;
    sync_parent_dir(final_path)
}

/// The outcome of a [`pull`]: where the model + tokenizer landed.
#[derive(Debug, Clone)]
pub struct PullOutcome {
    /// The installed `.focrq` path (a model search dir resolves it by name).
    pub focrq_path: PathBuf,
    /// The installed `tokenizer.json` path (sibling of the `.focrq`).
    pub tokenizer_path: PathBuf,
    /// Additional installed sidecar paths (siblings of the `.focrq`).
    pub sidecar_paths: Vec<PathBuf>,
    /// Quant level pulled (the manifest's actual tag — see the sole-quant
    /// fallback in [`pull`]).
    pub quant: String,
    /// The pulled model's license notice from the manifest (empty when the
    /// manifest entry carries none; the caller may substitute the primary
    /// model's built-in notice).
    pub license_notice: String,
    /// True iff every artifact was already cached (nothing downloaded).
    pub from_cache: bool,
}

/// Pick the quant entry for a pull. Exact match wins; otherwise, when the
/// model publishes EXACTLY ONE quant, fall back to it — failing a pull over
/// a default flag serves no one when the model's only artifact is
/// unambiguous. The returned tag is the ACTUAL quant
/// (callers report it; `pull` prints a visible note when it differs from
/// the request).
///
/// # Errors
/// [`FocrError::Usage`] when the quant is absent and the fallback is
/// ambiguous (zero or several published quants).
fn select_quant<'m>(
    quants: &'m BTreeMap<String, QuantEntry>,
    requested: &str,
) -> FocrResult<(String, &'m QuantEntry)> {
    match quants.get(requested) {
        Some(entry) => Ok((requested.to_owned(), entry)),
        None if quants.len() == 1 => {
            let (tag, entry) = quants.iter().next().expect("len==1 map has an entry");
            Ok((tag.clone(), entry))
        }
        None => Err(FocrError::Usage(format!(
            "manifest has no quant '{requested}' (available: {})",
            quants.keys().cloned().collect::<Vec<_>>().join(", ")
        ))),
    }
}

/// Fetch (or confirm-cached) the `quant` weights + tokenizer described by the
/// manifest at `manifest_source` (path or URL) into the model cache. `progress`
/// receives human status lines. Network only happens here — once it returns Ok,
/// the model loads offline.
pub fn pull(
    model: Option<&str>,
    quant: &str,
    manifest_source: &str,
    quiet: bool,
    mut progress: impl FnMut(&str),
) -> FocrResult<PullOutcome> {
    let runtime = RuntimeBuilder::new()
        .build()
        .map_err(|e| FocrError::Other(anyhow::anyhow!("runtime build: {e}")))?;

    // 1. Load the release-bound embedded manifest, an explicit local file, or
    // an explicit remote URL. The default never delegates artifact identity to
    // mutable branch state.
    let manifest_bytes = if manifest_source == DEFAULT_MANIFEST_SOURCE {
        progress("using release-bound embedded model manifest");
        BUILTIN_MANIFEST_JSON.as_bytes().to_vec()
    } else if is_url(manifest_source) {
        validate_https_url(manifest_source, "source")?;
        progress(&format!("fetching manifest {manifest_source}"));
        let url = manifest_source.to_string();
        runtime.block_on(async move {
            let cx = Cx::current().ok_or_else(|| {
                FocrError::Other(anyhow::anyhow!("runtime did not install an ambient Cx"))
            })?;
            fetch_manifest_bytes(&cx, &url).await
        })?
    } else {
        read_local_manifest(Path::new(manifest_source))?
    };
    let manifest = parse_manifest(&manifest_bytes)?;

    // Select the model: the primary top-level model (default, and the only one
    // old binaries know) unless `model` names a distinct entry in `models`.
    // Non-primary models install into their own `<cache>/<model-id>/` subdir:
    // sidecar filenames are NOT unique across models (smolvlm2 ships a
    // `tokenizer.json` that would clobber unlimited-ocr's in a flat cache),
    // and the loaders resolve sidecars beside the artifact, so isolation per
    // model is both necessary and sufficient. The primary model stays flat —
    // the layout every released binary already knows.
    static NO_SIDECARS: Vec<RemoteFile> = Vec::new();
    let (model_id, quants, tokenizer, sidecars, subdir, license_notice): ModelSelection<'_> =
        match model {
            None => (
                manifest.model.as_str(),
                &manifest.quants,
                &manifest.tokenizer,
                &NO_SIDECARS,
                None,
                manifest.license_notice.as_str(),
            ),
            Some(m) if m == manifest.model => (
                manifest.model.as_str(),
                &manifest.quants,
                &manifest.tokenizer,
                &NO_SIDECARS,
                None,
                manifest.license_notice.as_str(),
            ),
            Some(m) => {
                let entry = manifest.models.get(m).ok_or_else(|| {
                    let mut avail = vec![manifest.model.clone()];
                    avail.extend(manifest.models.keys().cloned());
                    FocrError::Usage(format!(
                        "manifest has no model '{m}' (available: {})",
                        avail.join(", ")
                    ))
                })?;
                (
                    m,
                    &entry.quants,
                    &entry.tokenizer,
                    &entry.sidecars,
                    Some(m),
                    entry.license_notice.as_str(),
                )
            }
        };

    let (quant_used, quant_entry) = select_quant(quants, quant)?;
    if quant_used != quant {
        progress(&format!(
            "manifest has no quant '{quant}' for this model; using the sole \
             published quant '{quant_used}'"
        ));
    }

    // Compatibility is a manifest contract, not a post-download surprise. The
    // currently committed primary entry intentionally fails here before cache
    // creation, cache hashing, or any multi-GB artifact request.
    validate_quant_compatibility(model_id, &quant_used, quant_entry)?;

    let cache_dir = cache_models_dir()?;
    ensure_real_directory(&cache_dir, "model cache")?;
    let mut dir = cache_dir;
    if let Some(sub) = subdir {
        dir = dir.join(sub);
        ensure_real_directory(&dir, "model cache subdirectory")?;
    }
    let focrq_path = dir.join(&quant_entry.focrq.filename);
    let tokenizer_path = dir.join(&tokenizer.filename);

    // 2. Download each artifact unless already byte-perfect in the cache. Each
    // call rechecks under an exclusive per-file install lock, closing the race
    // between an optimistic cache miss and another process's completed rename.
    let focrq_cached = install_file(
        &runtime,
        &quant_entry.focrq,
        &focrq_path,
        quiet,
        &mut progress,
    )?;
    let tokenizer_cached =
        install_file(&runtime, tokenizer, &tokenizer_path, quiet, &mut progress)?;
    let mut from_cache = focrq_cached && tokenizer_cached;
    let mut sidecar_paths = Vec::with_capacity(sidecars.len());
    for sidecar in sidecars {
        let path = dir.join(&sidecar.filename);
        let cached = install_file(&runtime, sidecar, &path, quiet, &mut progress)?;
        from_cache &= cached;
        sidecar_paths.push(path);
    }

    Ok(PullOutcome {
        focrq_path,
        tokenizer_path,
        sidecar_paths,
        quant: quant_used,
        license_notice: license_notice.to_owned(),
        from_cache,
    })
}

/// Ensure one cache file is byte-perfect. Returns `true` for a cache hit and
/// `false` when this call installed the file. A per-file `create_new` lock plus a
/// unique staging file prevents concurrent writers from sharing an inode.
fn install_file(
    runtime: &asupersync::runtime::Runtime,
    file: &RemoteFile,
    final_path: &Path,
    quiet: bool,
    progress: &mut impl FnMut(&str),
) -> FocrResult<bool> {
    if already_cached(final_path, file) {
        progress(&format!("cached: {}", final_path.display()));
        return Ok(true);
    }
    validate_install_target_type(final_path)?;

    let _lock = InstallLock::acquire(final_path)?;
    if already_cached(final_path, file) {
        progress(&format!("cached: {}", final_path.display()));
        return Ok(true);
    }
    validate_install_target_type(final_path)?;

    let (tmp, staging_file) = create_staging_file(final_path)?;
    let tmp_for_async = tmp.clone();
    let file_owned = file.clone();
    progress(&format!(
        "downloading {} ({:.2} GB, {} part(s))",
        file.filename,
        file.size as f64 / 1.0e9,
        file.parts.len()
    ));
    let download = runtime.block_on(async move {
        let cx = Cx::current().ok_or_else(|| {
            FocrError::Other(anyhow::anyhow!("runtime did not install an ambient Cx"))
        })?;
        download_remote_file(&cx, &file_owned, &tmp_for_async, staging_file, quiet).await
    });
    if let Err(e) = download {
        // Don't leave a half-written `.partial` littering the cache; the next
        // attempt would truncate it anyway, but a failed pull should be tidy.
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    // `rename` within one directory is the commit point: it replaces a regular
    // destination atomically, while a failure leaves the previously verified
    // file in place. Never unlink the working cache entry before this step.
    if let Err(error) = commit_staging_file(&tmp, final_path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    progress(&format!("installed {}", final_path.display()));
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_file(filename: &str) -> RemoteFile {
        RemoteFile {
            filename: filename.into(),
            size: 1,
            sha256: "ab".repeat(32),
            parts: vec![RemotePart {
                size: 1,
                sha256: "ab".repeat(32),
                urls: vec!["https://example.invalid/artifact".into()],
            }],
        }
    }

    fn tiny_model_entry() -> ModelEntry {
        ModelEntry {
            license_notice: "test license".into(),
            quants: BTreeMap::from([(
                "int8".into(),
                QuantEntry {
                    recipe: "test-decoder-int8-v1".into(),
                    focrq: tiny_file("model.focrq"),
                },
            )]),
            tokenizer: tiny_file("tokenizer.json"),
            sidecars: Vec::new(),
        }
    }

    fn tiny_manifest() -> Manifest {
        Manifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            model: "unlimited-ocr".into(),
            license_notice: "test license".into(),
            quants: BTreeMap::from([(
                DEFAULT_QUANT.into(),
                QuantEntry {
                    recipe: UNLIMITED_OCR_REQUIRED_RECIPE.into(),
                    focrq: tiny_file("model.focrq"),
                },
            )]),
            tokenizer: tiny_file("tokenizer.json"),
            models: BTreeMap::new(),
        }
    }

    fn parse_serialized(manifest: &Manifest) -> FocrResult<Manifest> {
        parse_manifest(&serde_json::to_vec(manifest).expect("serialize test manifest"))
    }

    #[test]
    fn hex32_is_lowercase_64() {
        let mut h = Sha256::new();
        h.update(b"franken_ocr");
        let d: [u8; 32] = h.finalize().into();
        let hex = hex32(&d);
        assert_eq!(hex.len(), 64);
        assert!(
            hex.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn sha256_matches_known_vector() {
        // sha256("") = e3b0c442...
        assert!(sha256_hex_matches(
            b"",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ));
        assert!(sha256_hex_matches(
            b"abc",
            "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD" // uppercase ok
        ));
        assert!(!sha256_hex_matches(b"abc", "00".repeat(32).as_str()));
    }

    #[test]
    fn manifest_round_trips_and_rejects_future_schema() {
        let m = Manifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            model: "unlimited-ocr".into(),
            license_notice: "MIT (Baidu)".into(),
            quants: BTreeMap::from([(
                "int8".to_string(),
                QuantEntry {
                    recipe: UNLIMITED_OCR_REQUIRED_RECIPE.into(),
                    focrq: RemoteFile {
                        filename: "unlimited-ocr.int8.focrq".into(),
                        size: 4157448783,
                        sha256: "ab".repeat(32),
                        parts: vec![RemotePart {
                            size: 4157448783,
                            sha256: "ab".repeat(32),
                            urls: vec!["https://example/part0".into()],
                        }],
                    },
                },
            )]),
            tokenizer: RemoteFile {
                filename: "tokenizer.json".into(),
                size: 9979544,
                sha256: "cd".repeat(32),
                parts: vec![RemotePart {
                    size: 9979544,
                    sha256: "cd".repeat(32),
                    urls: vec!["https://example/tok".into()],
                }],
            },
            models: BTreeMap::from([(
                "got-ocr2".to_string(),
                ModelEntry {
                    license_notice: "Apache-2.0 (GOT-OCR2.0)".into(),
                    quants: BTreeMap::from([(
                        "int8".to_string(),
                        QuantEntry {
                            recipe: GOT_OCR2_INT8_RECIPE.into(),
                            focrq: RemoteFile {
                                filename: "got-ocr2.int8.focrq".into(),
                                size: 813877416,
                                sha256: "ef".repeat(32),
                                parts: vec![RemotePart {
                                    size: 813877416,
                                    sha256: "ef".repeat(32),
                                    urls: vec!["https://example/got".into()],
                                }],
                            },
                        },
                    )]),
                    tokenizer: RemoteFile {
                        filename: "qwen.tiktoken".into(),
                        size: 2561218,
                        sha256: "12".repeat(32),
                        parts: vec![RemotePart {
                            size: 2561218,
                            sha256: "12".repeat(32),
                            urls: vec!["https://example/qwen".into()],
                        }],
                    },
                    sidecars: Vec::new(),
                },
            )]),
        };
        let json = serde_json::to_vec(&m).expect("serialize");
        let back = parse_manifest(&json).expect("parse");
        assert_eq!(back.model, "unlimited-ocr");
        assert_eq!(back.quants["int8"].focrq.size, 4157448783);
        assert_eq!(back.quants["int8"].recipe, UNLIMITED_OCR_REQUIRED_RECIPE);
        // The secondary model resolves by id with its own tokenizer filename.
        assert_eq!(back.models["got-ocr2"].tokenizer.filename, "qwen.tiktoken");
        assert_eq!(back.models["got-ocr2"].quants["int8"].focrq.size, 813877416);

        // A newer schema is rejected loudly.
        let future = br#"{"schema_version":999,"model":"x","quants":{},"tokenizer":{"filename":"t","size":0,"sha256":"","parts":[]}}"#;
        assert!(matches!(
            parse_manifest(future),
            Err(FocrError::FormatMismatch(_))
        ));
    }

    #[test]
    fn manifest_schema_must_match_exactly() {
        for schema_version in [0, MANIFEST_SCHEMA_VERSION - 1, MANIFEST_SCHEMA_VERSION + 1] {
            let mut manifest = tiny_manifest();
            manifest.schema_version = schema_version;
            let error = parse_serialized(&manifest).expect_err("schema must be rejected");
            assert!(matches!(error, FocrError::FormatMismatch(_)));
        }
    }

    #[test]
    fn every_published_model_quant_requires_its_exact_recipe() {
        let exact = [
            ("unlimited-ocr", "int8", UNLIMITED_OCR_REQUIRED_RECIPE),
            ("got-ocr2", "int8", GOT_OCR2_INT8_RECIPE),
            ("smolvlm2", "int8", SMOLVLM2_INT8_RECIPE),
            ("onechart", "int8", ONECHART_INT8_RECIPE),
            ("tromr", "int8", TROMR_INT8_RECIPE),
            ("tromr", "f32", TROMR_F32_RECIPE),
        ];
        for (model, quant, recipe) in exact {
            assert_eq!(required_quant_recipe(model, quant), Some(recipe));
            assert!(quant_recipe_is_compatible(model, quant, recipe));
            assert!(!quant_recipe_is_compatible(model, quant, "arbitrary-v1"));
        }
        assert!(!quant_recipe_is_compatible(
            "tromr",
            "int8",
            TROMR_F32_RECIPE
        ));
        assert!(!quant_recipe_is_compatible(
            "unknown-model",
            "int8",
            "arbitrary-v1"
        ));

        let entry = QuantEntry {
            recipe: "arbitrary-v1".into(),
            focrq: tiny_file("got.focrq"),
        };
        assert!(matches!(
            validate_quant_compatibility("got-ocr2", "int8", &entry),
            Err(FocrError::FormatMismatch(_))
        ));
    }

    #[test]
    fn legacy_default_pull_stops_before_any_artifact_request() {
        let suffix = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "focr_legacy_manifest_{}_{}.json",
            std::process::id(),
            suffix
        ));
        let mut manifest = tiny_manifest();
        manifest.quants.get_mut(DEFAULT_QUANT).unwrap().recipe =
            UNLIMITED_OCR_LEGACY_FULL_INT8_RECIPE.into();
        manifest.quants.get_mut(DEFAULT_QUANT).unwrap().focrq.parts[0].urls =
            vec!["https://127.0.0.1:9/must-not-be-requested".into()];
        std::fs::write(
            &path,
            serde_json::to_vec(&manifest).expect("serialize legacy manifest"),
        )
        .expect("write legacy manifest");

        let mut progress = Vec::new();
        let error = pull(
            None,
            DEFAULT_QUANT,
            path.to_str().expect("UTF-8 temp path"),
            true,
            |line| progress.push(line.to_owned()),
        )
        .expect_err("legacy recipe must fail before its unreachable artifact URL");
        assert!(matches!(error, FocrError::FormatMismatch(_)));
        assert!(
            progress.is_empty(),
            "artifact progress proves the fail-before-download order"
        );
        std::fs::remove_file(path).expect("remove test manifest");
    }

    #[test]
    fn manifest_rejects_traversal_and_nonportable_filenames() {
        for filename in [
            "../escape",
            "/absolute",
            "nested/file",
            r"nested\file",
            "C:drive",
            "CON",
            "com1.json",
            "trailing.",
            ".focr-stage-reserved.partial",
        ] {
            let mut manifest = tiny_manifest();
            manifest
                .quants
                .get_mut(DEFAULT_QUANT)
                .unwrap()
                .focrq
                .filename = filename.into();
            assert!(
                matches!(
                    parse_serialized(&manifest),
                    Err(FocrError::FormatMismatch(_))
                ),
                "accepted unsafe filename {filename:?}"
            );
        }

        let mut duplicate_install_name = tiny_manifest();
        duplicate_install_name.tokenizer.filename = "MODEL.FOCRQ".into();
        assert!(matches!(
            parse_serialized(&duplicate_install_name),
            Err(FocrError::FormatMismatch(_))
        ));

        let mut duplicate_model_id = tiny_manifest();
        duplicate_model_id
            .models
            .insert("Unlimited-OCR".into(), tiny_model_entry());
        assert!(matches!(
            parse_serialized(&duplicate_model_id),
            Err(FocrError::FormatMismatch(_))
        ));
    }

    #[test]
    fn manifest_rejects_bad_hash_url_and_size_contracts() {
        let mut uppercase_hash = tiny_manifest();
        uppercase_hash.tokenizer.sha256 = "AB".repeat(32);
        assert!(matches!(
            parse_serialized(&uppercase_hash),
            Err(FocrError::FormatMismatch(_))
        ));

        for url in ["http://example.invalid/file", "https:///missing-host"] {
            let mut manifest = tiny_manifest();
            manifest.tokenizer.parts[0].urls[0] = url.into();
            assert!(
                matches!(
                    parse_serialized(&manifest),
                    Err(FocrError::FormatMismatch(_))
                ),
                "accepted invalid URL {url:?}"
            );
        }

        let mut mismatched_size = tiny_manifest();
        mismatched_size.tokenizer.size = 2;
        assert!(matches!(
            parse_serialized(&mismatched_size),
            Err(FocrError::FormatMismatch(_))
        ));

        let mut oversized_artifact = tiny_manifest();
        oversized_artifact.tokenizer.size = MAX_ARTIFACT_BYTES + 1;
        oversized_artifact.tokenizer.parts[0].size = MAX_ARTIFACT_BYTES + 1;
        assert!(matches!(
            parse_serialized(&oversized_artifact),
            Err(FocrError::FormatMismatch(_))
        ));

        let mut oversized_part = tiny_manifest();
        oversized_part.tokenizer.size = MAX_PART_BYTES + 1;
        oversized_part.tokenizer.parts[0].size = MAX_PART_BYTES + 1;
        assert!(matches!(
            parse_serialized(&oversized_part),
            Err(FocrError::FormatMismatch(_))
        ));

        let mut too_many_parts = tiny_manifest();
        too_many_parts.tokenizer.size = (MAX_PARTS_PER_FILE + 1) as u64;
        too_many_parts.tokenizer.parts = (0..=MAX_PARTS_PER_FILE)
            .map(|_| RemotePart {
                size: 1,
                sha256: "ab".repeat(32),
                urls: vec!["https://example.invalid/part".into()],
            })
            .collect();
        assert!(matches!(
            parse_serialized(&too_many_parts),
            Err(FocrError::FormatMismatch(_))
        ));

        let mut too_many_urls = tiny_manifest();
        too_many_urls.tokenizer.parts[0].urls = (0..=MAX_URLS_PER_PART)
            .map(|index| format!("https://example.invalid/mirror-{index}"))
            .collect();
        assert!(matches!(
            parse_serialized(&too_many_urls),
            Err(FocrError::FormatMismatch(_))
        ));

        let mut missing_recipe = tiny_manifest();
        missing_recipe
            .quants
            .get_mut(DEFAULT_QUANT)
            .unwrap()
            .recipe
            .clear();
        assert!(matches!(
            parse_serialized(&missing_recipe),
            Err(FocrError::FormatMismatch(_))
        ));
    }

    #[test]
    fn manifest_rejects_excessive_collection_counts() {
        let mut too_many_quants = tiny_manifest();
        too_many_quants.quants = (0..=MAX_QUANTS_PER_MODEL)
            .map(|index| {
                (
                    format!("q{index}"),
                    QuantEntry {
                        recipe: format!("recipe-{index}"),
                        focrq: tiny_file(&format!("model-{index}.focrq")),
                    },
                )
            })
            .collect();
        assert!(matches!(
            parse_serialized(&too_many_quants),
            Err(FocrError::FormatMismatch(_))
        ));

        let mut too_many_sidecars = tiny_manifest();
        let mut model = tiny_model_entry();
        model.sidecars = (0..=MAX_SIDECARS_PER_MODEL)
            .map(|index| tiny_file(&format!("sidecar-{index}.json")))
            .collect();
        too_many_sidecars.models.insert("secondary".into(), model);
        assert!(matches!(
            parse_serialized(&too_many_sidecars),
            Err(FocrError::FormatMismatch(_))
        ));

        let mut too_many_models = tiny_manifest();
        too_many_models.models = (0..=MAX_MODELS)
            .map(|index| (format!("model-{index}"), tiny_model_entry()))
            .collect();
        assert!(matches!(
            parse_serialized(&too_many_models),
            Err(FocrError::FormatMismatch(_))
        ));
    }

    #[test]
    fn manifest_and_part_body_limits_fail_before_growth() {
        let oversized = vec![b' '; MAX_MANIFEST_BYTES + 1];
        assert!(matches!(
            parse_manifest(&oversized),
            Err(FocrError::FormatMismatch(_))
        ));

        let suffix = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "focr_oversized_manifest_{}_{}",
            std::process::id(),
            suffix
        ));
        std::fs::write(&path, &oversized).expect("write oversized local manifest");
        assert!(matches!(
            read_local_manifest(&path),
            Err(FocrError::FormatMismatch(_))
        ));
        std::fs::remove_file(path).expect("remove oversized local manifest");

        let mut buffer = vec![1, 2, 3];
        let before = buffer.clone();
        let error = extend_bounded(&mut buffer, &[4, 5], 4, "manifest")
            .expect_err("oversized chunk must be refused");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(buffer, before, "bounded sink must not partially append");

        assert_eq!(checked_part_response_size(3, 2, 5).unwrap(), 5);
        let error = checked_part_response_size(5, 1, 5)
            .expect_err("oversized part body must be refused before writing");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn cache_hashing_uses_a_fixed_size_streaming_buffer() {
        struct ProbeReader<'a> {
            remaining: usize,
            largest_request: &'a mut usize,
        }

        impl Read for ProbeReader<'_> {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                *self.largest_request = (*self.largest_request).max(buffer.len());
                let read = self.remaining.min(buffer.len());
                buffer[..read].fill(0x5a);
                self.remaining -= read;
                Ok(read)
            }
        }

        let mut largest_request = 0;
        let reader = ProbeReader {
            remaining: HASH_BUFFER_BYTES * 3 + 7,
            largest_request: &mut largest_request,
        };
        let _ = sha256_reader(reader).expect("streaming hash");
        assert_eq!(largest_request, HASH_BUFFER_BYTES);
    }

    #[test]
    fn install_coordination_is_exclusive_and_uses_unique_staging_files() {
        let suffix = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "focr_dist_coordination_{}_{}",
            std::process::id(),
            suffix
        ));
        std::fs::create_dir_all(&dir).expect("create test directory");
        let first = dir.join("artifact.bin");
        let same_stem = dir.join("artifact.json");
        assert_ne!(
            coordination_key(&first).unwrap(),
            coordination_key(&same_stem).unwrap(),
            "different filenames must not alias one lock"
        );

        let first_lock = InstallLock::acquire(&first).expect("first lock");
        assert!(matches!(
            InstallLock::acquire(&first),
            Err(FocrError::Timeout(_))
        ));
        let second_lock = InstallLock::acquire(&same_stem).expect("independent lock");

        let (stage_a, stage_a_file) = create_staging_file(&first).expect("first stage");
        let (stage_b, stage_b_file) = create_staging_file(&first).expect("second stage");
        assert_ne!(stage_a, stage_b);
        assert_eq!(
            pull_lock_path_for_staging(&stage_a),
            Some(dir.join(format!(
                ".focr-pull-{}.lock",
                coordination_key(&first).unwrap()
            )))
        );
        drop(stage_a_file);
        drop(stage_b_file);
        std::fs::remove_file(stage_a).expect("remove first stage");
        std::fs::remove_file(stage_b).expect("remove second stage");
        drop(first_lock);
        drop(second_lock);
        let recovered = InstallLock::acquire(&first)
            .expect("persistent lock pathname must be reusable after descriptor release");
        drop(recovered);
        std::fs::remove_file(dir.join(format!(
            ".focr-pull-{}.lock",
            coordination_key(&first).unwrap()
        )))
        .expect("remove first persistent lock file");
        std::fs::remove_file(dir.join(format!(
            ".focr-pull-{}.lock",
            coordination_key(&same_stem).unwrap()
        )))
        .expect("remove second persistent lock file");
        std::fs::remove_dir(dir).expect("remove test directory");
    }

    #[test]
    fn staged_commit_replaces_atomically_and_failure_preserves_old_file() {
        let suffix = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "focr_dist_atomic_commit_{}_{}",
            std::process::id(),
            suffix
        ));
        std::fs::create_dir_all(&dir).expect("create atomic commit directory");
        let final_path = dir.join("artifact.bin");
        let staging_path = dir.join("artifact.stage");
        std::fs::write(&final_path, b"verified-old").expect("write old artifact");
        std::fs::write(&staging_path, b"verified-new").expect("write staged artifact");

        commit_staging_file(&staging_path, &final_path).expect("atomic replacement");
        assert_eq!(
            std::fs::read(&final_path).expect("read replacement"),
            b"verified-new"
        );

        let missing_stage = dir.join("missing.stage");
        commit_staging_file(&missing_stage, &final_path)
            .expect_err("missing staging file must fail");
        assert_eq!(
            std::fs::read(&final_path).expect("old destination survives failed commit"),
            b"verified-new"
        );
        std::fs::remove_file(final_path).expect("remove committed test artifact");
        std::fs::remove_dir(dir).expect("remove atomic commit directory");
    }

    #[cfg(unix)]
    #[test]
    fn cache_artifact_and_model_directory_symlinks_fail_closed() {
        use std::os::unix::fs::symlink;

        let suffix = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "focr_dist_symlink_{}_{}",
            std::process::id(),
            suffix
        ));
        let real_model_dir = root.join("real-model-dir");
        std::fs::create_dir_all(&real_model_dir).expect("create real model directory");
        let linked_model_dir = root.join("got-ocr2");
        symlink(&real_model_dir, &linked_model_dir).expect("link model directory");
        assert!(matches!(
            ensure_real_directory(&linked_model_dir, "model cache subdirectory"),
            Err(FocrError::FormatMismatch(_))
        ));

        let real_artifact = root.join("real.focrq");
        std::fs::write(&real_artifact, b"model").expect("write real artifact");
        let linked_artifact = root.join("linked.focrq");
        symlink(&real_artifact, &linked_artifact).expect("link artifact");
        let mut hash = Sha256::new();
        hash.update(b"model");
        let digest: [u8; 32] = hash.finalize().into();
        let expected = RemoteFile {
            filename: "linked.focrq".into(),
            size: 5,
            sha256: hex32(&digest),
            parts: Vec::new(),
        };
        assert!(
            !already_cached(&linked_artifact, &expected),
            "matching target bytes must not turn a symlink into a cache hit"
        );
        assert!(matches!(
            validate_install_target_type(&linked_artifact),
            Err(FocrError::FormatMismatch(_))
        ));
        assert!(
            std::fs::symlink_metadata(&linked_artifact)
                .expect("symlink remains")
                .file_type()
                .is_symlink(),
            "fail-closed validation must not replace the symlink"
        );
        assert_eq!(
            std::fs::read(&real_artifact).expect("target remains readable"),
            b"model"
        );

        std::fs::remove_file(linked_artifact).expect("remove artifact link");
        std::fs::remove_file(real_artifact).expect("remove artifact target");
        std::fs::remove_file(linked_model_dir).expect("remove directory link");
        std::fs::remove_dir(real_model_dir).expect("remove real model directory");
        std::fs::remove_dir(root).expect("remove test root");
    }

    #[test]
    fn manifest_source_resolution_precedence() {
        // Explicit arg wins.
        assert_eq!(resolve_manifest_source(Some("/tmp/m.json")), "/tmp/m.json");
        // Default when nothing set (env not set in this unit context).
        // (We avoid mutating process env in tests; just assert the default const
        // is what falls through.)
        assert_eq!(DEFAULT_MANIFEST_SOURCE, "builtin:models/manifest-v2.json");
    }

    #[test]
    fn is_url_discriminates() {
        assert!(is_url("https://x/y"));
        assert!(is_url("http://x/y"));
        assert!(!is_url("/local/path"));
        assert!(!is_url("models/manifest.json"));
    }

    #[test]
    fn already_cached_matches_only_on_size_and_hash() {
        let dir = std::env::temp_dir().join(format!("focr_dist_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("blob.bin");
        std::fs::write(&p, b"hello franken").unwrap();
        let mut h = Sha256::new();
        h.update(b"hello franken");
        let d: [u8; 32] = h.finalize().into();
        let good = RemoteFile {
            filename: "blob.bin".into(),
            size: 13,
            sha256: hex32(&d),
            parts: vec![],
        };
        assert!(already_cached(&p, &good));
        let wrong_size = RemoteFile {
            size: 99,
            ..good.clone()
        };
        assert!(!already_cached(&p, &wrong_size));
        let wrong_hash = RemoteFile {
            sha256: "00".repeat(32),
            ..good.clone()
        };
        assert!(!already_cached(&p, &wrong_hash));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// bd-av64.7: the COMMITTED manifest is embedded and must lint clean —
    /// every sha 64-hex lowercase, every size positive, every URL https,
    /// part sizes summing to the whole, filenames unique per install dir —
    /// and it must publish the full runtime-ready zoo.
    #[test]
    fn builtin_manifest_publishes_the_zoo_and_lints_clean() {
        let m = builtin_manifest().expect("embedded manifest parses");
        assert_eq!(m.model, "unlimited-ocr");
        let primary = &m.quants[DEFAULT_QUANT];
        assert_eq!(primary.recipe, UNLIMITED_OCR_REQUIRED_RECIPE);
        validate_quant_compatibility(&m.model, DEFAULT_QUANT, primary)
            .expect("committed primary artifact must match the runtime recipe");
        assert_eq!(
            primary.focrq.filename,
            format!("unlimited-ocr.v{UNLIMITED_OCR_ARTIFACT_VERSION}.int8.focrq"),
            "distributed filename must be bound to the verified model release"
        );
        assert_eq!(primary.focrq.size, 4_157_448_783);
        assert_eq!(
            primary.focrq.sha256,
            "573340710167697891bf52dfa4cbb5d0a02a68f3011c01f8ef83fd34622fb592"
        );
        let expected_parts = [
            (
                1_957_046_720,
                "a45aa7674f38190974a2e61bdaeb8eca0d5039a6631406c1126f6614140ec7f6",
                "unlimited-ocr.v0.7.0.int8.focrq.part00",
            ),
            (
                1_957_046_720,
                "0081dbab8005f9bae0abae32fea6f85d20b507697ee55f2daff8d66137f9d5a8",
                "unlimited-ocr.v0.7.0.int8.focrq.part01",
            ),
            (
                243_355_343,
                "62d34bc6acb431e0b261e8d42c0834886f3b260083c3db2ba46fde5d0d6d2eec",
                "unlimited-ocr.v0.7.0.int8.focrq.part02",
            ),
        ];
        assert_eq!(primary.focrq.parts.len(), expected_parts.len());
        for (part, (size, sha256, filename)) in primary.focrq.parts.iter().zip(expected_parts) {
            assert_eq!(part.size, size);
            assert_eq!(part.sha256, sha256);
            assert_eq!(part.urls.len(), 1);
            assert!(part.urls.iter().all(|url| url.ends_with(filename)));
            assert!(part.urls[0].contains(&format!(
                "/releases/download/v{UNLIMITED_OCR_ARTIFACT_VERSION}/"
            )));
        }

        let mut legacy = primary.clone();
        legacy.recipe = UNLIMITED_OCR_LEGACY_FULL_INT8_RECIPE.into();
        validate_quant_compatibility(&m.model, DEFAULT_QUANT, &legacy)
            .expect_err("historical full-int8 artifacts must remain fail-closed");
        for id in ["got-ocr2", "smolvlm2", "onechart", "tromr"] {
            assert!(m.models.contains_key(id), "manifest missing model {id}");
        }
        // Runtime-required sidecar sets (the anti-broken-pull contract; the
        // loaders resolve these beside the artifact — see native_engine).
        let tromr = &m.models["tromr"];
        assert_eq!(
            tromr.quants.keys().cloned().collect::<Vec<_>>(),
            vec!["f32".to_string(), "int8".to_string()],
            "tromr publishes f32 AND int8 (bd-av64.12: 40 decoder GEMMs, \
             golden byte-identical, corpus gate delta 0)"
        );
        assert_eq!(tromr.tokenizer.filename, "tokenizer_rhythm.json");
        let mut tromr_sidecars: Vec<&str> =
            tromr.sidecars.iter().map(|f| f.filename.as_str()).collect();
        tromr_sidecars.sort_unstable();
        assert_eq!(
            tromr_sidecars,
            vec![
                "tokenizer_lift.json",
                "tokenizer_note.json",
                "tokenizer_pitch.json"
            ]
        );
        let onechart = &m.models["onechart"];
        assert_eq!(onechart.tokenizer.filename, "vocab.json");
        let mut oc_sidecars: Vec<&str> = onechart
            .sidecars
            .iter()
            .map(|f| f.filename.as_str())
            .collect();
        oc_sidecars.sort_unstable();
        assert_eq!(oc_sidecars, vec!["added_tokens.json", "merges.txt"]);
        assert_eq!(m.models["smolvlm2"].tokenizer.filename, "tokenizer.json");
        assert!(m.models["smolvlm2"].sidecars.is_empty());

        // The lint, over every file of every model.
        let lint = |file: &RemoteFile, ctx: &str| {
            assert!(file.size > 0, "{ctx}: zero size");
            assert_eq!(file.sha256.len(), 64, "{ctx}: sha length");
            assert!(
                file.sha256
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "{ctx}: sha not lowercase hex"
            );
            assert!(!file.parts.is_empty(), "{ctx}: no parts");
            assert_eq!(
                file.parts.iter().map(|p| p.size).sum::<u64>(),
                file.size,
                "{ctx}: part sizes do not sum to the whole"
            );
            for part in &file.parts {
                assert!(!part.urls.is_empty(), "{ctx}: part without urls");
                for url in &part.urls {
                    assert!(url.starts_with("https://"), "{ctx}: non-https url {url}");
                }
            }
        };
        let mut primary: Vec<&str> = vec![m.tokenizer.filename.as_str()];
        lint(&m.tokenizer, "primary tokenizer");
        for (tag, q) in &m.quants {
            lint(&q.focrq, &format!("primary quant {tag}"));
            primary.push(q.focrq.filename.as_str());
        }
        primary.sort_unstable();
        primary.dedup();
        assert_eq!(
            primary.len(),
            1 + m.quants.len(),
            "primary filename collision"
        );
        for (id, entry) in &m.models {
            let mut names: Vec<&str> = vec![entry.tokenizer.filename.as_str()];
            lint(&entry.tokenizer, &format!("{id} tokenizer"));
            assert!(
                !entry.license_notice.is_empty(),
                "{id}: empty license notice"
            );
            for (tag, q) in &entry.quants {
                assert!(!q.recipe.is_empty(), "{id}/{tag}: empty recipe");
                assert!(
                    quant_recipe_is_compatible(id, tag, &q.recipe),
                    "{id}/{tag}: manifest recipe {:?} is not the exact runtime recipe",
                    q.recipe
                );
                lint(&q.focrq, &format!("{id} quant {tag}"));
                names.push(q.focrq.filename.as_str());
            }
            for sc in &entry.sidecars {
                lint(sc, &format!("{id} sidecar {}", sc.filename));
                names.push(sc.filename.as_str());
            }
            let total = names.len();
            names.sort_unstable();
            names.dedup();
            assert_eq!(
                names.len(),
                total,
                "{id}: filename collision in its install dir"
            );
        }
    }

    /// bd-av64.7: exact quant wins; a sole published quant is the fallback
    /// (TrOMR is f32-only while the CLI default is int8); several published
    /// quants + a miss stays a loud Usage error.
    #[test]
    fn select_quant_exact_sole_and_ambiguous() {
        let file = RemoteFile {
            filename: "x.focrq".into(),
            size: 1,
            sha256: "ab".repeat(32),
            parts: vec![],
        };
        let entry = QuantEntry {
            recipe: "test-recipe-v1".into(),
            focrq: file,
        };
        let sole = BTreeMap::from([("f32".to_string(), entry.clone())]);
        let (tag, _) = select_quant(&sole, "int8").expect("sole quant falls back");
        assert_eq!(tag, "f32");
        let (tag, _) = select_quant(&sole, "f32").expect("exact match");
        assert_eq!(tag, "f32");
        let two = BTreeMap::from([
            ("int8".to_string(), entry.clone()),
            ("int4".to_string(), entry.clone()),
        ]);
        let (tag, _) = select_quant(&two, "int8").expect("exact among several");
        assert_eq!(tag, "int8");
        assert!(
            select_quant(&two, "f32").is_err(),
            "ambiguous miss must error"
        );
        assert!(select_quant(&BTreeMap::new(), "int8").is_err());
    }

    /// The optional sidecars collection defaults to empty within schema v2.
    #[test]
    fn model_entry_without_sidecars_parses_empty() {
        let json = r#"{
            "license_notice": "x",
            "quants": {"int8": {"recipe": "test-recipe-v1",
                "focrq": {"filename": "a.focrq", "size": 1,
                "sha256": "00", "parts": []}}},
            "tokenizer": {"filename": "t.json", "size": 1, "sha256": "00", "parts": []}
        }"#;
        let entry: ModelEntry = serde_json::from_str(json).expect("parses");
        assert!(entry.sidecars.is_empty());
    }
}
