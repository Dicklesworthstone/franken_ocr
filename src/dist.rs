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
use std::ffi::OsStr;
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
/// renaming or republishing them. The value lives at the crate root (so model
/// *resolution* in native_engine can use it in builds without this module);
/// within this module only the manifest tests reference it by name.
#[cfg(test)]
pub(crate) use crate::UNLIMITED_OCR_ARTIFACT_VERSION;

/// Recipe carried by the historical full-int8 artifact. It is retained for
/// provenance and fail-closed compatibility tests, but is not compatible with
/// the conservative default runtime.
pub const UNLIMITED_OCR_LEGACY_FULL_INT8_RECIPE: &str =
    "unlimited-ocr-full-int8-attn-int8-lmhead-int8-v1";

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
    /// The model's own release version (D22), for a model whose artifacts are
    /// versioned apart from franken_ocr: every artifact filename carries it as
    /// `.v<version>.` (`unlimited-ocr-2.v0.1.0.int8.focrq`), and it equals the
    /// version this runtime's architecture expects. `None` for the unversioned
    /// zoo entries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
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

/// A model entry's `version` (D22): a manifest atom that every artifact
/// filename carries as `.v<version>.`, and, for a model this runtime knows,
/// the version its architecture expects
/// ([`ModelArch::artifact_version`](crate::model_arch::ModelArch::artifact_version)).
fn validate_model_version(
    model_id: &str,
    version: &str,
    quants: &BTreeMap<String, QuantEntry>,
) -> FocrResult<()> {
    validate_manifest_atom(version, &format!("version for {model_id}"), MAX_NAME_BYTES)?;
    let tag = format!(".v{version}.");
    for (quant, entry) in quants {
        if !entry.focrq.filename.contains(&tag) {
            return Err(FocrError::FormatMismatch(format!(
                "manifest model {model_id:?} is version {version}, but its {quant} artifact \
                 {:?} does not carry {tag:?}",
                entry.focrq.filename
            )));
        }
    }
    if let Some(arch) = crate::model_arch::arch_by_id(model_id)
        && arch.artifact_version() != version
    {
        return Err(FocrError::FormatMismatch(format!(
            "manifest model {model_id:?} is version {version}, but this runtime runs version {}",
            arch.artifact_version()
        )));
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
        // A secondary model installs into `models/<id>/`, and the model search
        // passes over hidden directories (they hold pull state).
        if model_id.starts_with('.') {
            return Err(FocrError::FormatMismatch(format!(
                "manifest secondary model id {model_id:?} must not start with '.': its \
                 package directory would be hidden from the model search"
            )));
        }
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
        if let Some(version) = &entry.version {
            validate_model_version(model_id, version, &entry.quants)?;
        }
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

/// The exact recipe this runtime requires for one manifest model/quant pair
/// ([`ModelArch::required_quant_recipe`](crate::model_arch::ModelArch::required_quant_recipe));
/// `None` for an unknown model id or a pair without a contract.
fn required_recipe(model_id: &str, quant: &str) -> Option<&'static str> {
    crate::model_arch::arch_by_id(model_id).and_then(|arch| arch.required_quant_recipe(quant))
}

/// Whether a manifest quant exactly matches this runtime's model contract.
#[must_use]
pub fn quant_recipe_is_compatible(model_id: &str, quant: &str, recipe: &str) -> bool {
    required_recipe(model_id, quant) == Some(recipe)
}

fn validate_quant_compatibility(model_id: &str, quant: &str, entry: &QuantEntry) -> FocrResult<()> {
    let Some(required) = required_recipe(model_id, quant) else {
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

/// Where artifact part bytes come from: the HTTPS mirrors in a real pull, and
/// memory in the offline tests. Either way [`download_remote_file`] does the
/// per-part and whole-file verification.
trait PartSource {
    /// Stream `url`'s body into `sink`; returns the bytes delivered.
    async fn stream(
        &self,
        cx: &Cx,
        url: &str,
        sink: &mut dyn FnMut(&[u8]) -> std::io::Result<()>,
    ) -> Result<u64, DownloadError>;
}

/// The source of a real pull: each part URL is an HTTPS mirror.
struct HttpsMirrors {
    client: Client,
}

impl PartSource for HttpsMirrors {
    async fn stream(
        &self,
        cx: &Cx,
        url: &str,
        sink: &mut dyn FnMut(&[u8]) -> std::io::Result<()>,
    ) -> Result<u64, DownloadError> {
        stream_url(cx, &self.client, url, sink).await
    }
}

/// Downloads a [`RemoteFile`] from a [`PartSource`] into an open staging file,
/// blocking on the pull's runtime.
struct Downloader<'a, S> {
    runtime: &'a asupersync::runtime::Runtime,
    source: &'a S,
}

impl<S: PartSource> Downloader<'_, S> {
    fn fetch(
        &self,
        file: &RemoteFile,
        dest: &Path,
        out: std::fs::File,
        progress: &mut dyn FnMut(&PullEvent),
    ) -> FocrResult<()> {
        progress(&PullEvent::Note(format!(
            "downloading {} ({:.2} GB, {} part(s))",
            file.filename,
            file.size as f64 / 1.0e9,
            file.parts.len()
        )));
        self.runtime.block_on(async {
            let cx = Cx::current().ok_or_else(|| {
                FocrError::Other(anyhow::anyhow!("runtime did not install an ambient Cx"))
            })?;
            download_remote_file(&cx, self.source, file, dest, out, progress).await
        })
    }
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

/// Download + verify one [`RemoteFile`] from `source` into `dest`, a fresh
/// staging file. Each part is streamed straight into `dest` while a per-part AND
/// a whole-file hasher run; a failed mirror rolls `dest` + the whole-file hash
/// back to the part boundary and tries the next URL. The whole-file sha256 is
/// checked last. Every attempt and the whole-file check report to `progress`.
async fn download_remote_file(
    cx: &Cx,
    source: &impl PartSource,
    file: &RemoteFile,
    dest: &Path,
    mut out: std::fs::File,
    progress: &mut dyn FnMut(&PullEvent),
) -> FocrResult<()> {
    let mut full = Sha256::new();
    let mut committed: u64 = 0;

    for (i, part) in file.parts.iter().enumerate() {
        let full_checkpoint = full.clone();
        let mut part_ok = false;
        let mut last_err: Option<String> = None;
        let expected = part.sha256.trim().to_ascii_lowercase();

        for url in &part.urls {
            // Roll back to the part boundary before each attempt.
            out.set_len(committed)
                .and_then(|()| out.seek(SeekFrom::Start(committed)).map(|_| ()))
                .map_err(|e| FocrError::Other(anyhow::anyhow!("seek {}: {e}", dest.display())))?;
            full = full_checkpoint.clone();
            let mut part_hash = Sha256::new();

            progress(&PullEvent::PartStart {
                file: file.filename.clone(),
                part: i + 1,
                parts: file.parts.len(),
                bytes: part.size,
                url: url.clone(),
            });
            let mut received = 0u64;
            let res = source
                .stream(cx, url, &mut |chunk| {
                    let next = checked_part_response_size(received, chunk.len(), part.size)?;
                    out.write_all(chunk)?;
                    part_hash.update(chunk);
                    full.update(chunk);
                    received = next;
                    Ok(())
                })
                .await;

            let (computed, error) = match res {
                Ok(n) if n == part.size => {
                    let got: [u8; 32] = part_hash.finalize().into();
                    let got = hex32(&got);
                    let error = (got != expected)
                        .then(|| format!("part {} sha256 mismatch from {url}", i + 1));
                    (Some(got), error)
                }
                Ok(n) => (
                    None,
                    Some(format!("part {} size {n} != expected {}", i + 1, part.size)),
                ),
                Err(e) => (None, Some(format!("{e} (from {url})"))),
            };
            progress(&PullEvent::Part {
                file: file.filename.clone(),
                part: i + 1,
                parts: file.parts.len(),
                url: url.clone(),
                bytes: received,
                expected_sha256: expected.clone(),
                computed_sha256: computed,
                error: error.clone(),
            });
            match error {
                None => {
                    part_ok = true;
                    committed += part.size;
                    break;
                }
                Some(error) => last_err = Some(error),
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
    let got = hex32(&got);
    let expected = file.sha256.trim().to_ascii_lowercase();
    progress(&PullEvent::File {
        file: file.filename.clone(),
        bytes: committed,
        expected_sha256: expected.clone(),
        computed_sha256: got.clone(),
    });
    if got != expected {
        return Err(FocrError::FormatMismatch(format!(
            "reassembled {} sha256 {got} != manifest {}",
            file.filename, file.sha256
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

/// The coordination key of one install target name: a file of the flat layout
/// or a per-model package directory. Its lock, stages and aside copy all carry
/// it.
fn name_key(name: &str) -> String {
    let mut hash = Sha256::new();
    hash.update(name.as_bytes());
    let digest: [u8; 32] = hash.finalize().into();
    hex32(&digest)[..32].to_owned()
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
    Ok(name_key(filename))
}

/// The coordination key of model `model_id`'s package directory
/// (`models/<id>/`), for `focr doctor` to name the model behind pull state.
pub(crate) fn package_key(model_id: &str) -> String {
    name_key(model_id)
}

/// Hidden pull state in the model cache, parsed from its name. A pull creates
/// it only while it holds the lock with the same key, so `focr doctor` treats
/// it as interrupted once nothing holds that lock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PullState {
    /// A file download's staging file, `.focr-stage-<key>-<pid>-<n>.partial`.
    FileStage { key: String },
    /// A package install's stage directory, `.focr-stage-<key>-<pid>-<n>.pkg`.
    PackageStage { key: String },
    /// The previous package, moved aside by a commit: `.focr-prev-<key>.pkg`.
    PackageAside { key: String },
}

impl PullState {
    /// The coordination key the entry carries.
    pub(crate) fn key(&self) -> &str {
        match self {
            Self::FileStage { key } | Self::PackageStage { key } | Self::PackageAside { key } => {
                key
            }
        }
    }
}

fn is_key(key: &str) -> bool {
    key.len() == 32 && key.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Parse a model-cache entry name as [`PullState`]; `None` for anything else.
pub(crate) fn parse_pull_state(name: &str) -> Option<PullState> {
    if let Some(key) = name
        .strip_prefix(".focr-prev-")
        .and_then(|rest| rest.strip_suffix(".pkg"))
    {
        return is_key(key).then(|| PullState::PackageAside {
            key: key.to_owned(),
        });
    }
    let (key, suffix) = name.strip_prefix(".focr-stage-")?.split_once('-')?;
    if !is_key(key) {
        return None;
    }
    let key = key.to_owned();
    if suffix.ends_with(".partial") {
        Some(PullState::FileStage { key })
    } else if suffix.ends_with(".pkg") {
        Some(PullState::PackageStage { key })
    } else {
        None
    }
}

/// Map a pull-owned staging file, package stage or aside copy back to its
/// advisory lock. Doctor uses this to tell a live multi-gigabyte pull from
/// state a crashed process left behind.
pub(crate) fn pull_lock_path_for_staging(staging_path: &Path) -> Option<PathBuf> {
    let state = parse_pull_state(staging_path.file_name()?.to_str()?)?;
    Some(
        staging_path
            .parent()?
            .join(format!(".focr-pull-{}.lock", state.key())),
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

/// Create a fresh `.focr-stage-<key>-<pid>-<nonce><suffix>` entry beside
/// `final_path` with `create`, which must fail with `AlreadyExists` on a taken
/// name.
fn create_staging_entry<T>(
    final_path: &Path,
    suffix: &str,
    what: &str,
    create: impl Fn(&Path) -> std::io::Result<T>,
) -> FocrResult<(PathBuf, T)> {
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
            ".focr-stage-{key}-{}-{nonce}{suffix}",
            std::process::id(),
        ));
        match create(&path) {
            Ok(created) => return Ok((path, created)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "create {what} {}: {error}",
                    path.display()
                )));
            }
        }
    }
    Err(FocrError::Other(anyhow::anyhow!(
        "could not allocate a unique {what} for {}",
        final_path.display()
    )))
}

fn create_staging_file(final_path: &Path) -> FocrResult<(PathBuf, std::fs::File)> {
    create_staging_entry(final_path, ".partial", "staging file", |path| {
        OpenOptions::new().write(true).create_new(true).open(path)
    })
}

#[cfg(unix)]
fn sync_directory(directory: &Path) -> FocrResult<()> {
    std::fs::File::open(directory)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| {
            FocrError::Other(anyhow::anyhow!("sync {}: {error}", directory.display()))
        })
}

#[cfg(not(unix))]
fn sync_directory(_directory: &Path) -> FocrResult<()> {
    Ok(())
}

fn sync_parent_dir(path: &Path) -> FocrResult<()> {
    let parent = path.parent().ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "installed path {} has no parent",
            path.display()
        ))
    })?;
    sync_directory(parent)
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

/// What a pull did to the installed state, in [`PullEvent::Transaction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionOutcome {
    /// Already byte-perfect; nothing changed.
    Cached,
    /// The new file or package is in place.
    Installed,
    /// A step failed; the installed state is as it was before the pull.
    RolledBack,
    /// The manifest entry was refused before any download or cache change.
    Refused,
    /// A rename failed after some of the primary model's files were renamed in;
    /// pulling again completes it.
    Incomplete,
    /// State an interrupted pull left behind was restored or removed.
    Recovered,
}

impl TransactionOutcome {
    /// The NDJSON `outcome` value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cached => "cached",
            Self::Installed => "installed",
            Self::RolledBack => "rolled_back",
            Self::Refused => "refused",
            Self::Incomplete => "incomplete",
            Self::Recovered => "recovered",
        }
    }
}

/// One step of a [`pull`], reported as it happens. `Display` is the human
/// status line; [`PullEvent::to_json`] is the object `focr pull --json` writes
/// to stderr, one line per step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PullEvent {
    /// A status line with nothing to verify.
    Note(String),
    /// One mirror's download of one part is starting.
    PartStart {
        /// The file the part belongs to.
        file: String,
        /// 1-based part number.
        part: usize,
        /// The file's part count.
        parts: usize,
        /// The part's size in the manifest.
        bytes: u64,
        /// The mirror.
        url: String,
    },
    /// One mirror's download of one part ended.
    Part {
        /// The file the part belongs to.
        file: String,
        /// 1-based part number.
        part: usize,
        /// The file's part count.
        parts: usize,
        /// The mirror.
        url: String,
        /// Bytes the mirror delivered.
        bytes: u64,
        /// The part's SHA-256 in the manifest.
        expected_sha256: String,
        /// The delivered bytes' SHA-256; `None` when the part did not arrive
        /// whole.
        computed_sha256: Option<String>,
        /// Why the part was rejected; `None` when it verified.
        error: Option<String>,
    },
    /// A reassembled file's whole-file check.
    File {
        /// The file.
        file: String,
        /// Its reassembled size.
        bytes: u64,
        /// Its SHA-256 in the manifest.
        expected_sha256: String,
        /// The reassembled bytes' SHA-256.
        computed_sha256: String,
    },
    /// What happened to an installed file or package.
    Transaction {
        /// The file (flat layout), package directory (per-model layout) or
        /// pull state the step acted on.
        target: PathBuf,
        /// What happened.
        outcome: TransactionOutcome,
        /// Why; empty for `cached` and `installed`.
        reason: String,
    },
}

impl PullEvent {
    fn transaction(target: &Path, outcome: TransactionOutcome, reason: impl Into<String>) -> Self {
        Self::Transaction {
            target: target.to_owned(),
            outcome,
            reason: reason.into(),
        }
    }

    /// The machine-readable form: an `event` tag plus the variant's fields.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Note(message) => serde_json::json!({"event": "note", "message": message}),
            Self::PartStart {
                file,
                part,
                parts,
                bytes,
                url,
            } => serde_json::json!({
                "event": "part_start",
                "file": file,
                "part": part,
                "parts": parts,
                "bytes": bytes,
                "url": url,
            }),
            Self::Part {
                file,
                part,
                parts,
                url,
                bytes,
                expected_sha256,
                computed_sha256,
                error,
            } => serde_json::json!({
                "event": "part",
                "file": file,
                "part": part,
                "parts": parts,
                "url": url,
                "bytes": bytes,
                "expected_sha256": expected_sha256,
                "computed_sha256": computed_sha256,
                "verified": error.is_none(),
                "error": error,
            }),
            Self::File {
                file,
                bytes,
                expected_sha256,
                computed_sha256,
            } => serde_json::json!({
                "event": "file",
                "file": file,
                "bytes": bytes,
                "expected_sha256": expected_sha256,
                "computed_sha256": computed_sha256,
                "verified": expected_sha256 == computed_sha256,
            }),
            Self::Transaction {
                target,
                outcome,
                reason,
            } => serde_json::json!({
                "event": "transaction",
                "target": target.display().to_string(),
                "outcome": outcome.as_str(),
                "reason": reason,
            }),
        }
    }
}

impl std::fmt::Display for PullEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Note(message) => f.write_str(message),
            Self::PartStart {
                part,
                parts,
                bytes,
                url,
                ..
            } => write!(
                f,
                "  part {part}/{parts} ({:.1} MB) <- {url}",
                *bytes as f64 / 1.0e6
            ),
            Self::Part {
                part,
                parts,
                computed_sha256,
                error: None,
                ..
            } => write!(
                f,
                "  part {part}/{parts} verified (sha256 {})",
                computed_sha256.as_deref().unwrap_or("")
            ),
            Self::Part {
                part,
                parts,
                error: Some(error),
                ..
            } => write!(f, "  part {part}/{parts} rejected: {error}"),
            Self::File {
                file,
                bytes,
                expected_sha256,
                computed_sha256,
            } if expected_sha256 == computed_sha256 => {
                write!(f, "verified {file} ({bytes} bytes, sha256 {computed_sha256})")
            }
            Self::File {
                file,
                expected_sha256,
                computed_sha256,
                ..
            } => write!(
                f,
                "{file} reassembled to sha256 {computed_sha256}, but the manifest says \
                 {expected_sha256}"
            ),
            Self::Transaction {
                target,
                outcome,
                reason,
            } => {
                let target = target.display();
                match outcome {
                    TransactionOutcome::Cached => write!(f, "cached: {target}"),
                    TransactionOutcome::Installed => write!(f, "installed {target}"),
                    TransactionOutcome::RolledBack => {
                        write!(f, "rolled back {target}: {reason}")
                    }
                    TransactionOutcome::Refused => write!(f, "refused {target}: {reason}"),
                    TransactionOutcome::Incomplete => {
                        write!(f, "incomplete install of {target}: {reason}")
                    }
                    TransactionOutcome::Recovered => write!(f, "recovered {target}: {reason}"),
                }
            }
        }
    }
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
/// receives each step ([`PullEvent`]). Network only happens here — once it
/// returns Ok, the model loads offline.
pub fn pull(
    model: Option<&str>,
    quant: &str,
    manifest_source: &str,
    mut progress: impl FnMut(&PullEvent),
) -> FocrResult<PullOutcome> {
    let runtime = RuntimeBuilder::new()
        .build()
        .map_err(|e| FocrError::Other(anyhow::anyhow!("runtime build: {e}")))?;

    // 1. Load the release-bound embedded manifest, an explicit local file, or
    // an explicit remote URL. The default never delegates artifact identity to
    // mutable branch state.
    let manifest_bytes = if manifest_source == DEFAULT_MANIFEST_SOURCE {
        progress(&PullEvent::Note(
            "using release-bound embedded model manifest".into(),
        ));
        BUILTIN_MANIFEST_JSON.as_bytes().to_vec()
    } else if is_url(manifest_source) {
        validate_https_url(manifest_source, "source")?;
        progress(&PullEvent::Note(format!(
            "fetching manifest {manifest_source}"
        )));
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
    let cache_dir = cache_models_dir()?;
    let source = HttpsMirrors {
        client: streaming_client(),
    };
    let downloader = Downloader {
        runtime: &runtime,
        source: &source,
    };
    pull_from(
        &downloader,
        &manifest,
        model,
        quant,
        &cache_dir,
        &mut progress,
    )
}

/// [`pull`] once the manifest is parsed: select the model and quant, refuse an
/// incompatible recipe before touching the cache, then install into
/// `cache_dir` from `downloader`'s source.
fn pull_from<S: PartSource>(
    downloader: &Downloader<'_, S>,
    manifest: &Manifest,
    model: Option<&str>,
    quant: &str,
    cache_dir: &Path,
    progress: &mut dyn FnMut(&PullEvent),
) -> FocrResult<PullOutcome> {
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
        progress(&PullEvent::Note(format!(
            "manifest has no quant '{quant}' for this model; using the sole \
             published quant '{quant_used}'"
        )));
    }

    let dir = subdir.map_or_else(|| cache_dir.to_owned(), |sub| cache_dir.join(sub));
    // Compatibility is a manifest contract, not a post-download surprise. The
    // currently committed primary entry intentionally fails here before cache
    // creation, cache hashing, or any multi-GB artifact request.
    if let Err(error) = validate_quant_compatibility(model_id, &quant_used, quant_entry) {
        progress(&PullEvent::transaction(
            &dir,
            TransactionOutcome::Refused,
            error.to_string(),
        ));
        return Err(error);
    }

    ensure_real_directory(cache_dir, "model cache")?;
    let files: Vec<&RemoteFile> = [&quant_entry.focrq, tokenizer]
        .into_iter()
        .chain(sidecars)
        .collect();
    // 2. Download what is not already byte-perfect in the cache, then commit:
    // a per-model package as one directory swap, the primary model's flat files
    // only once every one of them has verified.
    let from_cache = match subdir {
        Some(_) => install_package(downloader, &dir, &files, progress)?,
        None => install_flat(downloader, &dir, &files, progress)?,
    };

    Ok(PullOutcome {
        focrq_path: dir.join(&quant_entry.focrq.filename),
        tokenizer_path: dir.join(&tokenizer.filename),
        sidecar_paths: sidecars
            .iter()
            .map(|sidecar| dir.join(&sidecar.filename))
            .collect(),
        quant: quant_used,
        license_notice: license_notice.to_owned(),
        from_cache,
    })
}

/// Install the primary model flat into `dir`, the layout every released binary
/// knows. Returns `true` when every file was already cached. Each missing file
/// downloads and verifies into its own staging file, under that file's install
/// lock, before any file is renamed into place: a failed download leaves every
/// installed file as it was, and only a crash between the closing renames can
/// leave a mix (a per-model package has no such window: [`install_package`]).
fn install_flat<S: PartSource>(
    downloader: &Downloader<'_, S>,
    dir: &Path,
    files: &[&RemoteFile],
    progress: &mut dyn FnMut(&PullEvent),
) -> FocrResult<bool> {
    let mut locks = Vec::new();
    let mut staged = Vec::new();
    if let Err(error) = stage_flat_files(downloader, dir, files, &mut locks, &mut staged, progress)
    {
        for (staging_path, _) in &staged {
            let _ = std::fs::remove_file(staging_path);
        }
        progress(&PullEvent::transaction(
            dir,
            TransactionOutcome::RolledBack,
            error.to_string(),
        ));
        return Err(error);
    }
    // `rename` within one directory is the commit point of each file: it
    // replaces a regular destination atomically, while a failure leaves the
    // previously verified file in place.
    for (index, (staging_path, final_path)) in staged.iter().enumerate() {
        if let Err(error) = commit_staging_file(staging_path, final_path) {
            for (rest, _) in &staged[index..] {
                let _ = std::fs::remove_file(rest);
            }
            let outcome = if index == 0 {
                TransactionOutcome::RolledBack
            } else {
                TransactionOutcome::Incomplete
            };
            progress(&PullEvent::transaction(dir, outcome, error.to_string()));
            return Err(error);
        }
        progress(&PullEvent::transaction(
            final_path,
            TransactionOutcome::Installed,
            "",
        ));
    }
    Ok(staged.is_empty())
}

/// Download every file of `files` that `dir` lacks into a staging file, under
/// the file's install lock (kept in `locks` until the commit). `staged` lists
/// each staging file as soon as it exists, so the caller can remove it.
fn stage_flat_files<S: PartSource>(
    downloader: &Downloader<'_, S>,
    dir: &Path,
    files: &[&RemoteFile],
    locks: &mut Vec<InstallLock>,
    staged: &mut Vec<(PathBuf, PathBuf)>,
    progress: &mut dyn FnMut(&PullEvent),
) -> FocrResult<()> {
    for file in files {
        let final_path = dir.join(&file.filename);
        if already_cached(&final_path, file) {
            progress(&PullEvent::transaction(
                &final_path,
                TransactionOutcome::Cached,
                "",
            ));
            continue;
        }
        validate_install_target_type(&final_path)?;
        // Recheck under the lock: another process may have finished this file
        // between the optimistic miss and the lock.
        let lock = InstallLock::acquire(&final_path)?;
        if already_cached(&final_path, file) {
            progress(&PullEvent::transaction(
                &final_path,
                TransactionOutcome::Cached,
                "",
            ));
            continue;
        }
        validate_install_target_type(&final_path)?;
        locks.push(lock);
        let (staging_path, out) = create_staging_file(&final_path)?;
        staged.push((staging_path.clone(), final_path));
        downloader.fetch(file, &staging_path, out, progress)?;
    }
    Ok(())
}

/// Install a non-primary model as one package: its directory `dir`
/// (`models/<id>/`) is replaced whole, so no reader ever finds a new artifact
/// beside an old or missing tokenizer (bd-jprb). Returns `true` when the
/// package was already complete.
///
/// Under the package lock the next package is assembled in a hidden sibling
/// stage ([`assemble_package`]), then committed by [`swap_package`]. A failure
/// before the commit leaves the live package untouched. A crash between the
/// commit's two renames leaves the previous package aside, which the next pull
/// restores before anything else ([`recover_package`]) and `focr doctor`
/// reports; the model search never reads hidden directories.
fn install_package<S: PartSource>(
    downloader: &Downloader<'_, S>,
    dir: &Path,
    files: &[&RemoteFile],
    progress: &mut dyn FnMut(&PullEvent),
) -> FocrResult<bool> {
    validate_package_dir(dir)?;
    if !package_has_pull_state(dir) && cached_files(dir, files).len() == files.len() {
        progress(&PullEvent::transaction(
            dir,
            TransactionOutcome::Cached,
            "",
        ));
        return Ok(true);
    }
    let _lock = InstallLock::acquire(dir)?;
    recover_package(dir, progress)?;
    validate_package_dir(dir)?;
    let cached = cached_files(dir, files);
    if cached.len() == files.len() {
        progress(&PullEvent::transaction(
            dir,
            TransactionOutcome::Cached,
            "",
        ));
        return Ok(true);
    }
    let (stage, ()) = create_staging_entry(dir, ".pkg", "package stage", |path| {
        std::fs::create_dir(path)
    })?;
    let committed = assemble_package(downloader, dir, &stage, files, &cached, progress)
        .and_then(|()| swap_package(&stage, dir, progress));
    if let Err(error) = committed {
        // `swap_package` puts the live package back before it fails, so the
        // stage is all there is to clean up.
        let _ = std::fs::remove_dir_all(&stage);
        progress(&PullEvent::transaction(
            dir,
            TransactionOutcome::RolledBack,
            error.to_string(),
        ));
        return Err(error);
    }
    progress(&PullEvent::transaction(
        dir,
        TransactionOutcome::Installed,
        "",
    ));
    Ok(false)
}

/// A package directory, when present, must be a real directory.
fn validate_package_dir(dir: &Path) -> FocrResult<()> {
    match std::fs::symlink_metadata(dir) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => Err(FocrError::FormatMismatch(format!(
            "model package {} must be a real directory, not a symlink or other file type",
            dir.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(FocrError::Other(anyhow::anyhow!(
            "inspect model package {}: {error}",
            dir.display()
        ))),
    }
}

/// Whether package `dir` has a stage or aside copy beside it: one a running
/// pull holds, or one an interrupted pull left. Even a complete package then
/// takes the lock, so the pull recovers that state rather than reporting a
/// cache hit over it.
fn package_has_pull_state(dir: &Path) -> bool {
    let (Some(parent), Ok(key)) = (dir.parent(), coordination_key(dir)) else {
        return false;
    };
    std::fs::read_dir(parent)
        .into_iter()
        .flatten()
        .flatten()
        .any(|entry| {
            entry
                .file_name()
                .to_str()
                .and_then(parse_pull_state)
                .is_some_and(|state| {
                    !matches!(state, PullState::FileStage { .. }) && state.key() == key
                })
        })
}

/// The names of `files` that `dir` already holds byte-perfect.
fn cached_files(dir: &Path, files: &[&RemoteFile]) -> BTreeSet<String> {
    files
        .iter()
        .filter(|file| already_cached(&dir.join(&file.filename), file))
        .map(|file| file.filename.clone())
        .collect()
}

/// Where a commit moves the previous package of `dir` aside.
fn package_aside_path(dir: &Path) -> FocrResult<PathBuf> {
    let parent = dir.parent().ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "model package {} has no parent directory",
            dir.display()
        ))
    })?;
    Ok(parent.join(format!(".focr-prev-{}.pkg", coordination_key(dir)?)))
}

/// Fill `stage` with the next package. Every live entry that stays (a file the
/// manifest still names and `cached` found byte-perfect, or a file the manifest
/// does not name, such as another quant) is hard-linked in, or copied on a
/// filesystem without hard links; every other manifest file downloads and
/// verifies in the stage. Leftover `.focr-` pull state is not carried over.
fn assemble_package<S: PartSource>(
    downloader: &Downloader<'_, S>,
    dir: &Path,
    stage: &Path,
    files: &[&RemoteFile],
    cached: &BTreeSet<String>,
    progress: &mut dyn FnMut(&PullEvent),
) -> FocrResult<()> {
    let replaced: BTreeSet<&OsStr> = files
        .iter()
        .filter(|file| !cached.contains(&file.filename))
        .map(|file| OsStr::new(file.filename.as_str()))
        .collect();
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => Some(entries),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(FocrError::Other(anyhow::anyhow!(
                "read model package {}: {error}",
                dir.display()
            )));
        }
    };
    for entry in entries.into_iter().flatten() {
        let entry = entry.map_err(|error| {
            FocrError::Other(anyhow::anyhow!(
                "read model package {}: {error}",
                dir.display()
            ))
        })?;
        let name = entry.file_name();
        if replaced.contains(name.as_os_str()) || name.as_encoded_bytes().starts_with(b".focr-") {
            continue;
        }
        let source = entry.path();
        let metadata = std::fs::symlink_metadata(&source).map_err(|error| {
            FocrError::Other(anyhow::anyhow!("inspect {}: {error}", source.display()))
        })?;
        if !metadata.is_file() {
            return Err(FocrError::FormatMismatch(format!(
                "model package {} holds {}, which is not a regular file; move it out of the \
                 model cache and pull again",
                dir.display(),
                source.display()
            )));
        }
        let dest = stage.join(&name);
        if std::fs::hard_link(&source, &dest).is_err() {
            std::fs::copy(&source, &dest)
                .and_then(|_| std::fs::File::open(&dest)?.sync_all())
                .map_err(|error| {
                    FocrError::Other(anyhow::anyhow!(
                        "keep {} in the new package: {error}",
                        source.display()
                    ))
                })?;
        }
    }
    for file in files {
        if cached.contains(&file.filename) {
            progress(&PullEvent::transaction(
                &dir.join(&file.filename),
                TransactionOutcome::Cached,
                "",
            ));
            continue;
        }
        let dest = stage.join(&file.filename);
        let out = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&dest)
            .map_err(|error| {
                FocrError::Other(anyhow::anyhow!("create {}: {error}", dest.display()))
            })?;
        downloader.fetch(file, &dest, out, progress)?;
    }
    sync_directory(stage)
}

/// Commit a staged package: move the live package (if any) aside, rename the
/// stage into its place, then remove the aside copy. When the second rename
/// fails the live package is put back.
fn swap_package(
    stage: &Path,
    dir: &Path,
    progress: &mut dyn FnMut(&PullEvent),
) -> FocrResult<()> {
    let aside = package_aside_path(dir)?;
    let had_live = std::fs::symlink_metadata(dir).is_ok();
    if had_live {
        std::fs::rename(dir, &aside).map_err(|error| {
            FocrError::Other(anyhow::anyhow!(
                "move {} aside to {}: {error}",
                dir.display(),
                aside.display()
            ))
        })?;
    }
    if let Err(error) = std::fs::rename(stage, dir) {
        if had_live && let Err(restore) = std::fs::rename(&aside, dir) {
            return Err(FocrError::Other(anyhow::anyhow!(
                "install {} -> {}: {error}; putting the previous package back also failed \
                 ({restore}); the next `focr pull` restores it from {}",
                stage.display(),
                dir.display(),
                aside.display()
            )));
        }
        return Err(FocrError::Other(anyhow::anyhow!(
            "install {} -> {}: {error}",
            stage.display(),
            dir.display()
        )));
    }
    // The new package is in place from here on; what follows cannot undo it.
    if let Err(error) = sync_parent_dir(dir) {
        progress(&PullEvent::Note(format!(
            "installed {}, but {error}",
            dir.display()
        )));
    }
    if had_live && let Err(error) = std::fs::remove_dir_all(&aside) {
        progress(&PullEvent::Note(format!(
            "could not remove the previous package {}: {error}; the next pull removes it",
            aside.display()
        )));
    }
    Ok(())
}

/// Undo what an interrupted install of package `dir` left behind; the caller
/// holds the package lock. An aside copy is the previous package: restored
/// when `dir` is missing (the crash fell between the commit's two renames) and
/// removed when `dir` is present (it fell after them). Every stage with this
/// package's key is abandoned, since stages exist only under the lock.
fn recover_package(dir: &Path, progress: &mut dyn FnMut(&PullEvent)) -> FocrResult<()> {
    let aside = package_aside_path(dir)?;
    match std::fs::symlink_metadata(&aside) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            if std::fs::symlink_metadata(dir).is_ok() {
                std::fs::remove_dir_all(&aside).map_err(|error| {
                    FocrError::Other(anyhow::anyhow!("remove {}: {error}", aside.display()))
                })?;
                progress(&PullEvent::transaction(
                    &aside,
                    TransactionOutcome::Recovered,
                    "removed the previous package an interrupted install left behind",
                ));
            } else {
                std::fs::rename(&aside, dir).map_err(|error| {
                    FocrError::Other(anyhow::anyhow!(
                        "restore {} -> {}: {error}",
                        aside.display(),
                        dir.display()
                    ))
                })?;
                sync_parent_dir(dir)?;
                progress(&PullEvent::transaction(
                    dir,
                    TransactionOutcome::Recovered,
                    format!(
                        "restored the previous package from {} after an interrupted install",
                        aside.display()
                    ),
                ));
            }
        }
        Ok(_) => {
            return Err(FocrError::FormatMismatch(format!(
                "{} must be a directory; move it out of the model cache",
                aside.display()
            )));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(FocrError::Other(anyhow::anyhow!(
                "inspect {}: {error}",
                aside.display()
            )));
        }
    }

    let parent = aside.parent().unwrap_or(dir);
    let key = coordination_key(dir)?;
    let entries = std::fs::read_dir(parent).map_err(|error| {
        FocrError::Other(anyhow::anyhow!("read {}: {error}", parent.display()))
    })?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let is_stage = name.to_str().and_then(parse_pull_state)
            == Some(PullState::PackageStage { key: key.clone() });
        let path = entry.path();
        if !is_stage
            || !std::fs::symlink_metadata(&path)
                .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
        {
            continue;
        }
        std::fs::remove_dir_all(&path).map_err(|error| {
            FocrError::Other(anyhow::anyhow!("remove {}: {error}", path.display()))
        })?;
        progress(&PullEvent::transaction(
            &path,
            TransactionOutcome::Recovered,
            "removed a package stage an interrupted install left behind",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_arch::{
        GOT_OCR2_INT8_RECIPE, ONECHART_INT8_RECIPE, SMOLVLM2_INT8_RECIPE, TROMR_F32_RECIPE,
        TROMR_INT8_RECIPE,
    };

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
            version: None,
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

    /// U2-E4 (D22): a model entry's `version` survives a round trip, every
    /// artifact filename must carry it, and a model this runtime knows must
    /// be at the version its architecture runs. Without it the entry parses
    /// as before and serializes without the key.
    #[test]
    fn model_version_round_trips_and_must_match_filenames_and_the_runtime() {
        let versioned = |version: &str, filename: &str| {
            let mut manifest = tiny_manifest();
            let mut entry = tiny_model_entry();
            entry.version = Some(version.into());
            entry.quants.get_mut("int8").expect("int8").focrq = tiny_file(filename);
            manifest.models.insert("unlimited-ocr-2".into(), entry);
            manifest
        };

        let manifest = versioned("0.1.0", "unlimited-ocr-2.v0.1.0.int8.focrq");
        let parsed = parse_serialized(&manifest).expect("a consistent version parses");
        assert_eq!(
            parsed.models["unlimited-ocr-2"].version.as_deref(),
            Some("0.1.0")
        );

        let unversioned = serde_json::to_string(&tiny_manifest()).expect("serialize");
        assert!(!unversioned.contains("\"version\""), "{unversioned}");
        assert!(
            parse_manifest(unversioned.as_bytes())
                .expect("no version still parses")
                .models
                .values()
                .all(|entry| entry.version.is_none())
        );

        for (manifest, needle) in [
            (
                versioned("0.1.0", "unlimited-ocr-2.int8.focrq"),
                "does not carry",
            ),
            (
                versioned("0.2.0", "unlimited-ocr-2.v0.2.0.int8.focrq"),
                "this runtime runs version 0.1.0",
            ),
            (
                versioned("0.1.0/../x", "unlimited-ocr-2.v0.1.0.int8.focrq"),
                "version for unlimited-ocr-2",
            ),
        ] {
            match parse_serialized(&manifest) {
                Err(FocrError::FormatMismatch(message)) => {
                    assert!(message.contains(needle), "{message}");
                }
                other => panic!("expected FormatMismatch containing {needle:?}, got {other:?}"),
            }
        }
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
                    version: None,
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

    /// Model open refuses a foreign `tokenizer.json` with "re-install its
    /// sidecars with `focr pull <model>`". That advice holds only for a model
    /// the builtin manifest publishes, with exactly the pinned sidecar.
    #[test]
    fn every_pinned_tokenizer_is_the_one_focr_pull_installs() {
        let manifest = builtin_manifest().expect("builtin manifest parses");
        for arch in crate::model_arch::registry() {
            let Some(pinned) = arch.tokenizer_sha256() else {
                assert!(
                    arch.required_token_ids().is_empty(),
                    "{} pins token ids but no sidecar hash",
                    arch.id()
                );
                continue;
            };
            let published = manifest.model == arch.id() || manifest.models.contains_key(arch.id());
            if let Some(source) = arch.tokenizer_source() {
                // Not published yet: its refusals name the pinned upstream file.
                assert!(
                    !published,
                    "`focr pull {}` publishes it; drop its tokenizer_source",
                    arch.id()
                );
                assert!(
                    source.starts_with("https://huggingface.co/")
                        && source.ends_with("/tokenizer.json"),
                    "{}: {source}",
                    arch.id()
                );
                continue;
            }
            let tokenizer = if manifest.model == arch.id() {
                &manifest.tokenizer
            } else {
                &manifest
                    .models
                    .get(arch.id())
                    .unwrap_or_else(|| {
                        panic!(
                            "{} pins its tokenizer, but `focr pull {}` cannot fetch it",
                            arch.id(),
                            arch.id()
                        )
                    })
                    .tokenizer
            };
            assert_eq!(tokenizer.filename, "tokenizer.json", "{}", arch.id());
            assert_eq!(tokenizer.sha256, pinned, "{}", arch.id());
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
            assert_eq!(required_recipe(model, quant), Some(recipe));
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
            |event| progress.push(event.clone()),
        )
        .expect_err("legacy recipe must fail before its unreachable artifact URL");
        assert!(matches!(error, FocrError::FormatMismatch(_)));
        assert!(
            matches!(
                progress.as_slice(),
                [PullEvent::Transaction {
                    outcome: TransactionOutcome::Refused,
                    ..
                }]
            ),
            "the refusal is the only step, which proves the fail-before-download order: \
             {progress:?}"
        );
        std::fs::remove_file(path).expect("remove test manifest");
    }

    /// Serves synthetic part bodies from memory by URL and records every
    /// request, so the offline pull tests run the real verification and
    /// install path (U2-E4; the HTTPS fetch itself is U2-T1a's).
    struct MemoryParts {
        bodies: BTreeMap<String, Vec<u8>>,
        requests: std::cell::RefCell<Vec<String>>,
    }

    impl MemoryParts {
        fn new(bodies: BTreeMap<String, Vec<u8>>) -> Self {
            Self {
                bodies,
                requests: std::cell::RefCell::new(Vec::new()),
            }
        }

        fn request_count(&self) -> usize {
            self.requests.borrow().len()
        }
    }

    impl PartSource for MemoryParts {
        async fn stream(
            &self,
            _cx: &Cx,
            url: &str,
            sink: &mut dyn FnMut(&[u8]) -> std::io::Result<()>,
        ) -> Result<u64, DownloadError> {
            self.requests.borrow_mut().push(url.to_owned());
            let Some(body) = self.bodies.get(url) else {
                return Err(DownloadError::UnexpectedStatus(404));
            };
            for chunk in body.chunks(7) {
                sink(chunk).map_err(DownloadError::Io)?;
            }
            Ok(body.len() as u64)
        }
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        hex32(&Sha256::digest(bytes).into())
    }

    /// `bytes` as a remote file of `parts` parts served from
    /// `https://mirror.invalid/<filename>.partNN`; the bodies go to `bodies`.
    fn split_file(
        filename: &str,
        bytes: &[u8],
        parts: usize,
        bodies: &mut BTreeMap<String, Vec<u8>>,
    ) -> RemoteFile {
        let remote_parts = bytes
            .chunks(bytes.len().div_ceil(parts))
            .enumerate()
            .map(|(index, part)| {
                let url = format!("https://mirror.invalid/{filename}.part{index:02}");
                bodies.insert(url.clone(), part.to_vec());
                RemotePart {
                    size: part.len() as u64,
                    sha256: sha256_hex(part),
                    urls: vec![url],
                }
            })
            .collect();
        RemoteFile {
            filename: filename.into(),
            size: bytes.len() as u64,
            sha256: sha256_hex(bytes),
            parts: remote_parts,
        }
    }

    const V2: &str = crate::model_arch::UNLIMITED_OCR_V2_ID;

    fn v2_artifact_name() -> String {
        let version = crate::model_arch::arch_by_id(V2)
            .expect("v2 arch")
            .artifact_version();
        format!("{V2}.v{version}.int8.focrq")
    }

    /// A manifest publishing v2 as a secondary model with these artifact
    /// (3 parts) and tokenizer bytes, and the bodies its URLs serve.
    fn v2_manifest(artifact: &[u8], tokenizer: &[u8]) -> (Manifest, BTreeMap<String, Vec<u8>>) {
        let mut bodies = BTreeMap::new();
        let mut manifest = tiny_manifest();
        manifest.models.insert(
            V2.into(),
            ModelEntry {
                license_notice: "MIT License, Copyright (c) 2026 Baidu".into(),
                version: Some(
                    crate::model_arch::arch_by_id(V2)
                        .expect("v2 arch")
                        .artifact_version()
                        .into(),
                ),
                quants: BTreeMap::from([(
                    DEFAULT_QUANT.into(),
                    QuantEntry {
                        recipe: required_recipe(V2, DEFAULT_QUANT)
                            .expect("v2 int8 recipe")
                            .into(),
                        focrq: split_file(&v2_artifact_name(), artifact, 3, &mut bodies),
                    },
                )]),
                tokenizer: split_file("tokenizer.json", tokenizer, 1, &mut bodies),
                sidecars: Vec::new(),
            },
        );
        parse_serialized(&manifest).expect("the synthetic v2 manifest is valid");
        (manifest, bodies)
    }

    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|index| (index as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    fn temp_cache(label: &str) -> (PathBuf, PathBuf) {
        let suffix = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "focr_dist_{label}_{}_{suffix}",
            std::process::id()
        ));
        (root.clone(), root.join("models"))
    }

    /// Pull `model` from `manifest` into `cache` from `source`, keeping every
    /// step.
    fn pull_offline(
        manifest: &Manifest,
        model: Option<&str>,
        cache: &Path,
        source: &MemoryParts,
    ) -> (FocrResult<PullOutcome>, Vec<PullEvent>) {
        let runtime = RuntimeBuilder::new().build().expect("runtime");
        let downloader = Downloader {
            runtime: &runtime,
            source,
        };
        let mut events = Vec::new();
        let result = pull_from(
            &downloader,
            manifest,
            model,
            DEFAULT_QUANT,
            cache,
            &mut |event| events.push(event.clone()),
        );
        (result, events)
    }

    fn entry_names(dir: &Path) -> BTreeSet<String> {
        std::fs::read_dir(dir)
            .expect("read directory")
            .map(|entry| {
                entry
                    .expect("directory entry")
                    .file_name()
                    .into_string()
                    .expect("UTF-8 name")
            })
            .collect()
    }

    fn package_lock_name() -> String {
        format!(".focr-pull-{}.lock", package_key(V2))
    }

    fn outcomes(events: &[PullEvent]) -> Vec<TransactionOutcome> {
        events
            .iter()
            .filter_map(|event| match event {
                PullEvent::Transaction { outcome, .. } => Some(*outcome),
                _ => None,
            })
            .collect()
    }

    /// U2-E4 happy path, offline: the parts reassemble, every part and the
    /// whole file verify, and the package lands in `models/unlimited-ocr-2/`
    /// with nothing left beside it but its lock. Pulling again is a cache hit
    /// that requests nothing.
    #[test]
    fn offline_v2_pull_installs_one_verified_package_then_hits_the_cache() {
        let (root, cache) = temp_cache("v2_pull");
        let artifact = pattern(1000, 7);
        let tokenizer = br#"{"model":"v2"}"#.to_vec();
        let (manifest, bodies) = v2_manifest(&artifact, &tokenizer);
        let source = MemoryParts::new(bodies);

        let (outcome, events) = pull_offline(&manifest, Some(V2), &cache, &source);
        let outcome = outcome.expect("offline v2 pull");
        let package = cache.join(V2);
        assert_eq!(outcome.focrq_path, package.join(v2_artifact_name()));
        assert_eq!(outcome.tokenizer_path, package.join("tokenizer.json"));
        assert!(!outcome.from_cache);
        assert_eq!(outcome.license_notice, "MIT License, Copyright (c) 2026 Baidu");
        assert_eq!(std::fs::read(&outcome.focrq_path).unwrap(), artifact);
        assert_eq!(std::fs::read(&outcome.tokenizer_path).unwrap(), tokenizer);
        assert_eq!(
            entry_names(&cache),
            BTreeSet::from([V2.to_owned(), package_lock_name()]),
            "no stage or aside copy outlives a successful install"
        );
        assert_eq!(
            entry_names(&package),
            BTreeSet::from([v2_artifact_name(), "tokenizer.json".to_owned()])
        );

        let verified_parts: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                PullEvent::Part {
                    expected_sha256,
                    computed_sha256,
                    error: None,
                    ..
                } => Some((expected_sha256, computed_sha256)),
                _ => None,
            })
            .collect();
        assert_eq!(verified_parts.len(), 4, "3 artifact parts + 1 tokenizer part");
        assert!(
            verified_parts
                .iter()
                .all(|(expected, computed)| computed.as_deref() == Some(expected.as_str()))
        );
        let whole_files = events
            .iter()
            .filter(|event| {
                matches!(event, PullEvent::File { expected_sha256, computed_sha256, .. }
                    if expected_sha256 == computed_sha256)
            })
            .count();
        assert_eq!(whole_files, 2);
        assert_eq!(outcomes(&events), [TransactionOutcome::Installed]);
        let part_line = events
            .iter()
            .find(|event| matches!(event, PullEvent::Part { .. }))
            .expect("a part step")
            .to_json();
        assert_eq!(part_line["event"], "part");
        assert_eq!(part_line["verified"], true);
        assert_eq!(part_line["expected_sha256"], part_line["computed_sha256"]);

        let requests = source.request_count();
        let (again, events) = pull_offline(&manifest, Some(V2), &cache, &source);
        assert!(again.expect("cached pull").from_cache);
        assert_eq!(outcomes(&events), [TransactionOutcome::Cached]);
        assert_eq!(source.request_count(), requests, "a cache hit requests nothing");
        std::fs::remove_dir_all(root).expect("remove test cache");
    }

    /// U2-E4: a corrupted part is refused and nothing is installed; a second
    /// mirror with the right bytes then completes the same pull, rolling the
    /// file back to the part boundary first.
    #[test]
    fn offline_v2_pull_refuses_a_corrupt_part_and_installs_nothing() {
        let (root, cache) = temp_cache("v2_corrupt");
        let artifact = pattern(1000, 11);
        let (mut manifest, mut bodies) = v2_manifest(&artifact, b"{}");
        let part_url = manifest.models[V2].quants[DEFAULT_QUANT].focrq.parts[1].urls[0].clone();
        let good = bodies[&part_url].clone();
        bodies.get_mut(&part_url).expect("part body")[5] ^= 0xff;

        let source = MemoryParts::new(bodies.clone());
        let (result, events) = pull_offline(&manifest, Some(V2), &cache, &source);
        result.expect_err("a corrupt part fails the pull");
        assert!(!cache.join(V2).exists(), "nothing installed");
        assert_eq!(entry_names(&cache), BTreeSet::from([package_lock_name()]));
        assert!(events.iter().any(|event| matches!(event,
            PullEvent::Part { part: 2, error: Some(error), computed_sha256: Some(_), .. }
                if error.contains("sha256 mismatch"))));
        assert_eq!(outcomes(&events), [TransactionOutcome::RolledBack]);

        let mirror = "https://mirror2.invalid/part01".to_owned();
        bodies.insert(mirror.clone(), good);
        manifest
            .models
            .get_mut(V2)
            .and_then(|entry| entry.quants.get_mut(DEFAULT_QUANT))
            .expect("v2 int8")
            .focrq
            .parts[1]
            .urls
            .push(mirror);
        let source = MemoryParts::new(bodies);
        let (outcome, events) = pull_offline(&manifest, Some(V2), &cache, &source);
        let outcome = outcome.expect("the second mirror completes the pull");
        assert_eq!(std::fs::read(outcome.focrq_path).unwrap(), artifact);
        assert_eq!(outcomes(&events), [TransactionOutcome::Installed]);
        std::fs::remove_dir_all(root).expect("remove test cache");
    }

    /// U2-E4 / bd-jprb: an install that fails keeps the previous package
    /// byte-identical; a crash between the commit's renames (previous package
    /// aside, none in place, a stage left over) is restored by the next pull
    /// before it does anything else; a crash after them leaves an aside copy
    /// the next pull removes. Files the manifest does not name (another quant)
    /// survive a real install.
    #[test]
    fn an_interrupted_v2_install_keeps_or_restores_the_previous_package() {
        let (root, cache) = temp_cache("v2_interrupted");
        let package = cache.join(V2);
        let old_artifact = pattern(900, 1);
        let (old_manifest, old_bodies) = v2_manifest(&old_artifact, b"{\"old\":1}");
        let old_source = MemoryParts::new(old_bodies);
        pull_offline(&old_manifest, Some(V2), &cache, &old_source)
            .0
            .expect("install the previous package");
        let other_quant = package.join(format!("{V2}.v0.1.0.int4.focrq"));
        std::fs::write(&other_quant, b"another quant").expect("write another quant");
        let snapshot = |dir: &Path| {
            entry_names(dir)
                .into_iter()
                .map(|name| {
                    let bytes = std::fs::read(dir.join(&name)).expect("read package file");
                    (name, bytes)
                })
                .collect::<BTreeMap<_, _>>()
        };
        let previous = snapshot(&package);

        // A download that fails part-way changes nothing.
        let new_artifact = pattern(1200, 2);
        let (new_manifest, mut new_bodies) = v2_manifest(&new_artifact, b"{\"new\":1}");
        let last_part = new_manifest.models[V2].quants[DEFAULT_QUANT].focrq.parts[2].urls[0].clone();
        let last_body = new_bodies.remove(&last_part).expect("last part body");
        let failing = MemoryParts::new(new_bodies.clone());
        let (result, events) = pull_offline(&new_manifest, Some(V2), &cache, &failing);
        result.expect_err("a missing part fails the pull");
        assert_eq!(snapshot(&package), previous);
        assert_eq!(
            entry_names(&cache),
            BTreeSet::from([V2.to_owned(), package_lock_name()])
        );
        assert_eq!(outcomes(&events), [TransactionOutcome::RolledBack]);

        // A crash between the commit's renames.
        let key = package_key(V2);
        let aside = cache.join(format!(".focr-prev-{key}.pkg"));
        let stage = cache.join(format!(".focr-stage-{key}-999999-0.pkg"));
        std::fs::create_dir(&stage).expect("abandoned stage");
        std::fs::write(stage.join(v2_artifact_name()), b"partial").expect("stage file");
        std::fs::rename(&package, &aside).expect("move the package aside");
        let (outcome, events) = pull_offline(&old_manifest, Some(V2), &cache, &old_source);
        assert!(outcome.expect("recovering pull").from_cache);
        assert_eq!(snapshot(&package), previous, "the previous package is back");
        assert_eq!(
            outcomes(&events),
            [
                TransactionOutcome::Recovered,
                TransactionOutcome::Recovered,
                TransactionOutcome::Cached
            ]
        );
        assert_eq!(
            entry_names(&cache),
            BTreeSet::from([V2.to_owned(), package_lock_name()])
        );

        // A crash after them: the aside copy is stale.
        std::fs::create_dir(&aside).expect("stale aside copy");
        std::fs::write(aside.join("tokenizer.json"), b"stale").expect("aside file");
        let (outcome, events) = pull_offline(&old_manifest, Some(V2), &cache, &old_source);
        assert!(outcome.expect("cleanup pull").from_cache);
        assert_eq!(
            outcomes(&events),
            [TransactionOutcome::Recovered, TransactionOutcome::Cached]
        );
        assert!(!aside.exists());

        // The real install: new artifact and tokenizer, the other quant kept.
        new_bodies.insert(last_part, last_body);
        let source = MemoryParts::new(new_bodies);
        let (outcome, events) = pull_offline(&new_manifest, Some(V2), &cache, &source);
        let outcome = outcome.expect("install the new package");
        assert_eq!(std::fs::read(&outcome.focrq_path).unwrap(), new_artifact);
        assert_eq!(std::fs::read(&outcome.tokenizer_path).unwrap(), b"{\"new\":1}");
        assert_eq!(std::fs::read(&other_quant).unwrap(), b"another quant");
        assert_eq!(outcomes(&events), [TransactionOutcome::Installed]);
        assert_eq!(
            entry_names(&cache),
            BTreeSet::from([V2.to_owned(), package_lock_name()])
        );
        std::fs::remove_dir_all(root).expect("remove test cache");
    }

    /// U2-E4: a local manifest whose v2 entry declares another recipe is
    /// refused before any part is requested or the cache is touched; the
    /// refusal is the only step.
    #[test]
    fn a_v2_recipe_mismatch_is_refused_before_any_request() {
        let (root, cache) = temp_cache("v2_recipe");
        let (mut manifest, bodies) = v2_manifest(&pattern(300, 3), b"{}");
        manifest
            .models
            .get_mut(V2)
            .and_then(|entry| entry.quants.get_mut(DEFAULT_QUANT))
            .expect("v2 int8")
            .recipe = "unlimited-ocr-2-full-int8-attn-int8-lmhead-int8-v1".into();
        let source = MemoryParts::new(bodies);
        let (result, events) = pull_offline(&manifest, Some(V2), &cache, &source);
        assert!(matches!(result, Err(FocrError::FormatMismatch(_))));
        assert_eq!(source.request_count(), 0);
        assert!(!root.exists(), "the cache is not created");
        assert!(matches!(
            events.as_slice(),
            [PullEvent::Transaction { target, outcome: TransactionOutcome::Refused, reason }]
                if *target == cache.join(V2) && reason.contains("requires")
        ));
    }

    /// bd-jprb: a second pull of a package another pull holds fails fast and
    /// changes nothing.
    #[test]
    fn a_held_package_lock_refuses_a_concurrent_pull() {
        let (root, cache) = temp_cache("v2_locked");
        std::fs::create_dir_all(&cache).expect("create cache");
        let (manifest, bodies) = v2_manifest(&pattern(300, 4), b"{}");
        let source = MemoryParts::new(bodies);
        let held = InstallLock::acquire(&cache.join(V2)).expect("hold the package lock");
        let (result, _) = pull_offline(&manifest, Some(V2), &cache, &source);
        assert!(matches!(result, Err(FocrError::Timeout(_))));
        assert_eq!(source.request_count(), 0);
        assert_eq!(entry_names(&cache), BTreeSet::from([package_lock_name()]));
        drop(held);
        std::fs::remove_dir_all(root).expect("remove test cache");
    }

    /// The primary model's flat layout: when one file fails to download, no
    /// file is renamed in, so the installed artifact keeps its previous bytes
    /// beside its previous tokenizer, and no staging file is left.
    #[test]
    fn a_failed_primary_download_leaves_every_flat_file_as_it_was() {
        let (root, cache) = temp_cache("flat");
        let mut bodies = BTreeMap::new();
        let mut manifest = tiny_manifest();
        manifest.quants.get_mut(DEFAULT_QUANT).expect("int8").focrq =
            split_file("unlimited-ocr.int8.focrq", &pattern(500, 5), 2, &mut bodies);
        manifest.tokenizer = split_file("tokenizer.json", b"{\"v\":1}", 1, &mut bodies);
        let source = MemoryParts::new(bodies);
        pull_offline(&manifest, None, &cache, &source)
            .0
            .expect("install the previous files");
        let installed = entry_names(&cache);

        let mut bodies = BTreeMap::new();
        manifest.quants.get_mut(DEFAULT_QUANT).expect("int8").focrq =
            split_file("unlimited-ocr.int8.focrq", &pattern(600, 6), 2, &mut bodies);
        manifest.tokenizer = split_file("tokenizer.json", b"{\"v\":2}", 1, &mut bodies);
        bodies.retain(|url, _| !url.contains("tokenizer"));
        let source = MemoryParts::new(bodies);
        let (result, events) = pull_offline(&manifest, None, &cache, &source);
        result.expect_err("the tokenizer is unreachable");
        assert_eq!(
            std::fs::read(cache.join("unlimited-ocr.int8.focrq")).unwrap(),
            pattern(500, 5),
            "the verified new artifact was not renamed in beside the old tokenizer"
        );
        assert_eq!(std::fs::read(cache.join("tokenizer.json")).unwrap(), b"{\"v\":1}");
        assert_eq!(entry_names(&cache), installed, "no staging file is left");
        assert_eq!(outcomes(&events), [TransactionOutcome::RolledBack]);
        std::fs::remove_dir_all(root).expect("remove test cache");
    }

    /// Every pull-owned hidden name maps to its lock; others do not parse.
    #[test]
    fn pull_state_names_map_to_their_lock() {
        let key = "c".repeat(32);
        let dir = Path::new("/cache/models");
        for (name, state) in [
            (
                format!(".focr-stage-{key}-12-3.partial"),
                PullState::FileStage { key: key.clone() },
            ),
            (
                format!(".focr-stage-{key}-12-3.pkg"),
                PullState::PackageStage { key: key.clone() },
            ),
            (
                format!(".focr-prev-{key}.pkg"),
                PullState::PackageAside { key: key.clone() },
            ),
        ] {
            assert_eq!(parse_pull_state(&name), Some(state), "{name}");
            assert_eq!(
                pull_lock_path_for_staging(&dir.join(&name)),
                Some(dir.join(format!(".focr-pull-{key}.lock"))),
                "{name}"
            );
        }
        for name in [
            format!(".focr-stage-{key}-12-3.tmp"),
            format!(".focr-prev-{key}"),
            ".focr-prev-short.pkg".to_owned(),
            format!(".focr-pull-{key}.lock"),
            "unlimited-ocr-2".to_owned(),
        ] {
            assert_eq!(parse_pull_state(&name), None, "{name}");
        }
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
            // Two sources, and the ORDER is the contract: HuggingFace is tried
            // first and GitHub is the fallback. GitHub Releases started
            // returning 503s under load (2026-08-12), which is what motivated
            // the mirror; a silent re-ordering here would put every download
            // back on the flaky source, so the order is pinned, not just the
            // membership.
            assert_eq!(part.urls.len(), 2);
            assert!(part.urls.iter().all(|url| url.ends_with(filename)));
            assert!(
                part.urls[0].starts_with("https://huggingface.co/")
                    && part.urls[0].contains("/resolve/"),
                "part {filename}: primary url is not the HuggingFace mirror: {}",
                part.urls[0]
            );
            assert!(
                part.urls[1].contains(&format!(
                    "/releases/download/v{UNLIMITED_OCR_ARTIFACT_VERSION}/"
                )),
                "part {filename}: fallback url is not the pinned GitHub release: {}",
                part.urls[1]
            );
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
