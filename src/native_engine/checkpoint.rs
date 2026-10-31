//! Durable resume for windowed documents (U2-D4, plan D14): the resume key a
//! continuation must match, the checkpoint a finished window leaves, and the
//! journal of completed pages.
//!
//! A checkpoint is replaced after every window by an atomic write (a temp
//! sibling, fsync, rename, then the parent directory's fsync), so a crash
//! leaves the previous checkpoint or the new one, never a torn file. Its last
//! 32 bytes are the SHA-256 of everything before them, so a damaged file is
//! refused before any field is read.
//!
//! The journal is append-only NDJSON: the run's key, then per window its
//! completed pages and a line that closes them, fsynced per append. It holds
//! the document so far, so a checkpoint holds only what crosses the boundary
//! and stays the same size however long the document gets. A window's lines
//! are appended before the checkpoint that counts them is written, so after a
//! crash the journal holds at least the checkpoint's lines;
//! [`Journal::reopen`] cuts it back to that count.
//!
//! All of it happens between windows, on the document loop's thread, in plain
//! blocking `std::fs` calls ([`DocumentRun`]).

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
#[cfg(all(feature = "pdf", not(target_arch = "wasm32")))]
use std::time::Instant;

use sha2::{Digest, Sha256};
#[cfg(all(feature = "pdf", target_arch = "wasm32"))]
use web_time::Instant;

use super::tensor::Mat;
#[cfg(feature = "pdf")]
use super::window::{WindowBoundary, WindowRecord, WindowResume, WindowedDocument};
#[cfg(feature = "pdf")]
use super::{PageFinish, PageOrigin, ResumePolicy, timing_log};
use crate::error::{FocrError, FocrResult};
#[cfg(feature = "pdf")]
use crate::{
    page_source::{MappedPage, WalkCursor},
    pdf::SkippedPage,
};

/// The fields a resumed run must share with the run it continues, by name, so
/// a mismatch can say which fields differ. Keys are ordered, so the digest is
/// canonical.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResumeKey(BTreeMap<String, String>);

impl ResumeKey {
    /// An empty key.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// This key with `field` set to `value`.
    #[must_use]
    pub fn with(mut self, field: &str, value: impl ToString) -> Self {
        self.insert(field, value);
        self
    }

    /// Set `field` to `value`.
    pub fn insert(&mut self, field: &str, value: impl ToString) {
        self.0.insert(field.to_owned(), value.to_string());
    }

    /// `field`'s value.
    #[must_use]
    pub fn get(&self, field: &str) -> Option<&str> {
        self.0.get(field).map(String::as_str)
    }

    /// The lowercase hex SHA-256 of the key's canonical JSON: the name of a
    /// stdout run's directory.
    #[must_use]
    pub fn digest(&self) -> String {
        let json = serde_json::to_vec(&self.0).expect("a string map serializes");
        hex(&Sha256::digest(&json))
    }

    /// The fields whose values differ between the two keys, or that only
    /// one holds, in order.
    #[must_use]
    pub fn differences(&self, other: &Self) -> Vec<String> {
        let mut fields: Vec<String> = self
            .0
            .iter()
            .filter(|(k, v)| other.0.get(*k) != Some(v))
            .map(|(k, _)| k.clone())
            .collect();
        fields.extend(other.0.keys().filter(|k| !self.0.contains_key(*k)).cloned());
        fields.sort();
        fields
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// Checkpoint format: this magic, a little-endian u32 version, the key and
/// the state as u64-length-prefixed JSON, a u64 blob count and each blob
/// u64-length-prefixed, then the SHA-256 of all of it.
const MAGIC: &[u8; 8] = b"FOCRCKPT";
const VERSION: u32 = 1;
const DIGEST: usize = 32;

/// A window boundary's saved state: the [`ResumeKey`] it belongs to, the
/// driver's metadata as JSON, and binary payloads the metadata refers to by
/// index (KV tails in U2-B4's tail format, vision blocks as [`mat_blob`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Checkpoint {
    /// What a continuation must match.
    pub key: ResumeKey,
    /// The driver's state.
    pub state: serde_json::Value,
    /// Binary payloads, by index.
    pub blobs: Vec<Vec<u8>>,
}

impl Checkpoint {
    /// This checkpoint's bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let key = serde_json::to_vec(&self.key).expect("a string map serializes");
        let state = serde_json::to_vec(&self.state).expect("a JSON value serializes");
        let blob_bytes: usize = self.blobs.iter().map(|b| 8 + b.len()).sum();
        let mut out =
            Vec::with_capacity(8 + 4 + 8 + key.len() + 8 + state.len() + 8 + blob_bytes + DIGEST);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        for part in [&key, &state] {
            out.extend_from_slice(&(part.len() as u64).to_le_bytes());
            out.extend_from_slice(part);
        }
        out.extend_from_slice(&(self.blobs.len() as u64).to_le_bytes());
        for blob in &self.blobs {
            out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
            out.extend_from_slice(blob);
        }
        let digest = Sha256::digest(&out);
        out.extend_from_slice(&digest);
        out
    }

    /// The checkpoint in `bytes`.
    ///
    /// # Errors
    /// [`FocrError::Other`] when the SHA-256 does not match (a damaged or
    /// truncated file), or on a bad magic, version or length.
    pub fn decode(bytes: &[u8]) -> FocrResult<Self> {
        let bad = |why: &str| FocrError::Other(anyhow::anyhow!("document checkpoint: {why}"));
        if bytes.len() < MAGIC.len() + 4 + DIGEST || &bytes[..MAGIC.len()] != MAGIC {
            return Err(bad("not a checkpoint (bad magic or short file)"));
        }
        let (body, digest) = bytes.split_at(bytes.len() - DIGEST);
        if Sha256::digest(body).as_slice() != digest {
            return Err(bad("its SHA-256 does not match its contents"));
        }
        let mut cursor = Cursor {
            bytes: body,
            at: MAGIC.len(),
        };
        let version = cursor
            .take(4)
            .map(|w| u32::from_le_bytes(w.try_into().expect("4 bytes")));
        if version != Some(VERSION) {
            return Err(bad(&format!(
                "version {version:?}, this build reads {VERSION}"
            )));
        }
        let short = || bad("a length runs past the end");
        let key = cursor.part().ok_or_else(short)?;
        let key = serde_json::from_slice(key).map_err(|e| bad(&format!("key: {e}")))?;
        let state = cursor.part().ok_or_else(short)?;
        let state = serde_json::from_slice(state).map_err(|e| bad(&format!("state: {e}")))?;
        let count = cursor.length().ok_or_else(short)?;
        let mut blobs = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            blobs.push(cursor.part().ok_or_else(short)?.to_vec());
        }
        if cursor.at != body.len() {
            return Err(bad("bytes follow the last blob"));
        }
        Ok(Self { key, state, blobs })
    }

    /// Write this checkpoint to `path` atomically ([`write_atomic`]).
    ///
    /// # Errors
    /// Any I/O error; the previous file at `path` is then untouched.
    pub fn write(&self, path: &Path) -> FocrResult<()> {
        write_atomic(path, &self.encode())
    }

    /// The checkpoint at `path`.
    ///
    /// # Errors
    /// An I/O error, or [`Self::decode`]'s.
    pub fn read(path: &Path) -> FocrResult<Self> {
        Self::decode(&std::fs::read(path).map_err(io_error(path, "read"))?)
    }
}

/// A read position in a checkpoint's body.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    /// The next `len` bytes, or `None` past the end.
    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self
            .at
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len())?;
        let part = &self.bytes[self.at..end];
        self.at = end;
        Some(part)
    }

    /// A little-endian u64 length.
    fn length(&mut self) -> Option<usize> {
        let word = self.take(8)?;
        usize::try_from(u64::from_le_bytes(word.try_into().ok()?)).ok()
    }

    /// A length-prefixed part.
    fn part(&mut self) -> Option<&'a [u8]> {
        let len = self.length()?;
        self.take(len)
    }
}

/// An I/O error on `path` as a [`FocrError`], naming what failed.
fn io_error<'a>(
    path: &'a Path,
    what: &'static str,
) -> impl FnOnce(std::io::Error) -> FocrError + 'a {
    move |e| FocrError::Other(anyhow::anyhow!("could not {what} {}: {e}", path.display()))
}

/// `m` as a blob: rows and cols as little-endian u64, then its elements as
/// little-endian f32.
#[must_use]
pub fn mat_blob(m: &Mat) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + 4 * m.data.len());
    out.extend_from_slice(&(m.rows as u64).to_le_bytes());
    out.extend_from_slice(&(m.cols as u64).to_le_bytes());
    for x in &m.data {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// The matrix [`mat_blob`] wrote.
///
/// # Errors
/// [`FocrError::Other`] when the length does not match the shape.
pub fn blob_mat(bytes: &[u8]) -> FocrResult<Mat> {
    let bad = || {
        FocrError::Other(anyhow::anyhow!(
            "document checkpoint: a matrix blob is damaged"
        ))
    };
    if bytes.len() < 16 {
        return Err(bad());
    }
    let word = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"));
    let rows = usize::try_from(word(0)).map_err(|_| bad())?;
    let cols = usize::try_from(word(8)).map_err(|_| bad())?;
    let elements = rows.checked_mul(cols).ok_or_else(bad)?;
    if elements.checked_mul(4).and_then(|b| b.checked_add(16)) != Some(bytes.len()) {
        return Err(bad());
    }
    let data = bytes[16..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok(Mat::from_vec(rows, cols, data))
}

/// Replace `path` with `bytes` so that a crash leaves the old file or the new
/// one: write a temp sibling, fsync it, rename it over `path`, fsync the
/// directory.
///
/// # Errors
/// Any I/O error; the temp file is removed and `path` is untouched.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> FocrResult<()> {
    let temp = temp_sibling(path);
    let written = (|| -> std::io::Result<()> {
        let mut file = std::fs::File::create(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&temp, path)
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_file(&temp);
        return Err(io_error(path, "write")(e));
    }
    sync_parent(path);
    Ok(())
}

/// The temp sibling [`write_atomic`] writes before its rename.
#[must_use]
pub fn temp_sibling(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp-{}", std::process::id()));
    path.with_file_name(name)
}

/// Make a rename in `path`'s directory durable. Best effort: a filesystem
/// that cannot sync a directory has already ordered the rename.
fn sync_parent(path: &Path) {
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        let parent = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}

/// A document's completed pages, one JSON value per line (U2-D4's
/// `<out>.partial`).
#[derive(Debug)]
pub struct Journal {
    path: PathBuf,
    lines: usize,
}

impl Journal {
    /// A new, empty journal at `path`, replacing any file there.
    ///
    /// # Errors
    /// Any I/O error.
    pub fn create(path: &Path) -> FocrResult<Self> {
        std::fs::File::create(path)
            .and_then(|file| file.sync_all())
            .map_err(io_error(path, "create"))?;
        sync_parent(path);
        Ok(Self {
            path: path.to_path_buf(),
            lines: 0,
        })
    }

    /// The journal at `path`, cut back to its first `lines` lines: the
    /// pages its checkpoint counts. A crash between a journal append and
    /// the checkpoint that counts it leaves extra lines, which a resumed run
    /// writes again.
    ///
    /// # Errors
    /// [`FocrError::Other`] when it holds fewer than `lines` whole lines,
    /// or any I/O error.
    pub fn reopen(path: &Path, lines: usize) -> FocrResult<(Self, Vec<serde_json::Value>)> {
        let (mut values, ends) = whole_lines(path)?;
        if values.len() < lines {
            return Err(FocrError::Other(anyhow::anyhow!(
                "document journal {} holds {} whole lines, its checkpoint counts {lines}",
                path.display(),
                values.len()
            )));
        }
        values.truncate(lines);
        let end = lines.checked_sub(1).map_or(0, |last| ends[last]);
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .and_then(|file| {
                file.set_len(end as u64)?;
                file.sync_all()
            })
            .map_err(io_error(path, "truncate"))?;
        Ok((
            Self {
                path: path.to_path_buf(),
                lines,
            },
            values,
        ))
    }

    /// Append `values` as one line each, in one write, and fsync them.
    ///
    /// # Errors
    /// Any I/O error.
    pub fn append(&mut self, values: &[serde_json::Value]) -> FocrResult<()> {
        let mut bytes = Vec::new();
        for value in values {
            serde_json::to_writer(&mut bytes, value).expect("a JSON value serializes");
            bytes.push(b'\n');
        }
        std::fs::OpenOptions::new()
            .append(true)
            .open(&self.path)
            .and_then(|mut file| {
                file.write_all(&bytes)?;
                file.sync_all()
            })
            .map_err(io_error(&self.path, "append to"))?;
        self.lines += values.len();
        Ok(())
    }

    /// Lines written.
    #[must_use]
    pub fn lines(&self) -> usize {
        self.lines
    }

    /// The journal's path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// The whole lines of the journal at `path` and where each ends: a crash can
/// tear the last line, which is no line yet.
///
/// # Errors
/// A whole line that does not parse, or any I/O error.
fn whole_lines(path: &Path) -> FocrResult<(Vec<serde_json::Value>, Vec<usize>)> {
    let bytes = std::fs::read(path).map_err(io_error(path, "read"))?;
    let (mut values, mut ends) = (Vec::new(), Vec::new());
    let mut end = 0;
    for line in bytes.split_inclusive(|&b| b == b'\n') {
        if line.last() != Some(&b'\n') {
            break;
        }
        values.push(serde_json::from_slice(line).map_err(|e| {
            FocrError::Other(anyhow::anyhow!(
                "document journal {}: line {}: {e}",
                path.display(),
                values.len() + 1
            ))
        })?);
        end += line.len();
        ends.push(end);
    }
    Ok((values, ends))
}

/// The key fields that name the document: pages of another document are no
/// pages of this one, so even a recompute refuses them.
pub const DOCUMENT_FIELDS: [&str; 2] = [INPUT_FIELD, SELECTION_FIELD];

/// The key field holding the input's hash ([`RunFiles::key`]).
pub const INPUT_FIELD: &str = "input_sha256";

/// The key field holding the page selection ([`RunFiles::key`]).
pub const SELECTION_FIELD: &str = "page_selection";

/// Where a document run keeps what a later run resumes from (U2-D4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunFiles {
    /// The journal of completed pages (`<out>.partial`).
    pub journal: PathBuf,
    /// The latest checkpoint (`<out>.focr-ckpt`).
    pub checkpoint: PathBuf,
    /// The resume-key fields only the caller knows: [`INPUT_FIELD`] and
    /// [`SELECTION_FIELD`]. The library adds the model's and the decode's.
    pub key: ResumeKey,
    /// The command that continues this run, which the journal keeps for
    /// `focr doctor` to report.
    pub command: Option<String>,
}

/// The journal's name under a keyed run directory.
pub const KEYED_JOURNAL: &str = "run.partial";

/// The checkpoint's name under a keyed run directory.
pub const KEYED_CHECKPOINT: &str = "run.focr-ckpt";

impl RunFiles {
    /// A run whose output is the file `output`: `<output>.partial` and
    /// `<output>.focr-ckpt` beside it.
    #[must_use]
    pub fn beside(output: &Path, key: ResumeKey) -> Self {
        let sibling = |suffix: &str| {
            let mut name = output.as_os_str().to_owned();
            name.push(suffix);
            PathBuf::from(name)
        };
        Self {
            journal: sibling(".partial"),
            checkpoint: sibling(".focr-ckpt"),
            key,
            command: None,
        }
    }

    /// A run with no output file: under `root`, in a directory named by the
    /// digest of `key`, so a rerun of the same command finds it.
    #[must_use]
    pub fn keyed(root: &Path, key: ResumeKey) -> Self {
        let dir = root.join(key.digest());
        Self {
            journal: dir.join(KEYED_JOURNAL),
            checkpoint: dir.join(KEYED_CHECKPOINT),
            key,
            command: None,
        }
    }

    /// This run with its resume `command`.
    #[must_use]
    pub fn with_command(mut self, command: String) -> Self {
        self.command = Some(command);
        self
    }

    /// Drop the finished run's journal, and its checkpoint unless
    /// `keep_checkpoint`. A file that is already gone is fine. Call this once
    /// the output is written: until then these files are the run.
    ///
    /// # Errors
    /// Any other I/O error.
    pub fn remove(&self, keep_checkpoint: bool) -> FocrResult<()> {
        let mut files = vec![&self.journal];
        if !keep_checkpoint {
            files.push(&self.checkpoint);
        }
        for path in files {
            if let Err(e) = std::fs::remove_file(path)
                && e.kind() != std::io::ErrorKind::NotFound
            {
                return Err(io_error(path, "remove")(e));
            }
        }
        // A keyed run's directory goes with its last file.
        if let Some(dir) = self.journal.parent()
            && self.journal.file_name() == Some(KEYED_JOURNAL.as_ref())
        {
            let _ = std::fs::remove_dir(dir);
        }
        Ok(())
    }

    /// Whether a checkpoint is there for a later run.
    #[must_use]
    pub fn has_checkpoint(&self) -> bool {
        self.checkpoint.exists()
    }
}

/// What a document run keeps of a page it read: enough to walk past it again
/// without rasterizing it, and to rebuild its result.
#[cfg(feature = "pdf")]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(super) struct ReadPage {
    /// Where it came from.
    pub origin: PageOrigin,
    /// The raster's pixels, which its boxes map through.
    pub size: (u32, u32),
    /// Where the walk stood past it.
    pub walk: WalkCursor,
    /// The pages that failed to rasterize just before it (D24).
    pub skipped: Vec<SkippedPage>,
}

#[cfg(feature = "pdf")]
impl ReadPage {
    /// The source-page map's slots the walk filled up to this page.
    pub fn map_slots(&self) -> impl Iterator<Item = MappedPage> + '_ {
        let failed = self.skipped.iter().map(|page| MappedPage {
            source_page: page.page,
            half: None,
            size: None,
        });
        failed.chain(std::iter::once(MappedPage {
            source_page: self.origin.source_page,
            half: self.origin.half,
            size: Some(self.size),
        }))
    }
}

/// A completed page's journal line: one for each page a window owned. A page
/// its output never reached has no body, and no result.
#[cfg(feature = "pdf")]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(super) struct JournalPage {
    /// [`PageResult::run_index`](super::PageResult::run_index).
    pub run_index: usize,
    /// [`PageResult::window`](super::PageResult::window).
    pub window: usize,
    /// The page as it was read.
    pub read: ReadPage,
    /// The raw body. The page's text, layout and figures are pure functions
    /// of it and `read.size` ([`PageResult::read`](super::PageResult)).
    pub body: Option<String>,
    /// [`PageResult::tokens`](super::PageResult::tokens).
    pub tokens: Option<usize>,
    /// [`PageResult::finish`](super::PageResult::finish).
    pub finish: Option<PageFinish>,
}

/// One line of a document run's journal: the run's key, then per window its
/// pages and the record that closes them.
#[cfg(feature = "pdf")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum JournalLine {
    /// The key of the run that wrote the lines after it, and the command
    /// that continues it.
    Key {
        key: ResumeKey,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        command: Option<String>,
    },
    /// A completed page.
    Page(Box<JournalPage>),
    /// The window the page lines before it belong to.
    Window { record: Box<WindowRecord> },
}

#[cfg(feature = "pdf")]
impl JournalLine {
    fn to_value(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("a journal line serializes")
    }
}

/// A checkpoint's state.
#[cfg(feature = "pdf")]
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct RunState {
    /// The journal lines this checkpoint counts.
    journal_lines: usize,
    /// [`WindowBoundary::encode`]'s state.
    boundary: serde_json::Value,
    /// The boundary's carried page, read but owned by no window yet.
    carried_page: ReadPage,
}

/// What `focr doctor` reports of an interrupted run: a pure read of its
/// journal.
#[cfg(feature = "pdf")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalSummary {
    /// Pages of the closed windows.
    pub pages: usize,
    /// Closed windows.
    pub windows: usize,
    /// The command that continues the run.
    pub command: Option<String>,
}

/// The run journaled at `path`.
///
/// # Errors
/// A whole line that does not parse, or any I/O error.
#[cfg(feature = "pdf")]
pub fn journal_summary(path: &Path) -> FocrResult<JournalSummary> {
    let (lines, _) = whole_lines(path)?;
    let journal = replay(&lines)?;
    Ok(JournalSummary {
        pages: journal.pages.len(),
        windows: journal.doc.windows.len(),
        command: journal.command,
    })
}

/// A journal's lines read back.
#[cfg(feature = "pdf")]
struct Replay {
    /// The last key line's key.
    key: Option<ResumeKey>,
    /// Its command.
    command: Option<String>,
    /// The pages of the closed windows.
    pages: Vec<JournalPage>,
    /// The closed windows.
    doc: WindowedDocument,
    /// Lines through the last closed window.
    closed: usize,
}

#[cfg(feature = "pdf")]
fn replay(lines: &[serde_json::Value]) -> FocrResult<Replay> {
    let mut replay = Replay {
        key: None,
        command: None,
        pages: Vec::new(),
        doc: WindowedDocument::empty(),
        closed: 0,
    };
    let mut open = Vec::new();
    for (n, line) in lines.iter().enumerate() {
        let line: JournalLine = serde::Deserialize::deserialize(line).map_err(|e| {
            FocrError::Other(anyhow::anyhow!("document journal: line {}: {e}", n + 1))
        })?;
        match line {
            JournalLine::Key { key, command } => {
                replay.key = Some(key);
                replay.command = command;
                if replay.closed == n {
                    replay.closed = n + 1;
                }
            }
            JournalLine::Page(page) => open.push(*page),
            JournalLine::Window { record } => {
                let mut record = *record;
                record.bodies = open.iter().filter_map(|p| p.body.clone()).collect();
                replay.doc.absorb(record);
                replay.pages.append(&mut open);
                replay.closed = n + 1;
            }
        }
    }
    Ok(replay)
}

/// What a run picks up from the files of an interrupted one.
#[cfg(feature = "pdf")]
#[derive(Debug)]
pub(super) struct Resumed {
    /// The completed pages, a window's unreached pages included.
    pub pages: Vec<JournalPage>,
    /// The page read past them, when the next window starts with its block.
    pub carried: Option<ReadPage>,
    /// Where the decode continues.
    pub from: WindowResume,
    /// The next window recomputes, so the result will differ from an
    /// uninterrupted run's.
    pub recomputed: bool,
}

#[cfg(feature = "pdf")]
impl Resumed {
    /// Every page read so far, in order.
    pub fn read(&self) -> Vec<ReadPage> {
        let pages = self.pages.iter().map(|p| &p.read);
        pages.chain(&self.carried).cloned().collect()
    }
}

/// A document run's journal and checkpoint writer. A write that fails ends
/// the writing, not the decode: hours of pages are worth more than the
/// ability to resume them, and the run says so
/// ([`DocumentRun::failure`]).
#[cfg(feature = "pdf")]
#[derive(Debug)]
pub(super) struct DocumentRun {
    journal_path: PathBuf,
    checkpoint_path: PathBuf,
    key: ResumeKey,
    command: Option<String>,
    /// `None` until the first window ends, so a document that fits one
    /// window leaves no files.
    journal: Option<Journal>,
    failure: Option<String>,
}

#[cfg(feature = "pdf")]
impl DocumentRun {
    /// The run over `files` under the full resume `key`, and what it picks up
    /// from an interrupted run, by `policy`.
    ///
    /// # Errors
    /// [`FocrError::Usage`] when [`ResumePolicy::IfMatching`] finds an
    /// interrupted run it cannot continue byte-identically (a differing key,
    /// a damaged or missing checkpoint), naming why and offering a recompute;
    /// or when [`ResumePolicy::Recompute`] finds another document's journal.
    pub(super) fn open(
        files: &RunFiles,
        key: ResumeKey,
        policy: ResumePolicy,
    ) -> FocrResult<(Self, Option<Resumed>)> {
        let mut run = Self {
            journal_path: files.journal.clone(),
            checkpoint_path: files.checkpoint.clone(),
            key,
            command: files.command.clone(),
            journal: None,
            failure: None,
        };
        let resumed = match policy {
            ResumePolicy::Never => None,
            ResumePolicy::IfMatching => run.continue_checkpoint()?,
            ResumePolicy::Recompute => run.recompute_journal()?,
        };
        Ok((run, resumed))
    }

    /// The closed windows of the journal, if there is one.
    fn read_journal(&self) -> FocrResult<Option<Replay>> {
        if !self.journal_path.exists() {
            return Ok(None);
        }
        let (lines, _) = whole_lines(&self.journal_path)?;
        replay(&lines).map(Some)
    }

    fn refusal(&self, why: &str) -> FocrError {
        FocrError::Usage(format!(
            "{why}. `--resume=recompute` continues from the journal's completed pages with a \
             result that differs from an uninterrupted run's; remove {} and {} to start over",
            self.journal_path.display(),
            self.checkpoint_path.display()
        ))
    }

    /// [`ResumePolicy::IfMatching`]: the byte-identical continuation, or
    /// `None` when no window completed.
    fn continue_checkpoint(&mut self) -> FocrResult<Option<Resumed>> {
        // A checkpoint with no journal is what `--keep-checkpoint` left of a
        // finished run.
        if !self.journal_path.exists() {
            return Ok(None);
        }
        if !self.checkpoint_path.exists() {
            let windows = self
                .read_journal()
                .map_err(|e| self.refusal(&e.to_string()))?
                .map_or(0, |journal| journal.doc.windows.len());
            return if windows == 0 {
                Ok(None)
            } else {
                Err(self.refusal(&format!(
                    "{} holds {windows} completed windows, but their checkpoint {} is missing",
                    self.journal_path.display(),
                    self.checkpoint_path.display()
                )))
            };
        }
        let saved =
            Checkpoint::read(&self.checkpoint_path).map_err(|e| self.refusal(&e.to_string()))?;
        let differs = saved.key.differences(&self.key);
        if !differs.is_empty() {
            return Err(self.refusal(&format!(
                "the interrupted run in {} differs from this one in {}",
                self.checkpoint_path.display(),
                differs.join(", ")
            )));
        }
        let state: RunState = serde::Deserialize::deserialize(&saved.state)
            .map_err(|e| self.refusal(&format!("document checkpoint: {e}")))?;
        let (journal, lines) = Journal::reopen(&self.journal_path, state.journal_lines)
            .map_err(|e| self.refusal(&e.to_string()))?;
        let Replay {
            pages, doc, closed, ..
        } = replay(&lines).map_err(|e| self.refusal(&e.to_string()))?;
        if closed != lines.len() {
            return Err(self.refusal("the checkpoint counts a window the journal does not close"));
        }
        let from = WindowResume::from_checkpoint(&state.boundary, &saved.blobs, doc)
            .map_err(|e| self.refusal(&e.to_string()))?;
        self.journal = Some(journal);
        Ok(Some(Resumed {
            pages,
            carried: Some(state.carried_page),
            from,
            recomputed: false,
        }))
    }

    /// [`ResumePolicy::Recompute`]: the journal's closed windows and a next
    /// window with nothing retained, or `None` when no window completed.
    fn recompute_journal(&mut self) -> FocrResult<Option<Resumed>> {
        let Some(journal) = self.read_journal()? else {
            return Ok(None);
        };
        let Replay {
            key,
            pages,
            doc,
            closed,
            ..
        } = journal;
        if doc.windows.is_empty() {
            return Ok(None);
        }
        for field in DOCUMENT_FIELDS {
            let wrote = key.as_ref().and_then(|key| key.get(field));
            if wrote != self.key.get(field) {
                return Err(FocrError::Usage(format!(
                    "{} belongs to another document ({field} differs), so its pages cannot \
                     continue this one; remove it to start over",
                    self.journal_path.display()
                )));
            }
        }
        let (mut journal, _) = Journal::reopen(&self.journal_path, closed)?;
        journal.append(&[self.key_line()])?;
        self.journal = Some(journal);
        Ok(Some(Resumed {
            pages,
            carried: None,
            from: WindowResume::recompute(doc),
            recomputed: true,
        }))
    }

    fn key_line(&self) -> serde_json::Value {
        JournalLine::Key {
            key: self.key.clone(),
            command: self.command.clone(),
        }
        .to_value()
    }

    /// Record a window that ended at `boundary`: its `pages` and `record`
    /// in the journal, then the checkpoint that counts them. `carried` is
    /// the page the boundary holds.
    pub(super) fn window_end(
        &mut self,
        pages: Vec<JournalPage>,
        record: &WindowRecord,
        boundary: &WindowBoundary<'_>,
        carried: &ReadPage,
    ) {
        if self.failure.is_some() {
            return;
        }
        if let Err(e) = self.write_window(pages, record, boundary, carried) {
            timing_log(&format!("checkpoint: no more writes: {e}"));
            self.failure = Some(e.to_string());
        }
    }

    fn write_window(
        &mut self,
        pages: Vec<JournalPage>,
        record: &WindowRecord,
        boundary: &WindowBoundary<'_>,
        carried: &ReadPage,
    ) -> FocrResult<()> {
        let t0 = Instant::now();
        if self.journal.is_none() {
            if let Some(dir) = self.journal_path.parent().filter(|d| !d.exists()) {
                std::fs::create_dir_all(dir).map_err(io_error(dir, "create"))?;
            }
            let mut journal = Journal::create(&self.journal_path)?;
            journal.append(&[self.key_line()])?;
            self.journal = Some(journal);
        }
        let journal = self.journal.as_mut().expect("the journal was just opened");
        let mut lines: Vec<serde_json::Value> = pages
            .into_iter()
            .map(|page| JournalLine::Page(Box::new(page)).to_value())
            .collect();
        lines.push(
            JournalLine::Window {
                record: Box::new(record.clone()),
            }
            .to_value(),
        );
        journal.append(&lines)?;
        let (boundary_state, blobs) = boundary.encode();
        let state = RunState {
            journal_lines: journal.lines(),
            boundary: boundary_state,
            carried_page: carried.clone(),
        };
        let checkpoint = Checkpoint {
            key: self.key.clone(),
            state: serde_json::to_value(&state).expect("a run state serializes"),
            blobs,
        };
        let bytes = checkpoint.encode();
        write_atomic(&self.checkpoint_path, &bytes)?;
        timing_log(&format!(
            "checkpoint: window {} written, {} bytes, {:.1} ms",
            record.stats.index,
            bytes.len(),
            t0.elapsed().as_secs_f64() * 1e3
        ));
        Ok(())
    }

    /// Why the run stopped writing its files, if it did.
    pub(super) fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh directory under the system temp dir for one test.
    fn scratch(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "focr-checkpoint-{test}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn key() -> ResumeKey {
        ResumeKey::new()
            .with("artifact_sha256", "54b8a1ec")
            .with("input_sha256", "abc")
            .with("window", 10)
            .with("max_page_tokens", 20_000)
    }

    fn sample() -> Checkpoint {
        let m = Mat::from_vec(2, 3, vec![1.0, -2.5, 3.25, f32::MIN_POSITIVE, 0.0, -0.0]);
        Checkpoint {
            key: key(),
            state: serde_json::json!({"lo": 10, "window": 1, "pages": ["a", "b"]}),
            blobs: vec![mat_blob(&m), Vec::new(), vec![7; 1000]],
        }
    }

    #[test]
    fn every_key_field_changes_the_digest_and_is_named() {
        let base = key();
        for field in [
            "artifact_sha256",
            "input_sha256",
            "window",
            "max_page_tokens",
        ] {
            let mut other = base.clone();
            other.insert(field, "changed");
            assert_ne!(base.digest(), other.digest(), "{field}");
            assert_eq!(base.differences(&other), [field], "{field}");
        }
        let extra = base.clone().with("crop_mode", "base");
        assert_eq!(base.differences(&extra), ["crop_mode"]);
        assert_eq!(extra.differences(&base), ["crop_mode"]);
        assert!(base.differences(&base.clone()).is_empty());
        // Insertion order does not matter: the key is canonical.
        let reordered = ResumeKey::new()
            .with("max_page_tokens", 20_000)
            .with("window", 10)
            .with("input_sha256", "abc")
            .with("artifact_sha256", "54b8a1ec");
        assert_eq!(base.digest(), reordered.digest());
        assert_eq!(base.digest().len(), 64);
    }

    #[test]
    fn a_checkpoint_round_trips_and_its_matrix_is_bit_exact() {
        let original = sample();
        let back = Checkpoint::decode(&original.encode()).unwrap();
        assert_eq!(back, original);
        let m = blob_mat(&back.blobs[0]).unwrap();
        assert_eq!((m.rows, m.cols), (2, 3));
        let bits: Vec<u32> = m.data.iter().map(|x| x.to_bits()).collect();
        let want: Vec<u32> = [1.0f32, -2.5, 3.25, f32::MIN_POSITIVE, 0.0, -0.0]
            .iter()
            .map(|x| x.to_bits())
            .collect();
        assert_eq!(bits, want);
        assert!(blob_mat(&back.blobs[0][..20]).is_err());
    }

    #[test]
    fn any_flipped_or_missing_byte_is_refused() {
        let bytes = sample().encode();
        for at in (0..bytes.len()).step_by(37).chain([bytes.len() - 1]) {
            let mut bad = bytes.clone();
            bad[at] ^= 0x10;
            assert!(Checkpoint::decode(&bad).is_err(), "byte {at}");
        }
        for len in [0, 8, 12, bytes.len() - 1] {
            assert!(Checkpoint::decode(&bytes[..len]).is_err(), "length {len}");
        }
    }

    #[test]
    fn a_crash_before_the_rename_leaves_the_previous_checkpoint() {
        let dir = scratch("atomic");
        let path = dir.join("doc.md.focr-ckpt");
        let first = sample();
        first.write(&path).unwrap();
        // A crash after the temp file was written but before the rename:
        // the temp sibling holds part of the next checkpoint.
        let mut next = sample();
        next.state = serde_json::json!({"lo": 20});
        let partial = next.encode();
        std::fs::write(temp_sibling(&path), &partial[..partial.len() / 2]).unwrap();
        assert_eq!(Checkpoint::read(&path).unwrap(), first);
        // The next write replaces both.
        next.write(&path).unwrap();
        assert_eq!(Checkpoint::read(&path).unwrap(), next);
        assert!(!temp_sibling(&path).exists());
    }

    #[test]
    fn a_reopened_journal_is_cut_to_its_checkpoint_and_continues() {
        let dir = scratch("journal");
        let path = dir.join("doc.md.partial");
        let mut journal = Journal::create(&path).unwrap();
        for page in 1..=3 {
            journal
                .append(&[serde_json::json!({"page": page})])
                .unwrap();
        }
        assert_eq!(journal.lines(), 3);
        // A crash mid-append leaves a torn fourth line.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(b"{\"page\":");
        std::fs::write(&path, &bytes).unwrap();
        // The checkpoint counted two pages: the third is written again.
        let (mut journal, pages) = Journal::reopen(&path, 2).unwrap();
        assert_eq!(
            pages,
            [
                serde_json::json!({"page": 1}),
                serde_json::json!({"page": 2})
            ]
        );
        journal.append(&[serde_json::json!({"page": 3})]).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text, "{\"page\":1}\n{\"page\":2}\n{\"page\":3}\n");
        // A journal shorter than its checkpoint is refused.
        assert!(Journal::reopen(&path, 4).is_err());
    }

    #[cfg(feature = "pdf")]
    mod runs {
        use super::*;
        use crate::native_engine::window::WindowStats;
        use crate::native_engine::{FinishReason, Half};

        fn files(dir: &Path) -> RunFiles {
            RunFiles {
                journal: dir.join("doc.md.partial"),
                checkpoint: dir.join("doc.md.focr-ckpt"),
                key: ResumeKey::new(),
                command: Some("focr ocr doc.pdf -o doc.md --multi-page".into()),
            }
        }

        fn key(window: usize) -> ResumeKey {
            ResumeKey::new()
                .with(INPUT_FIELD, "abc")
                .with(SELECTION_FIELD, "all")
                .with("window", window)
        }

        fn read(page: usize) -> ReadPage {
            ReadPage {
                origin: PageOrigin {
                    source_page: page + 1,
                    half: (page == 1).then_some(Half::Left),
                },
                size: (640, 480),
                walk: WalkCursor {
                    next: page + 1,
                    held: page == 1,
                },
                skipped: if page == 1 {
                    vec![SkippedPage {
                        page: 9,
                        reason: "no raster".into(),
                    }]
                } else {
                    Vec::new()
                },
            }
        }

        fn page(page: usize, window: usize) -> JournalPage {
            JournalPage {
                run_index: page + 1,
                window,
                read: read(page),
                body: Some(format!("body {page}")),
                tokens: Some(10 + page),
                finish: Some(PageFinish::Marker),
            }
        }

        fn record(index: usize, lo: usize) -> WindowRecord {
            WindowRecord {
                stats: WindowStats {
                    index,
                    pages: lo..lo + 1,
                    lookahead: true,
                    reused: index > 0,
                    fallback_reason: None,
                    straddle: false,
                    prefill_tokens: 100,
                    copied_rows: 0,
                    forced_tokens: 0,
                    generated_tokens: 10 + lo,
                    finish: FinishReason::Stop,
                    kv_bytes_peak: 1000 + lo,
                    retire_fallbacks: 0,
                    elapsed_ms: 5,
                },
                ids: vec![7, 8, u32::try_from(lo).unwrap()],
                bodies: vec![format!("body {lo}")],
                tokens: vec![10 + lo],
                finish: vec![PageFinish::Marker],
                leading_text: lo == 1,
                extra_markers: lo,
                forced_steps: 1,
                hf_all_neg_inf_steps: 0,
                retired_blocks: 2,
                vision_encodes: 2,
            }
        }

        /// A run that finished `windows` one-page windows, then stopped.
        fn interrupted(dir: &Path, windows: usize) -> RunFiles {
            let files = files(dir);
            let (mut run, resumed) =
                DocumentRun::open(&files, key(1), ResumePolicy::Never).unwrap();
            assert!(resumed.is_none());
            for lo in 0..windows {
                let carried = Mat::from_vec(2, 2, vec![0.5, -0.0, f32::MIN_POSITIVE, 3.0]);
                let boundary = WindowBoundary::bare(lo + 1, lo + 1, &carried);
                run.window_end(
                    vec![page(lo, lo)],
                    &record(lo, lo),
                    &boundary,
                    &read(lo + 1),
                );
                assert_eq!(run.failure(), None);
            }
            files
        }

        fn refusal(result: FocrResult<(DocumentRun, Option<Resumed>)>) -> String {
            match result {
                Err(FocrError::Usage(why)) => why,
                other => panic!("not refused: {other:?}"),
            }
        }

        #[test]
        fn a_matching_run_gets_its_pages_back_and_the_next_window_starts_at_the_boundary() {
            let dir = scratch("resume");
            let files = interrupted(&dir, 2);
            let (mut run, resumed) =
                DocumentRun::open(&files, key(1), ResumePolicy::IfMatching).unwrap();
            let resumed = resumed.expect("an interrupted run");
            assert!(!resumed.recomputed);
            assert_eq!(resumed.pages, [page(0, 0), page(1, 1)]);
            assert_eq!(resumed.carried, Some(read(2)));
            assert_eq!(resumed.read(), [read(0), read(1), read(2)]);
            assert_eq!(resumed.from.next_page(), 2);
            let slots: Vec<MappedPage> = resumed
                .read()
                .iter()
                .flat_map(ReadPage::map_slots)
                .collect();
            assert_eq!(slots.len(), 4, "a failed page keeps its slot");
            assert_eq!(slots[1].source_page, 9);
            assert_eq!(slots[1].size, None);
            let doc = resumed.from.into_document();
            assert_eq!(doc.pages.bodies, ["body 0", "body 1"]);
            assert_eq!(doc.pages.tokens, [10, 11]);
            assert!(doc.pages.leading_text);
            assert_eq!(doc.pages.extra_markers, 1);
            assert_eq!(doc.stats.vision_encodes, 4);
            assert_eq!(doc.stats.retired_blocks, 4);
            assert_eq!(doc.stats.peak_kv_bytes, 1001);
            assert_eq!(doc.window_ids, [vec![7, 8, 0], vec![7, 8, 1]]);
            // The run goes on writing where the journal stopped.
            let carried = Mat::from_vec(1, 1, vec![1.0]);
            let boundary = WindowBoundary::bare(3, 3, &carried);
            run.window_end(vec![page(2, 2)], &record(2, 2), &boundary, &read(3));
            assert_eq!(run.failure(), None);
            let (_, resumed) = DocumentRun::open(&files, key(1), ResumePolicy::IfMatching).unwrap();
            let doc = resumed.expect("still resumable").from.into_document();
            assert_eq!(doc.pages.bodies.len(), 3);
        }

        #[test]
        fn lines_past_the_checkpoint_are_cut() {
            let dir = scratch("past");
            let files = interrupted(&dir, 2);
            // A crash after a third window's lines, before its checkpoint.
            let (mut journal, _) = Journal::reopen(&files.journal, 5).unwrap();
            journal
                .append(&[JournalLine::Page(Box::new(page(2, 2))).to_value()])
                .unwrap();
            let (_, resumed) = DocumentRun::open(&files, key(1), ResumePolicy::IfMatching).unwrap();
            assert_eq!(resumed.expect("resumable").pages.len(), 2);
            assert_eq!(
                whole_lines(&files.journal).unwrap().0.len(),
                5,
                "the key and two windows of two lines each"
            );
        }

        #[test]
        fn a_differing_key_is_refused_by_name_and_a_recompute_continues() {
            let dir = scratch("key");
            let files = interrupted(&dir, 2);
            let why = refusal(DocumentRun::open(&files, key(2), ResumePolicy::IfMatching));
            assert!(why.contains("differs from this one in window"), "{why}");
            assert!(why.contains("--resume=recompute"), "{why}");
            let (_, resumed) = DocumentRun::open(&files, key(2), ResumePolicy::Recompute).unwrap();
            let resumed = resumed.expect("the journal's pages");
            assert!(resumed.recomputed);
            assert_eq!(resumed.pages.len(), 2);
            assert_eq!(resumed.carried, None);
            assert_eq!(resumed.read(), [read(0), read(1)]);
            assert_eq!(resumed.from.next_page(), 2);
            // The next line says which key wrote what follows.
            let (lines, _) = whole_lines(&files.journal).unwrap();
            assert_eq!(replay(&lines).unwrap().key, Some(key(2)));
        }

        #[test]
        fn another_documents_journal_is_refused_even_a_recompute() {
            let dir = scratch("other");
            let files = interrupted(&dir, 1);
            for field in DOCUMENT_FIELDS {
                let other = key(1).with(field, "different");
                let why = refusal(DocumentRun::open(&files, other, ResumePolicy::Recompute));
                assert!(why.contains(field), "{why}");
            }
        }

        #[test]
        fn a_damaged_or_missing_checkpoint_offers_a_recompute() {
            let dir = scratch("damaged");
            let files = interrupted(&dir, 1);
            let mut bytes = std::fs::read(&files.checkpoint).unwrap();
            let middle = bytes.len() / 2;
            bytes[middle] ^= 1;
            std::fs::write(&files.checkpoint, &bytes).unwrap();
            let why = refusal(DocumentRun::open(&files, key(1), ResumePolicy::IfMatching));
            assert!(why.contains("--resume=recompute"), "{why}");
            std::fs::remove_file(&files.checkpoint).unwrap();
            let why = refusal(DocumentRun::open(&files, key(1), ResumePolicy::IfMatching));
            assert!(why.contains("is missing"), "{why}");
            let (_, resumed) = DocumentRun::open(&files, key(1), ResumePolicy::Recompute).unwrap();
            assert!(resumed.expect("the journal's pages").recomputed);
        }

        #[test]
        fn nothing_to_pick_up_is_a_fresh_run() {
            let dir = scratch("fresh");
            let files = files(&dir);
            for policy in [ResumePolicy::IfMatching, ResumePolicy::Recompute] {
                let (_, resumed) = DocumentRun::open(&files, key(1), policy).unwrap();
                assert!(resumed.is_none());
            }
            assert!(!files.journal.exists(), "opening writes nothing");
            let mut journal = Journal::create(&files.journal).unwrap();
            let line = JournalLine::Key {
                key: key(1),
                command: None,
            };
            journal.append(&[line.to_value()]).unwrap();
            for policy in [ResumePolicy::IfMatching, ResumePolicy::Recompute] {
                let (_, resumed) = DocumentRun::open(&files, key(1), policy).unwrap();
                assert!(resumed.is_none(), "no window completed");
            }
        }

        #[test]
        fn a_run_that_cannot_write_says_so_and_stops_trying() {
            let dir = scratch("unwritable");
            // A file where the run's directory would go.
            let blocked = dir.join("blocked");
            std::fs::write(&blocked, b"").unwrap();
            let files = RunFiles::beside(&blocked.join("doc.md"), ResumeKey::new());
            let (mut run, _) = DocumentRun::open(&files, key(1), ResumePolicy::Never).unwrap();
            let carried = Mat::from_vec(1, 1, vec![1.0]);
            let boundary = WindowBoundary::bare(1, 1, &carried);
            run.window_end(vec![page(0, 0)], &record(0, 0), &boundary, &read(1));
            let first = run.failure().expect("the write failed").to_owned();
            run.window_end(vec![page(1, 1)], &record(1, 1), &boundary, &read(2));
            assert_eq!(run.failure(), Some(first.as_str()));
        }

        #[test]
        fn a_finished_run_removes_its_files_and_can_keep_the_checkpoint() {
            let dir = scratch("finish");
            let files = interrupted(&dir, 1);
            assert!(files.has_checkpoint());
            files.remove(true).unwrap();
            assert!(!files.journal.exists());
            assert!(files.has_checkpoint());
            files.remove(false).unwrap();
            assert!(!files.has_checkpoint());
            files.remove(false).unwrap();
        }

        #[test]
        fn a_keyed_run_lives_in_its_keys_directory_and_leaves_with_it() {
            let dir = scratch("keyed");
            let files = RunFiles::keyed(&dir, key(1));
            assert_eq!(
                files.journal.parent(),
                Some(dir.join(key(1).digest()).as_path())
            );
            let (mut run, _) = DocumentRun::open(&files, key(1), ResumePolicy::Never).unwrap();
            let carried = Mat::from_vec(1, 1, vec![1.0]);
            let boundary = WindowBoundary::bare(1, 1, &carried);
            run.window_end(vec![page(0, 0)], &record(0, 0), &boundary, &read(1));
            assert_eq!(run.failure(), None, "the directory is made as needed");
            assert!(files.journal.exists());
            files.remove(false).unwrap();
            assert!(!dir.join(key(1).digest()).exists());
        }

        #[test]
        fn a_journal_summary_reads_a_run_without_touching_it() {
            let dir = scratch("summary");
            let files = interrupted(&dir, 2);
            let before = std::fs::read(&files.journal).unwrap();
            let summary = journal_summary(&files.journal).unwrap();
            assert_eq!(summary.pages, 2);
            assert_eq!(summary.windows, 2);
            assert_eq!(summary.command, files.command);
            assert_eq!(std::fs::read(&files.journal).unwrap(), before);
        }

        #[test]
        fn beside_names_the_output_and_never_replaces_its_extension() {
            let files = RunFiles::beside(Path::new("out/doc.v1.md"), ResumeKey::new());
            assert_eq!(files.journal, Path::new("out/doc.v1.md.partial"));
            assert_eq!(files.checkpoint, Path::new("out/doc.v1.md.focr-ckpt"));
        }
    }
}
