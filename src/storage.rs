//! bd-223.4: the `RunStore` — durable local run state + telemetry on
//! **fsqlite** (frankensqlite; NEVER `rusqlite` — AGENTS.md/G3), backing
//! `focr runs` and `focr sync export-jsonl|import-jsonl`.
//!
//! * `_meta` is a versioned single-row table (`schema_version`, `created_at`,
//!   `franken_ocr_version`, `model_version_tag`). Opening an older store runs
//!   FORWARD migrations to [`SCHEMA_VERSION`]; a store whose version exceeds
//!   the binary's is REFUSED with [`FocrError::FormatMismatch`] (exit 7) —
//!   forward-only, no downgrade, no compatibility shims.
//! * `runs` records one row per recognition run (uuid, timestamps, input,
//!   mode, quant, model tag, exit code, status). Recording is BEST-EFFORT
//!   telemetry: a store failure never fails the user's run (the CLI logs and
//!   continues).
//! * JSONL sync (this module — the audit story): `export_jsonl` writes the
//!   canonically-ordered run set atomically (temp file in the same dir →
//!   fsync → rename) under an exclusive `.lock` sentinel; `import_jsonl`
//!   replays records idempotently (`INSERT OR REPLACE` by `run_id`).
//!
//! Doctrine note: no rayon runs under any store lock — the store is plain
//! sequential I/O on the CLI thread.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use fsqlite::{Connection, SqliteValue};

use crate::error::{FocrError, FocrResult};

/// The store schema version THIS binary writes (and migrates up to).
pub const SCHEMA_VERSION: i64 = 1;

/// Env override for the store path (default `~/.cache/franken_ocr/runs.db`).
pub const RUN_STORE_ENV: &str = "FOCR_RUN_STORE";

/// One recorded run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunRecord {
    /// UUID v4, the primary key.
    pub run_id: String,
    /// Unix milliseconds.
    pub started_at: i64,
    /// Unix milliseconds; `None` while in flight (never, for Phase-recorded
    /// completions).
    pub finished_at: Option<i64>,
    /// The input path as given.
    pub input_path: String,
    /// The command mode (`ocr`, `convert`, …).
    pub mode: String,
    /// The precision actually run (`int8`, `f32`, `unknown`).
    pub quant: String,
    /// The model provenance tag (artifact `model_id` + source sha when
    /// known; `unknown` otherwise — populated from the truth pack).
    pub model_version_tag: String,
    /// The process exit code the run mapped to.
    pub exit_code: i64,
    /// `ok` | `error` | `cancelled`.
    pub status: String,
}

/// The open store.
pub struct RunStore {
    conn: Connection,
    path: PathBuf,
}

impl std::fmt::Debug for RunStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunStore")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

fn text(v: &SqliteValue) -> String {
    match v {
        SqliteValue::Text(s) => s.as_str().to_owned(),
        other => format!("{other:?}"),
    }
}

fn int(v: &SqliteValue) -> i64 {
    match v {
        SqliteValue::Integer(i) => *i,
        _ => 0,
    }
}

impl RunStore {
    /// Resolve the default store path: [`RUN_STORE_ENV`] else
    /// `~/.cache/franken_ocr/runs.db` (directories created).
    ///
    /// # Errors
    /// No resolvable home directory, or directory creation failure.
    pub fn default_path() -> FocrResult<PathBuf> {
        if let Some(p) = std::env::var_os(RUN_STORE_ENV) {
            return Ok(PathBuf::from(p));
        }
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .ok_or_else(|| FocrError::Other(anyhow::anyhow!("no HOME for the run store")))?;
        let dir = home.join(".cache").join("franken_ocr");
        std::fs::create_dir_all(&dir)
            .map_err(|e| FocrError::Other(anyhow::anyhow!("create {}: {e}", dir.display())))?;
        Ok(dir.join("runs.db"))
    }

    /// Open (creating/migrating as needed) the store at `path`.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] when the on-disk `schema_version`
    /// EXCEEDS this binary's (forward-only); any fsqlite failure as
    /// [`FocrError::Other`].
    pub fn open(path: &Path) -> FocrResult<Self> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|e| {
                FocrError::Other(anyhow::anyhow!("create {}: {e}", parent.display()))
            })?;
        }
        let conn = Connection::open(path.display().to_string())
            .map_err(|e| FocrError::Other(anyhow::anyhow!("fsqlite open: {e}")))?;
        let store = Self {
            conn,
            path: path.to_path_buf(),
        };
        store.init_or_migrate()?;
        Ok(store)
    }

    /// The store's on-disk path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn sql(&self, sql: &str) -> FocrResult<usize> {
        self.conn
            .execute(sql)
            .map_err(|e| FocrError::Other(anyhow::anyhow!("fsqlite execute: {e}: {sql}")))
    }

    fn init_or_migrate(&self) -> FocrResult<()> {
        let has_meta = !self
            .conn
            .query("SELECT name FROM sqlite_master WHERE type='table' AND name='_meta'")
            .map_err(|e| FocrError::Other(anyhow::anyhow!("fsqlite query: {e}")))?
            .is_empty();
        if !has_meta {
            self.sql(
                "CREATE TABLE _meta (\n\
                     schema_version INTEGER NOT NULL,\n\
                     created_at INTEGER NOT NULL,\n\
                     franken_ocr_version TEXT NOT NULL,\n\
                     model_version_tag TEXT NOT NULL)",
            )?;
            self.sql(
                "CREATE TABLE runs (\n\
                     run_id TEXT PRIMARY KEY,\n\
                     started_at INTEGER NOT NULL,\n\
                     finished_at INTEGER,\n\
                     input_path TEXT NOT NULL,\n\
                     mode TEXT NOT NULL,\n\
                     quant TEXT NOT NULL,\n\
                     model_version_tag TEXT NOT NULL,\n\
                     exit_code INTEGER NOT NULL,\n\
                     status TEXT NOT NULL)",
            )?;
            self.conn
                .execute_with_params(
                    "INSERT INTO _meta (schema_version, created_at, franken_ocr_version, \
                     model_version_tag) VALUES (?, ?, ?, ?)",
                    &[
                        SqliteValue::Integer(SCHEMA_VERSION),
                        SqliteValue::Integer(now_millis()),
                        SqliteValue::Text(env!("CARGO_PKG_VERSION").into()),
                        SqliteValue::Text("unknown".into()),
                    ],
                )
                .map_err(|e| FocrError::Other(anyhow::anyhow!("fsqlite insert _meta: {e}")))?;
            return Ok(());
        }
        let version = self.schema_version()?;
        if version > SCHEMA_VERSION {
            return Err(FocrError::FormatMismatch(format!(
                "run store {} has schema_version {version}, newer than this binary's \
                 {SCHEMA_VERSION} — upgrade focr (forward-only migrations, no downgrade)",
                self.path.display()
            )));
        }
        // Forward migrations land here as versions grow:
        //   1 -> 2: ALTER/backfill, then bump _meta.schema_version — always
        //   additive, applied in order, each step leaving a valid store.
        // (v1 is current; the match keeps the migration seam explicit.)
        match version {
            SCHEMA_VERSION => Ok(()),
            older => Err(FocrError::Other(anyhow::anyhow!(
                "run store schema_version {older} has no migration path (bug: \
                 versions below {SCHEMA_VERSION} must be handled here)"
            ))),
        }
    }

    /// The on-disk `_meta.schema_version`.
    ///
    /// # Errors
    /// An fsqlite failure or a malformed `_meta`.
    pub fn schema_version(&self) -> FocrResult<i64> {
        let rows = self
            .conn
            .query("SELECT schema_version FROM _meta")
            .map_err(|e| FocrError::Other(anyhow::anyhow!("fsqlite query _meta: {e}")))?;
        rows.first()
            .and_then(|r| r.get(0).map(int))
            .ok_or_else(|| FocrError::Other(anyhow::anyhow!("empty _meta")))
    }

    /// Insert (or replace, keyed by `run_id`) one run record.
    ///
    /// # Errors
    /// An fsqlite failure.
    pub fn insert_run(&self, r: &RunRecord) -> FocrResult<()> {
        self.conn
            .execute_with_params(
                "INSERT OR REPLACE INTO runs (run_id, started_at, finished_at, input_path, \
                 mode, quant, model_version_tag, exit_code, status) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                &[
                    SqliteValue::Text(r.run_id.as_str().into()),
                    SqliteValue::Integer(r.started_at),
                    r.finished_at
                        .map_or(SqliteValue::Null, SqliteValue::Integer),
                    SqliteValue::Text(r.input_path.as_str().into()),
                    SqliteValue::Text(r.mode.as_str().into()),
                    SqliteValue::Text(r.quant.as_str().into()),
                    SqliteValue::Text(r.model_version_tag.as_str().into()),
                    SqliteValue::Integer(r.exit_code),
                    SqliteValue::Text(r.status.as_str().into()),
                ],
            )
            .map_err(|e| FocrError::Other(anyhow::anyhow!("fsqlite insert run: {e}")))?;
        Ok(())
    }

    fn rows_to_records(rows: &[fsqlite::Row]) -> Vec<RunRecord> {
        rows.iter()
            .filter_map(|row| {
                let v = row.values();
                if v.len() < 9 {
                    return None;
                }
                Some(RunRecord {
                    run_id: text(&v[0]),
                    started_at: int(&v[1]),
                    finished_at: match &v[2] {
                        SqliteValue::Null => None,
                        other => Some(int(other)),
                    },
                    input_path: text(&v[3]),
                    mode: text(&v[4]),
                    quant: text(&v[5]),
                    model_version_tag: text(&v[6]),
                    exit_code: int(&v[7]),
                    status: text(&v[8]),
                })
            })
            .collect()
    }

    /// Query runs: by exact id, else the most recent `limit` (started_at
    /// descending, run_id as the tiebreak).
    ///
    /// # Errors
    /// An fsqlite failure.
    pub fn query(&self, id: Option<&str>, limit: i64) -> FocrResult<Vec<RunRecord>> {
        const COLS: &str = "run_id, started_at, finished_at, input_path, mode, quant, \
                            model_version_tag, exit_code, status";
        let rows = match id {
            Some(id) => self
                .conn
                .query_with_params(
                    &format!("SELECT {COLS} FROM runs WHERE run_id = ?"),
                    &[SqliteValue::Text(id.into())],
                )
                .map_err(|e| FocrError::Other(anyhow::anyhow!("fsqlite query runs: {e}")))?,
            None => self
                .conn
                .query_with_params(
                    &format!(
                        "SELECT {COLS} FROM runs ORDER BY started_at DESC, run_id DESC LIMIT ?"
                    ),
                    &[SqliteValue::Integer(limit.max(0))],
                )
                .map_err(|e| FocrError::Other(anyhow::anyhow!("fsqlite query runs: {e}")))?,
        };
        Ok(Self::rows_to_records(&rows))
    }

    /// Every run in CANONICAL export order (`run_id` ascending — byte-stable
    /// across stores regardless of insertion order).
    ///
    /// # Errors
    /// An fsqlite failure.
    pub fn all_runs_canonical(&self) -> FocrResult<Vec<RunRecord>> {
        let rows = self
            .conn
            .query(
                "SELECT run_id, started_at, finished_at, input_path, mode, quant, \
                 model_version_tag, exit_code, status FROM runs ORDER BY run_id ASC",
            )
            .map_err(|e| FocrError::Other(anyhow::anyhow!("fsqlite query runs: {e}")))?;
        Ok(Self::rows_to_records(&rows))
    }
}

/// Unix milliseconds now.
#[must_use]
pub fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

fn record_to_json(r: &RunRecord) -> serde_json::Value {
    serde_json::json!({
        "schema_version": SCHEMA_VERSION,
        "run_id": r.run_id,
        "started_at": r.started_at,
        "finished_at": r.finished_at,
        "input_path": r.input_path,
        "mode": r.mode,
        "quant": r.quant,
        "model_version_tag": r.model_version_tag,
        "exit_code": r.exit_code,
        "status": r.status,
    })
}

/// One JSONL line back to a record.
///
/// # Errors
/// Malformed JSON or a missing required field.
pub fn record_from_json(line: &str) -> FocrResult<RunRecord> {
    let v: serde_json::Value = serde_json::from_str(line)
        .map_err(|e| FocrError::FormatMismatch(format!("run JSONL line: {e}")))?;
    let s = |k: &str| -> FocrResult<String> {
        v[k].as_str().map(str::to_owned).ok_or_else(|| {
            FocrError::FormatMismatch(format!("run JSONL line missing string field {k:?}"))
        })
    };
    let i = |k: &str| -> FocrResult<i64> {
        v[k].as_i64()
            .ok_or_else(|| FocrError::FormatMismatch(format!("run JSONL line missing int {k:?}")))
    };
    Ok(RunRecord {
        run_id: s("run_id")?,
        started_at: i("started_at")?,
        finished_at: v["finished_at"].as_i64(),
        input_path: s("input_path")?,
        mode: s("mode")?,
        quant: s("quant")?,
        model_version_tag: s("model_version_tag")?,
        exit_code: i("exit_code")?,
        status: s("status")?,
    })
}

/// RAII exclusive-lock sentinel (`<path>.lock`, `create_new`): a concurrent
/// export fails FAST with a clear error instead of corrupting the audit file
/// (a stale lock is reported with its path so the operator can remove it).
struct LockFile(PathBuf);

impl LockFile {
    fn acquire(target: &Path) -> FocrResult<Self> {
        let lock = target.with_extension("jsonl.lock");
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock)
        {
            Ok(_) => Ok(Self(lock)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(FocrError::Other(anyhow::anyhow!(
                    "run-store sync lock held: {} (another export/import in progress? \
                     remove the file if it is stale)",
                    lock.display()
                )))
            }
            Err(e) => Err(FocrError::Other(anyhow::anyhow!(
                "acquire {}: {e}",
                lock.display()
            ))),
        }
    }
}

impl Drop for LockFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Export every run as canonical JSONL — locked, ATOMIC (temp file in the
/// same directory → fsync → rename; a crash never leaves a partial file at
/// `out`). Returns the number of records written.
///
/// # Errors
/// Lock contention, I/O, or an fsqlite failure.
pub fn export_jsonl(store: &RunStore, out: &Path) -> FocrResult<usize> {
    let _lock = LockFile::acquire(out)?;
    let records = store.all_runs_canonical()?;
    let tmp = out.with_extension("jsonl.tmp");
    {
        let mut f = std::fs::File::create(&tmp)
            .map_err(|e| FocrError::Other(anyhow::anyhow!("create {}: {e}", tmp.display())))?;
        for r in &records {
            let line = serde_json::to_string(&record_to_json(r))
                .map_err(|e| FocrError::Other(anyhow::anyhow!("serialize run: {e}")))?;
            writeln!(f, "{line}")
                .map_err(|e| FocrError::Other(anyhow::anyhow!("write {}: {e}", tmp.display())))?;
        }
        f.sync_all()
            .map_err(|e| FocrError::Other(anyhow::anyhow!("fsync {}: {e}", tmp.display())))?;
    }
    std::fs::rename(&tmp, out).map_err(|e| {
        FocrError::Other(anyhow::anyhow!(
            "rename {} -> {}: {e}",
            tmp.display(),
            out.display()
        ))
    })?;
    Ok(records.len())
}

/// Import (replay) a JSONL audit file — locked; records replace by `run_id`
/// (idempotent). Returns the number of records imported.
///
/// # Errors
/// Lock contention, I/O, a malformed line (fails LOUD with its line number),
/// or an fsqlite failure.
pub fn import_jsonl(store: &RunStore, input: &Path) -> FocrResult<usize> {
    let _lock = LockFile::acquire(input)?;
    let text = std::fs::read_to_string(input)
        .map_err(|e| FocrError::Other(anyhow::anyhow!("read {}: {e}", input.display())))?;
    let mut n = 0usize;
    for (idx, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record = record_from_json(line).map_err(|e| {
            FocrError::FormatMismatch(format!("{}:{}: {e}", input.display(), idx + 1))
        })?;
        store.insert_run(&record)?;
        n += 1;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_store(name: &str) -> (RunStore, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "focr-runstore-{name}-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("runs.db");
        (RunStore::open(&path).expect("store opens"), dir)
    }

    fn sample(n: u8) -> RunRecord {
        RunRecord {
            run_id: format!("00000000-0000-4000-8000-0000000000{n:02x}"),
            started_at: 1_700_000_000_000 + i64::from(n),
            finished_at: Some(1_700_000_000_500 + i64::from(n)),
            input_path: format!("/pages/page_{n}.png"),
            mode: "ocr".into(),
            quant: "int8".into(),
            model_version_tag: "unlimited-ocr@sha256:2bc48a7a1100".into(),
            exit_code: 0,
            status: "ok".into(),
        }
    }

    #[test]
    fn meta_schema_created_and_versioned() {
        let (store, _dir) = scratch_store("meta");
        assert_eq!(store.schema_version().unwrap(), SCHEMA_VERSION);
        // Reopen: no re-init, version stable.
        let path = store.path().to_path_buf();
        drop(store);
        let again = RunStore::open(&path).expect("reopen migrates/no-ops");
        assert_eq!(again.schema_version().unwrap(), SCHEMA_VERSION);
    }

    #[test]
    fn too_new_store_refused_with_format_mismatch() {
        let (store, _dir) = scratch_store("toonew");
        let path = store.path().to_path_buf();
        store
            .sql(&format!(
                "UPDATE _meta SET schema_version = {}",
                SCHEMA_VERSION + 1
            ))
            .unwrap();
        drop(store);
        let err = RunStore::open(&path).expect_err("newer store must refuse");
        assert!(matches!(err, FocrError::FormatMismatch(_)), "{err:?}");
        assert_eq!(err.exit_code(), 7, "FormatMismatch is exit 7");
    }

    #[test]
    fn run_insert_and_query_by_id_and_limit() {
        let (store, _dir) = scratch_store("query");
        for n in 0..5u8 {
            store.insert_run(&sample(n)).unwrap();
        }
        // By id.
        let one = store.query(Some(&sample(3).run_id), 20).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0], sample(3));
        // Recent-first with limit.
        let recent = store.query(None, 2).unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0], sample(4), "most recent first");
        assert_eq!(recent[1], sample(3));
        // Unknown id = empty, not an error.
        assert!(store.query(Some("nope"), 20).unwrap().is_empty());
    }

    #[test]
    fn sync_jsonl_roundtrip_and_atomicity() {
        let (store, dir) = scratch_store("sync");
        for n in [3u8, 0, 4, 1, 2] {
            store.insert_run(&sample(n)).unwrap();
        }
        let out = dir.join("audit.jsonl");
        let n = export_jsonl(&store, &out).expect("export");
        assert_eq!(n, 5);
        // No temp/lock residue (atomic rename + RAII lock release).
        assert!(!out.with_extension("jsonl.tmp").exists(), "no temp residue");
        assert!(!out.with_extension("jsonl.lock").exists(), "lock released");
        // Every line parses and carries schema_version.
        let text = std::fs::read_to_string(&out).unwrap();
        for line in text.lines() {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(v["schema_version"].as_i64(), Some(SCHEMA_VERSION));
        }
        // Import into a FRESH store: sets equal; re-export byte-stable.
        let (fresh, dir2) = scratch_store("sync2");
        let m = import_jsonl(&fresh, &out).expect("import");
        assert_eq!(m, 5);
        assert_eq!(
            fresh.all_runs_canonical().unwrap(),
            store.all_runs_canonical().unwrap()
        );
        let out2 = dir2.join("audit.jsonl");
        export_jsonl(&fresh, &out2).unwrap();
        assert_eq!(
            std::fs::read_to_string(&out).unwrap(),
            std::fs::read_to_string(&out2).unwrap(),
            "canonical export is byte-stable across stores"
        );
        // Held lock fails fast with a clear error.
        let _held = LockFile::acquire(&out).unwrap();
        let err = export_jsonl(&store, &out).expect_err("lock contention");
        assert!(format!("{err}").contains("lock held"), "{err}");
    }

    #[test]
    fn malformed_import_fails_loud_with_line_number() {
        let (store, dir) = scratch_store("badline");
        let bad = dir.join("bad.jsonl");
        std::fs::write(&bad, "{\"run_id\": 42}\n").unwrap();
        let err = import_jsonl(&store, &bad).expect_err("malformed line");
        assert!(matches!(err, FocrError::FormatMismatch(_)), "{err:?}");
        assert!(
            format!("{err}").contains(":1:"),
            "carries the line number: {err}"
        );
    }
}
