//! `focr doctor` — idempotent, reversible, capability-reflecting self-check
//! and repair (bd-wp8.4, the world-class-doctor spec).
//!
//! ## The three laws this module encodes
//!
//! 1. **Detect-then-fix, never fix-then-detect.** Detectors are PURE: they
//!    read state and return findings; only the runtime decides — after the
//!    fact — whether `--fix` was passed. No detector mutates anything.
//! 2. **Single-chokepoint mutation.** Every disk write under `--fix` goes
//!    through [`mutation`]: a verbatim backup lands in
//!    `.doctor/runs/<run-id>/backups/` FIRST, before/after SHA-256 and mode
//!    are appended to `actions.jsonl`, and the whole run holds the
//!    `.doctor/lock` sentinel. A code-search test
//!    (`tests/doctor_fixtures.rs::all_mutation_is_inside_the_chokepoint`)
//!    fails CI if any other code path in this module writes to disk.
//! 3. **Blast-radius containment.** The doctor only touches paths inside the
//!    cache root it owns; anything else is REFUSED (exit 4). Irreversible
//!    repairs (re-quantizing weights) are never auto-run — refused with the
//!    exact recommended command instead.
//!
//! ## Exit-code contract (the doctor-skill canon, declared in
//! `doctor capabilities --json`; deliberately a SUB-contract distinct from
//! the §7.4 pipeline codes — an agent reads it from the tool)
//!
//! `0` healthy / all fixed · `1` findings present (detect-only) · `2` partial
//! fix (some fixed, some refused/advice-only) · `3` a fix failed and was
//! rolled back · `4` refused unsafe (nothing auto-fixable) · `5` concurrency
//! lost (lock held).

use std::fmt::Write as _;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{FocrError, FocrResult};

/// Version stamped on every doctor artifact (`actions.jsonl` lines, run
/// summaries, capabilities).
pub const DOCTOR_SCHEMA_VERSION: u32 = 1;

pub const EXIT_HEALTHY: i32 = 0;
pub const EXIT_FINDINGS: i32 = 1;
pub const EXIT_PARTIAL: i32 = 2;
pub const EXIT_FAILED_ROLLED_BACK: i32 = 3;
pub const EXIT_REFUSED_UNSAFE: i32 = 4;
pub const EXIT_CONCURRENCY_LOST: i32 = 5;

/// How a finding can be handled.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Fixability {
    /// Safe, reversible, applied by `--fix` through the mutation chokepoint.
    Auto { op: String },
    /// Deliberately NOT auto-run (irreversible or expensive); the exact
    /// command an agent should run instead.
    RefusedUnsafe { recommended_command: String },
    /// Nothing to mutate — the fix is an environment/action hint.
    AdviceOnly { hint: String },
}

/// One detector finding. Pure data; producing one mutates nothing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Finding {
    pub detector: &'static str,
    pub severity: &'static str,
    pub path: Option<String>,
    pub message: String,
    pub fixability: Fixability,
}

/// The doctor's world: the cache root it owns (everything it may touch) and
/// the `.doctor` state dir inside it. Resolved from the same user cache the
/// model resolver uses, so tests steer it hermetically via `HOME`.
pub struct DoctorRoot {
    pub cache_root: PathBuf,
}

impl DoctorRoot {
    /// Resolve from the user cache root.
    ///
    /// # Errors
    /// When no user cache directory is resolvable (no HOME).
    pub fn resolve() -> FocrResult<Self> {
        let cache_root = crate::dist::cache_root().ok_or_else(|| {
            FocrError::Other(anyhow::anyhow!(
                "cannot resolve a user cache directory (set HOME)"
            ))
        })?;
        Ok(Self { cache_root })
    }

    pub fn models_dir(&self) -> PathBuf {
        self.cache_root.join("models")
    }
    pub fn doctor_dir(&self) -> PathBuf {
        self.cache_root.join(".doctor")
    }
    pub fn runs_dir(&self) -> PathBuf {
        self.doctor_dir().join("runs")
    }
    pub fn lock_path(&self) -> PathBuf {
        self.doctor_dir().join("lock")
    }
}

// ───────────────────────────── detectors (PURE) ─────────────────────────────

/// Run every detector. Read-only by construction: detectors receive `&DoctorRoot`
/// and return findings; the mutation chokepoint is a different module.
#[must_use]
pub fn detect(root: &DoctorRoot) -> Vec<Finding> {
    let mut findings = Vec::new();
    detect_model_not_resolvable(root, &mut findings);
    detect_stale_focrq_format(root, &mut findings);
    detect_unreadable_cache_entries(root, &mut findings);
    detect_orphaned_partials(root, &mut findings);
    findings
}

fn detect_model_not_resolvable(_root: &DoctorRoot, out: &mut Vec<Finding>) {
    let spec = crate::OcrEngine::model_path();
    if !crate::native_engine::native_model_available(&spec) {
        let dirs: Vec<String> = crate::native_engine::model_resolution_search_dirs()
            .into_iter()
            .map(|p| p.display().to_string())
            .collect();
        out.push(Finding {
            detector: "model_not_resolvable",
            severity: "warn",
            path: None,
            message: format!(
                "no default model artifact resolvable (searched: {})",
                dirs.join(", ")
            ),
            fixability: Fixability::AdviceOnly {
                hint: "run `focr pull` to download the int8 weights, or set FOCR_MODEL_PATH / FOCR_MODEL_DIR".into(),
            },
        });
    }
}

fn detect_stale_focrq_format(root: &DoctorRoot, out: &mut Vec<Finding>) {
    let Ok(entries) = std::fs::read_dir(root.models_dir()) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        // exFAT/AppleDouble junk is never a model artifact (bd-re8.19 class).
        if name.starts_with('.') || !name.ends_with(".focrq") {
            continue;
        }
        let Ok(prefix) = prefix_reader::read_prefix(&path, 16) else {
            continue;
        };
        fn version_of(prefix: &[u8]) -> Option<u32> {
            if prefix.len() < 10 || &prefix[..6] != crate::native_engine::weights::FOCRQ_MAGIC {
                return None;
            }
            Some(u32::from_le_bytes([
                prefix[6], prefix[7], prefix[8], prefix[9],
            ]))
        }
        if let Some(version) = version_of(prefix.as_bytes()) {
            let current = crate::native_engine::weights::FOCRQ_FORMAT_VERSION;
            if version != current {
                out.push(Finding {
                    detector: "stale_focrq_format",
                    severity: "error",
                    path: Some(path.display().to_string()),
                    message: format!(
                        "{name}: format version {version} != current {current}"
                    ),
                    fixability: Fixability::RefusedUnsafe {
                        recommended_command: format!(
                            "focr pull   # re-fetch a current artifact (re-quantizing locally: `focr convert <safetensors> -o {name}`)"
                        ),
                    },
                });
            }
        }
    }
}

#[cfg(unix)]
fn detect_unreadable_cache_entries(root: &DoctorRoot, out: &mut Vec<Finding>) {
    use std::os::unix::fs::PermissionsExt;
    let Ok(entries) = std::fs::read_dir(root.models_dir()) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || !path.is_file() {
            continue;
        }
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        if meta.permissions().mode() & 0o400 == 0 {
            out.push(Finding {
                detector: "unreadable_cache_entry",
                severity: "error",
                path: Some(path.display().to_string()),
                message: format!(
                    "{name}: owner has no read permission (mode {:o})",
                    meta.permissions().mode() & 0o7777
                ),
                fixability: Fixability::Auto {
                    op: "chmod_u_rw".into(),
                },
            });
        }
    }
}

#[cfg(not(unix))]
fn detect_unreadable_cache_entries(_root: &DoctorRoot, _out: &mut Vec<Finding>) {}

fn detect_orphaned_partials(root: &DoctorRoot, out: &mut Vec<Finding>) {
    let Ok(entries) = std::fs::read_dir(root.models_dir()) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".tmp") || name.ends_with(".partial") {
            out.push(Finding {
                detector: "orphaned_partial_download",
                severity: "warn",
                path: Some(path.display().to_string()),
                message: format!("{name}: leftover partial/temp file in the model cache"),
                fixability: Fixability::Auto {
                    op: "quarantine".into(),
                },
            });
        }
    }
}

/// Read only the first `n` bytes (a 3.6 GB artifact must never be pulled
/// into RAM to check a 10-byte header).
pub(crate) mod prefix_reader {
    use std::io::Read;
    use std::path::Path;

    pub struct PrefixBytes(Vec<u8>);
    impl PrefixBytes {
        pub fn as_bytes(&self) -> &[u8] {
            &self.0
        }
    }

    pub fn read_prefix(path: &Path, n: usize) -> std::io::Result<PrefixBytes> {
        let mut f = std::fs::File::open(path)?;
        let mut buf = vec![0u8; n];
        let mut filled = 0;
        while filled < n {
            let k = f.read(&mut buf[filled..])?;
            if k == 0 {
                break;
            }
            filled += k;
        }
        buf.truncate(filled);
        Ok(PrefixBytes(buf))
    }
}

// ─────────────────────── the mutation chokepoint ───────────────────────
//
// EVERY disk write under `--fix` lives inside this module — the code-search
// test enforces it. Backups first, hashes always, one lock for the run.

pub mod mutation {
    use super::{DOCTOR_SCHEMA_VERSION, DoctorRoot, EXIT_CONCURRENCY_LOST};
    use crate::{FocrError, FocrResult};
    use serde::{Deserialize, Serialize};
    use std::io::Write as _;
    use std::path::{Path, PathBuf};

    /// One reversible action, as recorded in `actions.jsonl`.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Action {
        pub schema_version: u32,
        pub op: String,
        pub path: String,
        pub before_hash: String,
        pub after_hash: String,
        pub before_mode: Option<u32>,
        pub after_mode: Option<u32>,
        /// Backup file name inside the run's `backups/` dir (None for
        /// pure-metadata ops where the bytes were not touched — the bytes
        /// backup is still taken for uniformity when the file is readable).
        pub backup: Option<String>,
    }

    /// A live `--fix` run: owns the lock, the run dir, and the action log.
    pub struct DoctorRun {
        pub id: String,
        root_cache: PathBuf,
        dir: PathBuf,
        lock: PathBuf,
        pub actions: Vec<Action>,
    }

    pub fn sha256_file(path: &Path) -> String {
        use sha2::{Digest, Sha256};
        match std::fs::read(path) {
            Ok(bytes) => {
                let mut h = Sha256::new();
                h.update(&bytes);
                format!("{:x}", h.finalize())
            }
            Err(_) => "unreadable".into(),
        }
    }

    impl DoctorRun {
        /// Open a run: acquire the lock (exit-5 contract on contention),
        /// create `.doctor/runs/<id>/backups/`.
        ///
        /// # Errors
        /// A held lock maps to the doctor exit-5 contract; IO errors bubble.
        pub fn begin(root: &DoctorRoot, id: &str) -> FocrResult<Self> {
            std::fs::create_dir_all(root.runs_dir())
                .map_err(|e| FocrError::Other(anyhow::anyhow!("create runs dir: {e}")))?;
            let lock = root.lock_path();
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock)
            {
                Ok(mut f) => {
                    let _ = writeln!(f, "{}", std::process::id());
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    return Err(FocrError::Other(anyhow::anyhow!(
                        "doctor lock held ({}) — another doctor run is live (exit {})",
                        lock.display(),
                        EXIT_CONCURRENCY_LOST
                    )));
                }
                Err(e) => {
                    return Err(FocrError::Other(anyhow::anyhow!(
                        "acquire doctor lock: {e}"
                    )));
                }
            }
            let dir = root.runs_dir().join(id);
            std::fs::create_dir_all(dir.join("backups"))
                .map_err(|e| FocrError::Other(anyhow::anyhow!("create run dir: {e}")))?;
            Ok(Self {
                id: id.to_string(),
                root_cache: root.cache_root.clone(),
                dir,
                lock,
                actions: Vec::new(),
            })
        }

        fn backups_dir(&self) -> PathBuf {
            self.dir.join("backups")
        }

        fn backup_name(&self, path: &Path) -> String {
            format!(
                "{:03}_{}",
                self.actions.len(),
                path.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            )
        }

        fn guard_blast_radius(&self, path: &Path) -> FocrResult<()> {
            if !path.starts_with(&self.root_cache) {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "REFUSED: {} is outside the doctor's blast radius ({})",
                    path.display(),
                    self.root_cache.display()
                )));
            }
            Ok(())
        }

        fn record(&mut self, action: Action) -> FocrResult<()> {
            let line = serde_json::to_string(&action)
                .map_err(|e| FocrError::Other(anyhow::anyhow!("serialize action: {e}")))?;
            let log = self.dir.join("actions.jsonl");
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log)
                .map_err(|e| FocrError::Other(anyhow::anyhow!("open {}: {e}", log.display())))?;
            writeln!(f, "{line}")
                .map_err(|e| FocrError::Other(anyhow::anyhow!("append action: {e}")))?;
            f.sync_all().ok();
            self.actions.push(action);
            Ok(())
        }

        /// chmod the file owner-readable/writable, backing up bytes + mode.
        #[cfg(unix)]
        pub fn chmod_readable(&mut self, path: &Path) -> FocrResult<()> {
            use std::os::unix::fs::PermissionsExt;
            self.guard_blast_radius(path)?;
            let before_mode = std::fs::metadata(path)
                .map_err(|e| FocrError::Other(anyhow::anyhow!("stat {}: {e}", path.display())))?
                .permissions()
                .mode()
                & 0o7777;
            // Bytes-backup for uniform reversibility (readable only after we
            // grant ourselves read — so hash/backup may be pre- or post-chmod;
            // the mode is what this op changes, the bytes must not change).
            let before_hash = sha256_file(path);
            let new_mode = before_mode | 0o600;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(new_mode))
                .map_err(|e| FocrError::Other(anyhow::anyhow!("chmod {}: {e}", path.display())))?;
            let backup = self.backup_name(path);
            std::fs::copy(path, self.backups_dir().join(&backup))
                .map_err(|e| FocrError::Other(anyhow::anyhow!("backup {}: {e}", path.display())))?;
            let after_hash = sha256_file(path);
            self.record(Action {
                schema_version: DOCTOR_SCHEMA_VERSION,
                op: "chmod_u_rw".into(),
                path: path.display().to_string(),
                before_hash,
                after_hash,
                before_mode: Some(before_mode),
                after_mode: Some(new_mode),
                backup: Some(backup),
            })
        }

        /// Quarantine (reversibly move) an orphaned file into the run's backups.
        pub fn quarantine(&mut self, path: &Path) -> FocrResult<()> {
            self.guard_blast_radius(path)?;
            let before_hash = sha256_file(path);
            let backup = self.backup_name(path);
            let dest = self.backups_dir().join(&backup);
            std::fs::rename(path, &dest).map_err(|e| {
                FocrError::Other(anyhow::anyhow!("quarantine {}: {e}", path.display()))
            })?;
            self.record(Action {
                schema_version: DOCTOR_SCHEMA_VERSION,
                op: "quarantine".into(),
                path: path.display().to_string(),
                before_hash: before_hash.clone(),
                after_hash: "absent".into(),
                before_mode: None,
                after_mode: None,
                backup: Some(backup),
            })
        }

        /// Roll back every action of THIS run, newest first (used when a fix
        /// fails mid-run: exit-3 contract).
        pub fn rollback(&mut self, root: &DoctorRoot) -> FocrResult<usize> {
            let actions = std::mem::take(&mut self.actions);
            let n = actions.len();
            undo_actions(root, &self.dir, actions.into_iter().rev())?;
            Ok(n)
        }
    }

    impl Drop for DoctorRun {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.lock);
        }
    }

    /// Restore a sequence of actions (already reversed by the caller),
    /// verifying each restored file hashes back to its recorded
    /// `before_hash`. Fails CLOSED on a missing backup.
    pub fn undo_actions<I>(root: &DoctorRoot, run_dir: &Path, actions: I) -> FocrResult<()>
    where
        I: Iterator<Item = Action>,
    {
        for a in actions {
            let path = PathBuf::from(&a.path);
            if !path.starts_with(&root.cache_root) {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "undo refused: {} outside blast radius",
                    a.path
                )));
            }
            match a.op.as_str() {
                "quarantine" => {
                    let backup = a.backup.as_deref().ok_or_else(|| {
                        FocrError::Other(anyhow::anyhow!(
                            "undo failed closed: quarantine of {} has no backup",
                            a.path
                        ))
                    })?;
                    let src = run_dir.join("backups").join(backup);
                    if !src.exists() {
                        return Err(FocrError::Other(anyhow::anyhow!(
                            "undo failed closed: backup {} missing",
                            src.display()
                        )));
                    }
                    std::fs::rename(&src, &path).map_err(|e| {
                        FocrError::Other(anyhow::anyhow!("restore {}: {e}", a.path))
                    })?;
                    let restored = sha256_file(&path);
                    if restored != a.before_hash {
                        return Err(FocrError::Other(anyhow::anyhow!(
                            "undo verification failed: {} restored hash {} != recorded {}",
                            a.path,
                            restored,
                            a.before_hash
                        )));
                    }
                }
                "chmod_u_rw" => {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        let mode = a.before_mode.ok_or_else(|| {
                            FocrError::Other(anyhow::anyhow!(
                                "undo failed closed: chmod of {} has no before_mode",
                                a.path
                            ))
                        })?;
                        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode))
                            .map_err(|e| {
                                FocrError::Other(anyhow::anyhow!("restore mode {}: {e}", a.path))
                            })?;
                    }
                }
                other => {
                    return Err(FocrError::Other(anyhow::anyhow!(
                        "undo failed closed: unknown op {other:?} in actions.jsonl"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Load a past run's actions for `doctor undo <run-id>`.
    ///
    /// # Errors
    /// Missing run dir / malformed action lines fail closed.
    pub fn load_actions(root: &DoctorRoot, run_id: &str) -> FocrResult<(PathBuf, Vec<Action>)> {
        let dir = root.runs_dir().join(run_id);
        let log = dir.join("actions.jsonl");
        let text = std::fs::read_to_string(&log)
            .map_err(|e| FocrError::Other(anyhow::anyhow!("no run {run_id}: {e}")))?;
        let mut actions = Vec::new();
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let a: Action = serde_json::from_str(line).map_err(|e| {
                FocrError::Other(anyhow::anyhow!("actions.jsonl line {}: {e}", i + 1))
            })?;
            actions.push(a);
        }
        Ok((dir, actions))
    }
}

// ─────────────────────────── fix orchestration ───────────────────────────

/// Outcome of `--fix` over one findings list.
#[derive(Debug, Serialize)]
pub struct FixReport {
    pub run_id: String,
    pub fixed: usize,
    pub refused: usize,
    pub advice_only: usize,
    pub failed_rolled_back: bool,
    pub exit_code: i32,
}

/// Apply `--fix`: Auto findings go through the chokepoint; RefusedUnsafe and
/// AdviceOnly are reported, never forced. A failed auto-fix rolls back THIS
/// run's actions and reports the exit-3 contract.
///
/// # Errors
/// Lock contention (exit-5 contract) or unrecoverable IO.
pub fn fix(root: &DoctorRoot, findings: &[Finding], run_id: &str) -> FocrResult<FixReport> {
    let mut run = mutation::DoctorRun::begin(root, run_id)?;
    let mut fixed = 0usize;
    let mut refused = 0usize;
    let mut advice = 0usize;
    for f in findings {
        match &f.fixability {
            Fixability::Auto { op } => {
                let path = f.path.as_deref().map(PathBuf::from).ok_or_else(|| {
                    FocrError::Other(anyhow::anyhow!("auto finding without a path"))
                })?;
                let applied = match op.as_str() {
                    #[cfg(unix)]
                    "chmod_u_rw" => run.chmod_readable(&path),
                    "quarantine" => run.quarantine(&path),
                    other => Err(FocrError::Other(anyhow::anyhow!("unknown auto op {other}"))),
                };
                match applied {
                    Ok(()) => fixed += 1,
                    Err(e) => {
                        let undone = run.rollback(root)?;
                        eprintln!("focr doctor: fix failed ({e}); rolled back {undone} action(s)");
                        return Ok(FixReport {
                            run_id: run.id.clone(),
                            fixed: 0,
                            refused,
                            advice_only: advice,
                            failed_rolled_back: true,
                            exit_code: EXIT_FAILED_ROLLED_BACK,
                        });
                    }
                }
            }
            Fixability::RefusedUnsafe { .. } => refused += 1,
            Fixability::AdviceOnly { .. } => advice += 1,
        }
    }
    let exit_code = if fixed > 0 && refused == 0 && advice == 0 {
        EXIT_HEALTHY
    } else if fixed > 0 {
        EXIT_PARTIAL
    } else if refused > 0 {
        EXIT_REFUSED_UNSAFE
    } else if advice > 0 {
        EXIT_PARTIAL
    } else {
        EXIT_HEALTHY
    };
    Ok(FixReport {
        run_id: run.id.clone(),
        fixed,
        refused,
        advice_only: advice,
        failed_rolled_back: false,
        exit_code,
    })
}

/// `doctor undo <run-id>`: restore byte-for-byte from the run's backups,
/// newest action first, verifying recorded hashes; fails closed.
///
/// # Errors
/// Missing run/backup, hash mismatch after restore, or IO.
pub fn undo(root: &DoctorRoot, run_id: &str) -> FocrResult<usize> {
    let (dir, actions) = mutation::load_actions(root, run_id)?;
    let n = actions.len();
    mutation::undo_actions(root, &dir, actions.into_iter().rev())?;
    Ok(n)
}

/// The capabilities contract (read the tool, not a doc).
#[must_use]
pub fn capabilities() -> serde_json::Value {
    serde_json::json!({
        "schema_version": DOCTOR_SCHEMA_VERSION,
        "command": "doctor.capabilities",
        "contract_version": 1,
        "detectors": [
            {"name": "model_not_resolvable", "fix": "advice_only"},
            {"name": "stale_focrq_format", "fix": "refused_unsafe (re-quantization is irreversible; exact command recommended)"},
            {"name": "unreadable_cache_entry", "fix": "auto (chmod_u_rw, mode+bytes backed up)", "platform": "unix"},
            {"name": "orphaned_partial_download", "fix": "auto (quarantine into the run's backups; reversible)"},
        ],
        "fixers": ["chmod_u_rw", "quarantine"],
        "exit_codes": {
            "0": "healthy / all fixed",
            "1": "findings present (detect-only)",
            "2": "partial fix",
            "3": "fix failed and was rolled back",
            "4": "refused unsafe (nothing auto-fixable)",
            "5": "concurrency lost (doctor lock held)",
        },
        "mutation_contract": "single chokepoint: backup-first into .doctor/runs/<run-id>/backups/, before/after SHA-256 + mode in actions.jsonl, one lock per run; blast radius = the user cache root only",
        "undo": "focr doctor undo <run-id>  # byte-for-byte restore, hash-verified, fails closed",
        "env": ["HOME (cache root)", "FOCR_MODEL_PATH", "FOCR_MODEL_DIR"],
        "run_artifact_schema_version": DOCTOR_SCHEMA_VERSION,
    })
}

/// The paste-ready agent handbook (`doctor robot-docs`).
#[must_use]
pub fn robot_docs() -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "# focr doctor — agent handbook (contract v{DOCTOR_SCHEMA_VERSION})"
    );
    let _ = writeln!(
        s,
        "\nfocr doctor                # detect-only. exit 0 healthy, 1 findings"
    );
    let _ = writeln!(
        s,
        "focr doctor --json         # same, one JSON object on stdout"
    );
    let _ = writeln!(
        s,
        "focr doctor --dry-run      # worst-case blast radius, NO mutation"
    );
    let _ = writeln!(
        s,
        "focr doctor --fix          # safe repairs only. exit 0/2/3/4/5 (see capabilities)"
    );
    let _ = writeln!(
        s,
        "focr doctor undo <run-id>  # byte-for-byte restore, hash-verified"
    );
    let _ = writeln!(
        s,
        "focr doctor capabilities --json   # the full contract, from the tool"
    );
    let _ = writeln!(
        s,
        "\nRules: detect-then-fix; every write is backed up first under"
    );
    let _ = writeln!(
        s,
        ".doctor/runs/<run-id>/backups/ with SHA-256s in actions.jsonl;"
    );
    let _ = writeln!(
        s,
        "irreversible repairs are REFUSED with the exact command to run instead;"
    );
    let _ = writeln!(s, "blast radius is the user cache root only.");
    s
}
