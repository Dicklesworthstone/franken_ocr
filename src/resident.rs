//! Resident engine: a per-user background process that keeps the loaded model
//! in memory so consecutive `focr ocr` invocations skip the multi-gigabyte
//! artifact load (GH #9 — "run as a server like llama.cpp").
//!
//! Shape: `focr ocr` connects to a loopback TCP daemon (spawned on demand from
//! the same binary) and sends one recognition request; the daemon holds the
//! hydrated [`OcrEngine`] (and therefore the cached `OcrModel`) between
//! requests and exits by itself after a configurable idle period (default ten
//! minutes). Everything else — argument handling, output writing, robot
//! events, run telemetry — stays in the client process, so the observable
//! contract of `focr ocr` is unchanged.
//!
//! Loopback TCP is the one transport that behaves identically on Linux, macOS,
//! and Windows with only the standard library. Access control is the state
//! file, not the port: the daemon binds an ephemeral 127.0.0.1 port and writes
//! `{port, token, …}` to a file only the invoking user can read (0600 on
//! Unix), and every request must present that token. Document images are
//! sensitive, so the daemon holds no request history.
//!
//! Failure philosophy (inherited from the frankentts resident engine this
//! ports): the resident path may only ever make `focr ocr` FASTER, never break
//! it. Every transport-level problem (no daemon, stale state file, version or
//! artifact mismatch, differing inference environment, malformed reply) falls
//! back to the classic in-process load. Only a genuine recognition error
//! crosses the wire as an error, carrying its stable exit-code class.
//!
//! Because decode/preprocess overrides are resolved at model load
//! (`decode_params_from_env` runs inside `OcrModel::load`), the daemon keys
//! its hydrated engine by an option fingerprint: a request with different
//! load-resolved options drops the resident engine and rehydrates, so a warm
//! answer is never served with stale decode parameters. Environment variables
//! that steer inference cannot be re-applied to a live process, so an env
//! fingerprint mismatch refuses the request and the client runs inline.

use std::fs;
use std::hash::{BuildHasher, Hasher, RandomState};
use std::io::{Read, Write};
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime};

use serde_json::{Value, json};

use crate::error::{
    EXIT_CANCELLED, EXIT_FORMAT_MISMATCH, EXIT_INPUT_DECODE, EXIT_MODEL_NOT_FOUND, EXIT_TIMEOUT,
    EXIT_USAGE, FocrError, FocrResult,
};
use crate::native_engine::tromr::{MusicWarning, StaffBBox, StaffSkip};
use crate::native_engine::{
    ChartPageMeta, DecodeOverrides, FinishReason, ModelIdentity, MusicPageMeta, PreprocessOverrides,
};
use crate::{LayoutSpan, OcrEngine, RecognizedDocument};

/// Default idle unload period. Overridable through `FOCR_RESIDENT_IDLE_SECS`
/// (sub-second daemons are for tests; the floor is 1 second).
const DEFAULT_IDLE: Duration = Duration::from_secs(600);

/// How long the client waits for a reply. A 3B-parameter VLM page on CPU is
/// minutes, not seconds; the default mirrors the engine's own 10-minute
/// forward-stage budget. `FOCR_RESIDENT_CLIENT_TIMEOUT_SECS` overrides it.
const DEFAULT_CLIENT_READ_TIMEOUT: Duration = Duration::from_secs(600);

/// How long the client waits for the daemon's acknowledgement. The daemon's
/// acceptor thread answers at once, even while its model serves another
/// request, so silence this long means a stopped or wedged process, not a busy
/// one, and the client runs inline (bd-mapo).
const CLIENT_ACK_WAIT: Duration = Duration::from_secs(5);

/// The `kind` of the line the acceptor answers every connection with.
const ACK_KIND: &str = "accepted";

/// The `kind` of a line saying the request's run entered a stage. Any number
/// come between the acknowledgement and the reply.
const STAGE_KIND: &str = "stage";

/// Connections acknowledged and waiting for the model. Past this the acceptor
/// blocks, and later clients wait in the OS backlog for their acknowledgement.
const ACCEPT_QUEUE: usize = 16;

/// How long the client waits for a freshly spawned daemon to write its state
/// file and accept. The daemon binds BEFORE loading the model, so this covers
/// process start only; thirty seconds absorbs a first-launch antivirus scan on
/// Windows. `FOCR_RESIDENT_SPAWN_WAIT_SECS` overrides it.
const DEFAULT_SPAWN_WAIT: Duration = Duration::from_secs(30);

/// Reply size cap. A single page's markdown + layout JSON is well under a
/// megabyte; 64 MiB refuses a corrupt or hostile daemon before it can size an
/// absurd allocation in the client.
const MAX_REPLY_BYTES: usize = 64 * 1024 * 1024;

/// Request size cap (pre-auth): paths + option scalars run a few KB.
const MAX_REQUEST_BYTES: usize = 1024 * 1024;

const PROTOCOL: u64 = 4;

/// Upper clamp for user-supplied second counts (one year). `Instant + Duration`
/// PANICS on overflow, so an absurd `FOCR_RESIDENT_*_SECS` value must not reach
/// deadline arithmetic — in the daemon's case that panic would fire at startup
/// and turn every later client run into a full spawn-wait stall.
const MAX_CONFIG_SECS: u64 = 31_536_000;

/// `FOCR_*` variables EXEMPT from the inference-environment fingerprint.
///
/// The fingerprint is a prefix scan over every `FOCR_*` variable rather than a
/// curated allow-list: the daemon inherits the environment of the client that
/// spawned it, and a later client with a DIFFERENT inference environment must
/// not be served from that process, so any variable this list does not exempt
/// participates in the match and a mismatch refuses the request (the client
/// then runs inline, which honors its own environment). The scan direction is
/// deliberate — a NEW inference-steering variable is fingerprinted by default;
/// forgetting to exempt a harmless one costs only an unnecessary inline
/// fallback, while the inverse (a curated include-list missing a real steering
/// variable) would serve a warm answer computed under the wrong environment.
///
/// Exempt: the resident transport's own knobs (they select HOW the run is
/// served, not what it computes) and client-process-only surfaces (progress
/// rendering, run telemetry, pull-time manifest override).
const ENV_FINGERPRINT_EXEMPT: &[&str] = &[
    "FOCR_NO_RESIDENT",
    "FOCR_RESIDENT_IDLE_SECS",
    "FOCR_RESIDENT_DIR",
    "FOCR_RESIDENT_LOG",
    "FOCR_RESIDENT_SPAWN_WAIT_SECS",
    "FOCR_RESIDENT_CLIENT_TIMEOUT_SECS",
    "FOCR_NO_PROGRESS",
    "FOCR_RUN_STORE",
    "FOCR_MANIFEST_URL",
    // FOCR_TIMING additionally disables the resident client entirely (its
    // stderr trace would print invisibly in the daemon).
    "FOCR_TIMING",
];

fn client_read_timeout() -> Duration {
    std::env::var("FOCR_RESIDENT_CLIENT_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        // Zero is rejected rather than honored: `set_read_timeout(Some(ZERO))`
        // is an error in std, so a literal 0 would fail every connect.
        .filter(|&seconds| seconds > 0)
        .map_or(DEFAULT_CLIENT_READ_TIMEOUT, |seconds| {
            Duration::from_secs(seconds.min(MAX_CONFIG_SECS))
        })
}

fn spawn_wait() -> Duration {
    std::env::var("FOCR_RESIDENT_SPAWN_WAIT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map_or(DEFAULT_SPAWN_WAIT, |seconds| {
            Duration::from_secs(seconds.min(MAX_CONFIG_SECS))
        })
}

fn idle_period() -> Duration {
    std::env::var("FOCR_RESIDENT_IDLE_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&seconds| seconds > 0)
        .map_or(DEFAULT_IDLE, |seconds| {
            Duration::from_secs(seconds.min(MAX_CONFIG_SECS))
        })
}

/// Whether the resident path is enabled at all: on by default, disabled by the
/// `--no-resident` flag or by `FOCR_NO_RESIDENT=1|true` for callers that
/// cannot pass flags.
#[must_use]
pub fn enabled(no_resident_flag: bool) -> bool {
    if no_resident_flag {
        return false;
    }
    !matches!(
        std::env::var("FOCR_NO_RESIDENT").ok().as_deref(),
        Some("1") | Some("true")
    )
}

// -------------------------------------------------------------------- state file

fn resident_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("FOCR_RESIDENT_DIR") {
        return Some(PathBuf::from(dir));
    }
    crate::dist::cache_root()
}

/// A short stable digest of the resolved model artifact path, so distinct
/// models get distinct daemons. `RandomState` keys vary per process, so this
/// hand-rolls FNV-1a. Canonicalized first: two spellings of the same artifact
/// must share a daemon, while `./m.focrq` from two directories must not.
fn root_digest(root: &Path) -> u64 {
    let canonical = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in canonical.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn state_path(root: &Path) -> Option<PathBuf> {
    resident_dir().map(|dir| dir.join(format!("resident-{:016x}.json", root_digest(root))))
}

struct DaemonState {
    port: u16,
    token: String,
}

fn read_state(root: &Path) -> Option<DaemonState> {
    let path = state_path(root)?;
    let raw = fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&raw).ok()?;
    Some(DaemonState {
        port: u16::try_from(value.get("port")?.as_u64()?).ok()?,
        token: value.get("token")?.as_str()?.to_owned(),
    })
}

fn write_state(root: &Path, port: u16, token: &str) -> std::io::Result<PathBuf> {
    let path = state_path(root).ok_or_else(|| {
        std::io::Error::other("no cache directory and no FOCR_RESIDENT_DIR; cannot go resident")
    })?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let body = json!({
        "port": port,
        "token": token,
        "pid": std::process::id(),
        "version": env!("CARGO_PKG_VERSION"),
        "model_root": root.to_string_lossy(),
    })
    .to_string();
    // Write-then-rename so a client never reads a half-written file. The
    // staging file is born 0600 on unix — the token is inside, so there must
    // be no umask window in which another user can read it.
    let staging = path.with_extension("json.tmp");
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&staging)?;
        file.write_all(body.as_bytes())?;
    }
    #[cfg(not(unix))]
    fs::write(&staging, &body)?;
    fs::rename(&staging, &path)?;
    Ok(path)
}

/// 128 bits of `RandomState` entropy (seeded from the OS) as the session
/// token. The real gate is the 0600 state file; the token binds a socket
/// connection to that file.
fn fresh_token() -> String {
    let mut token = String::with_capacity(32);
    for _ in 0..2 {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u128(std::time::UNIX_EPOCH.elapsed().map_or(0, |d| d.as_nanos()));
        hasher.write_u32(std::process::id());
        token.push_str(&format!("{:016x}", hasher.finish()));
    }
    token
}

/// The artifact identity the daemon pins at hydration: a re-pull or re-convert
/// must not be served from a stale resident model. Full nanosecond mtime, not
/// whole seconds — a re-convert landing within the same second at the same
/// byte length would otherwise stamp identical and be served stale.
fn artifact_stamp(root: &Path) -> (u64, u64) {
    let Ok(meta) = fs::metadata(root) else {
        return (0, 0);
    };
    let mtime = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
    (mtime, meta.len())
}

fn env_fingerprint() -> Value {
    // BTreeMap for a deterministic key order (std::env::vars() order is
    // unspecified); Value equality is content-based either way, but the wire
    // bytes staying stable keeps diffs in FOCR_RESIDENT_LOG readable.
    let mut map = std::collections::BTreeMap::new();
    for (key, value) in std::env::vars() {
        if key.starts_with("FOCR_") && !ENV_FINGERPRINT_EXEMPT.contains(&key.as_str()) {
            map.insert(key, Value::String(value));
        }
    }
    Value::Object(map.into_iter().collect())
}

// ------------------------------------------------------------------------ client

/// A daemon's successful reply: the document plus how its decode ended
/// (U2-X2) and which model ran (U2-E1), so a resident run reports the same
/// `finish_reason` and `model_id` an inline one would.
pub struct ResidentReply {
    /// The recognized document.
    pub document: RecognizedDocument,
    /// `None` when the daemon's model recorded no decode report.
    pub finish_reason: Option<FinishReason>,
    /// `None` when the reply names no model this binary registers.
    pub model: Option<ModelIdentity>,
    /// A TrOMR run's staves and musical-sanity warnings; `None` for every
    /// other run.
    pub music: Option<MusicPageMeta>,
    /// A OneChart run's number head and verdict; `None` for every other run.
    pub chart: Option<ChartPageMeta>,
}

fn chart_to_wire(meta: &ChartPageMeta) -> Value {
    json!({
        "pred_locs": meta.pred_locs,
        "reliable_distance": meta.reliable_distance,
        "reliable": meta.reliable,
    })
}

/// `None` when any part cannot be read: a half-read head is worse than falling
/// back to an inline run.
fn chart_from_wire(value: &Value) -> Option<ChartPageMeta> {
    let pred_locs = value
        .get("pred_locs")?
        .as_array()?
        .iter()
        .map(|slot| slot.as_f64().map(|v| v as f32))
        .collect::<Option<Vec<f32>>>()?;
    let reliable_distance = match value.get("reliable_distance")? {
        Value::Null => None,
        distance => Some(distance.as_f64()?),
    };
    let reliable = match value.get("reliable")? {
        Value::Null => None,
        verdict => Some(verdict.as_bool()?),
    };
    Some(ChartPageMeta {
        pred_locs,
        reliable_distance,
        reliable,
    })
}

fn bbox_to_wire(bbox: StaffBBox) -> Value {
    json!([bbox.0, bbox.1, bbox.2, bbox.3])
}

fn bbox_from_wire(value: &Value) -> Option<StaffBBox> {
    let edges = value.as_array()?;
    let edge = |index: usize| -> Option<usize> {
        edges
            .get(index)?
            .as_u64()
            .and_then(|e| usize::try_from(e).ok())
    };
    Some((edge(0)?, edge(1)?, edge(2)?, edge(3)?))
}

fn music_to_wire(meta: &MusicPageMeta) -> Value {
    let staves: Vec<Value> = meta
        .staves
        .iter()
        .map(|(index, bbox)| json!({ "index": index, "bbox": bbox_to_wire(*bbox) }))
        .collect();
    let skips: Vec<Value> = meta
        .skips
        .iter()
        .map(|skip| {
            json!({ "index": skip.index, "bbox": bbox_to_wire(skip.bbox), "reason": skip.reason })
        })
        .collect();
    let warnings: Vec<Value> = meta
        .warnings
        .iter()
        .map(|warning| {
            json!({
                "kind": warning.kind,
                "part": warning.part,
                "measure": warning.measure,
                "detail": warning.detail,
            })
        })
        .collect();
    json!({ "staves": staves, "skips": skips, "warnings": warnings })
}

/// The metadata [`music_to_wire`] wrote, or `None` for anything else: the
/// client then reads the page itself rather than report part of what it found.
fn music_from_wire(value: &Value) -> Option<MusicPageMeta> {
    let uint = |item: &Value, key: &str| -> Option<usize> {
        item.get(key)?
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
    };
    let text =
        |item: &Value, key: &str| -> Option<String> { item.get(key)?.as_str().map(str::to_owned) };
    let list = |key: &str| value.get(key).and_then(Value::as_array);
    let staves = list("staves")?
        .iter()
        .map(|staff| Some((uint(staff, "index")?, bbox_from_wire(staff.get("bbox")?)?)))
        .collect::<Option<Vec<_>>>()?;
    let skips = list("skips")?
        .iter()
        .map(|skip| {
            Some(StaffSkip {
                index: uint(skip, "index")?,
                bbox: bbox_from_wire(skip.get("bbox")?)?,
                reason: text(skip, "reason")?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let warnings = list("warnings")?
        .iter()
        .map(|warning| {
            Some(MusicWarning {
                kind: crate::native_engine::tromr::music_warning_kind(
                    warning.get("kind")?.as_str()?,
                )?,
                part: uint(warning, "part")?,
                measure: uint(warning, "measure")?,
                detail: text(warning, "detail")?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(MusicPageMeta {
        staves,
        skips,
        warnings,
    })
}

/// The reply's `model_id`/`source_sha256` pair, its id mapped back onto the
/// registry.
fn model_from_reply(reply: &Value) -> Option<ModelIdentity> {
    let arch = reply
        .get("model_id")
        .and_then(Value::as_str)
        .and_then(crate::native_engine::model_arch::arch_by_id)?;
    Some(ModelIdentity {
        model_id: arch.id(),
        source_sha256: reply
            .get("source_sha256")
            .and_then(Value::as_str)
            .map(str::to_owned),
    })
}

/// One recognition request as it crosses the wire: everything the daemon needs
/// to reproduce the exact recognition the client would have run inline.
pub struct ResidentRequest<'a> {
    /// The input image path (the client absolutizes it; the daemon's cwd is
    /// unrelated).
    pub image: &'a Path,
    /// The RESOLVED model artifact path (a concrete `.focrq` file or
    /// safetensors directory) — the daemon key. The client resolves before
    /// trying the wire so an unresolvable model takes the inline path's
    /// download-offer/exit-3 contract unchanged.
    pub model: PathBuf,
    /// Load-resolved decode overrides (part of the rehydration fingerprint).
    pub decode: DecodeOverrides,
    /// Load-resolved preprocess overrides (part of the rehydration fingerprint).
    pub preprocess: PreprocessOverrides,
    /// GOT-OCR2 `--format` mode (forward-read; set per request in the daemon).
    pub format: bool,
    /// GOT-OCR2 square pad, from the task (forward-read; set per request).
    pub square_pad: bool,
    /// SmolVLM2 `--task describe` question (forward-read; set per request).
    pub question: Option<&'a str>,
    /// Told each stage the daemon enters for this request.
    pub on_stage: Option<OnStage<'a>>,
}

/// Told a stage's name and budget.
pub type OnStage<'a> = &'a dyn Fn(&str, Option<Duration>);

fn options_value(request: &ResidentRequest<'_>) -> Value {
    json!({
        "decode": {
            "max_length": request.decode.max_length,
            "temperature": request.decode.temperature,
            "no_repeat_ngram": request.decode.no_repeat_ngram,
            "ngram_window": request.decode.ngram_window,
            "max_page_tokens": request.decode.max_page_tokens,
        },
        "preprocess": {
            "base_size": request.preprocess.base_size,
            "image_size": request.preprocess.image_size,
            "gundam": request.preprocess.gundam,
        },
        "format": request.format,
        "square_pad": request.square_pad,
        "question": request.question,
    })
}

/// Try to recognize through a resident daemon, spawning one if none is
/// listening.
///
/// `Ok(None)` means "no resident path available, run inline" and is always
/// safe; a daemon spawned in the background (if any) serves the NEXT
/// invocation warm. `Ok(Some)` is a completed recognition; `Err` is a genuine
/// recognition error from the daemon, carrying the same exit-code class the
/// inline path would have produced.
///
/// # Errors
/// Recognition errors reported by the daemon (`kind: "ocr"`), plus
/// [`FocrError::Cancelled`] when cooperative shutdown fires while waiting for
/// the reply (matching the inline decode loop's Ctrl+C contract); every
/// transport-shaped failure is `Ok(None)`. A model the daemon does not serve
/// (U2-A7) is `Ok(None)` before any connect or spawn.
pub fn try_recognize(request: &ResidentRequest<'_>) -> FocrResult<Option<ResidentReply>> {
    if !crate::native_engine::resident_daemon_serves(&request.model) {
        return Ok(None);
    }
    match connect(&request.model) {
        Some(stream) => roundtrip(stream, request),
        None => Ok(None),
    }
}

/// Why one connect attempt did not produce a stream — the state-file removal
/// decision needs the kind.
enum ConnectFailure {
    /// No state file exists; nothing to retry against and nothing to clear.
    NoState,
    /// The kernel actively refused: nothing is listening on the recorded port,
    /// so the state file is provably stale and safe to clear.
    Refused,
    /// A timeout or any other error: the daemon may be alive but mid-request
    /// (a forward is minutes long); deleting its state file would orphan a
    /// healthy multi-gigabyte process.
    Other,
}

fn connect(root: &Path) -> Option<TcpStream> {
    // No resolvable state directory (no cache root, no FOCR_RESIDENT_DIR):
    // a spawned daemon would fail its state write immediately, and the poll
    // below would then burn the full spawn wait (30 s) on EVERY run. Bail to
    // the inline path up front instead.
    state_path(root)?;
    let mut last = ConnectFailure::NoState;
    for attempt in 0..3 {
        if attempt > 0 {
            std::thread::sleep(Duration::from_millis(300));
        }
        match connect_once(root) {
            Ok(stream) => return Some(stream),
            Err(ConnectFailure::NoState) => {
                last = ConnectFailure::NoState;
                break;
            }
            Err(failure) => last = failure,
        }
    }
    match last {
        // Provably nothing listening: clear the stale state, spawn fresh.
        ConnectFailure::Refused => {
            if let Some(path) = state_path(root)
                && path.exists()
            {
                let _ = fs::remove_file(&path);
            }
        }
        ConnectFailure::NoState => {}
        // Possibly a live daemon on a loaded machine, or with its accept queue
        // full. Removing its state here would orphan it — never again findable,
        // idling gigabytes until its timer — while a duplicate spawned beside
        // it. Fall back inline and leave the daemon to answer the next run.
        ConnectFailure::Other => return None,
    }
    spawn_daemon(root)?;
    let deadline = Instant::now() + spawn_wait();
    while Instant::now() < deadline {
        if let Ok(stream) = connect_once(root) {
            return Some(stream);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    None
}

fn connect_once(root: &Path) -> Result<TcpStream, ConnectFailure> {
    let state = read_state(root).ok_or(ConnectFailure::NoState)?;
    let stream = TcpStream::connect_timeout(
        &(Ipv4Addr::LOCALHOST, state.port).into(),
        Duration::from_millis(1000),
    )
    .map_err(|error| {
        if error.kind() == std::io::ErrorKind::ConnectionRefused {
            ConnectFailure::Refused
        } else {
            ConnectFailure::Other
        }
    })?;
    // A SHORT per-syscall timeout, looped in `roundtrip` up to the overall
    // client timeout: the reply wait spans a minutes-long forward, and a
    // single long blocking read would make the client deaf to Ctrl+C (the
    // inline path polls the cooperative-shutdown flag every decode step; the
    // resident wait must match that responsiveness).
    stream
        .set_read_timeout(Some(Duration::from_millis(500)))
        .map_err(|_| ConnectFailure::Other)?;
    stream.set_nodelay(true).ok();
    Ok(stream)
}

fn spawn_daemon(root: &Path) -> Option<()> {
    let exe = std::env::current_exe().ok()?;
    let mut command = Command::new(exe); // ubs:ignore — this binary's own path, never input
    command
        .arg("resident-daemon")
        .arg("--model-root")
        .arg(root)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    // Field diagnostics: the daemon's stderr is discarded by default;
    // FOCR_RESIDENT_LOG routes it to a file instead, which is how a silent
    // failure to serve gets a voice.
    if let Ok(log) = std::env::var("FOCR_RESIDENT_LOG")
        && let Ok(file) = fs::OpenOptions::new().create(true).append(true).open(&log)
        && let Ok(err) = file.try_clone()
    {
        command.stdout(file).stderr(err);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // DETACHED_PROCESS | CREATE_NO_WINDOW: outlive the console, show nothing.
        command.creation_flags(0x0000_0008 | 0x0800_0000);
    }
    command.spawn().ok().map(|_child| ())
}

fn roundtrip(
    mut stream: TcpStream,
    request: &ResidentRequest<'_>,
) -> FocrResult<Option<ResidentReply>> {
    let Some(state) = read_state(&request.model) else {
        return Ok(None);
    };
    let image = std::path::absolute(request.image).unwrap_or_else(|_| request.image.to_path_buf());
    let header = json!({
        "protocol": PROTOCOL,
        "op": "ocr",
        "token": state.token,
        "version": env!("CARGO_PKG_VERSION"),
        "model_root": request.model.to_string_lossy(),
        "image": image.to_string_lossy(),
        "options": options_value(request),
        "env": env_fingerprint(),
    });
    if stream
        .write_all(format!("{header}\n").as_bytes())
        .and_then(|()| stream.flush())
        .is_err()
    {
        return Ok(None);
    }

    let waits = (CLIENT_ACK_WAIT, client_read_timeout());
    let on_stage = |name: &str, budget: Option<Duration>| {
        if let Some(on_stage) = request.on_stage {
            on_stage(name, budget);
        }
    };
    let Some(line) = read_reply(&mut stream, waits, crate::cancel_checkpoint, on_stage)? else {
        return Ok(None);
    };
    reply_from_wire(&line)
}

/// What the daemon's reply `line` means for this run; see [`try_recognize`].
fn reply_from_wire(line: &[u8]) -> FocrResult<Option<ResidentReply>> {
    let Ok(reply) = serde_json::from_slice::<Value>(line) else {
        return Ok(None);
    };
    if reply.get("ok").and_then(Value::as_bool) == Some(true) {
        let markdown = reply
            .get("markdown")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        // A reply without it comes from a daemon that predates `--raw`, which
        // an inline run can still answer.
        let Some(raw) = reply.get("raw").and_then(Value::as_str) else {
            return Ok(None);
        };
        let layout = reply
            .get("layout")
            .and_then(Value::as_array)
            .map(|spans| spans.iter().filter_map(LayoutSpan::from_json).collect())
            .unwrap_or_default();
        // Absent from an older daemon's reply: the run then reports none.
        let finish_reason = match reply.get("finish_reason").and_then(Value::as_str) {
            Some("stop") => Some(FinishReason::Stop),
            Some("length") => Some(FinishReason::Length),
            _ => None,
        };
        let music = match reply.get("music").map(music_from_wire) {
            None => None,
            Some(Some(music)) => Some(music),
            Some(None) => return Ok(None),
        };
        let chart = match reply.get("chart").map(chart_from_wire) {
            None => None,
            Some(Some(chart)) => Some(chart),
            Some(None) => return Ok(None),
        };
        return Ok(Some(ResidentReply {
            document: RecognizedDocument {
                markdown,
                raw: raw.to_owned(),
                layout,
            },
            finish_reason,
            model: model_from_reply(&reply),
            music,
            chart,
        }));
    }
    // A recognition error is real and final; anything transport-shaped means
    // fallback.
    match reply.get("kind").and_then(Value::as_str) {
        Some("ocr") => {
            let code = reply
                .get("exit_code")
                .and_then(Value::as_i64)
                .and_then(|c| i32::try_from(c).ok())
                .unwrap_or(crate::error::EXIT_GENERIC);
            let message = reply
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("resident recognition failed")
                .to_owned();
            Err(wire_error(code, message))
        }
        _ => Ok(None),
    }
}

/// The daemon's reply line, or `None` to run inline: it did not acknowledge
/// within `ack_wait`, the reply took past `reply_wait`, or the transport failed.
///
/// A daemon that refuses a request answers with the refusal alone, so any first
/// line but the acknowledgement or a stage is the reply, and `on_stage` hears
/// each stage on the way to it. The reply is one JSON line; the cap refuses a
/// corrupt daemon before it sizes an allocation in this process. The socket's
/// per-syscall timeout is short (see `connect_once`), so this loop can poll
/// `checkpoint` between reads: Ctrl+C aborts the wait with the same
/// Cancelled/exit-6 contract as the inline decode loop.
fn read_reply(
    stream: &mut TcpStream,
    (ack_wait, reply_wait): (Duration, Duration),
    checkpoint: impl Fn() -> FocrResult<()>,
    on_stage: impl Fn(&str, Option<Duration>),
) -> FocrResult<Option<Vec<u8>>> {
    let started = Instant::now();
    let mut acknowledged = false;
    let mut buffer: Vec<u8> = Vec::new();
    let mut chunk = [0_u8; 65536];
    loop {
        checkpoint()?;
        if let Some(end) = buffer.iter().position(|&byte| byte == b'\n') {
            let mut line: Vec<u8> = buffer.drain(..=end).collect();
            line.pop();
            if !acknowledged && is_acknowledgement(&line) {
                acknowledged = true;
                continue;
            }
            if let Some((name, budget)) = stage_from_wire(&line) {
                on_stage(&name, budget);
                continue;
            }
            return Ok(Some(line));
        }
        if buffer.len() >= MAX_REPLY_BYTES {
            return Ok(None);
        }
        let waited = started.elapsed();
        if waited >= reply_wait || (!acknowledged && waited >= ack_wait) {
            return Ok(None);
        }
        match stream.read(&mut chunk) {
            Ok(0) => return Ok(None),
            Ok(count) => buffer.extend_from_slice(&chunk[..count]),
            // A per-syscall timeout is the expected idle tick while the
            // daemon's forward runs; anything else is transport-shaped.
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) => {}
            Err(_) => return Ok(None),
        }
    }
}

fn is_acknowledgement(line: &[u8]) -> bool {
    serde_json::from_slice::<Value>(line)
        .is_ok_and(|reply| reply.get("kind").and_then(Value::as_str) == Some(ACK_KIND))
}

/// A stage line's name and budget; `None` for any other line.
fn stage_from_wire(line: &[u8]) -> Option<(String, Option<Duration>)> {
    let line = serde_json::from_slice::<Value>(line).ok()?;
    if line.get("kind")?.as_str()? != STAGE_KIND {
        return None;
    }
    let budget = line.get("budget_ms").and_then(Value::as_u64);
    Some((
        line.get("name")?.as_str()?.to_owned(),
        budget.map(Duration::from_millis),
    ))
}

/// The text [`wire_error`] rebuilds `error` from. A variant it rebuilds writes
/// its own prefix back, so that variant crosses as its message alone.
fn wire_message(error: &FocrError) -> String {
    match error {
        FocrError::Usage(message)
        | FocrError::ModelNotFound(message)
        | FocrError::InputDecode(message)
        | FocrError::Timeout(message)
        | FocrError::FormatMismatch(message) => message.clone(),
        other => other.to_string(),
    }
}

fn wire_error(exit_code: i32, message: String) -> FocrError {
    match exit_code {
        EXIT_USAGE => FocrError::Usage(message),
        EXIT_MODEL_NOT_FOUND => FocrError::ModelNotFound(message),
        EXIT_INPUT_DECODE => FocrError::InputDecode(message),
        EXIT_TIMEOUT => FocrError::Timeout(message),
        EXIT_CANCELLED => FocrError::Cancelled,
        EXIT_FORMAT_MISMATCH => FocrError::FormatMismatch(message),
        _ => FocrError::Other(anyhow::anyhow!(message)),
    }
}

// ------------------------------------------------------------------------ daemon

/// The daemon's hydrated engine plus the identity it was hydrated under.
struct ResidentEngine {
    engine: OcrEngine,
    /// Canonical JSON of the load-resolved options the engine was built with.
    fingerprint: String,
    /// `(mtime_nanos, len)` of the model artifact at hydration.
    stamp: (u64, u64),
}

/// Answer every connection at once and hand it to the serial worker, whatever
/// the worker is doing, so a client can tell a busy daemon from a dead one.
fn accept_connections(listener: &TcpListener, queue: &std::sync::mpsc::SyncSender<TcpStream>) {
    let acknowledgement = format!("{}\n", json!({ "ok": true, "kind": ACK_KIND }));
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else {
            return;
        };
        let _ = stream.write_all(acknowledgement.as_bytes());
        if queue.send(stream).is_err() {
            return;
        }
    }
}

/// Run the resident daemon until the idle period passes without a request.
///
/// Binds first and hydrates lazily on the first request, so the process is
/// connectable within milliseconds of spawning and a daemon that never gets a
/// request costs no model memory before its idle exit.
///
/// # Errors
/// Fails only on bind/state-file setup problems; per-request failures are
/// reported to that request's client and never end the daemon.
pub fn run_daemon(model_root: &Path) -> FocrResult<()> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .map_err(|error| FocrError::Other(anyhow::anyhow!("resident daemon bind: {error}")))?;
    let port = listener
        .local_addr()
        .map_err(|error| FocrError::Other(anyhow::anyhow!("resident daemon addr: {error}")))?
        .port();
    let token = fresh_token();
    let state_file = write_state(model_root, port, &token)
        .map_err(|error| FocrError::Other(anyhow::anyhow!("resident state write: {error}")))?;
    let (queue, connections) = std::sync::mpsc::sync_channel(ACCEPT_QUEUE);
    std::thread::Builder::new()
        .name("focr-resident-accept".into())
        .spawn(move || accept_connections(&listener, &queue))
        .map_err(|error| FocrError::Other(anyhow::anyhow!("resident acceptor: {error}")))?;
    eprintln!(
        "focr resident daemon serving {} on 127.0.0.1:{port}",
        model_root.display()
    );

    let idle = idle_period();
    let mut resident: Option<ResidentEngine> = None;
    let mut deadline = Instant::now() + idle;

    loop {
        match connections.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(stream) => {
                // Un-poison the cooperative-shutdown flag before serving. On
                // Unix the daemon shares its spawner's process group, so a
                // Ctrl+C during a resident-served run delivers SIGINT here
                // too: the in-flight forward correctly aborts Cancelled, but
                // the process-global flag would otherwise stay set and make
                // every LATER request fail Cancelled instantly — a poisoned
                // daemon squatting on gigabytes. The abort-the-current-run
                // semantics are preserved (the flag is only cleared between
                // requests); a second Ctrl+C still hard-exits.
                crate::reset_shutdown();
                // Serve strictly serially; a second client queues here, already
                // acknowledged. Panic-isolated: this process exists to hold a
                // hydrated multi-gigabyte model across many calls, so one
                // malformed request must not cost every later caller that
                // work. `AssertUnwindSafe` is sound for `resident` because it
                // is only ever replaced wholesale (`Some(...)` after the
                // engine is fully built), never mutated in place.
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    handle_connection(stream, model_root, &token, port, &mut resident);
                }));
                if outcome.is_err() {
                    eprintln!(
                        "focr resident daemon: request panicked; connection dropped, model kept"
                    );
                }
                deadline = Instant::now() + idle;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                eprintln!("focr resident daemon idle exit");
                remove_state_if_ours(&state_file, port);
                return Ok(());
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                remove_state_if_ours(&state_file, port);
                return Ok(());
            }
        }
    }
}

/// Removes the state file only when it still describes THIS daemon.
///
/// Two cold-start invocations racing both spawn a daemon; both write the same
/// state path and the second write wins, orphaning the first. The orphan
/// serves nobody and idle-exits later — and an unconditional remove at that
/// exit would delete the SUCCESSOR's state file, making the healthy daemon
/// undiscoverable and spawning a third copy of the model. A retiring daemon
/// therefore checks that the file still names its own port.
fn remove_state_if_ours(state_file: &Path, port: u16) {
    let ours = fs::read_to_string(state_file)
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .and_then(|value| value.get("port")?.as_u64())
        .is_some_and(|recorded| recorded == u64::from(port));
    if ours {
        let _ = fs::remove_file(state_file);
    }
}

fn decode_overrides_from_wire(options: &Value) -> DecodeOverrides {
    let decode = options.get("decode").cloned().unwrap_or(Value::Null);
    let usize_of = |key: &str| {
        decode
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
    };
    DecodeOverrides {
        max_length: usize_of("max_length"),
        temperature: decode
            .get("temperature")
            .and_then(Value::as_f64)
            .map(|t| t as f32),
        no_repeat_ngram: usize_of("no_repeat_ngram"),
        ngram_window: usize_of("ngram_window"),
        max_page_tokens: usize_of("max_page_tokens"),
    }
}

fn preprocess_overrides_from_wire(options: &Value) -> PreprocessOverrides {
    let preprocess = options.get("preprocess").cloned().unwrap_or(Value::Null);
    let usize_of = |key: &str| {
        preprocess
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
    };
    PreprocessOverrides {
        base_size: usize_of("base_size"),
        image_size: usize_of("image_size"),
        gundam: preprocess.get("gundam").and_then(Value::as_bool),
    }
}

/// The load-resolved slice of the wire options — the rehydration fingerprint.
/// Forward-read options (`format`, `question`) are set per request instead.
fn load_fingerprint(options: &Value) -> String {
    json!({
        "decode": options.get("decode").cloned().unwrap_or(Value::Null),
        "preprocess": options.get("preprocess").cloned().unwrap_or(Value::Null),
    })
    .to_string()
}

fn handle_connection(
    stream: TcpStream,
    model_root: &Path,
    token: &str,
    port: u16,
    resident: &mut Option<ResidentEngine>,
) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
    // A write timeout is the daemon's survival: replies are written inline in
    // the accept loop, so one client that stops reading would otherwise park
    // this thread in `write_all` forever.
    let _ = stream.set_write_timeout(Some(Duration::from_secs(60)));
    let _ = stream.set_nodelay(true);

    // BOUNDED read, in bytes AND in time. The byte cap stops a pre-auth OOM;
    // the wall-clock deadline stops a byte-trickling peer from wedging the
    // single-threaded accept loop (the 10 s read timeout above is per SYSCALL).
    const REQUEST_DEADLINE: Duration = Duration::from_secs(30);
    let mut stream = stream;
    let deadline = Instant::now() + REQUEST_DEADLINE;
    let mut buffer: Vec<u8> = Vec::new();
    let mut chunk = [0_u8; 4096];
    let newline = loop {
        if Instant::now() >= deadline {
            return;
        }
        match stream.read(&mut chunk) {
            Ok(0) => return,
            Ok(count) => {
                buffer.extend_from_slice(&chunk[..count]);
                if let Some(position) = buffer.iter().position(|&byte| byte == b'\n') {
                    break position;
                }
                if buffer.len() >= MAX_REQUEST_BYTES {
                    let reply =
                        json!({ "ok": false, "kind": "request", "message": "request too large" });
                    let _ = stream.write_all(format!("{reply}\n").as_bytes());
                    return;
                }
            }
            Err(_) => return,
        }
    };
    let Ok(request) = serde_json::from_slice::<Value>(&buffer[..newline]) else {
        return;
    };

    let refuse = |stream: &mut TcpStream, kind: &str, message: &str| {
        let reply = json!({ "ok": false, "kind": kind, "message": message });
        let _ = stream.write_all(format!("{reply}\n").as_bytes());
    };

    // Token first: an unauthenticated peer learns nothing but "no".
    // Constant-time compare (XOR-fold) so a loopback peer cannot walk the
    // token off an early-exit comparison.
    let token_matches = |candidate: &str| {
        let (a, b) = (candidate.as_bytes(), token.as_bytes());
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b).fold(0_u8, |acc, (x, y)| acc | (x ^ y)) == 0
    };
    if !request
        .get("token")
        .and_then(Value::as_str)
        .is_some_and(token_matches)
    {
        refuse(&mut stream, "auth", "bad token");
        return;
    }
    if request.get("protocol").and_then(Value::as_u64) != Some(PROTOCOL)
        || request.get("version").and_then(Value::as_str) != Some(env!("CARGO_PKG_VERSION"))
    {
        // A different binary version must not be served by this process; the
        // client falls back inline and this daemon retires so the next spawn
        // matches. Ownership-checked: the state file may already belong to a
        // successor daemon.
        refuse(&mut stream, "version", "resident daemon version mismatch");
        if let Some(state) = state_path(model_root) {
            remove_state_if_ours(&state, port);
        }
        std::process::exit(0);
    }

    // The client says which model it thinks it is talking to; a daemon keyed
    // by a colliding or stale state file must refuse rather than recognize
    // with the wrong model. Compared canonically.
    if let Some(wire_root) = request.get("model_root").and_then(Value::as_str) {
        let wire_canonical =
            fs::canonicalize(wire_root).unwrap_or_else(|_| PathBuf::from(wire_root));
        let own_canonical =
            fs::canonicalize(model_root).unwrap_or_else(|_| model_root.to_path_buf());
        if wire_canonical != own_canonical {
            refuse(
                &mut stream,
                "request",
                "resident daemon serves a different model",
            );
            return;
        }
    }

    // The daemon inherits its spawner's environment; a client whose
    // inference-steering env differs must run inline instead (env cannot be
    // re-applied to a live process at arbitrary points).
    if request.get("env").cloned().unwrap_or_else(|| json!({})) != env_fingerprint() {
        refuse(
            &mut stream,
            "env",
            "inference environment differs from the resident daemon's",
        );
        return;
    }

    // A model with a lever policy (Unlimited-OCR-2), or one whose header the
    // bounded probe refuses, runs inline until its reply carries the decode
    // report and lever warnings (bd-u77k, U2-A7); the client falls back.
    if !crate::native_engine::resident_daemon_serves(model_root) {
        refuse(
            &mut stream,
            "model",
            "the resident daemon does not serve this model; run inline",
        );
        return;
    }

    // A re-pulled or re-converted artifact invalidates the resident weights.
    let stamp_now = artifact_stamp(model_root);
    if let Some(engine) = resident.as_ref()
        && engine.stamp != stamp_now
    {
        refuse(&mut stream, "stale", "model artifact changed since load");
        if let Some(state) = state_path(model_root) {
            remove_state_if_ours(&state, port);
        }
        std::process::exit(0);
    }

    let options = request.get("options").cloned().unwrap_or_else(|| json!({}));
    let fingerprint = load_fingerprint(&options);

    // Decode/preprocess overrides are resolved at MODEL LOAD, so a request
    // with different load-resolved options cannot be served by the current
    // hydration: drop it and rehydrate under the new fingerprint. (Alternating
    // fingerprints thrash the load; identical repeat runs — the whole point of
    // residency — hit the warm path.)
    if resident
        .as_ref()
        .is_some_and(|engine| engine.fingerprint != fingerprint)
    {
        *resident = None;
    }

    // Forward-read globals are (re)set on every request; load-resolved globals
    // must be in place BEFORE the engine hydrates.
    crate::native_engine::set_decode_overrides(decode_overrides_from_wire(&options));
    crate::native_engine::set_preprocess_overrides(preprocess_overrides_from_wire(&options));
    crate::native_engine::force_got_format(
        options
            .get("format")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    );
    crate::native_engine::force_got_square_pad(
        options
            .get("square_pad")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    );
    crate::native_engine::set_smolvlm2_question(
        options
            .get("question")
            .and_then(Value::as_str)
            .map(str::to_owned),
    );

    if resident.is_none() {
        match OcrEngine::new() {
            Ok(engine) => {
                *resident = Some(ResidentEngine {
                    engine,
                    fingerprint,
                    stamp: stamp_now,
                });
            }
            Err(error) => {
                // Engine START failure (runtime build, pool install) is a
                // property of this daemon process, not of the request or the
                // model — refuse transport-shaped so the client falls back to
                // its own inline engine rather than surfacing a fatal error.
                // Model-load and recognition failures surface later, inside
                // `recognize_with_layout_model`, as real `kind:"ocr"` errors.
                refuse(
                    &mut stream,
                    "engine",
                    &format!("resident engine start failed: {error}"),
                );
                return;
            }
        }
    }
    let Some(engine) = resident.as_ref() else {
        return;
    };

    let image = PathBuf::from(request.get("image").and_then(Value::as_str).unwrap_or(""));
    if image.as_os_str().is_empty() {
        refuse(&mut stream, "request", "missing image path");
        return;
    }

    // The client shows these as its run's stages. A line that cannot be
    // written costs the client that line.
    if let Ok(client) = stream.try_clone() {
        engine
            .engine
            .set_stage_sink(Some(Box::new(move |name, budget| {
                let mut line = json!({ "kind": STAGE_KIND, "name": name });
                if let Some(budget) = budget {
                    line["budget_ms"] =
                        u64::try_from(budget.as_millis()).unwrap_or(u64::MAX).into();
                }
                let _ = (&client).write_all(format!("{line}\n").as_bytes());
            })));
    }
    let outcome = engine
        .engine
        .recognize_with_layout_model(model_root, &image);
    engine.engine.set_stage_sink(None);
    match outcome {
        Ok(document) => {
            let layout: Vec<Value> = document.layout.iter().map(LayoutSpan::to_json).collect();
            // Taken per request: the warm model's report would otherwise
            // accumulate across requests.
            let finish_reason = engine
                .engine
                .take_last_decode_report()
                .map(|report| report.run_finish_reason().as_str());
            let mut reply = json!({
                "ok": true,
                "markdown": document.markdown,
                "raw": document.raw,
                "layout": layout,
                "finish_reason": finish_reason,
            });
            if let Some(music) = engine.engine.take_music_page_meta() {
                reply["music"] = music_to_wire(&music);
            }
            if let Some(chart) = engine.engine.take_chart_page_meta() {
                reply["chart"] = chart_to_wire(&chart);
            }
            if let Some(model) = engine.engine.loaded_model_identity() {
                model.merge_into(&mut reply);
            }
            let _ = stream.write_all(format!("{reply}\n").as_bytes());
            let _ = stream.flush();
        }
        Err(error) => {
            let reply = json!({
                "ok": false,
                "kind": "ocr",
                "exit_code": error.exit_code(),
                "message": wire_message(&error),
            });
            let _ = stream.write_all(format!("{reply}\n").as_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_digest_distinguishes_roots_and_is_stable() {
        let a = root_digest(Path::new("/tmp/model-a.focrq"));
        let b = root_digest(Path::new("/tmp/model-b.focrq"));
        assert_ne!(a, b);
        assert_eq!(a, root_digest(Path::new("/tmp/model-a.focrq")));
    }

    #[test]
    fn tokens_are_distinct_and_hex() {
        let one = fresh_token();
        let two = fresh_token();
        assert_eq!(one.len(), 32);
        assert!(one.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_ne!(one, two, "two RandomState-seeded tokens collided");
    }

    const ACK: &[u8] = b"{\"ok\":true,\"kind\":\"accepted\"}\n";
    const LONG: Duration = Duration::from_secs(30);
    const SHORT: Duration = Duration::from_millis(500);
    /// How long a test's daemon outlives the wait it tests.
    const HOLD: Duration = Duration::from_secs(8);

    /// The test's read, with no cancel flag: another test sets the global one.
    fn read(stream: &mut TcpStream, waits: (Duration, Duration)) -> Option<Vec<u8>> {
        read_reply(stream, waits, || Ok(()), |_, _| {}).unwrap()
    }

    /// A client's socket to a thread playing the daemon.
    fn client_of(daemon: impl FnOnce(TcpStream) + Send + 'static) -> TcpStream {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || daemon(listener.accept().unwrap().0));
        let stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_millis(10)))
            .unwrap();
        stream
    }

    #[test]
    fn a_silent_daemon_is_given_up_on_at_the_ack_wait() {
        let mut stream = client_of(|stream| {
            std::thread::sleep(HOLD);
            drop(stream);
        });
        let started = Instant::now();
        assert_eq!(read(&mut stream, (SHORT, LONG)), None);
        assert!(started.elapsed() < HOLD);
    }

    #[test]
    fn a_cancel_ends_the_wait() {
        let mut stream = client_of(|stream| {
            std::thread::sleep(HOLD);
            drop(stream);
        });
        let out = read_reply(
            &mut stream,
            (LONG, LONG),
            || Err(FocrError::Cancelled),
            |_, _| {},
        );
        assert!(matches!(out, Err(FocrError::Cancelled)), "{out:?}");
    }

    #[test]
    fn stages_reach_the_client_on_the_way_to_the_reply() {
        let mut stream = client_of(|mut stream| {
            stream.write_all(ACK).unwrap();
            stream
                .write_all(b"{\"kind\":\"stage\",\"name\":\"load\"}\n")
                .unwrap();
            stream
                .write_all(b"{\"budget_ms\":600000,\"kind\":\"stage\",\"name\":\"forward\"}\n")
                .unwrap();
            stream.write_all(b"{\"ok\":true,\"raw\":\"r\"}\n").unwrap();
        });
        let heard = std::cell::RefCell::new(Vec::new());
        let reply = read_reply(
            &mut stream,
            (LONG, LONG),
            || Ok(()),
            |name, budget| heard.borrow_mut().push((name.to_owned(), budget)),
        )
        .unwrap()
        .unwrap();
        assert_eq!(reply, b"{\"ok\":true,\"raw\":\"r\"}");
        assert_eq!(
            heard.into_inner(),
            [
                ("load".to_owned(), None),
                ("forward".to_owned(), Some(Duration::from_secs(600))),
            ]
        );
    }

    fn sample_music() -> MusicPageMeta {
        MusicPageMeta {
            staves: vec![(0, (1, 2, 3, 4)), (2, (5, 6, 7, 8))],
            skips: vec![StaffSkip {
                index: 1,
                bbox: (9, 10, 11, 12),
                reason: "width guard".into(),
            }],
            warnings: vec![
                MusicWarning {
                    kind: "overfull_bar",
                    part: 1,
                    measure: 3,
                    detail: "5/4 in 4/4".into(),
                },
                MusicWarning {
                    kind: "key_mismatch",
                    part: 2,
                    measure: 0,
                    detail: "G against D".into(),
                },
            ],
        }
    }

    #[test]
    fn music_metadata_crosses_the_wire_whole() {
        let wire = music_to_wire(&sample_music());
        let back = music_from_wire(&wire).expect("the wire shape parses");
        assert_eq!(music_to_wire(&back), wire);
        assert_eq!(back.staves, sample_music().staves);
        assert_eq!(back.skips[0].reason, "width guard");
        assert_eq!(back.warnings[1].kind, "key_mismatch");
    }

    #[test]
    fn music_metadata_that_cannot_be_read_is_not_half_reported() {
        let wire = music_to_wire(&sample_music());
        for (pointer, broken) in [
            ("/staves/0/bbox", json!([1, 2, 3])),
            ("/skips/0/reason", json!(7)),
            ("/warnings/0/kind", json!("no_such_warning")),
            ("/warnings/1/part", json!(-1)),
        ] {
            let mut bad = wire.clone();
            *bad.pointer_mut(pointer).expect(pointer) = broken;
            assert!(music_from_wire(&bad).is_none(), "{pointer}");
        }
        assert!(music_from_wire(&json!({ "staves": [] })).is_none());
    }

    #[test]
    fn a_chart_head_crosses_the_wire_whole_or_not_at_all() {
        let meta = ChartPageMeta {
            pred_locs: vec![0.25, 1.0],
            reliable_distance: Some(0.5),
            reliable: Some(false),
        };
        assert_eq!(chart_from_wire(&chart_to_wire(&meta)), Some(meta.clone()));
        let unverifiable = ChartPageMeta {
            reliable_distance: None,
            reliable: None,
            ..meta
        };
        assert_eq!(
            chart_from_wire(&chart_to_wire(&unverifiable)),
            Some(unverifiable.clone())
        );
        let wire = chart_to_wire(&unverifiable);
        for (key, broken) in [
            ("pred_locs", json!(["x"])),
            ("reliable_distance", json!("far")),
            ("reliable", json!(1)),
        ] {
            let mut bad = wire.clone();
            bad[key] = broken;
            assert!(chart_from_wire(&bad).is_none(), "{key}");
        }
        let reply = json!({ "ok": true, "markdown": "m", "raw": "r", "chart": { "reliable": 1 } });
        let served = reply_from_wire(reply.to_string().as_bytes()).unwrap();
        assert!(served.is_none(), "an unreadable head falls back inline");
    }

    #[test]
    fn a_served_run_reports_music_it_can_read_and_otherwise_runs_inline() {
        let served = |music: Value| {
            let reply = json!({ "ok": true, "markdown": "m", "raw": "r", "music": music });
            reply_from_wire(reply.to_string().as_bytes()).unwrap()
        };
        let reply = served(music_to_wire(&sample_music())).expect("a whole reply");
        assert_eq!(reply.music.expect("music").staves.len(), 2);
        assert!(served(json!({ "staves": 1 })).is_none());

        let plain = json!({ "ok": true, "markdown": "m", "raw": "r" });
        let reply = reply_from_wire(plain.to_string().as_bytes()).unwrap();
        assert!(reply.expect("a whole reply").music.is_none());
    }

    #[test]
    fn a_served_error_reads_as_the_inline_one_does() {
        let errors = [
            FocrError::Usage("bad flag".into()),
            FocrError::ModelNotFound("gone".into()),
            FocrError::InputDecode("not an image".into()),
            FocrError::Timeout("forward stage exceeded 1ms budget".into()),
            FocrError::FormatMismatch("newer format".into()),
            FocrError::Cancelled,
            FocrError::Other(anyhow::anyhow!("broke")),
        ];
        for error in errors {
            let served = wire_error(error.exit_code(), wire_message(&error));
            assert_eq!(served.to_string(), error.to_string());
            assert_eq!(served.exit_code(), error.exit_code());
        }
    }

    #[test]
    fn an_acknowledged_request_may_outlast_the_ack_wait() {
        let mut stream = client_of(|mut stream| {
            stream.write_all(ACK).unwrap();
            std::thread::sleep(SHORT * 2);
            stream.write_all(b"{\"ok\":true}\n").unwrap();
        });
        let reply = read(&mut stream, (SHORT, LONG));
        assert_eq!(reply.as_deref(), Some(&b"{\"ok\":true}"[..]));
    }

    #[test]
    fn an_acknowledged_request_is_still_bounded_by_the_reply_wait() {
        let mut stream = client_of(|mut stream| {
            stream.write_all(ACK).unwrap();
            std::thread::sleep(HOLD);
            drop(stream);
        });
        let started = Instant::now();
        assert_eq!(read(&mut stream, (SHORT, SHORT * 2)), None);
        assert!(started.elapsed() < HOLD);
    }

    #[test]
    fn an_acknowledgement_and_a_reply_in_one_read_are_both_taken() {
        let mut stream = client_of(|mut stream| {
            stream
                .write_all(&[ACK, b"{\"ok\":true}\n"].concat())
                .unwrap();
            std::thread::sleep(HOLD);
        });
        let reply = read(&mut stream, (SHORT, LONG));
        assert_eq!(reply.as_deref(), Some(&b"{\"ok\":true}"[..]));
    }

    #[test]
    fn a_refusal_needs_no_acknowledgement() {
        let refusal = b"{\"ok\":false,\"kind\":\"version\"}\n";
        let mut stream = client_of(move |mut stream| {
            stream.write_all(refusal).unwrap();
            std::thread::sleep(HOLD);
        });
        let reply = read(&mut stream, (SHORT, LONG));
        assert_eq!(reply.as_deref(), Some(&refusal[..refusal.len() - 1]));
    }

    /// Nothing receives from the queue here, as when the worker is mid-page,
    /// and a client is still answered.
    #[test]
    fn a_busy_worker_does_not_delay_the_acknowledgement() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let (queue, _connections) = std::sync::mpsc::sync_channel(ACCEPT_QUEUE);
        std::thread::spawn(move || accept_connections(&listener, &queue));
        for _ in 0..2 {
            let mut stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).unwrap();
            stream.set_read_timeout(Some(LONG)).unwrap();
            let mut line = vec![0_u8; ACK.len()];
            stream.read_exact(&mut line).unwrap();
            assert!(is_acknowledgement(line.trim_ascii_end()), "{line:?}");
        }
    }

    #[test]
    fn wire_errors_keep_their_exit_class() {
        let cases: &[(i32, i32)] = &[
            (EXIT_USAGE, EXIT_USAGE),
            (EXIT_MODEL_NOT_FOUND, EXIT_MODEL_NOT_FOUND),
            (EXIT_INPUT_DECODE, EXIT_INPUT_DECODE),
            (EXIT_TIMEOUT, EXIT_TIMEOUT),
            (EXIT_CANCELLED, EXIT_CANCELLED),
            (EXIT_FORMAT_MISMATCH, EXIT_FORMAT_MISMATCH),
            (crate::error::EXIT_GENERIC, crate::error::EXIT_GENERIC),
            (99, crate::error::EXIT_GENERIC),
        ];
        for &(wire, expected) in cases {
            assert_eq!(wire_error(wire, "x".into()).exit_code(), expected);
        }
    }

    #[test]
    fn options_round_trip_through_the_wire_shape() {
        let request = ResidentRequest {
            image: Path::new("/tmp/x.png"),
            model: PathBuf::from("/tmp/m.focrq"),
            decode: DecodeOverrides {
                max_length: Some(4096),
                temperature: Some(0.5),
                no_repeat_ngram: None,
                ngram_window: Some(512),
                max_page_tokens: Some(0),
            },
            preprocess: PreprocessOverrides {
                base_size: Some(1024),
                image_size: None,
                gundam: Some(true),
            },
            format: true,
            square_pad: true,
            question: Some("what?"),
            on_stage: None,
        };
        let wire = options_value(&request);
        assert_eq!(wire["square_pad"], true);
        let decode = decode_overrides_from_wire(&wire);
        assert_eq!(decode.max_length, Some(4096));
        assert_eq!(decode.temperature, Some(0.5));
        assert_eq!(decode.no_repeat_ngram, None);
        assert_eq!(decode.ngram_window, Some(512));
        assert_eq!(decode.max_page_tokens, Some(0));
        let preprocess = preprocess_overrides_from_wire(&wire);
        assert_eq!(preprocess.base_size, Some(1024));
        assert_eq!(preprocess.image_size, None);
        assert_eq!(preprocess.gundam, Some(true));
        assert_eq!(wire.get("format").and_then(Value::as_bool), Some(true));
        assert_eq!(wire.get("question").and_then(Value::as_str), Some("what?"));
    }

    #[test]
    fn load_fingerprint_ignores_forward_read_options() {
        let base = json!({
            "decode": {"max_length": 100},
            "preprocess": {"gundam": null},
            "format": false,
            "question": null,
        });
        let mut format_flipped = base.clone();
        format_flipped["format"] = json!(true);
        assert_eq!(load_fingerprint(&base), load_fingerprint(&format_flipped));
        let mut decode_changed = base.clone();
        decode_changed["decode"]["max_length"] = json!(200);
        assert_ne!(load_fingerprint(&base), load_fingerprint(&decode_changed));
    }

    /// The daemon protocol refuses a bad token and answers nothing else.
    #[test]
    fn daemon_refuses_bad_token() {
        use std::io::{BufRead, BufReader};

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut resident = None;
            handle_connection(
                stream,
                Path::new("/nonexistent/model.focrq"),
                "right-token",
                0,
                &mut resident,
            );
        });
        let mut stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).unwrap();
        let request = json!({
            "protocol": PROTOCOL,
            "op": "ocr",
            "token": "wrong-token",
            "version": env!("CARGO_PKG_VERSION"),
            "model_root": "/nonexistent/model.focrq",
            "image": "/tmp/x.png",
            "options": {},
            "env": {},
        });
        stream.write_all(format!("{request}\n").as_bytes()).unwrap();
        let mut reply = String::new();
        BufReader::new(&mut stream).read_line(&mut reply).unwrap();
        let value: Value = serde_json::from_str(&reply).unwrap();
        assert_eq!(value.get("ok").and_then(Value::as_bool), Some(false));
        assert_eq!(value.get("kind").and_then(Value::as_str), Some("auth"));
        handle.join().unwrap();
    }

    /// An env-fingerprint mismatch refuses (client runs inline) rather than
    /// serving with the wrong inference environment.
    #[test]
    fn daemon_refuses_mismatched_env_fingerprint() {
        use std::io::{BufRead, BufReader};

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut resident = None;
            handle_connection(
                stream,
                Path::new("/nonexistent/model.focrq"),
                "tok",
                0,
                &mut resident,
            );
        });
        let mut stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).unwrap();
        let request = json!({
            "protocol": PROTOCOL,
            "op": "ocr",
            "token": "tok",
            "version": env!("CARGO_PKG_VERSION"),
            "model_root": "/nonexistent/model.focrq",
            "image": "/tmp/x.png",
            "options": {},
            "env": {"FOCR_DECODE_INT8": "1", "__focr_test_unlikely": "x"},
        });
        stream.write_all(format!("{request}\n").as_bytes()).unwrap();
        let mut reply = String::new();
        BufReader::new(&mut stream).read_line(&mut reply).unwrap();
        let value: Value = serde_json::from_str(&reply).unwrap();
        assert_eq!(value.get("ok").and_then(Value::as_bool), Some(false));
        assert_eq!(value.get("kind").and_then(Value::as_str), Some("env"));
        handle.join().unwrap();
    }

    /// U2-A7: a daemon keyed to an Unlimited-OCR-2 checkpoint refuses
    /// transport-shaped, before building an engine; the client runs it
    /// inline. (The client's own skip is pinned by the e2e CLI matrix: no
    /// daemon state appears.)
    #[test]
    fn daemon_refuses_unlimited_ocr_2_transport_shaped() {
        use std::io::{BufRead, BufReader};

        let dir =
            std::env::temp_dir().join(format!("franken_ocr_resident_v2_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let index = crate::native_engine::write_sparse_v2_stand_in(&dir);
        assert!(!crate::native_engine::resident_daemon_serves(&index));

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let root = index.clone();
        let handle = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut resident = None;
            handle_connection(stream, &root, "tok", 0, &mut resident);
            assert!(resident.is_none(), "no engine was built for v2");
        });
        let mut stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).unwrap();
        let request = json!({
            "protocol": PROTOCOL,
            "op": "ocr",
            "token": "tok",
            "version": env!("CARGO_PKG_VERSION"),
            "model_root": index.display().to_string(),
            "image": "/nonexistent/page.png",
            "options": {},
            "env": env_fingerprint(),
        });
        stream.write_all(format!("{request}\n").as_bytes()).unwrap();
        let mut reply = String::new();
        BufReader::new(&mut stream).read_line(&mut reply).unwrap();
        let value: Value = serde_json::from_str(&reply).unwrap();
        assert_eq!(value.get("ok").and_then(Value::as_bool), Some(false));
        assert_eq!(value.get("kind").and_then(Value::as_str), Some("model"));
        handle.join().unwrap();
    }
}
