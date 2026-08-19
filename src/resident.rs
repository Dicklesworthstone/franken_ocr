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
use crate::native_engine::{DecodeOverrides, PreprocessOverrides};
use crate::{LayoutSpan, OcrEngine, RecognizedDocument};

/// Default idle unload period. Overridable through `FOCR_RESIDENT_IDLE_SECS`
/// (sub-second daemons are for tests; the floor is 1 second).
const DEFAULT_IDLE: Duration = Duration::from_secs(600);

/// How long the client waits for a reply. A 3B-parameter VLM page on CPU is
/// minutes, not seconds; the default mirrors the engine's own 10-minute
/// forward-stage budget. `FOCR_RESIDENT_CLIENT_TIMEOUT_SECS` overrides it.
const DEFAULT_CLIENT_READ_TIMEOUT: Duration = Duration::from_secs(600);

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

const PROTOCOL: u64 = 1;

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
        .map_or(DEFAULT_CLIENT_READ_TIMEOUT, Duration::from_secs)
}

fn spawn_wait() -> Duration {
    std::env::var("FOCR_RESIDENT_SPAWN_WAIT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map_or(DEFAULT_SPAWN_WAIT, Duration::from_secs)
}

fn idle_period() -> Duration {
    std::env::var("FOCR_RESIDENT_IDLE_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&seconds| seconds > 0)
        .map_or(DEFAULT_IDLE, Duration::from_secs)
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
    /// SmolVLM2 `--task describe` question (forward-read; set per request).
    pub question: Option<&'a str>,
}

fn options_value(request: &ResidentRequest<'_>) -> Value {
    json!({
        "decode": {
            "max_length": request.decode.max_length,
            "temperature": request.decode.temperature,
            "no_repeat_ngram": request.decode.no_repeat_ngram,
            "ngram_window": request.decode.ngram_window,
        },
        "preprocess": {
            "base_size": request.preprocess.base_size,
            "image_size": request.preprocess.image_size,
            "gundam": request.preprocess.gundam,
        },
        "format": request.format,
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
/// transport-shaped failure is `Ok(None)`.
pub fn try_recognize(request: &ResidentRequest<'_>) -> FocrResult<Option<RecognizedDocument>> {
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
        // Possibly a live-but-busy daemon (an OCR forward can hold the serial
        // accept loop for minutes). Removing its state here would orphan it —
        // never again findable, idling gigabytes until its timer — while a
        // duplicate spawned beside it. Fall back inline and leave the daemon
        // to answer the next run.
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
    let mut command = Command::new(exe);
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
) -> FocrResult<Option<RecognizedDocument>> {
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

    // Bounded single-line reply read: the reply is one JSON line; the cap
    // refuses a corrupt daemon before it sizes an allocation in this process.
    // The socket's per-syscall timeout is short (see `connect_once`), so this
    // loop can poll cooperative cancellation between reads — Ctrl+C aborts
    // the wait with the same Cancelled/exit-6 contract as the inline decode
    // loop — while the overall deadline still bounds a wedged daemon.
    let overall_deadline = Instant::now() + client_read_timeout();
    let mut buffer: Vec<u8> = Vec::new();
    let mut chunk = [0_u8; 65536];
    let newline = loop {
        crate::cancel_checkpoint()?;
        if Instant::now() >= overall_deadline {
            return Ok(None);
        }
        match stream.read(&mut chunk) {
            Ok(0) => return Ok(None),
            Ok(count) => {
                buffer.extend_from_slice(&chunk[..count]);
                if let Some(position) = buffer.iter().position(|&byte| byte == b'\n') {
                    break position;
                }
                if buffer.len() >= MAX_REPLY_BYTES {
                    return Ok(None);
                }
            }
            // A per-syscall timeout is the expected idle tick while the
            // daemon's forward runs; anything else is transport-shaped.
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) => {}
            Err(_) => return Ok(None),
        }
    };
    let Ok(reply) = serde_json::from_slice::<Value>(&buffer[..newline]) else {
        return Ok(None);
    };
    if reply.get("ok").and_then(Value::as_bool) == Some(true) {
        let markdown = reply
            .get("markdown")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let layout = reply
            .get("layout")
            .and_then(Value::as_array)
            .map(|spans| {
                spans
                    .iter()
                    .filter_map(|span| {
                        let label = span.get("label")?.as_str()?.to_owned();
                        let boxes = span
                            .get("boxes")?
                            .as_array()?
                            .iter()
                            .filter_map(|b| {
                                let coords = b.as_array()?;
                                if coords.len() != 4 {
                                    return None;
                                }
                                let mut out = [0_i64; 4];
                                for (slot, coord) in out.iter_mut().zip(coords) {
                                    *slot = coord.as_i64()?;
                                }
                                Some(out)
                            })
                            .collect();
                        Some(LayoutSpan { label, boxes })
                    })
                    .collect()
            })
            .unwrap_or_default();
        return Ok(Some(RecognizedDocument { markdown, layout }));
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
    listener
        .set_nonblocking(true)
        .map_err(|error| FocrError::Other(anyhow::anyhow!("resident socket mode: {error}")))?;
    eprintln!(
        "focr resident daemon serving {} on 127.0.0.1:{port}",
        model_root.display()
    );

    let idle = idle_period();
    let mut resident: Option<ResidentEngine> = None;
    let mut deadline = Instant::now() + idle;

    loop {
        match listener.accept() {
            Ok((stream, _peer)) => {
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
                // Serve strictly serially; a second client queues in the OS
                // backlog. Panic-isolated: this process exists to hold a
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
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    eprintln!("focr resident daemon idle exit");
                    remove_state_if_ours(&state_file, port);
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(_) => {
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

    match engine
        .engine
        .recognize_with_layout_model(model_root, &image)
    {
        Ok(document) => {
            let layout: Vec<Value> = document
                .layout
                .iter()
                .map(|span| json!({ "label": span.label, "boxes": span.boxes }))
                .collect();
            let reply = json!({
                "ok": true,
                "markdown": document.markdown,
                "layout": layout,
            });
            let _ = stream.write_all(format!("{reply}\n").as_bytes());
            let _ = stream.flush();
        }
        Err(error) => {
            let reply = json!({
                "ok": false,
                "kind": "ocr",
                "exit_code": error.exit_code(),
                "message": error.to_string(),
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
            },
            preprocess: PreprocessOverrides {
                base_size: Some(1024),
                image_size: None,
                gundam: Some(true),
            },
            format: true,
            question: Some("what?"),
        };
        let wire = options_value(&request);
        let decode = decode_overrides_from_wire(&wire);
        assert_eq!(decode.max_length, Some(4096));
        assert_eq!(decode.temperature, Some(0.5));
        assert_eq!(decode.no_repeat_ngram, None);
        assert_eq!(decode.ngram_window, Some(512));
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
}
