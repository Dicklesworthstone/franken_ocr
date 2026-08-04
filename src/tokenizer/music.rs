//! E6 (bd-3jo6.5.6): the TrOMR music tokenizer — four DECODE-ONLY WordLevel
//! tables (`tokenizer_{rhythm,pitch,lift,note}.json`, committed upstream and
//! shipped beside the artifact), the fourth — and by far the simplest —
//! tokenizer family in the zoo (tromr-spec §9).
//!
//! WordLevel means: no merges, no normalizer, no byte fallback — the id→token
//! table IS the whole tokenizer. Inference needs only decode; encode is a
//! training/fixture-side concern and deliberately does not exist here.
//!
//! ## The detokenize contract (upstream `staff2score.py::detokenize`, pinned)
//!
//! Per id, in order: `convert_ids_to_tokens` (out-of-range ⇒ `None`) →
//! `None`→`""` → `replace('Ġ', ' ')` then `strip()` (a no-op on these tables,
//! carried verbatim from the upstream code) → DELETE the literal token strings
//! `[BOS]`/`[EOS]`/`[PAD]` — by TOKEN, not by id. Two census subtleties the
//! goldens lock in (`tests/fixtures/tromr/detokenize_goldens.json`):
//!
//! * only the RHYTHM table contains the specials; pitch/lift/note ids 0..2 are
//!   real tokens (`nonote`, `note-C0`, …) and MUST survive decoding;
//! * an out-of-range id decodes to `""` and is KEPT (empty string is not a
//!   special) — exactly what upstream emits.

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::{FocrError, FocrResult};

/// The four parallel TrOMR output streams (spec §4/§9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    /// 260 ids — carries the specials ([PAD] 0, [BOS] 1, [EOS] 2) + `+`/`|`/
    /// clefs/keys/rests/notes/time signatures. EOS on THIS stream stops decode.
    Rhythm,
    /// 71 ids — `nonote` + `note-{C..B}{0..9}` (accidentals live in lift).
    Pitch,
    /// 7 ids — `nonote` + the 6 lift/accidental classes.
    Lift,
    /// 2 ids — `nonote`/`note` (train-time consistency head; inference-dead
    /// upstream, decoded here only for completeness).
    Note,
}

/// One decode-only WordLevel table: dense id→token.
#[derive(Debug)]
struct WordLevelTable {
    id_to_token: Vec<String>,
}

impl WordLevelTable {
    /// Parse one HF `tokenizers` JSON: `model.type` must be `"WordLevel"` and
    /// `model.vocab` (token→id) must form a DENSE, duplicate-free id space —
    /// gaps or dupes mean the table is not the censused upstream file.
    fn from_json(text: &str, origin: &str) -> FocrResult<Self> {
        let doc: serde_json::Value = serde_json::from_str(text).map_err(|e| {
            FocrError::FormatMismatch(format!("{origin}: not valid tokenizer JSON: {e}"))
        })?;
        let model = &doc["model"];
        if model["type"].as_str() != Some("WordLevel") {
            return Err(FocrError::FormatMismatch(format!(
                "{origin}: model.type {:?} is not WordLevel",
                model["type"]
            )));
        }
        let vocab = model["vocab"].as_object().ok_or_else(|| {
            FocrError::FormatMismatch(format!("{origin}: model.vocab is not an object"))
        })?;
        let mut by_id: BTreeMap<u64, &str> = BTreeMap::new();
        for (token, id) in vocab {
            let id = id.as_u64().ok_or_else(|| {
                FocrError::FormatMismatch(format!("{origin}: id for {token:?} is not a u64"))
            })?;
            if by_id.insert(id, token).is_some() {
                return Err(FocrError::FormatMismatch(format!(
                    "{origin}: duplicate id {id}"
                )));
            }
        }
        let n = by_id.len() as u64;
        if by_id.keys().next_back().map(|&k| k + 1) != Some(n) {
            return Err(FocrError::FormatMismatch(format!(
                "{origin}: id space is not dense 0..{n}"
            )));
        }
        Ok(Self {
            id_to_token: by_id.into_values().map(str::to_owned).collect(),
        })
    }

    fn token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(String::as_str)
    }
}

/// The TrOMR music tokenizer: the four stream tables, loaded from the model
/// dir (`tokenizer_{rhythm,pitch,lift,note}.json` beside the `.focrq`, the
/// zoo files-beside convention).
#[derive(Debug)]
pub struct MusicTokenizer {
    rhythm: WordLevelTable,
    pitch: WordLevelTable,
    lift: WordLevelTable,
    note: WordLevelTable,
}

/// The rhythm-stream special ids (spec §9 config anchors; the OTHER streams
/// have no specials — their low ids are real tokens).
pub const PAD_ID: u32 = 0;
/// `[BOS]` on the rhythm stream.
pub const BOS_ID: u32 = 1;
/// `[EOS]` on the rhythm stream — the generate stop condition (spec §5).
pub const EOS_ID: u32 = 2;

impl MusicTokenizer {
    /// Load the four tables from `dir`, validating each against the censused
    /// vocab size (260/71/7/2 — spec §9). Missing file or wrong table shape is
    /// a clean [`FocrError::FormatMismatch`]/IO error, never a fallback.
    ///
    /// # Errors
    /// A missing/unreadable file, malformed JSON, a non-WordLevel model, a
    /// non-dense id space, or a vocab-size mismatch vs the census.
    pub fn from_dir(dir: &Path) -> FocrResult<Self> {
        let load = |stem: &str, want: usize| -> FocrResult<WordLevelTable> {
            let path = dir.join(format!("tokenizer_{stem}.json"));
            let text = std::fs::read_to_string(&path).map_err(|e| {
                FocrError::ModelNotFound(format!(
                    "TrOMR tokenizer table missing: {}: {e}",
                    path.display()
                ))
            })?;
            let table = WordLevelTable::from_json(&text, &path.display().to_string())?;
            if table.id_to_token.len() != want {
                return Err(FocrError::FormatMismatch(format!(
                    "{}: {} ids, census expects {want} (spec §9)",
                    path.display(),
                    table.id_to_token.len()
                )));
            }
            Ok(table)
        };
        Ok(Self {
            rhythm: load("rhythm", 260)?,
            pitch: load("pitch", 71)?,
            lift: load("lift", 7)?,
            note: load("note", 2)?,
        })
    }

    fn table(&self, stream: Stream) -> &WordLevelTable {
        match stream {
            Stream::Rhythm => &self.rhythm,
            Stream::Pitch => &self.pitch,
            Stream::Lift => &self.lift,
            Stream::Note => &self.note,
        }
    }

    /// Raw id→token lookup on one stream (`None` = out of range, exactly the
    /// upstream `convert_ids_to_tokens` behavior).
    #[must_use]
    pub fn token(&self, stream: Stream, id: u32) -> Option<&str> {
        self.table(stream).token(id)
    }

    /// The pinned upstream detokenize (`staff2score.py::detokenize`): per id,
    /// `None`→`""`, `Ġ`→space + trim, then DROP the literal specials
    /// `[BOS]`/`[EOS]`/`[PAD]` (by token string — pitch/lift/note low ids are
    /// real tokens and survive; an out-of-range `""` is kept).
    #[must_use]
    pub fn detokenize(&self, stream: Stream, ids: &[u32]) -> Vec<String> {
        ids.iter()
            .filter_map(|&id| {
                let tok = self.token(stream, id).unwrap_or("");
                let cleaned = tok.replace('Ġ', " ");
                let cleaned = cleaned.trim();
                (!matches!(cleaned, "[BOS]" | "[EOS]" | "[PAD]")).then(|| cleaned.to_owned())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tromr")
    }

    fn load() -> MusicTokenizer {
        MusicTokenizer::from_dir(&fixture_dir()).expect("committed tables load")
    }

    #[test]
    fn tables_match_the_census_layout() {
        // Spec §9 exact-id anchors across all four streams.
        let tk = load();
        for (id, want) in [
            (0, "[PAD]"),
            (1, "[BOS]"),
            (2, "[EOS]"),
            (3, "+"),
            (4, "|"),
            (5, "barline"),
        ] {
            assert_eq!(tk.token(Stream::Rhythm, id), Some(want));
        }
        assert_eq!(tk.token(Stream::Pitch, 0), Some("nonote"));
        assert_eq!(tk.token(Stream::Pitch, 1), Some("note-C0"));
        assert_eq!(tk.token(Stream::Pitch, 70), Some("note-B9"));
        assert_eq!(tk.token(Stream::Lift, 0), Some("nonote"));
        assert_eq!(tk.token(Stream::Lift, 1), Some("lift_null"));
        assert_eq!(tk.token(Stream::Lift, 6), Some("lift_N"));
        assert_eq!(tk.token(Stream::Note, 0), Some("nonote"));
        assert_eq!(tk.token(Stream::Note, 1), Some("note"));
        // Out of range = None on every stream (the upstream contract).
        assert_eq!(tk.token(Stream::Note, 2), None);
        assert_eq!(tk.token(Stream::Rhythm, 260), None);
        // The special-id constants are the rhythm anchors.
        assert_eq!(tk.token(Stream::Rhythm, PAD_ID), Some("[PAD]"));
        assert_eq!(tk.token(Stream::Rhythm, BOS_ID), Some("[BOS]"));
        assert_eq!(tk.token(Stream::Rhythm, EOS_ID), Some("[EOS]"));
    }

    #[test]
    fn detokenize_matches_the_oracle_goldens() {
        // tests/fixtures/tromr/detokenize_goldens.json — generated 2026-07-05
        // by the HF `tokenizers` WordLevel oracle over the SAME committed
        // tables, applying the upstream staff2score.py arithmetic.
        let tk = load();
        let text = std::fs::read_to_string(fixture_dir().join("detokenize_goldens.json"))
            .expect("goldens committed");
        let gold: serde_json::Value = serde_json::from_str(&text).expect("goldens parse");
        for (name, stream) in [
            ("rhythm", Stream::Rhythm),
            ("pitch", Stream::Pitch),
            ("lift", Stream::Lift),
            ("note", Stream::Note),
        ] {
            let entry = &gold[name];
            let ids: Vec<u32> = entry["probe_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
                .collect();
            let want: Vec<String> = entry["detokenized"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect();
            assert_eq!(tk.detokenize(stream, &ids), want, "{name}");
            assert!(
                entry["oor_token_is_none"].as_bool().unwrap(),
                "{name}: oracle confirms out-of-range => None"
            );
        }
    }

    #[test]
    fn detokenize_keeps_oor_empty_and_drops_specials_by_token() {
        let tk = load();
        // Rhythm: specials dropped; a real token + an OOR id (kept as "").
        assert_eq!(
            tk.detokenize(Stream::Rhythm, &[BOS_ID, 5, 9999, EOS_ID, PAD_ID]),
            vec!["barline".to_owned(), String::new()]
        );
        // Pitch: ids 0..2 are REAL tokens, none dropped.
        assert_eq!(
            tk.detokenize(Stream::Pitch, &[1, 0, 2]),
            vec![
                "note-C0".to_owned(),
                "nonote".to_owned(),
                "note-D0".to_owned()
            ]
        );
    }

    #[test]
    fn loader_error_paths_are_clean() {
        // Missing dir → ModelNotFound-class error, never a fallback table.
        let err = MusicTokenizer::from_dir(std::path::Path::new("/nonexistent/tromr"))
            .expect_err("missing tables must error");
        assert!(matches!(err, FocrError::ModelNotFound(_)), "{err:?}");

        // A structurally wrong table refuses with FormatMismatch.
        for (bad, why) in [
            (r#"{"model":{"type":"BPE","vocab":{}}}"#, "not WordLevel"),
            (
                r#"{"model":{"type":"WordLevel","vocab":{"a":0,"b":2}}}"#,
                "gap",
            ),
            (
                r#"{"model":{"type":"WordLevel","vocab":"x"}}"#,
                "not an object",
            ),
            ("not json", "not JSON"),
        ] {
            let err = WordLevelTable::from_json(bad, "synthetic").expect_err(why);
            assert!(
                matches!(err, FocrError::FormatMismatch(_)),
                "{why}: {err:?}"
            );
        }
    }
}
