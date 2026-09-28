//! Byte-level integrity of committed files whose exact bytes are pinned by
//! sha256 somewhere else in the tree.
//!
//! JSON formatter sweeps (5847dee, b28d97f) re-indented several of these files.
//! The JSON stayed semantically identical, but every sha256 pin on the old bytes
//! broke silently: the gauntlet certification bundle no longer matched its own
//! artifact hashes, and the TrOMR tokenizer fixtures no longer matched the model
//! manifests that ship them. The embedded native-engine manifests already
//! verify their own hashes at load time; this test covers the pins nothing else
//! checks, so a future reformat fails CI instead of landing.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn sha256_hex(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    format!("{:x}", Sha256::digest(&bytes))
}

fn read_text(rel: &str) -> String {
    let path = root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Collect every `{ "artifact": <repo-relative path>, "sha256": <hex> }` pair.
fn collect_artifact_pins(value: &serde_json::Value, out: &mut Vec<(String, String)>) {
    match value {
        serde_json::Value::Object(map) => {
            if let (
                Some(serde_json::Value::String(artifact)),
                Some(serde_json::Value::String(sha)),
            ) = (map.get("artifact"), map.get("sha256"))
            {
                out.push((artifact.clone(), sha.clone()));
            }
            for v in map.values() {
                collect_artifact_pins(v, out);
            }
        }
        serde_json::Value::Array(items) => {
            for v in items {
                collect_artifact_pins(v, out);
            }
        }
        _ => {}
    }
}

#[test]
fn gauntlet_certification_bundle_artifact_hashes_match_committed_bytes() {
    let bundle: serde_json::Value =
        serde_json::from_str(&read_text("docs/gauntlet/bundle/certification_bundle.json"))
            .expect("certification_bundle.json is valid JSON");
    let mut pins = Vec::new();
    collect_artifact_pins(&bundle, &mut pins);
    assert!(
        !pins.is_empty(),
        "certification bundle lists no artifact hashes"
    );

    let mut mismatches = Vec::new();
    for (artifact, expected) in &pins {
        let path = root().join(artifact);
        if !path.is_file() {
            // Pins on generated outputs (e.g. .gauntlet-output) are not committed.
            continue;
        }
        let actual = sha256_hex(&path);
        if &actual != expected {
            mismatches.push(format!(
                "{artifact}: pinned {expected}, committed bytes {actual}"
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "committed gauntlet bundle files no longer match their pinned sha256 \
         (restore the original bytes; do not reformat pinned JSON):\n{}",
        mismatches.join("\n")
    );
}

#[test]
fn tromr_tokenizer_fixture_hashes_match_every_manifest_that_pins_them() {
    let pinning_docs = [
        "models/manifest.json",
        "models/manifest-v2.json",
        "src/native_engine/tromr_lineage_manifest.json",
        "site/model-manifest.js",
        "ios/Sources/ModelStore.swift",
    ];
    let docs: Vec<(&str, String)> = pinning_docs.iter().map(|d| (*d, read_text(d))).collect();

    let mut missing = Vec::new();
    for name in ["lift", "note", "pitch", "rhythm"] {
        let rel = format!("tests/fixtures/tromr/tokenizer_{name}.json");
        let actual = sha256_hex(&root().join(&rel));
        for (doc, text) in &docs {
            if !text.contains(&actual) {
                missing.push(format!("{rel} (sha256 {actual}) is not pinned in {doc}"));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "TrOMR tokenizer fixture bytes drifted from the shipped model manifests \
         (restore the original bytes; do not reformat pinned JSON):\n{}",
        missing.join("\n")
    );
}

#[test]
fn moe_torch_oracle_fixture_hash_matches_fixture_manifest() {
    let actual = sha256_hex(&root().join("tests/fixtures/moe_torch_2_10_cpu.json"));
    assert!(
        read_text("tests/fixtures/MANIFEST.toml").contains(&actual),
        "tests/fixtures/moe_torch_2_10_cpu.json sha256 {actual} is not the value pinned in \
         tests/fixtures/MANIFEST.toml (restore the original bytes; do not reformat pinned JSON)"
    );
}
