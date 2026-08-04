# Gauntlet Integrity Audit

Date: 2026-07-10
Bead: `bd-2mo.30.1`
Disposition: unresolved, release readiness must remain red

Fresh-eyes tracing from measurement producers through row bundling and strict
certificate verification found these fail-closed gaps:

1. Timing aggregates were internally self-consistent but were not recomputed
   from the raw focr and reference observations.
2. CER receipts carried claimed aggregate values without cryptographic bindings
   to the exact reference and hypothesis texts used to compute them.
3. A measured executable hash was recorded after invocation but was not bound to
   a trusted current-source build receipt.
4. Reference provenance did not bind every load-bearing model/config/tokenizer
   and remote-code file, and reference determinism was not a hard eligibility
   condition.
5. Several evidence readers lacked complete count, size, path-containment, and
   malformed-structure limits.

The current certificate correctly remains red. Closure requires adversarial
self-tests proving that each mismatch is rejected, plus a usable producer path
for generating the newly required evidence rather than a verifier-only schema.
