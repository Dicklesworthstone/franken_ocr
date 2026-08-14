#!/usr/bin/env python3
"""Finite authority-state tests for the offline TrOMR exporter."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("gen_tromr_safetensors.py")
SPEC = importlib.util.spec_from_file_location("gen_tromr_safetensors", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load exporter at {SCRIPT}")
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


class ReplayOutcomeTests(unittest.TestCase):
    def classify(
        self,
        *,
        exact_pin_supplied: bool,
        expected_match: bool,
        accepted_comparison_supplied: bool,
        tolerance_match: bool,
    ) -> str:
        return EXPORTER.classify_replay_outcome(
            exact_pin_supplied=exact_pin_supplied,
            expected_match=expected_match,
            accepted_comparison_supplied=accepted_comparison_supplied,
            tolerance_match=tolerance_match,
        )

    def test_exact_pin_match_is_exact_bytes(self) -> None:
        self.assertEqual(
            self.classify(
                exact_pin_supplied=True,
                expected_match=True,
                accepted_comparison_supplied=False,
                tolerance_match=False,
            ),
            "exact_bytes",
        )

    def test_accepted_value_match_is_value_tolerance(self) -> None:
        self.assertEqual(
            self.classify(
                exact_pin_supplied=True,
                expected_match=False,
                accepted_comparison_supplied=True,
                tolerance_match=True,
            ),
            "value_tolerance",
        )

    def test_no_authority_is_unverified_not_exact(self) -> None:
        self.assertEqual(
            self.classify(
                exact_pin_supplied=False,
                expected_match=False,
                accepted_comparison_supplied=False,
                tolerance_match=False,
            ),
            "unverified",
        )

    def test_failed_exact_pin_is_mismatch(self) -> None:
        self.assertEqual(
            self.classify(
                exact_pin_supplied=True,
                expected_match=False,
                accepted_comparison_supplied=False,
                tolerance_match=False,
            ),
            "mismatch",
        )

    def test_failed_accepted_comparison_is_mismatch(self) -> None:
        self.assertEqual(
            self.classify(
                exact_pin_supplied=False,
                expected_match=False,
                accepted_comparison_supplied=True,
                tolerance_match=False,
            ),
            "mismatch",
        )


if __name__ == "__main__":
    unittest.main()
