#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for diagnostic isolation, timeouts and result claims."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import diagnose_iluvatar_runtime as diagnostic  # noqa: E402


def result_row(probe, status, variant="current"):
    return {
        "probe": probe,
        "status": status,
        "device": 0,
        "variant": variant,
        "library_path": "/example/sdk/lib64:/example/dependency/lib",
    }


class LibraryEnvironmentTests(unittest.TestCase):
    def test_preserves_input_environment_masks_and_dependency_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corex"
            (root / "lib64").mkdir(parents=True)
            (root / "lib").mkdir()
            original = {
                "CUDA_VISIBLE_DEVICES": "3,5",
                "IX_VISIBLE_DEVICES": "device-allocation",
                "LD_LIBRARY_PATH": f"/dependency/lib:{root / 'lib64'}::/other/lib:",
                "LD_PRELOAD": "/existing/preload.so",
            }
            before = dict(original)
            variants = diagnostic.library_environments(original, [root])
            self.assertEqual(original, before)
            self.assertEqual(len(variants), 2)
            self.assertEqual(variants[0], ("current", before, None))
            preferred = variants[1][1]
            self.assertEqual(
                preferred["LD_LIBRARY_PATH"].split(":"),
                [
                    str(root / "lib64"),
                    str(root / "lib"),
                    "/dependency/lib",
                    "/other/lib",
                ],
            )
            for key in ("CUDA_VISIBLE_DEVICES", "IX_VISIBLE_DEVICES", "LD_PRELOAD"):
                self.assertEqual(preferred[key], before[key])
            preferred["CUDA_VISIBLE_DEVICES"] = "changed-child-only"
            variants[0][1]["LD_PRELOAD"] = "changed-current-copy-only"
            self.assertEqual(original, before)

    def test_sdk_roots_deduplicate_resolved_paths_and_skip_missing_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corex-real"
            (root / "lib64").mkdir(parents=True)
            alias = Path(directory) / "corex-alias"
            alias.symlink_to(root, target_is_directory=True)
            empty = Path(directory) / "empty-sdk"
            empty.mkdir()
            variants = diagnostic.library_environments(
                {}, [root, alias, root, empty, Path(directory) / "missing"]
            )
            self.assertEqual(len(variants), 2)
            self.assertEqual(variants[1][2], str(root.resolve()))
            self.assertEqual(variants[1][1]["LD_LIBRARY_PATH"], str(root / "lib64"))


class ProbeResultTests(unittest.TestCase):
    def test_exit_zero_without_final_probe_pass_is_error(self):
        for events in ([], [{"stage": "runtime.initial_sync", "event": "PASS"}]):
            with self.subTest(events=events):
                result = diagnostic.classify(
                    {"status": "EXITED", "returncode": 0}, events
                )
                self.assertEqual(result["status"], "ERROR")

    def test_only_successful_exit_with_final_probe_pass_is_pass(self):
        events = [{"stage": "probe", "event": "PASS"}]
        result = diagnostic.classify({"status": "EXITED", "returncode": 0}, events)
        self.assertEqual(result["status"], "PASS")
        result = diagnostic.classify({"status": "EXITED", "returncode": 1}, events)
        self.assertEqual(result["status"], "ERROR")

    def test_api_error_and_failed_stage_are_preserved(self):
        events = [
            {"stage": "cudaDeviceSynchronize.initial", "event": "BEGIN"},
            {
                "stage": "probe",
                "event": "ERROR",
                "category": "API_ERROR",
                "return_code": 17,
            },
        ]
        result = diagnostic.classify({"status": "EXITED", "returncode": 2}, events)
        self.assertEqual(result["status"], "API_ERROR")
        self.assertEqual(result["last_stage"], "cudaDeviceSynchronize.initial")
        self.assertEqual(result["events"][-1]["return_code"], 17)

    def test_timeout_reaps_own_sleep_process_and_remains_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "sleep.log"
            result = diagnostic.run_command(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                dict(os.environ),
                log,
                timeout=0.1,
            )
            self.assertEqual(result["status"], "TIMEOUT")
            self.assertIsNotNone(result["returncode"])
            with self.assertRaises(ProcessLookupError):
                os.kill(result["pid"], 0)
            classified = diagnostic.classify(
                result,
                [
                    {"stage": "cudaDeviceSynchronize.initial", "event": "BEGIN"},
                    {"stage": "probe", "event": "PASS"},
                ],
            )
            self.assertEqual(classified["status"], "TIMEOUT")

    def test_reader_tolerates_warnings_and_only_accepts_event_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "mixed.log"
            log.write_text(
                'warning from optional dependency\n{"unrelated": true}\n'
                '[]\n{"stage": "probe", "event": "PASS"}\n'
            )
            self.assertEqual(
                diagnostic.read_events(log), [{"stage": "probe", "event": "PASS"}]
            )


class ReportClaimTests(unittest.TestCase):
    def test_runtime_and_driver_passes_do_not_claim_torch_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = {
                "results": [
                    result_row("runtime", "PASS"),
                    result_row("driver", "PASS"),
                    result_row("torch", "TIMEOUT"),
                ]
            }
            diagnostic.write_reports(output, report)
            summary = (output / "summary.txt").read_text()
            self.assertIn("No tested Torch environment completed", summary)
            self.assertNotIn("Torch compute passed in these", summary)
            self.assertNotIn("LD_LIBRARY_PATH=", summary)
            self.assertEqual(json.loads((output / "report.json").read_text()), report)

    def test_only_passing_torch_environment_is_recommended(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = {
                "results": [
                    result_row("torch", "TIMEOUT", "failed-environment"),
                    result_row("runtime", "PASS", "runtime-only"),
                    result_row("torch", "PASS", "working-environment"),
                ]
            }
            diagnostic.write_reports(output, report)
            summary = (output / "summary.txt").read_text()
            recommendations = summary.split(
                "Torch compute passed in these process environments", 1
            )[1]
            self.assertIn("variant=working-environment", recommendations)
            self.assertNotIn("failed-environment", recommendations)
            self.assertNotIn("runtime-only", recommendations)
            self.assertIn("MQA/TLE remains unverified", recommendations)
            self.assertEqual(recommendations.count("LD_LIBRARY_PATH="), 1)


if __name__ == "__main__":
    unittest.main()
