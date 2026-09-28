#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; run with python -m unittest discover -s tools/tests -v."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark_mqa_suite as suite  # noqa: E402


def passed_row(value=2.0, signature="same"):
    return {
        "quant": "fp8",
        "status": "PASS",
        "comparison_signature": signature,
        "timing": {p: {"critical_ms_median": value} for p in suite.PHASES},
    }


class SourceAndReportingTests(unittest.TestCase):
    def test_five_sizes_and_argument_rejection(self):
        args = suite.parse_args(["--plan"])
        self.assertEqual(len(args.cases), 5)
        self.assertEqual(args.cases[-1]["batch"], 256)
        self.assertEqual(args.cases[-1]["context"], 1048576)
        self.assertEqual(args.operators, ["dense", "paged"])
        self.assertEqual(args.backends, ["torch", "baseline", "tle"])
        for invalid in (["0x32"], ["4x-1"], ["4x32", "4x32"]):
            with self.assertRaises(ValueError):
                suite.parse_shapes(invalid)

    def test_full_snapshot_is_independent_and_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "src" / "flaggems_vllm"
            (package / "ops").mkdir(parents=True)
            (package / "runtime").mkdir()
            (package / "__init__.py").write_text("")
            for name in suite.OPS.values():
                (package / "ops" / f"{name}.py").write_text("original = True\n")
            # Paged imports a dense helper: each process must see its own source tree.
            (package / "ops" / "fp8_fp4_paged_mqa_logits.py").write_text(
                "from flaggems_vllm.ops.fp8_fp4_mqa_logits import original\n"
            )
            helper = package / "runtime" / "tune.yaml"
            helper.write_text("block: 32\n")
            frozen = root / "original"
            manifest = suite.freeze_source(root / "src", frozen)
            helper.write_text("block: 64\n")
            self.assertEqual(
                suite.verify_snapshot(frozen)["tree_sha256"], manifest["tree_sha256"]
            )
            copied = frozen / "src" / "flaggems_vllm" / "runtime" / "tune.yaml"
            self.assertEqual(copied.read_text(), "block: 32\n")
            (package / "ops" / "fp8_fp4_mqa_logits.py").write_text("original = False\n")
            code = (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from flaggems_vllm.ops.fp8_fp4_paged_mqa_logits import original; "
                "print(original)"
            )
            for src, expected in ((frozen / "src", "True"), (root / "src", "False")):
                imported = subprocess.check_output(
                    [sys.executable, "-c", code, str(src)], text=True
                ).strip()
                self.assertEqual(imported, expected)
            with self.assertRaises(FileExistsError):
                suite.freeze_source(root / "src", frozen)
            copied.write_text("block: 128\n")
            with self.assertRaisesRegex(ValueError, "modified"):
                suite.verify_snapshot(frozen)

    def test_observed_launcher_and_restoration_after_exception(self):
        module = SimpleNamespace(launch=lambda: 42)
        original = module.launch
        value, count = suite.observed_call(module, "launch", lambda: module.launch())
        self.assertEqual((value, count), (42, 1))
        self.assertIs(module.launch, original)
        self.assertEqual(suite.observed_call(module, "launch", lambda: 7), (7, 0))
        self.assertEqual(suite.observed_call(module, "absent", lambda: 7), (7, None))

        def fail():
            module.launch()
            raise RuntimeError("candidate failed")

        with self.assertRaises(RuntimeError):
            suite.observed_call(module, "launch", fail)
        self.assertIs(module.launch, original)

    def test_fallback_unverified_and_empty_ranks_do_not_claim_tle(self):
        def check(calls, empty=False):
            return {
                "local": {"passed": True},
                "global_samples": {"passed": True},
                "empty_shard": empty,
                "tle_launcher_calls": calls,
            }

        self.assertEqual(suite.row_status([check(1), check(None, True)], "tle"), "PASS")
        self.assertEqual(suite.row_status([check(1), check(0)], "tle"), "FALLBACK")
        self.assertEqual(suite.row_status([check(None)], "tle"), "UNVERIFIED_TLE")
        self.assertEqual(suite.row_status([check(1)], "baseline"), "BASELINE_USED_TLE")
        bad = check(1)
        bad["global_samples"]["passed"] = False
        self.assertEqual(suite.row_status([bad], "tle"), "INCORRECT")

    def test_ratios_require_valid_matching_measurements(self):
        rows = {"torch": passed_row(8), "baseline": passed_row(4), "tle": passed_row(2)}
        ratios = suite.comparison(rows, "end_to_end")
        self.assertEqual(ratios["baseline_vs_torch"], 2)
        self.assertEqual(ratios["tle_vs_torch"], 4)
        self.assertEqual(ratios["tle_vs_baseline"], 2)
        rows["tle"]["status"] = "FALLBACK"
        self.assertIsNone(suite.comparison(rows, "end_to_end")["tle_vs_baseline"])
        rows["tle"] = passed_row(signature="different input or environment")
        self.assertFalse(suite.comparison(rows, "end_to_end")["comparable"])
        self.assertIsNone(suite.comparison(rows, "end_to_end")["baseline_vs_torch"])
        self.assertIsNone(suite.finite_latency(passed_row(float("nan")), "end_to_end"))
        self.assertIsNone(suite.finite_latency(passed_row(0), "end_to_end"))

    def test_incomplete_or_failed_process_group_cannot_be_overall_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = {"id": "small", "batch": 1, "context": 16}
            jobs = [
                {
                    "id": b,
                    "case": case,
                    "operator": "dense",
                    "backend": b,
                    "report": f"{b}.json",
                    "log": f"{b}.log",
                }
                for b in suite.BACKENDS
            ]
            run = {
                "jobs": jobs,
                "cases": [case],
                "operators": ["dense"],
                "quants": ["fp8", "fp4"],
                "settings": {"device": "cpu", "nproc_per_node": 2},
            }
            suite.write_json(root / "run.json", run)
            for backend in suite.BACKENDS:
                suite.write_json(
                    root / f"{backend}.json",
                    {"status": "PASS", "results": [passed_row()]},
                )
            summary = suite.summarize(root)
            self.assertEqual(summary["status"], "ERROR")
            self.assertTrue(
                any(i["status"] == "NOT_COMPLETED" for i in summary["issues"])
            )
            run["quants"] = ["fp8"]
            suite.write_json(root / "run.json", run)
            suite.write_json(
                root / "tle.json",
                {
                    "status": "ERROR",
                    "error": "worker failed at shutdown",
                    "results": [passed_row()],
                },
            )
            self.assertEqual(suite.summarize(root)["status"], "ERROR")
            self.assertTrue((root / "summary.csv").is_file())
            self.assertIn("CPU 检查模式", (root / "summary.md").read_text())
            saved = json.loads((root / "summary.json").read_text())
            self.assertEqual(len(saved["records"]), 3)

            suite.write_json(
                root / "tle.json", {"status": "PASS", "results": [passed_row(0)]}
            )
            invalid = suite.summarize(root)
            self.assertEqual(invalid["status"], "ERROR")
            self.assertTrue(
                any(i["status"] == "INVALID_TIMING" for i in invalid["issues"])
            )
            run["status"] = "RUNNING"
            suite.write_json(root / "run.json", run)
            (root / "tle.json").unlink()
            pending = suite.summarize(root)
            self.assertEqual(pending["status"], "RUNNING")
            self.assertTrue(any(i["status"] == "PENDING" for i in pending["issues"]))


class DenseContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mqa_suite_inputs
        import torch

        cls.torch, cls.inputs = torch, mqa_suite_inputs
        torch.set_num_threads(1)

    def args(self, context):
        return SimpleNamespace(batch=3, context=context, page_size=16, heads=16, seed=7)

    def test_shards_reconstruct_same_dense_inputs_and_reference(self):
        torch, inputs = self.torch, self.inputs
        for quant in ("fp8", "fp4"):
            for context, world in ((70, 2), (17, 4)):
                with self.subTest(quant=quant, context=context, world=world):
                    args = self.args(context)
                    whole, _ = inputs.make_dense_inputs(args, quant, 0, 1, "cpu")
                    outputs, keys = [], []
                    for rank in range(world):
                        data, plan = inputs.make_dense_inputs(
                            args, quant, rank, world, "cpu"
                        )
                        result = inputs.make_dense_torch_call(
                            data, 16, compiled=False
                        )()
                        reference = inputs.dense_reference(data, 13)
                        torch.testing.assert_close(
                            result, reference, rtol=1e-5, atol=1e-5
                        )
                        self.assertTrue(
                            torch.equal(
                                data.q.view(torch.uint8), whole.q.view(torch.uint8)
                            )
                        )
                        outputs.append(result[:, : plan["length"]])
                        keys.append(data.k[: plan["length"]].view(torch.uint8))
                    self.assertTrue(
                        torch.equal(
                            torch.cat(keys), whole.k[:context].view(torch.uint8)
                        )
                    )
                    positions = list(range(context))
                    expected = inputs.dense_sample_reference(args, quant, positions)
                    torch.testing.assert_close(
                        torch.cat(outputs, 1), expected, rtol=1e-5, atol=1e-5
                    )

    def test_hand_computed_relu_before_negative_weight_and_fp4_decode(self):
        torch, inputs = self.torch, self.inputs
        for quant in ("fp8", "fp4"):
            if quant == "fp8":
                q = torch.zeros(1, 1, 2, 128)
                q[0, 0, :, 0] = torch.tensor([1.0, -1.0])
                q, scales = q.to(torch.float8_e4m3fn), None
            else:
                q = torch.zeros(1, 1, 2, 64, dtype=torch.uint8)
                q[0, 0, :, 0] = torch.tensor([2, 10], dtype=torch.uint8)
                scales = torch.full((1, 1, 2), 0x7F7F7F7F, dtype=torch.int32)
            k = torch.zeros(2, 128)
            k[:, 0] = torch.tensor([1.0, -2.0])
            data = inputs.DenseInputs(
                quant,
                q,
                scales,
                k.to(torch.float8_e4m3fn),
                torch.ones(2),
                torch.tensor([[2.0, -3.0]]),
                torch.zeros(1, dtype=torch.int32),
                torch.tensor([2], dtype=torch.int32),
                [2],
                2,
            )
            actual = inputs.make_dense_torch_call(data, 1, compiled=False)()
            torch.testing.assert_close(
                actual, torch.tensor([[2.0, -6.0]]), rtol=0, atol=0
            )
            torch.testing.assert_close(
                inputs.dense_reference(data, 1), actual, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
