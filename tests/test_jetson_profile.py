"""CPU-only regression tests; no dataset or CUDA execution."""
import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("jetson_profile_under_test", ROOT / "tools/jetson_profile.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def result(model="tresnet", method="full", batch=1, status="ok"):
    return {
        "config": {"model": model, "method": method, "batch": batch, "rank": 2},
        "status": status,
        "checks": {"lora_edge_class_matches_request": True},
        "metrics": {
            "training_state_peak_bytes": 9003836,
            "saved_backward_end_forward_bytes": 758748,
            "optimizer_total_bytes": 4487320,
            "cuda_requested_peak_bytes": 11516408,
        },
    }


class AccountingTests(unittest.TestCase):
    def analyse(self, trace, tags, forward, before=()):
        return profile.analyse_trace(trace, before, 4096, tags, {"forward": forward})["metrics"]

    def test_repeated_references_count_once(self):
        trace = [{"action": "alloc", "addr": 100, "size": 64}]
        tag = {"event": 1, "address": 100, "bytes": 64, "kind": "saved"}
        m = self.analyse(trace, [tag, tag], 1)
        self.assertEqual(m["saved_backward_end_forward_bytes"], 64)

    def test_view_counts_backing_allocation(self):
        trace = [{"action": "alloc", "addr": 100, "size": 64}]
        m = self.analyse(trace, [{"event": 1, "address": 100, "bytes": 16, "kind": "saved"}], 1)
        self.assertEqual(m["training_state_peak_bytes"], 64)

    def test_model_and_input_aliases_not_counted_as_saved(self):
        before = [{"address": 100, "bytes": 64, "owner": "model"},
                  {"address": 200, "bytes": 32, "owner": "input"}]
        tags = [{"event": 0, "address": address, "bytes": size, "kind": "saved"}
                for address, size in [(100, 64), (200, 32)]]
        m = self.analyse([], tags, 0, before)
        self.assertEqual(m["training_state_peak_bytes"], 64)
        self.assertEqual(m["saved_backward_end_forward_bytes"], 0)

    def test_address_reuse_is_a_new_generation(self):
        trace = [{"action": "alloc", "addr": 100, "size": 64},
                 {"action": "free_requested", "addr": 100},
                 {"action": "free_completed", "addr": 100},
                 {"action": "alloc", "addr": 100, "size": 16}]
        tags = [{"event": 1, "address": 100, "bytes": 64, "kind": "saved"},
                {"event": 4, "address": 100, "bytes": 16, "kind": "gradient"}]
        m = self.analyse(trace, tags, 1)
        self.assertEqual(m["training_state_peak_bytes"], 64)
        self.assertEqual(m["parameter_gradients_peak_bytes"], 16)

    def test_concurrent_sum_not_sum_of_component_maxima(self):
        trace = [{"action": "alloc", "addr": 100, "size": 64},
                 {"action": "free_requested", "addr": 100},
                 {"action": "alloc", "addr": 200, "size": 32}]
        tags = [{"event": 1, "address": 100, "bytes": 64, "kind": "saved"},
                {"event": 3, "address": 200, "bytes": 32, "kind": "gradient"}]
        self.assertEqual(self.analyse(trace, tags, 1)["training_state_peak_bytes"], 64)

    def test_sg_replay_overlap_is_measured(self):
        trace = [{"action": "alloc", "addr": 100, "size": 64},
                 {"action": "alloc", "addr": 200, "size": 32}]
        tags = [{"event": i, "address": address, "bytes": size, "kind": "sg_replay_input"}
                for i, address, size in [(1, 100, 64), (2, 200, 32)]]
        m = self.analyse(trace, tags, 0)
        self.assertEqual(m["training_state_peak_bytes"], 96)
        self.assertEqual(m["training_state_core_peak_bytes"], 0)

    def test_unknown_pointer_rejected(self):
        with self.assertRaises(profile.AccountingError):
            self.analyse([], [{"event": 0, "address": 99, "bytes": 1, "kind": "saved"}], 0)

    def test_unmatched_free_rejected(self):
        with self.assertRaises(profile.AccountingError):
            self.analyse([{"action": "free_requested", "addr": 99}], [], 0)


class ComparisonTests(unittest.TestCase):
    def export(self, directory, results, args=None):
        args = args or profile.parser().parse_args([])
        profile.export_results(Path(directory), args, results)
        with (Path(directory) / "table3_comparison.csv").open() as stream:
            return list(csv.DictReader(stream))

    def test_default_grid_and_original_reference(self):
        self.assertEqual(len(profile.sweep_cases(profile.parser().parse_args([]))), 40)
        self.assertEqual(len(profile.PAPER_TABLE3), 10)
        self.assertEqual(profile.PAPER_TABLE3["mobilenetv2", "full"][0][0], (43.20, 16.12))
        self.assertEqual(profile.PAPER_TABLE3["mobilenetv2", "full"][0][1], (109.00, 81.91))
        self.assertEqual(profile.PAPER_TABLE3["mobilenetv2", "memflora_sg"][1], .16)

    def test_empty_sweep_is_pending_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.export(directory, [])
            self.assertEqual(len(rows), 40)
            self.assertTrue(all(r["status"] == "PENDING" for r in rows))
            self.assertTrue(all(r["jetson_peak_bytes"] == "" for r in rows))

    def test_unrounded_bytes_and_printed_reference_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.export(directory, [result()])
            self.assertEqual(rows[0]["jetson_peak_bytes"], "9003836")
            self.assertAlmostEqual(float(rows[0]["delta_peak_mb"]), 9.003836 - 9.02)
            text = (Path(directory) / "table3_comparison.md").read_text()
            self.assertIn("9.00 / 0.76", text)
            self.assertIn("9.02 / 0.90", text)
            self.assertIn("INCOMPLETE", text)

    def test_failed_and_invalid_cases_have_no_numeric_comparison(self):
        for status in ["failed", "invalid"]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                rows = self.export(directory, [result(status=status)])
                self.assertEqual(rows[0]["status"], "FAILED")
                self.assertEqual(rows[0]["delta_peak_mb"], "")

    def test_legacy_lora_edge_requires_implementation_evidence(self):
        r = result(method="lora_edge")
        r.pop("checks")
        with tempfile.TemporaryDirectory() as directory:
            rows = self.export(directory, [r])
            row = next(r for r in rows if r["method"] == "LoRA-Edge" and r["batch"] == "1")
            self.assertEqual(row["status"], "UNVERIFIED IMPLEMENTATION")
            self.assertEqual(row["jetson_peak_bytes"], "")

    def test_rank_mismatch_does_not_compare(self):
        args = profile.parser().parse_args(["--rank", "4"])
        with tempfile.TemporaryDirectory() as directory:
            rows = self.export(directory, [result()], args)
            self.assertTrue(all(r["status"] == "RANK MISMATCH" for r in rows))
            self.assertTrue(all(r["delta_peak_mb"] == "" for r in rows))

    def test_result_rank_checked_independently(self):
        r = result()
        r["config"]["rank"] = 4
        self.assertEqual(profile.comparison_status(r, 2), "RANK MISMATCH")

    def test_subset_does_not_drop_paper_rows(self):
        args = profile.parser().parse_args(["--models", "tresnet", "--methods", "full", "--batches", "1"])
        with tempfile.TemporaryDirectory() as directory:
            rows = self.export(directory, [result()], args)
            self.assertEqual(len(rows), 40)
            self.assertEqual(sum(r["status"] == "ok" for r in rows), 1)

    def test_complete_sg_optimizer_is_total_not_paper_value(self):
        results = [result("mobilenetv2", "memflora_sg", b) for b in profile.PAPER_BATCHES]
        for r in results:
            r["metrics"]["optimizer_total_bytes"] = 296448
        with tempfile.TemporaryDirectory() as directory:
            rows = self.export(directory, results)
            row = next(r for r in rows if r["method"] == "MemFLoRA-SG" and r["backbone"] == "MobileNetV2")
            self.assertEqual(row["paper_optimizer_mb"], "0.16")
            self.assertEqual(row["jetson_optimizer_bytes"], "296448")
            text = (Path(directory) / "table3_comparison.md").read_text()
            line = next(line for line in text.splitlines() if "| MobileNetV2 | MemFLoRA-SG | Jetson |" in line)
            self.assertTrue(line.endswith("| 0.30 |"))

    def test_all_40_results_export_in_paper_order(self):
        results = [result(model, method, batch) for model, method, batch in
                   profile.sweep_cases(profile.parser().parse_args([]))]
        with tempfile.TemporaryDirectory() as directory:
            rows = self.export(directory, list(reversed(results)))
            self.assertEqual(len(rows), 40)
            self.assertTrue(all(r["status"] == "ok" for r in rows))
            self.assertEqual((rows[0]["backbone"], rows[0]["method"], rows[0]["batch"]),
                             ("T-ResNet", "Full FT", "1"))
            self.assertEqual((rows[-1]["backbone"], rows[-1]["method"], rows[-1]["batch"]),
                             ("MobileNetV2", "MemFLoRA-SG", "64"))


class EnvironmentTests(unittest.TestCase):
    def test_dry_run_never_checks_environment(self):
        with patch.object(profile, "check_environment", side_effect=AssertionError("must not run")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(profile.main(["--dry-run"]), 0)

    def test_preflight_failure_does_not_create_output_or_launch_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "new-run"
            with patch.object(profile, "check_environment", return_value=False), patch.object(profile.subprocess, "run", side_effect=AssertionError("must not run")):
                self.assertEqual(profile.main(["--out", str(out)]), 1)
            self.assertFalse(out.exists())

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(profile, "check_environment", side_effect=AssertionError("must not run")), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    profile.main(["--out", directory])

    def test_actionable_preflight_error_and_same_python(self):
        args = profile.parser().parse_args([])
        completed = SimpleNamespace(returncode=1, stdout="", stderr="ModuleNotFoundError: No module named 'torch'")
        output = io.StringIO()
        with patch.object(profile.subprocess, "run", return_value=completed) as run, contextlib.redirect_stderr(output), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(profile.check_environment(args))
        self.assertEqual(run.call_args.args[0][0], sys.executable)
        self.assertIn("--preflight", run.call_args.args[0])
        self.assertIn("No cases were started", output.getvalue())
        self.assertIn("venvs/memflora/bin/python", output.getvalue())

    def test_preflight_success(self):
        completed = SimpleNamespace(returncode=0, stdout="Environment OK", stderr="")
        with patch.object(profile.subprocess, "run", return_value=completed), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(profile.check_environment(profile.parser().parse_args([])))


if __name__ == "__main__":
    unittest.main()
