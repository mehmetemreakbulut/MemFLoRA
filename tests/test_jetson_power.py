"""Stdlib-only tests: no CUDA, telemetry process, or benchmark execution."""
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("jetson_power", Path(__file__).resolve().parents[1] / "tools/jetson_power.py")
power = importlib.util.module_from_spec(spec)
spec.loader.exec_module(power)


def samples(values):
    return [{"t": float(i), "w": float(w)} for i, w in enumerate(values)]


class PowerTests(unittest.TestCase):
    def test_parse_instantaneous_not_average(self):
        self.assertEqual(power.parse_power("CPU [1%@100] VDD_IN 4567mW/3999mW"), 4.567)
        self.assertEqual(power.parse_power("VDD_IN 4.5W/3.2W"), 4.5)

    def test_missing_rail_not_zero(self):
        self.assertIsNone(power.parse_power("VDD_CPU_GPU_CV 500mW/400mW"))
        self.assertIsNone(power.parse_power("permission denied"))

    def test_constant_power(self):
        got = power.power_window(samples([5] * 14), 0.5, 12.5, 2)
        self.assertEqual(got["energy_j"], 60)
        self.assertEqual(got["mean_w"], 5)
        self.assertEqual(got["samples"], 12)

    def test_interpolated_boundaries(self):
        got = power.power_window(samples([0, 2, 4, 6]), 0.5, 2.5, 2, min_samples=2)
        self.assertEqual(got["energy_j"], 6)

    def test_irregular_samples_time_weighted(self):
        got = power.power_window([{"t": 0, "w": 2}, {"t": 1, "w": 4}, {"t": 4, "w": 4}], 0, 4, 4, min_samples=3)
        self.assertEqual(got["mean_w"], 3.75)

    def test_reject_extrapolation(self):
        with self.assertRaisesRegex(ValueError, "bracket"):
            power.power_window(samples([5] * 14), 0, 14, 2)

    def test_reject_sparse_power(self):
        with self.assertRaisesRegex(ValueError, "few"):
            power.power_window(samples([5] * 4), 0, 3, 2)

    def test_reject_telemetry_gap(self):
        with self.assertRaisesRegex(ValueError, "gap"):
            power.power_window(samples([5] * 14), 0, 12, 0.5)

    def test_reject_invalid_samples(self):
        for values in [[5, math.nan, 5], [5, -1, 5]]:
            with self.assertRaisesRegex(ValueError, "Invalid power"):
                power.power_window(samples(values), 0, 2, 2, min_samples=1)

    def test_reject_duplicate_timestamps(self):
        with self.assertRaisesRegex(ValueError, "strictly"):
            power.power_window([{"t": 0, "w": 5}] * 3, 0, 1, 2)

    def test_idle_subtraction_and_units(self):
        got = power.make_metrics({"mean_w": 3}, {"mean_w": 5, "duration_s": 10, "energy_j": 50}, 100)
        self.assertEqual(got["above_idle_w"], 2)
        self.assertEqual(got["ms_per_update"], 100)
        self.assertEqual(got["total_j_per_update"], 0.5)
        self.assertEqual(got["above_idle_j_per_update"], 0.2)

    def test_negative_difference_not_clipped(self):
        got = power.make_metrics({"mean_w": 6}, {"mean_w": 5, "duration_s": 10, "energy_j": 50}, 100)
        self.assertEqual(got["above_idle_w"], -1)
        self.assertEqual(got["above_idle_j_per_update"], -0.1)

    def test_summary_requires_all_repeats(self):
        r = {"model": "tresnet", "method": "full", "status": "ok", "metrics": dict.fromkeys(power.METRICS, 2)}
        row = power.summary_rows([r], ["tresnet"], ["full"], 3)[0]
        self.assertEqual(row["status"], "incomplete")
        self.assertIsNone(row["total_w"])

    def test_runtime_mode_default_and_mixed_results(self):
        self.assertEqual(power.parse_args([]).runtime_mode, "reference")
        self.assertEqual(power.parse_args(["--runtime-mode", "optimized"]).runtime_mode, "optimized")
        rows = [{"model": "tresnet", "method": "full", "status": "ok",
                 "runtime_mode": mode, "metrics": dict.fromkeys(power.METRICS, 1)}
                for mode in ("reference", "optimized")]
        with self.assertRaisesRegex(ValueError, "Cannot average"):
            power.summary_rows(rows, ["tresnet"], ["full"], 2)

    def test_summary_mean_and_sample_sd(self):
        rows = [{"model": "tresnet", "method": "full", "status": "ok", "metrics": dict.fromkeys(power.METRICS, v)} for v in (1, 3)]
        got = power.summary_rows(rows, ["tresnet"], ["full"], 2)[0]
        self.assertEqual(got["total_w"], 2)
        self.assertAlmostEqual(got["total_w_sd"], math.sqrt(2))
        single = power.summary_rows(rows[:1], ["tresnet"], ["full"], 1)[0]
        self.assertIsNone(single["total_w_sd"])

    def test_bad_durations_fail_before_launch(self):
        for args in (["--seconds", "0"], ["--seconds", "nan"], ["--idle-seconds", "1"], ["--settle-seconds", "-1"], ["--methods", "full", "full"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    power.parse_args(args)

    def test_dry_run_does_not_import_torch(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(power.main(["--dry-run"]), 0)
        self.assertEqual(len(output.getvalue().splitlines()), 10)

    def test_timed_block_counts_complete_updates_and_synchronizes(self):
        update = Mock(return_value=Mock(item=Mock(return_value=1.0)))
        sync = Mock()
        with patch.object(power.time, "monotonic", side_effect=[10, 10, 11, 11]):
            start, end, count = power.timed_updates(update, sync, 0.5, 2)
        self.assertEqual((start, end, count), (10, 11, 2))
        self.assertEqual(update.call_count, 2)
        self.assertEqual(sync.call_count, 2)

    def test_nonfinite_loss_rejects_performance_result(self):
        update = Mock(return_value=Mock(item=Mock(return_value=math.nan)))
        with patch.object(power.time, "monotonic", side_effect=[0, 0]):
            with self.assertRaisesRegex(RuntimeError, "Non-finite"):
                power.timed_updates(update, Mock(), 1, 1)


if __name__ == "__main__":
    unittest.main()
