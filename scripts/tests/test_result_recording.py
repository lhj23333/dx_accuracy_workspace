"""Test bookkeeping with synthetic CLI output, without model inference."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_results
import results_store as store
import run_eval


class ResultRecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dxmz-recording-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def rows(self):
        return {row["model"]: row for row in store.read_summary(self.root)}

    def result(self, backend="gpu", score=50):
        return {"model": "model-a", "profile": store.PROFILES[backend], "dataset": "COCO",
                "metrics": [{"name": "mAP", "metric_value": score}, {"name": "mAP50", "metric_value": score+10}]}

    def run_stub(self, backend="gpu", result=None, extra="", code=None):
        if code is None:
            result = self.result(backend) if result is None else result
            filename = f"eval_model-a_{store.PROFILES[backend]}.json"
            code = ("from pathlib import Path\n"
                    "Path('result').mkdir(exist_ok=True)\n"
                    f"Path('result/{filename}').write_text({json.dumps(result)!r})\n"
                    "print('Synthetic result writer; no inference')\n" + extra)
        item = {"model": "model-a", "backend": backend, "profile": store.PROFILES[backend],
                "command": [sys.executable, "-c", code]}
        with redirect_stdout(io.StringIO()):
            run_eval.run_evaluation(item, os.environ.copy(), self.root)

    def test_independent_backends_update_one_model_row(self):
        self.run_stub("gpu")
        self.assertEqual(self.rows()["model-a"]["npu_q_pro_mAP"], "")
        self.run_stub("npu", self.result("npu", 49.5))
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows["model-a"]["delta_mAP_pp"], "-0.5")
        self.assertNotIn("model-b", rows)
        self.assertEqual(len(rows["model-a"]), 7)
        directory = self.root / "results/model-a/gpu"
        self.assertFalse((directory / "record.json").exists())
        self.assertIn("Success", (directory / "eval.log").read_text())
        self.assertFalse((self.root / "runs").exists())

    def test_rerun_missing_json_clears_stale_score_and_preserves_other_backend(self):
        self.run_stub("gpu")
        self.run_stub("npu")
        with self.assertRaises(FileNotFoundError):
            self.run_stub(code="print('No result was saved')")
        row = self.rows()["model-a"]
        self.assertEqual(row["gpu_fp32_mAP"], "")
        self.assertEqual(row["delta_mAP_pp"], "")
        self.assertEqual(row["npu_q_pro_mAP"], "50.0")
        directory = self.root / "results/model-a/gpu"
        self.assertIn("Failed:", (directory / "eval.log").read_text())
        self.assertFalse((directory / "result/eval_model-a_onnx.json").exists())

    def test_nonzero_exit_does_not_publish_metrics_even_with_valid_json(self):
        with self.assertRaises(RuntimeError):
            self.run_stub(extra="raise SystemExit(2)\n")
        self.assertEqual(self.rows()["model-a"]["gpu_fp32_mAP"], "")
        self.assertIn("code 2", (self.root / "results/model-a/gpu/eval.log").read_text())

    def test_caught_metric_failure_is_not_published_as_zero(self):
        with self.assertRaises(RuntimeError):
            self.run_stub(result=self.result(score=0), extra="print('Object detection evaluation failed: synthetic')\n")
        self.assertEqual(self.rows()["model-a"]["gpu_fp32_mAP"], "")

    def test_interrupt_is_logged_and_old_scores_are_cleared(self):
        self.run_stub()
        with patch("run_eval.subprocess.Popen", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_stub()
        self.assertEqual(self.rows()["model-a"]["gpu_fp32_mAP"], "")
        self.assertIn("Interrupted by user", (self.root / "results/model-a/gpu/eval.log").read_text())

    def test_interrupt_stops_running_process(self):
        popen = subprocess.Popen
        launched = []

        class InterruptingOutput:
            def __init__(self, stream):
                self.stream = stream

            def __iter__(self):
                raise KeyboardInterrupt

            def close(self):
                self.stream.close()

        def launch(*args, **kwargs):
            process = popen(*args, **kwargs)
            launched.append(process)
            process.stdout = InterruptingOutput(process.stdout)
            return process

        with patch("run_eval.subprocess.Popen", side_effect=launch):
            with self.assertRaises(KeyboardInterrupt):
                self.run_stub(code="import time; time.sleep(30)")
        self.assertEqual(launched[0].returncode, -signal.SIGTERM)

    def test_invalid_metrics_and_mismatched_results_are_rejected(self):
        path = self.root / "result.json"
        for value in (float("nan"), float("inf"), -1, 101):
            with self.subTest(value=value):
                path.write_text(json.dumps(self.result(score=value)))
                with self.assertRaises(ValueError):
                    store.read_result(path, "model-a", "gpu")
        for field, value in (("profile", "q-pro"), ("model", "other"), ("dataset", "other"), ("error", "failed")):
            result = {**self.result(), field: value}
            path.write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                store.read_result(path, "model-a", "gpu")
        path.write_text(json.dumps(self.result(score=0)))
        self.assertEqual(store.read_result(path, "model-a", "gpu")["mAP"], 0)

    def test_concurrent_updates_do_not_lose_rows(self):
        pairs = [(model, backend) for model in ("model-a", "model-b") for backend in ("gpu", "npu")]

        def write(pair):
            model, backend = pair
            store.update(self.root, model, backend, {"mAP": 50 if backend == "gpu" else 49, "mAP50": 60})

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(write, pairs))
        for row in self.rows().values():
            self.assertEqual(row["delta_mAP_pp"], "-1.0")

    def test_different_sample_counts_clear_other_backend_scores(self):
        self.run_stub("gpu", {**self.result("gpu"), "num_samples": 2})
        self.run_stub("npu", {**self.result("npu", 49), "num_samples": 3})
        row = self.rows()["model-a"]
        self.assertEqual(row["gpu_fp32_mAP"], "")
        self.assertEqual(row["npu_q_pro_mAP"], "49.0")
        self.assertEqual(row["delta_mAP_pp"], "")

    def test_matching_sample_counts_preserve_both_backends(self):
        self.run_stub("gpu", {**self.result("gpu"), "num_samples": 2})
        self.run_stub("npu", {**self.result("npu", 49), "num_samples": 2})
        self.assertEqual(self.rows()["model-a"]["delta_mAP_pp"], "-1.0")

    def test_legacy_results_without_sample_counts_are_not_mixed_with_new_subset(self):
        self.run_stub("gpu")
        self.run_stub("npu", {**self.result("npu"), "num_samples": 2})
        row = self.rows()["model-a"]
        self.assertEqual(row["gpu_fp32_mAP"], "")
        self.assertEqual(row["delta_mAP_pp"], "")

    def test_duplicate_evaluation_is_rejected_before_modifying_outputs(self):
        self.run_stub()
        summary = (self.root / store.SUMMARY_NAME).read_bytes()
        log = self.root / "results/model-a/gpu/eval.log"
        original = log.read_bytes()
        with (log.parent / ".eval.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.assertRaisesRegex(RuntimeError, "already running"):
                self.run_stub()
        self.assertEqual(log.read_bytes(), original)
        self.assertEqual((self.root / store.SUMMARY_NAME).read_bytes(), summary)

    def test_viewing_summary_does_not_write_files(self):
        with patch("compare_results.read_summary", side_effect=lambda: store.read_summary(self.root)):
            with patch("sys.argv", ["compare_results.py"]), redirect_stdout(io.StringIO()):
                compare_results.main()
        self.assertEqual(list(self.root.iterdir()), [])
        self.run_stub()
        path = self.root / store.SUMMARY_NAME
        original = (path.stat().st_mtime_ns, path.read_bytes())
        store.read_summary(self.root)
        self.assertEqual((path.stat().st_mtime_ns, path.read_bytes()), original)

    def test_padded_csv_is_read_without_rewriting_and_updates_preserve_other_rows(self):
        path = self.root / store.SUMMARY_NAME
        original = ("\ufeff" + " , ".join(store.FIELDNAMES) +
                    "\n model-a , 50 , 60 , 49.5 , 59.5 , -0.5 , -0.5 \n"
                    " model-b , 51 , 61 , , , , \n").encode("utf-8")
        path.write_bytes(original)
        before = (path.stat().st_mtime_ns, path.read_bytes())
        self.assertEqual(self.rows()["model-a"]["gpu_fp32_mAP"], "50")
        self.assertEqual((path.stat().st_mtime_ns, path.read_bytes()), before)
        store.update(self.root, "model-a", "npu", {"mAP": 49, "mAP50": 59})
        self.assertEqual(self.rows()["model-a"]["delta_mAP_pp"], "-1.0")
        self.assertEqual(self.rows()["model-b"]["gpu_fp32_mAP"], "51")
        self.assertEqual(len(self.rows()), 2)

    def test_invalid_csv_is_not_overwritten(self):
        path = self.root / store.SUMMARY_NAME
        for original in (b"custom,data\nkeep,this\n",
                         (",".join(store.FIELDNAMES) + "\nmodel-a,1\n").encode(),
                         (",".join(store.FIELDNAMES) + "\nmodel-a,1,2,,,,\n model-a ,1,2,,,,\n").encode()):
            path.write_bytes(original)
            with self.assertRaises(ValueError):
                store.update(self.root, "model-a", "gpu")
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
