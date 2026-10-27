"""Test YAML discovery, minimal preflight, and downloads using local fixtures."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import download
import workspace


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dxmz-workspace-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        configs = self.root / "configs/detectors"
        configs.mkdir(parents=True)
        self.config = {"name": "model-a", "profiles": {"onnx": {"target": "onnx", "runtime": {"device": "gpu"}}},
                       "dataset": {"type": "COCO", "eval_path": "${DATA_ROOT}/COCO"},
                       "artifacts": {"path": "${MODEL_ROOT}/${MODEL_NAME}"}}
        (configs / "model-a.yaml").write_text(json.dumps(self.config))
        (configs / "model-b.yml").write_text(json.dumps({**self.config, "name": "model-b"}))
        self.env = {"MODEL_ROOT": str(self.root / "models"), "DATA_ROOT": str(self.root / "datasets")}
        model = self.root / "models/model-a/model-a.onnx"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"manually replaced model")
        self.coco = self.root / "datasets/COCO"
        (self.coco / "val2017").mkdir(parents=True)
        (self.coco / "val2017/1.jpg").write_bytes(b"image")
        (self.coco / "annotations").mkdir()
        self.annotation = {"images": [{"id": 1, "file_name": "1.jpg"}]}
        (self.coco / "annotations/instances_val2017.json").write_text(json.dumps(self.annotation))

    def test_new_yaml_is_discovered_without_registration(self):
        self.assertEqual(set(workspace.discover_models(self.root)), {"model-a", "model-b"})
        path = self.root / "configs/new-model.yaml"
        path.write_text(json.dumps({**self.config, "name": "new-model"}))
        self.assertEqual(workspace.select_models("new-model", self.root), {"new-model": path})

    def test_selected_gpu_accepts_manual_files_without_other_models_or_npu(self):
        workspace.check_inputs(workspace.select_models("model-a", self.root), ["gpu"], self.env, self.root)
        self.assertFalse((self.root / "sources").exists())
        self.assertFalse((self.root / "manifest.json").exists())

    def test_empty_model_and_cpu_profile_are_rejected(self):
        model = self.root / "models/model-a/model-a.onnx"
        model.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "empty model"):
            workspace.check_inputs(workspace.select_models("model-a", self.root), ["gpu"], self.env, self.root)
        model.write_bytes(b"model")
        config = {**self.config, "profiles": {"onnx": {"target": "onnx", "runtime": {"device": "cpu"}}}}
        (self.root / "configs/detectors/model-a.yaml").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "GPU runtime"):
            workspace.check_inputs(workspace.select_models("model-a", self.root), ["gpu"], self.env, self.root)

    def test_missing_dataset_image_is_rejected(self):
        (self.coco / "val2017/1.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete COCO"):
            workspace.check_coco(self.coco)

    def test_explicit_model_file_and_environment_paths_are_respected(self):
        config = {**self.config, "artifacts": {"onnx": "external/custom.onnx"}}
        self.assertEqual(workspace.model_file(config, "gpu", self.env, self.root), self.root / "external/custom.onnx")
        with patch.dict("os.environ", self.env):
            self.assertEqual(workspace.environment(self.root)["MODEL_ROOT"], self.env["MODEL_ROOT"])

    def test_explicit_backend_file_overrides_shared_artifact_directory(self):
        config = {**self.config, "artifacts": {
            "path": "${MODEL_ROOT}/${MODEL_NAME}", "onnx": "external/custom.onnx"}}
        self.assertEqual(workspace.model_file(config, "gpu", self.env, self.root),
                         self.root / "external/custom.onnx")

    def test_relative_resource_roots_are_absolute_in_evaluator_environment(self):
        with patch.dict("os.environ", {"DATA_ROOT": "datasets", "MODEL_ROOT": "models"}):
            env = workspace.environment(self.root)
        self.assertEqual(env["DATA_ROOT"], str(self.root / "datasets"))
        self.assertEqual(env["MODEL_ROOT"], str(self.root / "models"))

    def test_existing_models_and_dataset_do_not_download_or_create_records(self):
        with patch("download.environment", return_value=self.env), patch("download.subprocess.run") as curl:
            with redirect_stdout(io.StringIO()):
                download.download_models("model-a", "gpu", self.root)
                download.download_dataset(self.root)
            curl.assert_not_called()
        self.assertFalse((self.root / "downloads").exists())
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_successful_download_uses_partial_file_without_metadata(self):
        target = self.root / "new.onnx"

        def write_partial(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_bytes(b"downloaded model")

        with patch("download.subprocess.run", side_effect=write_partial), redirect_stdout(io.StringIO()):
            download.download_file("https://example.invalid/new.onnx", target)
        self.assertEqual(target.read_bytes(), b"downloaded model")
        self.assertFalse(target.with_name("new.onnx.part").exists())
        self.assertFalse((self.root / "sources").exists())

    def test_failed_download_preserves_partial_without_publishing_model(self):
        target = self.root / "new.onnx"

        def fail_download(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_bytes(b"partial")
            raise subprocess.CalledProcessError(22, command)

        with patch("download.subprocess.run", side_effect=fail_download), redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                download.download_file("https://example.invalid/new.onnx", target)
        self.assertFalse(target.exists())
        self.assertEqual(target.with_name("new.onnx.part").read_bytes(), b"partial")

    def test_dataset_download_removes_extracted_archives(self):
        (self.coco / "val2017/1.jpg").unlink()
        cache = self.root / "downloads"
        cache.mkdir()
        with zipfile.ZipFile(cache / "val2017.zip", "w") as archive:
            archive.writestr("val2017/1.jpg", b"image")
        with zipfile.ZipFile(cache / "annotations_trainval2017.zip", "w") as archive:
            archive.writestr("annotations/instances_val2017.json", json.dumps(self.annotation))
        with patch("download.environment", return_value=self.env), patch("download.subprocess.run") as curl:
            with redirect_stdout(io.StringIO()):
                download.download_dataset(self.root)
            curl.assert_not_called()
        workspace.check_coco(self.coco)
        self.assertFalse(cache.exists())


if __name__ == "__main__":
    unittest.main()
