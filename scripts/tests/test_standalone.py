"""Exercise a relocated workspace using the installed ModelZoo, without inference."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

SYNTHETIC_CLI = '''
import json
from pathlib import Path
import sys
import types
import yaml

def main():
    model = yaml.safe_load(Path(sys.argv[2]).read_text())["name"]
    profile = sys.argv[sys.argv.index("--profile") + 1]
    model_path = Path(sys.argv[sys.argv.index("--model-path") + 1])
    assert model_path.read_bytes() == b"fixture", model_path
    result = {"model": model, "profile": profile, "dataset": "COCO", "num_samples": 1,
              "metrics": [{"name": "mAP", "metric_value": 50},
                          {"name": "mAP50", "metric_value": 60}]}
    Path("result").mkdir(exist_ok=True)
    Path(f"result/eval_{model}_{profile}.json").write_text(json.dumps(result))
    Path("interpreter.json").write_text(json.dumps({"executable": sys.executable, "prefix": sys.prefix}))
    print("Synthetic evaluation; no inference")
    return 0

cli = types.ModuleType("dx_modelzoo.main")
cli.main = main
sys.modules["dx_modelzoo.main"] = cli
policy = types.ModuleType("fp32_policy")
policy.configure = lambda: None
sys.modules["fp32_policy"] = policy
'''


class StandaloneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dxmz-standalone-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "standalone workspace"
        self.root.mkdir()
        for directory in ("scripts", "configs"):
            shutil.copytree(ROOT / directory, self.root / directory, symlinks=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT / "env.sh", self.root / "env.sh")
        self.env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        for variable in ("DATA_ROOT", "MODEL_ROOT", "DXNN_DEVICES", "DXMZ_MAX_EVAL_IMAGES", "PYTHONPATH"):
            self.env.pop(variable, None)
        self.assertFalse((self.root.parent / ".venv").exists())
        self.assertFalse((self.root.parent / "src").exists())

    def run_python(self, *args, env=None):
        run = subprocess.run([sys.executable, "-B", *map(str, args)], cwd=self.temp.name,
                             env=self.env if env is None else env, capture_output=True, text=True, timeout=45)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        return run.stdout

    def shell_environment(self, overrides=None):
        probe = ("import json, os, sys; "
                 "print(json.dumps({'variables': {key: os.environ.get(key) for key in "
                 "('DATA_ROOT', 'MODEL_ROOT', 'DXNN_DEVICES', 'NVIDIA_TF32_OVERRIDE', 'VIRTUAL_ENV')}, "
                 "'executable': sys.executable}))")
        run = subprocess.run([
            "bash", "--noprofile", "--norc", "-c", 'source "$1" || exit $?\n"$2" -B -c "$3"',
            "standalone-env", str(self.root / "env.sh"), sys.executable, probe,
        ], cwd=self.temp.name, env={**self.env, **(overrides or {})},
            capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        return json.loads(run.stdout)

    def prepare_resources(self):
        for config in (self.root / "configs").rglob("*.yaml"):
            directory = self.root / "models" / config.stem
            directory.mkdir(parents=True)
            for filename in (f"{config.stem}.onnx", f"{config.stem}_q-pro.dxnn"):
                (directory / filename).write_bytes(b"fixture")
        coco = self.root / "datasets/COCO"
        (coco / "val2017").mkdir(parents=True)
        (coco / "val2017/000000000001.jpg").write_bytes(b"fixture")
        (coco / "annotations").mkdir()
        (coco / "annotations/instances_val2017.json").write_text(json.dumps({
            "images": [{"id": 1, "file_name": "000000000001.jpg"}],
        }))

    def test_env_defaults_do_not_select_another_python_environment(self):
        result = self.shell_environment()
        self.assertEqual(result["executable"], sys.executable)
        self.assertEqual(result["variables"]["DATA_ROOT"], str(self.root / "datasets"))
        self.assertEqual(result["variables"]["MODEL_ROOT"], str(self.root / "models"))
        self.assertEqual(result["variables"]["DXNN_DEVICES"], "0")
        self.assertEqual(result["variables"]["NVIDIA_TF32_OVERRIDE"], "0")
        self.assertEqual(result["variables"]["VIRTUAL_ENV"], self.env.get("VIRTUAL_ENV"))

    def test_env_preserves_external_resources_and_selected_environment(self):
        supplied = {"DATA_ROOT": "/external data", "MODEL_ROOT": "relative models", "DXNN_DEVICES": "2,3",
                    "VIRTUAL_ENV": "/selected environment", "NVIDIA_TF32_OVERRIDE": "1"}
        result = self.shell_environment(supplied)
        for key, value in supplied.items():
            self.assertEqual(result["variables"][key], "0" if key == "NVIDIA_TF32_OVERRIDE" else value)
        self.assertEqual(result["executable"], sys.executable)

    def test_relocated_cli_lists_checks_and_previews_using_current_interpreter(self):
        runner = self.root / "scripts/run_eval.py"
        models = self.run_python(runner, "--list").splitlines()
        self.assertEqual(len(models), 6)
        preview = self.run_python(runner, "--model", "all", "--backend", "both", "--dry-run")
        commands = [shlex.split(line) for line in preview.splitlines() if "scripts/dxmz_eval.py" in line]
        self.assertEqual(len(commands), len(models) * 2)
        for command in commands:
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(command[1], str(self.root / "scripts/dxmz_eval.py"))
            self.assertTrue(Path(command[3]).is_relative_to(self.root / "configs"))
        self.assertFalse((self.root / "results").exists())
        self.assertFalse((self.root / "accuracy_results.csv").exists())
        self.prepare_resources()
        self.assertIn("Ready: 6 model(s)", self.run_python(runner, "--check"))

    def test_relocated_configs_reuse_installed_decoders_across_repeated_loads(self):
        adapters = list((self.root / "configs").rglob("custom_ops.py"))
        self.assertEqual(len(adapters), 4)
        self.assertTrue(all(path.is_file() and not path.is_symlink() for path in adapters))
        second = Path(self.temp.name) / "second workspace/configs"
        shutil.copytree(self.root / "configs", second, symlinks=True)
        probe = '''
import json
from pathlib import Path
import sys
from dx_modelzoo.loader.model_builder import ModelBuilder
from dx_modelzoo.postprocessing import POSTPROCESSING_REGISTRY

loaded = []
for directory in sys.argv[1:]:
    for path in sorted(Path(directory).rglob("*.yaml")):
        builder = ModelBuilder(path)
        decoder = POSTPROCESSING_REGISTRY.get(builder.config["postprocessing"][0]["type"])
        assert decoder.__module__ == f"dx_modelzoo.models.cv.object_detection.{path.parent.name}.custom_ops"
        assert type(builder.build_postprocessing().steps[0]) is decoder
        loaded.append(builder.name)
print(json.dumps(loaded))
'''
        loaded = json.loads(self.run_python("-c", probe, self.root / "configs", second, self.root / "configs"))
        self.assertEqual(len(loaded), 18)
        self.assertEqual(loaded[:6], loaded[6:12])
        self.assertEqual(loaded[:6], loaded[12:])

    def test_relocated_evaluation_children_use_current_interpreter(self):
        self.prepare_resources()
        stubs = Path(self.temp.name) / "runtime stubs"
        stubs.mkdir()
        (stubs / "sitecustomize.py").write_text(SYNTHETIC_CLI)
        env = {**self.env, "PYTHONPATH": str(stubs)}
        model = "yolov8-m_640x640"
        self.run_python(self.root / "scripts/run_eval.py", "--model", model, "--backend", "both", env=env)
        for backend in ("gpu", "npu"):
            path = self.root / "results" / model / backend / "interpreter.json"
            self.assertEqual(json.loads(path.read_text()), {"executable": sys.executable, "prefix": sys.prefix})
        summary = self.run_python(self.root / "scripts/compare_results.py")
        self.assertIn(f"{model} | 50.0 | 60.0 | 50.0 | 60.0 | 0.0 | 0.0", summary)


if __name__ == "__main__":
    unittest.main()
