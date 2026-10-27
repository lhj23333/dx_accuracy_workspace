"""Discover evaluation YAMLs and check the files needed by selected models."""

import json
import os
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {"gpu": "onnx", "npu": "q-pro"}


def discover_models(root=ROOT):
    import yaml

    models = {}
    for path in sorted((root / "configs").rglob("*")):
        if path.suffix not in {".yaml", ".yml"}:
            continue
        config = yaml.safe_load(path.read_text())
        name = config.get("name") if isinstance(config, dict) else None
        if not isinstance(name, str) or not name or name in models:
            raise ValueError(f"Missing or duplicate model name: {path}")
        models[name] = path
    return models


def select_models(name="all", root=ROOT):
    models = discover_models(root)
    if not models:
        raise ValueError(f"No model YAMLs found in {root / 'configs'}")
    if name == "all":
        return models
    if name not in models:
        raise ValueError(f"Unknown model: {name}; use --list to see available models")
    return {name: models[name]}


def environment(root=ROOT):
    env = os.environ.copy()
    env.setdefault("DATA_ROOT", str(root / "datasets"))
    env.setdefault("MODEL_ROOT", str(root / "models"))
    for variable in ("DATA_ROOT", "MODEL_ROOT"):
        env[variable] = str((root / Path(env[variable]).expanduser()).resolve())
    env.setdefault("DXNN_DEVICES", "0")
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    return env


def resolve_path(value, config, env, root=ROOT):
    path = Path(Template(value).substitute(env, MODEL_NAME=config["name"])).expanduser()
    return path if path.is_absolute() else root / path


def model_file(config, backend, env, root=ROOT):
    profile = PROFILES[backend]
    target = "onnx" if backend == "gpu" else "dxnn"
    if config["profiles"][profile].get("target", target) != target:
        raise ValueError(f"{config['name']}: {profile} must target {target}")
    artifacts = config["artifacts"]
    base = resolve_path(artifacts.get(target) or artifacts["path"], config, env, root)
    if base.suffix == f".{target}":
        return base
    filename = f"{config['name']}.onnx" if backend == "gpu" else f"{config['name']}_{profile}.dxnn"
    return base / filename


def check_coco(directory):
    annotation = directory / "annotations/instances_val2017.json"
    images = json.loads(annotation.read_text())["images"]
    expected = {image["file_name"] for image in images}
    actual = {path.name for path in (directory / "val2017").glob("*.jpg") if path.is_file()}
    if not expected or len(expected) != len(images) or expected != actual:
        raise ValueError(f"Incomplete COCO dataset: {directory}; missing={len(expected-actual)}, extra={len(actual-expected)}")


def check_inputs(models, backends, env, root=ROOT):
    import yaml

    datasets = set()
    for name, path in models.items():
        config = yaml.safe_load(path.read_text())
        for backend in backends:
            model = model_file(config, backend, env, root)
            if not model.is_file() or model.stat().st_size == 0:
                raise ValueError(f"Missing or empty model file: {model}")
            if backend == "gpu":
                device = config["profiles"][PROFILES[backend]]["runtime"]["device"]
                if not (device == "gpu" or type(device) is int or isinstance(device, list) and device):
                    raise ValueError(f"{name}: GPU evaluation requires a GPU runtime device")
        dataset = config["dataset"]
        if dataset["type"] != "COCO":
            raise ValueError(f"{name}: this accuracy runner expects COCO detection data")
        datasets.add(resolve_path(dataset["eval_path"], config, env, root))
    for dataset in datasets:
        check_coco(dataset)
