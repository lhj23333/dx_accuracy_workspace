#!/usr/bin/env python3
"""Download model files or COCO val2017; existing files need no registration."""

import argparse
from pathlib import Path
import subprocess
import zipfile

from workspace import PROFILES, ROOT, check_coco, environment, model_file, select_models

MODEL_URL = "https://sdk.deepx.ai/modelzoo"
Q_PRO_RELEASE = "2_4_0"
COCO_URL = "https://s3.amazonaws.com/images.cocodataset.org"


def download_file(url, target):
    if target.is_file() and target.stat().st_size:
        print(f"Exists: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    print(f"Downloading: {url}", flush=True)
    subprocess.run([
        "curl", "--fail", "--location", "--show-error", "--retry", "3",
        "--connect-timeout", "20", "--continue-at", "-", "--output", str(partial), url,
    ], check=True)
    if not partial.is_file() or partial.stat().st_size == 0:
        raise ValueError(f"Downloaded file is empty: {url}")
    partial.replace(target)


def download_models(name, backend, root=ROOT):
    import yaml

    env = environment(root)
    backends = list(PROFILES) if backend == "both" else [backend]
    for model, path in select_models(name, root).items():
        config = yaml.safe_load(path.read_text())
        for device in backends:
            suffix = f"onnx/{model}.onnx" if device == "gpu" else f"q-pro-dxnn/{Q_PRO_RELEASE}/{model}.dxnn"
            download_file(f"{MODEL_URL}/{suffix}", model_file(config, device, env, root))


def download_dataset(root=ROOT):
    destination = Path(environment(root)["DATA_ROOT"]) / "COCO"
    try:
        check_coco(destination)
    except (OSError, ValueError, KeyError):
        pass
    else:
        print(f"Exists: {destination}")
        return
    cache = root / "downloads"
    images = cache / "val2017.zip"
    annotations = cache / "annotations_trainval2017.zip"
    download_file(f"{COCO_URL}/zips/val2017.zip", images)
    download_file(f"{COCO_URL}/annotations/annotations_trainval2017.zip", annotations)
    with zipfile.ZipFile(images) as archive:
        members = [name for name in archive.namelist() if name.startswith("val2017/") and name.endswith(".jpg")]
        if not members:
            raise ValueError(f"No COCO images in {images}")
        for name in members:
            if Path(name).parts != ("val2017", Path(name).name):
                raise ValueError(f"Unexpected archive member: {name}")
            archive.extract(name, destination)
    with zipfile.ZipFile(annotations) as archive:
        archive.extract("annotations/instances_val2017.json", destination)
    check_coco(destination)
    for archive in (images, annotations):
        archive.unlink()
    if not any(cache.iterdir()):
        cache.rmdir()
    print(f"Ready: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    models = commands.add_parser("models", help="Download selected models from the DEEPX model server")
    models.add_argument("--model", default="all", help="all or a model name shown by run_eval.py --list")
    models.add_argument("--backend", choices=["gpu", "npu", "both"], default="both")
    commands.add_parser("dataset", help="Download and extract COCO val2017")
    args = parser.parse_args()
    try:
        if args.command == "models":
            download_models(args.model, args.backend)
        else:
            download_dataset()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Download failed: {error}\n")


if __name__ == "__main__":
    main()
