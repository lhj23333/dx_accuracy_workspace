"""Store accuracy metrics in one CSV, using only the standard library."""

import csv
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile

from workspace import PROFILES, ROOT

SUMMARY_NAME = "accuracy_results.csv"
PREFIXES = {"gpu": "gpu_fp32", "npu": "npu_q_pro"}
METRICS = ("mAP", "mAP50")


def column(backend, metric):
    return f"{PREFIXES[backend]}_{metric}"


FIELDNAMES = ["model"] + [
    column(backend, metric) for backend in PROFILES for metric in METRICS
] + [f"delta_{metric}_pp" for metric in METRICS]


def read_summary(root=ROOT):
    """Read without creating or rewriting files; missing results stay blank."""
    rows = []
    path = root / SUMMARY_NAME
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or [name.strip() for name in reader.fieldnames] != FIELDNAMES:
                raise ValueError(f"Unexpected CSV columns in {path}")
            reader.fieldnames = FIELDNAMES
            seen = set()
            for row in reader:
                row = {key: value.strip() if isinstance(value, str) else value for key, value in row.items()}
                name = row.get("model")
                if not name or name in seen or None in row or None in row.values():
                    raise ValueError(f"Invalid or duplicate model row in {path}")
                seen.add(name)
                rows.append(row)
    return rows


def update(root, model_name, backend, metrics=None, *, clear_other=False):
    """Replace one backend's scores; None clears its previous attempt."""
    if backend not in PROFILES:
        raise ValueError(f"Unknown backend: {backend}")
    with (root / ".accuracy_results.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = read_summary(root)
        row = next((row for row in rows if row["model"] == model_name), None)
        if row is None:
            row = dict.fromkeys(FIELDNAMES, "")
            row["model"] = model_name
            rows.append(row)
        if clear_other:
            other = "npu" if backend == "gpu" else "gpu"
            for metric in METRICS:
                row[column(other, metric)] = ""
        for metric in METRICS:
            row[column(backend, metric)] = metrics[metric] if metrics is not None else ""
            gpu, npu = (row[column(device, metric)] for device in PROFILES)
            row[f"delta_{metric}_pp"] = round(float(npu) - float(gpu), 4) if gpu != "" and npu != "" else ""
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=root,
                                             prefix=".accuracy_results-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, root / SUMMARY_NAME)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def read_result(path, model_name, backend):
    """Validate native COCO results before publishing accuracy values."""
    result = json.loads(path.read_text())
    if result.get("model") != model_name or result.get("profile") != PROFILES[backend]:
        raise ValueError(f"Model/profile mismatch in {path}")
    if result.get("error") is not None:
        raise ValueError(f"Evaluation failed: {result['error']}")
    if result.get("dataset") != "COCO":
        raise ValueError(f"Expected COCO results in {path}")
    values = {metric["name"]: float(metric["metric_value"]) for metric in result.get("metrics", [])}
    for metric in METRICS:
        if metric not in values or not math.isfinite(values[metric]) or not 0 <= values[metric] <= 100:
            raise ValueError(f"Missing/invalid {metric} in {path}")
    return {metric: values[metric] for metric in METRICS}


def print_summary(rows, selected="all"):
    print(" | ".join(FIELDNAMES))
    for row in rows:
        if selected == "all" or row["model"] == selected:
            print(" | ".join(str(row[field]) if row[field] != "" else "-" for field in FIELDNAMES))
