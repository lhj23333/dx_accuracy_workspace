#!/usr/bin/env python3
"""Check forkserver data loading through the workspace FP32 launcher."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/dxmz_eval.py"

HARNESS = '''
import json
import os
from pathlib import Path

class TinyDataset:
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return index * 2

def record(event):
    with Path(os.environ["DXMZ_ENTRYPOINT_EVENTS"]).open("a") as stream:
        stream.write(event + "\\n")

def configure():
    record("configure")

def main():
    record("main")
    from dx_modelzoo.common.dataloader import DataLoader
    values = list(DataLoader(TinyDataset(), batch_size=1, num_workers=2))
    assert values == [0, 2, 4, 6, 8, 10], values
    print(json.dumps({"start_method": "forkserver", "workers": 2, "values": values}))
    return 0
'''

SITECUSTOMIZE = '''
import sys
import types
import entrypoint_harness

# Stub only the FP32 policy and CLI callback; use the actual launcher and
# actual ModelZoo DataLoader. No model, GPU or NPU session is constructed.
policy = types.ModuleType("fp32_policy")
policy.configure = entrypoint_harness.configure
sys.modules["fp32_policy"] = policy
cli = types.ModuleType("dx_modelzoo.main")
cli.main = entrypoint_harness.main
sys.modules["dx_modelzoo.main"] = cli
'''


def main():
    with tempfile.TemporaryDirectory(prefix="dxmz-entrypoint-") as directory:
        temp = Path(directory)
        (temp / "entrypoint_harness.py").write_text(HARNESS)
        (temp / "sitecustomize.py").write_text(SITECUSTOMIZE)
        events_path = temp / "events.txt"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(temp)
        env["DXMZ_ENTRYPOINT_EVENTS"] = str(events_path)
        run = subprocess.run([sys.executable, str(LAUNCHER)], env=env, capture_output=True, text=True, timeout=45)
        assert run.returncode == 0, run.stdout + run.stderr
        events = events_path.read_text().splitlines()
        assert events == ["configure", "main"], events
        worker_result = json.loads(run.stdout)
        report = {"success": True, "launcher": str(LAUNCHER), "callback_events": events,
                  **worker_result, "inference_or_evaluation_started": False}
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
