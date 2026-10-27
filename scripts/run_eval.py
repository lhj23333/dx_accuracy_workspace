#!/usr/bin/env python3
"""Run ModelZoo evaluation and save accuracy metrics, a log, and native JSON."""

import argparse
from datetime import datetime
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys

from results_store import SUMMARY_NAME, print_summary, read_result, read_summary, update
from workspace import PROFILES, ROOT, check_inputs, environment, model_file, select_models

MAX_EVAL_IMAGES = 5000  # Edit this limit; 0 uses the full dataset.


def run_evaluation(item, env, root=ROOT):
    model, backend, profile = item["model"], item["backend"], item["profile"]
    directory = root / "results" / model / backend
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".eval.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"This model/backend is already running: {model}/{backend}") from None
        log_path = directory / "eval.log"
        result_path = directory / "result" / f"eval_{model}_{profile}.json"
        with log_path.open("w", buffering=1) as log:
            def report(message):
                print(message, flush=True)
                log.write(message + "\n")

            report(f"Started: {datetime.now().astimezone().isoformat(timespec='seconds')}")
            report(shlex.join(item["command"]))
            report(f"Result directory: {directory}")
            report(f"Max evaluation images: {env.get('DXMZ_MAX_EVAL_IMAGES', '0')} (0=all)")
            try:
                # Clear old scores and native JSON before starting a new attempt.
                update(root, model, backend)
                result_path.unlink(missing_ok=True)
                metric_error = False
                process = subprocess.Popen(item["command"], cwd=directory, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        metric_error |= "Object detection evaluation failed:" in line
                    code = process.wait()
                finally:
                    # Stop the evaluator and its data-loader workers on cancellation.
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.wait()
                    process.stdout.close()
                if code:
                    raise RuntimeError(f"Evaluation exited with code {code}")
                if metric_error:
                    raise RuntimeError("COCO metric computation failed")
                metrics = read_result(result_path, model, backend)
                count = json.loads(result_path.read_text()).get("num_samples")
                other = "npu" if backend == "gpu" else "gpu"
                other_path = root / "results" / model / other / "result" / f"eval_{model}_{PROFILES[other]}.json"
                try:
                    other_count = json.loads(other_path.read_text()).get("num_samples")
                except (OSError, ValueError):
                    other_count = None
                clear_other = count is not None and count != other_count
                update(root, model, backend, metrics, clear_other=clear_other)
                if clear_other:
                    report(f"Cleared {other} scores: sample counts differ or the previous count is unknown")
            except KeyboardInterrupt:
                report("Interrupted by user")
                raise
            except Exception as error:
                report(f"Failed: {error}")
                raise
            else:
                report("Success")
            finally:
                report(f"Finished: {datetime.now().astimezone().isoformat(timespec='seconds')}")


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["gpu", "npu", "both"], default="both")
    parser.add_argument("--model", default="all", help="all or a model name shown by --list")
    parser.add_argument("--list", action="store_true", help="List models discovered from configs/")
    parser.add_argument("--check", action="store_true", help="Check selected model files and dataset without evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without writing files or running models")
    args = parser.parse_args()
    try:
        models = select_models(args.model)
        if args.list:
            print("\n".join(models))
            return
        if type(MAX_EVAL_IMAGES) is not int or MAX_EVAL_IMAGES < 0:
            raise ValueError("MAX_EVAL_IMAGES must be a non-negative integer; 0 means all images")
        env = environment()
        env["DXMZ_MAX_EVAL_IMAGES"] = str(MAX_EVAL_IMAGES)
        backends = list(PROFILES) if args.backend == "both" else [args.backend]
        commands = []
        for name, path in models.items():
            config = yaml.safe_load(path.read_text())
            for backend in backends:
                profile = PROFILES[backend]
                launcher = [sys.executable, str(ROOT / "scripts/dxmz_eval.py")]
                command = launcher + ["eval", str(path), "--profile", profile, "--seed", "42",
                                      "--model-path", str(model_file(config, backend, env)), "--save"]
                commands.append({"backend": backend, "model": name, "profile": profile, "command": command})
        if args.dry_run:
            print("Summary CSV: " + str(ROOT / SUMMARY_NAME))
            print(f"MAX_EVAL_IMAGES={MAX_EVAL_IMAGES} (0=all)")
            for variable in ("DATA_ROOT", "MODEL_ROOT"):
                print(variable + "=" + shlex.quote(env[variable]))
            for item in commands:
                print(f"Working directory: {ROOT / 'results' / item['model'] / item['backend']}")
                print(shlex.join(item["command"]))
            return
        check_inputs(models, backends, env)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        parser.exit(1, f"{error}\n")
    if args.check:
        print(f"Ready: {len(models)} model(s), backend={args.backend}, max_images={MAX_EVAL_IMAGES} (0=all)")
        return
    print("Summary CSV: " + str(ROOT / SUMMARY_NAME), flush=True)
    for item in commands:
        try:
            run_evaluation(item, env)
        except KeyboardInterrupt:
            raise SystemExit(130) from None
        except Exception as error:
            raise SystemExit(str(error)) from error
        print_summary(read_summary(), item["model"])


if __name__ == "__main__":
    main()
