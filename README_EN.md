# YOLO GPU / NPU Accuracy Evaluation

[简体中文](README.md) | **English**

Compare object detection accuracy between GPU FP32 ONNX and DEEPX NPU Q-Pro DXNN models on the same COCO validation set, using DX-ModelZoo for data loading, preprocessing, postprocessing, and metric computation.

## 1. Project Structure

```text
accuracy_workspace/
├── env.sh                       # Resource paths and runtime settings
├── configs/cv/object_detection/  # Model YAML files and decoder adapters
├── scripts/                     # Downloads, evaluation, summaries, and regression checks
├── models/<model>/              # ONNX / DXNN models
├── datasets/COCO/               # val2017 images and detection annotations
├── accuracy_results.csv         # Accuracy summary
└── results/<model>/<gpu|npu>/    # eval.log and native result JSON
```

## 2. Test Environment

| Item               | Configuration / Version                                                             |
| ----------------------- | ----------------------------------------------------------------------------------- |
| OS                      | Ubuntu 22.04.5 LTS, x86_64, Linux 6.8.0-138-generic                                 |
| CPU / RAM               | Intel Core i7-14700KF; approximately 46.9 GiB of system-reported RAM                |
| GPU                     | NVIDIA GeForce RTX 5060 Ti, 16 GB; driver 595.99.02                                 |
| NPU                     | DEEPX M1, M.2, 4 GB LPDDR5; device 0                                                |
| NPU software            | DX Runtime / DX Engine 3.4.2; RT driver 2.6.0, PCIe driver 2.5.0, firmware 2.7.4    |
| Python / ModelZoo       | Python 3.10.12; DX-ModelZoo 0.10.2 (ModelZoo source commit `f620eae`)               |
| GPU libraries           | ONNX Runtime GPU 1.23.2; CUDA Runtime 12.8.90, cuDNN 9.10.2.21                      |
| Evaluation dependencies | NumPy 1.26.4, faster-coco-eval 1.8.0, PyTorch 2.8.0+cu128, TorchVision 0.23.0+cu128 |

## 3. Evaluation and Results

Evaluation uses all 5,000 images in COCO val2017 across 80 object categories. Both backends use the same images and the corresponding model configuration, with batch size 1 and random seed 42. NMS uses the ModelZoo defaults: confidence threshold 0.001, IoU threshold 0.7, and a maximum of 300 detections per image. The current GPU launcher disables TF32 and stops if CUDA is unavailable or fails to initialize.

The table below comes from the existing `accuracy_results.csv` and has been checked against the native JSON results. AP values are percentages; differences are in percentage points (pp).

| Model               | GPU mAP | NPU mAP | ΔmAP    | GPU mAP50 | NPU mAP50 | ΔmAP50  |
| ------------------- | ------- | ------- | ------- | --------- | --------- | ------- |
| yolov5-m6_1280x1280 | 50.5768 | 50.2846 | -0.2922 | 67.9465   | 67.8021   | -0.1444 |
| yolov7-e6_1280x1280 | 55.4078 | 55.0586 | -0.3492 | 72.5881   | 72.4268   | -0.1613 |
| yolov8-m_640x640    | 49.4159 | 49.1907 | -0.2252 | 66.0576   | 65.9029   | -0.1547 |
| yolov8-l_640x640    | 52.0510 | 51.8103 | -0.2407 | 68.6871   | 68.6147   | -0.0724 |
| yolov8-x_640x640    | 53.0295 | 52.8445 | -0.1850 | 69.7482   | 69.6082   | -0.1400 |
| yolov9-c_640x640    | 52.1278 | 51.7349 | -0.3929 | 69.0627   | 68.8761   | -0.1866 |

### Metrics

- **mAP**: COCO bbox AP@IoU 0.50:0.05:0.95, averaged across 80 categories.
- **mAP50**: COCO bbox AP@IoU 0.50.
- **Δ**: NPU − GPU. Negative values indicate a lower NPU score. These correspond to `delta_mAP_pp` / `delta_mAP50_pp` in the CSV.

## 4. Setup and Reproduction

### 4.1 Install the Environment

Follow the [DX-ModelZoo installation instructions](https://github.com/DEEPX-AI/dx-modelzoo#installation) to set up the NVIDIA driver, DX Runtime, and NPU device drivers. The workspace requires Python 3.10 or later.

The example below is for a first-time install. Replace the paths with your actual locations, and use the ModelZoo source version listed in the table above.

```bash
# Create the Python environment and install ModelZoo
python3 -m venv /path/to/modelzoo-venv
source /path/to/modelzoo-venv/bin/activate
python -m pip install -e "/path/to/dx-modelzoo[gpu]" \
    "onnxruntime-gpu==1.23.2" "faster-coco-eval==1.8.0" \
    "numpy==1.26.4" "torch==2.8.0" "torchvision==0.23.0"

# Install DX Engine
python -m pip install /path/to/dx_engine-3.4.2-*.whl
```

Before each use, select the Python environment, then enter the workspace:

```bash
source /path/to/modelzoo-venv/bin/activate
cd /path/to/accuracy_workspace
source env.sh
```

`env.sh` only sets resource paths and runtime settings; it does not switch the Python environment.

Defaults are `DATA_ROOT=<workspace>/datasets` and `MODEL_ROOT=<workspace>/models`.

### 4.2 Prepare Data and Models

```bash
# List models available for download
python scripts/run_eval.py --list

# Download the dataset
python scripts/download.py dataset

# Download models
python scripts/download.py models --model yolov8-m_640x640
```

Model files are laid out as follows:

```text
models/<model>/<model>.onnx
models/<model>/<model>_q-pro.dxnn
datasets/COCO/val2017/*.jpg
datasets/COCO/annotations/instances_val2017.json
```

Reproducing the results above requires the same ONNX / DXNN model files. Rerun evaluation after replacing models or using a different compiler release.

### 4.3 Run Evaluation

```bash
# Check all model files and the dataset
python scripts/run_eval.py --model all --backend both --check
python scripts/run_eval.py --model all --backend both --dry-run

# Evaluate one model
python scripts/run_eval.py --model yolov8-m_640x640 --backend both

# Evaluate all models
python scripts/run_eval.py --model all --backend both

# View the summary
python scripts/compare_results.py
```

`--backend` accepts `gpu`, `npu`, or `both`, and defaults to `both`. `--model` selects the model name. `--check` checks files and the dataset. `--dry-run` previews the commands.

The number of images is controlled by `MAX_EVAL_IMAGES = 5000` at the top of [scripts/run_eval.py](scripts/run_eval.py). Set it to `0` to use all images. Any other positive integer selects the first N images sorted by filename and trims the evaluation annotations to match.

## 5. Regression Checks and Extensions

```bash
python -B -m unittest discover -s scripts/tests -v
python -B scripts/tests/check_fp32_entrypoint.py
```

Regression checks cover configuration and path resolution, result recording, COCO subset metrics, GPU FP32 safeguards, and forkserver data loading.

To add a model, create a YAML file with a unique name under `configs/`. Configure its inputs, preprocessing, postprocessing, model paths, and `onnx` / `q-pro` profiles, and make sure the corresponding `custom_ops.py` can be loaded from the same directory. Then validate it with `--list`, `--check`, and a single-model evaluation.
