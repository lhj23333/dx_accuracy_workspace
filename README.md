# YOLO GPU / NPU 精度测试

**简体中文** | [English](README_EN.md)

基于 DX-ModelZoo，在同一份 COCO 验证集上比较 GPU FP32 ONNX 与 DEEPX NPU Q-Pro DXNN 的目标检测精度，复用 ModelZoo 的数据加载、前后处理和指标计算。

## 1. 项目结构

```text
accuracy_workspace/
├── env.sh                       # 资源路径与运行参数
├── configs/cv/object_detection/  # 模型 YAML 与解码器适配文件
├── scripts/                     # 下载、评测、结果汇总与回归测试
├── models/<model>/              # ONNX / DXNN 模型
├── datasets/COCO/               # val2017 图像与检测标注
├── accuracy_results.csv         # 精度汇总
└── results/<model>/<gpu|npu>/    # eval.log 与原生结果 JSON
```

## 2. 测试环境

| 条目                | 配置 / 版本                                                                          |
| ----------------- | -------------------------------------------------------------------------------- |
| 系统                | Ubuntu 22.04.5 LTS，x86_64，Linux 6.8.0-138-generic                                |
| CPU / 内存          | Intel Core i7-14700KF；系统识别内存约 46.9 GiB                                           |
| GPU               | NVIDIA GeForce RTX 5060 Ti，16 GB；驱动 595.99.02                                    |
| NPU               | DEEPX M1，M.2，4 GB LPDDR5；设备 0                                                    |
| NPU 软件            | DX Runtime / DX Engine 3.4.2；RT 驱动 2.6.0，PCIe 驱动 2.5.0，固件 2.7.4                  |
| Python / ModelZoo | Python 3.10.12；DX-ModelZoo 0.10.2（ModelZoo 源码提交 `f620eae`）                       |
| GPU 运行库           | ONNX Runtime GPU 1.23.2；CUDA Runtime 12.8.90，cuDNN 9.10.2.21                     |
| 评测依赖              | NumPy 1.26.4，faster-coco-eval 1.8.0，PyTorch 2.8.0+cu128，TorchVision 0.23.0+cu128 |

## 3. 测试内容与结果

测试集为 COCO val2017 全量 5,000 张图像、80 类目标。各模型两端使用相同图像和对应配置，batch size 为 1，随机种子为 42。NMS 使用 ModelZoo 默认参数：置信度阈值 0.001、IoU 阈值 0.7、最多保留 300 个检测框。当前 GPU 入口禁用 TF32，并在 CUDA 不可用或初始化失败时停止。

下表取自现有 `accuracy_results.csv`，已逐项与原生 JSON 核对；AP 数值单位为 %，差值单位为百分点（pp）。

| 模型                  | GPU mAP | NPU mAP | ΔmAP    | GPU mAP50 | NPU mAP50 | ΔmAP50  |
| ------------------- | ------- | ------- | ------- | --------- | --------- | ------- |
| yolov5-m6_1280x1280 | 50.5768 | 50.2846 | -0.2922 | 67.9465   | 67.8021   | -0.1444 |
| yolov7-e6_1280x1280 | 55.4078 | 55.0586 | -0.3492 | 72.5881   | 72.4268   | -0.1613 |
| yolov8-m_640x640    | 49.4159 | 49.1907 | -0.2252 | 66.0576   | 65.9029   | -0.1547 |
| yolov8-l_640x640    | 52.0510 | 51.8103 | -0.2407 | 68.6871   | 68.6147   | -0.0724 |
| yolov8-x_640x640    | 53.0295 | 52.8445 | -0.1850 | 69.7482   | 69.6082   | -0.1400 |
| yolov9-c_640x640    | 52.1278 | 51.7349 | -0.3929 | 69.0627   | 68.8761   | -0.1866 |

### 指标说明

- **mAP**：COCO bbox AP@IoU 0.50:0.05:0.95，跨 80 类取平均。
- **mAP50**：COCO bbox AP@IoU 0.50。
- **Δ**：NPU − GPU；负值表示 NPU 分数低于 GPU，对应 CSV 的 `delta_mAP_pp` / `delta_mAP50_pp`。

## 4. 环境搭建与复现

### 4.1 安装环境

按 [DX-ModelZoo 安装说明](https://github.com/DEEPX-AI/dx-modelzoo#installation)准备 NVIDIA 驱动、DX Runtime 和 NPU 设备驱动。工作区使用 Python 3.10 或以上。

首次安装示例如下，需将路径替换为实际位置，并使用上表对应版本的 ModelZoo 源码。

```bash
# 创建 Python 环境并安装 ModelZoo
python3 -m venv /path/to/modelzoo-venv
source /path/to/modelzoo-venv/bin/activate
python -m pip install -e "/path/to/dx-modelzoo[gpu]" \
    "onnxruntime-gpu==1.23.2" "faster-coco-eval==1.8.0" \
    "numpy==1.26.4" "torch==2.8.0" "torchvision==0.23.0"

# 安装 DX Engine
python -m pip install /path/to/dx_engine-3.4.2-*.whl
```

每次使用时，先选择 Python 环境，再进入工作区：

```bash
source /path/to/modelzoo-venv/bin/activate
cd /path/to/accuracy_workspace
source env.sh
```

`env.sh` 仅设置资源路径和运行参数，不切换 Python 环境；

默认 `DATA_ROOT=<workspace>/datasets`、`MODEL_ROOT=<workspace>/models`

### 4.2 准备数据与模型

```bash
# 查看可下载的模型列表
python scripts/run_eval.py --list

# 下载数据集
python scripts/download.py dataset

# 下载模型
python scripts/download.py models --model yolov8-m_640x640
```

模型文件结构如下：

```text
models/<model>/<model>.onnx
models/<model>/<model>_q-pro.dxnn
datasets/COCO/val2017/*.jpg
datasets/COCO/annotations/instances_val2017.json
```

复现上述结果需使用相同的 ONNX / DXNN 模型文件；替换模型或使用其他编译版本后应重新评测。

### 4.3 执行评测

```bash
# 检查全部模型文件及数据集
python scripts/run_eval.py --model all --backend both --check
python scripts/run_eval.py --model all --backend both --dry-run

# 评测单模型
python scripts/run_eval.py --model yolov8-m_640x640 --backend both

# 评测全部模型
python scripts/run_eval.py --model all --backend both

# 查看结果汇总
python scripts/compare_results.py
```

主命令 `--backend` 可选 `gpu`、`npu`、`both`，默认 `both`；`--model` 指定模型名称；`--check` 检查文件及数据集；`--dry-run` 预览命令。

测试数量由 [scripts/run_eval.py](scripts/run_eval.py) 顶部的 `MAX_EVAL_IMAGES = 5000` 控制；设为 `0` 使用全部图像，其他正整数按文件名排序截取前 N 张，同时裁剪评测标注。

## 5. 回归检查与扩展

```bash
python -B -m unittest discover -s scripts/tests -v
python -B scripts/tests/check_fp32_entrypoint.py
```

回归检查覆盖配置与路径解析、结果记录、COCO 子集指标、GPU FP32 保护、forkserver 数据加载。

新增模型时，在 `configs/` 下添加名称唯一的 YAML，配置输入、前后处理、模型路径及 `onnx` / `q-pro` profile，并确保同目录可加载相应 `custom_ops.py`；随后通过 `--list`、`--check` 和单模型评测验证。