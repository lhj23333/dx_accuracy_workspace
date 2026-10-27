"""Reuse YOLOV5 decoders from the installed DX-ModelZoo package."""

from dx_modelzoo.models.cv.object_detection.yolov5.custom_ops import (
    YOLOv5Decode,
    YOLOv5MultiscaleDecode,
)

__all__ = ["YOLOv5Decode", "YOLOv5MultiscaleDecode"]
