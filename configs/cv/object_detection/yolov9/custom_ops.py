"""Reuse YOLOV9 decoders from the installed DX-ModelZoo package."""

from dx_modelzoo.models.cv.object_detection.yolov9.custom_ops import YOLOv9Decode

__all__ = ["YOLOv9Decode"]
