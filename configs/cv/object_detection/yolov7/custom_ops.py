"""Reuse YOLOV7 decoders from the installed DX-ModelZoo package."""

from dx_modelzoo.models.cv.object_detection.yolov7.custom_ops import YOLOv7Decode

__all__ = ["YOLOv7Decode"]
