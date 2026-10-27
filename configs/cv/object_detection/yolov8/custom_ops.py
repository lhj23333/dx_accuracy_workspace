"""Reuse YOLOV8 decoders from the installed DX-ModelZoo package."""

from dx_modelzoo.models.cv.object_detection.yolov8.custom_ops import YOLOv8Decode

__all__ = ["YOLOv8Decode"]
