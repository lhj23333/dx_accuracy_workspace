"""Validate COCO subsets and AP denominators without model inference."""

from contextlib import ExitStack, redirect_stdout
from datetime import datetime
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxmz_eval import configure_sample_limit
from dx_modelzoo.dataset.coco import COCO
from dx_modelzoo.evaluator.object_detection_evaluator import ObjectDetectionEvaluator


class SampleLimitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dxmz-subset-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "val2017").mkdir()
        (self.root / "annotations").mkdir()
        for image_id in (3, 1, 2):
            (self.root / f"val2017/{image_id:012d}.jpg").write_bytes(b"fixture")
        annotation = {"info": {}, "images": [
            {"id": image_id, "file_name": f"{image_id:012d}.jpg", "width": 20, "height": 20}
            for image_id in (3, 1, 2)], "categories": [{"id": 1, "name": "box"}], "annotations": [
            {"id": image_id, "image_id": image_id, "category_id": 1, "bbox": [0, 0, 10, 10],
             "area": 100, "iscrowd": 0} for image_id in (1, 2, 3)]}
        self.annotation = self.root / "annotations/instances_val2017.json"
        self.annotation.write_text(json.dumps(annotation))
        self.original = self.annotation.read_bytes()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(COCO, "__init__", COCO.__init__))
        self.stack.enter_context(patch.object(ObjectDetectionEvaluator, "compute_final_metrics",
                                             ObjectDetectionEvaluator.compute_final_metrics))

    def dataset(self, limit):
        configure_sample_limit(limit)
        with redirect_stdout(io.StringIO()):
            return COCO(str(self.root))

    def test_limit_selects_sorted_images_and_matching_ground_truth(self):
        dataset = self.dataset(2)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.ids, ["000000000001", "000000000002"])
        self.assertEqual(set(dataset.coco_annotation.getImgIds()), {1, 2})
        self.assertEqual({obj["image_id"] for obj in dataset.coco_annotation.dataset["annotations"]}, {1, 2})
        self.assertEqual(self.annotation.read_bytes(), self.original)
        self.assertEqual(len(list((self.root / "val2017").iterdir())), 3)

    def test_zero_uses_all_images(self):
        dataset = self.dataset(0)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(set(dataset.coco_annotation.getImgIds()), {1, 2, 3})

    def test_limit_above_dataset_size_uses_all_images(self):
        dataset = self.dataset(100)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(set(dataset.coco_annotation.getImgIds()), {1, 2, 3})

    def test_invalid_limits_are_rejected(self):
        for limit in (-1, 1.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                configure_sample_limit(limit)

    def test_ap_counts_selected_images_with_no_detections_and_excludes_unselected_images(self):
        dataset = self.dataset(2)
        evaluator = ObjectDetectionEvaluator(SimpleNamespace(_use_async=False), dataset)
        evaluator._start_time = datetime.now()
        evaluator.total_inference_time = 1
        evaluator._detections = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 1}]
        with redirect_stdout(io.StringIO()):
            result = evaluator.compute_final_metrics({"image_count": 2})
        metrics = {metric["name"]: metric["metric_value"] for metric in result["metrics"]}
        # One of two selected images is detected; including the third image would produce about 33 AP.
        self.assertTrue(49 < metrics["mAP"] < 51, metrics)
        self.assertEqual(result["num_samples"], 2)


if __name__ == "__main__":
    unittest.main()
