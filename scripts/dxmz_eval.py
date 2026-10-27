#!/usr/bin/env python3
"""Launch ModelZoo with a fixed COCO subset and GPU FP32 safeguards."""

import os


def configure_sample_limit(limit):
    if type(limit) is not int or limit < 0:
        raise ValueError("MAX_EVAL_IMAGES must be a non-negative integer; 0 means all images")

    from dx_modelzoo.dataset.coco import COCO
    from dx_modelzoo.evaluator.object_detection_evaluator import ObjectDetectionEvaluator

    original_init = COCO.__init__
    original_metrics = ObjectDetectionEvaluator.compute_final_metrics

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        total = len(self.img_files)
        if limit and limit < total:
            # COCO already sorts file names, so both backends select the same images.
            self.img_files = self.img_files[:limit]
            self.ids = self.ids[:limit]
            selected = {int(image_id) for image_id in self.ids}
            coco = self.coco_annotation
            if coco is None:
                raise ValueError("COCO annotations are required for subset evaluation")
            coco.dataset["images"] = [image for image in coco.dataset["images"] if image["id"] in selected]
            coco.dataset["annotations"] = [obj for obj in coco.dataset["annotations"] if obj["image_id"] in selected]
            coco.createIndex()
        print(f"COCO samples: {len(self.img_files)}/{total} (limit={limit}, 0=all)", flush=True)

    def compute_metrics(self, metrics_state):
        result = original_metrics(self, metrics_state)
        result["num_samples"] = len(self.dataset)
        return result

    COCO.__init__ = initialize
    ObjectDetectionEvaluator.compute_final_metrics = compute_metrics


def launch():
    configure_sample_limit(int(os.environ.get("DXMZ_MAX_EVAL_IMAGES", "0")))
    from fp32_policy import configure

    configure()
    from dx_modelzoo.main import main

    return main()


if __name__ == "__main__":
    raise SystemExit(launch())
