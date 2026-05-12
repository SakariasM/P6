#!/usr/bin/env python3
"""
coco_benchmark.py — evaluate a TFLite segmentation model on COCO val2017
                    (person class only), reporting mAP@50-95 (mask).

No dilation or blur is applied — raw thresholded model output only.
For semantic models (selfie_segmenter, student_seg_320), the merged mask is
split into per-person instances via connected components.
For YOLO TFLite models, per-instance masks come directly from the detections.

Usage:
  python3 scripts/coco_benchmark.py --model models/student_seg_320.tflite
  python3 scripts/coco_benchmark.py --model models/selfie_segmenter_float32.tflite
  python3 scripts/coco_benchmark.py --model models/yolo26n-seg_float32.tflite

Options:
  --model        Path to TFLite model file (required)
  --coco-dir     Directory to store COCO data (default: datasets/coco_data)
  --threshold    Mask probability threshold — set to 0.5 for benchmarking (default: 0.5)
  --min-area     Min connected-component area in pixels to count as an instance (default: 100)
  --max-images   Cap the number of images evaluated (useful for quick sanity checks)
"""

import argparse
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

COCO_PERSON_CAT_ID = 1


# ── COCO download ─────────────────────────────────────────────────────────────

def _download(url, dest):
    if os.path.exists(dest):
        return
    print(f"  Downloading {os.path.basename(dest)} ...", flush=True)

    def _progress(block, block_size, total):
        if total > 0:
            print(f"\r  {min(block * block_size / total * 100, 100):.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def setup_coco(coco_dir):
    """Download COCO val2017 images + annotations if not already present."""
    coco_dir   = Path(coco_dir)
    images_dir = coco_dir / "val2017"
    ann_file   = coco_dir / "annotations" / "instances_val2017.json"

    if images_dir.exists() and ann_file.exists():
        return str(images_dir), str(ann_file)

    coco_dir.mkdir(parents=True, exist_ok=True)
    (coco_dir / "annotations").mkdir(exist_ok=True)

    if not ann_file.exists():
        ann_zip = coco_dir / "annotations_trainval2017.zip"
        _download(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            str(ann_zip),
        )
        print("  Extracting annotations ...", flush=True)
        with zipfile.ZipFile(ann_zip) as z:
            z.extract("annotations/instances_val2017.json", str(coco_dir))
        ann_zip.unlink()

    if not images_dir.exists():
        img_zip = coco_dir / "val2017.zip"
        _download(
            "http://images.cocodataset.org/zips/val2017.zip",
            str(img_zip),
        )
        print("  Extracting images ...", flush=True)
        with zipfile.ZipFile(img_zip) as z:
            z.extractall(str(coco_dir))
        img_zip.unlink()

    return str(images_dir), str(ann_file)


# ── TFLite inference ──────────────────────────────────────────────────────────

class TFLiteInferencer:
    """
    Single-image inference wrapper, replicating the model-loading logic from
    live_mask.py without dilation, blur, or shared-memory overhead.
    """

    def __init__(self, model_path, threshold=0.5, min_area=100):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter

        self.threshold = threshold
        self.min_area  = min_area

        interp = Interpreter(model_path=model_path, num_threads=4)
        interp.allocate_tensors()
        self.interp = interp

        inp_detail  = interp.get_input_details()[0]
        out_details = interp.get_output_details()
        self.inp_idx = inp_detail["index"]
        inp_shape    = inp_detail["shape"]

        # NCHW (PyTorch-exported) vs NHWC (standard TFLite) — same heuristic as live_mask.py
        if inp_shape[1] <= 4 and inp_shape[2] > 4:
            self.nchw = True
            _, _, self.model_h, self.model_w = inp_shape
        else:
            self.nchw = False
            _, self.model_h, self.model_w, _ = inp_shape

        # Detect semantic (single output) vs YOLO (detection + prototype outputs)
        self.yolo        = False
        self.selfie_idx  = None
        self.det_idx     = None
        self.proto_idx   = None
        self.proto_h = self.proto_w = self.n_proto = None

        if len(out_details) == 1 and len(out_details[0]["shape"]) == 4:
            self.selfie_idx = out_details[0]["index"]
        else:
            self.yolo = True
            for od in out_details:
                if len(od["shape"]) == 3:
                    self.det_idx = od["index"]
                elif len(od["shape"]) == 4:
                    self.proto_idx = od["index"]
                    _, self.proto_h, self.proto_w, self.n_proto = od["shape"]
            if self.det_idx is None or self.proto_idx is None:
                raise RuntimeError(
                    f"Unexpected YOLO output shapes: {[od['shape'] for od in out_details]}"
                )

        model_type = "YOLO" if self.yolo else "semantic"
        layout     = "NCHW" if self.nchw else "NHWC"
        print(
            f"[model] {os.path.basename(model_path)}: "
            f"{self.model_w}×{self.model_h}, {model_type}, {layout}, "
            f"threshold={threshold}, min_area={min_area}px"
        )

    def infer(self, image_bgr):
        """
        Run inference on a single BGR image.
        Returns a list of (mask, confidence) tuples where mask is bool (H, W)
        in original image coordinates.
        """
        h, w = image_bgr.shape[:2]

        # Preprocess: resize → RGB → [0, 1]
        small = cv2.resize(image_bgr, (self.model_w, self.model_h))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        if self.nchw:
            inp = (rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)[np.newaxis]
        else:
            inp = (rgb.astype(np.float32) / 255.0)[np.newaxis]

        self.interp.set_tensor(self.inp_idx, inp)
        self.interp.invoke()

        if not self.yolo:
            return self._infer_semantic(h, w)
        else:
            return self._infer_yolo(h, w)

    def _infer_semantic(self, h, w):
        out = self.interp.get_tensor(self.selfie_idx)
        prob_small = out[0, 0, :, :] if self.nchw else out[0, :, :, 0]  # (model_h, model_w)

        # Upsample probability map to original image resolution before thresholding
        prob = cv2.resize(prob_small, (w, h), interpolation=cv2.INTER_LINEAR)
        binary = prob > self.threshold

        return self._split_components(binary, prob)

    def _infer_yolo(self, h, w):
        detections = self.interp.get_tensor(self.det_idx)[0]    # (n_dets, 38+)
        prototypes = self.interp.get_tensor(self.proto_idx)[0]  # (proto_h, proto_w, n_proto)

        valid = (
            (detections[:, 4] > self.threshold) &
            (detections[:, 5].astype(int) == 0)   # class 0 = person
        )
        if not valid.any():
            return []

        proto_flat = prototypes.reshape(-1, self.n_proto)   # (proto_h*proto_w, n_proto)
        coeffs     = detections[valid, 6:]                  # (n_valid, n_proto)
        confs      = detections[valid, 4].tolist()

        logits       = coeffs @ proto_flat.T                        # (n_valid, proto_h*proto_w)
        person_probs = 1.0 / (1.0 + np.exp(-logits))               # sigmoid
        person_probs = person_probs.reshape(-1, self.proto_h, self.proto_w)

        instances = []
        for pmask_small, conf in zip(person_probs, confs):
            pmask  = cv2.resize(pmask_small, (w, h), interpolation=cv2.INTER_LINEAR)
            binary = pmask > 0.5
            if binary.sum() >= self.min_area:
                instances.append((binary, float(conf)))
        return instances

    def _split_components(self, binary_mask, prob_map):
        """Split merged binary mask into instances via connected components."""
        n_labels, labels = cv2.connectedComponents(
            binary_mask.astype(np.uint8), connectivity=8
        )
        instances = []
        for lbl in range(1, n_labels):
            component = labels == lbl
            if component.sum() < self.min_area:
                continue
            conf = float(prob_map[component].max())
            instances.append((component, conf))
        return instances


# ── COCO prediction formatting ────────────────────────────────────────────────

def mask_to_rle(binary_mask):
    """Encode a bool (H, W) mask as a pycocotools compressed RLE."""
    from pycocotools import mask as mask_util
    # pycocotools requires Fortran-order (column-major) uint8
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return rle  # counts stays as bytes — loadRes handles it directly


# ── evaluation ────────────────────────────────────────────────────────────────

def run_eval(model_path, coco_dir, threshold, min_area, max_images):
    images_dir, ann_file = setup_coco(coco_dir)

    print(f"\nLoading COCO annotations ...")
    with open(ann_file) as f:
        coco_raw = json.load(f)

    # Find image IDs that have at least one person instance annotation with a mask
    person_img_ids = set()
    for ann in coco_raw["annotations"]:
        if ann["category_id"] == COCO_PERSON_CAT_ID and ann.get("segmentation"):
            person_img_ids.add(ann["image_id"])

    images_by_id  = {img["id"]: img for img in coco_raw["images"]}
    person_img_ids = sorted(person_img_ids)

    if max_images:
        person_img_ids = person_img_ids[:max_images]

    total = len(person_img_ids)
    cap   = f" (capped at {max_images})" if max_images else ""
    print(f"Person images to evaluate: {total}{cap}")

    model = TFLiteInferencer(model_path, threshold=threshold, min_area=min_area)

    predictions     = []
    evaluated_ids   = []
    skipped         = 0

    for i, img_id in enumerate(person_img_ids):
        img_info = images_by_id.get(img_id)
        if not img_info:
            skipped += 1
            continue

        img_path = os.path.join(images_dir, img_info["file_name"])
        image    = cv2.imread(img_path)
        if image is None:
            skipped += 1
            continue

        instances = model.infer(image)
        evaluated_ids.append(img_id)

        for mask, conf in instances:
            predictions.append({
                "image_id":     img_id,
                "category_id":  COCO_PERSON_CAT_ID,
                "segmentation": mask_to_rle(mask),
                "score":        conf,
            })

        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total} images  ({len(predictions)} predictions so far) ...")

    if skipped:
        print(f"[warn] {skipped} images skipped (not found or unreadable)")

    print(f"\nTotal predictions: {len(predictions)}")
    print("Running COCOeval ...")

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(ann_file)

    if not predictions:
        print("No predictions — check model path and threshold.")
        return

    coco_dt = coco_gt.loadRes(predictions)

    evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
    evaluator.params.imgIds = evaluated_ids
    evaluator.params.catIds = [COCO_PERSON_CAT_ID]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    sep = "=" * 52
    lines = [
        "",
        sep,
        f"  DATE      : {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  MODEL     : {os.path.basename(model_path)}",
        f"  THRESHOLD : {threshold}",
        f"  IMAGES    : {len(evaluated_ids)}",
        f"  PREDS     : {len(predictions)}",
        sep,
        "  MASK QUALITY (COCO val2017 — person only)",
        sep,
        f"  mAP@50-95 (mask) : {evaluator.stats[0]:.4f}",
        f"  mAP@50    (mask) : {evaluator.stats[1]:.4f}",
        f"  mAP@75    (mask) : {evaluator.stats[2]:.4f}",
        sep,
        "  BY OBJECT SIZE",
        sep,
        f"  mAP small  (<32²px)  : {evaluator.stats[3]:.4f}",
        f"  mAP medium (32-96px) : {evaluator.stats[4]:.4f}",
        f"  mAP large  (>96²px)  : {evaluator.stats[5]:.4f}",
        sep,
        "  RECALL",
        sep,
        f"  AR@1              : {evaluator.stats[6]:.4f}",
        f"  AR@10             : {evaluator.stats[7]:.4f}",
        f"  AR@100            : {evaluator.stats[8]:.4f}",
        f"  AR@100 small      : {evaluator.stats[9]:.4f}",
        f"  AR@100 medium     : {evaluator.stats[10]:.4f}",
        f"  AR@100 large      : {evaluator.stats[11]:.4f}",
        sep,
        "",
    ]

    for line in lines:
        print(line)

    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "coco_logs.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Results appended to {log_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="COCO mAP@50-95 (mask) evaluation for TFLite person segmentation models"
    )
    ap.add_argument("--model",       required=True,                     help="Path to .tflite model file")
    ap.add_argument("--coco-dir",    default="./datasets/coco_data",    help="COCO data directory (default: ./datasets/coco_data)")
    ap.add_argument("--threshold",   type=float, default=0.5,           help="Mask probability threshold (default: 0.5)")
    ap.add_argument("--min-area",    type=int,   default=100,           help="Min component area in pixels (default: 100)")
    ap.add_argument("--max-images",  type=int,   default=None,          help="Limit images evaluated (e.g. 500 for a quick check)")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: model not found: {args.model}")
        sys.exit(1)

    try:
        from pycocotools import mask as _
    except ImportError:
        print("Error: pycocotools not installed.")
        print("  pip install pycocotools")
        sys.exit(1)

    run_eval(args.model, args.coco_dir, args.threshold, args.min_area, args.max_images)


if __name__ == "__main__":
    main()
