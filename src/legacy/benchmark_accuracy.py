"""
Ekstern benchmark: YOLO-variants + student model on COCO val2017.
 
Runs all models på COCO val2017 (eksternal dataset - not trained on).
YOLO-models evaluated with mAP50/mAP50-95 through Ultralytics .val().
Student model evaluated with IoU and Dice against ground truth masks.
 
Usage:
    python src/benchmark_accuracy.py
    python src/benchmark_accuracy.py --skip-student
    python src/benchmark_accuracy.py --skip-yolo
    python src/benchmark_accuracy.py --n-max 50
"""
 
import sys
import json
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
 
sys.path.insert(0, str(Path(__file__).parent))
from student.student_model import StudentSegmentation
 
PROJECT       = Path('/ceph/project/P6-Machine-Vision/P6')
COCO_IMG_DIR  = PROJECT / 'data' / 'images' / 'val2017'
COCO_MASK_DIR = PROJECT / 'data' / 'labels' / 'val2017'
COCO_YAML     = PROJECT / 'data' / 'coco_local.yaml'
 
# YOLO-variants (downloaded automatically by Ultralytics)
YOLO_MODELS = [
    ('yolo26n-seg.pt', 'YOLO26n-seg'),
    ('yolo26s-seg.pt', 'YOLO26s-seg'),
    ('yolo26m-seg.pt', 'YOLO26m-seg'),
    ('yolo26l-seg.pt', 'YOLO26l-seg'),
    ('yolo26x-seg.pt', 'YOLO26x-seg'),
]
 
# Student models
STUDENT_MODELS = [
    (PROJECT / 'trained_models/best_model.pt',        'Student (best)'),
    (PROJECT / 'trained_models/best_model_deploy.pt', 'Student (deploy)'),
]
 
# Reference values from Ultralytics dokumentation
YOLO_REFERENCE = {
    'yolo26n-seg.pt': {'mask_mAP50_95': 33.9},
    'yolo26s-seg.pt': {'mask_mAP50_95': 40.0},
    'yolo26m-seg.pt': {'mask_mAP50_95': 44.1},
}
 
# ── YOLO IoU benchmark (same 200 images like Student) ──────────────────────
def run_yolo_iou_benchmark(model_name, label, device, n_max=200):
    """
    Evaluates a YOLO model with IoU and Dice against COCO ground truth masks.
 
    Runs inference on the first n_max images in COCO val2017, combines all
    person masks into a single prediction mask, and compares against ground truth.
 
    Args:
        model_name: YOLO model filename, e.g. 'yolo26n-seg.pt'
        label:      Human-readable name for printing, e.g. 'YOLO26n-seg'
        device:     'cuda' or 'cpu'
        n_max:      Maximum number of images to evaluate (default: 200)
 
    Returns:
        Tuple (mean_iou, mean_dice, n_evaluated).
    """
    from ultralytics import YOLO
    model     = YOLO(model_name)
    img_files = sorted(COCO_IMG_DIR.glob('*.jpg'))[:n_max]
    ious, dices = [], []
    for img_path in img_files:
        gt = load_gt_mask(img_path.stem)
        if gt is None:
            continue
        results = model.predict(str(img_path), imgsz=640, device=device,
                                classes=[0], verbose=False)
        r = results[0]
        pred_mask = np.zeros((640, 640), dtype=np.float32)
        if r.masks is not None:
            for m in r.masks.data:
                m_np = m.cpu().numpy()

            # BUG: resize is outside the loop — only the last mask is checked.
            # Each mask should be resized individually inside the loop above.
            if m_np.shape != pred_mask.shape:
                m_np = cv2.resize(
                    m_np,
                    (pred_mask.shape[1], pred_mask.shape[0]),  # cv2 expects (width, height)
                    interpolation=cv2.INTER_NEAREST,
                )

            pred_mask = np.maximum(pred_mask, m_np)
        pred_t = torch.from_numpy(pred_mask)
        iou, dice = compute_metrics(pred_t, gt)
        ious.append(iou)
        dices.append(dice)
    return round(float(np.mean(ious)), 4) if ious else 0.0, round(float(np.mean(dices)), 4) if dices else 0.0, len(ious)

# ── YOLO benchmark ────────────────────────────────────────────────────────────
 
def run_yolo_benchmark(model_name, label, device, n_max=5000):
    """
    Full YOLO evaluation: mAP via Ultralytics .val() + IoU/Dice on a subset.
 
    Args:
        model_name: YOLO model filename, e.g. 'yolo26n-seg.pt'
        label:      Human-readable name for printing
        device:     'cuda' or 'cpu'
        n_max:      Maximum number of images for IoU/Dice evaluation (default: 5000)
 
    Returns:
        Dict with keys: model, type, box_mAP50, box_mAP50_95, mask_mAP50,
        mask_mAP50_95, iou, dice, n_images.
    """
    from ultralytics import YOLO
    print(f'\nLoader: {label}')
    model   = YOLO(model_name)
    results = model.val(
        data=str(COCO_YAML),
        split='val',
        imgsz=640,
        batch=16,
        device=device,
        verbose=True,
        classes=[0],
    )
    output = {
        'model':         label,
        'type':          'yolo',
        'box_mAP50':     round(float(results.box.map50), 4),
        'box_mAP50_95':  round(float(results.box.map),   4),
        'mask_mAP50':    None,
        'mask_mAP50_95': None,
        'iou':           None,
        'dice':          None,
        'n_images':      None,
    }
    if hasattr(results, 'seg') and results.seg is not None:
        output['mask_mAP50']    = round(float(results.seg.map50), 4)
        output['mask_mAP50_95'] = round(float(results.seg.map),   4)
    iou, dice, n = run_yolo_iou_benchmark(model_name, label, device, n_max=n_max)
    output['iou']      = iou
    output['dice']     = dice
    output['n_images'] = n
    return output
 
 
# ── Student benchmark ─────────────────────────────────────────────────────────
 
def load_student(ckpt_path, device):
    """
    Loads a saved student model from a checkpoint file.
 
    Reads model architecture parameters from the checkpoint's 'args' field
    and loads the weights into a new StudentSegmentation instance.
 
    Args:
        ckpt_path: Path to .pt checkpoint file
        device:    'cuda' or 'cpu'
 
    Returns:
        StudentSegmentation model in eval mode.
    """
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    args  = ckpt.get('args', {})
    tc    = ckpt.get('teacher_channels', [128, 128, 256])
    model = StudentSegmentation(
        in_channels=3,
        base_channels=args.get('base_channels', 8),
        depth=args.get('depth', 4),
        teacher_channels=tc,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model
 
 
def load_gt_mask(image_stem, size=640):
    """
    Loads ground truth segmentation mask from a YOLO-format .txt file.
 
    Each polygon (class 0 = person) is drawn into a binary mask.
    Coordinates are normalised [0,1] and scaled to size.
 
    Args:
        image_stem: Filename without extension, e.g. '000000001000'
        size:       Output mask size in pixels (default: 640)
 
    Returns:
        np.ndarray of shape (size, size) with values in [0, 1],
        or None if no person annotations are found.
    """
    txt_path = COCO_MASK_DIR / (image_stem + '.txt')
    if not txt_path.exists():
        return None
    mask = np.zeros((size, size), dtype=np.float32)
    lines = txt_path.read_text().strip().split('\n')
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        if parts[0] != '0':  # skip if not class 0 (person)
            continue

        # Convert normalised coordinates [0,1] to pixel coordinates
        coords = list(map(float, parts[1:]))
        pts = np.array(coords).reshape(-1, 2)
        pts[:, 0] *= size
        pts[:, 1] *= size
        from PIL import ImageDraw

        # Draw polygon on temporary mask and merge into main mask
        tmp = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(tmp)
        draw.polygon([tuple(p) for p in pts], fill=255)
        mask = np.maximum(mask, np.array(tmp) / 255.0)
    if mask.max() == 0:
        return None
    return mask
 
 
def compute_metrics(pred, gt, threshold=0.5):
    """
    Computes IoU and Dice coefficient between prediction and ground truth.
 
    Args:
        pred:      Tensor of predicted probabilities (float)
        gt:        np.ndarray ground truth mask (float, values in [0,1])
        threshold: Threshold for binarising pred (default: 0.5)
 
    Returns:
        Tuple (iou, dice) as Python floats.
    """
    pred = (pred > threshold).float()
    gt   = torch.from_numpy(gt).to(pred.device)
    intersection = (pred * gt).sum()
    union        = pred.sum() + gt.sum()
    iou  = (intersection + 1e-6) / (union - intersection + 1e-6)
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return iou.item(), dice.item()
 
 
def run_student_benchmark(ckpt_path, label, device, n_max=200):
    """
    Evaluates a student model with IoU and Dice against COCO ground truth masks.
 
    Args:
        ckpt_path: Path to .pt checkpoint file
        label:     Human-readable name for printing
        device:    'cuda' or 'cpu'
        n_max:     Maximum number of images to evaluate (default: 200)
 
    Returns:
        Dict with keys: model, type, box_mAP50, box_mAP50_95, mask_mAP50,
        mask_mAP50_95, iou, dice, n_images.
    """
    print(f'\nLoader: {label}')
    model     = load_student(ckpt_path, device)
    img_files = sorted(COCO_IMG_DIR.glob('*.jpg'))[:n_max]
    ious, dices = [], []
 
    for i, img_path in enumerate(img_files):
        print(f'\r  [{i+1}/{len(img_files)}]', end='', flush=True)
        gt = load_gt_mask(img_path.stem)
        if gt is None:
            continue
        img_t = TF.to_tensor(Image.open(img_path).convert('RGB'))
        img_t = TF.resize(img_t, [640, 640]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred, _ = model(img_t)
        iou, dice = compute_metrics(pred.squeeze(), gt)
        ious.append(iou)
        dices.append(dice)
 
    print()
    return {
        'model':         label,
        'type':          'student',
        'box_mAP50':     None,
        'box_mAP50_95':  None,
        'mask_mAP50':    None,
        'mask_mAP50_95': None,
        'iou':           round(float(np.mean(ious)),  4) if ious else 0.0,
        'dice':          round(float(np.mean(dices)), 4) if dices else 0.0,
        'n_images':      len(ious),
    }
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(description='Ekstern benchmark på COCO val2017')
    parser.add_argument('--n-max', type=int, default=200,
                        help='Max billeder til student benchmark (default: 200)')
    parser.add_argument('--skip-yolo', action='store_true',
                        help='Spring YOLO-varianter over')
    parser.add_argument('--skip-student', action='store_true',
                        help='Spring student over')
    args = parser.parse_args()
 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device:  {device}')
    print(f'Dataset: COCO val2017 ({COCO_IMG_DIR})')
 
    if not COCO_IMG_DIR.exists():
        print(f'FEJL: COCO billeder ikke fundet: {COCO_IMG_DIR}')
        return
 
    results = []
 
    # YOLO variants
    if not args.skip_yolo:
        print(f'\n{"=" * 60}')
        print('YOLO-varianter (mAP via Ultralytics .val())')
        print(f'{"=" * 60}')
        for model_name, label in YOLO_MODELS:
            r = run_yolo_benchmark(model_name, label, device, n_max=args.n_max)
            results.append(r)
 
    # Student models
    if not args.skip_student:
        if not COCO_MASK_DIR.exists():
            print(f'\nSKIP student: PNG-masker ikke fundet i {COCO_MASK_DIR}')
            print('Kør først: python src/convert_labels_to_masks.py')
        else:
            print(f'\n{"=" * 60}')
            print('Student-modeller (IoU/Dice mod ground truth)')
            print(f'{"=" * 60}')
            for ckpt_path, label in STUDENT_MODELS:
                if not ckpt_path.exists():
                    print(f'SKIP {label}: ikke fundet')
                    continue
                r = run_student_benchmark(ckpt_path, label, device, n_max=args.n_max)
                results.append(r)
 
    # Print results table
    print(f"\n{'─' * 75}")
    print(f"{'Model':<22} {'Type':<10} {'box mAP50-95':>13} {'mask mAP50-95':>14} {'IoU':>8} {'Dice':>8}")
    print(f"{'─' * 75}")
    for r in results:
        print(
            f"{r['model']:<22} "
            f"{r['type']:<10} "
            f"{str(r['box_mAP50_95'] or '—'):>13} "
            f"{str(r['mask_mAP50_95'] or '—'):>14} "
            f"{str(r['iou'] or '—'):>8} "
            f"{str(r['dice'] or '—'):>8}"
        )
    print(f"{'─' * 75}")
 
    # Save results
    out = PROJECT / 'results' / 'benchmark_accuracy_results.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nGemt: {out}')
 
 
if __name__ == '__main__':
    main()