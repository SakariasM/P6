import sys, json, torch, numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, '/ceph/project/P6-Machine-Vision/P6/src')
from student.student_model import StudentSegmentation

FIFTYONE_DIR = Path('/ceph/home/student.aau.dk/mg67xn/fiftyone/open-images-v7/validation')
IMG_DIR      = FIFTYONE_DIR / 'data'
MASK_DIR     = FIFTYONE_DIR / 'labels' / 'masks'
PROJECT      = Path('/ceph/project/P6-Machine-Vision/P6')

def load_student(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt['args']
    tc   = ckpt.get('teacher_channels', [128, 128, 256])
    model = StudentSegmentation(
        in_channels=3,
        base_channels=args.get('base_channels', 8),
        depth=args.get('depth', 4),
        teacher_channels=tc,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model

def compute_metrics(pred, gt, threshold=0.5):
    pred = (pred > threshold).float()
    gt   = gt.float()
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    iou  = (intersection + 1e-6) / (union - intersection + 1e-6)
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return iou.item(), dice.item()

def load_gt_mask(image_stem, size=640):
    # Find alle PNG masker der starter med image_stem
    masks = list(MASK_DIR.glob(f'*/{image_stem}_*.png'))
    if not masks:
        return None
    gt = np.zeros((size, size), dtype=np.float32)
    for mask_path in masks:
        m = np.array(Image.open(mask_path).convert('L').resize((size, size)))
        gt = np.maximum(gt, (m > 128).astype(np.float32))
    return gt

def run_benchmark(ckpt_path, label, n_max=200):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    model = load_student(ckpt_path, device)

    img_files = sorted(IMG_DIR.glob('*.jpg'))[:n_max]
    ious, dices = [], []

    for img_path in img_files:
        gt = load_gt_mask(img_path.stem)
        if gt is None:
            continue

        img   = Image.open(img_path).convert('RGB')
        img_t = TF.to_tensor(img)
        img_t = TF.resize(img_t, [640, 640]).unsqueeze(0).to(device)
        gt_t  = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred, _ = model(img_t)

        iou, dice = compute_metrics(pred, gt_t)
        ious.append(iou)
        dices.append(dice)

    return {
        "name":      label,
        "iou":       round(float(np.mean(ious)),  4) if ious else 0.0,
        "dice":      round(float(np.mean(dices)), 4) if dices else 0.0,
        "n_images":  len(ious),
    }

models = [
    (str(PROJECT / 'trained_models/best_model.pt'),        'best_model'),
    (str(PROJECT / 'trained_models/best_model_deploy.pt'), 'best_model_deploy'),
]

results = []
for ckpt_path, label in models:
    print(f'Benchmarker: {label}')
    r = run_benchmark(ckpt_path, label)
    results.append(r)
    print(f'  IoU:  {r["iou"]}')
    print(f'  Dice: {r["dice"]}')
    print(f'  N:    {r["n_images"]}')

out = str(PROJECT / 'benchmark_results.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)

print('\n-- Resultater --')
print(json.dumps(results, indent=2))
