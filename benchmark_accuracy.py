import sys, json, torch, numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, '/ceph/project/P6-Machine-Vision/P6/src')
from student.student_model import StudentSegmentation

def load_student(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt['args']
    tc = ckpt.get('teacher_channels', [128, 128, 256])
    model = StudentSegmentation(
        in_channels=3,
        base_channels=args.get('base_channels', 8),
        depth=args.get('depth', 4),
        teacher_channels=tc,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model, ckpt

def compute_metrics(pred, gt, threshold=0.5):
    pred = (pred > threshold).float()
    gt = gt.float()
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    iou  = (intersection + 1e-6) / (union - intersection + 1e-6)
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return iou.item(), dice.item()

def poly_to_mask(coords, size=640):
    from PIL import ImageDraw
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)
    pts = np.array(coords).reshape(-1, 2)
    pts[:, 0] *= size
    pts[:, 1] *= size
    draw.polygon([tuple(p) for p in pts], fill=255)
    return np.array(mask, dtype=np.float32) / 255.0

def run_benchmark(ckpt_path, val_dir, device, label, model_type, n_max=200):
    model, ckpt = load_student(ckpt_path, device)
    img_dir = Path(val_dir) / 'data'
    lbl_dir = Path(val_dir) / 'labels'
    img_files = sorted(img_dir.glob('*.jpg'))[:n_max]
    ious, dices = [], []
    for img_path in img_files:
        lbl_path = lbl_dir / (img_path.stem + '.txt')
        if not lbl_path.exists():
            continue
        img = Image.open(img_path).convert('RGB')
        img_t = TF.to_tensor(img)
        img_t = TF.resize(img_t, [640, 640]).unsqueeze(0).to(device)
        gt = np.zeros((640, 640), dtype=np.float32)
        for line in lbl_path.read_text().strip().split('\n'):
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = list(map(float, parts[1:]))
            gt = np.maximum(gt, poly_to_mask(coords))
        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred, _ = model(img_t)
        iou, dice = compute_metrics(pred, gt_t)
        ious.append(iou)
        dices.append(dice)
    return {
        "name": label, "type": model_type,
        "map50": None, "map5095": None, "prec": None, "rec": None,
        "iou":  round(float(np.mean(ious)),  4) if ious else 0.0,
        "dice": round(float(np.mean(dices)), 4) if dices else 0.0,
        "n_images": len(ious),
    }

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: ' + device)
val_dir = '/ceph/project/P6-Machine-Vision/P6/data/open-images-v7/validation'
project = '/ceph/project/P6-Machine-Vision/P6'

models = [
    ('trained_models/best_model.pt',        'best_model',        'student'),
    ('trained_models/best_model_deploy.pt', 'best_model_deploy', 'student'),
]

results = []
for ckpt_rel, label, mtype in models:
    ckpt_path = str(Path(project) / ckpt_rel)
    print('Benchmarker: ' + label)
    r = run_benchmark(ckpt_path, val_dir, device, label, mtype)
    results.append(r)
    print('  IoU:  ' + str(r['iou']))
    print('  Dice: ' + str(r['dice']))
    print('  N:    ' + str(r['n_images']))

out = str(Path(project) / 'benchmark_results.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print('\n-- Dashboard JSON --')
print(json.dumps(results, indent=2))
