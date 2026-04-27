import json, os
from pathlib import Path

PROJECT = Path('/ceph/project/P6-Machine-Vision/P6')

print('')
print('===== Student ablation (val IoU/Dice) =====')
print('%-25s %8s %8s %6s' % ('Variant', 'IoU', 'Dice', 'Epoch'))
print('-' * 52)
base = PROJECT / 'trained_models' / 'ablation'
for v in sorted(os.listdir(base)):
    f = base / v / 'training_history.json'
    if not f.exists():
        print('%-25s  ingen data' % v)
        continue
    data = json.load(open(f))
    best = max(data, key=lambda x: x.get('val_iou', 0))
    marker = ' <-- bedst' if v == 'backbone_plus_neck1' else ''
    print('%-25s %8.4f %8.4f %6d%s' % (v, best['val_iou'], best['val_dice'], best['epoch'], marker))

print('')
print('===== Teacher COCO benchmark =====')
coco_file = PROJECT / 'coco_benchmark_results.json'
if coco_file.exists():
    r = json.load(open(coco_file))
    print('Model:          ' + r.get('model', '?'))
    print('Dataset:        ' + r.get('dataset', '?'))
    if 'box_mAP50' in r:
        print('box mAP50:      ' + str(r['box_mAP50']))
        print('box mAP50:95:   ' + str(r['box_mAP50_95']))
    if 'mask_mAP50' in r:
        print('mask mAP50:     ' + str(r['mask_mAP50']))
        print('mask mAP50:95:  ' + str(r['mask_mAP50_95']))
    print('')
    print('Reference (yolo26n-seg officiel): box=39.6  mask=33.9')
else:
    print('Ikke klar endnu -- job korer stadig')
print('')
