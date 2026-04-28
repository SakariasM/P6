import json, os
from pathlib import Path

ANN = Path('/ceph/project/P6-Machine-Vision/P6/data/instances_val2017.json')
OUT = Path('/ceph/project/P6-Machine-Vision/P6/data/labels/val2017')
OUT.mkdir(parents=True, exist_ok=True)

with open(ANN) as f:
    coco = json.load(f)

img_info = {img['id']: img for img in coco['images']}
anns_by_img = {}
for ann in coco['annotations']:
    anns_by_img.setdefault(ann['image_id'], []).append(ann)

for img_id, anns in anns_by_img.items():
    info = img_info[img_id]
    w, h = info['width'], info['height']
    lines = []
    for ann in anns:
        if ann.get('iscrowd') or not ann.get('segmentation'):
            continue
        cat = ann['category_id'] - 1
        for seg in ann['segmentation']:
            if len(seg) < 6:
                continue
            pts = []
            for i in range(0, len(seg), 2):
                pts.append(str(round(seg[i]/w, 6)))
                pts.append(str(round(seg[i+1]/h, 6)))
            lines.append(str(cat) + ' ' + ' '.join(pts))
    if lines:
        out_file = OUT / (Path(info['file_name']).stem + '.txt')
        out_file.write_text('\n'.join(lines))

print('Konverteret: ' + str(len(list(OUT.glob('*.txt')))) + ' label-filer')
