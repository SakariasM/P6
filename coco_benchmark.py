import json, torch
from pathlib import Path

PROJECT = Path('/ceph/project/P6-Machine-Vision/P6')
OUT_PATH = PROJECT / 'coco_benchmark_results.json'

def run():
    from ultralytics import YOLO

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ' + device)

    yaml_path = PROJECT / 'data' / 'coco_local.yaml'
    with open(yaml_path, 'w') as f:
        f.write('path: ' + str(PROJECT / 'data') + '\n')
        f.write('train: val2017\n')
        f.write('val: val2017\n')
        f.write('nc: 80\n')
        f.write('names:\n')
        names = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
        for i, n in enumerate(names):
            f.write('  ' + str(i) + ': ' + n + '\n')

    model = YOLO('yolo26n-seg.pt')
    print('Korer val() pa COCO val2017...')
    results = model.val(
        data=str(yaml_path),
        split='val',
        imgsz=640,
        batch=16,
        device=device,
        verbose=True,
    )

    output = {
        "model": "yolo26n-seg",
        "dataset": "COCO val2017",
        "box_mAP50":     round(float(results.box.map50), 4),
        "box_mAP50_95":  round(float(results.box.map),   4),
        "box_precision": round(float(results.box.mp),    4),
        "box_recall":    round(float(results.box.mr),    4),
    }
    if hasattr(results, 'seg') and results.seg is not None:
        output["mask_mAP50"]    = round(float(results.seg.map50), 4)
        output["mask_mAP50_95"] = round(float(results.seg.map),   4)

    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print('\n===== Teacher COCO Benchmark =====')
    for k, v in output.items():
        print(k + ': ' + str(v))
    print('==================================')
    print('Reference: box=39.6  mask=33.9')

if __name__ == '__main__':
    run()
