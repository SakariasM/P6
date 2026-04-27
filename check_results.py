import json, os

base = "/ceph/project/P6-Machine-Vision/P6/trained_models/ablation"
variants = sorted(os.listdir(base))

print("")
print("%-25s %8s %8s %6s" % ("Variant", "IoU", "Dice", "Epoch"))
print("-" * 52)
for v in variants:
    f = os.path.join(base, v, "training_history.json")
    if not os.path.exists(f):
        print("%-25s  ingen data" % v)
        continue
    data = json.load(open(f))
    best = max(data, key=lambda x: x.get("val_iou", 0))
    print("%-25s %8.4f %8.4f %6d" % (v, best["val_iou"], best["val_dice"], best["epoch"]))
print("")
