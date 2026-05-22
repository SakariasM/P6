Knowledge Distillation for Person Segmentation on the Edge
Offline knowledge distillation from a YOLO26n-seg teacher into a lightweight
U-Net student for binary person segmentation, targeting CPU-only
deployment on a Raspberry Pi 5.
The student is trained to mimic the teacher's intermediate feature maps and
pseudo-labels rather than learning from ground-truth masks directly. A
multi-component distillation loss (attention transfer + feature mimicry +
relational + segmentation) is used, and an ablation study sweeps which teacher
layers are most useful to distil from.

This repository accompanies our P6 project report (P6.pdf)

Installation:
bashgit clone <TODO: repo url>
cd <TODO: repo name>

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt   # TODO: add/verify requirements.txt
export PYTHONPATH=src

Download Openimages v7 dataset on their website. Store the data in same folder as the project.

## 1. Extract teacher features

Features are captured with PyTorch forward hooks (`YOLOFeatureExtractor`) on
five candidate teacher layers: `[4, 6, 9, 13, 16]`. Extraction is done in
chunks (~500 images each) so only one chunk is loaded into memory at a time
during training.

Quick smoke test on a single image:

```bash
python -m teacher.feature_extractor \
    --model yolo26n-seg.pt \
    --image path/to/test.jpg \
    --show-shapes
```


## 2. Train the student

The training entry point is `training.hybrid_distillation_train`. Example
(values mirror the SLURM defaults):

```bash
python -m training.hybrid_distillation_train \
    --predictions   /path/to/teacher_predictions \
    --image-root    /path/to/openimages/train/data \
    --epochs        50 \
    --batch-size    8 \
    --lr            1e-4 \
    --base-channels 8 \
    --depth         4 \
    --img-size      640 \
    --attention-weight 1.0 \
    --mimicry-weight   2.0 \
    --relation-weight  1.0 \
    --seg-weight       0.5 \
    --grad-clip     1.0 \
    --val-split     0.1 \
    --patience      5 \
    --augment \
    --teacher-layers model.4 model.6 model.9 \
    --output-dir    trained_models/student
```

On SLURM:

```bash
sbatch slurm/train_student.slurm
```


## 3. Run the ablation study

The ablation study trains the student against different subsets of teacher
layers, defined in `configs/ablation_configs.json`. Pick a config by name:

```bash
# Full-CBAM baseline for a given config
CONFIG=backbone_3 sbatch slurm/train_ablation.slurm

# Same config, but CBAM disabled at encoder level 0 (--cbam-levels 1 2 3)
CONFIG=backbone_3 sbatch slurm/train_ablation_scratch.slurm
```







training flags:

| Flag | Default (SLURM) | Meaning |
|------|-----------------|---------|
| `--base-channels` | `8` | U-Net base channel count (script default is 32 — override!) |
| `--depth` | `4` | Encoder/decoder levels |
| `--teacher-layers` | per config | Teacher layers to distil from, e.g. `model.4 model.9` |
| `--cbam-levels` | all | Encoder levels that keep CBAM, e.g. `1 2 3` disables level 0 |
| `--img-size` | `640` | Training resolution |
| `--*-weight` | see above | Per-term loss weights |





## Slurm wiring

Which entry point each job runs:

| Slurm job | Entry point |
| --- | --- |
| `extract_teacher.slurm` | `teacher.hybrid_predictions` |
| `extract_all_layers.slurm` | `teacher.hybrid_predictions` |
| `full_pipeline.slurm` | `teacher.hybrid_predictions` → `training.hybrid_distillation_train` |
| `train_student.slurm` | `training.hybrid_distillation_train` |
| `train_ablation.slurm` | `training.hybrid_distillation_train` |
| `train_ablation_scratch.slurm` | `training.hybrid_distillation_train` |
| `backfill_val.slurm` | `backfill_val_metrics` |
| `run_benchmark.sh` | `benchmark_accuracy.py` |
| `run_unit_benchmarking.slurm` | `benchmark_accuracy.py` |
| `run_tests.slurm` | `tests/` |

```
             *     ,MMM8&&&.            *
                  MMMM88&&&&&    .
                 MMMM88&&&&&&&
     *           MMM88&&&&&&&&
                 MMM88&&&&&&&&
                 'MMM88&&&&&&'
                   'MMM8&&&'      *
           /\/|_      __/\\
          /    -\    /-   ~\  .              '
          \    = Y =T_ =   /
           )==*(`     `) ~ \
          /     \     /     \
          |     |     ) ~   (
         /       \   /     ~ \
         \       /   \~     ~/
  jgs_/\_/\__  _/_/\_/\__~__/_/\_/\_/\_/\_/\_
  |  |  |  | ) ) |  |  | ((  |  |  |  |  |  |
  |  |  |  |( (  |  |  |  \\ |  |  |  |  |  |
  |  |  |  | )_) |  |  |  |))|  |  |  |  |  |
  |  |  |  |  |  |  |  |  (/ |  |  |  |  |  |
  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
```
