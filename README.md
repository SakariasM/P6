# P6 Benchmark Workspace

This repo benchmarks segmentation masks generated on the Pi against SAM2 ground truth.

## Layout

- `tools/` - runnable scripts (`benchmark.py`, `stream.sh`, GT generators)
- `docs/` - runbooks, notes, and setup instructions
- `data/gt/` - source footage and pre-generated ground-truth videos
- `data/preds/` - prediction masks pulled back from the Pi
- `runs/benchmark/` - benchmark logs and comparison images
- `runs/stream/` - stream-run logs and optional debug video
- `runs/gt-gen/` - GT-generation debug output
- `cache/gt-gen/` - GT-generation frames, masks, and scan caches

## Common commands

```bash
bash tools/stream.sh data/gt/test_footage_40s_1080p_30fps.mp4
python3 tools/benchmark.py --gt data/gt/gt_mask_1080p_30fps.mp4 --pred data/preds/pred_mask_yolo26n-seg_1080x1920.mp4
```

See `docs/HOW_TO_RUN.md` for the full workflow.
