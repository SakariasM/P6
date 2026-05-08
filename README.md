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
