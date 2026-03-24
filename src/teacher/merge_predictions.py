"""Merge worker prediction files from a parallel extraction run."""
import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description="Merge parallel worker prediction files")
    parser.add_argument("--input", type=str, required=True,
                        help="Directory containing worker prediction files")
    parser.add_argument("--output", type=str, required=True,
                        help="Path for merged output file")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    input_dir = Path(args.input)
    all_predictions = []

    for worker_id in range(args.num_workers):
        # Prefer final output over checkpoint
        final = input_dir / f"hybrid_teacher_predictions_worker{worker_id}.torch"
        checkpoint = input_dir / f"checkpoint_worker{worker_id}.torch"

        if final.exists():
            path = final
        elif checkpoint.exists():
            path = checkpoint
            print(f"Warning: worker {worker_id} has no final output, using checkpoint")
        else:
            print(f"Warning: no file found for worker {worker_id}, skipping")
            continue

        preds = torch.load(path, weights_only=False)
        print(f"Worker {worker_id}: {len(preds)} predictions from {path.name}")
        all_predictions.extend(preds)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_predictions, output_path)
    print(f"\nMerged {len(all_predictions)} total predictions → {output_path}")


if __name__ == "__main__":
    main()
