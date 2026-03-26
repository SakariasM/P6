"""Merge chunked prediction files from a parallel extraction run."""
import argparse
from pathlib import Path
import torch


def merge_worker(input_dir: Path, worker_id: int, save_format: str) -> list:
    """Load and concatenate all chunk files for one worker."""
    suffix = f"_worker{worker_id}"
    chunks = sorted(input_dir.glob(f"chunk_*{suffix}.{save_format}"))

    if not chunks:
        # Fall back to old single-file format
        single = input_dir / f"hybrid_teacher_predictions{suffix}.{save_format}"
        if single.exists():
            print(f"  Worker {worker_id}: loading single file {single.name}")
            return torch.load(single, weights_only=False)
        print(f"  Worker {worker_id}: no files found, skipping")
        return []

    predictions = []
    for chunk in chunks:
        preds = torch.load(chunk, weights_only=False)
        predictions.extend(preds)
        print(f"  {chunk.name}: {len(preds)} predictions")

    print(f"  Worker {worker_id} total: {len(predictions)} predictions from {len(chunks)} chunks")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Merge chunked worker prediction files")
    parser.add_argument("--input", type=str, required=True,
                        help="Directory containing chunk files")
    parser.add_argument("--output", type=str, required=True,
                        help="Path for merged output file")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--format", type=str, default="torch")
    args = parser.parse_args()

    input_dir = Path(args.input)
    all_predictions = []

    for worker_id in range(args.num_workers):
        print(f"\nWorker {worker_id}:")
        preds = merge_worker(input_dir, worker_id, args.format)
        all_predictions.extend(preds)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_predictions, output_path)
    print(f"\nMerged {len(all_predictions)} total predictions → {output_path}")


if __name__ == "__main__":
    main()
