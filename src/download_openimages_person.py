#!/usr/bin/env python3
"""
Download Open Images V7 - Person class images only (no annotations).
Downloads both train and validation splits.
"""

import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path


OUTPUT_DIR = Path("data/open_images_person")


def download_split(split: str, max_samples: int = None):
    """Download a single split of Person images (no annotations)."""
    print(f"\n{'='*60}")
    print(f"Downloading Open Images V7 - Person images ({split})")
    if max_samples:
        print(f"Limiting to {max_samples} images")
    print(f"{'='*60}\n")

    dataset_name = f"open-images-v7-person-{split}"

    # Delete existing fiftyone dataset entry if it exists (from a previous partial run)
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["classifications"],
        classes=["Person"],
        only_matching=True,
        max_samples=max_samples,
        shuffle=True,
        num_workers=16,
        dataset_name=dataset_name,
        dataset_dir=str(OUTPUT_DIR / f"raw_{split}"),
    )

    num_samples = len(dataset)
    print(f"\n{split}: {num_samples} images containing Person")

    return dataset_name, num_samples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 64K training images, 16K validation images
    max_samples = {
        "train": 64000,
        "validation": 16000
    }

    total = 0
    dataset_names = []

    for split in ["train", "validation"]:
        name, count = download_split(split, max_samples=max_samples[split])
        dataset_names.append(name)
        total += count

    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Total images: {total}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"{'='*60}")

    # Clean up fiftyone dataset entries
    for name in dataset_names:
        if fo.dataset_exists(name):
            fo.delete_dataset(name)


if __name__ == "__main__":
    main()
