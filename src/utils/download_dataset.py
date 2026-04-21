#!/usr/bin/env python3
"""
Download image datasets directly to the cluster for teacher model inference.
No annotations needed - just raw images for YOLO to process.
"""

import argparse
import os
from pathlib import Path
import urllib.request
import zipfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path):
    """Download a file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_coco_images(output_dir: Path, split: str = "val2017"):
    """
    Download COCO images only (no annotations).

    Args:
        output_dir: Directory to save images
        split: Which split to download (train2017, val2017, test2017)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # URL for COCO images
    image_url = f"http://images.cocodataset.org/zips/{split}.zip"

    print(f"Downloading COCO {split} images...")
    image_zip = output_dir / f"{split}.zip"

    if not image_zip.exists():
        download_url(image_url, image_zip)
    else:
        print(f"Images already downloaded: {image_zip}")

    # Extract images
    print(f"Extracting {split} images...")
    images_dir = output_dir / split

    if not images_dir.exists():
        with zipfile.ZipFile(image_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"✓ Extracted to {images_dir}")

        # Optionally remove zip to save space
        print(f"Cleaning up zip file...")
        image_zip.unlink()
    else:
        print(f"✓ Images already extracted: {images_dir}")

    # Count images
    num_images = len(list(images_dir.glob("*.jpg")))
    print(f"\n✓ Ready: {num_images} images in {images_dir}")

    return images_dir


def download_open_images(output_dir: Path, num_images: int = 10000):
    """
    Download Open Images Dataset samples.
    Uses fiftyone to download a subset.
    Requires: pip install fiftyone
    """
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        print("ERROR: 'fiftyone' package not found. Install with: pip install fiftyone")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {num_images} images from Open Images Dataset...")
    print("This may take a while on first run...")

    # Download dataset
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        max_samples=num_images,
        dataset_dir=str(output_dir / "open_images_raw"),
        shuffle=True
    )

    # Export to simple image directory
    images_dir = output_dir / "open_images"
    images_dir.mkdir(exist_ok=True)

    print(f"Organizing images to {images_dir}...")
    dataset.export(
        export_dir=str(images_dir),
        dataset_type=fo.types.ImageDirectory
    )

    num_images = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
    print(f"\n✓ Ready: {num_images} images in {images_dir}")

    return images_dir


def download_custom_urls(url_file: Path, output_dir: Path):
    """
    Download images from a text file containing URLs (one per line).

    Args:
        url_file: Path to text file with image URLs
        output_dir: Directory to save images
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Downloading {len(urls)} images from {url_file}...")

    success_count = 0
    for i, url in enumerate(urls):
        filename = url.split('/')[-1].split('?')[0]  # Remove query params
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            filename = f"image_{i:05d}.jpg"

        output_path = output_dir / filename

        if output_path.exists():
            success_count += 1
            continue

        try:
            urllib.request.urlretrieve(url, output_path)
            success_count += 1
            if (i + 1) % 100 == 0:
                print(f"Downloaded {i + 1}/{len(urls)} images...")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

    print(f"\n✓ Ready: {success_count}/{len(urls)} images in {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download image datasets for teacher model inference (no annotations needed)"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['coco', 'open-images', 'custom'],
        default='coco',
        help='Which dataset to download'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='Output directory for images'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val2017',
        choices=['train2017', 'val2017', 'test2017'],
        help='COCO split to download (val2017 has ~5K images, train2017 has ~118K)'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=10000,
        help='Number of images for Open Images'
    )
    parser.add_argument(
        '--url-file',
        type=str,
        help='Text file with image URLs (for custom dataset)'
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    print("="*60)
    print("Teacher Model Inference - Dataset Downloader")
    print("="*60)

    if args.dataset == 'coco':
        images_dir = download_coco_images(output_dir, args.split)
        print(f"\n✓ Use this path for inference:")
        print(f"  {images_dir}")
        print(f"\nExample command:")
        print(f"  python src/predictions.py --model yolo11n-seg.pt --input {images_dir} --output results")

    elif args.dataset == 'open-images':
        images_dir = download_open_images(output_dir, args.num_images)
        if images_dir:
            print(f"\n✓ Use this path for inference:")
            print(f"  {images_dir}")

    elif args.dataset == 'custom':
        if not args.url_file:
            print("ERROR: --url-file required for custom dataset")
            return
        images_dir = download_custom_urls(Path(args.url_file), output_dir / "custom")
        print(f"\n✓ Use this path for inference:")
        print(f"  {images_dir}")


if __name__ == '__main__':
    main()
