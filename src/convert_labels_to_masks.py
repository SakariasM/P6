"""
Convert COCO YOLO polygon .txt labels to PNG masks
Only needs to be run once.
Usage:
    python3 src/convert_labels_to_masks.py
    python3 src/convert_labels_to_masks.py --labels-dir data/labels/val2017 --masks-dir data/masks/val2017
    python3 src/convert_labels_to_masks.py --img-size 320 --class-id 0
"""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# --- Default Paths and sizes, if nothing is specified through the command line ---
DEFAULT_LABELS_DIR = Path('/ceph/project/P6-Machine-Vision/P6/data/labels/val2017')
DEFAULT_MASKS_DIR  = Path('/ceph/project/P6-Machine-Vision/P6/data/masks/val2017')
DEFAULT_SIZE       = 640

def parse_args():
    # --- Reads and returns command line arguments ---
    parser = argparse.ArgumentParser(description='Convert YOLO polygon labels to PNG masks')
    parser.add_argument('--labels-dir', type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument('--masks-dir',  type=Path, default=DEFAULT_MASKS_DIR)
    parser.add_argument('--img-size',   type=int,  default=DEFAULT_SIZE,
                        help='Output mask resolution (default: 640)')
    parser.add_argument('--class-id',   type=int,  default=0,
                        help='Only convert this class ID (default: 0 = person)')
    return parser.parse_args()

def poly_to_mask(coords, size):
    """
        Converts a list of normalized (x, y) coordinates to binary PNG mask.

        The YOLO format stores polygon coordinates as values between 0 and 1
        relative to the image width / height.
        The picture scale them up to pixel and draw the polygon as a white filled
        shape on a black background.

        Args:
            coords: flat list of normalized coordinates [x1, y1, x2, y2, ...]
            size: Images size in pixels (both width and height)

        Returns:
            NumPy array (size * size) with 255 inside the polygon, 0 outside
    """

    # Create a blank black image in grayscale ('L' = 8-bit grayscale)
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)

    # Convert the flat list to (N, 2) array and scale to pixel coordinates
    pts  = np.array(coords).reshape(-1, 2)
    pts[:, 0] *= size
    pts[:, 1] *= size

    # Draw the filled polygon with white (255) on the black images
    draw.polygon([tuple(p) for p in pts], fill=255)

    return np.array(mask)

def convert(args):
    """
        Main conversion loop: iterates over all .txt label files
        and saves the corresponding binary PNG masks

        For each image, all polygons for the selected class, are combined into one
        overall mask by taking the pixel-wise maximum (union)

        Images without the selected class are skipped.
    """

    # Create the output directory if it does not exist
    args.masks_dir.mkdir(parents=True, exist_ok=True)
    txt_files = list(args.labels_dir.glob('*.txt'))
    print(f'Converting {len(txt_files)} label files...')
    converted, skipped = 0, 0

    for i, txt_path in enumerate(txt_files):
        # Show continuous progress in the terminal without newline for each file
        print(f'\r  [{i+1}/{len(txt_files)}]', end='', flush=True)

        # Reads all lines from label file
        lines    = txt_path.read_text().strip().split('\n')

        # Start with a completely black (empty) mask for this image
        combined = np.zeros((args.img_size, args.img_size), dtype=np.uint8)

        for line in lines:
            parts = line.strip().split()

            # A valid polygon line must have at least class_id + 3 points (6 numbers) = 7 parts
            if len(parts) < 7:
                continue

            # Skip if the line does not belong to the desired class
            if int(parts[0]) != args.class_id:
                continue

            # The rest of the line is the polygon coordinates
            coords = list(map(float, parts[1:]))

            # Convert polygon to mask and add it to the combined mask (union)
            mask   = poly_to_mask(coords, args.img_size)
            combined = np.maximum(combined, mask)

        # if no pixels are set, there are no relevant objects in the images
        if combined.max() == 0:
            skipped += 1
            continue

        # Save the combined mask as PNG with the same filename as the label file
        out_path = args.masks_dir / (txt_path.stem + '.png')
        Image.fromarray(combined).save(out_path)
        converted += 1

    print(f'\nDone: {converted} masks saved, {skipped} without class {args.class_id} skipped')
    print(f'Masks saved to: {args.masks_dir}')

if __name__ == '__main__':
    # Parse command linje-arguments and run conversion
    convert(parse_args())