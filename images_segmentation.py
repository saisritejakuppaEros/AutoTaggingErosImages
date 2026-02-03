import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import torch
from ultralytics.models.sam import SAM3SemanticPredictor
import shutil

def get_mask_union(masks):
    """Combine multiple masks into single binary mask"""
    if not masks:
        return None
    combined = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        combined |= mask
    return combined

def process_single_image(image_path: Path, tags: list, predictor, min_area: int):
    """Segment image iteratively until covered"""
    try:
        predictor.set_image(str(image_path))

        all_boxes = []
        all_tags = []
        used_mask = None
        remaining_tags = tags.copy()

        iteration = 0
        max_iterations = 3

        while remaining_tags and iteration < max_iterations:
            # Query with remaining tags
            results = predictor(text=remaining_tags)

            if results is None or len(results) == 0:
                break

            result = results[0]
            if not hasattr(result, 'boxes') or result.boxes is None:
                break

            boxes = result.boxes.xyxy.cpu().numpy()
            masks = result.masks.data.cpu().numpy() if hasattr(result, 'masks') and result.masks is not None else None

            if len(boxes) == 0:
                break

            new_boxes = []
            new_tags = []
            detected_tags = set()

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)

                if area < min_area:
                    continue

                # Check overlap with already detected regions
                if masks is not None and used_mask is not None:
                    mask = masks[idx]
                    overlap = np.sum(mask & used_mask) / np.sum(mask) if np.sum(mask) > 0 else 1.0
                    if overlap > 0.7:  # Skip if >70% overlap
                        continue

                tag_idx = idx % len(remaining_tags)
                tag = remaining_tags[tag_idx]

                new_boxes.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "label": tag,
                    "area": float(area)
                })
                new_tags.append(tag)
                detected_tags.add(tag)

                # Update used mask
                if masks is not None:
                    if used_mask is None:
                        used_mask = masks[idx].copy()
                    else:
                        used_mask |= masks[idx]

            all_boxes.extend(new_boxes)
            all_tags.extend(new_tags)

            # Remove detected tags
            remaining_tags = [t for t in remaining_tags if t not in detected_tags]

            if len(new_boxes) == 0:
                break

            iteration += 1

        return all_boxes

    except Exception as e:
        print(f"Error segmenting {image_path.name}: {e}")
        return []

def process_folder(folder_name: str, device_id: int, dataset_dir: str, output_dir: str,
                   min_area_threshold: int, model_path: str):
    """Process a single folder with images and JSON"""

    INPUT_JSON = f"{dataset_dir}/{folder_name}.json"
    INPUT_FOLDER = f"{dataset_dir}/{folder_name}"
    OUTPUT_JSON = f"{output_dir}/{folder_name}.json"

    # Check if inputs exist
    if not os.path.exists(INPUT_JSON):
        print(f"Skipping {folder_name}: JSON not found at {INPUT_JSON}")
        return

    if not os.path.exists(INPUT_FOLDER):
        print(f"Skipping {folder_name}: folder not found at {INPUT_FOLDER}")
        return

    # Skip if output exists
    if os.path.exists(OUTPUT_JSON):
        print(f"Skipping {folder_name}: output exists at {OUTPUT_JSON}")
        return

    print(f"\n{'='*60}")
    print(f"Processing {folder_name} on GPU {device_id}")
    print(f"{'='*60}")

    # Set device
    torch.cuda.set_device(device_id)

    # Initialize predictor
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=model_path,
        half=True,
        save=False,
        device=device_id
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)

    # Load JSON data
    print(f"Loading JSON data for {folder_name}...")
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    print(f"Found {len(data)} images in JSON")

    # Segment images
    print(f"Segmenting images for {folder_name}...")
    results = []
    skipped = 0
    processed = 0

    for item in tqdm(data, desc=f"Segmenting {folder_name}"):
        key = item.get('key', '')
        tags = item.get('tags', [])
        url = item.get('url', '')
        width = item.get('width', 0)
        height = item.get('height', 0)
        json_image_path = item.get('image_path', '')

        # Extract just the filename from JSON image_path
        # e.g., "/mnt/data0/.../002900012.jpg" -> "002900012.jpg"
        if json_image_path:
            image_filename = os.path.basename(json_image_path)
        else:
            # Fallback: use key with .jpg extension
            image_filename = f"{key}.jpg"

        # Construct correct path using our base directory
        actual_image_path = os.path.join(INPUT_FOLDER, image_filename)

        # Verify image exists
        if not os.path.exists(actual_image_path):
            skipped += 1
            results.append({
                "key": key,
                "url": url,
                "width": width,
                "height": height,
                "boxes": []
            })
            continue

        # Skip if no tags
        if not tags:
            results.append({
                "key": key,
                "url": url,
                "width": width,
                "height": height,
                "boxes": []
            })
            continue

        # Process image with iterative segmentation
        boxes = process_single_image(Path(actual_image_path), tags, predictor, min_area_threshold)
        processed += 1

        results.append({
            "key": key,
            "url": url,
            "width": width,
            "height": height,
            "boxes": boxes
        })

    # Save results
    print(f"Saving results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    total_boxes = sum(len(r['boxes']) for r in results)
    print(f"\n{'='*60}")
    print(f"Done! Segmented {processed} images for {folder_name}")
    print(f"Skipped: {skipped} images (not found)")
    print(f"Total boxes detected: {total_boxes}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description='SAM3 Segmentation for folder-based datasets')
    parser.add_argument('--folder', type=str, help='Single folder name to process (e.g., 00290)')
    parser.add_argument('--start', type=int, help='Start folder number (e.g., 290 for 00290)')
    parser.add_argument('--end', type=int, help='End folder number (exclusive)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--dataset-dir', type=str, default='/data/corerndimage/image_tagging/target_dataset')
    parser.add_argument('--output-dir', type=str, default='/data/corerndimage/image_tagging/Dataset/segmentation_output')
    parser.add_argument('--min-area', type=int, default=100)
    parser.add_argument('--model', type=str, default='sam3.pt')

    args = parser.parse_args()

    # Determine which folders to process
    if args.folder:
        folders_to_process = [args.folder]
    elif args.start is not None and args.end is not None:
        folders_to_process = [f"{i:05d}" for i in range(args.start, args.end)]
    else:
        parser.error("Either --folder or both --start and --end must be provided")
        return

    print(f"Processing {len(folders_to_process)} folder(s) on GPU {args.gpu}")
    print(f"Folders: {folders_to_process}")

    for folder_name in folders_to_process:
        try:
            process_folder(
                folder_name, args.gpu,
                dataset_dir=args.dataset_dir,
                output_dir=args.output_dir,
                min_area_threshold=args.min_area,
                model_path=args.model
            )
        except Exception as e:
            print(f"Error processing folder {folder_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()


