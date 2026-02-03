import os
import json
import tarfile
import pyarrow.parquet as pq
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

def process_file(file_num: int, device_id: int, base_dataset_dir: str, base_tags_dir: str,
                 base_tmp_dir: str, base_output_dir: str, min_area_threshold: int, model_path: str):
    """Process a single parquet/tar file pair"""
    
    file_id = f"{file_num:05d}"
    
    INPUT_PARQUET = f"{base_dataset_dir}/{file_id}.parquet"
    INPUT_TAR = f"{base_dataset_dir}/{file_id}.tar"
    INPUT_TAGS = f"{base_tags_dir}/{file_id}.json"
    TMP_IMAGE_DIR = f"{base_tmp_dir}/{file_id}_gpu{device_id}"
    OUTPUT_JSON = f"{base_output_dir}/{file_id}.json"
    
    if not os.path.exists(INPUT_PARQUET):
        print(f"Skipping {file_id}: parquet not found")
        return
    if not os.path.exists(INPUT_TAR):
        print(f"Skipping {file_id}: tar not found")
        return
    if not os.path.exists(INPUT_TAGS):
        print(f"Skipping {file_id}: tags not found")
        return
    if os.path.exists(OUTPUT_JSON):
        print(f"Skipping {file_id}: output exists")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {file_id} on GPU {device_id}")
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
    
    os.makedirs(TMP_IMAGE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)
    
    # Load tags
    print(f"Loading tags for {file_id}...")
    with open(INPUT_TAGS, 'r') as f:
        tags_data = json.load(f)
    
    tags_map = {item['key']: item['tags'] for item in tags_data}
    
    # Read metadata
    print(f"Reading parquet metadata for {file_id}...")
    pf = pq.ParquetFile(INPUT_PARQUET)
    metadata_map = {}
    for batch in pf.iter_batches(batch_size=1000):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            key = row.get('__key__', row.get('key', ''))
            metadata_map[key] = {
                'url': row.get('url', ''),
                'width': row.get('width', 0),
                'height': row.get('height', 0)
            }
    
    # Extract images
    print(f"Extracting images to {TMP_IMAGE_DIR}...")
    image_keys = []
    with tarfile.open(INPUT_TAR, "r") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(('.jpg', '.png', '.jpeg'))]
        for member in tqdm(members, desc=f"Extracting {file_id}"):
            key = Path(member.name).stem
            ext = Path(member.name).suffix
            output_path = Path(TMP_IMAGE_DIR) / f"{key}{ext}"
            
            with open(output_path, "wb") as f:
                f.write(tar.extractfile(member).read())
            
            image_keys.append((key, output_path))
    
    print(f"Extracted {len(image_keys)} images")
    
    # Segment images
    print(f"Segmenting images for {file_id}...")
    results = []
    
    for key, img_path in tqdm(image_keys, desc=f"Segmenting {file_id}"):
        tags = tags_map.get(key, [])
        if not tags:
            results.append({
                "key": key,
                "url": metadata_map.get(key, {}).get('url', ''),
                "width": metadata_map.get(key, {}).get('width', 0),
                "height": metadata_map.get(key, {}).get('height', 0),
                "boxes": []
            })
            continue
        
        boxes = process_single_image(img_path, tags, predictor, min_area_threshold)
        
        meta = metadata_map.get(key, {})
        results.append({
            "key": key,
            "url": meta.get('url', ''),
            "width": meta.get('width', 0),
            "height": meta.get('height', 0),
            "boxes": boxes
        })
    
    # Save results
    print(f"Saving results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Done! Segmented {len(results)} images for {file_id}")
    
    # Cleanup
    print(f"Cleaning up tmp directory for {file_id}...")
    shutil.rmtree(TMP_IMAGE_DIR, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Segment images using SAM3 with VLM tags")
    parser.add_argument('--start', type=int, required=True, help='Start file number (inclusive)')
    parser.add_argument('--end', type=int, required=True, help='End file number (exclusive)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--dataset-dir', type=str,
                        default='/data/corerndimage/image_tagging/Dataset/internet_indian_dataset',
                        help='Base directory for parquet/tar dataset files')
    parser.add_argument('--tags-dir', type=str,
                        default='/data/corerndimage/image_tagging/Dataset/vlm_tags',
                        help='Base directory for VLM tag JSON files')
    parser.add_argument('--tmp-dir', type=str, default='./tmp_seg',
                        help='Temporary directory for extracted images')
    parser.add_argument('--output-dir', type=str,
                        default='/data/corerndimage/image_tagging/Dataset/segmentation',
                        help='Output directory for segmentation JSON results')
    parser.add_argument('--min-area', type=int, default=100,
                        help='Minimum bbox area in pixels to keep')
    parser.add_argument('--model', type=str, default='sam3.pt',
                        help='Path to SAM3 model weights')
    
    args = parser.parse_args()
    
    print(f"Processing files {args.start:05d} to {args.end-1:05d} on GPU {args.gpu}")
    print(f"Dataset: {args.dataset_dir}, Tags: {args.tags_dir}, Output: {args.output_dir}")
    
    for file_num in range(args.start, args.end):
        try:
            process_file(
                file_num, args.gpu,
                base_dataset_dir=args.dataset_dir,
                base_tags_dir=args.tags_dir,
                base_tmp_dir=args.tmp_dir,
                base_output_dir=args.output_dir,
                min_area_threshold=args.min_area,
                model_path=args.model
            )
        except Exception as e:
            print(f"Error processing file {file_num:05d}: {e}")
            continue

if __name__ == "__main__":
    main()
