#!/usr/bin/env python3
"""
Tag test image segments by finding nearest neighbors in source embeddings using autofaiss
"""

import os
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import pyarrow.parquet as pq
from collections import Counter

# ============================================================
# STEP 1: BUILD AUTOFAISS INDEX FROM SOURCE EMBEDDINGS
# ============================================================

def build_source_index(embedding_dir, parquet_dir, index_dir):
    """Build faiss index from source embeddings with themes"""
    
    print("="*80)
    print("BUILDING FAISS INDEX FROM SOURCE EMBEDDINGS")
    print("="*80)
    
    # Load all source embeddings
    pkl_files = sorted(Path(embedding_dir).glob("*.pkl"))
    
    all_embeddings = []
    all_labels = []
    all_keys = []  # Track which image each segment belongs to
    
    print(f"Loading {len(pkl_files)} embedding files...")
    for pkl_file in tqdm(pkl_files):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        
        embeddings = data["embeddings"]
        metadata = data["metadata"]
        labels = [item["label"] for item in metadata]
        keys = [item["key"] for item in metadata]
        
        all_embeddings.append(embeddings)
        all_labels.extend(labels)
        all_keys.extend(keys)
    
    all_embeddings = np.vstack(all_embeddings).astype(np.float32)
    
    print(f"\nTotal embeddings: {all_embeddings.shape[0]}")
    print(f"Embedding dim: {all_embeddings.shape[1]}")
    print(f"Unique tags: {len(set(all_labels))}")
    print(f"Unique images: {len(set(all_keys))}")
    
    # Load themes from parquet files
    print("\nLoading themes from parquet files...")
    key_to_theme = {}
    
    parquet_files = sorted(Path(parquet_dir).glob("*.parquet"))
    for pq_file in tqdm(parquet_files):
        try:
            pf = pq.ParquetFile(str(pq_file))
            for batch in pf.iter_batches(batch_size=1000):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    key = row.get('key', row.get('__key__', ''))
                    theme = row.get('theme', '')
                    if key and theme:
                        key_to_theme[key] = theme
        except Exception as e:
            print(f"Error reading {pq_file}: {e}")
    
    print(f"Loaded themes for {len(key_to_theme)} images")
    
    # Map themes to embeddings
    all_themes = []
    for key in all_keys:
        theme = key_to_theme.get(key, '')
        all_themes.append(theme)
    
    # Save embeddings and metadata
    os.makedirs(index_dir, exist_ok=True)
    embeddings_npy = f"{index_dir}/embeddings.npy"
    metadata_pkl = f"{index_dir}/metadata.pkl"
    
    print(f"\nSaving embeddings to {embeddings_npy}...")
    np.save(embeddings_npy, all_embeddings)
    
    print(f"Saving metadata to {metadata_pkl}...")
    metadata = {
        "labels": all_labels,
        "themes": all_themes,
        "keys": all_keys
    }
    with open(metadata_pkl, "wb") as f:
        pickle.dump(metadata, f)
    
    # Build faiss index (using simple FlatIP for reliability)
    print(f"\nBuilding faiss index...")
    dimension = all_embeddings.shape[1]
    
    # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(dimension)
    index.add(all_embeddings)
    
    # Save index
    index_path = f"{index_dir}/knn.index"
    faiss.write_index(index, index_path)
    
    print(f"\nIndex built successfully!")
    print(f"Index type: IndexFlatIP")
    print(f"Total vectors: {index.ntotal}")
    
    return index, metadata


def parse_themes(theme_string):
    """Parse theme string into list of individual themes"""
    if not theme_string:
        return []
    # Split by ' - ' and clean up
    themes = [t.strip() for t in theme_string.split(' - ') if t.strip()]
    return themes


def get_most_common_themes(theme_list, top_k=5):
    """Get most common themes from a list of theme strings"""
    all_themes = []
    for theme_string in theme_list:
        all_themes.extend(parse_themes(theme_string))
    
    if not all_themes:
        return []
    
    theme_counts = Counter(all_themes)
    return [theme for theme, count in theme_counts.most_common(top_k)]


# ============================================================
# STEP 2: EXTRACT TEST IMAGE EMBEDDINGS
# ============================================================

def extract_crop(image_path, bbox):
    """Extract crop from image"""
    try:
        img = Image.open(image_path).convert("RGB")
        x1, y1, x2, y2 = bbox
        return img.crop((int(x1), int(y1), int(x2), int(y2)))
    except Exception as e:
        print(f"Crop error {image_path}: {e}")
        return None


def compute_test_embeddings(seg_json_path, images_dir, model, processor, device, batch_size=32):
    """Compute embeddings for test image segments"""
    
    with open(seg_json_path, "r") as f:
        seg_data = json.load(f)
    
    # Extract folder name from JSON path
    folder_name = Path(seg_json_path).stem
    image_folder = Path(images_dir) / folder_name
    
    print(f"  Loading {len(seg_data)} images from {image_folder}")
    
    # First pass: load all images into memory (faster than repeated disk access)
    image_cache = {}
    for item in tqdm(seg_data, desc="  Loading images", leave=False):
        key = item["key"]
        
        if key in image_cache:
            continue
        
        # Find image file
        image_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            candidate = image_folder / f"{key}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            continue
        
        try:
            img = Image.open(image_path).convert("RGB")
            image_cache[key] = img
        except Exception as e:
            print(f"  Error loading {image_path}: {e}")
            continue
    
    print(f"  Loaded {len(image_cache)} images into cache")
    
    # Second pass: extract crops
    crops = []
    metadata = []
    
    for item in tqdm(seg_data, desc="  Extracting crops", leave=False):
        key = item["key"]
        
        if key not in image_cache:
            continue
        
        img = image_cache[key]
        
        for box in item["boxes"]:
            try:
                x1, y1, x2, y2 = box["bbox"]
                crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
                
                crops.append(crop)
                metadata.append({
                    "key": key,
                    "bbox": box["bbox"],
                    "original_label": box.get("label", ""),
                    "area": box.get("area", 0),
                })
            except Exception as e:
                print(f"  Crop error for {key}: {e}")
                continue
    
    if len(crops) == 0:
        return np.array([]), []
    
    print(f"  Extracted {len(crops)} crops")
    
    # Compute embeddings in batches
    embeddings_all = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(crops), batch_size), desc="  Computing embeddings", leave=False):
            batch = crops[i:i + batch_size]
            
            inputs = processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            
            vision_out = model.vision_model(pixel_values=pixel_values)
            pooled = vision_out.pooler_output
            feats = model.visual_projection(pooled)
            feats = F.normalize(feats, dim=-1)
            
            embeddings_all.append(feats.cpu().numpy())
    
    embeddings_all = np.vstack(embeddings_all).astype(np.float32)
    
    return embeddings_all, metadata


# ============================================================
# STEP 3: TAG TEST IMAGES USING KNN
# ============================================================

def tag_test_images(index, source_metadata, test_embeddings, test_metadata, k=5):
    """Tag test segments using KNN on source embeddings with themes"""
    
    if len(test_embeddings) == 0:
        return []
    
    source_labels = source_metadata["labels"]
    source_themes = source_metadata["themes"]
    source_keys = source_metadata["keys"]
    
    # Search for k nearest neighbors
    distances, indices = index.search(test_embeddings, k)
    
    # Assign tags and themes based on neighbors
    results = []
    
    for i, meta in enumerate(test_metadata):
        # Get all k neighbors
        neighbor_indices = indices[i]
        neighbor_distances = distances[i]
        
        # Collect neighbor information
        neighbors = []
        neighbor_themes_list = []
        
        for idx, dist in zip(neighbor_indices, neighbor_distances):
            neighbor_label = source_labels[idx]
            neighbor_theme = source_themes[idx]
            neighbor_key = source_keys[idx]
            
            neighbors.append({
                "tag": neighbor_label,
                "distance": float(dist),
                "source_key": neighbor_key,
                "theme": neighbor_theme
            })
            
            neighbor_themes_list.append(neighbor_theme)
        
        # Get most common tag (majority vote)
        neighbor_labels = [n["tag"] for n in neighbors]
        tag_counts = Counter(neighbor_labels)
        predicted_tag = tag_counts.most_common(1)[0][0]
        
        # Get most common themes
        most_common_themes = get_most_common_themes(neighbor_themes_list, top_k=5)
        
        results.append({
            "key": meta["key"],
            "bbox": meta["bbox"],
            "area": meta["area"],
            "original_label": meta["original_label"],
            "predicted_tag": predicted_tag,
            "neighbors": neighbors,  # All 5 neighbors with their info
            "most_common_themes": most_common_themes  # Top 5 most common themes
        })
    
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_embeddings", type=str, default="./clip_embeddings")
    parser.add_argument("--source_parquet", type=str, default="/data/corerndimage/image_tagging/Dataset/internet_indian_dataset")
    parser.add_argument("--test_seg_dir", type=str, default="/data/corerndimage/image_tagging/Dataset/segmentation_output")
    parser.add_argument("--test_images_dir", type=str, default="/data/corerndimage/image_tagging/target_dataset")
    parser.add_argument("--output_dir", type=str, default="./test_tags")
    parser.add_argument("--index_dir", type=str, default="./faiss_index")
    parser.add_argument("--rebuild_index", action="store_true", help="Rebuild index from scratch")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--test_folder", type=str, help="Single test folder to process (e.g., 00290)")
    
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Build or load index
    index_path = f"{args.index_dir}/knn.index"
    metadata_path = f"{args.index_dir}/metadata.pkl"
    
    if args.rebuild_index or not os.path.exists(index_path):
        index, source_metadata = build_source_index(
            args.source_embeddings, 
            args.source_parquet,
            args.index_dir
        )
    else:
        print("Loading existing index...")
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            source_metadata = pickle.load(f)
        print(f"Loaded index with {index.ntotal} vectors")
        print(f"Loaded metadata with {len(source_metadata['labels'])} labels")
    
    # Load CLIP model
    print("\nLoading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    
    # Find test segmentation files
    if args.test_folder:
        seg_files = [Path(args.test_seg_dir) / f"{args.test_folder}.json"]
    else:
        seg_files = sorted(Path(args.test_seg_dir).glob("*.json"))
    
    print(f"\nProcessing {len(seg_files)} test files...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for seg_file in seg_files:
        folder_name = seg_file.stem
        output_file = Path(args.output_dir) / f"{folder_name}.json"
        
        if output_file.exists():
            print(f"\nSkipping {folder_name}: output exists")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing {folder_name}")
        print('='*80)
        
        # Compute test embeddings
        test_embeddings, test_metadata = compute_test_embeddings(
            seg_file, args.test_images_dir, model, processor, device, args.batch_size
        )
        
        if len(test_embeddings) == 0:
            print(f"No embeddings for {folder_name}")
            continue
        
        print(f"  Computed {len(test_embeddings)} embeddings")
        
        # Tag using KNN
        print(f"  Finding nearest neighbors...")
        results = tag_test_images(index, source_metadata, test_embeddings, test_metadata, args.k)
        
        # Save results
        print(f"  Saving results...")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"  ✓ Saved to {output_file}")
        print(f"  ✓ Tagged {len(results)} segments")
        
        # Show sample
        if len(results) > 0:
            print(f"\n  Sample result:")
            sample = results[0]
            print(f"    Key: {sample['key']}")
            print(f"    Predicted tag: {sample['predicted_tag']}")
            print(f"    Top themes: {', '.join(sample['most_common_themes'][:3])}")
    
    print("\n" + "="*80)
    print(f"COMPLETED! Processed {len(seg_files)} files")
    print("="*80)


if __name__ == "__main__":
    main()