import json
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import faiss
import ray
from tqdm import tqdm
from logzero import logger


# ============================================================
# Ray task
# ============================================================
@ray.remote
def load_pickle_shard(pickle_path: str):
    """Load embeddings and metadata from a pickle file."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    
    embeddings = data["embeddings"]  # Already np.array[float32]
    metadata = data["metadata"]  # List of dicts with theme, url, key, bbox
    
    if len(embeddings) == 0:
        return None
    
    # Ensure embeddings are float32
    embeddings = embeddings.astype("float32")
    
    return embeddings, metadata


# ============================================================
# Build FAISS index
# ============================================================
def build_faiss_index(config: dict) -> None:
    pipe = config["tool"]["pipeline"]

    # Use source_embeddings directory (where embed_source saves pickle files)
    embed_dir = Path(pipe.get("source_embeddings", "./source_embeddings"))
    faiss_dir = Path(pipe.get("faiss_index_dir", "./faiss_index"))
    faiss_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Find pickle files
    # --------------------------------------------------
    pickle_files = sorted(embed_dir.glob("*.pkl"))
    if not pickle_files:
        raise RuntimeError(f"No pickle files found in {embed_dir}")

    logger.info(f"Found {len(pickle_files)} pickle shards in {embed_dir}")

    # --------------------------------------------------
    # Ray-parallel loading
    # --------------------------------------------------
    # Note: Ray should be initialized by main.py with config settings
    if not ray.is_initialized():
        logger.warning("Ray not initialized, initializing with defaults...")
        ray.init()

    futures = [
        load_pickle_shard.remote(str(pkl))
        for pkl in pickle_files
    ]

    all_embs = []
    id_to_meta = []

    for fut in tqdm(futures, desc="Loading embedding shards"):
        res = ray.get(fut)
        if res is None:
            continue
        embs, metas = res
        all_embs.append(embs)
        # Metadata already contains theme, url, key, bbox from embed_source
        id_to_meta.extend(metas)

    if not all_embs:
        raise RuntimeError("No embeddings loaded from pickle files")

    X = np.vstack(all_embs)
    dim = X.shape[1]

    logger.info(f"Total vectors: {X.shape[0]}, dim={dim}")
    logger.info(f"Sample meta: {id_to_meta[:2] if id_to_meta else 'None'}")

    # --------------------------------------------------
    # FAISS
    # --------------------------------------------------
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer,
        dim,
        pipe.get("faiss_nlist", 4096),
        pipe.get("faiss_pq_m", 64),
        8,
        faiss.METRIC_INNER_PRODUCT,
    )

    logger.info("Training FAISS index...")
    index.train(X)
    index.add(X)

    faiss.write_index(index, str(faiss_dir / "index.faiss"))

    with open(faiss_dir / "id_to_meta.json", "w") as f:
        json.dump(id_to_meta, f, indent=2)

    logger.info(f"[DONE] FAISS index built successfully")
    logger.info(f"  - Index: {faiss_dir / 'index.faiss'}")
    logger.info(f"  - Metadata: {faiss_dir / 'id_to_meta.json'}")
    logger.info(f"  - Total vectors: {len(id_to_meta)}")


# ============================================================
# Tag Target Embeddings
# ============================================================

def _load_source_labels(source_output_dir: Path):
    """Load labels from source segmentation JSON files.
    
    Returns a dict mapping (key, bbox_tuple) -> label
    where bbox_tuple is (x1, y1, x2, y2) rounded to ints for matching.
    """
    logger.info("Loading source segmentation labels...")
    source_seg_files = sorted(source_output_dir.glob("*.json"))
    
    if not source_seg_files:
        logger.warning(f"No source segmentation files found in {source_output_dir}")
        return {}
    
    key_bbox_to_label = {}
    
    for seg_file in tqdm(source_seg_files, desc="Loading source labels", leave=False):
        try:
            with open(seg_file) as f:
                records = json.load(f)
            
            for record in records:
                key = record.get("key")
                if not key:
                    continue
                
                boxes = record.get("boxes", [])
                for box in boxes:
                    bbox = box.get("bbox")
                    label = box.get("label")
                    
                    if not bbox or len(bbox) != 4 or not label:
                        continue
                    
                    # Round bbox to ints for matching
                    x1, y1, x2, y2 = map(int, bbox)
                    bbox_tuple = (x1, y1, x2, y2)
                    key_bbox_to_label[(key, bbox_tuple)] = label
                    
        except Exception as e:
            logger.warning(f"Error loading {seg_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(key_bbox_to_label)} source label mappings")
    return key_bbox_to_label


def _match_bbox(bbox1, bbox2, tolerance=2):
    """Check if two bboxes match within tolerance."""
    if len(bbox1) != 4 or len(bbox2) != 4:
        return False
    
    x1_1, y1_1, x2_1, y2_1 = map(int, bbox1)
    x1_2, y1_2, x2_2, y2_2 = map(int, bbox2)
    
    return (
        abs(x1_1 - x1_2) <= tolerance and
        abs(y1_1 - y1_2) <= tolerance and
        abs(x2_1 - x2_2) <= tolerance and
        abs(y2_1 - y2_2) <= tolerance
    )


def _parse_theme_tags(theme_string: str) -> list:
    """Parse theme string into list of individual tags.
    
    Args:
        theme_string: Theme string with tags separated by ' - '
    
    Returns:
        List of individual tags (stripped and non-empty)
    """
    if not theme_string:
        return []
    return [t.strip() for t in theme_string.split(' - ') if t.strip()]


def tag_target(folder: str, config: dict) -> None:
    """Tag target embeddings using FAISS KNN search on source embeddings.
    
    Args:
        folder: Target folder name (e.g., "00290")
        config: Configuration dict
    """
    pipe = config["tool"]["pipeline"]
    
    # Read all paths from config
    faiss_dir = Path(pipe.get("faiss_index_dir", "./faiss_index"))
    source_output_dir = Path(pipe.get("source_output", "./source_processed"))
    target_output_dir = Path(pipe.get("target_output", "./target_processed"))
    csv_path = Path(pipe.get("source_csv"))
    faiss_k = pipe.get("faiss_k", 10)
    
    # Target paths
    target_root = target_output_dir / folder
    target_embeddings_path = target_root / "embeddings" / "embeddings.parquet"
    target_seg_path = target_root / "segmentation" / "results.json"
    tagging_output_dir = target_root / "tagging"
    tagging_output_path = tagging_output_dir / "results.json"
    
    logger.info("="*80)
    logger.info(f"TAGGING TARGET: {folder}")
    logger.info("="*80)
    logger.info(f"FAISS index dir: {faiss_dir}")
    logger.info(f"Source output dir: {source_output_dir}")
    logger.info(f"Target embeddings: {target_embeddings_path}")
    logger.info(f"Target segmentation: {target_seg_path}")
    logger.info(f"K nearest neighbors: {faiss_k}")
    
    # Check required files exist
    if not target_embeddings_path.exists():
        raise FileNotFoundError(f"Target embeddings not found: {target_embeddings_path}")
    
    if not target_seg_path.exists():
        raise FileNotFoundError(f"Target segmentation not found: {target_seg_path}")
    
    faiss_index_path = faiss_dir / "index.faiss"
    faiss_metadata_path = faiss_dir / "id_to_meta.json"
    
    if not faiss_index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_index_path}")
    
    if not faiss_metadata_path.exists():
        raise FileNotFoundError(f"FAISS metadata not found: {faiss_metadata_path}")
    
    # --------------------------------------------------
    # Load CSV for theme mapping
    # --------------------------------------------------
    logger.info("\nLoading CSV for theme mapping...")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df_csv = pd.read_csv(csv_path)
    logger.info(f"CSV loaded: {len(df_csv)} rows")
    logger.info(f"CSV columns: {list(df_csv.columns)}")
    
    # Build mapping by full URL (not normalized key)
    # Strip whitespace from URLs to avoid matching issues
    df_csv["image_url"] = df_csv["image_url"].astype(str).str.strip()
    url_to_theme = dict(zip(df_csv["image_url"], df_csv["theme"]))
    
    logger.info(f"✓ Built {len(url_to_theme)} URL→theme mappings")
    
    # --------------------------------------------------
    # Load FAISS index and metadata
    # --------------------------------------------------
    logger.info("\nLoading FAISS index...")
    index = faiss.read_index(str(faiss_index_path))
    logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
    
    logger.info("Loading FAISS metadata...")
    with open(faiss_metadata_path) as f:
        source_metadata = json.load(f)
    logger.info(f"Loaded {len(source_metadata)} metadata entries")
    
    # --------------------------------------------------
    # Load source labels from segmentation JSONs
    # --------------------------------------------------
    source_labels_map = _load_source_labels(source_output_dir)
    
    # --------------------------------------------------
    # Load target embeddings
    # --------------------------------------------------
    logger.info("\nLoading target embeddings...")
    df_embeddings = pd.read_parquet(target_embeddings_path)
    logger.info(f"Loaded {len(df_embeddings)} target embeddings")
    
    # Convert embeddings to numpy array
    embeddings_list = [np.array(emb, dtype=np.float32) for emb in df_embeddings["embedding"]]
    target_embeddings = np.vstack(embeddings_list).astype(np.float32)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(target_embeddings)
    
    # --------------------------------------------------
    # Load target segmentation results
    # --------------------------------------------------
    logger.info("\nLoading target segmentation...")
    with open(target_seg_path) as f:
        target_seg_data = json.load(f)
    logger.info(f"Loaded {len(target_seg_data)} target images")
    
    # Build crop_path -> (record_idx, box_idx, box) mapping
    crop_to_segment = {}
    for record_idx, record in enumerate(target_seg_data):
        boxes = record.get("boxes", [])
        for box_idx, box in enumerate(boxes):
            crop_path = box.get("crop_path")
            if crop_path:
                # Normalize path for matching
                crop_path_normalized = str(Path(crop_path))
                crop_to_segment[crop_path_normalized] = {
                    "record_idx": record_idx,
                    "box_idx": box_idx,
                    "box": box,
                }
    
    logger.info(f"Built mapping for {len(crop_to_segment)} crops")
    
    # Match embeddings to segments
    matched_indices = []
    matched_segment_info = []
    
    for idx, row in df_embeddings.iterrows():
        crop_path = row["crop_path"]
        crop_path_normalized = str(Path(crop_path))
        
        if crop_path_normalized in crop_to_segment:
            matched_indices.append(idx)
            matched_segment_info.append(crop_to_segment[crop_path_normalized])
    
    logger.info(f"Matched {len(matched_indices)} embeddings to segments")
    
    if len(matched_indices) == 0:
        logger.warning("No embeddings matched to segments!")
        return
    
    # Get embeddings for matched indices
    matched_embeddings = target_embeddings[matched_indices]
    
    # --------------------------------------------------
    # Perform KNN search
    # --------------------------------------------------
    logger.info(f"\nPerforming KNN search (k={faiss_k})...")
    distances, indices = index.search(matched_embeddings, faiss_k)
    logger.info(f"Found neighbors for {len(matched_embeddings)} target embeddings")
    
    # --------------------------------------------------
    # Extract labels and assign tags
    # --------------------------------------------------
    logger.info("\nAssigning tags via majority vote...")
    
    tagged_count = 0
    
    for i, (dist_row, idx_row) in enumerate(tqdm(zip(distances, indices), total=len(matched_embeddings), desc="Tagging")):
        segment_info = matched_segment_info[i]
        record_idx = segment_info["record_idx"]
        box_idx = segment_info["box_idx"]
        box = segment_info["box"]
        
        # Get neighbors and their labels
        neighbor_labels = []
        neighbor_info = []
        all_neighbor_csv_themes = []  # Collect CSV themes from ALL neighbors for tag extraction
        
        for neighbor_idx, dist in zip(idx_row, dist_row):
            if neighbor_idx < 0 or neighbor_idx >= len(source_metadata):
                continue
            
            neighbor_meta = source_metadata[neighbor_idx]
            neighbor_key = neighbor_meta.get("key")
            neighbor_bbox = neighbor_meta.get("bbox")
            neighbor_url = neighbor_meta.get("url", "").strip()
            neighbor_theme = neighbor_meta.get("theme", "")
            
            # Look up theme from CSV using URL
            csv_theme = url_to_theme.get(neighbor_url, "")
            if csv_theme:
                all_neighbor_csv_themes.append(csv_theme)
            
            if not neighbor_key or not neighbor_bbox:
                continue
            
            # Try to find label for this neighbor
            neighbor_bbox_tuple = tuple(map(int, neighbor_bbox))
            label = source_labels_map.get((neighbor_key, neighbor_bbox_tuple))
            
            if label:
                neighbor_labels.append(label)
                neighbor_info.append({
                    "label": label,
                    "distance": float(dist),
                    "key": neighbor_key,
                    "bbox": neighbor_bbox,
                    "theme": neighbor_theme,
                })
        
        # Extract all tags from CSV theme strings (separated by ' - ')
        all_tags_set = set()
        for csv_theme_str in all_neighbor_csv_themes:
            if csv_theme_str:
                tags = _parse_theme_tags(csv_theme_str)
                all_tags_set.update(tags)
        
        # Convert set to sorted list for consistent output
        tags = sorted(list(all_tags_set))
        
        # Majority vote for predicted tag
        if neighbor_labels:
            label_counts = Counter(neighbor_labels)
            predicted_tag = label_counts.most_common(1)[0][0]
            tagged_count += 1
            
            # Get top themes from neighbors
            neighbor_themes = [info["theme"] for info in neighbor_info if info.get("theme")]
            theme_strings = [t for t in neighbor_themes if t]
            
            # Parse themes (split by ' - ')
            all_themes = []
            for theme_str in theme_strings:
                if theme_str:
                    themes = [t.strip() for t in theme_str.split(' - ') if t.strip()]
                    all_themes.extend(themes)
            
            top_themes = [theme for theme, count in Counter(all_themes).most_common(5)]
        else:
            predicted_tag = ""
            top_themes = []
            neighbor_info = []
        
        # Update box in place
        target_seg_data[record_idx]["boxes"][box_idx]["predicted_tag"] = predicted_tag
        target_seg_data[record_idx]["boxes"][box_idx]["neighbors"] = neighbor_info[:faiss_k]  # Store up to k neighbors
        target_seg_data[record_idx]["boxes"][box_idx]["top_themes"] = top_themes
        target_seg_data[record_idx]["boxes"][box_idx]["tags"] = tags  # Add tags from source themes
    
    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    tagging_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving tagged results to {tagging_output_path}...")
    with open(tagging_output_path, "w") as f:
        json.dump(target_seg_data, f, indent=2)
    
    logger.info(f"[DONE] Tagged {tagged_count} segments in {len(target_seg_data)} images")
    logger.info(f"Output saved to: {tagging_output_path}")
