import json
import io
import tarfile
import pickle
from pathlib import Path
from typing import List
import math

import numpy as np
import ray
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from logzero import logger


def chunk_list(lst: List, size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]
def embed_source(config: dict) -> None:
    """
    Generate CLIP embeddings for source dataset crops.
    
    Args:
        config: Configuration dict with pipeline settings
    """
    
    pipe = config["tool"]["pipeline"]

    source_seg_dir = Path(pipe["source_output"])          # contains *.json
    source_dataset_dir = Path(pipe["source_dataset"])     # contains *.tar
    csv_path = Path(pipe["source_csv"])

    out_dir = Path(pipe.get("source_embeddings", "./source_embeddings"))
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_model_name = pipe["clip_model"]
    num_gpus = pipe["num_gpus"]
    min_area = pipe.get("min_area", 100)
    BATCH_SIZE = 32

    # ============================================================
    # CSV → URL → theme (Match by full URL)
    # ============================================================
    logger.info("="*80)
    logger.info("LOADING CSV FOR THEME MAPPING")
    logger.info("="*80)
    
    df_csv = pd.read_csv(csv_path)
    logger.info(f"CSV loaded: {len(df_csv)} rows")
    logger.info(f"CSV columns: {list(df_csv.columns)}")

    # Build mapping by full URL (not normalized key)
    # Strip whitespace from URLs to avoid matching issues
    df_csv["image_url"] = df_csv["image_url"].astype(str).str.strip()
    url_to_theme = dict(zip(df_csv["image_url"], df_csv["theme"]))
    
    logger.info(f"✓ Built {len(url_to_theme)} URL→theme mappings")
    
    # Log sample URLs for debugging
    sample_urls = list(url_to_theme.keys())[:5]
    logger.info("Sample CSV URLs:")
    for url in sample_urls:
        logger.info(f"  {url}")

    # ============================================================
    # Ray Actor for CLIP Embedding
    # ============================================================
    @ray.remote(num_gpus=1)
    class CLIPEmbedActor:
        def __init__(self):
            import clip
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(
                clip_model_name, device=self.device
            )
            self.model.eval()
            logger.info(f"CLIPEmbedActor initialized on {self.device}")

        def embed(self, batch):
            """Process a batch of image crops and return embeddings."""
            out_embs = []
            out_meta = []

            for item in batch:
                try:
                    img = Image.open(io.BytesIO(item["image_bytes"])).convert("RGB")
                except Exception as e:
                    logger.debug(f"Failed to load image: {e}")
                    continue

                # Validate and clamp bbox coordinates
                x1, y1, x2, y2 = map(int, item["bbox"])
                x1 = max(0, min(x1, img.width - 1))
                y1 = max(0, min(y1, img.height - 1))
                x2 = max(0, min(x2, img.width))
                y2 = max(0, min(y2, img.height))

                if x2 <= x1 or y2 <= y1:
                    logger.debug(f"Invalid bbox: {item['bbox']}")
                    continue

                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    logger.debug(f"Bbox too small: {area} < {min_area}")
                    continue

                try:
                    # Crop and compute embedding
                    crop = img.crop((x1, y1, x2, y2))
                    crop_t = self.preprocess(crop).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        emb = self.model.encode_image(crop_t)
                        emb = emb / emb.norm(dim=-1, keepdim=True)

                    out_embs.append(emb.cpu().numpy()[0])
                    out_meta.append(item["meta"])

                except Exception as e:
                    logger.debug(f"Failed to compute embedding: {e}")
                    continue

            return out_embs, out_meta

    # ============================================================
    # Initialize Ray actors
    # ============================================================
    actors = []
    seg_files = sorted(source_seg_dir.glob("*.json"))
    logger.info(f"Found {len(seg_files)} source segmentation shards")

    # ============================================================
    # Main loop: Process each shard
    # ============================================================
    logger.info("="*80)
    logger.info("PROCESSING SOURCE SHARDS")
    logger.info("="*80)
    
    for seg_json in tqdm(seg_files, desc="Embedding source shards"):
        shard_id = seg_json.stem
        out_file = out_dir / f"{shard_id}.pkl"

        if out_file.exists():
            logger.info(f"[SKIP] {shard_id} (output already exists)")
            continue

        tar_path = source_dataset_dir / f"{shard_id}.tar"
        if not tar_path.exists():
            logger.warning(f"[MISS] TAR not found: {tar_path}")
            continue

        # Load segmentation JSON
        with open(seg_json) as f:
            records = json.load(f)

        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING SHARD: {shard_id}")
        logger.info(f"{'='*80}")
        logger.info(f"Loaded {len(records)} records from JSON")

        # Log sample record to see structure
        if records:
            sample = records[0]
            logger.info(f"Sample record keys: {list(sample.keys())}")
            logger.info(f"Sample record: key={sample.get('key')}, url={sample.get('url')[:80] if sample.get('url') else 'N/A'}...")

        batch_data = []
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

        # Open TAR file and build member index
        with tarfile.open(tar_path, "r") as tar:
            # Build member index: stem → TarInfo
            members = {}
            for m in tar.getmembers():
                if not m.isfile():
                    continue
                if Path(m.name).suffix.lower() not in IMAGE_EXTS:
                    continue
                members[Path(m.name).stem] = m

            logger.info(f"Found {len(members)} images in TAR")

            matched_count = 0
            no_theme_count = 0
            no_boxes_count = 0
            missing_url_count = 0
            missing_key_count = 0
            
            # Log first few URLs from JSON for debugging
            sample_json_urls = []
            
            # Process each record
            for idx, r in enumerate(records):
                key = r.get("key")
                url = r.get("url", "").strip()  # Get URL from JSON and strip whitespace
                boxes = r.get("boxes", [])
                
                # Collect sample URLs for debugging
                if idx < 5:
                    sample_json_urls.append(url)
                
                if not key:
                    missing_key_count += 1
                    continue
                
                if not url:
                    missing_url_count += 1
                    continue
                    
                if key not in members:
                    continue
                
                if not boxes:
                    no_boxes_count += 1
                    continue
                
                # Match by URL, not key
                theme = url_to_theme.get(url, "")
                if not theme:
                    no_theme_count += 1
                    # For first few misses, log the URL to debug
                    if no_theme_count <= 3:
                        logger.debug(f"No theme for URL: {url}")
                    continue  # Skip if no theme found
                
                matched_count += 1
                
                # Extract image bytes from TAR
                fobj = tar.extractfile(members[key])
                if not fobj:
                    continue

                img_bytes = fobj.read()
                if not img_bytes:
                    continue

                # Process all bounding boxes for this image
                for b in boxes:
                    bbox = b.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    
                    # Skip invalid coordinates
                    if any(math.isnan(c) or math.isinf(c) for c in bbox):
                        continue

                    batch_data.append({
                        "image_bytes": img_bytes,
                        "bbox": bbox,
                        "meta": {
                            "key": key,
                            "bbox": bbox,
                            "theme": theme,
                            "url": url,
                        }
                    })
            
            # Log sample URLs from JSON
            logger.info("Sample JSON URLs:")
            for url in sample_json_urls[:3]:
                logger.info(f"  {url}")
            
            logger.info(f"\nProcessing Stats:")
            logger.info(f"  Total records: {len(records)}")
            logger.info(f"  Missing key: {missing_key_count}")
            logger.info(f"  Missing URL: {missing_url_count}")
            logger.info(f"  No boxes: {no_boxes_count}")
            logger.info(f"  No theme match: {no_theme_count}")
            logger.info(f"  ✓ Matched with theme: {matched_count}")
            logger.info(f"  ✓ Total crops created: {len(batch_data)}")

        if not batch_data:
            logger.warning(f"[EMPTY] {shard_id} - no valid crops to process")
            continue

        # Initialize actors on first use
        if not actors:
            logger.info(f"Initializing {num_gpus} CLIP actors...")
            actors = [CLIPEmbedActor.remote() for _ in range(num_gpus)]

        # Distribute batches across actors
        all_embeddings = []
        all_metadata = []

        futures = []
        for i in range(0, len(batch_data), BATCH_SIZE):
            actor = actors[(i // BATCH_SIZE) % len(actors)]
            batch = batch_data[i:i + BATCH_SIZE]
            futures.append(actor.embed.remote(batch))

        # Collect results
        logger.info(f"[{shard_id}] Processing {len(futures)} batches...")
        for fut in tqdm(futures, desc=f"  {shard_id}", leave=False):
            embs, metas = ray.get(fut)
            all_embeddings.extend(embs)
            all_metadata.extend(metas)

        if not all_embeddings:
            logger.warning(f"[NO OUTPUT] {shard_id} - no embeddings generated")
            continue

        # Save embeddings and metadata
        with open(out_file, "wb") as f:
            pickle.dump(
                {
                    "embeddings": np.asarray(all_embeddings, dtype="float32"),
                    "metadata": all_metadata,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        logger.info(
            f"[DONE] {shard_id}: "
            f"{len(all_embeddings)} embeddings saved to {out_file}"
        )

    logger.info("="*80)
    logger.info("Source embeddings generation complete!")
    logger.info("="*80)

def embed_target(folder: str, config: dict) -> None:
    pipe = config["tool"]["pipeline"]

    target_root = Path(pipe["target_output"]) / folder
    crops_dir = target_root / "segmentation" / "crops"
    embed_out_dir = target_root / "embeddings"

    embed_out_dir.mkdir(parents=True, exist_ok=True)

    clip_model_name = pipe["clip_model"]
    num_gpus = pipe["num_gpus"]
    BATCH_SIZE = 32

    crop_files = sorted(crops_dir.glob("*.jpg"))
    if not crop_files:
        logger.warning(f"No crops found in {crops_dir}")
        return

    @ray.remote(num_gpus=1)
    class CLIPActor:
        def __init__(self):
            import clip
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(
                clip_model_name, device=self.device
            )
            self.model.eval()

        def embed_batch(self, paths: List[str]):
            out = []
            for p in paths:
                try:
                    img = Image.open(p).convert("RGB")
                    x = self.preprocess(img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        emb = self.model.encode_image(x)
                        emb = emb / emb.norm(dim=-1, keepdim=True)

                    out.append({
                        "crop_path": str(p),
                        "embedding": emb.cpu().numpy()[0],
                    })
                except Exception:
                    continue
            return out

    if not ray.is_initialized():
        ray.init()

    actors = [CLIPActor.remote() for _ in range(num_gpus)]

    futures = []
    for i in range(0, len(crop_files), BATCH_SIZE):
        actor = actors[(i // BATCH_SIZE) % len(actors)]
        batch = [str(p) for p in crop_files[i:i + BATCH_SIZE]]
        futures.append(actor.embed_batch.remote(batch))

    results = []
    for fut in tqdm(futures, desc=f"Embedding target {folder}"):
        results.extend(ray.get(fut))

    if not results:
        logger.warning("No target embeddings produced")
        return

    df = pd.DataFrame(results)
    out_file = embed_out_dir / "embeddings.parquet"
    df.to_parquet(out_file, index=False)

    logger.info(f"[DONE] Saved {len(df)} target embeddings → {out_file}")