import os
import io
import json
import tarfile
from pathlib import Path
from typing import List
from tqdm import tqdm
import ray
import torch
import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError
from ultralytics.models.sam import SAM3SemanticPredictor

# ============================================================
# Utilities
# ============================================================

def chunk_list(lst: List[str], size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def write_jsonl(path: Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def finalize_json(tmp_jsonl: Path, final_json: Path):
    with open(final_json, "w") as out:
        out.write("[\n")
        with open(tmp_jsonl) as f:
            for line in f:
                out.write(line.rstrip())
                out.write(",\n")
        out.seek(out.tell() - 2)
        out.write("\n]\n")
    tmp_jsonl.unlink()


def validate_and_clip_bbox(x1, y1, x2, y2, img_w, img_h):
    x1 = max(0.0, min(x1, img_w))
    y1 = max(0.0, min(y1, img_h))
    x2 = max(0.0, min(x2, img_w))
    y2 = max(0.0, min(y2, img_h))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


# ============================================================
# SOURCE SEGMENTATION
# ============================================================
def segment_source(config: dict) -> None:

    pipe = config["tool"]["pipeline"]

    dataset_dir = Path(pipe["source_dataset"])
    tags_dir = Path(pipe["source_tags"])
    output_dir = Path(pipe["source_output"])

    sam_model = pipe["sam_model"]
    min_area = pipe["min_area"]
    num_gpus = pipe["num_gpus"]

    start = pipe["start_shard"]
    end = pipe["end_shard"]

    TAG_CHUNK = 16
    MAX_INFLIGHT = num_gpus * 2

    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Ray Actor
    # ============================================================

    @ray.remote(num_gpus=1)
    class SAMOnlyActor:
        def __init__(self):
            print(
                f"[SAMOnlyActor INIT] PID={os.getpid()} | "
                f"Ray GPU IDs={ray.get_gpu_ids()} | "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )

            gpu_id = ray.get_gpu_ids()[0] if ray.get_gpu_ids() else None
            device_str = str(gpu_id) if gpu_id is not None else ""

            self.predictor = SAM3SemanticPredictor(
                overrides=dict(
                    task="segment",
                    mode="predict",
                    model=sam_model,
                    device=device_str,
                    half=True,
                    save=False,
                    imgsz=644,
                    verbose=False,
                )
            )

        def segment(self, image_bytes: bytes, tags: list):
            if not tags or not image_bytes:
                return []

            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError):
                return []

            img_w, img_h = img.size
            self.predictor.set_image(img)

            out = []

            with torch.no_grad():
                for tag_chunk in chunk_list(tags, TAG_CHUNK):
                    results = self.predictor(text=tag_chunk)
                    if not results or not hasattr(results[0], "boxes"):
                        continue

                    boxes = results[0].boxes.xyxy.cpu().numpy()

                    for i, (x1, y1, x2, y2) in enumerate(boxes):
                        clipped = validate_and_clip_bbox(
                            x1, y1, x2, y2, img_w, img_h
                        )
                        if clipped is None:
                            continue

                        x1, y1, x2, y2 = clipped

                        # IMPORTANT: cast to float to avoid float32 overflow
                        w = float(x2) - float(x1)
                        h = float(y2) - float(y1)
                        area = w * h

                        if area < min_area:
                            continue

                        out.append({
                            "bbox": [x1, y1, x2, y2],
                            "label": tag_chunk[i % len(tag_chunk)],
                            "area": area,
                        })

            torch.cuda.empty_cache()
            return out

    # ============================================================
    # Actor Pool
    # ============================================================

    actors = [SAMOnlyActor.remote() for _ in range(num_gpus)]

    # ============================================================
    # Shard Loop (THIS is where tqdm belongs)
    # ============================================================

    for shard in tqdm(range(start, end), desc="Source shards"):
        shard_id = f"{shard:05d}"

        parquet_path = dataset_dir / f"{shard_id}.parquet"
        tar_path = dataset_dir / f"{shard_id}.tar"
        tag_path = tags_dir / f"{shard_id}.json"

        final_json = output_dir / f"{shard_id}.json"
        tmp_jsonl = output_dir / f"{shard_id}.tmp.jsonl"

        if final_json.exists():
            continue

        with open(tag_path) as f:
            tag_map = {x["key"]: x["tags"] for x in json.load(f)}

        meta = {}
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=2000):
            df = batch.to_pandas()
            for _, r in df.iterrows():
                k = r.get("__key__", r.get("key"))
                meta[k] = {
                    "url": r.get("url", ""),
                    "width": r.get("width", 0),
                    "height": r.get("height", 0),
                }

        inflight = []

        with tarfile.open(tar_path, "r") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]

            for idx, m in enumerate(
                tqdm(members, desc=f"Shard {shard_id}", leave=False)
            ):
                key = Path(m.name).stem
                tags = tag_map.get(key, [])

                fobj = tar.extractfile(m)
                if not fobj or not tags:
                    write_jsonl(
                        tmp_jsonl,
                        {"key": key, **meta.get(key, {}), "boxes": []},
                    )
                    continue

                img_bytes = fobj.read()
                actor = actors[idx % len(actors)]
                inflight.append((key, actor.segment.remote(img_bytes, tags)))

                if len(inflight) >= MAX_INFLIGHT:
                    k, fut = inflight.pop(0)
                    boxes = ray.get(fut)
                    write_jsonl(
                        tmp_jsonl,
                        {"key": k, **meta.get(k, {}), "boxes": boxes},
                    )

        for k, fut in inflight:
            boxes = ray.get(fut)
            write_jsonl(
                tmp_jsonl,
                {"key": k, **meta.get(k, {}), "boxes": boxes},
            )

        finalize_json(tmp_jsonl, final_json)

    print("Source segmentation complete.")

# ============================================================
# TARGET SEGMENTATION
# ============================================================

def segment_target(folder: str, config: dict) -> None:

    pipe = config["tool"]["pipeline"]

    dataset_root = Path(pipe["target_dataset"])
    output_root = Path(pipe["target_output"])

    sam_model = pipe["sam_model"]
    min_area = pipe["min_area"]
    num_gpus = pipe["num_gpus"]

    input_folder = dataset_root / folder
    input_json = dataset_root / f"{folder}.json"

    output_dir = output_root / folder / "segmentation"
    crops_dir = output_dir / "crops"
    tmp_jsonl = output_dir / "results.tmp.jsonl"
    final_json = output_dir / "results.json"

    if final_json.exists():
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    @ray.remote(num_gpus=1)
    class SAMOnlyActor:
        def __init__(self):
            gpu_id = ray.get_gpu_ids()[0] if ray.get_gpu_ids() else None
            device_str = str(gpu_id) if gpu_id is not None else ""

            self.predictor = SAM3SemanticPredictor(
                overrides=dict(
                    task="segment",
                    mode="predict",
                    model=sam_model,
                    device=device_str,
                    half=True,
                    save=False,
                    imgsz=644,
                    verbose=False,
                )
            )

        def segment(self, image_bytes: bytes, tags: list):
            if not tags:
                return []

            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError):
                return []

            img_w, img_h = image.size
            self.predictor.set_image(image)

            out = []
            MAX_TAGS = 12

            with torch.no_grad():
                for chunk in tqdm(chunk_list(tags, MAX_TAGS)):
                    results = self.predictor(text=chunk)
                    if not results or not hasattr(results[0], "boxes"):
                        continue

                    boxes = results[0].boxes.xyxy.cpu().numpy()

                    for i, (x1, y1, x2, y2) in enumerate(boxes):
                        clipped = validate_and_clip_bbox(
                            x1, y1, x2, y2, img_w, img_h
                        )
                        if clipped is None:
                            continue

                        x1, y1, x2, y2 = clipped
                        area = (x2 - x1) * (y2 - y1)

                        if area < min_area or not torch.isfinite(torch.tensor(area)):
                            continue

                        out.append({
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "label": chunk[i % len(chunk)],
                            "area": float(area),
                        })

            torch.cuda.empty_cache()
            return out

    actors = [SAMOnlyActor.remote() for _ in range(num_gpus)]

    with open(input_json) as f:
        data = json.load(f)

    MAX_INFLIGHT = num_gpus * 2
    inflight = []

    for idx, item in enumerate(data):
        key = item["key"]
        tags = item.get("tags", [])

        img_file = input_folder / os.path.basename(
            item.get("image_path", f"{key}.jpg")
        )

        if not img_file.exists() or not tags:
            write_jsonl(tmp_jsonl, {
                "key": key,
                "url": item.get("url", ""),
                "width": item.get("width", 0),
                "height": item.get("height", 0),
                "boxes": [],
            })
            continue

        img_bytes = img_file.read_bytes()
        actor = actors[idx % len(actors)]
        inflight.append((key, img_file, actor.segment.remote(img_bytes, tags), item))

        if len(inflight) >= MAX_INFLIGHT:
            _flush_target(inflight.pop(0), crops_dir, tmp_jsonl)

    for item in inflight:
        _flush_target(item, crops_dir, tmp_jsonl)

    finalize_json(tmp_jsonl, final_json)
    print(f"[DONE] Target segmentation complete for {folder}")


def _flush_target(item, crops_dir: Path, tmp_jsonl: Path):
    key, img_file, fut, meta = item
    boxes = ray.get(fut)
    image = Image.open(img_file).convert("RGB")

    saved = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b["bbox"])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.width, x2)
        y2 = min(image.height, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image.crop((x1, y1, x2, y2))
        crop_path = crops_dir / f"{key}_{i}.jpg"
        crop.save(crop_path)

        saved.append({
            "bbox": [x1, y1, x2, y2],
            "label": b["label"],
            "area": b["area"],
            "crop_path": str(crop_path),
        })

    write_jsonl(tmp_jsonl, {
        "key": key,
        "url": meta.get("url", ""),
        "width": meta.get("width", 0),
        "height": meta.get("height", 0),
        "boxes": saved,
    })
