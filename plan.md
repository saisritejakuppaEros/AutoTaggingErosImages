# Minimal Ray Pipeline - File Structure

```
AutoTaggingErosImages/
├── main.py                          # Single entry point - all modes
├── pyproject.toml                   # Config + dependencies
├── scripts/
│   ├── segmentation.py              # SAM segmentation (source + target)
│   ├── embeddings.py                # CLIP embeddings (source + target)
│   └── tagging.py                   # FAISS build + inference
└── utils.py                         # All helpers + Ray actors
```

---

## File Contents

### 1. `main.py`
**Single CLI entry point for all operations**

**Modes:**
- `segment-source` - Segment webdataset (done once)
- `segment-target` - Segment target folder (run per folder)
- `embed-source` - CLIP embeddings from source (done once)
- `embed-target` - CLIP embeddings from target (run per folder)
- `build-index` - Build FAISS index (done once after source embedding)
- `tag-target` - FAISS inference on target (run per folder)

**Usage:**
```bash
# Source pipeline (run once)
python main.py segment-source --config config.toml
python main.py embed-source --config config.toml
python main.py build-index --config config.toml

# Target pipeline (run per folder)
python main.py segment-target --folder 00290 --config config.toml
python main.py embed-target --folder 00290 --config config.toml
python main.py tag-target --folder 00290 --config config.toml

# Or combined target pipeline
python main.py process-target --folder 00290 --config config.toml
```

---

### 2. `pyproject.toml`
**All configuration in one place**

```toml
[project]
name = "eros-autotagging"
version = "2.0.0"
dependencies = [
    "ray[data]",
    "torch",
    "transformers",
    "ultralytics",
    "faiss-gpu",
    "pillow",
    "numpy",
    "pyarrow"
]

[tool.pipeline]
# Hardware
num_gpus = 8
num_cpus = 240
object_store_gb = 100

# Models
sam_model = "sam3.pt"
clip_model = "ViT-B/32"
min_area = 150

# Batching (H200 optimized)
batch_images = 24           # Segmentation batch
batch_clips = 192           # CLIP batch
batch_faiss = 20000         # FAISS search batch

# Paths - Source (run once)
source_dataset = "/data/corerndimage/image_tagging/Dataset/internet_indian_dataset"
source_tags = "/data/corerndimage/image_tagging/Dataset/vlm_tags"
source_output = "/data/corerndimage/image_tagging/Dataset/source_processed"

# Paths - Target (per folder)
target_dataset = "/data/corerndimage/image_tagging/target_dataset"
target_output = "/data/corerndimage/image_tagging/Dataset/target_processed"

# FAISS
faiss_index_dir = "/data/corerndimage/image_tagging/faiss_index"
faiss_k = 10

# Shards (source only)
start_shard = 0
end_shard = 100
```

---

### 3. `scripts/segmentation.py`

**Functions:**

```python
def segment_source(config: dict) -> None:
    """
    Segments webdataset tar files using SAM
    - Reads tar + parquet shards (streaming, no extraction)
    - Ray map_batches with UnifiedActor (SAM+CLIP co-located)
    - Crops images in-memory, computes embeddings inline
    - NO crops saved to disk
    - Only saves embeddings + metadata with bbox coordinates
    """

def segment_target(folder: str, config: dict) -> None:
    """
    Segments single target folder using SAM
    - Reads plain images from target_dataset/{folder}/
    - Ray map_batches with SAMOnlyActor
    - Saves crops + metadata to target_output/{folder}/segmentation/
    - (Target saves crops since folders are small, ~1k images)
    """
```

**Implementation:**
- Source: UnifiedActor does SAM → crop in memory → CLIP → discard crop → save embedding + bbox metadata
- Target: Saves actual crops (small scale, needed for re-tagging experiments)

---

### 4. `scripts/embeddings.py`

**Functions:**

```python
def embed_source(config: dict) -> None:
    """
    NOT USED FOR SOURCE - embeddings computed inline during segmentation
    Source pipeline: segment_source() does SAM+CLIP together
    """

def embed_target(folder: str, config: dict) -> None:
    """
    Computes CLIP embeddings for target segmented crops
    - Reads saved crops from target_output/{folder}/segmentation/crops/
    - Ray map_batches with CLIPActor
    - Saves embeddings to target_output/{folder}/embeddings/
    """
```

**Implementation:**
- Source: Embeddings already computed by UnifiedActor, skip this script
- Target: Separate stage for flexibility (can re-embed without re-segmenting)

---

### 5. `scripts/tagging.py`

**Functions:**

```python
def build_faiss_index(config: dict) -> None:
    """
    Builds FAISS index from source embeddings
    - Loads all embeddings from source_output/embeddings/
    - Trains FAISS index (IVF + PQ)
    - Saves to faiss_index_dir/
      - index.faiss
      - id_to_tags.json (mapping embedding_id -> VLM tags)
    """

def tag_target(folder: str, config: dict) -> None:
    """
    Tags target folder using FAISS similarity search
    - Loads FAISS index + id_to_tags mapping
    - Loads target embeddings from target_output/{folder}/embeddings/
    - Ray map_batches with FAISSActor (GPU search)
    - Aggregates tags per image (vote from all crops)
    - Saves to target_output/{folder}/tags.json
    """
```

**Implementation:**
- FAISS index loaded on all 8 GPUs (replicated)
- Batch search: 20k queries per GPU per batch
- Tag voting: top-k neighbors → most common tags

---

### 6. `utils.py`

**Ray Actors:**

```python
@ray.remote(num_gpus=1, num_cpus=30)
class UnifiedActor:
    """SAM + CLIP co-located - for source pipeline (100M images)"""
    __init__(gpu_id, sam_path, clip_name, min_area)
    process_batch(images, keys, tags) -> embeddings + metadata
    # Process: Load image → SAM segment → crop in-memory → CLIP embed → discard crop
    # Output: Only embeddings + metadata (key, bbox, tags) - NO crop images saved

@ray.remote(num_gpus=1, num_cpus=30)
class SAMOnlyActor:
    """SAM only - for target segmentation"""
    __init__(gpu_id, sam_path, min_area)
    process_batch(images, keys) -> crops + metadata
    # Saves actual crop images (target is small scale ~1k images)

@ray.remote(num_gpus=1, num_cpus=25)
class CLIPActor:
    """CLIP only - for target embedding (reads saved crops)"""
    __init__(gpu_id, clip_name)
    process_batch(crops, metadata) -> embeddings

@ray.remote(num_gpus=1, num_cpus=15)
class FAISSActor:
    """FAISS GPU search - for inference"""
    __init__(gpu_id, index_path, id_to_tags)
    search_batch(embeddings, k) -> tags
```

**Helper Functions:**

```python
def init_ray(config) -> None:
    """Initialize Ray cluster"""

def read_webdataset(dataset_dir, tags_dir, start, end) -> ray.data.Dataset:
    """
    Stream tar files directly, no disk extraction
    Yields images as np.ndarray in-memory
    """

def read_image_folder(folder_path) -> ray.data.Dataset:
    """Load plain images from folder"""

def save_embeddings_with_metadata(dataset, output_dir) -> None:
    """
    Save embeddings + metadata (NO crop images for source)
    Metadata includes: {key, bbox, area, vlm_tags, crop_id}
    """

def save_crops(dataset, output_dir) -> None:
    """
    Save crops to disk (TARGET only)
    Not used for source (100M images)
    """

def aggregate_tags(crop_tags, min_votes=3) -> List[str]:
    """Vote-based tag aggregation per image"""

def load_config(toml_path) -> dict:
    """Parse pyproject.toml"""
```

**Model Loaders:**

```python
def load_sam_model(model_path, device) -> SAMModel:
    """Load SAM with proper device placement"""

def load_clip_model(model_name, device) -> CLIPModel:
    """Load CLIP with proper device placement"""

def build_faiss_index_cpu(embeddings, index_type) -> faiss.Index:
    """Train FAISS index on CPU (multi-threaded)"""

def load_faiss_to_gpu(index_path, gpu_id) -> faiss.GpuIndex:
    """Load index to GPU memory"""
```

---

## Execution Flow

### Source Pipeline (Once)
```bash
# Step 1: Segment + Embed in one pass (UnifiedActor)
python main.py segment-source

# Step 2: Build FAISS index
python main.py build-index

# (No separate embed-source - done inline during segmentation)
```

### Target Pipeline (Per Folder)
```bash
# Option A: Step-by-step
python main.py segment-target --folder 00290
python main.py embed-target --folder 00290
python main.py tag-target --folder 00290

# Option B: All-in-one
python main.py process-target --folder 00290
```

---

## Data Layout

```
source_processed/
└── embeddings/
    ├── embeddings_00000.npy       # [10000, 512]
    └── metadata_00000.json        # [{key, bbox, area, vlm_tags, crop_id}, ...]
    # NO crops/ directory - crops never saved to disk

target_processed/
└── 00290/
    ├── segmentation/
    │   ├── crops/                 # Actual crop images saved
    │   └── metadata.json
    ├── embeddings/
    │   ├── embeddings_00000.npy
    │   └── metadata_00000.json
    └── tags.json                  # Final predictions

faiss_index/
├── index.faiss
└── id_to_tags.json                # {0: ["person", "clothing"], 1: [...]}
```

---

## Key Design Decisions

1. **Source: Streaming pipeline** - Tar files read directly, images cropped in-memory, NO disk writes for crops (100M images)
2. **Source uses UnifiedActor** - SAM+CLIP together, segment → crop → embed → discard, only save embeddings+metadata
3. **Target: Separate stages** - Saves crops (small scale ~1k images), flexibility to re-tag without re-segmenting
4. **FAISS index replicated on all 8 GPUs** - parallel search, no bottleneck
5. **Config in TOML** - single source of truth, easy editing
6. **All Ray init/shutdown handled in main.py** - clean resource management

---

## Performance Expectations (H200)

**Source Pipeline (100M images, run once):**
- Segment + Embed (unified): 30-40 hours (streaming, in-memory crops)
- FAISS Build: 30-45 min
- **Total: ~31-41 hours** (single pass, no intermediate I/O)

**Target Pipeline (1k images per folder):**
- Segmentation: 3-5 min
- Embedding: 1-2 min
- Tagging: 30 sec
- **Total: ~5-7 min per folder**

---

## Files Count: 4 core files
- main.py (CLI router)
- pyproject.toml (config)
- utils.py (all actors + helpers)
- scripts/segmentation.py
- scripts/embeddings.py
- scripts/tagging.py

**Total: 6 files** (vs 17 in detailed version)