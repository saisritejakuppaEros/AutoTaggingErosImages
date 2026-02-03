Nice, this is a solid pipeline 👍
Here’s a **clean, professional README.md** you can drop directly into the repo. I’ve written it in a way that future interns / collaborators won’t bug you every 10 minutes 😄

---

# AutoTagging Pipeline (VLM + Segmentation + CLIP)

This repository contains an end-to-end **image auto-tagging pipeline** using:

* **VLM-based coarse tags**
* **SAM-based segmentation**
* **CLIP embeddings**
* **FAISS indexing** for fast similarity search
* **Zero-shot tagging** on a target dataset

The pipeline is designed to work with **WebDataset-style inputs** as well as **plain image folders**.

---

## 1. Environment Setup

Create and activate the conda environment:

```bash
conda create -n rnd_autotag_v2 python=3.12
conda activate rnd_autotag_v2
```

Install PyTorch (CUDA-compatible version assumed):

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
```

Install remaining dependencies:

```bash
pip install pyarrow numpy pillow tqdm opencv-python pandas
pip install accelerate ultralytics transformers streamlit autofaiss
```

---

## 2. Dataset Structure

### 2.1 Source Dataset (WebDataset-style)

```text
/data/corerndimage/image_tagging/Dataset/internet_indian_dataset/
├── 00000.tar
└── 00000.parquet
```

* `.tar` contains images
* `.parquet` contains metadata (keys, URLs, etc.)

### 2.2 VLM Tags

```text
/data/corerndimage/image_tagging/Dataset/vlm_tags/
└── 00000.json
```

Example VLM tag format:

```json
[
  {
    "key": "000000010",
    "image_path": "/path/to/image.jpg",
    "url": "http://example.com/image.jpg",
    "width": 2048,
    "height": 1536,
    "tags": [
      "people",
      "animals",
      "vehicles",
      "buildings",
      "vegetation",
      "ground",
      "sky"
    ]
  }
]
```

These tags are used to guide **segmentation and downstream embedding extraction**.

---

## 3. Target Dataset (Inference)

Target dataset contains **raw images** and an optional JSON file with object-level metadata.

```text
/data/corerndimage/image_tagging/target_dataset/
├── 00290/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
└── 00290.json
```

---

## 4. Pipeline Overview

```text
VLM Tags
   ↓
Segmentation (SAM)
   ↓
Object Crops
   ↓
CLIP Embeddings
   ↓
FAISS Index
   ↓
Zero-shot Tagging on Target Dataset
```

---

## 5. Running the Pipeline

### 5.1 Segmentation on WebDataset (Source)

Runs segmentation on images inside `.tar` files using VLM tags.

```bash
python vllm_segmentation.py \
  --start 0 \
  --end 10 \
  --gpu 3 \
  --dataset-dir /data/corerndimage/image_tagging/Dataset/internet_indian_dataset \
  --tags-dir /data/corerndimage/image_tagging/Dataset/vlm_tags \
  --output-dir /data/corerndimage/image_tagging/Dataset/segmentation_output \
  --tmp-dir ./tmp/my_seg \
  --min-area 150 \
  --model sam3.pt
```

**Notes**

* `start/end` control shard range
* `min-area` filters tiny segments
* Output contains segmented object masks/crops

---

### 5.2 Segmentation on Target Image Folder

For plain image folders (no `.tar`):

```bash
python images_segmentation.py \
  --folder 00290 \
  --dataset-dir /data/corerndimage/image_tagging/target_dataset \
  --output-dir /data/corerndimage/image_tagging/Dataset/segmentation_output \
  --min-area 100 \
  --model sam3.pt \
  --gpu 0
```

---

### 5.3 Compute CLIP Embeddings (Source Dataset)

Generates CLIP embeddings for segmented objects.

```bash
python compute_clip_embeddings.py \
  --start 0 \
  --end 10 \
  --dataset-dir /data/corerndimage/image_tagging/Dataset/internet_indian_dataset \
  --segmentation-dir /data/corerndimage/image_tagging/Dataset/segmentation \
  --tmp-dir ./tmp/tmp_clip \
  --output_dir ./clip_embeddings
```

Output:

* CLIP embeddings per object
* Stored for FAISS indexing

---

### 5.4 Tagging on Target Dataset (Inference)

Runs similarity-based tagging using FAISS.

```bash
python image_tagging.py \
  --source_embeddings ./clip_embeddings \
  --source_parquet /data/corerndimage/image_tagging/Dataset/internet_indian_dataset \
  --test_seg_dir /data/corerndimage/image_tagging/Dataset/segmentation_output \
  --test_images_dir /data/corerndimage/image_tagging/target_dataset \
  --output_dir /data/corerndimage/image_tagging/test_tags \
  --index_dir /data/corerndimage/image_tagging/faiss_index
```

Output:

* JSON files containing predicted tags per object / image

---

## 6. Outputs

```text
segmentation_output/   → segmented objects
clip_embeddings/       → CLIP feature vectors
faiss_index/           → FAISS search index
test_tags/             → final predicted tags
```

---