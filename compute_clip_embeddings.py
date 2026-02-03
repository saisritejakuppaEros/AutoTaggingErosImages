import os
import json
import argparse
import pickle
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel

# ============================================================
# UTILS
# ============================================================

def extract_crop(image_path: Path, bbox):
    try:
        img = Image.open(image_path).convert("RGB")
        x1, y1, x2, y2 = bbox
        return img.crop((int(x1), int(y1), int(x2), int(y2)))
    except Exception as e:
        print(f"Crop error {image_path}: {e}")
        return None

# ============================================================
# CORE
# ============================================================

def process_file(
    file_num: int,
    model,
    processor,
    accelerator,
    output_dir: str,
    batch_size: int,
    dataset_dir: str,
    segmentation_dir: str,
    tmp_dir_base: str,
):
    file_id = f"{file_num:05d}"

    input_tar = f"{dataset_dir}/{file_id}.tar"
    input_seg = f"{segmentation_dir}/{file_id}.json"
    tmp_dir = f"{tmp_dir_base}/{file_id}"
    output_pkl = f"{output_dir}/{file_id}.pkl"

    if not os.path.exists(input_tar):
        if accelerator.is_main_process:
            print(f"Skipping {file_id}: tar not found")
        return

    if not os.path.exists(input_seg):
        if accelerator.is_main_process:
            print(f"Skipping {file_id}: segmentation not found")
        return

    if os.path.exists(output_pkl):
        if accelerator.is_main_process:
            print(f"Skipping {file_id}: embeddings exist")
        return

    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print(f"Processing {file_id}")
        print("=" * 60)

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load segmentation
    # --------------------------------------------------------

    with open(input_seg, "r") as f:
        seg_data = json.load(f)

    # --------------------------------------------------------
    # Extract images (main process)
    # --------------------------------------------------------

    if accelerator.is_main_process:
        print(f"Extracting images for {file_id}...")
        image_map = {}

        with tarfile.open(input_tar, "r") as tar:
            members = [
                m for m in tar.getmembers()
                if m.name.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            for m in tqdm(members, desc=f"Extracting {file_id}"):
                key = Path(m.name).stem
                ext = Path(m.name).suffix
                out = Path(tmp_dir) / f"{key}{ext}"

                with open(out, "wb") as f:
                    f.write(tar.extractfile(m).read())

                image_map[key] = out

    accelerator.wait_for_everyone()

    # --------------------------------------------------------
    # Rebuild image map (workers)
    # --------------------------------------------------------

    if not accelerator.is_main_process:
        image_map = {
            p.stem: p
            for p in Path(tmp_dir).glob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        }

    # --------------------------------------------------------
    # Collect crops
    # --------------------------------------------------------

    crops = []
    metadata = []

    for item in seg_data:
        key = item["key"]
        if key not in image_map:
            continue

        img_path = image_map[key]

        for box in item["boxes"]:
            crop = extract_crop(img_path, box["bbox"])
            if crop is None:
                continue

            crops.append(crop)
            metadata.append({
                "key": key,
                "bbox": box["bbox"],
                "label": box.get("label", ""),
                "area": box.get("area", 0),
            })

    if len(crops) == 0:
        if accelerator.is_main_process:
            print(f"No crops for {file_id}")
        return

    if accelerator.is_main_process:
        print(f"Computing embeddings for {len(crops)} crops...")

    # --------------------------------------------------------
    # CLIP IMAGE EMBEDDINGS (SAFE PATH)
    # --------------------------------------------------------

    # Handle Accelerate wrapping
    clip = model.module if hasattr(model, "module") else model
    vision_model = clip.vision_model
    visual_projection = clip.visual_projection

    embeddings_all = []

    with torch.no_grad():
        for i in tqdm(
            range(0, len(crops), batch_size),
            desc=f"CLIP {file_id}",
            disable=not accelerator.is_main_process,
        ):
            batch = crops[i:i + batch_size]

            inputs = processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(accelerator.device)

            vision_out = vision_model(pixel_values=pixel_values)
            pooled = vision_out.pooler_output              # [B, hidden_dim]
            feats = visual_projection(pooled)              # [B, 512]
            feats = F.normalize(feats, dim=-1)

            embeddings_all.append(feats.cpu().numpy())

    embeddings_all = np.concatenate(embeddings_all, axis=0)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if accelerator.is_main_process:
        print(f"Saving {embeddings_all.shape[0]} embeddings")

        with open(output_pkl, "wb") as f:
            pickle.dump({
                "file_id": file_id,
                "embeddings": embeddings_all,
                "metadata": metadata,
            }, f)

        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./clip_embeddings")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--dataset-dir", type=str, default="/data/corerndimage/image_tagging/Dataset/internet_indian_dataset")
    parser.add_argument("--segmentation-dir", type=str, default="/data/corerndimage/image_tagging/Dataset/segmentation")
    parser.add_argument("--tmp-dir", type=str, default="./tmp_clip")
    
    args = parser.parse_args()

    accelerator = Accelerator()

    if accelerator.is_main_process:
        print(f"Processing files {args.start:05d} to {args.end - 1:05d}")
        print(f"Output directory: {args.output_dir}")
        print(f"CLIP model: {args.model_name}")

    model = CLIPModel.from_pretrained(args.model_name)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    model = accelerator.prepare(model)
    model.eval()

    for f in range(args.start, args.end):
        try:
            process_file(
                f,
                model,
                processor,
                accelerator,
                args.output_dir,
                args.batch_size,
                args.dataset_dir,
                args.segmentation_dir,
                args.tmp_dir,
            )
        except Exception as e:
            if accelerator.is_main_process:
                print(f"Error processing {f:05d}: {e}")

if __name__ == "__main__":
    main()
