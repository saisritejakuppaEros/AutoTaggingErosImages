import os
import json
import tarfile
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
import io
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
import re
import argparse

# CONFIG
BASE_DATASET_DIR = "/mnt/data0/teja/internet_dataset/internet_indian_dataset/internet_indian_dataset"
BASE_TMP_DIR = "/mnt/data0/data0/mouryesh/T2I/tagging/object_tagging/pipeline_generation/tmp"
BASE_OUTPUT_DIR = "./output/vlm_tags"
NUM_WORKERS = 20
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

OBJECT_PROMPT = (
    "Extract ALL visible objects and background regions from this image.\n"
    "Include: people, animals, vehicles, tools, buildings, vegetation, ground, water, sky.\n"
    "Rules: Output ONLY JSON. Use lowercase noun phrases (1-3 words). No duplicates. No text/logos/UI.\n"
    'Format: {"prompts": ["object1", "object2"]}\n'
    "No markdown. No explanations."
)

def safe_parse_json(text: str) -> Dict[str, List[str]]:
    if not text or not text.strip():
        return {"prompts": []}
    
    text = re.sub(r"```json\s*|```\s*", "", text.strip())
    
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "prompts" in obj:
            return {"prompts": [str(p).strip().lower() for p in obj["prompts"] if str(p).strip()]}
    except:
        pass
    
    match = re.search(r'\[(.*?)\]', text, flags=re.DOTALL)
    if match:
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', match.group(1))
        if quoted:
            return {"prompts": [str(a or b).strip().lower() for a, b in quoted if (a or b).strip()]}
    
    return {"prompts": []}

def tag_single_image(image_path: Path, client: OpenAI) -> List[str]:
    try:
        abs_path = str(image_path.resolve())
        
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{abs_path}"}},
                    {"type": "text", "text": OBJECT_PROMPT}
                ]
            }],
            max_tokens=2000,
            temperature=0.0
        )
        
        content = resp.choices[0].message.content
        result = safe_parse_json(content)
        return result["prompts"]
    except Exception as e:
        print(f"Error tagging {image_path.name}: {e}")
        return []

def process_file(file_num: int, port: int):
    """Process a single parquet/tar file pair"""
    
    # Format file number with leading zeros
    file_id = f"{file_num:05d}"
    
    INPUT_PARQUET = f"{BASE_DATASET_DIR}/{file_id}.parquet"
    INPUT_TAR = f"{BASE_DATASET_DIR}/{file_id}.tar"
    TMP_IMAGE_DIR = f"{BASE_TMP_DIR}/{file_id}"
    OUTPUT_JSON = f"{BASE_OUTPUT_DIR}/{file_id}.json"
    BASE_URL = f"http://localhost:{port}/v1"
    
    # Check if files exist
    if not os.path.exists(INPUT_PARQUET):
        print(f"Skipping {file_id}: parquet file not found")
        return
    if not os.path.exists(INPUT_TAR):
        print(f"Skipping {file_id}: tar file not found")
        return
    
    # Check if output already exists
    if os.path.exists(OUTPUT_JSON):
        print(f"Skipping {file_id}: output already exists")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {file_id} on port {port}")
    print(f"{'='*60}")
    
    client = OpenAI(base_url=BASE_URL, api_key="EMPTY")
    
    os.makedirs(TMP_IMAGE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)
    
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
    
    print(f"Loaded {len(metadata_map)} metadata entries")
    
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
    
    print(f"Tagging images for {file_id}...")
    results = []
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(tag_single_image, img_path, client): (key, img_path) for key, img_path in image_keys}
        
        with tqdm(total=len(futures), desc=f"Tagging {file_id}", unit="img") as pbar:
            for future in as_completed(futures):
                key, img_path = futures[future]
                tags = future.result()
                
                meta = metadata_map.get(key, {})
                results.append({
                    "key": key,
                    "image_path": str(img_path.resolve()),
                    "url": meta.get('url', ''),
                    "width": meta.get('width', 0),
                    "height": meta.get('height', 0),
                    "tags": tags
                })
                
                pbar.update(1)
    
    print(f"Saving results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Done! Tagged {len(results)} images for {file_id}")
    
    # Clean up tmp directory to save space
    print(f"Cleaning up tmp directory for {file_id}...")
    import shutil
    shutil.rmtree(TMP_IMAGE_DIR, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description='Tag images from dataset using VLM')
    parser.add_argument('--start', type=int, required=True, help='Start file number (e.g., 0)')
    parser.add_argument('--end', type=int, required=True, help='End file number (exclusive, e.g., 70)')
    parser.add_argument('--port', type=int, required=True, help='VLLM server port (e.g., 8000)')
    
    args = parser.parse_args()
    
    print(f"Processing files {args.start:05d} to {args.end-1:05d} using port {args.port}")
    
    for file_num in range(args.start, args.end):
        try:
            process_file(file_num, args.port)
        except Exception as e:
            print(f"Error processing file {file_num:05d}: {e}")
            continue

if __name__ == "__main__":
    main()