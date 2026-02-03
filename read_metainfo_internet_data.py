import pandas as pd
import pyarrow.parquet as pq
import tarfile
import json
import io
from pathlib import Path
from PIL import Image

# File paths
PARQUET_FILE = "/data/corerndimage/image_tagging/Dataset/internet_indian_dataset/00000.parquet"
TAR_FILE = "/data/corerndimage/image_tagging/Dataset/internet_indian_dataset/00000.tar"

print("=" * 80)
print("READING PARQUET FILE")
print("=" * 80)

# Read parquet file
try:
    # Read first row from parquet
    pf = pq.ParquetFile(PARQUET_FILE)
    batch = next(pf.iter_batches(batch_size=1))
    df = batch.to_pandas()
    
    print(f"\nParquet file schema:")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst sample from parquet:")
    print("-" * 80)
    
    # Print all columns and their values
    for col in df.columns:
        value = df[col].iloc[0]
        print(f"{col}: {value}")
    
    print("\n" + "=" * 80)
    print("First sample as dictionary:")
    print("-" * 80)
    print(df.iloc[0].to_dict())
    
except Exception as e:
    print(f"Error reading parquet file: {e}")

print("\n" + "=" * 80)
print("READING TAR FILE")
print("=" * 80)

# Read tar file
try:
    with tarfile.open(TAR_FILE, "r") as tar:
        members = tar.getmembers()
        print(f"\nTotal files in tar: {len(members)}")
        
        # Find first image and metadata pair
        image_files = sorted([m for m in members if m.name.endswith(('.jpg', '.jpeg', '.png'))], key=lambda m: m.name)
        json_files = sorted([m for m in members if m.name.endswith('.json')], key=lambda m: m.name)
        
        if image_files and json_files:
            # Get first image
            first_image = image_files[0]
            print(f"\nFirst image file: {first_image.name}")
            print(f"Image size: {first_image.size} bytes")
            
            # Extract image
            image_data = tar.extractfile(first_image).read()
            print(f"Image data length: {len(image_data)} bytes")
            
            # Get corresponding JSON (assuming same base name)
            base_name = Path(first_image.name).stem
            matching_json = None
            for jf in json_files:
                if Path(jf.name).stem == base_name:
                    matching_json = jf
                    break
            
            if matching_json:
                print(f"\nMatching metadata file: {matching_json.name}")
                json_data = tar.extractfile(matching_json).read()
                metadata = json.loads(json_data.decode('utf-8'))
                
                print("\nFirst sample metadata from tar:")
                print("-" * 80)
                for key, value in metadata.items():
                    if key not in ('theme', 'caption'):  # Skip theme and caption fields
                        print(f"{key}: {value}")
            else:
                print("\nNo matching JSON file found for first image")
                # Try to read first JSON anyway
                if json_files:
                    first_json = json_files[0]
                    print(f"\nReading first JSON file instead: {first_json.name}")
                    json_data = tar.extractfile(first_json).read()
                    metadata = json.loads(json_data.decode('utf-8'))
                    print("\nFirst JSON metadata:")
                    print("-" * 80)
                    for key, value in metadata.items():
                        if key not in ('theme', 'caption'):  # Skip theme and caption fields
                            print(f"{key}: {value}")
        else:
            print("\nNo image or JSON files found in tar")
            # List all files
            print("\nAll files in tar:")
            for member in members[:10]:  # Show first 10
                print(f"  {member.name} ({member.size} bytes)")
        
        # Print shapes of first 100 images
        print("\n" + "=" * 80)
        print("FIRST 100 IMAGE SHAPES")
        print("=" * 80)
        if image_files:
            print(f"\nTotal images found: {len(image_files)}")
            print(f"\nShapes of first 100 images:")
            print("-" * 80)
            for i, img_file in enumerate(image_files[:100], 1):
                try:
                    image_data = tar.extractfile(img_file).read()
                    img = Image.open(io.BytesIO(image_data))
                    width, height = img.size
                    print(f"{i:3d}. {img_file.name}: {width}x{height} (shape: ({height}, {width}, {len(img.getbands())}))")
                except Exception as e:
                    print(f"{i:3d}. {img_file.name}: Error reading image - {e}")
        else:
            print("\nNo image files found in tar")
                
except Exception as e:
    print(f"Error reading tar file: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
