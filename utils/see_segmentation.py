import streamlit as st
import json
import os
from PIL import Image, ImageDraw, ImageFont
import random


# streamlit run see_segmentation.py

# CONFIG
BASE_DATASET_DIR = "/data/corerndimage/image_tagging/target_dataset"
BASE_OUTPUT_DIR = "/data/corerndimage/image_tagging/Dataset/segmentation_output"

st.set_page_config(page_title="SAM3 Segmentation Viewer", layout="wide")

st.title("🎯 SAM3 Segmentation Results Viewer")

# Sidebar for folder selection
st.sidebar.header("Select Folder")

# Get available folders
available_folders = []
if os.path.exists(BASE_OUTPUT_DIR):
    json_files = [f for f in os.listdir(BASE_OUTPUT_DIR) if f.endswith('.json')]
    available_folders = sorted([f.replace('.json', '') for f in json_files])

if not available_folders:
    st.error(f"No segmentation results found in {BASE_OUTPUT_DIR}")
    st.stop()

selected_folder = st.sidebar.selectbox("Choose folder:", available_folders)

# Load results
OUTPUT_JSON = f"{BASE_OUTPUT_DIR}/{selected_folder}.json"
INPUT_FOLDER = f"{BASE_DATASET_DIR}/{selected_folder}"

with open(OUTPUT_JSON, 'r') as f:
    results = json.load(f)

st.sidebar.write(f"**Total images:** {len(results)}")
total_boxes = sum(len(r['boxes']) for r in results)
st.sidebar.write(f"**Total boxes:** {total_boxes}")

# Filter options
show_only_with_boxes = st.sidebar.checkbox("Show only images with boxes", value=True)
min_boxes = st.sidebar.slider("Minimum boxes per image", 0, 20, 0)

# Filter results
filtered_results = [r for r in results if len(r['boxes']) >= min_boxes]
if show_only_with_boxes:
    filtered_results = [r for r in filtered_results if len(r['boxes']) > 0]

st.sidebar.write(f"**Filtered images:** {len(filtered_results)}")

# Image selection
if not filtered_results:
    st.warning("No images match the filter criteria")
    st.stop()

image_idx = st.sidebar.number_input(
    "Image index", 
    min_value=0, 
    max_value=len(filtered_results)-1, 
    value=0
)

# Navigation buttons
col1, col2, col3 = st.sidebar.columns(3)
if col1.button("⬅️ Prev"):
    if image_idx > 0:
        st.rerun()
if col2.button("🎲 Random"):
    st.rerun()
if col3.button("Next ➡️"):
    if image_idx < len(filtered_results) - 1:
        st.rerun()

# Get selected image data
selected_data = filtered_results[image_idx]

# Display image info
st.subheader(f"Image: {selected_data['key']}")
col1, col2, col3 = st.columns(3)
col1.metric("Width", selected_data.get('width', 'N/A'))
col2.metric("Height", selected_data.get('height', 'N/A'))
col3.metric("Boxes", len(selected_data['boxes']))

if selected_data.get('url'):
    st.caption(f"URL: {selected_data['url']}")

# Load and display image
json_image_path = selected_data.get('image_path', '')
if json_image_path:
    image_filename = os.path.basename(json_image_path)
else:
    image_filename = f"{selected_data['key']}.jpg"

actual_image_path = os.path.join(INPUT_FOLDER, image_filename)

if not os.path.exists(actual_image_path):
    st.error(f"Image not found: {actual_image_path}")
    st.stop()

# Load image
image = Image.open(actual_image_path)
image_with_boxes = image.copy()
draw = ImageDraw.Draw(image_with_boxes)

# Color map for labels
colors = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
    '#FFA500', '#800080', '#008000', '#FFC0CB', '#A52A2A', '#808080'
]

label_colors = {}
color_idx = 0

# Draw boxes
boxes_data = selected_data['boxes']
for box in boxes_data:
    bbox = box['bbox']
    label = box['label']
    area = box.get('area', 0)

    # Assign color to label
    if label not in label_colors:
        label_colors[label] = colors[color_idx % len(colors)]
        color_idx += 1

    color = label_colors[label]

    # Draw rectangle
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    # Draw label background
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    text = f"{label}"
    bbox_text = draw.textbbox((x1, y1), text, font=font)
    text_width = bbox_text[2] - bbox_text[0]
    text_height = bbox_text[3] - bbox_text[1]

    # Draw text background
    draw.rectangle(
        [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
        fill=color
    )

    # Draw text
    draw.text((x1 + 2, y1 - text_height - 2), text, fill='white', font=font)

# Display images side by side
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

with col2:
    st.subheader("With Bounding Boxes")
    st.image(image_with_boxes, use_container_width=True)

# Display box details
st.subheader("Detected Objects")

if boxes_data:
    # Create a table
    table_data = []
    for idx, box in enumerate(boxes_data):
        table_data.append({
            "#": idx + 1,
            "Label": box['label'],
            "BBox": f"({box['bbox'][0]:.1f}, {box['bbox'][1]:.1f}, {box['bbox'][2]:.1f}, {box['bbox'][3]:.1f})",
            "Area": f"{box.get('area', 0):.1f}",
            "Color": label_colors.get(box['label'], '#000000')
        })

    # Display as dataframe
    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True
    )

    # Display label statistics
    st.subheader("Label Statistics")
    label_counts = {}
    for box in boxes_data:
        label = box['label']
        label_counts[label] = label_counts.get(label, 0) + 1

    cols = st.columns(min(len(label_counts), 4))
    for idx, (label, count) in enumerate(sorted(label_counts.items())):
        cols[idx % len(cols)].metric(label, count)
else:
    st.info("No bounding boxes detected for this image")

# Export option
st.sidebar.markdown("---")
if st.sidebar.button("📥 Export Current Image"):
    output_path = f"export_{selected_data['key']}.jpg"
    image_with_boxes.save(output_path)
    st.sidebar.success(f"Saved to {output_path}")