#!/usr/bin/env python3
"""
Streamlit app to visualize image tagging results.
Loads images, segmentations, tags, and labels from the tagging pipeline results.

Usage:
    streamlit run utils/visualize_tags.py

Requirements:
    - streamlit
    - pillow
    - toml (python-toml or tomli)
"""

import json
import toml
from pathlib import Path
from typing import Optional, Dict, List
from collections import Counter
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Configuration Loading
# ============================================================
@st.cache_data
def load_config(config_path: str = "config.toml") -> dict:
    """Load configuration from config.toml"""
    config_path = Path(config_path)
    if not config_path.exists():
        st.error(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, "r") as f:
        return toml.load(f)


# ============================================================
# Data Loading Functions
# ============================================================
@st.cache_data
def load_results_json(results_path: Path) -> List[dict]:
    """Load tagging results JSON file"""
    if not results_path.exists():
        return []
    
    with open(results_path, "r") as f:
        return json.load(f)


@st.cache_data
def load_original_tags(tags_json_path: Path) -> Dict[str, List[str]]:
    """Load original tags from target dataset JSON"""
    if not tags_json_path.exists():
        return {}
    
    with open(tags_json_path, "r") as f:
        data = json.load(f)
    
    return {item["key"]: item.get("tags", []) for item in data}


def find_image_file(image_dir: Path, key: str) -> Optional[Path]:
    """Find image file by key, trying different extensions"""
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        candidate = image_dir / f"{key}{ext}"
        if candidate.exists():
            return candidate
    return None


# ============================================================
# Image Processing Functions
# ============================================================
def draw_boxes_on_image(image: Image.Image, boxes: List[dict], selected_box_idx: Optional[int] = None) -> Image.Image:
    """Draw bounding boxes on image"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
        "#FFA500", "#800080", "#FFC0CB", "#A52A2A", "#808080", "#000080"
    ]
    
    for idx, box in enumerate(boxes):
        bbox = box.get("bbox", [])
        if len(bbox) != 4:
            continue
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Use different color for selected box
        if selected_box_idx == idx:
            color = "#FF00FF"  # Magenta for selected
            width = 4
        else:
            color = colors[idx % len(colors)]
            width = 2
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        
        # Draw label
        label = box.get("label", "")
        predicted_tag = box.get("predicted_tag", "")
        top_themes = box.get("top_themes", [])
        
        # Create label text
        if predicted_tag:
            label_text = f"{label} → {predicted_tag}"
        else:
            label_text = label
        
        # Draw text background for label
        if font:
            bbox_text = draw.textbbox((x1, y1 - 20), label_text, font=font)
        else:
            bbox_text = (x1, y1 - 20, x1 + len(label_text) * 8, y1)
        
        draw.rectangle(bbox_text, fill=color, outline=color)
        draw.text((x1, y1 - 20), label_text, fill="white", font=font)
        
        # Draw top themes below the label if available
        if top_themes:
            themes_text = ", ".join(top_themes[:2])  # Show top 2 themes to save space
            if len(top_themes) > 2:
                themes_text += f" (+{len(top_themes) - 2})"
            
            # Position themes text below the label (y1 - 20 for label, so y1 - 5 for themes)
            # But if that's too close, put it at y2 + 5 (below the box)
            themes_y = y2 + 5  # Position below the box
            
            # Try smaller font for themes
            try:
                themes_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except:
                themes_font = font
            
            # Draw themes text background
            if themes_font:
                themes_bbox = draw.textbbox((x1, themes_y), themes_text, font=themes_font)
            else:
                themes_bbox = (x1, themes_y, x1 + len(themes_text) * 6, themes_y + 12)
            
            # Use a slightly different color for themes
            themes_color = "#444444"  # Dark gray background
            draw.rectangle(themes_bbox, fill=themes_color, outline=themes_color)
            draw.text((x1, themes_y), themes_text, fill="white", font=themes_font)
    
    return img_copy


# ============================================================
# Streamlit App
# ============================================================
def main():
    st.set_page_config(page_title="Image Tagging Visualizer", layout="wide")
    st.title("Image Tagging Visualizer")
    
    # Load config
    config = load_config()
    if not config:
        st.stop()
    
    pipe = config.get("tool", {}).get("pipeline", {})
    target_dataset = Path(pipe.get("target_dataset", "./target_dataset"))
    target_output = Path(pipe.get("target_output", "./target_processed"))
    
    # Sidebar for folder selection
    st.sidebar.header("Configuration")
    
    # Find available folders
    available_folders = []
    if target_output.exists():
        for folder in sorted(target_output.iterdir()):
            if folder.is_dir():
                tagging_results = folder / "tagging" / "results.json"
                if tagging_results.exists():
                    available_folders.append(folder.name)
    
    if not available_folders:
        st.error("No processed folders found. Please run the tagging pipeline first.")
        st.stop()
    
    selected_folder = st.sidebar.selectbox("Select Folder", available_folders)
    
    # Load results
    results_path = target_output / selected_folder / "tagging" / "results.json"
    results_data = load_results_json(results_path)
    
    if not results_data:
        st.error(f"No results found in {results_path}")
        st.stop()
    
    # Load original tags
    tags_json_path = target_dataset / f"{selected_folder}.json"
    original_tags = load_original_tags(tags_json_path)
    
    # Image selection
    st.sidebar.header("Image Selection")
    image_keys = [item["key"] for item in results_data]
    selected_key = st.sidebar.selectbox("Select Image", image_keys)
    
    # Find selected image data
    selected_image_data = next((item for item in results_data if item["key"] == selected_key), None)
    if not selected_image_data:
        st.error(f"Image {selected_key} not found")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"Image: {selected_key}")
        
        # Load and display original image
        image_dir = target_dataset / selected_folder
        image_path = find_image_file(image_dir, selected_key)
        
        if not image_path:
            st.error(f"Image file not found for key {selected_key} in {image_dir}")
        else:
            try:
                original_image = Image.open(image_path).convert("RGB")
                
                # Draw boxes on image (will be updated when box is selected)
                boxes = selected_image_data.get("boxes", [])
                annotated_image = draw_boxes_on_image(original_image, boxes)
                
                st.image(annotated_image, caption=f"Image with {len(boxes)} segments", use_container_width=True)
                
                # Image metadata
                st.subheader("Image Metadata")
                st.write(f"**URL:** {selected_image_data.get('url', 'N/A')}")
                st.write(f"**Dimensions:** {selected_image_data.get('width', 0)} x {selected_image_data.get('height', 0)}")
                st.write(f"**Number of segments:** {len(boxes)}")
                
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    with col2:
        st.header("Tags & Labels")
        
        # Original tags
        tags = original_tags.get(selected_key, [])
        if tags:
            st.subheader("Original Tags")
            st.write(", ".join(tags))
        
        # Segmentation boxes
        boxes = selected_image_data.get("boxes", [])
        if boxes:
            st.subheader(f"Segments ({len(boxes)})")
            
            # Show all boxes summary
            with st.expander("View All Boxes", expanded=False):
                for idx, box in enumerate(boxes):
                    label = box.get("label", "unknown")
                    predicted_tag = box.get("predicted_tag", "")
                    top_themes = box.get("top_themes", [])
                    tags_list = box.get("tags", [])
                    
                    st.write(f"**Box {idx + 1}:**")
                    st.write(f"  Label: {label}")
                    if predicted_tag:
                        st.write(f"  Predicted Tag: :green[{predicted_tag}]")
                    if top_themes:
                        st.write(f"  Top Themes: {', '.join(top_themes[:5])}")
                    if tags_list:
                        st.write(f"  Tags ({len(tags_list)}): {', '.join(tags_list[:10])}")
                        if len(tags_list) > 10:
                            st.write(f"    ... and {len(tags_list) - 10} more")
                    st.write("---")
            
            # Box selection
            box_labels = []
            for idx, box in enumerate(boxes):
                label = box.get("label", "unknown")
                predicted_tag = box.get("predicted_tag", "")
                if predicted_tag:
                    box_labels.append(f"{idx}: {label} → {predicted_tag}")
                else:
                    box_labels.append(f"{idx}: {label}")
            
            selected_box_idx = st.selectbox("Select Segment", range(len(boxes)), format_func=lambda x: box_labels[x])
            selected_box = boxes[selected_box_idx]
            
            # Display selected box details
            st.write("**Bounding Box:**", selected_box.get("bbox", []))
            st.write("**Label:**", selected_box.get("label", "N/A"))
            st.write("**Area:**", f"{selected_box.get('area', 0):.2f}")
            
            predicted_tag = selected_box.get("predicted_tag")
            if predicted_tag:
                st.write("**Predicted Tag:**", f":green[{predicted_tag}]")
            
            top_themes = selected_box.get("top_themes", [])
            if top_themes:
                st.write("**Top Themes:**", ", ".join(top_themes[:5]))
            
            # Show tags from CSV themes
            tags_list = selected_box.get("tags", [])
            if tags_list:
                st.write(f"**Tags ({len(tags_list)}):**")
                # Display tags in a more readable format
                tags_display = ", ".join(tags_list[:20])
                st.write(tags_display)
                if len(tags_list) > 20:
                    st.write(f"  ... and {len(tags_list) - 20} more")
            
            neighbors = selected_box.get("neighbors", [])
            if neighbors:
                with st.expander(f"Neighbors ({len(neighbors)})"):
                    for i, neighbor in enumerate(neighbors[:5]):  # Show top 5
                        st.write(f"**{i+1}.** {neighbor.get('label', 'N/A')} "
                               f"(distance: {neighbor.get('distance', 0):.3f})")
                        st.write(f"   Key: {neighbor.get('key', 'N/A')}")
    
    # Statistics section
    st.header("Statistics")
    
    boxes = selected_image_data.get("boxes", [])
    if boxes:
        all_labels = [box.get("label", "") for box in boxes]
        all_predicted_tags = [box.get("predicted_tag", "") for box in boxes if box.get("predicted_tag")]
        all_tags = []
        all_top_themes = []
        
        for box in boxes:
            tags_list = box.get("tags", [])
            all_tags.extend(tags_list)
            top_themes = box.get("top_themes", [])
            all_top_themes.extend(top_themes)
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.subheader("Labels")
            label_counts = Counter(all_labels)
            st.write(f"Unique: {len(label_counts)}")
            st.write("Top 5:")
            for label, count in label_counts.most_common(5):
                st.write(f"  {label}: {count}")
        
        with col_stat2:
            st.subheader("Predicted Tags")
            if all_predicted_tags:
                tag_counts = Counter(all_predicted_tags)
                st.write(f"Unique: {len(tag_counts)}")
                st.write("Top 5:")
                for tag, count in tag_counts.most_common(5):
                    st.write(f"  {tag}: {count}")
            else:
                st.write("No predicted tags")
        
        with col_stat3:
            st.subheader("CSV Tags")
            if all_tags:
                csv_tag_counts = Counter(all_tags)
                st.write(f"Unique: {len(csv_tag_counts)}")
                st.write("Top 5:")
                for tag, count in csv_tag_counts.most_common(5):
                    st.write(f"  {tag}: {count}")
            else:
                st.write("No CSV tags")
        
        if all_top_themes:
            st.subheader("Top Themes")
            theme_counts = Counter(all_top_themes)
            st.write(f"Unique themes: {len(theme_counts)}")
            st.write("Top 10 themes:")
            for theme, count in theme_counts.most_common(10):
                st.write(f"  {theme}: {count}")
    
    # Crop images section
    st.header("Segmentation Crops")
    
    boxes = selected_image_data.get("boxes", [])
    if boxes:
        # Filter boxes with valid crop paths
        valid_boxes = [(idx, box) for idx, box in enumerate(boxes) if box.get("crop_path")]
        
        if valid_boxes:
            num_cols = 4
            cols = st.columns(num_cols)
            
            for idx, (box_idx, box) in enumerate(valid_boxes):
                col = cols[idx % num_cols]
                
                crop_path_str = box.get("crop_path", "")
                # Handle relative paths - resolve relative to workspace root
                crop_path = Path(crop_path_str)
                if not crop_path.is_absolute():
                    # Resolve relative to workspace root (where config.toml is)
                    workspace_root = Path(".").resolve()
                    crop_path = workspace_root / crop_path_str
                
                if crop_path.exists():
                    try:
                        crop_image = Image.open(crop_path).convert("RGB")
                        
                        label = box.get("label", "")
                        predicted_tag = box.get("predicted_tag", "")
                        top_themes = box.get("top_themes", [])
                        
                        caption = label
                        if predicted_tag:
                            caption += f" → {predicted_tag}"
                        
                        col.image(crop_image, caption=caption, use_container_width=True)
                        col.write(f"Area: {box.get('area', 0):.0f}")
                        
                        # Show top themes
                        if top_themes:
                            themes_display = ", ".join(top_themes[:3])
                            if len(top_themes) > 3:
                                themes_display += f" (+{len(top_themes) - 3} more)"
                            col.write(f"**Themes:** {themes_display}")
                    except Exception as e:
                        col.error(f"Error loading crop: {e}")
                else:
                    col.warning(f"Crop not found: {crop_path}")


if __name__ == "__main__":
    main()
