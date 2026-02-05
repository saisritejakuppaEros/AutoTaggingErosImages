#!/usr/bin/env python3
"""
Print labels and tags from tagging results JSON file.
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def print_labels(results_path: Path, show_tags: bool = False, show_stats: bool = False, limit: int = None):
    """Print labels from tagging results.
    
    Args:
        results_path: Path to tagging results JSON file
        show_tags: If True, also print tags from CSV themes
        show_stats: If True, print statistics about labels/tags
        limit: Limit number of images to process (None = all)
    """
    print(f"Loading results from: {results_path}")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} images\n")
    print("=" * 80)
    
    all_labels = []
    all_predicted_tags = []
    all_tags = []
    
    images_to_process = data[:limit] if limit else data
    
    for idx, record in enumerate(images_to_process):
        key = record.get("key", "unknown")
        url = record.get("url", "")
        boxes = record.get("boxes", [])
        
        print(f"\nImage {idx + 1}: {key}")
        if url:
            print(f"  URL: {url[:80]}...")
        print(f"  Boxes: {len(boxes)}")
        
        for box_idx, box in enumerate(boxes):
            label = box.get("label", "")
            predicted_tag = box.get("predicted_tag", "")
            tags = box.get("tags", [])
            top_themes = box.get("top_themes", [])
            
            all_labels.append(label)
            if predicted_tag:
                all_predicted_tags.append(predicted_tag)
            all_tags.extend(tags)
            
            print(f"\n  Box {box_idx + 1}:")
            print(f"    Label: {label}")
            if predicted_tag:
                print(f"    Predicted Tag: {predicted_tag}")
            if top_themes:
                print(f"    Top Themes: {', '.join(top_themes[:5])}")
            if show_tags and tags:
                print(f"    Tags ({len(tags)}): {', '.join(tags[:20])}")
                if len(tags) > 20:
                    print(f"      ... and {len(tags) - 20} more")
    
    if show_stats:
        print("\n" + "=" * 80)
        print("STATISTICS")
        print("=" * 80)
        
        label_counts = Counter(all_labels)
        print(f"\nUnique Labels: {len(label_counts)}")
        print("Top 10 Labels:")
        for label, count in label_counts.most_common(10):
            print(f"  {label}: {count}")
        
        if all_predicted_tags:
            tag_counts = Counter(all_predicted_tags)
            print(f"\nUnique Predicted Tags: {len(tag_counts)}")
            print("Top 10 Predicted Tags:")
            for tag, count in tag_counts.most_common(10):
                print(f"  {tag}: {count}")
        
        if all_tags:
            csv_tag_counts = Counter(all_tags)
            print(f"\nUnique CSV Tags: {len(csv_tag_counts)}")
            print("Top 20 CSV Tags:")
            for tag, count in csv_tag_counts.most_common(20):
                print(f"  {tag}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Print labels and tags from tagging results")
    parser.add_argument("results_path", type=str, help="Path to tagging results JSON file")
    parser.add_argument("--tags", action="store_true", help="Show tags from CSV themes")
    parser.add_argument("--stats", action="store_true", help="Show statistics about labels/tags")
    parser.add_argument("--limit", type=int, help="Limit number of images to process")
    
    args = parser.parse_args()
    
    results_path = Path(args.results_path)
    if not results_path.exists():
        print(f"Error: File not found: {results_path}")
        return
    
    print_labels(results_path, show_tags=args.tags, show_stats=args.stats, limit=args.limit)


if __name__ == "__main__":
    main()
