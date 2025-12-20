"""
Merge 62 existing + 42 new annotated images into 104-image dataset.
Steps:
1. Merge COCO JSONs (62 + 42)
2. Remap to 13 classes
3. Copy all images to unified folder
4. Generate masks for all 104 images
"""

import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict

# Paths
OLD_62_JSON = "../data/annotations/coco-annotation-merged-62images_BACKUP_16classes.json"
NEW_42_JSON = "../data/floorplan_42/result.json"
OLD_62_IMAGES = "../data/floorplan"
NEW_42_IMAGES = "../data/floorplan_42"

OUTPUT_JSON_16_CLASSES = "../data/annotations/coco-annotation-merged-104images_16classes.json"
OUTPUT_JSON_13_CLASSES = "../data/annotations/coco-annotation-merged-104images_13classes.json"
OUTPUT_IMAGES_DIR = "../data/floorplan_104"
OUTPUT_MASKS_DIR = "../data/floorplan_masks_104_13classes"

# Class remapping 16→13
OLD_TO_NEW = {
    "DK": "dining_area",
    "LDK": "dining_area",
    "dinning room": "dining_area",
    "dining_area": "dining_area",
    
    "toilet": "bathroom",
    "washroom": "bathroom",
    "bathroom": "bathroom",
    
    "living room": "room",
    "room": "room",
    
    # Unchanged classes
    "bedroom": "bedroom",
    "closet": "closet",
    "door": "door",
    "entrance": "entrance",
    "kitchen": "kitchen",
    "outdoor space": "outdoor_space",
    "outdoor_space": "outdoor_space",
    "sliding door": "sliding_door",
    "sliding_door": "sliding_door",
    "stairs": "stairs",
    "window": "window",
    "windows": "window",
    "balcony": "balcony"
}

# New 13 class schema
NEW_CLASS_NAMES = [
    "dining_area", "bathroom", "bedroom", "closet", "room",
    "door", "entrance", "kitchen", "outdoor_space",
    "sliding_door", "stairs", "window", "balcony"
]

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def normalize_filename(filename):
    """Extract actual filename from path."""
    return Path(filename).name

def merge_coco_jsons(json1_path, json2_path):
    """Merge two COCO JSON files."""
    print("Loading JSONs...")
    data1 = load_json(json1_path)
    data2 = load_json(json2_path)
    
    # Normalize filenames in both datasets
    for img in data1['images']:
        img['file_name'] = normalize_filename(img['file_name'])
    for img in data2['images']:
        img['file_name'] = normalize_filename(img['file_name'])
    
    # Build category name to id mapping for both datasets
    cat1_map = {cat['name']: cat['id'] for cat in data1['categories']}
    cat2_map = {cat['name']: cat['id'] for cat in data2['categories']}
    
    # Create unified category list (16 classes)
    all_cat_names = set()
    for cat in data1['categories']:
        all_cat_names.add(cat['name'])
    for cat in data2['categories']:
        all_cat_names.add(cat['name'])
    
    unified_categories = [{"id": i, "name": name} for i, name in enumerate(sorted(all_cat_names))]
    unified_cat_map = {cat['name']: cat['id'] for cat in unified_categories}
    
    print(f"Unified categories ({len(unified_categories)}): {[c['name'] for c in unified_categories]}")
    
    # Merge images
    merged_images = data1['images'].copy()
    img_id_offset = max([img['id'] for img in data1['images']]) + 1
    
    for img in data2['images']:
        new_img = img.copy()
        new_img['id'] = img['id'] + img_id_offset
        merged_images.append(new_img)
    
    # Merge annotations with category remapping
    merged_annotations = []
    ann_id = 0
    
    # Process data1 annotations
    for ann in data1['annotations']:
        old_cat_id = ann['category_id']
        old_cat_name = next(cat['name'] for cat in data1['categories'] if cat['id'] == old_cat_id)
        new_cat_id = unified_cat_map[old_cat_name]
        
        new_ann = ann.copy()
        new_ann['id'] = ann_id
        new_ann['category_id'] = new_cat_id
        merged_annotations.append(new_ann)
        ann_id += 1
    
    # Process data2 annotations (with image_id offset)
    for ann in data2['annotations']:
        old_cat_id = ann['category_id']
        old_cat_name = next(cat['name'] for cat in data2['categories'] if cat['id'] == old_cat_id)
        new_cat_id = unified_cat_map[old_cat_name]
        
        new_ann = ann.copy()
        new_ann['id'] = ann_id
        new_ann['image_id'] = ann['image_id'] + img_id_offset
        new_ann['category_id'] = new_cat_id
        merged_annotations.append(new_ann)
        ann_id += 1
    
    merged_data = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": unified_categories
    }
    
    print(f"✓ Merged: {len(merged_images)} images, {len(merged_annotations)} annotations")
    return merged_data

def remap_to_13_classes(coco_data):
    """Remap 16 classes to 13 classes."""
    print("\nRemapping to 13 classes...")
    
    # Create new categories
    new_categories = [{"id": i, "name": name} for i, name in enumerate(NEW_CLASS_NAMES)]
    new_cat_map = {cat['name']: cat['id'] for cat in new_categories}
    
    # Remap annotations
    old_cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    remapped_annotations = []
    mapping_stats = defaultdict(int)
    
    for ann in coco_data['annotations']:
        old_cat_name = old_cat_id_to_name[ann['category_id']]
        
        if old_cat_name not in OLD_TO_NEW:
            print(f"⚠️  Unknown class: {old_cat_name}")
            continue
        
        new_cat_name = OLD_TO_NEW[old_cat_name]
        new_cat_id = new_cat_map[new_cat_name]
        
        new_ann = ann.copy()
        new_ann['category_id'] = new_cat_id
        remapped_annotations.append(new_ann)
        
        mapping_stats[f"{old_cat_name} → {new_cat_name}"] += 1
    
    remapped_data = {
        "images": coco_data['images'],
        "annotations": remapped_annotations,
        "categories": new_categories
    }
    
    print("\nRemapping statistics:")
    for mapping, count in sorted(mapping_stats.items()):
        print(f"  {mapping}: {count} annotations")
    
    print(f"\n✓ Remapped to 13 classes: {len(remapped_annotations)} annotations")
    return remapped_data

def copy_images():
    """Copy all images to unified folder."""
    print("\nCopying images to unified folder...")
    output_dir = Path(OUTPUT_IMAGES_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Copy from old 62
    old_dir = Path(OLD_62_IMAGES)
    for img_file in old_dir.glob("*.[jp][pn]g"):
        shutil.copy2(img_file, output_dir / img_file.name)
    
    # Copy from new 42
    new_dir = Path(NEW_42_IMAGES)
    for img_file in new_dir.glob("*.[jp][pn]g"):
        shutil.copy2(img_file, output_dir / img_file.name)
    
    total = len(list(output_dir.glob("*.[jp][pn]g")))
    print(f"✓ Copied {total} images to {output_dir}")

def generate_masks(coco_json_path):
    """Generate mask images from COCO JSON."""
    print("\nGenerating masks...")
    coco_data = load_json(coco_json_path)
    
    output_dir = Path(OUTPUT_MASKS_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Group annotations by image
    img_id_to_file = {img['id']: normalize_filename(img['file_name']) for img in coco_data['images']}
    img_id_to_size = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
    
    anns_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        anns_by_image[ann['image_id']].append(ann)
    
    processed = 0
    skipped = 0
    
    for img_id, anns in anns_by_image.items():
        filename = img_id_to_file[img_id]
        width, height = img_id_to_size[img_id]
        
        # Create blank mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Draw each annotation
        for ann in anns:
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            
            cat_id = ann['category_id']
            
            # Handle polygon segmentation
            for seg in ann['segmentation']:
                if len(seg) < 6:  # Need at least 3 points
                    continue
                
                # Convert to polygon
                poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                
                # Fill polygon
                from cv2 import fillPoly
                fillPoly(mask, [poly], color=cat_id)
        
        # Save mask
        output_path = output_dir / filename
        Image.fromarray(mask).save(output_path)
        processed += 1
        
        if processed % 10 == 0:
            print(f"  Processed {processed}/{len(anns_by_image)} images...")
    
    print(f"✓ Generated {processed} masks (skipped {skipped})")

# MAIN
if __name__ == "__main__":
    print("="*70)
    print("MERGING 62 + 42 IMAGES → 104 IMAGE DATASET")
    print("="*70)
    
    # Step 1: Merge JSONs (16 classes)
    print("\n[1/5] Merging COCO JSONs...")
    merged_16 = merge_coco_jsons(OLD_62_JSON, NEW_42_JSON)
    save_json(merged_16, OUTPUT_JSON_16_CLASSES)
    print(f"✓ Saved: {OUTPUT_JSON_16_CLASSES}")
    
    # Step 2: Remap to 13 classes
    print("\n[2/5] Remapping to 13 classes...")
    merged_13 = remap_to_13_classes(merged_16)
    save_json(merged_13, OUTPUT_JSON_13_CLASSES)
    print(f"✓ Saved: {OUTPUT_JSON_13_CLASSES}")
    
    # Step 3: Copy images
    print("\n[3/5] Copying images...")
    copy_images()
    
    # Step 4: Generate masks
    print("\n[4/5] Generating masks for 13 classes...")
    generate_masks(OUTPUT_JSON_13_CLASSES)
    
    print("\n" + "="*70)
    print("✓ DONE! 104-image dataset ready for training")
    print("="*70)
    print(f"\nDataset locations:")
    print(f"  Images:  {OUTPUT_IMAGES_DIR}")
    print(f"  Masks:   {OUTPUT_MASKS_DIR}")
    print(f"  JSON:    {OUTPUT_JSON_13_CLASSES}")
    print(f"\n🚀 Ready to train FPN model!")
