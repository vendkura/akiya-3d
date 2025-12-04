"""
Remap COCO Classes from 16 to 13
=================================
This script consolidates similar classes to reduce confusion:
- living room → room
- DK, LDK, dinning room → dining_area
- toilet, washroom → bathroom

Input:  coco-annotation-merged-62images.json (16 classes)
Output: coco-annotation-merged-62images_13classes.json (13 classes)

Usage:
    python remap_classes.py
"""

import json
from pathlib import Path

# Configuration
INPUT_JSON = "../data/annotations/coco-annotation-merged-62images.json"
OUTPUT_JSON = "../data/annotations/coco-annotation-merged-62images_13classes.json"

# Old class name to new class name mapping
OLD_TO_NEW = {
    "DK": "dining_area",
    "LDK": "dining_area",
    "dinning room": "dining_area",
    "living room": "room",
    "toilet": "bathroom",
    "washroom": "bathroom",
    # Keep others the same
    "bedroom": "bedroom",
    "room": "room",
    "closet": "closet",
    "entrance": "entrance",
    "kitchen": "kitchen",
    "outdoor space": "outdoor_space",
    "stairs": "stairs",
    "sliding door": "sliding_door",
    "door": "door",
    "windows": "windows",
}

# New class list (13 classes)
NEW_CLASSES = [
    "dining_area",    # 0 - merged from DK, LDK, dinning room
    "bathroom",       # 1 - merged from toilet, washroom
    "bedroom",        # 2
    "room",           # 3 - merged includes living room
    "closet",         # 4
    "entrance",       # 5
    "kitchen",        # 6
    "outdoor_space",  # 7 - normalized from "outdoor space"
    "stairs",         # 8
    "sliding_door",   # 9 - normalized from "sliding door"
    "door",           # 10
    "windows",        # 11
    "background",     # 12 - if needed
]

def remap_coco_json(input_path, output_path):
    """Remap COCO JSON from 16 classes to 13 classes."""
    
    print("Loading original COCO JSON...")
    with open(input_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    print(f"Original categories: {len(coco_data['categories'])}")
    print(f"Original annotations: {len(coco_data['annotations'])}")
    
    # Step 1: Create old_id to new_id mapping
    old_id_to_new_id = {}
    
    for old_cat in coco_data['categories']:
        old_name = old_cat['name']
        new_name = OLD_TO_NEW.get(old_name, old_name)
        
        # Find new ID for this class
        if new_name in NEW_CLASSES:
            new_id = NEW_CLASSES.index(new_name)
            old_id_to_new_id[old_cat['id']] = new_id
            print(f"  {old_cat['id']:2d} '{old_name:20s}' → {new_id:2d} '{new_name}'")
        else:
            print(f"  WARNING: '{old_name}' not found in new classes!")
    
    # Step 2: Create new categories list
    new_categories = []
    for idx, name in enumerate(NEW_CLASSES):
        new_categories.append({
            "id": idx,
            "name": name
        })
    
    # Step 3: Update all annotations with new category IDs
    updated_annotations = []
    for ann in coco_data['annotations']:
        old_cat_id = ann['category_id']
        if old_cat_id in old_id_to_new_id:
            ann['category_id'] = old_id_to_new_id[old_cat_id]
            updated_annotations.append(ann)
        else:
            print(f"  WARNING: Annotation with category_id {old_cat_id} not mapped!")
    
    # Step 4: Create new COCO structure
    new_coco_data = {
        "images": coco_data['images'],
        "annotations": updated_annotations,
        "categories": new_categories,
        "info": {
            "year": 2025,
            "version": "2.0",
            "description": "Remapped from 16 to 13 classes",
            "contributor": "Label Studio",
            "url": "",
            "date_created": coco_data['info'].get('date_created', '')
        }
    }
    
    # Step 5: Save new JSON
    print(f"\nSaving remapped COCO JSON...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Done!")
    print(f"  New categories: {len(new_coco_data['categories'])} (was {len(coco_data['categories'])})")
    print(f"  Annotations updated: {len(new_coco_data['annotations'])}")
    print(f"  Output: {output_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("NEW CLASS MAPPING (13 classes):")
    print(f"{'='*70}")
    for idx, name in enumerate(NEW_CLASSES):
        print(f"  {idx:2d}: {name}")
    print(f"{'='*70}")

if __name__ == "__main__":
    input_path = Path(__file__).parent / INPUT_JSON
    output_path = Path(__file__).parent / OUTPUT_JSON
    
    print("="*70)
    print("COCO CLASS REMAPPING: 16 → 13 Classes")
    print("="*70)
    
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        exit(1)
    
    remap_coco_json(input_path, output_path)
    
    print("\n✓ Remapping complete!")
    print("\nNext steps:")
    print("  1. Run: python convert-coco-to-unet.py (to regenerate masks)")
    print("  2. Update unet-training.py (NUM_CLASSES=13)")
    print("  3. Train with new 13-class configuration")
