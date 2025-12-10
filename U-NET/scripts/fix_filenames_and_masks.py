"""
Fix: Update JSON filenames to match actual files in floorplan_104 and regenerate masks.
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
import cv2

JSON_PATH = "../data/annotations/coco-annotation-merged-104images_13classes.json"
IMAGES_DIR = Path("../data/floorplan_104")
MASKS_DIR = Path("../data/floorplan_masks_104_13classes")

print("Loading JSON...")
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

print("Getting actual image files...")
actual_files = {f.name: f for f in IMAGES_DIR.glob("*.[jp][pn]g")}
print(f"Found {len(actual_files)} actual image files")

print("\nUpdating JSON filenames...")
updated = 0
not_found = 0

for img in data['images']:
    old_name = img['file_name']
    # Extract just the filename
    basename = Path(old_name).name
    
    if basename in actual_files:
        img['file_name'] = basename
        updated += 1
    else:
        print(f"  ⚠️  Not found: {basename}")
        not_found += 1

print(f"✓ Updated {updated} filenames")
if not_found > 0:
    print(f"⚠️  {not_found} files not found")

# Save updated JSON
print(f"\nSaving updated JSON...")
with open(JSON_PATH, 'w') as f:
    json.dump(data, f, indent=2)
print(f"✓ Saved: {JSON_PATH}")

# Regenerate masks
print("\nRegenerating masks with correct filenames...")
MASKS_DIR.mkdir(exist_ok=True)

img_id_to_file = {img['id']: img['file_name'] for img in data['images']}
img_id_to_size = {img['id']: (img['width'], img['height']) for img in data['images']}

anns_by_image = defaultdict(list)
for ann in data['annotations']:
    anns_by_image[ann['image_id']].append(ann)

processed = 0
for img_id, anns in anns_by_image.items():
    filename = img_id_to_file[img_id]
    width, height = img_id_to_size[img_id]
    
    # Create blank mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw annotations
    for ann in anns:
        if 'segmentation' not in ann or not ann['segmentation']:
            continue
        
        cat_id = ann['category_id']
        
        for seg in ann['segmentation']:
            if len(seg) < 6:
                continue
            
            poly = np.array(seg).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [poly], color=cat_id)
    
    # Save with matching filename
    output_path = MASKS_DIR / filename
    Image.fromarray(mask).save(output_path)
    processed += 1
    
    if processed % 20 == 0:
        print(f"  Processed {processed}/{len(anns_by_image)}...")

print(f"\n✓ Generated {processed} masks")
print(f"✓ Masks saved to: {MASKS_DIR}")
print("\n🚀 Ready to train FPN!")
