"""
Simple consolidation: Copy 62 existing images+masks, add 42 new images with generated masks.
"""

import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# ==================== CONFIGURATION ====================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'

# Input directories
OLD_IMAGES_DIR = DATA_DIR / 'floorplan'  # 62 images
OLD_MASKS_DIR = DATA_DIR / 'floorplan_masks_13classes'  # 62 masks
NEW_IMAGES_DIR = DATA_DIR / 'floorplan_42'  # 42 images
NEW_COCO_JSON = DATA_DIR / 'floorplan_42' / 'result.json'  # 42 annotations

# Output directories
OUTPUT_IMAGES_DIR = DATA_DIR / 'floorplan_104'
OUTPUT_MASKS_DIR = DATA_DIR / 'floorplan_masks_104_13classes'

# Clean output directories
if OUTPUT_IMAGES_DIR.exists():
    shutil.rmtree(OUTPUT_IMAGES_DIR)
if OUTPUT_MASKS_DIR.exists():
    shutil.rmtree(OUTPUT_MASKS_DIR)

OUTPUT_IMAGES_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_MASKS_DIR.mkdir(exist_ok=True, parents=True)

# Category mapping: 18 categories in new data -> 13 unified classes
CATEGORY_MAPPING = {
    "background": 0,  # Not used in masks
    "dining_area": 0,
    "bathroom": 1,
    "bedroom": 2,
    "closet": 3,
    "room": 4,
    "corridor": 4,  # Map to room
    "living_room": 4,  # Map to room
    "door": 5,
    "entrance": 6,
    "kitchen": 7,
    "outdoor_space": 8,
    "sliding_door": 9,
    "stairs": 10,
    "window": 11,
    "windows": 11,  # Plural -> singular
    "balcony": 12
}

print("="*70)
print("SIMPLE 104-IMAGE CONSOLIDATION")
print("="*70)

# ==================== STEP 1: Copy 62 existing images + masks ====================
print("\n[1/3] Copying 62 existing images and masks...")

old_images = sorted(list(OLD_IMAGES_DIR.glob('*.png')))
copied_count = 0

for i, img_path in enumerate(tqdm(old_images, desc="Copying old dataset")):
    # Copy image
    new_name = f"old_{i:04d}.png"
    shutil.copy2(img_path, OUTPUT_IMAGES_DIR / new_name)
    
    # Copy corresponding mask
    mask_name = img_path.stem + "_mask.png"
    mask_path = OLD_MASKS_DIR / mask_name
    
    if mask_path.exists():
        new_mask_name = f"old_{i:04d}_mask.png"
        shutil.copy2(mask_path, OUTPUT_MASKS_DIR / new_mask_name)
        copied_count += 1

print(f"✓ Copied {copied_count} image-mask pairs from old dataset")

# ==================== STEP 2: Load new COCO annotations ====================
print("\n[2/3] Loading COCO annotations for 42 new images...")

with open(NEW_COCO_JSON, 'r') as f:
    coco_data = json.load(f)

print(f"✓ Found {len(coco_data['images'])} images")
print(f"✓ Found {len(coco_data['annotations'])} annotations")
print(f"✓ Found {len(coco_data['categories'])} categories")

# Build category ID mapping
category_id_to_class = {}
category_names = {}
for cat in coco_data['categories']:
    category_names[cat['id']] = cat['name']
    if cat['name'] in CATEGORY_MAPPING:
        category_id_to_class[cat['id']] = CATEGORY_MAPPING[cat['name']]
    else:
        print(f"  Warning: Unknown category '{cat['name']}' (id={cat['id']})")

print(f"✓ Mapped {len(category_id_to_class)} categories")

# ==================== STEP 3: Generate masks for 42 new images ====================
print("\n[3/3] Processing 42 new images and generating masks...")

new_count = 0
skipped = []

for img_info in tqdm(coco_data['images'], desc="Processing new images"):
    img_id = img_info['id']
    file_name = img_info['file_name']
    
    # Extract actual filename (it has hash prefix in path)
    actual_filename = Path(file_name).name
    
    # Find the actual file in floorplan_42 directory
    source_path = NEW_IMAGES_DIR / actual_filename
    
    if not source_path.exists():
        skipped.append(actual_filename)
        continue
    
    # Copy image with new name
    new_name = f"new_{img_id:04d}{source_path.suffix}"
    shutil.copy2(source_path, OUTPUT_IMAGES_DIR / new_name)
    
    # Create mask from COCO polygons
    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
    
    # Get all annotations for this image
    for ann in coco_data['annotations']:
        if ann['image_id'] == img_id:
            category_id = ann['category_id']
            
            if category_id not in category_id_to_class:
                continue
            
            class_id = category_id_to_class[category_id]
            
            # Convert segmentation polygon to mask
            if 'segmentation' in ann and len(ann['segmentation']) > 0:
                for seg in ann['segmentation']:
                    if len(seg) < 6:  # Need at least 3 points
                        continue
                    
                    # Convert flat list to list of tuples
                    polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                    
                    # Draw polygon on mask
                    mask_img = Image.fromarray(mask)
                    draw = ImageDraw.Draw(mask_img)
                    draw.polygon(polygon, fill=class_id)
                    mask = np.array(mask_img)
    
    # Save mask
    new_mask_name = f"new_{img_id:04d}_mask.png"
    Image.fromarray(mask).save(OUTPUT_MASKS_DIR / new_mask_name)
    
    new_count += 1

print(f"✓ Processed {new_count} new images")
if skipped:
    print(f"⚠ Skipped {len(skipped)} images (not found)")

# ==================== SUMMARY ====================
total_images = len(list(OUTPUT_IMAGES_DIR.glob('*')))
total_masks = len(list(OUTPUT_MASKS_DIR.glob('*')))

print("\n" + "="*70)
print("✓ CONSOLIDATION COMPLETE!")
print("="*70)
print(f"\nDataset Summary:")
print(f"  Old dataset: {copied_count} image-mask pairs")
print(f"  New dataset: {new_count} image-mask pairs")
print(f"  Total: {total_images} images, {total_masks} masks")
print(f"\nOutput directories:")
print(f"  Images: {OUTPUT_IMAGES_DIR}")
print(f"  Masks:  {OUTPUT_MASKS_DIR}")
print(f"\nReady for training!")
