"""
Consolidate 62+42 images into unified 104-image dataset.
Uses COCO annotations to handle filename mapping.
"""

import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==================== CONFIGURATION ====================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'

# Input directories
OLD_IMAGES_DIR = DATA_DIR / 'floorplan'  # 62 images (French names)
NEW_IMAGES_DIR = DATA_DIR / 'floorplan_42'  # 42 images (hash names)
COCO_JSON = DATA_DIR / 'annotations' / 'coco-annotation-merged-104images_13classes.json'

# Output directories
OUTPUT_IMAGES_DIR = DATA_DIR / 'floorplan_104'
OUTPUT_MASKS_DIR = DATA_DIR / 'floorplan_masks_104_13classes'

# Create output directories
OUTPUT_IMAGES_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_MASKS_DIR.mkdir(exist_ok=True, parents=True)

# 13 class mapping
CLASS_MAPPING = {
    "dining_area": 0,
    "bathroom": 1,
    "bedroom": 2,
    "closet": 3,
    "room": 4,
    "door": 5,
    "entrance": 6,
    "kitchen": 7,
    "outdoor_space": 8,
    "sliding_door": 9,
    "stairs": 10,
    "window": 11,
    "balcony": 12
}

print("="*70)
print("CONSOLIDATING 104-IMAGE DATASET")
print("="*70)

# Load COCO annotations
print(f"\nLoading COCO annotations: {COCO_JSON}")
with open(COCO_JSON, 'r') as f:
    coco_data = json.load(f)

print(f"✓ Found {len(coco_data['images'])} images in COCO")
print(f"✓ Found {len(coco_data['annotations'])} annotations")
print(f"✓ Found {len(coco_data['categories'])} categories")

# Build category mapping
category_id_to_class = {}
for cat in coco_data['categories']:
    if cat['name'] in CLASS_MAPPING:
        category_id_to_class[cat['id']] = CLASS_MAPPING[cat['name']]

print(f"\n✓ Mapped {len(category_id_to_class)} categories to 13 classes")

# Process each image
print(f"\nProcessing images...")
successful = 0
failed = []

for img_info in tqdm(coco_data['images'], desc="Consolidating"):
    img_id = img_info['id']
    file_name = img_info['file_name']
    
    # Extract actual filename from path
    # Examples:
    # "/data/upload/5/3fa1f399-Capture_d%C3%A9cran_2025-06-05_222845.png"
    # "G:\\My Drive\\LabelStudio-Data\\media\\upload\\5\\03c96914-5601.jpg"
    
    actual_filename = Path(file_name).name
    
    # The hash prefix is before the first hyphen
    # e.g., "03c96914-5601.jpg" -> hash is "03c96914"
    # e.g., "3fa1f399-Capture_d%C3%A9cran_2025-06-05_222845.png" -> hash is "3fa1f399"
    
    # Try to find the image in either directory
    source_path = None
    
    # Check in new 42 images (hash names)
    new_img_path = NEW_IMAGES_DIR / actual_filename
    if new_img_path.exists():
        source_path = new_img_path
    else:
        # Check in old 62 images (need to match by the part after hash)
        # Extract the part after the hash prefix
        if '-' in actual_filename:
            # Remove hash prefix and URL encoding
            parts = actual_filename.split('-', 1)
            if len(parts) > 1:
                name_part = parts[1].replace('%C3%A9', 'é').replace('_', ' ').replace('%20', ' ')
                
                # Try to find matching file in old directory
                for old_file in OLD_IMAGES_DIR.glob('*'):
                    if name_part.lower() in old_file.name.lower():
                        source_path = old_file
                        break
    
    if source_path is None:
        failed.append(actual_filename)
        continue
    
    # Copy image to output directory
    # Use a clean filename: just the image ID
    output_image_name = f"image_{img_id:04d}{source_path.suffix}"
    output_image_path = OUTPUT_IMAGES_DIR / output_image_name
    
    shutil.copy2(source_path, output_image_path)
    
    # Create mask from COCO annotations
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
                from PIL import ImageDraw
                
                for seg in ann['segmentation']:
                    if len(seg) < 6:  # Need at least 3 points (6 coords)
                        continue
                    
                    # Convert flat list to list of tuples
                    polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                    
                    # Draw polygon on mask
                    mask_img = Image.fromarray(mask)
                    draw = ImageDraw.Draw(mask_img)
                    draw.polygon(polygon, fill=class_id)
                    mask = np.array(mask_img)
    
    # Save mask
    output_mask_name = f"image_{img_id:04d}_mask.png"
    output_mask_path = OUTPUT_MASKS_DIR / output_mask_name
    Image.fromarray(mask).save(output_mask_path)
    
    successful += 1

print(f"\n{'='*70}")
print(f"✓ CONSOLIDATION COMPLETE!")
print(f"{'='*70}")
print(f"\nSuccessful: {successful}/{len(coco_data['images'])}")
if failed:
    print(f"Failed: {len(failed)}")
    print("Failed files:")
    for f in failed[:10]:
        print(f"  - {f}")

print(f"\nOutput directories:")
print(f"  Images: {OUTPUT_IMAGES_DIR} ({len(list(OUTPUT_IMAGES_DIR.glob('*')))} files)")
print(f"  Masks:  {OUTPUT_MASKS_DIR} ({len(list(OUTPUT_MASKS_DIR.glob('*')))} files)")
