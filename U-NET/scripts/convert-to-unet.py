import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# ==================== CONFIGURATION ====================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
INPUT_JSON = DATA_DIR / 'annotations' / 'coco-annotation.json'
OUTPUT_MASK_DIR = DATA_DIR / 'floorplan_masks'           # For training
OUTPUT_VIZ_DIR = DATA_DIR / 'floorplan_masks_viz'        # For visualization
OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIZ_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("🎨 MASK GENERATION SCRIPT")
print("=" * 60)
print(f"📖 Reading annotations from: {INPUT_JSON}")
print(f"💾 Training masks → {OUTPUT_MASK_DIR}")
print(f"🎨 Visualization masks → {OUTPUT_VIZ_DIR}")
print("=" * 60)

# ==================== LOAD DATA ====================
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

total_images = len(coco_data.get('images', []))
total_annotations = len(coco_data.get('annotations', []))
print(f"📊 Dataset: {total_images} images, {total_annotations} annotations\n")

# ==================== PROCESS EACH IMAGE ====================
processed_count = 0
skipped_count = 0

for img_info in coco_data.get('images', []):
    # Get image dimensions
    width = img_info.get('width')
    height = img_info.get('height')
    img_filename = img_info.get('file_name', '')

    if not width or not height:
        # Try to read from actual image file
        img_path = SCRIPT_DIR / img_filename.replace('\\', '/')
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception as e:
            print(f"⚠️  Skipping {img_filename}: {e}")
            skipped_count += 1
            continue

    # Create TWO mask images
    mask_train = Image.new('L', (width, height), 0)  # Training (class IDs)
    mask_viz = Image.new('L', (width, height), 0)    # Visualization (visible)
    
    draw_train = ImageDraw.Draw(mask_train)
    draw_viz = ImageDraw.Draw(mask_viz)

    # Get annotations for this image
    img_anns = [a for a in coco_data.get('annotations', []) 
                if a.get('image_id') == img_info.get('id')]
    
    print(f"🖼️  [{processed_count + 1}/{total_images}] {Path(img_filename).name}")
    print(f"    Size: {width}x{height}, Annotations: {len(img_anns)}")

    # Draw each annotation
    for ann in img_anns:
        class_id = int(ann.get('category_id', 0))
        visible_class_id = min((class_id + 1) * 15, 255)  # For visualization
        
        segs = ann.get('segmentation', [])
        
        # Draw polygons
        if segs and any(segs):
            for poly in segs:
                if not poly:
                    continue
                
                # Convert to (x,y) tuples
                polygon = [(int(round(poly[i])), int(round(poly[i+1]))) 
                          for i in range(0, len(poly), 2)]
                
                try:
                    draw_train.polygon(polygon, fill=class_id)          # Raw ID
                    draw_viz.polygon(polygon, fill=visible_class_id)    # Visible
                except Exception as e:
                    print(f"    ⚠️  Failed polygon for ann {ann.get('id')}: {e}")
        
        # Fallback to bounding box
        elif 'bbox' in ann:
            bbox = ann['bbox']
            if len(bbox) == 4:
                x, y, w, h = bbox
                x1, y1 = int(round(x)), int(round(y))
                x2, y2 = int(round(x + w)), int(round(y + h))
                
                draw_train.rectangle([x1, y1, x2, y2], fill=class_id)
                draw_viz.rectangle([x1, y1, x2, y2], fill=visible_class_id)

    # Save both masks with clean filenames
    # Remove any hash prefixes and URL encoding from the filename
    from urllib.parse import unquote
    clean_filename = unquote(Path(img_filename).name)  # Decode URL encoding
    basename = Path(clean_filename).stem
    
    mask_train_path = OUTPUT_MASK_DIR / f"{basename}_mask.png"
    mask_viz_path = OUTPUT_VIZ_DIR / f"{basename}_mask_viz.png"
    
    mask_train.save(mask_train_path)
    mask_viz.save(mask_viz_path)
    
    processed_count += 1
    print(f"    ✅ Saved: {mask_train_path.name} + {mask_viz_path.name}\n")

# ==================== SUMMARY ====================
print("=" * 60)
print("🎉 GENERATION COMPLETE!")
print("=" * 60)
print(f"✅ Processed: {processed_count} images")
print(f"⚠️  Skipped: {skipped_count} images")
print(f"\n📂 Training masks: {OUTPUT_MASK_DIR}")
print(f"📂 Visualization masks: {OUTPUT_VIZ_DIR}")
print("=" * 60)
print("\n🚀 Ready for U-Net training!")