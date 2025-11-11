"""
Merge two COCO annotation JSON files safely.
Handles ID conflicts by offsetting new IDs appropriately.
"""

import json
from pathlib import Path

# File paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
ANNOTATIONS_DIR = DATA_DIR / 'annotations'

ORIGINAL_FILE = ANNOTATIONS_DIR / 'coco-annotation.json'  # 30 images
NEW_FILE = ANNOTATIONS_DIR / 'coco-annotation-32images-more.json'  # 32 new images
OUTPUT_FILE = ANNOTATIONS_DIR / 'coco-annotation-merged-62images.json'
BACKUP_FILE = ANNOTATIONS_DIR / 'coco-annotation-30images-backup.json'

print("=" * 70)
print("🔄 MERGING COCO ANNOTATION FILES")
print("=" * 70)

# Load original file (30 images)
print(f"📖 Loading original file: {ORIGINAL_FILE.name}")
with open(ORIGINAL_FILE, 'r', encoding='utf-8') as f:
    original = json.load(f)

print(f"   Images: {len(original['images'])}")
print(f"   Annotations: {len(original['annotations'])}")
print(f"   Categories: {len(original['categories'])}")

# Load new file (32 images)
print(f"\n📖 Loading new file: {NEW_FILE.name}")
with open(NEW_FILE, 'r', encoding='utf-8') as f:
    new = json.load(f)

print(f"   Images: {len(new['images'])}")
print(f"   Annotations: {len(new['annotations'])}")
print(f"   Categories: {len(new['categories'])}")

# Backup original file
print(f"\n💾 Creating backup: {BACKUP_FILE.name}")
with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
    json.dump(original, f, indent=2, ensure_ascii=False)

# Find max IDs in original file
max_image_id = max([img['id'] for img in original['images']]) if original['images'] else 0
max_annotation_id = max([ann['id'] for ann in original['annotations']]) if original['annotations'] else 0

print(f"\n🔢 Original file max IDs:")
print(f"   Max image_id: {max_image_id}")
print(f"   Max annotation_id: {max_annotation_id}")

# Calculate offsets for new file
image_id_offset = max_image_id
annotation_id_offset = max_annotation_id

print(f"\n➕ ID offsets for new file:")
print(f"   Image ID offset: +{image_id_offset}")
print(f"   Annotation ID offset: +{annotation_id_offset}")

# Create merged data structure
merged = {
    'images': original['images'].copy(),
    'annotations': original['annotations'].copy(),
    'categories': original['categories'].copy(),  # Categories should be identical
    'info': original.get('info', {})
}

# Process new images with offset IDs
print(f"\n🖼️  Processing new images...")
image_id_mapping = {}  # Map old image IDs to new ones

for img in new['images']:
    old_id = img['id']
    new_id = old_id + image_id_offset
    image_id_mapping[old_id] = new_id
    
    # Create new image entry with updated ID
    new_img = img.copy()
    new_img['id'] = new_id
    merged['images'].append(new_img)

print(f"   ✅ Added {len(new['images'])} images (IDs remapped)")

# Process new annotations with offset IDs
print(f"\n📝 Processing new annotations...")
for ann in new['annotations']:
    old_id = ann['id']
    old_image_id = ann['image_id']
    
    # Create new annotation with updated IDs
    new_ann = ann.copy()
    new_ann['id'] = old_id + annotation_id_offset
    new_ann['image_id'] = image_id_mapping[old_image_id]
    
    merged['annotations'].append(new_ann)

print(f"   ✅ Added {len(new['annotations'])} annotations (IDs remapped)")

# Verify categories match
print(f"\n🔍 Verifying categories match...")
original_cats = sorted([cat['name'] for cat in original['categories']])
new_cats = sorted([cat['name'] for cat in new['categories']])

if original_cats == new_cats:
    print(f"   ✅ Categories match perfectly!")
else:
    print(f"   ⚠️  WARNING: Category mismatch detected!")
    print(f"   Original: {original_cats}")
    print(f"   New: {new_cats}")

# Save merged file
print(f"\n💾 Saving merged file: {OUTPUT_FILE.name}")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

# Print summary
print("\n" + "=" * 70)
print("✅ MERGE COMPLETE!")
print("=" * 70)
print(f"📊 Merged dataset statistics:")
print(f"   Total images: {len(merged['images'])} (30 + 32)")
print(f"   Total annotations: {len(merged['annotations'])}")
print(f"   Categories: {len(merged['categories'])}")
print(f"\n📂 Files created:")
print(f"   Backup: {BACKUP_FILE}")
print(f"   Merged: {OUTPUT_FILE}")
print("\n💡 Next steps:")
print(f"   1. Verify merged file: Open {OUTPUT_FILE.name}")
print(f"   2. Copy new images to: U-NET/data/floorplan/")
print(f"   3. Update conversion script to use: {OUTPUT_FILE.name}")
print(f"   4. Regenerate all masks for 62 images")
print("=" * 70)
