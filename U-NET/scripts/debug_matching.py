from pathlib import Path

# Same config as training script
IMAGES_DIR = Path(__file__).parent.parent / 'data' / 'floorplan'
MASKS_DIR = Path(__file__).parent.parent / 'data' / 'floorplan_masks'

# Get all image and mask paths
image_files = sorted(list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.jpg")))
mask_files = []
all_masks = list(MASKS_DIR.glob("*_mask.png"))

print(f"Found {len(image_files)} images")
print(f"Found {len(all_masks)} masks")
print()

for img_path in image_files[:3]:  # Just check first 3
    # Normalize filename: remove extra spaces
    img_stem_normalized = ' '.join(img_path.stem.split())
    
    print(f"Image: {img_path.name}")
    print(f"  Stem: [{img_stem_normalized}]")
    
    # Find matching mask
    matched = False
    for mask_path in all_masks:
        mask_stem_normalized = ' '.join(mask_path.stem.replace('_mask', '').split())
        
        if img_stem_normalized == mask_stem_normalized:
            print(f"  ✓ Matched with: {mask_path.name}")
            mask_files.append(mask_path)
            matched = True
            break
    
    if not matched:
        print(f"  ✗ No match found")
        print(f"  First mask stem: [{' '.join(all_masks[0].stem.replace('_mask', '').split())}]")
    print()

print(f"\nTotal matches: {len(mask_files)}")
