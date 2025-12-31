"""
Test script for boundary_extraction.py
"""

import sys
import os

# Detect if running from WSL or Windows
if os.path.exists("/mnt/e"):
    # Running from WSL
    mask_dir = "/mnt/e/github.com/akiya-3d-thesis/U-NET/data/floorplan_masks_13classes"
    image_dir = "/mnt/e/github.com/akiya-3d-thesis/U-NET/data/floorplan"
    output_base = "/mnt/e/github.com/akiya-3d-thesis/3D-Pipeline"
else:
    # Running from Windows
    mask_dir = r"E:\github.com\akiya-3d-thesis\U-NET\data\floorplan_masks_13classes"
    image_dir = r"E:\github.com\akiya-3d-thesis\U-NET\data\floorplan"
    output_base = r"E:\github.com\akiya-3d-thesis\3D-Pipeline"

sys.path.insert(0, output_base)

from boundary_extraction import BoundaryExtractor
from pathlib import Path

# Get first mask file
mask_files = list(Path(mask_dir).glob("*_mask.png"))
if not mask_files:
    print("❌ No mask files found")
    print(f"Checked in: {mask_dir}")
    sys.exit(1)

mask_path = str(mask_files[0])
print(f"Testing with mask: {mask_path}")

# Find corresponding image
# Mask filename format: "Capture d'écran ... 224143_mask.png"
# Image filename format: "Capture d'écran ... 224143.png"
base_name = mask_files[0].stem.replace("_mask", "")
image_path = None

# Try exact match
candidate = Path(image_dir) / (base_name + ".png")
if candidate.exists():
    image_path = str(candidate)

if image_path:
    print(f"Found corresponding image: {image_path}")
else:
    print(f"⚠️ No corresponding image found, using mask only")

# Create output directory
output_dir = os.path.join(output_base, "test_output")
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Run extraction
print("\n" + "="*60)
print("Running boundary extraction...")
print("="*60)

try:
    extractor = BoundaryExtractor()
    result = extractor.process(mask_path, image_path, output_dir)
    
    print("\n" + "="*60)
    print("✅ SUCCESS!")
    print("="*60)
    print(f"Extracted rooms: {len(result['rooms'])}")
    print(f"Detected walls: {len(result['walls'])}")
    print(f"Scale factor: {result['scale_factor']:.6f} m/pixel")
    
    print(f"\n📁 Output files:")
    print(f"  - {output_dir}/*_boundaries.png (visualization)")
    print(f"  - {output_dir}/*_boundaries.json (geometry data)")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
