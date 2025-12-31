"""
Test script for visualize_geometry.py
"""

import sys
import os
from pathlib import Path

# Detect if running from WSL or Windows
if os.path.exists("/mnt/e"):
    output_base = "/mnt/e/github.com/akiya-3d-thesis/3D-Pipeline"
else:
    output_base = r"E:\github.com\akiya-3d-thesis\3D-Pipeline"

sys.path.insert(0, output_base)

from visualize_geometry import visualize_from_json

# Find the geometry JSON from previous test
test_output_dir = os.path.join(output_base, "test_output")
geometry_files = list(Path(test_output_dir).glob("geometry_*.json"))

if not geometry_files:
    print("❌ No geometry JSON found")
    print(f"Checked in: {test_output_dir}")
    sys.exit(1)

geometry_path = str(geometry_files[0])
print(f"Visualizing geometry: {geometry_path}\n")

print("="*60)
print("Creating 3D visualizations...")
print("="*60)

try:
    visualizations_dir = os.path.join(test_output_dir, "visualizations")
    visualize_from_json(geometry_path, visualizations_dir)
    
    print("\n" + "="*60)
    print("✅ SUCCESS!")
    print("="*60)
    print(f"\n📊 Visualizations saved to:")
    print(f"   {visualizations_dir}")
    
    # List created files
    viz_files = list(Path(visualizations_dir).glob("*.png"))
    print(f"\n📁 Generated images ({len(viz_files)}):")
    for f in sorted(viz_files):
        file_size = f.stat().st_size / (1024 * 1024)
        print(f"   - {f.name} ({file_size:.2f} MB)")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
