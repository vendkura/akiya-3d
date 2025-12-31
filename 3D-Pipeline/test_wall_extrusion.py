"""
Test script for wall_extrusion.py
"""

import sys
import os
from pathlib import Path

# Detect if running from WSL or Windows
if os.path.exists("/mnt/e"):
    # Running from WSL
    output_base = "/mnt/e/github.com/akiya-3d-thesis/3D-Pipeline"
else:
    # Running from Windows
    output_base = r"E:\github.com\akiya-3d-thesis\3D-Pipeline"

sys.path.insert(0, output_base)

from wall_extrusion import WallExtrusion
import json

# Find the boundaries JSON from previous test
test_output_dir = os.path.join(output_base, "test_output")
boundaries_files = list(Path(test_output_dir).glob("*_boundaries.json"))

if not boundaries_files:
    print("❌ No boundaries JSON found")
    print(f"Checked in: {test_output_dir}")
    sys.exit(1)

boundaries_path = str(boundaries_files[0])
print(f"Testing with boundaries: {boundaries_path}")

# Output path
output_geometry_path = os.path.join(test_output_dir, "geometry_3d.json")

print("\n" + "="*60)
print("Running wall extrusion...")
print("="*60)

try:
    # Create extrusion engine
    extrusion = WallExtrusion(
        room_height=2.5,
        wall_thickness=0.2
    )
    
    # Process
    geometry = extrusion.process(boundaries_path, output_geometry_path)
    
    print("\n" + "="*60)
    print("✅ SUCCESS!")
    print("="*60)
    print(f"Generated 3D geometry:")
    print(f"  Vertices: {geometry['metadata']['num_vertices']}")
    print(f"  Faces: {geometry['metadata']['num_faces']}")
    print(f"  Rooms: {len(geometry['room_mapping'])}")
    print(f"  Room height: {geometry['metadata']['room_height']}m")
    print(f"  Wall thickness: {geometry['metadata']['wall_thickness']}m")
    
    print(f"\n📁 Output files:")
    print(f"  - {output_geometry_path}")
    
    # Print sample of geometry structure
    print(f"\n📊 Geometry structure:")
    print(f"  Sample vertex: {geometry['vertices'][0] if geometry['vertices'] else 'None'}")
    print(f"  Sample face: {geometry['faces'][0] if geometry['faces'] else 'None'}")
    print(f"  Sample normal: {geometry['normals'][0] if geometry['normals'] else 'None'}")
    print(f"  Sample color: {geometry['colors'][0] if geometry['colors'] else 'None'}")
    
    # Verify geometry integrity
    print(f"\n✓ Geometry validation:")
    
    # Check all face indices are valid
    max_vertex_idx = max(max(face) for face in geometry['faces'] if face)
    if max_vertex_idx < len(geometry['vertices']):
        print(f"  ✓ All face indices valid (max {max_vertex_idx} < {len(geometry['vertices'])})")
    else:
        print(f"  ❌ Invalid face indices detected!")
    
    # Check vertex counts match
    if len(geometry['vertices']) == len(geometry['normals']) == len(geometry['colors']):
        print(f"  ✓ Vertex/normal/color counts match")
    else:
        print(f"  ❌ Mismatch: V={len(geometry['vertices'])}, N={len(geometry['normals'])}, C={len(geometry['colors'])}")
    
    # Check for orphaned vertices
    used_vertices = set()
    for face in geometry['faces']:
        used_vertices.update(face)
    
    orphaned = len(geometry['vertices']) - len(used_vertices)
    if orphaned == 0:
        print(f"  ✓ No orphaned vertices")
    else:
        print(f"  ⚠️  {orphaned} orphaned vertices")
    
    print("\n" + "="*60)
    print("Ready for OBJ export!")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
