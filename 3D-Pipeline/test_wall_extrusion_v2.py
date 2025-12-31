"""
Test the new wall_extrusion.py v2
"""

import json
from pathlib import Path
from wall_extrusion import SimpleWallExtrusion

"""
Test the new wall_extrusion.py v2
"""

import json
from pathlib import Path
from wall_extrusion import SimpleWallExtrusion

# Find first boundary file in test_output
test_dir = Path("test_output")
boundary_files = list(test_dir.glob("*_boundaries.json"))

if not boundary_files:
    print(f"✗ No boundaries files found in test_output/")
    exit(1)

boundaries_file = boundary_files[0]
print(f"Testing wall_extrusion.py v2...")
print(f"Input: {boundaries_file}")

try:
    extrusion = SimpleWallExtrusion()
    geometry = extrusion.process(
        str(boundaries_file), 
        "test_output/geometry_3d_v2.json"
    )
    
    print(f"\n✓ Extrusion successful!")
    print(f"  Vertices: {geometry['metadata']['num_vertices']}")
    print(f"  Faces: {geometry['metadata']['num_faces']}")
    print(f"  Room height: {geometry['metadata']['room_height']}m")
    print(f"  Scale factor: {geometry['metadata']['scale_factor']} m/pixel")
    print(f"  Rooms processed: {len(geometry['room_mapping'])}")
    
    # Geometry validation
    num_verts = len(geometry['vertices'])
    num_faces = len(geometry['faces'])
    
    print(f"\nGeometry validation:")
    print(f"  Total vertices: {num_verts}")
    print(f"  Total faces: {num_faces}")
    
    # Check for valid face indices
    invalid_faces = 0
    for face in geometry['faces']:
        if any(idx < 0 or idx >= num_verts for idx in face):
            invalid_faces += 1
    
    print(f"  Invalid face indices: {invalid_faces}")
    
    if invalid_faces == 0:
        print(f"\n✓ All geometry is valid!")
    else:
        print(f"\n✗ {invalid_faces} faces have invalid vertex indices")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
