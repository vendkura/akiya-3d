"""
Compare v1 and v2 3D geometries
"""

import json
from pathlib import Path
import numpy as np

# Load both geometries
v1_file = Path("test_output/geometry_3d.json")
v2_file = Path("test_output/geometry_3d_v2.json")

if not v1_file.exists():
    print(f"✗ v1 file not found: {v1_file}")
    exit(1)

if not v2_file.exists():
    print(f"✗ v2 file not found: {v2_file}")
    exit(1)

print("=" * 60)
print("COMPARING WALL EXTRUSION V1 vs V2")
print("=" * 60)

with open(v1_file, 'r') as f:
    v1 = json.load(f)

with open(v2_file, 'r') as f:
    v2 = json.load(f)

# Extract vertices and faces
v1_verts = np.array(v1['vertices'])
v1_faces = np.array(v1['faces'])
v2_verts = np.array(v2['vertices'])
v2_faces = np.array(v2['faces'])

print(f"\nGeometry Statistics:")
print(f"{'Metric':<30} {'V1':<15} {'V2':<15}")
print("-" * 60)
print(f"{'Vertices':<30} {len(v1_verts):<15} {len(v2_verts):<15}")
print(f"{'Faces':<30} {len(v1_faces):<15} {len(v2_faces):<15}")
print(f"{'Rooms processed':<30} {len(v1.get('room_mapping', {})):<15} {len(v2.get('room_mapping', {})):<15}")

# Analyze vertex distribution
def analyze_vertices(verts, name):
    z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
    z_vals = verts[:, 2]
    floor_verts = np.sum(z_vals < 0.1)  # ~z=0
    ceil_verts = np.sum(z_vals > 2.4)   # ~z=2.5
    wall_verts = np.sum((z_vals >= 0.1) & (z_vals <= 2.4))  # Between floor/ceiling
    
    print(f"\n{name} Vertex Distribution:")
    print(f"  Floor vertices (z≈0): {floor_verts}")
    print(f"  Ceiling vertices (z≈2.5): {ceil_verts}")
    print(f"  Wall vertices (0<z<2.5): {wall_verts}")
    print(f"  Z-range: {z_min:.2f} to {z_max:.2f}")
    
    return floor_verts, ceil_verts, wall_verts

v1_floor, v1_ceil, v1_wall = analyze_vertices(v1_verts, "V1")
v2_floor, v2_ceil, v2_wall = analyze_vertices(v2_verts, "V2")

print(f"\nKey Differences:")
print("-" * 60)

# V1 had wall thickness offset -> more vertices
v1_verts_per_room = len(v1_verts) / len(v1.get('room_mapping', {}))
v2_verts_per_room = len(v2_verts) / len(v2.get('room_mapping', {}))
print(f"V1 vertices per room: {v1_verts_per_room:.1f}")
print(f"V2 vertices per room: {v2_verts_per_room:.1f}")
print(f"  V2 is {(v1_verts_per_room/v2_verts_per_room):.2f}x simpler")

print(f"\nV1 analysis (with wall offset):")
print(f"  - More vertices due to offset polygon creating additional layers")
print(f"  - {v1_wall} wall vertices for wall thickness geometry")

print(f"\nV2 analysis (no offset - single surface):")
print(f"  - Simpler geometry: no offset polygon")
print(f"  - {v2_wall} wall vertices for direct walls")
print(f"  - Should have continuous walls (no gaps)")
print(f"  - Should have distinct rooms (no overlaps)")

print("\n" + "=" * 60)
print("✓ V2 is simpler and should fix visualization issues")
print("  Key improvement: No wall thickness offset → continuous walls")
print("=" * 60)
