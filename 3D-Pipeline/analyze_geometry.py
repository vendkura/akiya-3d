"""
Detailed geometry analysis script
"""

import json
import numpy as np
from pathlib import Path
import sys
import os

if os.path.exists("/mnt/e"):
    output_base = "/mnt/e/github.com/akiya-3d-thesis/3D-Pipeline"
else:
    output_base = r"E:\github.com\akiya-3d-thesis\3D-Pipeline"

# Load geometry
geometry_file = Path(output_base) / "test_output" / "geometry_3d.json"

with open(geometry_file, 'r') as f:
    geometry = json.load(f)

print("="*70)
print("DETAILED GEOMETRY ANALYSIS")
print("="*70)

vertices = np.array(geometry['vertices'])
faces = np.array(geometry['faces'])
normals = np.array(geometry['normals'])
colors = np.array(geometry['colors'])

print(f"\n📊 GEOMETRY DIMENSIONS:")
print(f"  Total vertices: {len(vertices)}")
print(f"  Total faces: {len(faces)}")
print(f"  Total normals: {len(normals)}")
print(f"  Total colors: {len(colors)}")

print(f"\n📐 COORDINATE RANGES:")
print(f"  X range: {vertices[:, 0].min():.2f} to {vertices[:, 0].max():.2f} meters")
print(f"  Y range: {vertices[:, 1].min():.2f} to {vertices[:, 1].max():.2f} meters")
print(f"  Z range: {vertices[:, 2].min():.2f} to {vertices[:, 2].max():.2f} meters")

print(f"\n📍 BUILDING DIMENSIONS:")
width = vertices[:, 0].max() - vertices[:, 0].min()
depth = vertices[:, 1].max() - vertices[:, 1].min()
height = vertices[:, 2].max() - vertices[:, 2].min()
print(f"  Width (X): {width:.2f} m")
print(f"  Depth (Y): {depth:.2f} m")
print(f"  Height (Z): {height:.2f} m")
print(f"  Floor area: {width * depth:.2f} m²")

print(f"\n🔍 FACE ANALYSIS:")
print(f"  Min vertex index in faces: {faces.min()}")
print(f"  Max vertex index in faces: {faces.max()}")
print(f"  Total vertices in geometry: {len(vertices)}")

# Check for degenerate faces (zero area)
degenerate_count = 0
for face in faces:
    v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
    if area < 1e-6:
        degenerate_count += 1

print(f"  Degenerate faces (area < 1e-6): {degenerate_count}")

print(f"\n🎨 COLOR ANALYSIS:")
unique_colors = set(tuple(c) for c in colors)
print(f"  Unique colors: {len(unique_colors)}")
color_freq = {}
for color in colors:
    color_tuple = tuple(np.round(color, 3))
    color_freq[color_tuple] = color_freq.get(color_tuple, 0) + 1

print(f"  Top 10 colors by frequency:")
sorted_colors = sorted(color_freq.items(), key=lambda x: x[1], reverse=True)[:10]
for i, (color, count) in enumerate(sorted_colors, 1):
    print(f"    {i}. RGB{color} - {count} vertices")

print(f"\n🏠 ROOM MAPPING:")
rooms = geometry['room_mapping']
print(f"  Total rooms: {len(rooms)}")
print(f"  Vertices per room:")
room_vertex_counts = [len(v) for v in rooms.values()]
print(f"    Min: {min(room_vertex_counts)}")
print(f"    Max: {max(room_vertex_counts)}")
print(f"    Average: {np.mean(room_vertex_counts):.1f}")

print(f"\n⚠️  POTENTIAL ISSUES TO CHECK:")

# Check for overlapping Z coordinates (rooms on top of each other)
floor_vertices = vertices[vertices[:, 2] < 0.1]
ceiling_vertices = vertices[vertices[:, 2] > 2.4]
print(f"  Floor vertices (z < 0.1m): {len(floor_vertices)}")
print(f"  Ceiling vertices (z > 2.4m): {len(ceiling_vertices)}")

# Check for isolated vertices
used_vertices = set()
for face in faces:
    used_vertices.update(face)
orphaned = len(vertices) - len(used_vertices)
print(f"  Orphaned vertices (not in faces): {orphaned}")

# Check normal directions
up_normals = np.sum(normals[:, 2] > 0.9)
down_normals = np.sum(normals[:, 2] < -0.9)
print(f"  Normals pointing up (ceiling): {up_normals}")
print(f"  Normals pointing down (floor): {down_normals}")

print(f"\n✓ SUMMARY:")
if faces.max() < len(vertices) and orphaned == 0 and degenerate_count == 0:
    print("  ✅ Geometry appears valid")
else:
    print("  ⚠️  Potential issues detected - review above")

print("\n" + "="*70)
