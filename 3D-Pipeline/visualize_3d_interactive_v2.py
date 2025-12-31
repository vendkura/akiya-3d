"""
Visualize 3D geometry (v2) using Plotly interactive viewer
"""

import json
from pathlib import Path
import plotly.graph_objects as go
import numpy as np

# Load geometry
geometry_file = Path("test_output/geometry_3d_v2.json")

if not geometry_file.exists():
    print(f"Error: Geometry file not found: {geometry_file}")
    exit(1)

print(f"Loading geometry from {geometry_file}...")
with open(geometry_file, 'r') as f:
    geometry = json.load(f)

vertices = np.array(geometry['vertices'])
faces = np.array(geometry['faces'])
colors = geometry['colors']

print(f"Vertices: {len(vertices)}")
print(f"Faces: {len(faces)}")

# Create Plotly mesh
fig = go.Figure(data=[go.Mesh3d(
    x=vertices[:, 0],
    y=vertices[:, 1],
    z=vertices[:, 2],
    i=faces[:, 0],
    j=faces[:, 1],
    k=faces[:, 2],
    vertexcolor=[f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},1.0)' for c in colors],
    name='3D Model',
    showlegend=False,
    flatshading=False
)])

fig.update_layout(
    title="3D Floor Plan Model (v2 - Simplified Walls)",
    scene=dict(
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        zaxis_title="Z (meters)",
        aspectmode='data'
    ),
    width=1200,
    height=800
)

output_file = Path("test_output/3d_model_interactive_v2.html")
fig.write_html(str(output_file))
print(f"\n✓ Visualization saved to {output_file}")
print(f"  Open in web browser to inspect the 3D model")
