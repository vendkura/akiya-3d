"""
Interactive 3D visualization using Plotly
Creates an HTML file you can open in browser to rotate/zoom the 3D model
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

# Try to import plotly
try:
    import plotly.graph_objects as go
    from plotly.offline import plot
    print("✓ Plotly available")
except ImportError:
    print("❌ Plotly not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "-q"])
    import plotly.graph_objects as go
    from plotly.offline import plot

# Load geometry
geometry_file = Path(output_base) / "test_output" / "geometry_3d.json"
print(f"Loading geometry from {geometry_file}...")

with open(geometry_file, 'r') as f:
    geometry = json.load(f)

vertices = np.array(geometry['vertices'])
faces = np.array(geometry['faces'])
colors = np.array(geometry['colors'])

print(f"Loaded {len(vertices)} vertices, {len(faces)} faces")

# Create plotly figure
fig = go.Figure()

# Create mesh3d plot
# For Plotly, we need to extract vertex coordinates and face connectivity
x = vertices[:, 0].tolist()
y = vertices[:, 1].tolist()
z = vertices[:, 2].tolist()

# Convert RGB (0-1) to hex for each vertex
vertex_colors_hex = []
for color in colors:
    r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
    hex_color = f'rgb({r},{g},{b})'
    vertex_colors_hex.append(hex_color)

# Extract face vertex indices
i_indices = []
j_indices = []
k_indices = []

for face in faces:
    i_indices.append(face[0])
    j_indices.append(face[1])
    k_indices.append(face[2])

# Create mesh surface
fig.add_trace(go.Mesh3d(
    x=x, y=y, z=z,
    i=i_indices, j=j_indices, k=k_indices,
    vertexcolor=vertex_colors_hex,
    opacity=0.9,
    name='Building Model',
    showlegend=False,
    hovertemplate='<b>Vertex</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<extra></extra>'
))

# Update layout
fig.update_layout(
    title='Floor Plan 3D Model - Interactive View',
    scene=dict(
        xaxis=dict(title='X (meters)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        yaxis=dict(title='Y (meters)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        zaxis=dict(title='Z (meters)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)  # Isometric view
        )
    ),
    width=1200,
    height=800,
    showlegend=True,
    hovermode='closest',
)

# Save as HTML
output_html = Path(output_base) / "test_output" / "3d_model_interactive.html"
fig.write_html(str(output_html))

print(f"\n✅ Interactive 3D visualization created!")
print(f"📄 Open this file in your browser: {output_html}")
print(f"\nFeatures:")
print(f"  - Rotate: Click and drag")
print(f"  - Zoom: Scroll wheel")
print(f"  - Pan: Right-click and drag")
print(f"  - Hover: See vertex coordinates")
