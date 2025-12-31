"""
Visualization script for 3D geometry from wall extrusion

Creates interactive 3D plots and static images to verify geometry correctness.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeometryVisualizer:
    """Visualize 3D geometry from wall extrusion."""
    
    def __init__(self, geometry_dict: dict):
        """
        Initialize visualizer.
        
        Args:
            geometry_dict: Dict with vertices, faces, normals, colors from wall_extrusion
        """
        self.vertices = np.array(geometry_dict["vertices"])
        self.faces = np.array(geometry_dict["faces"])
        self.normals = np.array(geometry_dict["normals"])
        self.colors = np.array(geometry_dict["colors"])
        self.room_mapping = geometry_dict.get("room_mapping", {})
        self.metadata = geometry_dict.get("metadata", {})
        
        logger.info(f"Loaded geometry: {len(self.vertices)} vertices, {len(self.faces)} faces")
    
    def get_face_vertices(self, face_idx: int):
        """
        Get 3D coordinates of vertices for a face.
        
        Args:
            face_idx: Index of face
        
        Returns:
            3x3 array of [x, y, z] coordinates
        """
        face = self.faces[face_idx]
        return self.vertices[face]
    
    def get_face_color(self, face_idx: int):
        """
        Get average color for a face (from vertex colors).
        
        Args:
            face_idx: Index of face
        
        Returns:
            RGB color tuple (0-1 range)
        """
        face = self.faces[face_idx]
        # Average color of the three vertices
        colors = self.colors[face]
        avg_color = np.mean(colors, axis=0)
        return tuple(avg_color)
    
    def plot_3d_static(self, output_path: str = "geometry_3d.png", 
                       view_angle: str = "isometric"):
        """
        Create static 3D plot using Matplotlib.
        
        Args:
            output_path: Path to save PNG
            view_angle: 'isometric', 'top', 'front', 'side'
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        logger.info(f"Creating 3D plot ({view_angle} view)...")
        
        # Collect faces with colors
        face_list = []
        face_colors = []
        
        for face_idx in range(len(self.faces)):
            face_verts = self.get_face_vertices(face_idx)
            face_color = self.get_face_color(face_idx)
            
            face_list.append(face_verts)
            face_colors.append(face_color)
        
        # Create poly collection
        poly = Poly3DCollection(face_list, alpha=0.8, edgecolor='black', linewidth=0.2)
        poly.set_facecolor(face_colors)
        ax.add_collection3d(poly)
        
        # Set limits
        ax.set_xlim([self.vertices[:, 0].min(), self.vertices[:, 0].max()])
        ax.set_ylim([self.vertices[:, 1].min(), self.vertices[:, 1].max()])
        ax.set_zlim([self.vertices[:, 2].min(), self.vertices[:, 2].max()])
        
        # Labels
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title(f'3D Floor Plan Model - {view_angle.title()} View')
        
        # Set view angle
        if view_angle == "isometric":
            ax.view_init(elev=20, azim=45)
        elif view_angle == "top":
            ax.view_init(elev=90, azim=0)
        elif view_angle == "front":
            ax.view_init(elev=0, azim=0)
        elif view_angle == "side":
            ax.view_init(elev=0, azim=90)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved {view_angle} view to {output_path}")
        plt.close()
    
    def plot_3d_multiview(self, output_path: str = "geometry_3d_multiview.png"):
        """
        Create figure with multiple views (isometric, top, front, side).
        
        Args:
            output_path: Path to save PNG
        """
        fig = plt.figure(figsize=(16, 12))
        
        views = [
            (111, "Isometric", 20, 45),
            (222, "Top", 90, 0),
            (223, "Front", 0, 0),
            (224, "Side", 0, 90)
        ]
        
        logger.info("Creating multi-view 3D plot...")
        
        for subplot_idx, view_name, elev, azim in views:
            ax = fig.add_subplot(subplot_idx, projection='3d')
            
            # Collect faces with colors
            face_list = []
            face_colors = []
            
            for face_idx in range(len(self.faces)):
                face_verts = self.get_face_vertices(face_idx)
                face_color = self.get_face_color(face_idx)
                
                face_list.append(face_verts)
                face_colors.append(face_color)
            
            # Create poly collection
            poly = Poly3DCollection(face_list, alpha=0.8, edgecolor='gray', linewidth=0.1)
            poly.set_facecolor(face_colors)
            ax.add_collection3d(poly)
            
            # Set limits
            ax.set_xlim([self.vertices[:, 0].min(), self.vertices[:, 0].max()])
            ax.set_ylim([self.vertices[:, 1].min(), self.vertices[:, 1].max()])
            ax.set_zlim([self.vertices[:, 2].min(), self.vertices[:, 2].max()])
            
            # Labels (only on outer plots to reduce clutter)
            if subplot_idx == 222:
                ax.set_xlabel('X (m)', fontsize=8)
                ax.set_ylabel('Y (m)', fontsize=8)
            if subplot_idx == 223:
                ax.set_xlabel('X (m)', fontsize=8)
                ax.set_zlabel('Z (m)', fontsize=8)
            if subplot_idx == 224:
                ax.set_ylabel('Y (m)', fontsize=8)
                ax.set_zlabel('Z (m)', fontsize=8)
            
            ax.set_title(f'{view_name} View', fontsize=10)
            ax.view_init(elev=elev, azim=azim)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved multi-view to {output_path}")
        plt.close()
    
    def plot_geometry_stats(self, output_path: str = "geometry_stats.png"):
        """
        Create visualization of geometry statistics.
        
        Args:
            output_path: Path to save PNG
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Vertex distribution in Z (height)
        ax = axes[0, 0]
        z_values = self.vertices[:, 2]
        ax.hist(z_values, bins=30, color='skyblue', edgecolor='black')
        ax.set_xlabel('Z Height (meters)')
        ax.set_ylabel('Number of Vertices')
        ax.set_title('Vertex Height Distribution')
        ax.grid(alpha=0.3)
        
        # 2. X-Y top-down view (point cloud)
        ax = axes[0, 1]
        x_values = self.vertices[:, 0]
        y_values = self.vertices[:, 1]
        scatter = ax.scatter(x_values, y_values, c=z_values, cmap='viridis', s=5, alpha=0.6)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Top-Down View (colored by height)')
        ax.axis('equal')
        plt.colorbar(scatter, ax=ax, label='Z (meters)')
        
        # 3. Face normal distribution
        ax = axes[1, 0]
        z_normals = self.normals[:, 2]
        ax.hist(z_normals, bins=30, color='lightcoral', edgecolor='black')
        ax.set_xlabel('Normal Z Component')
        ax.set_ylabel('Number of Normals')
        ax.set_title('Normal Vector Z-Component Distribution')
        ax.grid(alpha=0.3)
        
        # 4. Geometry statistics text
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = f"""
GEOMETRY STATISTICS

Vertices: {len(self.vertices):,}
Faces: {len(self.faces):,}
Normals: {len(self.normals):,}
Colors: {len(self.colors):,}

BOUNDS:
  X: [{self.vertices[:, 0].min():.2f}, {self.vertices[:, 0].max():.2f}] m
  Y: [{self.vertices[:, 1].min():.2f}, {self.vertices[:, 1].max():.2f}] m
  Z: [{self.vertices[:, 2].min():.2f}, {self.vertices[:, 2].max():.2f}] m

ROOM INFORMATION:
  Number of rooms: {len(self.room_mapping)}

EXTRUSION PARAMETERS:
  Room height: {self.metadata.get('room_height', 'N/A')} m
  Wall thickness: {self.metadata.get('wall_thickness', 'N/A')} m
  Scale factor: {self.metadata.get('scale_factor', 'N/A')} m/pixel

MESH QUALITY:
  Min vertices per room: {min(len(v) for v in self.room_mapping.values())} (room {min(self.room_mapping, key=lambda k: len(self.room_mapping[k]))})
  Max vertices per room: {max(len(v) for v in self.room_mapping.values())} (room {max(self.room_mapping, key=lambda k: len(self.room_mapping[k]))})
  Avg vertices per room: {np.mean([len(v) for v in self.room_mapping.values()]):.1f}
        """
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
                fontfamily='monospace', fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved statistics to {output_path}")
        plt.close()
    
    def create_all_visualizations(self, output_dir: str = "./visualizations"):
        """
        Create all visualization outputs.
        
        Args:
            output_dir: Directory to save visualizations
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating visualizations in {output_dir}...")
        
        # Multi-view
        self.plot_3d_multiview(f"{output_dir}/geometry_multiview.png")
        
        # Single high-quality isometric
        self.plot_3d_static(f"{output_dir}/geometry_isometric.png", "isometric")
        
        # Statistics
        self.plot_geometry_stats(f"{output_dir}/geometry_statistics.png")
        
        logger.info(f"✅ All visualizations saved to {output_dir}")


def visualize_from_json(geometry_json_path: str, output_dir: str = None):
    """
    Load geometry JSON and create visualizations.
    
    Args:
        geometry_json_path: Path to geometry JSON from wall_extrusion
        output_dir: Output directory (default: same as input, subdir 'visualizations')
    """
    # Load geometry
    with open(geometry_json_path, 'r') as f:
        geometry = json.load(f)
    
    # Determine output directory
    if not output_dir:
        base_dir = Path(geometry_json_path).parent
        output_dir = str(base_dir / "visualizations")
    
    # Create visualizer and generate plots
    visualizer = GeometryVisualizer(geometry)
    visualizer.create_all_visualizations(output_dir)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_geometry.py <geometry_json> [output_dir]")
        sys.exit(1)
    
    geometry_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    visualize_from_json(geometry_path, output_dir)
