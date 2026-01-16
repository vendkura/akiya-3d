"""
Wall-Based Extrusion - Dollhouse Style 3D Floor Plans
======================================================

Creates 3D floor plans where:
- WALLS are solid geometry (with thickness)
- ROOMS are empty spaces (voids you can see into)
- FLOOR is a solid plane at the bottom
- CEILING is open (dollhouse view)

This produces architectural-style 3D models where you can
look inside and see each room clearly defined.

Author: Asheleyine's Master thesis project
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class WallSegment:
    """A single wall segment with start/end points."""
    p1: Tuple[float, float]  # Start point (x, y)
    p2: Tuple[float, float]  # End point (x, y)
    is_exterior: bool  # True if this is an exterior wall
    room_ids: List[int]  # IDs of rooms on either side


@dataclass
class WallGeometry:
    """3D geometry for a wall segment with thickness."""
    vertices: np.ndarray  # Nx3 array of vertices
    faces: List[List[int]]  # Face indices


# ============================================================================
# WALL DETECTOR
# ============================================================================

class WallDetector:
    """
    Detects walls from segmentation mask.
    
    Walls are found at:
    1. Boundaries between different rooms
    2. Outer boundary of the entire floor plan
    """
    
    def __init__(self, wall_thickness: float = 0.12, min_wall_length: float = 0.3):
        """
        Initialize wall detector.
        
        Args:
            wall_thickness: Wall thickness in meters
            min_wall_length: Minimum wall length in meters (filters small fragments)
        """
        self.wall_thickness = wall_thickness
        self.min_wall_length = min_wall_length
        self.walls: List[WallSegment] = []
        
    def detect_from_mask(self, 
                         mask: np.ndarray, 
                         scale_factor: float) -> List[WallSegment]:
        """
        Detect wall segments from segmentation mask.
        
        Args:
            mask: Segmentation mask (H, W) with room class IDs
            scale_factor: Meters per pixel
        
        Returns:
            List of WallSegment objects
        """
        self.walls = []
        h, w = mask.shape
        
        # Create binary mask of all non-background pixels
        floorplan_mask = (mask > 0).astype(np.uint8)
        
        # Find exterior boundary (outer walls)
        exterior_contours, _ = cv2.findContours(
            floorplan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in exterior_contours:
            if cv2.contourArea(contour) < 1000:
                continue
            segments = self._contour_to_segments(contour, scale_factor, is_exterior=True)
            self.walls.extend(segments)
        
        # Find interior walls (boundaries between different room classes)
        interior_walls = self._detect_interior_walls(mask, scale_factor)
        self.walls.extend(interior_walls)
        
        logger.info(f"Detected {len(self.walls)} wall segments")
        return self.walls
    
    def _contour_to_segments(self, 
                             contour: np.ndarray, 
                             scale_factor: float,
                             is_exterior: bool = False) -> List[WallSegment]:
        """Convert contour to wall segments."""
        segments = []
        points = contour.squeeze()
        
        if len(points.shape) == 1:
            return segments
        
        n = len(points)
        for i in range(n):
            p1 = points[i] * scale_factor
            p2 = points[(i + 1) % n] * scale_factor
            
            # Skip walls shorter than minimum length
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            if length < self.min_wall_length:
                continue
            
            segments.append(WallSegment(
                p1=(float(p1[0]), float(p1[1])),
                p2=(float(p2[0]), float(p2[1])),
                is_exterior=is_exterior,
                room_ids=[]
            ))
        
        return segments
    
    def _detect_interior_walls(self, 
                               mask: np.ndarray, 
                               scale_factor: float) -> List[WallSegment]:
        """
        Detect walls between rooms (interior walls).
        
        Uses edge detection between different class regions.
        """
        segments = []
        
        # Find edges where adjacent pixels have different non-zero values
        # This indicates a boundary between two rooms
        
        # Horizontal edges
        h_diff = np.abs(mask[:-1, :].astype(int) - mask[1:, :].astype(int))
        h_edges = (h_diff > 0) & (mask[:-1, :] > 0) & (mask[1:, :] > 0)
        
        # Vertical edges
        v_diff = np.abs(mask[:, :-1].astype(int) - mask[:, 1:].astype(int))
        v_edges = (v_diff > 0) & (mask[:, :-1] > 0) & (mask[:, 1:] > 0)
        
        # Convert edge pixels to line segments
        # Group connected edge pixels into lines
        h_segments = self._edges_to_segments(h_edges, scale_factor, horizontal=True)
        v_segments = self._edges_to_segments(v_edges, scale_factor, horizontal=False)
        
        segments.extend(h_segments)
        segments.extend(v_segments)
        
        return segments
    
    def _edges_to_segments(self, 
                           edge_mask: np.ndarray, 
                           scale_factor: float,
                           horizontal: bool) -> List[WallSegment]:
        """Convert edge mask to line segments."""
        segments = []
        
        # Find contours in edge mask
        edge_uint8 = (edge_mask * 255).astype(np.uint8)
        
        # Dilate slightly to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edge_uint8 = cv2.dilate(edge_uint8, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edge_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) < 2:
                continue
            
            # Simplify contour to line segments
            epsilon = 0.02 * cv2.arcLength(contour, False)
            approx = cv2.approxPolyDP(contour, epsilon, False)
            
            points = approx.squeeze()
            if len(points.shape) == 1:
                continue
            
            for i in range(len(points) - 1):
                p1 = points[i] * scale_factor
                p2 = points[i + 1] * scale_factor
                
                length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                if length < self.min_wall_length:  # Use configurable min length
                    continue
                
                segments.append(WallSegment(
                    p1=(float(p1[0]), float(p1[1])),
                    p2=(float(p2[0]), float(p2[1])),
                    is_exterior=False,
                    room_ids=[]
                ))
        
        return segments


# ============================================================================
# WALL-BASED EXTRUDER
# ============================================================================

class WallBasedExtruder:
    """
    Creates 3D geometry from wall segments.
    
    Each wall becomes a 3D box with:
    - Thickness (width)
    - Height (floor to ceiling)
    - Proper orientation based on wall direction
    """
    
    def __init__(self,
                 wall_thickness: float = 0.12,
                 wall_height: float = 2.5,
                 floor_thickness: float = 0.1):
        """
        Initialize extruder.
        
        Args:
            wall_thickness: Wall thickness in meters (default 12cm)
            wall_height: Wall height in meters (default 2.5m)
            floor_thickness: Floor slab thickness in meters
        """
        self.wall_thickness = wall_thickness
        self.wall_height = wall_height
        self.floor_thickness = floor_thickness
        
        # Output geometry
        self.vertices: List[List[float]] = []
        self.faces: List[List[int]] = []
        self.colors: List[List[float]] = []
        
        # Color scheme
        self.wall_color = [0.9, 0.9, 0.9]  # Light gray walls
        self.floor_color = [0.8, 0.7, 0.5]  # Tan/wood floor
        
    def extrude_walls(self, walls: List[WallSegment]) -> Dict:
        """
        Extrude all wall segments to 3D.
        
        Args:
            walls: List of WallSegment objects
        
        Returns:
            Dict with vertices, faces, colors
        """
        self.vertices = []
        self.faces = []
        self.colors = []
        
        logger.info(f"Extruding {len(walls)} wall segments...")
        
        for wall in walls:
            self._extrude_single_wall(wall)
        
        logger.info(f"  Created {len(self.vertices)} vertices, {len(self.faces)} faces")
        
        return {
            "vertices": np.array(self.vertices),
            "faces": np.array(self.faces),
            "colors": np.array(self.colors)
        }
    
    def _extrude_single_wall(self, wall: WallSegment):
        """
        Extrude a single wall segment to a 3D box.
        
        Creates a box along the wall line with proper thickness.
        """
        p1 = np.array(wall.p1)
        p2 = np.array(wall.p2)
        
        # Wall direction vector
        direction = p2 - p1
        length = np.linalg.norm(direction)
        
        if length < 0.01:
            return
        
        direction = direction / length
        
        # Perpendicular vector (for wall thickness)
        perp = np.array([-direction[1], direction[0]])
        
        # Half thickness offset
        offset = perp * (self.wall_thickness / 2)
        
        # Four corners of wall base (at floor level, y=0)
        c1 = p1 - offset  # Bottom-left
        c2 = p1 + offset  # Bottom-right
        c3 = p2 + offset  # Top-right
        c4 = p2 - offset  # Top-left
        
        # Create 8 vertices for the wall box
        # Bottom 4 (y = 0)
        v0 = [c1[0], 0.0, c1[1]]
        v1 = [c2[0], 0.0, c2[1]]
        v2 = [c3[0], 0.0, c3[1]]
        v3 = [c4[0], 0.0, c4[1]]
        
        # Top 4 (y = wall_height)
        v4 = [c1[0], self.wall_height, c1[1]]
        v5 = [c2[0], self.wall_height, c2[1]]
        v6 = [c3[0], self.wall_height, c3[1]]
        v7 = [c4[0], self.wall_height, c4[1]]
        
        # Add vertices
        base_idx = len(self.vertices)
        self.vertices.extend([v0, v1, v2, v3, v4, v5, v6, v7])
        
        # Add colors for each vertex
        for _ in range(8):
            self.colors.append(self.wall_color)
        
        # Create faces (triangles)
        # Each face of the box needs 2 triangles
        
        # Bottom face (y = 0) - not needed for dollhouse view
        # self.faces.append([base_idx + 0, base_idx + 2, base_idx + 1])
        # self.faces.append([base_idx + 0, base_idx + 3, base_idx + 2])
        
        # Top face (y = wall_height)
        self.faces.append([base_idx + 4, base_idx + 5, base_idx + 6])
        self.faces.append([base_idx + 4, base_idx + 6, base_idx + 7])
        
        # Front face
        self.faces.append([base_idx + 0, base_idx + 1, base_idx + 5])
        self.faces.append([base_idx + 0, base_idx + 5, base_idx + 4])
        
        # Back face
        self.faces.append([base_idx + 2, base_idx + 3, base_idx + 7])
        self.faces.append([base_idx + 2, base_idx + 7, base_idx + 6])
        
        # Left face
        self.faces.append([base_idx + 0, base_idx + 4, base_idx + 7])
        self.faces.append([base_idx + 0, base_idx + 7, base_idx + 3])
        
        # Right face
        self.faces.append([base_idx + 1, base_idx + 2, base_idx + 6])
        self.faces.append([base_idx + 1, base_idx + 6, base_idx + 5])
    
    def add_floor(self, mask: np.ndarray, scale_factor: float):
        """
        Add floor plane covering the entire floorplan.
        
        Args:
            mask: Segmentation mask to determine floor extent
            scale_factor: Meters per pixel
        """
        # Find bounding box of floorplan
        non_zero = np.argwhere(mask > 0)
        
        if len(non_zero) == 0:
            return
        
        min_y, min_x = non_zero.min(axis=0)
        max_y, max_x = non_zero.max(axis=0)
        
        # Add padding
        padding = 5
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(mask.shape[1], max_x + padding)
        max_y = min(mask.shape[0], max_y + padding)
        
        # Convert to world coordinates
        x1 = min_x * scale_factor
        z1 = min_y * scale_factor
        x2 = max_x * scale_factor
        z2 = max_y * scale_factor
        
        # Floor at y = 0
        y = 0.0
        
        # Create floor vertices
        base_idx = len(self.vertices)
        self.vertices.extend([
            [x1, y, z1],  # 0: bottom-left
            [x2, y, z1],  # 1: bottom-right
            [x2, y, z2],  # 2: top-right
            [x1, y, z2],  # 3: top-left
        ])
        
        # Floor color
        for _ in range(4):
            self.colors.append(self.floor_color)
        
        # Floor face (facing up)
        self.faces.append([base_idx + 0, base_idx + 1, base_idx + 2])
        self.faces.append([base_idx + 0, base_idx + 2, base_idx + 3])
        
        logger.info(f"  Added floor plane: {x2-x1:.1f}m x {z2-z1:.1f}m")
    
    def add_floor_from_contour(self, mask: np.ndarray, scale_factor: float):
        """
        Add floor that follows the actual floorplan shape (not just bounding box).
        
        Args:
            mask: Segmentation mask
            scale_factor: Meters per pixel
        """
        # Create binary mask
        binary = (mask > 0).astype(np.uint8)
        
        # Find outer contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Simplify
        epsilon = 0.01 * cv2.arcLength(largest, True)
        simplified = cv2.approxPolyDP(largest, epsilon, True)
        
        points = simplified.squeeze() * scale_factor
        
        if len(points.shape) == 1:
            return
        
        # Triangulate the floor polygon
        # Using fan triangulation (works for convex, approximate for non-convex)
        base_idx = len(self.vertices)
        
        # Add vertices
        for p in points:
            self.vertices.append([float(p[0]), 0.0, float(p[1])])
            self.colors.append(self.floor_color)
        
        # Triangulate using ear clipping (simple fan for now)
        n = len(points)
        for i in range(1, n - 1):
            self.faces.append([base_idx, base_idx + i, base_idx + i + 1])
        
        logger.info(f"  Added floor with {n} vertices")
    
    def add_colored_floor(self, mask: np.ndarray, scale_factor: float):
        """
        Add floor plane with colors based on room types.
        
        Each room class gets its own color on the floor.
        
        Args:
            mask: Segmentation mask with class IDs
            scale_factor: Meters per pixel
        """
        # Color mapping for room types (RGB normalized)
        room_colors = {
            0: [0.2, 0.2, 0.2],      # background - dark gray
            1: [0.95, 0.75, 0.8],    # dining_area - light pink
            2: [0.7, 0.85, 0.95],    # bathroom - light blue
            3: [0.8, 0.9, 0.95],     # bedroom - pale blue
            4: [0.95, 0.9, 0.75],    # closet - beige
            5: [0.95, 0.95, 0.8],    # room - light yellow
            6: [0.55, 0.35, 0.2],    # door - brown
            7: [0.9, 0.95, 0.9],     # entrance - honeydew
            8: [0.6, 0.9, 0.6],      # kitchen - light green
            9: [0.65, 0.95, 0.65],   # outdoor_space - pale green
            10: [0.6, 0.6, 0.6],     # sliding_door - gray
            11: [0.75, 0.75, 0.75],  # stairs - light gray
            12: [0.6, 0.8, 0.95],    # window - sky blue
            13: [0.75, 0.88, 0.9],   # balcony - powder blue
        }
        
        # Process each room class
        for class_id in range(1, 14):
            # Get binary mask for this class
            class_mask = (mask == class_id).astype(np.uint8)
            
            if class_mask.sum() == 0:
                continue
            
            # Find contours for this class
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 500:  # Skip tiny regions
                    continue
                
                # Simplify contour
                epsilon = 0.01 * cv2.arcLength(contour, True)
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                
                points = simplified.squeeze() * scale_factor
                
                if len(points.shape) == 1 or len(points) < 3:
                    continue
                
                # Get color for this room type
                color = room_colors.get(class_id, [0.8, 0.8, 0.8])
                
                # Add floor polygon for this room
                base_idx = len(self.vertices)
                
                for p in points:
                    self.vertices.append([float(p[0]), 0.0, float(p[1])])
                    self.colors.append(color)
                
                # Triangulate (fan triangulation)
                n = len(points)
                for i in range(1, n - 1):
                    self.faces.append([base_idx, base_idx + i, base_idx + i + 1])
        
        logger.info(f"  Added colored floor for {len(np.unique(mask)) - 1} room types")
    
    def get_geometry(self) -> Dict:
        """Get complete geometry."""
        return {
            "vertices": np.array(self.vertices),
            "faces": np.array(self.faces),
            "colors": np.array(self.colors)
        }
    
    def get_bounds(self) -> Dict:
        """Get bounding box."""
        if not self.vertices:
            return {"min": [0, 0, 0], "max": [0, 0, 0], "size": [0, 0, 0]}
        
        verts = np.array(self.vertices)
        min_pt = verts.min(axis=0)
        max_pt = verts.max(axis=0)
        
        return {
            "min": min_pt.tolist(),
            "max": max_pt.tolist(),
            "size": (max_pt - min_pt).tolist()
        }


# ============================================================================
# IMPROVED WALL DETECTOR (From Room Polygons)
# ============================================================================

class RoomBasedWallDetector:
    """
    Detects walls from room polygons rather than raw mask.
    
    This is more accurate because it uses the cleaned/rectangularized
    room boundaries we've already computed.
    """
    
    def __init__(self, wall_thickness: float = 0.12):
        self.wall_thickness = wall_thickness
        self.walls: List[WallSegment] = []
        
    def detect_from_rooms(self, 
                          rooms: List[Dict],
                          scale_factor: float) -> List[WallSegment]:
        """
        Detect walls from room polygon data.
        
        Args:
            rooms: List of room dicts with 'vertices' key
            scale_factor: Already applied to vertices
        
        Returns:
            List of WallSegment objects
        """
        self.walls = []
        all_edges: Set[Tuple] = set()
        
        for room in rooms:
            vertices = room.get('vertices', [])
            if len(vertices) < 3:
                continue
            
            # Extract edges from this room
            n = len(vertices)
            for i in range(n):
                p1 = tuple(vertices[i])
                p2 = tuple(vertices[(i + 1) % n])
                
                # Normalize edge direction (smaller point first)
                if p1 > p2:
                    p1, p2 = p2, p1
                
                edge = (p1, p2)
                
                # Check if this edge already exists (shared wall)
                if edge in all_edges:
                    # This is an interior wall (shared between rooms)
                    # We might want to skip it or mark it differently
                    pass
                else:
                    all_edges.add(edge)
                    
                    self.walls.append(WallSegment(
                        p1=p1,
                        p2=p2,
                        is_exterior=False,  # Will determine later
                        room_ids=[]
                    ))
        
        logger.info(f"Detected {len(self.walls)} wall segments from {len(rooms)} rooms")
        return self.walls


# ============================================================================
# OBJ EXPORTER
# ============================================================================

class WallOBJExporter:
    """Export wall-based geometry to OBJ format."""
    
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.colors = []
    
    def load_geometry(self, geometry: Dict):
        """Load geometry from extruder."""
        self.vertices = geometry.get("vertices", [])
        self.faces = geometry.get("faces", [])
        self.colors = geometry.get("colors", [])
        
        if isinstance(self.vertices, np.ndarray):
            self.vertices = self.vertices.tolist()
        if isinstance(self.faces, np.ndarray):
            self.faces = self.faces.tolist()
        if isinstance(self.colors, np.ndarray):
            self.colors = self.colors.tolist()
    
    def export_obj(self, obj_path: str, mtl_filename: str = "model.mtl"):
        """Export to OBJ file."""
        with open(obj_path, 'w') as f:
            f.write("# Wall-Based Floorplan 3D Model\n")
            f.write(f"# Vertices: {len(self.vertices)}\n")
            f.write(f"# Faces: {len(self.faces)}\n")
            f.write(f"mtllib {mtl_filename}\n\n")
            
            # Write vertices with colors
            for i, v in enumerate(self.vertices):
                if i < len(self.colors):
                    c = self.colors[i]
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
                else:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            f.write("\n")
            f.write("usemtl walls\n")
            
            # Write faces (1-indexed)
            for face in self.faces:
                face_str = " ".join(str(idx + 1) for idx in face)
                f.write(f"f {face_str}\n")
        
        logger.info(f"Exported OBJ to {obj_path}")
    
    def export_mtl(self, mtl_path: str):
        """Export MTL material file."""
        with open(mtl_path, 'w') as f:
            f.write("# Material file for Wall-Based Floorplan\n\n")
            
            f.write("newmtl walls\n")
            f.write("Ka 0.2 0.2 0.2\n")
            f.write("Kd 0.9 0.9 0.9\n")
            f.write("Ks 0.1 0.1 0.1\n")
            f.write("Ns 10.0\n")
            f.write("d 1.0\n\n")
            
            f.write("newmtl floor\n")
            f.write("Ka 0.2 0.18 0.12\n")
            f.write("Kd 0.8 0.7 0.5\n")
            f.write("Ks 0.05 0.05 0.05\n")
            f.write("Ns 5.0\n")
            f.write("d 1.0\n")
        
        logger.info(f"Exported MTL to {mtl_path}")


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def create_dollhouse_model(mask: np.ndarray,
                           scale_factor: float,
                           output_dir: str,
                           output_name: str = "floorplan",
                           wall_thickness: float = 0.12,
                           wall_height: float = 2.5) -> Dict:
    """
    Create dollhouse-style 3D model from segmentation mask.
    
    Args:
        mask: Segmentation mask
        scale_factor: Meters per pixel
        output_dir: Output directory
        output_name: Base name for output files
        wall_thickness: Wall thickness in meters
        wall_height: Wall height in meters
    
    Returns:
        Dict with file paths and geometry info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect walls
    detector = WallDetector(wall_thickness=wall_thickness)
    walls = detector.detect_from_mask(mask, scale_factor)
    
    # Extrude walls
    extruder = WallBasedExtruder(
        wall_thickness=wall_thickness,
        wall_height=wall_height
    )
    extruder.extrude_walls(walls)
    
    # Add floor
    extruder.add_floor_from_contour(mask, scale_factor)
    
    # Get geometry
    geometry = extruder.get_geometry()
    bounds = extruder.get_bounds()
    
    # Export
    exporter = WallOBJExporter()
    exporter.load_geometry(geometry)
    
    obj_path = output_dir / f"{output_name}.obj"
    mtl_path = output_dir / f"{output_name}.mtl"
    
    exporter.export_obj(str(obj_path), f"{output_name}.mtl")
    exporter.export_mtl(str(mtl_path))
    
    return {
        "obj_path": str(obj_path),
        "mtl_path": str(mtl_path),
        "bounds": bounds,
        "num_walls": len(walls),
        "num_vertices": len(geometry["vertices"]),
        "num_faces": len(geometry["faces"])
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create dollhouse-style 3D from mask")
    parser.add_argument("--mask", "-m", required=True, help="Input segmentation mask")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--name", "-n", default="floorplan", help="Output name")
    parser.add_argument("--scale", type=float, default=0.01, help="Scale factor (m/px)")
    parser.add_argument("--wall-thickness", type=float, default=0.12, help="Wall thickness (m)")
    parser.add_argument("--wall-height", type=float, default=2.5, help="Wall height (m)")
    
    args = parser.parse_args()
    
    # Load mask
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Cannot load {args.mask}")
        exit(1)
    
    # Create model
    result = create_dollhouse_model(
        mask=mask,
        scale_factor=args.scale,
        output_dir=args.output,
        output_name=args.name,
        wall_thickness=args.wall_thickness,
        wall_height=args.wall_height
    )
    
    print(f"\n✓ Created dollhouse model:")
    print(f"  OBJ: {result['obj_path']}")
    print(f"  Walls: {result['num_walls']}")
    print(f"  Bounds: {result['bounds']['size']}")