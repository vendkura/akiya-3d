"""
Wall Extrusion v6 - Robust 3D extrusion with ear-clipping triangulation

Key improvements over v5:
- Ear-clipping triangulation (handles non-convex polygons correctly)
- Polygon validation and repair
- Proper winding order enforcement
- Robust normal calculation
- Better error handling

Author: Asheleyine's Master thesis project
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# EAR CLIPPING TRIANGULATION
# ============================================================================

class EarClipTriangulator:
    """
    Ear-clipping algorithm for triangulating simple polygons.
    
    Works correctly for both convex and non-convex (concave) polygons.
    
    Algorithm:
    1. Find an "ear" - a vertex where the triangle formed with neighbors
       is inside the polygon and contains no other vertices
    2. Cut off the ear (add triangle to result)
    3. Repeat until only 3 vertices remain
    """
    
    @staticmethod
    def triangulate(vertices: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Triangulate a simple polygon using ear clipping.
        
        Args:
            vertices: Nx2 array of polygon vertices in CCW order
            
        Returns:
            List of triangle indices [(i0, i1, i2), ...]
        """
        n = len(vertices)
        
        if n < 3:
            return []
        
        if n == 3:
            return [(0, 1, 2)]
        
        # Ensure CCW winding
        if EarClipTriangulator._signed_area(vertices) < 0:
            vertices = vertices[::-1]
        
        # Create index list (we'll remove vertices as we clip ears)
        indices = list(range(n))
        triangles = []
        
        # Safety counter to prevent infinite loops
        max_iterations = n * n
        iteration = 0
        
        while len(indices) > 3 and iteration < max_iterations:
            iteration += 1
            ear_found = False
            
            for i in range(len(indices)):
                # Get three consecutive vertices
                i0 = indices[(i - 1) % len(indices)]
                i1 = indices[i]
                i2 = indices[(i + 1) % len(indices)]
                
                v0 = vertices[i0]
                v1 = vertices[i1]
                v2 = vertices[i2]
                
                # Check if this is a convex vertex (ear candidate)
                if not EarClipTriangulator._is_convex(v0, v1, v2):
                    continue
                
                # Check if any other vertex is inside this triangle
                is_ear = True
                for j in indices:
                    if j in (i0, i1, i2):
                        continue
                    if EarClipTriangulator._point_in_triangle(vertices[j], v0, v1, v2):
                        is_ear = False
                        break
                
                if is_ear:
                    # Found an ear! Add triangle and remove vertex
                    triangles.append((i0, i1, i2))
                    indices.remove(i1)
                    ear_found = True
                    break
            
            if not ear_found:
                # Fallback: if no ear found (degenerate polygon), 
                # try to salvage by removing smallest triangle
                logger.warning("No ear found, using fallback triangulation")
                if len(indices) >= 3:
                    triangles.append((indices[0], indices[1], indices[2]))
                    indices.pop(1)
        
        # Add final triangle
        if len(indices) == 3:
            triangles.append((indices[0], indices[1], indices[2]))
        
        return triangles
    
    @staticmethod
    def _signed_area(vertices: np.ndarray) -> float:
        """Calculate signed area (positive = CCW, negative = CW)."""
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return area / 2.0
    
    @staticmethod
    def _is_convex(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> bool:
        """Check if vertex v1 is convex (cross product > 0 for CCW polygon)."""
        edge1 = v1 - v0
        edge2 = v2 - v1
        cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
        return cross > 0
    
    @staticmethod
    def _point_in_triangle(p: np.ndarray, 
                           v0: np.ndarray, 
                           v1: np.ndarray, 
                           v2: np.ndarray) -> bool:
        """Check if point p is inside triangle v0-v1-v2."""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign(p, v0, v1)
        d2 = sign(p, v1, v2)
        d3 = sign(p, v2, v0)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)


# ============================================================================
# POLYGON UTILITIES
# ============================================================================

def ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    """Ensure vertices are in counter-clockwise order."""
    n = len(vertices)
    signed_area = 0.0
    for i in range(n):
        j = (i + 1) % n
        signed_area += vertices[i][0] * vertices[j][1]
        signed_area -= vertices[j][0] * vertices[i][1]
    
    if signed_area < 0:  # Clockwise, need to reverse
        return vertices[::-1].copy()
    return vertices


def validate_polygon(vertices: np.ndarray) -> Tuple[bool, str]:
    """
    Validate polygon for extrusion.
    
    Returns:
        (is_valid, message)
    """
    if len(vertices) < 3:
        return False, "Less than 3 vertices"
    
    # Check for duplicate consecutive vertices
    for i in range(len(vertices)):
        j = (i + 1) % len(vertices)
        if np.allclose(vertices[i], vertices[j], atol=1e-6):
            return False, f"Duplicate vertex at index {i}"
    
    # Check for zero area
    area = 0.0
    n = len(vertices)
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    
    if abs(area) < 1e-6:
        return False, "Zero area polygon"
    
    return True, "OK"


def remove_duplicate_vertices(vertices: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Remove consecutive duplicate vertices."""
    if len(vertices) < 3:
        return vertices
    
    cleaned = [vertices[0]]
    for i in range(1, len(vertices)):
        if not np.allclose(vertices[i], cleaned[-1], atol=tolerance):
            cleaned.append(vertices[i])
    
    # Check last vs first
    if len(cleaned) > 1 and np.allclose(cleaned[-1], cleaned[0], atol=tolerance):
        cleaned.pop()
    
    return np.array(cleaned)


# ============================================================================
# WALL EXTRUSION V6
# ============================================================================

class WallExtrusionV6:
    """
    Robust wall extrusion with proper triangulation.
    
    For each room polygon:
    1. Validate and clean polygon
    2. Create floor vertices at z=0
    3. Create ceiling vertices at z=height
    4. Triangulate floor/ceiling using ear-clipping
    5. Create wall quads connecting floor to ceiling
    """
    
    def __init__(self, 
                 room_height: float = 2.5,
                 scale_factor: float = 1.0):
        """
        Initialize wall extrusion.
        
        Args:
            room_height: Room height in meters (default 2.5m for Japanese houses)
            scale_factor: Scale factor from boundary extraction (m/pixel)
        """
        self.room_height = room_height
        self.scale_factor = scale_factor
        
        # Geometry output
        self.vertices: List[List[float]] = []
        self.faces: List[List[int]] = []
        self.vertex_colors: List[List[float]] = []
        self.face_materials: List[int] = []  # Material index per face
        
        # Vertex cache for deduplication
        self.vertex_cache: Dict[tuple, int] = {}
        
        # Room tracking
        self.room_data: List[Dict] = []
        
        # Statistics
        self.stats = {
            "rooms_processed": 0,
            "rooms_skipped": 0,
            "triangles_created": 0,
            "vertices_created": 0,
        }
    
    def load_boundaries(self, boundaries: Dict):
        """
        Load boundary data from extraction.
        
        Args:
            boundaries: Dict with 'rooms', 'walls', 'scale_factor'
        """
        self.room_data = boundaries.get("rooms", [])
        self.scale_factor = boundaries.get("scale_factor", self.scale_factor)
        
        logger.info(f"Loaded {len(self.room_data)} rooms, scale={self.scale_factor:.6f}")
    
    def load_boundaries_json(self, json_path: str):
        """Load boundaries from JSON file."""
        with open(json_path, 'r') as f:
            boundaries = json.load(f)
        self.load_boundaries(boundaries)
    
    def _get_or_create_vertex(self, 
                               pos: np.ndarray, 
                               color: List[float]) -> int:
        """Get existing vertex or create new one."""
        # Round position for cache key
        pos_key = tuple(np.round(pos * 10000).astype(int))
        
        if pos_key in self.vertex_cache:
            return self.vertex_cache[pos_key]
        
        idx = len(self.vertices)
        self.vertices.append(pos.tolist())
        self.vertex_colors.append(color)
        self.vertex_cache[pos_key] = idx
        
        return idx
    
    def _create_room_geometry(self, room: Dict) -> bool:
        """
        Create 3D geometry for a single room.
        
        Args:
            room: Room dict with vertices, color, etc.
            
        Returns:
            True if successful, False if skipped
        """
        # Get vertices
        vertices_2d = np.array(room.get("vertices", []))
        
        if len(vertices_2d) < 3:
            logger.warning(f"Room {room.get('polygon_id', '?')}: insufficient vertices")
            return False
        
        # Apply scale factor
        vertices_2d = vertices_2d * self.scale_factor
        
        # Clean and validate
        vertices_2d = remove_duplicate_vertices(vertices_2d)
        is_valid, msg = validate_polygon(vertices_2d)
        
        if not is_valid:
            logger.warning(f"Room {room.get('polygon_id', '?')}: {msg}")
            return False
        
        # Ensure CCW winding
        vertices_2d = ensure_ccw(vertices_2d)
        
        # Get color (normalized to 0-1)
        color_rgb = room.get("color", [128, 128, 128])
        color_normalized = [c / 255.0 for c in color_rgb]
        
        n = len(vertices_2d)
        
        # ===== CREATE FLOOR VERTICES =====
        floor_indices = []
        for v2d in vertices_2d:
            # Map 2D to 3D: x -> X, y -> Z, height -> Y
            pos = np.array([v2d[0], 0.0, v2d[1]])
            idx = self._get_or_create_vertex(pos, color_normalized)
            floor_indices.append(idx)
        
        # ===== CREATE CEILING VERTICES =====
        ceiling_indices = []
        for v2d in vertices_2d:
            pos = np.array([v2d[0], self.room_height, v2d[1]])
            idx = self._get_or_create_vertex(pos, color_normalized)
            ceiling_indices.append(idx)
        
        # ===== TRIANGULATE FLOOR & CEILING =====
        # Use ear clipping for robust triangulation
        triangles = EarClipTriangulator.triangulate(vertices_2d)
        
        if not triangles:
            logger.warning(f"Room {room.get('polygon_id', '?')}: triangulation failed")
            return False
        
        # Floor triangles (facing down, so reverse winding)
        for tri in triangles:
            i0, i1, i2 = tri
            self.faces.append([
                floor_indices[i0],
                floor_indices[i2],  # Reversed
                floor_indices[i1]
            ])
            self.stats["triangles_created"] += 1
        
        # Ceiling triangles (facing up)
        for tri in triangles:
            i0, i1, i2 = tri
            self.faces.append([
                ceiling_indices[i0],
                ceiling_indices[i1],
                ceiling_indices[i2]
            ])
            self.stats["triangles_created"] += 1
        
        # ===== CREATE WALLS =====
        for i in range(n):
            # Four corners of wall quad
            f0 = floor_indices[i]
            f1 = floor_indices[(i + 1) % n]
            c0 = ceiling_indices[i]
            c1 = ceiling_indices[(i + 1) % n]
            
            # Wall as two triangles
            # Triangle 1: f0 -> f1 -> c1
            self.faces.append([f0, f1, c1])
            # Triangle 2: f0 -> c1 -> c0
            self.faces.append([f0, c1, c0])
            
            self.stats["triangles_created"] += 2
        
        return True
    
    def extrude_all_rooms(self):
        """Process all rooms and create 3D geometry."""
        if not self.room_data:
            raise ValueError("No room data loaded")
        
        logger.info(f"Extruding {len(self.room_data)} rooms...")
        
        for room in self.room_data:
            try:
                success = self._create_room_geometry(room)
                if success:
                    self.stats["rooms_processed"] += 1
                else:
                    self.stats["rooms_skipped"] += 1
            except Exception as e:
                logger.error(f"Room {room.get('polygon_id', '?')}: {e}")
                self.stats["rooms_skipped"] += 1
        
        self.stats["vertices_created"] = len(self.vertices)
        
        logger.info(f"✓ Created {self.stats['vertices_created']} vertices, "
                   f"{self.stats['triangles_created']} triangles")
        logger.info(f"  Rooms: {self.stats['rooms_processed']} processed, "
                   f"{self.stats['rooms_skipped']} skipped")
    
    def get_geometry(self) -> Dict:
        """
        Get complete 3D geometry.
        
        Returns:
            Dict with vertices, faces, colors (as numpy arrays)
        """
        return {
            "vertices": np.array(self.vertices),
            "faces": np.array(self.faces),
            "colors": np.array(self.vertex_colors),
            "stats": self.stats,
        }
    
    def get_bounds(self) -> Dict:
        """Get bounding box of geometry."""
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
# OBJ EXPORTER
# ============================================================================

class OBJExporter:
    """Export geometry to OBJ format with MTL materials."""
    
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.colors = []
        self.materials = {}
    
    def load_geometry(self, geometry: Dict):
        """Load geometry from extrusion."""
        self.vertices = geometry.get("vertices", [])
        self.faces = geometry.get("faces", [])
        self.colors = geometry.get("colors", [])
        
        if isinstance(self.vertices, np.ndarray):
            self.vertices = self.vertices.tolist()
        if isinstance(self.faces, np.ndarray):
            self.faces = self.faces.tolist()
        if isinstance(self.colors, np.ndarray):
            self.colors = self.colors.tolist()
    
    def create_materials(self):
        """Create materials from vertex colors."""
        self.materials = {}
        
        for i, color in enumerate(self.colors):
            # Create color key
            color_key = tuple(int(c * 255) for c in color)
            
            if color_key not in self.materials:
                mat_name = f"mat_{len(self.materials)}"
                self.materials[color_key] = {
                    "name": mat_name,
                    "color": color,
                    "faces": []
                }
        
        logger.info(f"Created {len(self.materials)} materials")
    
    def export_obj(self, obj_path: str, mtl_filename: str = "model.mtl"):
        """Export to OBJ file."""
        obj_path = Path(obj_path)
        
        with open(obj_path, 'w') as f:
            f.write(f"# Floorplan 3D Model\n")
            f.write(f"# Vertices: {len(self.vertices)}\n")
            f.write(f"# Faces: {len(self.faces)}\n")
            f.write(f"mtllib {mtl_filename}\n\n")
            
            # Write vertices with colors
            for i, v in enumerate(self.vertices):
                # OBJ supports vertex colors as extension
                if i < len(self.colors):
                    c = self.colors[i]
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
                else:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            f.write("\n")
            
            # Write faces (OBJ uses 1-indexed vertices)
            f.write("usemtl default\n")
            for face in self.faces:
                # OBJ face indices are 1-based
                face_str = " ".join(str(idx + 1) for idx in face)
                f.write(f"f {face_str}\n")
        
        logger.info(f"Exported OBJ to {obj_path}")
    
    def export_mtl(self, mtl_path: str):
        """Export MTL material file."""
        with open(mtl_path, 'w') as f:
            f.write("# Material file for Floorplan 3D Model\n\n")
            
            # Default material
            f.write("newmtl default\n")
            f.write("Ka 0.2 0.2 0.2\n")  # Ambient
            f.write("Kd 0.8 0.8 0.8\n")  # Diffuse
            f.write("Ks 0.1 0.1 0.1\n")  # Specular
            f.write("Ns 10.0\n")          # Shininess
            f.write("d 1.0\n\n")          # Opacity
            
            # Additional materials from colors
            for color_key, mat_data in self.materials.items():
                f.write(f"newmtl {mat_data['name']}\n")
                c = mat_data['color']
                f.write(f"Ka {c[0]*0.2:.3f} {c[1]*0.2:.3f} {c[2]*0.2:.3f}\n")
                f.write(f"Kd {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
                f.write(f"Ks 0.1 0.1 0.1\n")
                f.write("Ns 10.0\n")
                f.write("d 1.0\n\n")
        
        logger.info(f"Exported MTL to {mtl_path}")


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def extrude_boundaries_to_obj(boundaries_json: str,
                               output_dir: str,
                               room_height: float = 2.5) -> Dict:
    """
    Convenience function: Load boundaries and export to OBJ.
    
    Args:
        boundaries_json: Path to boundaries JSON from extraction
        output_dir: Output directory for OBJ/MTL files
        room_height: Room height in meters
    
    Returns:
        Dict with geometry and file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(boundaries_json).stem.replace("_boundaries", "")
    
    # Load and extrude
    extruder = WallExtrusionV6(room_height=room_height)
    extruder.load_boundaries_json(boundaries_json)
    extruder.extrude_all_rooms()
    
    geometry = extruder.get_geometry()
    
    # Export
    exporter = OBJExporter()
    exporter.load_geometry(geometry)
    exporter.create_materials()
    
    obj_path = output_dir / f"{base_name}.obj"
    mtl_path = output_dir / f"{base_name}.mtl"
    
    exporter.export_obj(str(obj_path), f"{base_name}.mtl")
    exporter.export_mtl(str(mtl_path))
    
    return {
        "geometry": geometry,
        "obj_path": str(obj_path),
        "mtl_path": str(mtl_path),
        "bounds": extruder.get_bounds(),
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extrude room boundaries to 3D OBJ")
    parser.add_argument("--boundaries", "-b", required=True, help="Path to boundaries JSON")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--height", type=float, default=2.5, help="Room height in meters")
    
    args = parser.parse_args()
    
    result = extrude_boundaries_to_obj(
        args.boundaries,
        args.output,
        args.height
    )
    
    print(f"\n✓ Created OBJ: {result['obj_path']}")
    print(f"✓ Created MTL: {result['mtl_path']}")
    print(f"✓ Bounds: {result['bounds']}")