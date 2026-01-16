"""
Wall Extrusion v4 - FIXED: Walls at boundaries, no offset gaps
Creates walls along room edges WITHOUT separating adjacent rooms.

Key improvements:
- Walls created AT room boundaries (not offset away)
- Creates inner and outer wall faces at the same boundary
- Adjacent rooms remain connected (no gaps)
- Proper manifold topology
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WallExtrusion4:
    """
    Convert 2D room polygons to 3D geometry with walls at boundaries.
    
    Wall structure:
    - Inner wall face: along room boundary, facing inward
    - Outer wall face: along room boundary, facing outward (offset 10cm)
    - Top/bottom: ceiling and floor
    
    KEY: Walls extend FROM the boundary line, not AROUND it
    """
    
    def __init__(self, room_height: float = 2.5, wall_thickness: float = 0.1):
        """
        Initialize wall extrusion with thick walls.
        
        Args:
            room_height: Room height in meters (default 2.5m)
            wall_thickness: Wall thickness in meters (default 0.1m = 10cm)
        """
        self.room_height = room_height
        self.wall_thickness = wall_thickness
        
        # Global geometry
        self.vertices = []
        self.faces = []
        self.normals = []
        self.vertex_colors = []
        self.room_mapping = {}
        
        self.boundaries_data = None
        self.scale_factor = 1.0
        self.rooms = {}
        self.vertex_cache = {}  # Cache for shared vertices
    
    def load_boundaries_json(self, json_path):
        """Load boundary data from JSON file or dict."""
        if isinstance(json_path, dict):
            self.boundaries_data = json_path
        else:
            with open(json_path, 'r') as f:
                self.boundaries_data = json.load(f)
        
        self.scale_factor = self.boundaries_data.get("scale_factor", 1.0)
        logger.info(f"Loaded {len(self.boundaries_data['rooms'])} rooms, scale: {self.scale_factor}")
        
        return self.boundaries_data
    
    def extrude_all_rooms(self):
        """Process all rooms and create 3D geometry with thick walls."""
        if not self.boundaries_data:
            raise ValueError("Boundaries data not loaded")
        
        rooms = self.boundaries_data.get('rooms', [])
        logger.info(f"Extruding {len(rooms)} rooms with {self.wall_thickness*100}cm walls...")
        
        # Process each room
        for room_id, room_data in enumerate(rooms):
            try:
                vertices = np.array(room_data.get('vertices', [])) * self.scale_factor
                color_rgb = room_data.get('color', [128, 128, 128])
                
                if len(vertices) < 3:
                    logger.warning(f"Room {room_id}: invalid polygon (< 3 vertices)")
                    continue
                
                self.extrude_room(room_id, vertices, tuple(color_rgb))
            
            except Exception as e:
                logger.error(f"Room {room_id}: {e}")
                continue
        
        logger.info(f"✓ Created {len(self.vertices)} vertices, {len(self.faces)} faces")
    
    def extrude_room(self, room_id: int, vertices: np.ndarray, color: Tuple[int, int, int]):
        """
        Extrude a single room to 3D.
        
        Creates:
        - Floor mesh (triangulated, at original boundary)
        - Ceiling mesh (triangulated, at original boundary)
        - Wall mesh (vertical faces extending from boundary)
        """
        # Ensure correct polygon winding (CCW when viewed from above)
        vertices = self._ensure_ccw_winding(vertices)
        
        # Create floor and ceiling at the room boundary (no offset)
        self.create_floor(room_id, vertices, color)
        self.create_ceiling(room_id, vertices, color)
        
        # Create walls extending from the boundary
        self.create_walls_at_boundary(room_id, vertices, color)
    
    def _ensure_ccw_winding(self, vertices: np.ndarray) -> np.ndarray:
        """
        Ensure polygon is wound counter-clockwise (CCW) when viewed from above.
        This is important for correct normal calculation.
        """
        # Calculate signed area using shoelace formula
        n = len(vertices)
        signed_area = 0.0
        for i in range(n):
            v0 = vertices[i]
            v1 = vertices[(i + 1) % n]
            signed_area += (v1[0] - v0[0]) * (v1[1] + v0[1])
        
        # If negative, polygon is CW - reverse it to make it CCW
        if signed_area > 0:
            vertices = vertices[::-1]
        
        return vertices
    
    def triangulate_simple(self, vertices: np.ndarray) -> List[Tuple[int, int, int]]:
        """Simple fan triangulation for polygons."""
        if len(vertices) < 3:
            return []
        
        triangles = []
        for i in range(1, len(vertices) - 1):
            triangles.append((0, i, i + 1))
        
        return triangles
    
    def get_or_create_vertex(self, pos: np.ndarray, normal: np.ndarray, color: Tuple[float, float, float]) -> int:
        """Get existing vertex or create new one with position-based caching."""
        pos_rounded = np.round(pos * 1000000) / 1000000
        cache_key = tuple(pos_rounded)
        
        if cache_key in self.vertex_cache:
            return self.vertex_cache[cache_key]
        
        # Create new vertex
        idx = len(self.vertices)
        self.vertices.append(pos.tolist())
        self.normals.append(normal.tolist())
        self.vertex_colors.append([c / 255.0 for c in color])
        self.vertex_cache[cache_key] = idx
        
        return idx
    
    def create_floor(self, room_id: int, vertices: np.ndarray, color: Tuple[int, int, int]):
        """Create floor mesh at Z=0, at the room boundary (no offset)."""
        triangles = self.triangulate_simple(vertices)
        
        if not triangles:
            logger.warning(f"Room {room_id}: floor triangulation failed")
            return
        
        floor_normal = np.array([0, 0, -1])
        vertex_indices = []
        
        # Add vertices at floor level (Z=0)
        for vert in vertices:
            pos_3d = np.array([vert[0], vert[1], 0.0], dtype=float)
            idx = self.get_or_create_vertex(pos_3d, floor_normal, color)
            vertex_indices.append(idx)
        
        # Add faces
        for tri in triangles:
            v0, v1, v2 = vertex_indices[tri[0]], vertex_indices[tri[1]], vertex_indices[tri[2]]
            self.faces.append([v0, v1, v2])
        
        logger.debug(f"Room {room_id}: floor {len(triangles)} triangles")
    
    def create_ceiling(self, room_id: int, vertices: np.ndarray, color: Tuple[int, int, int]):
        """Create ceiling mesh at Z=room_height, at the room boundary (no offset)."""
        triangles = self.triangulate_simple(vertices)
        
        if not triangles:
            logger.warning(f"Room {room_id}: ceiling triangulation failed")
            return
        
        ceiling_normal = np.array([0, 0, 1])
        vertex_indices = []
        
        # Add vertices at ceiling level
        for vert in vertices:
            pos_3d = np.array([vert[0], vert[1], self.room_height], dtype=float)
            idx = self.get_or_create_vertex(pos_3d, ceiling_normal, color)
            vertex_indices.append(idx)
        
        # Add faces (reversed winding for outward normals)
        for tri in triangles:
            v0, v1, v2 = vertex_indices[tri[0]], vertex_indices[tri[2]], vertex_indices[tri[1]]
            self.faces.append([v0, v1, v2])
        
        logger.debug(f"Room {room_id}: ceiling {len(triangles)} triangles")
    
    def create_walls_at_boundary(self, room_id: int, vertices: np.ndarray, color: Tuple[int, int, int]):
        """
        Create walls that extend FROM the boundary line.
        
        For each edge of the room boundary (CCW winding from above):
        - Creates inner wall face (along boundary, facing into room)
        - Creates outer wall face (offset outward by wall_thickness)
        
        For CCW winding:
        - Left perpendicular = outward normal
        """
        n = len(vertices)
        
        for i in range(n):
            p0 = vertices[i]
            p1 = vertices[(i + 1) % n]
            
            # Edge vector (along boundary)
            edge_dir = p1 - p0
            edge_len = np.linalg.norm(edge_dir)
            if edge_len < 1e-6:
                continue
            
            # Outward perpendicular (left of edge direction for CCW polygon)
            # For CCW: rotate edge 90° counter-clockwise = [-dy, dx]
            outward_perp = np.array([-edge_dir[1], edge_dir[0]])
            outward_perp = outward_perp / (np.linalg.norm(outward_perp) + 1e-8)
            
            # Inward is opposite
            inward_perp = -outward_perp
            
            # Normal vectors (perpendicular to wall face)
            inward_normal = np.array([inward_perp[0], inward_perp[1], 0.0])
            outward_normal = np.array([outward_perp[0], outward_perp[1], 0.0])
            
            # INNER WALL: along boundary, facing INTO room
            p0_inner_f = np.array([p0[0], p0[1], 0.0])
            p0_inner_c = np.array([p0[0], p0[1], self.room_height])
            p1_inner_f = np.array([p1[0], p1[1], 0.0])
            p1_inner_c = np.array([p1[0], p1[1], self.room_height])
            
            # OUTER WALL: offset outward by wall thickness
            p0_outer = p0 + outward_perp * self.wall_thickness
            p1_outer = p1 + outward_perp * self.wall_thickness
            
            p0_outer_f = np.array([p0_outer[0], p0_outer[1], 0.0])
            p0_outer_c = np.array([p0_outer[0], p0_outer[1], self.room_height])
            p1_outer_f = np.array([p1_outer[0], p1_outer[1], 0.0])
            p1_outer_c = np.array([p1_outer[0], p1_outer[1], self.room_height])
            
            # Inner wall face (facing inward to room)
            # Winding: p0_f -> p1_f -> p1_c -> p0_c (CCW from inside)
            v0 = self.get_or_create_vertex(p0_inner_f, inward_normal, color)
            v1 = self.get_or_create_vertex(p1_inner_f, inward_normal, color)
            v2 = self.get_or_create_vertex(p1_inner_c, inward_normal, color)
            v3 = self.get_or_create_vertex(p0_inner_c, inward_normal, color)
            
            # Correct winding for inward-facing wall
            self.faces.append([v0, v1, v2])  # Triangle 1: base -> adjacent -> top-adjacent
            self.faces.append([v0, v2, v3])  # Triangle 2: base -> top-adjacent -> top
            
            # Outer wall face (facing outward away from room)
            # Winding must be opposite of inner wall (reversed)
            v4 = self.get_or_create_vertex(p0_outer_f, outward_normal, color)
            v5 = self.get_or_create_vertex(p1_outer_f, outward_normal, color)
            v6 = self.get_or_create_vertex(p1_outer_c, outward_normal, color)
            v7 = self.get_or_create_vertex(p0_outer_c, outward_normal, color)
            
            # Correct winding for outward-facing wall (opposite of inner)
            self.faces.append([v4, v6, v5])  # Triangle 1: reversed
            self.faces.append([v4, v7, v6])  # Triangle 2: reversed
            
            # Side caps (connecting inner to outer walls, perpendicular to edge)
            # These close the wall "thickness"
            # Left cap (at start of edge)
            self.faces.append([v0, v7, v4])  # Inner-top -> Outer-top -> Outer-bottom
            self.faces.append([v0, v4, v3])  # Inner-bottom -> Outer-bottom -> Inner-top
            
            # Right cap (at end of edge)
            self.faces.append([v1, v5, v6])  # Inner-bottom -> Outer-bottom -> Outer-top
            self.faces.append([v1, v6, v2])  # Inner-top -> Outer-top -> Inner-bottom
    
    def _rotate_to_horizontal(self, vertices: np.ndarray, normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate model 90 degrees so it lies flat horizontally (floor plan view).
        
        Transform: (X, Y, Z) → (X, Z, Y)
        This makes the model lie flat with Y pointing up (height).
        """
        # Rotate vertices: swap Y and Z, negate new Y for correct orientation
        rotated_verts = np.zeros_like(vertices)
        rotated_verts[:, 0] = vertices[:, 0]  # X stays X
        rotated_verts[:, 1] = vertices[:, 2]  # Y becomes Z (height)
        rotated_verts[:, 2] = -vertices[:, 1]  # Z becomes -Y
        
        # Rotate normals the same way
        rotated_norms = np.zeros_like(normals)
        rotated_norms[:, 0] = normals[:, 0]   # X stays X
        rotated_norms[:, 1] = normals[:, 2]   # Y becomes Z
        rotated_norms[:, 2] = -normals[:, 1]  # Z becomes -Y
        
        return rotated_verts, rotated_norms
    
    def get_geometry(self) -> Dict:
        """Get complete 3D geometry with horizontal orientation."""
        verts = np.array(self.vertices)
        norms = np.array(self.normals)
        
        # Rotate to horizontal (floor plan lying flat)
        verts, norms = self._rotate_to_horizontal(verts, norms)
        
        return {
            'vertices': verts,
            'faces': np.array(self.faces),
            'normals': norms,
            'colors': np.array(self.vertex_colors),
            'room_mapping': self.room_mapping
        }
    
    def get_statistics(self) -> Dict:
        """Get geometry statistics."""
        return {
            'num_vertices': len(self.vertices),
            'num_faces': len(self.faces),
            'num_rooms': len(self.room_mapping),
            'wall_thickness_cm': self.wall_thickness * 100,
            'bounds': self._get_bounds()
        }
    
    def _get_bounds(self) -> Dict:
        """Get bounding box."""
        if not self.vertices:
            return {'min': [0, 0, 0], 'max': [0, 0, 0]}
        
        verts = np.array(self.vertices)
        return {
            'min': verts.min(axis=0).tolist(),
            'max': verts.max(axis=0).tolist(),
            'size': (verts.max(axis=0) - verts.min(axis=0)).tolist()
        }


def main():
    """Test wall extrusion v4 with thick walls."""
    from boundary_extraction import BoundaryExtractor
    
    # Load test mask
    mask_path = "../U-NET/data/floorplan/Capture d'écran 2025-06-05 224143.png"
    
    # Extract boundaries
    extractor = BoundaryExtractor()
    extractor.load_mask_and_image(mask_path, None)
    boundaries = extractor.process()
    
    # Extrude with v4
    extrusion = WallExtrusion4(wall_thickness=0.1)  # 10cm walls
    extrusion.load_boundaries_json(boundaries)
    extrusion.extrude_all_rooms()
    
    # Get stats
    stats = extrusion.get_statistics()
    print(f"\n✓ Extrusion v4 complete:")
    print(f"  Vertices: {stats['num_vertices']}")
    print(f"  Faces: {stats['num_faces']}")
    print(f"  Rooms: {stats['num_rooms']}")
    print(f"  Wall thickness: {stats['wall_thickness_cm']}cm")
    print(f"  Bounds: {stats['bounds']}")


if __name__ == "__main__":
    main()
