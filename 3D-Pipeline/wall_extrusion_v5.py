"""
Wall Extrusion v5 - CLEAN APPROACH: Simple, minimal, correct geometry

Strategy:
- NO complex offset logic
- NO backface issues
- Just: vertices at room corners + floor/ceiling triangles + simple wall quads
- Rooms stay connected at boundaries
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WallExtrusion5:
    """
    Simple wall extrusion: room boundaries → 3D floor plan model
    
    For each room:
    1. Create floor vertices (at z=0)
    2. Create ceiling vertices (at z=height)
    3. Create wall faces connecting floor to ceiling
    """
    
    def __init__(self, room_height: float = 2.5, wall_thickness: float = 0.0):
        """
        Initialize simple wall extrusion.
        
        Args:
            room_height: Room height in meters (default 2.5m)
            wall_thickness: Currently unused (we keep rooms adjacent, not offset)
        """
        self.room_height = room_height
        self.vertices = []
        self.faces = []
        self.normals = []
        self.vertex_colors = []
        self.vertex_cache = {}
        
        self.boundaries_data = None
        self.scale_factor = 1.0
    
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
        """Process all rooms and create 3D geometry."""
        if not self.boundaries_data:
            raise ValueError("Boundaries data not loaded")
        
        rooms = self.boundaries_data.get('rooms', [])
        logger.info(f"Extruding {len(rooms)} rooms...")
        
        for room_id, room_data in enumerate(rooms):
            try:
                vertices_2d = np.array(room_data.get('vertices', [])) * self.scale_factor
                color_rgb = room_data.get('color', [128, 128, 128])
                
                if len(vertices_2d) < 3:
                    logger.warning(f"Room {room_id}: invalid polygon")
                    continue
                
                self._extrude_single_room(room_id, vertices_2d, color_rgb)
            except Exception as e:
                logger.error(f"Room {room_id}: {e}")
                continue
        
        logger.info(f"✓ Created {len(self.vertices)} vertices, {len(self.faces)} faces")
    
    def _extrude_single_room(self, room_id: int, vertices_2d: np.ndarray, color_rgb: List[int]):
        """
        Extrude a single room.
        
        Creates:
        - Floor (triangulated polygon at z=0)
        - Ceiling (triangulated polygon at z=height)
        - Walls (vertical quads connecting floor to ceiling)
        """
        color_normalized = [c / 255.0 for c in color_rgb]
        
        # ===== FLOOR =====
        floor_verts = []
        for vertex_2d in vertices_2d:
            # Floor at z=0, rotated to horizontal
            pos = np.array([vertex_2d[0], 0.0, vertex_2d[1]])  # X, height=0, Z
            floor_verts.append(self._get_or_create_vertex(pos, [0, -1, 0], color_normalized))
        
        # Triangulate floor
        for tri in self._fan_triangulate(len(floor_verts)):
            v0, v1, v2 = floor_verts[tri[0]], floor_verts[tri[1]], floor_verts[tri[2]]
            self.faces.append([v0, v2, v1])  # Reversed for downward normal
        
        # ===== CEILING =====
        ceiling_verts = []
        for vertex_2d in vertices_2d:
            # Ceiling at z=height
            pos = np.array([vertex_2d[0], self.room_height, vertex_2d[1]])
            ceiling_verts.append(self._get_or_create_vertex(pos, [0, 1, 0], color_normalized))
        
        # Triangulate ceiling
        for tri in self._fan_triangulate(len(ceiling_verts)):
            v0, v1, v2 = ceiling_verts[tri[0]], ceiling_verts[tri[1]], ceiling_verts[tri[2]]
            self.faces.append([v0, v1, v2])  # Normal facing up
        
        # ===== WALLS =====
        n = len(vertices_2d)
        for i in range(n):
            v0_floor = floor_verts[i]
            v1_floor = floor_verts[(i + 1) % n]
            v0_ceil = ceiling_verts[i]
            v1_ceil = ceiling_verts[(i + 1) % n]
            
            # Calculate outward-facing normal for this wall
            edge_2d = vertices_2d[(i + 1) % n] - vertices_2d[i]
            # Perpendicular to edge, pointing outward
            normal_2d = np.array([-edge_2d[1], edge_2d[0]])
            normal_2d = normal_2d / (np.linalg.norm(normal_2d) + 1e-8)
            wall_normal = np.array([normal_2d[0], 0, normal_2d[1]])
            
            # Create wall quad as two triangles
            # Triangle 1: floor-floor-ceil
            self.faces.append([v0_floor, v1_floor, v1_ceil])
            # Triangle 2: floor-ceil-ceil
            self.faces.append([v0_floor, v1_ceil, v0_ceil])
    
    def _get_or_create_vertex(self, pos: np.ndarray, normal: np.ndarray, color: List[float]) -> int:
        """Get or create vertex with caching."""
        pos_key = tuple(np.round(pos * 1000000) / 1000000)
        
        if pos_key in self.vertex_cache:
            return self.vertex_cache[pos_key]
        
        idx = len(self.vertices)
        self.vertices.append(pos.tolist())
        self.normals.append(normal.tolist())
        self.vertex_colors.append(color)
        self.vertex_cache[pos_key] = idx
        
        return idx
    
    def _fan_triangulate(self, n_verts: int) -> List[Tuple[int, int, int]]:
        """Fan triangulation for simple polygons."""
        triangles = []
        for i in range(1, n_verts - 1):
            triangles.append((0, i, i + 1))
        return triangles
    
    def get_geometry(self) -> Dict:
        """Get complete 3D geometry."""
        return {
            'vertices': np.array(self.vertices),
            'faces': np.array(self.faces),
            'normals': np.array(self.normals),
            'colors': np.array(self.vertex_colors),
            'room_mapping': {}
        }
    
    def get_statistics(self) -> Dict:
        """Get geometry statistics."""
        return {
            'num_vertices': len(self.vertices),
            'num_faces': len(self.faces),
            'room_height': self.room_height,
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
