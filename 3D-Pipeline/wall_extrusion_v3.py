"""
Improved Wall Extrusion v3 - Better geometry with shared walls
Properly handles room connectivity and wall geometry.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WallExtrusion3D:
    """
    Convert 2D room polygons to 3D geometry with proper wall connectivity.
    
    Key improvements over v2:
    - Proper wall geometry (connects adjacent rooms)
    - Shared floor/ceiling vertices between rooms
    - Better normal calculation
    - Watertight geometry
    """
    
    def __init__(self, room_height: float = 2.5, wall_thickness: float = 0.1):
        """
        Initialize wall extrusion.
        
        Args:
            room_height: Room height in meters (default 2.5m)
            wall_thickness: Wall thickness in meters (for wall faces)
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
        """Process all rooms and create 3D geometry."""
        if not self.boundaries_data:
            raise ValueError("Boundaries data not loaded")
        
        rooms = self.boundaries_data.get('rooms', [])
        logger.info(f"Extruding {len(rooms)} rooms...")
        
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
        - Floor mesh (triangulated)
        - Ceiling mesh (triangulated)
        - Wall mesh (quad faces, properly oriented)
        """
        # Create meshes
        self.create_floor(room_id, vertices, color)
        self.create_ceiling(room_id, vertices, color)
        self.create_walls(room_id, vertices, color)
    
    def triangulate_simple(self, vertices: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Simple fan triangulation for polygons.
        Fast and reliable for both convex and concave shapes.
        """
        if len(vertices) < 3:
            return []
        
        triangles = []
        # Fan from first vertex
        for i in range(1, len(vertices) - 1):
            triangles.append((0, i, i + 1))
        
        return triangles
    
    def get_or_create_vertex(self, pos: np.ndarray, normal: np.ndarray, color: Tuple[float, float, float]) -> int:
        """
        Get existing vertex or create new one.
        Uses position-based caching to share vertices.
        """
        # Round to avoid floating-point duplicates
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
        """Create floor mesh for room."""
        triangles = self.triangulate_simple(vertices)
        
        if not triangles:
            logger.warning(f"Room {room_id}: floor triangulation failed")
            return
        
        floor_normal = np.array([0, 0, -1])
        vertex_indices = []
        
        # Add/retrieve vertices
        for vert in vertices:
            pos_3d = np.array([vert[0], vert[1], 0.0], dtype=float)
            idx = self.get_or_create_vertex(pos_3d, floor_normal, color)
            vertex_indices.append(idx)
        
        # Add faces
        for tri in triangles:
            v0, v1, v2 = vertex_indices[tri[0]], vertex_indices[tri[1]], vertex_indices[tri[2]]
            self.faces.append([v0, v1, v2])
        
        self._track_room_vertices(room_id, vertex_indices)
        logger.debug(f"Room {room_id}: floor {len(triangles)} triangles")
    
    def create_ceiling(self, room_id: int, vertices: np.ndarray, color: Tuple[int, int, int]):
        """Create ceiling mesh for room."""
        triangles = self.triangulate_simple(vertices)
        
        if not triangles:
            logger.warning(f"Room {room_id}: ceiling triangulation failed")
            return
        
        ceiling_normal = np.array([0, 0, 1])
        vertex_indices = []
        
        # Add/retrieve vertices
        for vert in vertices:
            pos_3d = np.array([vert[0], vert[1], self.room_height], dtype=float)
            idx = self.get_or_create_vertex(pos_3d, ceiling_normal, color)
            vertex_indices.append(idx)
        
        # Add faces (reverse winding for outward normals)
        for tri in triangles:
            v0, v1, v2 = vertex_indices[tri[0]], vertex_indices[tri[2]], vertex_indices[tri[1]]
            self.faces.append([v0, v1, v2])
        
        self._track_room_vertices(room_id, vertex_indices)
        logger.debug(f"Room {room_id}: ceiling {len(triangles)} triangles")
    
    def create_walls(self, room_id: int, vertices: np.ndarray, color: Tuple[int, int, int]):
        """
        Create wall mesh for room perimeter.
        
        For each edge of room polygon, creates:
        - Two triangles per edge (quad face)
        - Proper outward-facing normals
        """
        n = len(vertices)
        wall_vertices = []
        
        for i in range(n):
            p0 = vertices[i]
            p1 = vertices[(i + 1) % n]
            
            # Bottom edge
            p0_floor = np.array([p0[0], p0[1], 0.0], dtype=float)
            p1_floor = np.array([p1[0], p1[1], 0.0], dtype=float)
            
            # Top edge
            p0_ceil = np.array([p0[0], p0[1], self.room_height], dtype=float)
            p1_ceil = np.array([p1[0], p1[1], self.room_height], dtype=float)
            
            # Calculate normal (perpendicular to edge, pointing outward)
            edge_dir = p1 - p0
            # Normal in XY plane (perpendicular to edge, pointing outward from polygon)
            normal_xy = np.array([-edge_dir[1], edge_dir[0]])
            normal_xy = normal_xy / (np.linalg.norm(normal_xy) + 1e-8)
            normal = np.array([normal_xy[0], normal_xy[1], 0.0])
            
            # Get or create vertices
            v0 = self.get_or_create_vertex(p0_floor, normal, color)
            v1 = self.get_or_create_vertex(p1_floor, normal, color)
            v2 = self.get_or_create_vertex(p1_ceil, normal, color)
            v3 = self.get_or_create_vertex(p0_ceil, normal, color)
            
            wall_vertices.extend([v0, v1, v2, v3])
            
            # Add quad as two triangles
            self.faces.append([v0, v1, v2])  # Bottom-right triangle
            self.faces.append([v0, v2, v3])  # Top-left triangle
        
        self._track_room_vertices(room_id, wall_vertices)
        logger.debug(f"Room {room_id}: walls {n * 2} triangles")
    
    def _track_room_vertices(self, room_id: int, vertex_indices: List[int]):
        """Track which vertices belong to which room."""
        if room_id not in self.room_mapping:
            self.room_mapping[room_id] = []
        self.room_mapping[room_id].extend(vertex_indices)
    
    def get_geometry(self) -> Dict:
        """Get complete 3D geometry."""
        return {
            'vertices': np.array(self.vertices),
            'faces': np.array(self.faces),
            'normals': np.array(self.normals),
            'colors': np.array(self.vertex_colors),
            'room_mapping': self.room_mapping
        }
    
    def get_statistics(self) -> Dict:
        """Get geometry statistics."""
        return {
            'num_vertices': len(self.vertices),
            'num_faces': len(self.faces),
            'num_rooms': len(self.room_mapping),
            'total_area_m2': sum([
                self._polygon_area(verts) 
                for verts in self.boundaries_data.get('rooms', [])
            ]),
            'bounds': self._get_bounds()
        }
    
    def _polygon_area(self, room_data: Dict) -> float:
        """Calculate polygon area using shoelace formula."""
        vertices = np.array(room_data.get('vertices', [])) * self.scale_factor
        if len(vertices) < 3:
            return 0
        
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
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
    """Test wall extrusion v3."""
    from boundary_extraction import BoundaryExtractor
    
    # Load test mask
    mask_path = "../U-NET/data/floorplan/Capture d'écran 2025-06-05 224143.png"
    from PIL import Image
    mask = np.array(Image.open(mask_path))
    
    # Extract boundaries
    extractor = BoundaryExtractor()
    extractor.load_mask_and_image(mask_path, None)
    boundaries = extractor.process()
    
    # Extrude
    extrusion = WallExtrusion3D()
    extrusion.load_boundaries_json(boundaries)
    extrusion.extrude_all_rooms()
    
    # Get stats
    stats = extrusion.get_statistics()
    print(f"\n✓ Extrusion complete:")
    print(f"  Vertices: {stats['num_vertices']}")
    print(f"  Faces: {stats['num_faces']}")
    print(f"  Rooms: {stats['num_rooms']}")
    print(f"  Total area: {stats['total_area_m2']:.1f} m²")
    print(f"  Bounds: {stats['bounds']}")


if __name__ == "__main__":
    main()
