"""
Wall Extrusion - Simplified Version
Convert 2D Room Polygons to 3D Geometry (No offset, clean geometry)

Simplified approach:
- Single-surface walls (no thickness offset)
- Direct floor-to-ceiling wall connection
- Proper polygon triangulation using ear clipping
- Per-room coloring
- Clean, continuous geometry
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleWallExtrusion:
    """Convert 2D room polygons to 3D geometry with simple, clean walls."""
    
    def __init__(self, room_height: float = 2.5):
        """
        Initialize wall extrusion engine.
        
        Args:
            room_height: Height of rooms in meters (default 2.5m)
        """
        self.room_height = room_height
        
        # Global geometry storage
        self.vertices = []          # List of [x, y, z]
        self.faces = []             # List of [v0, v1, v2]
        self.normals = []           # List of [nx, ny, nz]
        self.vertex_colors = []     # List of [r, g, b]
        self.room_mapping = {}      # {room_id: [vertex_indices]}
        
        self.boundaries_data = None
        self.scale_factor = 1.0
        
    def load_boundaries_json(self, json_path: str) -> Dict:
        """
        Load boundary extraction results.
        
        Args:
            json_path: Path to JSON from boundary_extraction.py
        
        Returns:
            Boundaries data dict
        """
        with open(json_path, 'r') as f:
            self.boundaries_data = json.load(f)
        
        self.scale_factor = self.boundaries_data.get("scale_factor", 1.0)
        
        logger.info(f"Loaded {len(self.boundaries_data['rooms'])} rooms from {json_path}")
        logger.info(f"Scale factor: {self.scale_factor} m/pixel")
        
        return self.boundaries_data
    
    def polygon_to_meters(self, vertices: List[List[float]]) -> np.ndarray:
        """
        Convert polygon vertices from pixels to meters.
        
        Args:
            vertices: List of [x, y] in pixels
        
        Returns:
            Nx2 numpy array in meters
        """
        return np.array(vertices) * self.scale_factor
    
    def triangulate_polygon_ear_clipping(self, vertices: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Triangulate polygon using ear clipping algorithm.
        
        Robust method that works for both convex and concave polygons.
        
        Args:
            vertices: Nx2 array of polygon vertices (in order)
        
        Returns:
            List of triangle vertex index tuples
        """
        if len(vertices) < 3:
            logger.warning(f"Polygon has {len(vertices)} vertices, need at least 3")
            return []
        
        if len(vertices) == 3:
            # Already a triangle
            return [(0, 1, 2)]
        
        triangles = []
        vertex_indices = list(range(len(vertices)))
        
        # Ear clipping
        while len(vertex_indices) > 3:
            found_ear = False
            
            for i in range(len(vertex_indices)):
                prev_idx = (i - 1) % len(vertex_indices)
                curr_idx = i
                next_idx = (i + 1) % len(vertex_indices)
                
                prev_vert = vertices[vertex_indices[prev_idx]]
                curr_vert = vertices[vertex_indices[curr_idx]]
                next_vert = vertices[vertex_indices[next_idx]]
                
                # Check if this forms an ear (convex angle)
                v1 = curr_vert - prev_vert
                v2 = next_vert - curr_vert
                
                # Cross product to check convexity
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                
                if cross <= 0:
                    # Not convex, skip
                    continue
                
                # Check if any other vertex is inside this triangle
                is_ear = True
                for j in range(len(vertex_indices)):
                    if j == prev_idx or j == curr_idx or j == next_idx:
                        continue
                    
                    test_vert = vertices[vertex_indices[j]]
                    
                    # Check if point is inside triangle using barycentric coordinates
                    if self.point_in_triangle(test_vert, prev_vert, curr_vert, next_vert):
                        is_ear = False
                        break
                
                if is_ear:
                    # Found an ear, add triangle and remove middle vertex
                    triangles.append((vertex_indices[prev_idx], vertex_indices[curr_idx], vertex_indices[next_idx]))
                    vertex_indices.pop(curr_idx)
                    found_ear = True
                    break
            
            if not found_ear:
                # Couldn't find ear, use fan triangulation as fallback
                logger.warning("Ear clipping failed, using fan triangulation")
                remaining = vertex_indices
                for i in range(1, len(remaining) - 1):
                    triangles.append((remaining[0], remaining[i], remaining[i + 1]))
                break
        
        # Add final triangle
        if len(vertex_indices) == 3:
            triangles.append((vertex_indices[0], vertex_indices[1], vertex_indices[2]))
        
        return triangles
    
    def point_in_triangle(self, p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        """
        Check if point p is inside triangle abc using cross products.
        
        Args:
            p: Point to test [x, y]
            a, b, c: Triangle vertices [x, y]
        
        Returns:
            True if p is inside triangle (excluding edges)
        """
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign(p, a, b)
        d2 = sign(p, b, c)
        d3 = sign(p, c, a)
        
        # Check if all same sign (point is inside)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)
    
    def add_vertex(self, 
                   position: np.ndarray, 
                   normal: np.ndarray,
                   color: Tuple[float, float, float]) -> int:
        """
        Add vertex to global list.
        
        Args:
            position: [x, y, z]
            normal: [nx, ny, nz]
            color: [r, g, b] in 0-255 range
        
        Returns:
            Vertex index
        """
        idx = len(self.vertices)
        self.vertices.append(position.tolist())
        self.normals.append(normal.tolist())
        # Normalize color to 0-1 range
        self.vertex_colors.append([c / 255.0 for c in color])
        return idx
    
    def add_face(self, v0: int, v1: int, v2: int):
        """
        Add triangular face to global list.
        
        Args:
            v0, v1, v2: Vertex indices
        """
        self.faces.append([v0, v1, v2])
    
    def create_floor_mesh(self, room_id: int, vertices: np.ndarray, color: Tuple[int, int, int]):
        """
        Create floor mesh for room.
        
        Args:
            room_id: Room identifier
            vertices: Nx2 array of polygon vertices (meters)
            color: RGB color tuple
        """
        # Triangulate
        triangles = self.triangulate_polygon_ear_clipping(vertices)
        
        if not triangles:
            logger.warning(f"Room {room_id}: floor triangulation failed")
            return
        
        # Add vertices (all at z=0)
        vertex_indices = []
        floor_normal = np.array([0, 0, -1])  # Pointing down
        
        for vert in vertices:
            pos_3d = np.array([vert[0], vert[1], 0])
            idx = self.add_vertex(pos_3d, floor_normal, color)
            vertex_indices.append(idx)
        
        # Add faces
        for tri in triangles:
            v0 = vertex_indices[tri[0]]
            v1 = vertex_indices[tri[1]]
            v2 = vertex_indices[tri[2]]
            self.add_face(v0, v1, v2)
        
        # Track room mapping
        if room_id not in self.room_mapping:
            self.room_mapping[room_id] = []
        self.room_mapping[room_id].extend(vertex_indices)
        
        logger.debug(f"Room {room_id}: created floor with {len(triangles)} triangles")
    
    def create_ceiling_mesh(self, room_id: int, vertices: np.ndarray, color: Tuple[int, int, int]):
        """
        Create ceiling mesh for room.
        
        Args:
            room_id: Room identifier
            vertices: Nx2 array of polygon vertices (meters)
            color: RGB color tuple
        """
        # Triangulate
        triangles = self.triangulate_polygon_ear_clipping(vertices)
        
        if not triangles:
            logger.warning(f"Room {room_id}: ceiling triangulation failed")
            return
        
        # Add vertices (all at z=room_height)
        vertex_indices = []
        ceiling_normal = np.array([0, 0, 1])  # Pointing up
        
        for vert in vertices:
            pos_3d = np.array([vert[0], vert[1], self.room_height])
            idx = self.add_vertex(pos_3d, ceiling_normal, color)
            vertex_indices.append(idx)
        
        # Add faces (reverse winding for outward normals)
        for tri in triangles:
            v0 = vertex_indices[tri[0]]
            v2 = vertex_indices[tri[1]]  # Swap to reverse winding
            v1 = vertex_indices[tri[2]]
            self.add_face(v0, v1, v2)
        
        # Track room mapping
        if room_id not in self.room_mapping:
            self.room_mapping[room_id] = []
        self.room_mapping[room_id].extend(vertex_indices)
        
        logger.debug(f"Room {room_id}: created ceiling with {len(triangles)} triangles")
    
    def create_wall_mesh(self, room_id: int, vertices: np.ndarray, color: Tuple[int, int, int]):
        """
        Create simple wall mesh for room (NO thickness offset).
        
        Args:
            room_id: Room identifier
            vertices: Nx2 array of polygon vertices (meters)
            color: RGB color tuple
        """
        n = len(vertices)
        
        # For each edge of polygon, create wall faces
        for i in range(n):
            # Edge endpoints (outer polygon at ground level)
            p0_floor = vertices[i]
            p1_floor = vertices[(i + 1) % n]
            
            # Corresponding ceiling points
            p0_ceil = vertices[i]
            p1_ceil = vertices[(i + 1) % n]
            
            # Create 3D vertices
            v0_floor = np.array([p0_floor[0], p0_floor[1], 0])
            v1_floor = np.array([p1_floor[0], p1_floor[1], 0])
            v0_ceil = np.array([p0_ceil[0], p0_ceil[1], self.room_height])
            v1_ceil = np.array([p1_ceil[0], p1_ceil[1], self.room_height])
            
            # Compute outward wall normal (perpendicular to edge)
            edge = p1_floor - p0_floor
            # Normal perpendicular to edge, pointing outward (rotate 90° CW)
            normal_2d = np.array([edge[1], -edge[0]])
            normal_2d_norm = np.linalg.norm(normal_2d)
            if normal_2d_norm > 1e-6:
                normal_2d = normal_2d / normal_2d_norm
            wall_normal = np.array([normal_2d[0], normal_2d[1], 0])
            
            # Add 4 vertices for this wall quad
            idx_floor_0 = self.add_vertex(v0_floor, wall_normal, color)
            idx_floor_1 = self.add_vertex(v1_floor, wall_normal, color)
            idx_ceil_0 = self.add_vertex(v0_ceil, wall_normal, color)
            idx_ceil_1 = self.add_vertex(v1_ceil, wall_normal, color)
            
            # Create 2 triangles for quad
            # Triangle 1: floor_0, ceil_0, ceil_1
            self.add_face(idx_floor_0, idx_ceil_0, idx_ceil_1)
            # Triangle 2: floor_0, ceil_1, floor_1
            self.add_face(idx_floor_0, idx_ceil_1, idx_floor_1)
        
        logger.debug(f"Room {room_id}: created walls for {n} edges")
    
    def extrude_room(self, room_id: int, room_data: Dict):
        """
        Extrude a single room to 3D geometry.
        
        Args:
            room_id: Room identifier
            room_data: Room dict with vertices and color
        """
        vertices_px = np.array(room_data["vertices"])
        color = tuple(room_data["color"])
        
        # Convert to meters
        vertices_m = self.polygon_to_meters(vertices_px)
        
        # Validate polygon (at least 3 vertices)
        if len(vertices_m) < 3:
            logger.warning(f"Room {room_id}: has {len(vertices_m)} vertices, skipping")
            return
        
        # Create meshes
        self.create_floor_mesh(room_id, vertices_m, color)
        self.create_ceiling_mesh(room_id, vertices_m, color)
        self.create_wall_mesh(room_id, vertices_m, color)
    
    def extrude_all_rooms(self):
        """
        Extrude all rooms from loaded boundaries.
        """
        if not self.boundaries_data:
            raise ValueError("No boundaries data loaded. Call load_boundaries_json() first.")
        
        rooms = self.boundaries_data.get("rooms", [])
        logger.info(f"Extruding {len(rooms)} rooms...")
        
        for room_id, room_data in enumerate(rooms):
            try:
                self.extrude_room(room_id, room_data)
            except Exception as e:
                logger.warning(f"Failed to extrude room {room_id}: {e}")
        
        logger.info(f"Extrusion complete: {len(self.vertices)} vertices, {len(self.faces)} faces")
    
    def get_geometry(self) -> Dict:
        """
        Get geometry as dict ready for OBJ export.
        
        Returns:
            Dict with vertices, faces, normals, colors
        """
        return {
            "vertices": self.vertices,
            "faces": self.faces,
            "normals": self.normals,
            "colors": self.vertex_colors,
            "room_mapping": self.room_mapping,
            "metadata": {
                "num_vertices": len(self.vertices),
                "num_faces": len(self.faces),
                "room_height": self.room_height,
                "scale_factor": self.scale_factor
            }
        }
    
    def process(self, 
                boundaries_json_path: str,
                output_geometry_path: str = None) -> Dict:
        """
        Full pipeline: load boundaries and extrude to 3D.
        
        Args:
            boundaries_json_path: Path to JSON from boundary_extraction.py
            output_geometry_path: Optional path to save geometry JSON
        
        Returns:
            Geometry dict
        """
        # Load boundaries
        self.load_boundaries_json(boundaries_json_path)
        
        # Extrude all rooms
        self.extrude_all_rooms()
        
        # Get geometry
        geometry = self.get_geometry()
        
        # Save if requested
        if output_geometry_path:
            with open(output_geometry_path, 'w') as f:
                json.dump(geometry, f, indent=2)
            logger.info(f"Saved geometry to {output_geometry_path}")
        
        return geometry


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python wall_extrusion.py <boundaries_json> [output_geometry_json]")
        sys.exit(1)
    
    boundaries_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    extrusion = SimpleWallExtrusion()
    extrusion.process(boundaries_path, output_path)
