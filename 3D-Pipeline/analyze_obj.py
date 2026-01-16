"""
Quick OBJ file analyzer and validator.
Check geometry quality of generated models.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class OBJAnalyzer:
    """Analyze OBJ file geometry."""
    
    def __init__(self, obj_path: str):
        """Load and parse OBJ file."""
        self.obj_path = Path(obj_path)
        self.vertices = []
        self.faces = []
        self.normals = []
        
        self.load_obj()
    
    def load_obj(self):
        """Parse OBJ file."""
        with open(self.obj_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if parts[0] == 'v':
                    self.vertices.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'vn':
                    self.normals.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'f':
                    face = []
                    for part in parts[1:]:
                        indices = part.split('/')
                        face.append(int(indices[0]) - 1)  # Convert to 0-indexed
                    if len(face) == 3:
                        self.faces.append(face)
        
        self.vertices = np.array(self.vertices)
        self.faces = np.array(self.faces)
        if self.normals:
            self.normals = np.array(self.normals)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive geometry statistics."""
        stats = {
            'num_vertices': len(self.vertices),
            'num_faces': len(self.faces),
            'num_normals': len(self.normals),
            'file_size_kb': self.obj_path.stat().st_size / 1024,
            'bounds': self._get_bounds(),
            'vertex_density': len(self.vertices) / (len(self.faces) / 3) if len(self.faces) > 0 else 0,
            'degenerate_faces': self._count_degenerate_faces(),
            'face_quality': self._analyze_face_quality(),
            'manifold_check': self._check_manifold()
        }
        return stats
    
    def _get_bounds(self) -> Dict:
        """Get bounding box."""
        min_point = self.vertices.min(axis=0)
        max_point = self.vertices.max(axis=0)
        return {
            'min': min_point.tolist(),
            'max': max_point.tolist(),
            'size': (max_point - min_point).tolist()
        }
    
    def _count_degenerate_faces(self) -> int:
        """Count faces with zero area (degenerate)."""
        count = 0
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            if area < 1e-6:
                count += 1
        return count
    
    def _analyze_face_quality(self) -> Dict:
        """Analyze face quality metrics."""
        areas = []
        aspect_ratios = []
        
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            
            # Area
            edge1 = v1 - v0
            edge2 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            areas.append(area)
            
            # Aspect ratio (longest / shortest edge)
            e0_len = np.linalg.norm(v1 - v0)
            e1_len = np.linalg.norm(v2 - v1)
            e2_len = np.linalg.norm(v0 - v2)
            
            min_edge = min(e0_len, e1_len, e2_len)
            max_edge = max(e0_len, e1_len, e2_len)
            
            if min_edge > 1e-8:
                aspect_ratios.append(max_edge / min_edge)
        
        areas = np.array(areas)
        aspect_ratios = np.array(aspect_ratios)
        
        return {
            'min_area': float(areas.min()),
            'max_area': float(areas.max()),
            'avg_area': float(areas.mean()),
            'avg_aspect_ratio': float(aspect_ratios.mean()) if len(aspect_ratios) > 0 else 0
        }
    
    def _check_manifold(self) -> Dict:
        """Check if geometry is manifold."""
        # Count edge usage
        edge_count = {}
        
        for face in self.faces:
            for i in range(3):
                v0 = face[i]
                v1 = face[(i + 1) % 3]
                
                # Normalize edge (always smaller index first)
                edge = tuple(sorted([v0, v1]))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Manifold check: each edge should be used exactly 2 times (shared by 2 faces)
        non_manifold_edges = sum(1 for count in edge_count.values() if count != 2)
        
        return {
            'is_manifold': non_manifold_edges == 0,
            'non_manifold_edges': non_manifold_edges,
            'total_edges': len(edge_count)
        }
    
    def print_report(self):
        """Print detailed analysis report."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print(f"OBJ FILE ANALYSIS: {self.obj_path.name}")
        print("="*60)
        
        print(f"\n📊 GEOMETRY:")
        print(f"  Vertices: {stats['num_vertices']}")
        print(f"  Faces: {stats['num_faces']}")
        print(f"  Normals: {stats['num_normals']}")
        print(f"  File size: {stats['file_size_kb']:.1f} KB")
        
        print(f"\n📐 BOUNDS:")
        bounds = stats['bounds']
        print(f"  Min: {[f'{x:.2f}' for x in bounds['min']]}")
        print(f"  Max: {[f'{x:.2f}' for x in bounds['max']]}")
        print(f"  Size: {[f'{x:.2f}' for x in bounds['size']]} m")
        
        print(f"\n✓ QUALITY:")
        face_q = stats['face_quality']
        print(f"  Degenerate faces: {stats['degenerate_faces']}")
        print(f"  Min face area: {face_q['min_area']:.6f} m²")
        print(f"  Max face area: {face_q['max_area']:.6f} m²")
        print(f"  Avg aspect ratio: {face_q['avg_aspect_ratio']:.2f}")
        
        print(f"\n🔄 TOPOLOGY:")
        manifold = stats['manifold_check']
        is_manifold = "✓ YES" if manifold['is_manifold'] else "✗ NO"
        print(f"  Manifold: {is_manifold}")
        print(f"  Non-manifold edges: {manifold['non_manifold_edges']}")
        print(f"  Total edges: {manifold['total_edges']}")
        
        print("\n" + "="*60 + "\n")


def main():
    """Analyze generated OBJ files."""
    # Analyze v3 output
    obj_path = Path("pipeline_output/Capture d’écran 2025-06-05 224143/Capture d’écran 2025-06-05 224143.obj")
    
    
    if obj_path.exists():
        analyzer = OBJAnalyzer(str(obj_path))
        analyzer.print_report()
    else:
        print(f"OBJ file not found: {obj_path}")
        print("Run main_pipeline.py first to generate files")


if __name__ == "__main__":
    main()
