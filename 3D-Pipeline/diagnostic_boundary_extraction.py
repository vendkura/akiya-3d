"""
Diagnostic Tool: Analyze Boundary Extraction Quality
Compares original contours vs simplified contours to quantify distortion
"""

import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageDraw
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BoundaryDiagnostics:
    """Analyze boundary extraction quality and identify issues."""
    
    def __init__(self, mask_path: str):
        """Load mask for analysis."""
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        self.mask_shape = self.mask.shape
        logger.info(f"Loaded mask: {self.mask_shape}")
        
        self.rooms_original = []  # Original (non-simplified) contours
        self.rooms_simplified = []  # Simplified contours
        self.distortion_metrics = {}
    
    def extract_with_and_without_simplification(self):
        """Extract rooms both with and without Douglas-Peucker simplification."""
        logger.info("\n=== EXTRACTING ROOMS ===")
        
        for class_id in range(1, 14):  # Skip background (0)
            binary_mask = (self.mask == class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            for contour_idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < 100:  # Filter noise
                    continue
                
                # Original contour (no simplification)
                original_vertices = contour.squeeze().tolist()
                if not isinstance(original_vertices[0], list):
                    original_vertices = [original_vertices]
                
                # Simplified contour (Douglas-Peucker)
                perimeter = cv2.arcLength(contour, True)
                epsilon = 0.02 * perimeter  # Current setting
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                simplified_vertices = simplified.squeeze().tolist()
                if not isinstance(simplified_vertices[0], list):
                    simplified_vertices = [simplified_vertices]
                
                # Store for analysis
                room_data = {
                    'class_id': class_id,
                    'area': float(area),
                    'perimeter': float(perimeter),
                    'original_vertex_count': len(original_vertices),
                    'simplified_vertex_count': len(simplified_vertices),
                    'original_vertices': original_vertices,
                    'simplified_vertices': simplified_vertices,
                    'epsilon_used': float(epsilon)
                }
                
                self.rooms_original.append(room_data)
        
        logger.info(f"Extracted {len(self.rooms_original)} rooms")
    
    def calculate_distortion_metrics(self):
        """Calculate how much simplification distorts room shapes."""
        logger.info("\n=== CALCULATING DISTORTION ===")
        
        vertex_reductions = []
        area_changes = []
        
        for room in self.rooms_original:
            # Vertex reduction
            orig_count = room['original_vertex_count']
            simp_count = room['simplified_vertex_count']
            reduction = (1 - simp_count / orig_count) * 100 if orig_count > 0 else 0
            vertex_reductions.append(reduction)
            
            # Area change (use polygon area before/after)
            orig_area = self._polygon_area(np.array(room['original_vertices']))
            simp_area = self._polygon_area(np.array(room['simplified_vertices']))
            area_change = abs(simp_area - orig_area) / orig_area * 100 if orig_area > 0 else 0
            area_changes.append(area_change)
            
            room['vertex_reduction_percent'] = float(reduction)
            room['area_change_percent'] = float(area_change)
        
        self.distortion_metrics = {
            'avg_vertex_reduction': float(np.mean(vertex_reductions)),
            'max_vertex_reduction': float(np.max(vertex_reductions)),
            'avg_area_change': float(np.mean(area_changes)),
            'max_area_change': float(np.max(area_changes)),
            'rooms_with_high_distortion': sum(1 for ac in area_changes if ac > 10),
            'rooms_with_moderate_distortion': sum(1 for ac in area_changes if 5 <= ac <= 10),
            'rooms_with_low_distortion': sum(1 for ac in area_changes if ac < 5)
        }
        
        logger.info(f"Average vertex reduction: {self.distortion_metrics['avg_vertex_reduction']:.1f}%")
        logger.info(f"Average area change: {self.distortion_metrics['avg_area_change']:.3f}%")
        logger.info(f"Rooms with HIGH distortion (>10%): {self.distortion_metrics['rooms_with_high_distortion']}")
    
    def verify_room_connectivity(self):
        """Check if adjacent rooms should share walls (are they touching?)."""
        logger.info("\n=== CHECKING ROOM CONNECTIVITY ===")
        
        # For each pair of rooms, check if they share an edge
        connections = []
        for i, room1 in enumerate(self.rooms_original):
            verts1 = np.array(room1['simplified_vertices'])
            
            for j, room2 in enumerate(self.rooms_original[i+1:], i+1):
                verts2 = np.array(room2['simplified_vertices'])
                
                # Check for shared vertices or close proximity
                for v1 in verts1:
                    for v2 in verts2:
                        dist = np.linalg.norm(v1 - v2)
                        if dist < 1.0:  # Within 1 pixel
                            connections.append({
                                'room1': i,
                                'room2': j,
                                'distance': float(dist)
                            })
        
        logger.info(f"Found {len(connections)} adjacent room pairs")
        return connections
    
    def _polygon_area(self, vertices: np.ndarray) -> float:
        """Calculate polygon area using shoelace formula."""
        if len(vertices) < 3:
            return 0
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    def generate_diagnostic_report(self, output_path: str = "diagnostic_report.json"):
        """Save all diagnostic data to JSON."""
        report = {
            'mask_shape': list(self.mask_shape),
            'total_rooms_extracted': len(self.rooms_original),
            'distortion_metrics': self.distortion_metrics,
            'rooms_detail': self.rooms_original
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Report saved to {output_path}")
        return report
    
    def create_visualization(self, output_path: str = "diagnostic_visualization.png"):
        """Create side-by-side visualization of original vs simplified."""
        logger.info(f"\nGenerating visualization...")
        
        # Create image showing both original and simplified contours
        height, width = self.mask_shape
        viz = Image.new('RGB', (width * 2, height), color=(50, 50, 50))
        draw = ImageDraw.Draw(viz)
        
        # Draw original contours (left side)
        for room_idx, room in enumerate(self.rooms_original[:10]):  # First 10 rooms
            color = (100 + room_idx * 15, 150, 200)
            
            # Original vertices
            orig_verts = room['original_vertices']
            if len(orig_verts) > 2:
                orig_verts = [(int(v[0]), int(v[1])) for v in orig_verts]
                draw.polygon(orig_verts, outline=color, width=2)
        
        # Draw simplified contours (right side)
        for room_idx, room in enumerate(self.rooms_original[:10]):
            color = (100 + room_idx * 15, 150, 200)
            
            # Simplified vertices
            simp_verts = room['simplified_vertices']
            if len(simp_verts) > 2:
                simp_verts = [(int(v[0]) + width, int(v[1])) for v in simp_verts]
                draw.polygon(simp_verts, outline=color, width=2)
        
        # Add labels
        draw.text((10, 10), "ORIGINAL", fill=(255, 255, 255))
        draw.text((width + 10, 10), "SIMPLIFIED", fill=(255, 255, 255))
        
        viz.save(output_path)
        logger.info(f"✓ Visualization saved to {output_path}")


def main():
    """Run full diagnostic."""
    # Find first available floor plan image
    floorplan_dir = Path("../U-NET/data/floorplan")
    images = list(floorplan_dir.glob("*.png"))
    
    if not images:
        logger.error(f"No images found in {floorplan_dir}")
        return
    
    mask_path = str(images[0])
    logger.info(f"Using image: {images[0].name}")
    
    # Run diagnostics
    diag = BoundaryDiagnostics(mask_path)
    diag.extract_with_and_without_simplification()
    diag.calculate_distortion_metrics()
    connections = diag.verify_room_connectivity()
    diag.create_visualization()
    report = diag.generate_diagnostic_report()
    
    # Print summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"\nPolygon Simplification Impact:")
    print(f"  Average vertex reduction: {report['distortion_metrics']['avg_vertex_reduction']:.1f}%")
    print(f"  Average area change: {report['distortion_metrics']['avg_area_change']:.3f}%")
    print(f"\nDistortion Distribution:")
    print(f"  HIGH distortion (>10%): {report['distortion_metrics']['rooms_with_high_distortion']} rooms")
    print(f"  MODERATE (5-10%): {report['distortion_metrics']['rooms_with_moderate_distortion']} rooms")
    print(f"  LOW (<5%): {report['distortion_metrics']['rooms_with_low_distortion']} rooms")
    print(f"\nRoom Connectivity:")
    print(f"  Adjacent room pairs: {len(connections)}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
