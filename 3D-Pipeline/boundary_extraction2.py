"""
Boundary Extraction v2 - Improved polygon extraction with noise filtering

Changes from v1:
- min_contour_area increased to 3000 (was 100)
- Morphological cleanup before extraction
- Polygon simplification and validation
- Quality metadata for each polygon
- Convexity analysis for downstream triangulation hints

Author: Asheleyine's Master thesis project
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

CLASS_NAMES = {
    0: "background",
    1: "dining_area",
    2: "bathroom",
    3: "bedroom",
    4: "closet",
    5: "room",
    6: "door",
    7: "entrance",
    8: "kitchen",
    9: "outdoor_space",
    10: "sliding_door",
    11: "stairs",
    12: "window",
    13: "balcony"
}

# RGB colors for visualization
CLASS_COLORS = {
    "background": (50, 50, 50),
    "dining_area": (255, 182, 193),
    "bathroom": (255, 200, 124),
    "bedroom": (173, 216, 230),
    "closet": (255, 228, 181),
    "room": (255, 255, 224),
    "door": (160, 82, 45),
    "entrance": (240, 255, 240),
    "kitchen": (144, 238, 144),
    "outdoor_space": (152, 251, 152),
    "sliding_door": (169, 169, 169),
    "stairs": (211, 211, 211),
    "window": (135, 206, 250),
    "balcony": (176, 224, 230),
}

# Classes that should be extruded as rooms (have floors/ceilings)
ROOM_CLASSES = {
    "dining_area", "bathroom", "bedroom", "closet", "room",
    "entrance", "kitchen", "outdoor_space", "balcony", "stairs"
}

# Classes that are architectural features (doors, windows) - handled differently
FEATURE_CLASSES = {
    "door", "sliding_door", "window"
}


# ============================================================================
# POLYGON DATA STRUCTURE
# ============================================================================

@dataclass
class ExtractedPolygon:
    """Data structure for an extracted room polygon with quality metadata."""
    polygon_id: int
    class_name: str
    class_id: int
    vertices: List[List[float]]  # [[x, y], ...]
    area: float
    perimeter: float
    centroid: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    
    # Quality metrics
    vertex_count: int
    is_convex: bool
    convexity_ratio: float  # 1.0 = perfectly convex
    solidity: float  # area / convex_hull_area
    
    # Color for visualization/export
    color: Tuple[int, int, int]
    
    # Triangulation hint
    needs_ear_clipping: bool  # True if non-convex


# ============================================================================
# BOUNDARY EXTRACTOR V2
# ============================================================================

class BoundaryExtractorV2:
    """
    Improved boundary extraction with noise filtering and quality analysis.
    
    Key improvements:
    - Higher min_contour_area (3000 vs 100) to filter noise
    - Morphological operations to clean up mask
    - Polygon simplification with quality preservation
    - Convexity analysis for triangulation hints
    """
    
    def __init__(self,
                 min_contour_area: int = 3000,
                 contour_approximation_epsilon: float = 0.015,
                 morphology_kernel_size: int = 3,
                 enable_morphology: bool = True):
        """
        Initialize boundary extractor.
        
        Args:
            min_contour_area: Minimum pixel area to consider valid (default 3000)
            contour_approximation_epsilon: Douglas-Peucker epsilon as fraction of perimeter
            morphology_kernel_size: Kernel size for morphological cleanup
            enable_morphology: Whether to apply morphological cleanup
        """
        self.min_contour_area = min_contour_area
        self.contour_approximation_epsilon = contour_approximation_epsilon
        self.morphology_kernel_size = morphology_kernel_size
        self.enable_morphology = enable_morphology
        
        # Results
        self.polygons: List[ExtractedPolygon] = []
        self.walls: List[Dict] = []
        self.scale_factor: float = 1.0
        
        # Input data
        self.mask: Optional[np.ndarray] = None
        self.original_image: Optional[np.ndarray] = None
        self.mask_shape: Optional[Tuple[int, int]] = None
        
        # Statistics
        self.stats = {
            "total_contours_found": 0,
            "contours_filtered_by_area": 0,
            "valid_polygons": 0,
            "convex_polygons": 0,
            "non_convex_polygons": 0,
        }
    
    def load_mask(self, mask_path: str, image_path: Optional[str] = None):
        """Load segmentation mask from file."""
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            raise FileNotFoundError(f"Cannot load mask: {mask_path}")
        
        self.mask_shape = self.mask.shape
        
        if image_path:
            self.original_image = cv2.imread(image_path)
        
        logger.info(f"Loaded mask: {self.mask_shape}, unique classes: {np.unique(self.mask)}")
    
    def load_mask_array(self, mask_array: np.ndarray):
        """Load mask from numpy array."""
        if mask_array.ndim != 2:
            raise ValueError(f"Mask must be 2D, got shape {mask_array.shape}")
        
        self.mask = mask_array.astype(np.uint8)
        self.mask_shape = self.mask.shape
        
        logger.info(f"Loaded mask array: {self.mask_shape}")
    
    def _apply_morphology(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up binary mask.
        
        Operations:
        1. Close small holes (morphological closing)
        2. Remove small noise (morphological opening)
        """
        if not self.enable_morphology:
            return binary_mask
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (self.morphology_kernel_size, self.morphology_kernel_size)
        )
        
        # Close small holes
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return opened
    
    def _simplify_polygon(self, contour: np.ndarray) -> np.ndarray:
        """
        Simplify polygon using Douglas-Peucker algorithm.
        
        Args:
            contour: OpenCV contour array
            
        Returns:
            Simplified contour
        """
        perimeter = cv2.arcLength(contour, True)
        epsilon = self.contour_approximation_epsilon * perimeter
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        
        return simplified
    
    def _ensure_ccw_winding(self, vertices: np.ndarray) -> np.ndarray:
        """
        Ensure polygon vertices are in counter-clockwise order.
        
        This is important for consistent normal direction in 3D.
        """
        # Calculate signed area (positive = CCW, negative = CW)
        n = len(vertices)
        signed_area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            signed_area += vertices[i][0] * vertices[j][1]
            signed_area -= vertices[j][0] * vertices[i][1]
        
        signed_area /= 2.0
        
        # If clockwise, reverse
        if signed_area < 0:
            return vertices[::-1]
        
        return vertices
    
    def _analyze_polygon_quality(self, contour: np.ndarray) -> Dict:
        """
        Analyze polygon quality metrics.
        
        Returns dict with:
        - is_convex: bool
        - convexity_ratio: float (0-1)
        - solidity: float (0-1)
        """
        # Convexity test
        is_convex = cv2.isContourConvex(contour)
        
        # Convex hull analysis
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        
        # Solidity = area / convex_hull_area
        solidity = contour_area / (hull_area + 1e-8)
        
        # Convexity ratio (alternative measure)
        hull_perimeter = cv2.arcLength(hull, True)
        contour_perimeter = cv2.arcLength(contour, True)
        convexity_ratio = hull_perimeter / (contour_perimeter + 1e-8)
        
        return {
            "is_convex": is_convex,
            "convexity_ratio": min(1.0, convexity_ratio),
            "solidity": min(1.0, solidity),
        }
    
    def extract_polygons(self) -> List[ExtractedPolygon]:
        """
        Extract room polygons from segmentation mask.
        
        Returns:
            List of ExtractedPolygon objects
        """
        if self.mask is None:
            raise ValueError("Mask not loaded")
        
        self.polygons = []
        self.stats = {k: 0 for k in self.stats}
        polygon_id = 0
        
        # Process each class
        for class_id in range(1, 14):
            class_name = CLASS_NAMES.get(class_id, f"unknown_{class_id}")
            
            # Skip feature classes for room extraction
            # (they can be handled separately if needed)
            if class_name in FEATURE_CLASSES:
                continue
            
            # Create binary mask for this class
            binary_mask = (self.mask == class_id).astype(np.uint8) * 255
            
            # Skip if no pixels of this class
            if binary_mask.sum() == 0:
                continue
            
            # Apply morphological cleanup
            cleaned_mask = self._apply_morphology(binary_mask)
            
            # Find contours
            contours, _ = cv2.findContours(
                cleaned_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            self.stats["total_contours_found"] += len(contours)
            
            # Process each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by minimum area
                if area < self.min_contour_area:
                    self.stats["contours_filtered_by_area"] += 1
                    continue
                
                # Simplify polygon
                simplified = self._simplify_polygon(contour)
                
                # Need at least 3 vertices for a valid polygon
                if len(simplified) < 3:
                    continue
                
                # Extract vertices as list
                vertices = simplified.squeeze().tolist()
                if not isinstance(vertices[0], list):
                    vertices = [vertices]
                
                # Ensure CCW winding
                vertices_array = np.array(vertices)
                vertices_array = self._ensure_ccw_winding(vertices_array)
                vertices = vertices_array.tolist()
                
                # Calculate metrics
                perimeter = cv2.arcLength(simplified, True)
                moments = cv2.moments(simplified)
                
                if moments["m00"] > 0:
                    cx = moments["m10"] / moments["m00"]
                    cy = moments["m01"] / moments["m00"]
                else:
                    cx, cy = vertices_array.mean(axis=0)
                
                bbox = cv2.boundingRect(simplified)
                
                # Quality analysis
                quality = self._analyze_polygon_quality(simplified)
                
                # Create polygon object
                polygon = ExtractedPolygon(
                    polygon_id=polygon_id,
                    class_name=class_name,
                    class_id=class_id,
                    vertices=vertices,
                    area=float(area),
                    perimeter=float(perimeter),
                    centroid=(float(cx), float(cy)),
                    bounding_box=bbox,
                    vertex_count=len(vertices),
                    is_convex=quality["is_convex"],
                    convexity_ratio=quality["convexity_ratio"],
                    solidity=quality["solidity"],
                    color=CLASS_COLORS.get(class_name, (200, 200, 200)),
                    needs_ear_clipping=not quality["is_convex"] or quality["solidity"] < 0.85
                )
                
                self.polygons.append(polygon)
                polygon_id += 1
                
                # Update stats
                self.stats["valid_polygons"] += 1
                if quality["is_convex"]:
                    self.stats["convex_polygons"] += 1
                else:
                    self.stats["non_convex_polygons"] += 1
        
        logger.info(f"Extracted {len(self.polygons)} valid polygons")
        logger.info(f"  Filtered {self.stats['contours_filtered_by_area']} noise contours")
        logger.info(f"  Convex: {self.stats['convex_polygons']}, Non-convex: {self.stats['non_convex_polygons']}")
        
        return self.polygons
    
    def detect_walls(self) -> List[Dict]:
        """
        Detect wall segments between rooms.
        
        Note: This is a simplified wall detection. For thesis purposes,
        the room polygons themselves define the walls.
        """
        if self.mask is None:
            raise ValueError("Mask not loaded")
        
        self.walls = []
        
        # Find edges using gradient
        sobelx = cv2.Sobel(self.mask, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.mask, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        _, edges = cv2.threshold(gradient, 0.5, 255, cv2.THRESH_BINARY)
        edges = edges.astype(np.uint8)
        
        # Find edge contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            length = cv2.arcLength(contour, False)
            if length < 20:  # Filter tiny segments
                continue
            
            points = contour.squeeze()
            if len(points.shape) == 1 or len(points) < 2:
                continue
            
            self.walls.append({
                "p1": [float(points[0][0]), float(points[0][1])],
                "p2": [float(points[-1][0]), float(points[-1][1])],
                "length": float(length)
            })
        
        logger.info(f"Detected {len(self.walls)} wall segments")
        return self.walls
    
    def calculate_scale_factor(self, 
                               reference_room_meters: float = 4.0,
                               use_median: bool = True) -> float:
        """
        Calculate scale factor (meters per pixel) based on room sizes.
        
        Args:
            reference_room_meters: Assumed average room width in meters
            use_median: Use median instead of mean (more robust to outliers)
        
        Returns:
            Scale factor (meters per pixel)
        """
        if not self.polygons:
            logger.warning("No polygons extracted, using default scale")
            self.scale_factor = 0.01  # 1cm per pixel
            return self.scale_factor
        
        # Get room widths
        widths = []
        for poly in self.polygons:
            if poly.class_name in ROOM_CLASSES:
                x, y, w, h = poly.bounding_box
                widths.append(max(w, h))  # Use larger dimension
        
        if not widths:
            self.scale_factor = 0.01
            return self.scale_factor
        
        # Calculate representative width
        if use_median:
            representative_width = np.median(widths)
        else:
            representative_width = np.mean(widths)
        
        self.scale_factor = reference_room_meters / representative_width
        
        logger.info(f"Scale factor: {self.scale_factor:.6f} m/px "
                   f"(reference: {representative_width:.0f}px = {reference_room_meters}m)")
        
        return self.scale_factor
    
    def get_rooms_for_extrusion(self) -> List[Dict]:
        """
        Get room data in format expected by wall extrusion.
        
        Returns:
            List of room dicts with vertices, color, class info
        """
        rooms = []
        for poly in self.polygons:
            rooms.append({
                "polygon_id": poly.polygon_id,
                "class": poly.class_name,
                "class_id": poly.class_id,
                "vertices": poly.vertices,
                "color": list(poly.color),
                "area": poly.area,
                "is_convex": poly.is_convex,
                "needs_ear_clipping": poly.needs_ear_clipping,
                "centroid": list(poly.centroid),
            })
        return rooms
    
    def export_json(self, output_path: str):
        """Export boundaries to JSON file."""
        data = {
            "rooms": self.get_rooms_for_extrusion(),
            "walls": self.walls,
            "scale_factor": self.scale_factor,
            "metadata": {
                "mask_shape": list(self.mask_shape) if self.mask_shape else None,
                "total_rooms": len(self.polygons),
                "total_walls": len(self.walls),
                "extraction_params": {
                    "min_contour_area": self.min_contour_area,
                    "contour_epsilon": self.contour_approximation_epsilon,
                    "morphology_enabled": self.enable_morphology,
                },
                "stats": self.stats,
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported boundaries to {output_path}")
    
    def visualize(self, output_path: str, show_labels: bool = True):
        """Create visualization of extracted polygons."""
        if self.mask is None:
            raise ValueError("Mask not loaded")
        
        # Create colored base
        h, w = self.mask_shape
        viz = np.zeros((h, w, 3), dtype=np.uint8)
        viz[:] = (50, 50, 50)
        
        # Fill rooms with colors
        for poly in self.polygons:
            vertices = np.array(poly.vertices, dtype=np.int32)
            color_bgr = (poly.color[2], poly.color[1], poly.color[0])
            cv2.fillPoly(viz, [vertices], color_bgr)
        
        # Draw boundaries
        for poly in self.polygons:
            vertices = np.array(poly.vertices, dtype=np.int32)
            # Green for convex, orange for non-convex
            if poly.is_convex:
                border_color = (0, 200, 0)
            else:
                border_color = (0, 165, 255)
            cv2.polylines(viz, [vertices], True, border_color, 2)
        
        # Add labels
        if show_labels:
            for poly in self.polygons:
                cx, cy = int(poly.centroid[0]), int(poly.centroid[1])
                label = f"#{poly.polygon_id} {poly.class_name[:4]}"
                
                # Background for readability
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(viz, (cx-2, cy-th-2), (cx+tw+2, cy+2), (0, 0, 0), -1)
                cv2.putText(viz, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imwrite(output_path, viz)
        logger.info(f"Saved visualization to {output_path}")
    
    def process(self, 
                mask_path: str, 
                image_path: Optional[str] = None,
                output_dir: str = "./output") -> Dict:
        """
        Full extraction pipeline.
        
        Args:
            mask_path: Path to segmentation mask
            image_path: Path to original image (optional)
            output_dir: Output directory
        
        Returns:
            Dict with rooms, walls, scale_factor
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(mask_path).stem
        
        # Load
        self.load_mask(mask_path, image_path)
        
        # Extract
        self.extract_polygons()
        self.detect_walls()
        self.calculate_scale_factor()
        
        # Export
        self.export_json(str(output_dir / f"{base_name}_boundaries_v2.json"))
        self.visualize(str(output_dir / f"{base_name}_boundaries_v2.png"))
        
        return {
            "rooms": self.get_rooms_for_extrusion(),
            "walls": self.walls,
            "scale_factor": self.scale_factor,
        }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract room boundaries from segmentation mask")
    parser.add_argument("--mask", "-m", required=True, help="Path to segmentation mask")
    parser.add_argument("--image", "-i", help="Path to original floorplan image")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--min-area", type=int, default=3000, help="Minimum contour area")
    
    args = parser.parse_args()
    
    extractor = BoundaryExtractorV2(min_contour_area=args.min_area)
    result = extractor.process(args.mask, args.image, args.output)
    
    print(f"\n✓ Extracted {len(result['rooms'])} rooms")
    print(f"✓ Scale factor: {result['scale_factor']:.6f} m/px")