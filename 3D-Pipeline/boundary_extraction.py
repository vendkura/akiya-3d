"""
Boundary Extraction from FPN Segmentation Masks

Extracts room polygons and wall segments from segmentation masks.
Outputs JSON with room geometry and visualization overlays.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Room class names (13 classes)
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

# Color mapping for visualization (per room type)
CLASS_COLORS = {
    "dining_area": (255, 182, 193),      # Light Pink
    "bathroom": (255, 200, 124),         # Light Orange
    "bedroom": (173, 216, 230),          # Light Blue
    "closet": (255, 228, 181),           # Bisque
    "room": (255, 255, 224),             # Light Yellow
    "door": (160, 82, 45),               # Brown
    "entrance": (240, 255, 240),         # Honeydew
    "kitchen": (144, 238, 144),          # Light Green
    "outdoor_space": (152, 251, 152),    # Pale Green
    "sliding_door": (169, 169, 169),     # Dark Gray
    "stairs": (211, 211, 211),           # Light Gray
    "window": (135, 206, 250),           # Sky Blue
    "balcony": (176, 224, 230),          # Powder Blue
}


class BoundaryExtractor:
    """Extract room boundaries and walls from segmentation masks."""
    
    def __init__(self, 
                 min_contour_area: int = 100,
                 contour_approximation_epsilon: float = 0.02,
                 wall_detection_kernel_size: int = 3):
        """
        Initialize boundary extractor.
        
        Args:
            min_contour_area: Minimum pixels to consider as valid room (filters noise)
            contour_approximation_epsilon: Douglas-Peucker epsilon (0.02 = 2% of perimeter)
            wall_detection_kernel_size: Kernel size for wall detection
        """
        self.min_contour_area = min_contour_area
        self.contour_approximation_epsilon = contour_approximation_epsilon
        self.wall_detection_kernel_size = wall_detection_kernel_size
        
        self.rooms = []
        self.walls = []
        self.mask = None
        self.original_image = None
        self.scale_factor = 1.0  # pixels to meters
        
    def load_mask_and_image(self, mask_path: str, image_path: str = None):
        """
        Load segmentation mask and original image.
        
        Args:
            mask_path: Path to segmentation mask (single channel, pixel values 0-13)
            image_path: Path to original floor plan image
        """
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            raise ValueError(f"Cannot load mask: {mask_path}")
        
        if image_path:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                logger.warning(f"Cannot load image: {image_path}, using mask only")
        else:
            self.original_image = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        
        logger.info(f"Loaded mask shape: {self.mask.shape}, Image shape: {self.original_image.shape}")
    
    def load_mask_array(self, mask_array: np.ndarray):
        """
        Load mask from numpy array directly (no file I/O).
        
        Args:
            mask_array: Numpy array with shape (H, W), values 0-13
        """
        if not isinstance(mask_array, np.ndarray):
            raise TypeError("mask_array must be a numpy array")
        
        if mask_array.ndim != 2:
            raise ValueError(f"mask_array must be 2D, got shape {mask_array.shape}")
        
        self.mask = mask_array.astype(np.uint8)
        self.original_image = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        
        logger.info(f"Loaded mask array with shape: {self.mask.shape}")
    
    def extract_room_polygons(self) -> Dict:
        """
        Extract room polygons for each class.
        
        Returns:
            Dict with room polygons: {"rooms": [{"class": str, "vertices": [[x,y], ...], "color": (r,g,b)}]}
        """
        if self.mask is None:
            raise ValueError("Mask not loaded. Call load_mask_and_image() first.")
        
        self.rooms = []
        
        # Process each class (skip 0=background)
        for class_id in range(1, 14):
            class_name = CLASS_NAMES.get(class_id, f"unknown_{class_id}")
            
            # Create binary mask for this class
            binary_mask = (self.mask == class_id).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # Process each contour (multiple rooms of same type possible)
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter small contours (noise)
                if area < self.min_contour_area:
                    continue
                
                # Simplify contour using Douglas-Peucker
                perimeter = cv2.arcLength(contour, True)
                epsilon = self.contour_approximation_epsilon * perimeter
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                
                # Convert to list of [x, y] coordinates
                vertices = simplified.squeeze().tolist()
                
                # Handle case where simplified becomes single point
                if not isinstance(vertices[0], list):
                    vertices = [vertices]
                
                if len(vertices) >= 3:  # Valid polygon needs at least 3 points
                    color = CLASS_COLORS.get(class_name, (200, 200, 200))
                    
                    self.rooms.append({
                        "class": class_name,
                        "class_id": class_id,
                        "vertices": vertices,
                        "color": color,
                        "area": float(area)
                    })
                    
                    logger.debug(f"Extracted {class_name} polygon with {len(vertices)} vertices, area {area}")
        
        logger.info(f"Extracted {len(self.rooms)} room polygons")
        return {"rooms": self.rooms}
    
    def detect_walls(self) -> Dict:
        """
        Detect wall segments between rooms (boundaries between different classes).
        
        Returns:
            Dict with wall segments: {"walls": [{"p1": [x,y], "p2": [x,y], "length": float}]}
        """
        if self.mask is None:
            raise ValueError("Mask not loaded. Call load_mask_and_image() first.")
        
        self.walls = []
        
        # Find edges using Sobel (gradient magnitude)
        # Walls are where adjacent pixels have different class values
        sobelx = cv2.Sobel(self.mask, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.mask, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        # Threshold to get edges
        _, edges = cv2.threshold(gradient, 0.5, 255, cv2.THRESH_BINARY)
        edges = edges.astype(np.uint8)
        
        # Find contours of edges (wall segments)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            length = cv2.arcLength(contour, False)
            
            # Filter very small wall segments
            if length < 5:
                continue
            
            # Get endpoints
            points = contour.squeeze()
            if len(points.shape) == 1:
                continue
            
            # Simplified: use first and last point
            p1 = [float(points[0][0]), float(points[0][1])]
            p2 = [float(points[-1][0]), float(points[-1][1])]
            
            self.walls.append({
                "p1": p1,
                "p2": p2,
                "length": float(length)
            })
        
        logger.info(f"Detected {len(self.walls)} wall segments")
        return {"walls": self.walls}
    
    def calculate_scale_factor(self, assumed_room_width_m: float = 4.0) -> float:
        """
        Calculate dynamic scale factor from room dimensions.
        
        Assumes average room width = assumed_room_width_m meters
        Measures average room width in pixels and calculates scale.
        
        Args:
            assumed_room_width_m: Assumed real-world room width (meters)
        
        Returns:
            Scale factor (meters per pixel)
        """
        if not self.rooms:
            logger.warning("No rooms extracted. Scale factor set to 1.0 (1 pixel = 1 cm)")
            self.scale_factor = 0.01  # 1 pixel = 1 cm
            return self.scale_factor
        
        # Calculate bounding box width for each room
        widths = []
        for room in self.rooms:
            vertices = np.array(room["vertices"])
            min_x, max_x = vertices[:, 0].min(), vertices[:, 0].max()
            width_px = max_x - min_x
            if width_px > 0:
                widths.append(width_px)
        
        if widths:
            avg_width_px = np.mean(widths)
            # scale_factor = real_world_meters / pixel_width
            self.scale_factor = assumed_room_width_m / avg_width_px
            logger.info(f"Calculated scale: {self.scale_factor:.6f} m/pixel (avg room width {avg_width_px:.1f} px = {assumed_room_width_m} m)")
        else:
            self.scale_factor = 0.01
            logger.warning("Could not calculate scale factor, defaulting to 0.01 m/pixel")
        
        return self.scale_factor
    
    def visualize_boundaries(self, output_path: str = "boundaries.png"):
        """
        Create visualization with boundaries overlaid on original image.
        
        Args:
            output_path: Path to save visualization PNG
        """
        if self.original_image is None:
            logger.warning("No original image available for visualization")
            return
        
        viz = self.original_image.copy()
        overlay = viz.copy()
        
        # Draw room polygons with colors on overlay
        for room in self.rooms:
            vertices = np.array(room["vertices"], dtype=np.int32)
            color = room["color"]
            color_bgr = (color[2], color[1], color[0])  # Convert RGB to BGR for OpenCV
            
            cv2.fillPoly(overlay, [vertices], color_bgr)
            cv2.polylines(overlay, [vertices], True, color_bgr, 2)
        
        # Blend overlay with original (alpha blending)
        alpha = 0.3
        viz = cv2.addWeighted(overlay, alpha, viz, 1 - alpha, 0)
        
        # Draw walls on top
        for wall in self.walls:
            p1 = tuple(map(int, wall["p1"]))
            p2 = tuple(map(int, wall["p2"]))
            cv2.line(viz, p1, p2, (0, 0, 0), 2)  # Black walls
        
        cv2.imwrite(output_path, viz)
        logger.info(f"Saved visualization to {output_path}")
    
    def export_json(self, output_path: str = "boundaries.json"):
        """
        Export room and wall data as JSON.
        
        Args:
            output_path: Path to save JSON file
        """
        data = {
            "rooms": self.rooms,
            "walls": self.walls,
            "scale_factor": self.scale_factor,
            "metadata": {
                "total_rooms": len(self.rooms),
                "total_walls": len(self.walls),
                "mask_size": list(self.mask.shape) if self.mask is not None else None
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved boundaries to {output_path}")
    
    def process(self, mask_path: str, image_path: str = None, output_dir: str = "./output"):
        """
        Full pipeline: extract boundaries and save outputs.
        
        Args:
            mask_path: Path to segmentation mask
            image_path: Path to original floor plan image
            output_dir: Directory to save outputs
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load inputs
        self.load_mask_and_image(mask_path, image_path)
        
        # Extract boundaries
        self.extract_room_polygons()
        self.detect_walls()
        
        # Calculate scale
        self.calculate_scale_factor()
        
        # Save outputs
        base_name = Path(mask_path).stem
        self.visualize_boundaries(f"{output_dir}/{base_name}_boundaries.png")
        self.export_json(f"{output_dir}/{base_name}_boundaries.json")
        
        logger.info(f"Boundary extraction complete for {base_name}")
        
        return {
            "rooms": self.rooms,
            "walls": self.walls,
            "scale_factor": self.scale_factor
        }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python boundary_extraction.py <mask_path> [image_path] [output_dir]")
        sys.exit(1)
    
    mask_path = sys.argv[1]
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "./output"
    
    extractor = BoundaryExtractor()
    extractor.process(mask_path, image_path, output_dir)
