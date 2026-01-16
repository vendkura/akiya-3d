"""
Segmentation Post-Processor
===========================
Cleans up FPN segmentation masks before polygon extraction.

Fixes:
1. Jagged boundaries → Morphological smoothing
2. Small holes inside rooms → Hole filling
3. Small fragments → Removal/merging
4. Wobbly edges → Line regularization (straightening)

Usage:
    from segmentation_postprocess import SegmentationPostProcessor
    
    processor = SegmentationPostProcessor()
    clean_mask = processor.process(raw_mask)

Author: Asheleyine's Master thesis project
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from scipy import ndimage
from pathlib import Path
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

# Classes that represent rooms (should have clean rectangular shapes)
ROOM_CLASSES = {1, 2, 3, 4, 5, 7, 8, 9, 13}  # dining, bath, bed, closet, room, entrance, kitchen, outdoor, balcony

# Classes that are thin features (doors, windows) - different processing
FEATURE_CLASSES = {6, 10, 12}  # door, sliding_door, window

# Stairs - special handling
STAIR_CLASSES = {11}


class SegmentationPostProcessor:
    """
    Post-processes segmentation masks to improve polygon extraction quality.
    
    Pipeline:
    1. Per-class morphological cleanup
    2. Small region removal
    3. Hole filling
    4. Boundary smoothing
    5. Optional: Edge straightening
    """
    
    def __init__(self,
                 min_region_area: int = 2000,
                 smoothing_kernel_size: int = 5,
                 hole_fill_threshold: int = 1000,
                 enable_straightening: bool = True,
                 straightening_tolerance: float = 0.02):
        """
        Initialize post-processor.
        
        Args:
            min_region_area: Minimum pixels for a valid region (smaller = removed)
            smoothing_kernel_size: Kernel size for morphological smoothing
            hole_fill_threshold: Maximum hole size to fill (pixels)
            enable_straightening: Whether to straighten edges
            straightening_tolerance: Tolerance for line fitting (fraction of perimeter)
        """
        self.min_region_area = min_region_area
        self.smoothing_kernel_size = smoothing_kernel_size
        self.hole_fill_threshold = hole_fill_threshold
        self.enable_straightening = enable_straightening
        self.straightening_tolerance = straightening_tolerance
        
        # Statistics
        self.stats = {
            "regions_removed": 0,
            "holes_filled": 0,
            "pixels_changed": 0,
        }
    
    def process(self, mask: np.ndarray) -> np.ndarray:
        """
        Full post-processing pipeline.
        
        Args:
            mask: Input segmentation mask (H, W) with values 0-13
            
        Returns:
            Cleaned mask with same shape and value range
        """
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
        
        logger.info(f"Post-processing mask {mask.shape}...")
        self.stats = {k: 0 for k in self.stats}
        
        original_mask = mask.copy()
        clean_mask = np.zeros_like(mask)
        
        # Process each class separately
        unique_classes = np.unique(mask)
        logger.info(f"  Classes present: {[CLASS_NAMES.get(c, c) for c in unique_classes if c != 0]}")
        
        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue
            
            # Extract binary mask for this class
            binary = (mask == class_id).astype(np.uint8)
            
            # Apply appropriate processing based on class type
            if class_id in ROOM_CLASSES:
                processed = self._process_room_class(binary, class_id)
            elif class_id in FEATURE_CLASSES:
                processed = self._process_feature_class(binary, class_id)
            elif class_id in STAIR_CLASSES:
                processed = self._process_stair_class(binary, class_id)
            else:
                processed = self._process_generic_class(binary, class_id)
            
            # Merge back - only where not already assigned
            # (handles overlaps by giving priority to first-processed)
            update_mask = (processed > 0) & (clean_mask == 0)
            clean_mask[update_mask] = class_id
        
        # Calculate change statistics
        self.stats["pixels_changed"] = int(np.sum(original_mask != clean_mask))
        change_percent = 100 * self.stats["pixels_changed"] / mask.size
        
        logger.info(f"  ✓ Post-processing complete")
        logger.info(f"    Regions removed: {self.stats['regions_removed']}")
        logger.info(f"    Holes filled: {self.stats['holes_filled']}")
        logger.info(f"    Pixels changed: {self.stats['pixels_changed']} ({change_percent:.1f}%)")
        
        return clean_mask
    
    def _process_room_class(self, binary: np.ndarray, class_id: int) -> np.ndarray:
        """
        Process room-type classes (should be clean, rectangular-ish).
        
        Steps:
        1. Remove small fragments
        2. Fill holes
        3. Smooth boundaries
        4. Optional: Straighten edges
        """
        result = binary.copy()
        
        # Step 1: Remove small fragments
        result = self._remove_small_regions(result, self.min_region_area)
        
        # Step 2: Fill small holes
        result = self._fill_holes(result, self.hole_fill_threshold)
        
        # Step 3: Morphological smoothing (close then open)
        result = self._smooth_boundaries(result)
        
        # Step 4: Straighten edges (optional)
        if self.enable_straightening:
            result = self._straighten_edges(result)
        
        return result
    
    def _process_feature_class(self, binary: np.ndarray, class_id: int) -> np.ndarray:
        """
        Process feature classes (doors, windows) - thin elements.
        
        Lighter processing to preserve thin shapes.
        """
        result = binary.copy()
        
        # Only remove very small fragments
        result = self._remove_small_regions(result, self.min_region_area // 4)
        
        # Light smoothing with smaller kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return result
    
    def _process_stair_class(self, binary: np.ndarray, class_id: int) -> np.ndarray:
        """
        Process stairs - can have complex shapes.
        """
        result = binary.copy()
        
        # Remove small fragments
        result = self._remove_small_regions(result, self.min_region_area // 2)
        
        # Fill holes
        result = self._fill_holes(result, self.hole_fill_threshold // 2)
        
        # Light smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return result
    
    def _process_generic_class(self, binary: np.ndarray, class_id: int) -> np.ndarray:
        """
        Generic processing for unknown classes.
        """
        result = binary.copy()
        result = self._remove_small_regions(result, self.min_region_area)
        result = self._fill_holes(result, self.hole_fill_threshold)
        result = self._smooth_boundaries(result)
        return result
    
    def _remove_small_regions(self, binary: np.ndarray, min_area: int) -> np.ndarray:
        """
        Remove connected components smaller than min_area.
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        result = np.zeros_like(binary)
        
        for label_id in range(1, num_labels):  # Skip background (0)
            area = stats[label_id, cv2.CC_STAT_AREA]
            
            if area >= min_area:
                result[labels == label_id] = 1
            else:
                self.stats["regions_removed"] += 1
        
        return result
    
    def _fill_holes(self, binary: np.ndarray, max_hole_size: int) -> np.ndarray:
        """
        Fill small holes inside regions.
        """
        # Invert to find holes
        inverted = 1 - binary
        
        # Find connected components in inverted (these are holes + background)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            inverted, connectivity=8
        )
        
        result = binary.copy()
        
        # The largest component in inverted is the background, skip it
        if num_labels <= 1:
            return result
        
        # Find background label (largest component)
        areas = stats[1:, cv2.CC_STAT_AREA]  # Skip label 0
        background_label = np.argmax(areas) + 1
        
        # Fill small holes (not background)
        for label_id in range(1, num_labels):
            if label_id == background_label:
                continue
            
            area = stats[label_id, cv2.CC_STAT_AREA]
            
            if area <= max_hole_size:
                result[labels == label_id] = 1
                self.stats["holes_filled"] += 1
        
        return result
    
    def _smooth_boundaries(self, binary: np.ndarray) -> np.ndarray:
        """
        Smooth jagged boundaries using morphological operations.
        
        Uses closing followed by opening to:
        - Close small gaps and smooth outward bumps
        - Remove small protrusions and smooth inward bumps
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.smoothing_kernel_size, self.smoothing_kernel_size)
        )
        
        # Close: fills small gaps, connects nearby regions
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Open: removes small protrusions
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return opened
    
    def _straighten_edges(self, binary: np.ndarray) -> np.ndarray:
        """
        Straighten edges by fitting lines to boundary segments.
        
        For architectural floor plans, most edges should be:
        - Horizontal
        - Vertical  
        - At 45° angles (rare)
        
        This function detects dominant edge orientations and snaps to them.
        """
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return binary
        
        result = np.zeros_like(binary)
        
        for contour in contours:
            if len(contour) < 4:
                cv2.drawContours(result, [contour], -1, 1, -1)
                continue
            
            # Approximate polygon
            perimeter = cv2.arcLength(contour, True)
            epsilon = self.straightening_tolerance * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Straighten each edge
            straightened = self._straighten_polygon(approx)
            
            # Draw filled polygon
            cv2.drawContours(result, [straightened], -1, 1, -1)
        
        return result
    
    def _straighten_polygon(self, polygon: np.ndarray) -> np.ndarray:
        """
        Straighten polygon edges by snapping to horizontal/vertical.
        """
        points = polygon.squeeze()
        if len(points.shape) == 1:
            return polygon
        
        n = len(points)
        straightened = []
        
        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            
            # Calculate edge direction
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Snap to horizontal or vertical if close
            angle = np.arctan2(abs(dy), abs(dx)) * 180 / np.pi
            
            # Threshold for snapping (15 degrees)
            snap_threshold = 15
            
            if angle < snap_threshold:
                # Nearly horizontal - snap to horizontal
                new_p1 = p1.copy()
                # Keep p1, adjust will happen at next vertex
            elif angle > (90 - snap_threshold):
                # Nearly vertical - snap to vertical
                new_p1 = p1.copy()
            else:
                # Diagonal - keep as is
                new_p1 = p1.copy()
            
            straightened.append(new_p1)
        
        # Now snap vertices to create clean horizontal/vertical edges
        straightened = np.array(straightened)
        straightened = self._snap_vertices(straightened)
        
        return straightened.reshape(-1, 1, 2).astype(np.int32)
    
    def _snap_vertices(self, points: np.ndarray) -> np.ndarray:
        """
        Snap vertices to create cleaner horizontal/vertical edges.
        """
        n = len(points)
        result = points.copy().astype(float)
        
        for i in range(n):
            p_prev = result[(i - 1) % n]
            p_curr = result[i]
            p_next = result[(i + 1) % n]
            
            # Check incoming edge
            dx_in = p_curr[0] - p_prev[0]
            dy_in = p_curr[1] - p_prev[1]
            
            # Check outgoing edge
            dx_out = p_next[0] - p_curr[0]
            dy_out = p_next[1] - p_curr[1]
            
            # Snap threshold (pixels)
            snap_px = 10
            
            # If incoming is nearly horizontal, snap y to match
            if abs(dy_in) < snap_px and abs(dx_in) > snap_px:
                result[i][1] = p_prev[1]
            
            # If incoming is nearly vertical, snap x to match
            if abs(dx_in) < snap_px and abs(dy_in) > snap_px:
                result[i][0] = p_prev[0]
        
        return result.astype(np.int32)
    
    def visualize_comparison(self, 
                             original: np.ndarray, 
                             processed: np.ndarray,
                             output_path: str):
        """
        Create side-by-side comparison visualization.
        """
        # Colorize masks
        original_color = self._colorize_mask(original)
        processed_color = self._colorize_mask(processed)
        
        # Create difference map
        diff = (original != processed).astype(np.uint8) * 255
        diff_color = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        diff_color[diff > 0] = [0, 0, 255]  # Red for changes
        
        # Stack horizontally
        h, w = original.shape
        comparison = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
        comparison[:, :w] = original_color
        comparison[:, w+10:w*2+10] = processed_color
        comparison[:, w*2+20:] = diff_color
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Processed", (w + 20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Changes", (w * 2 + 30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, comparison)
        logger.info(f"Saved comparison to {output_path}")
    
    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create colored visualization of mask."""
        colors = {
            0: (50, 50, 50),
            1: (193, 182, 255),   # dining - pink
            2: (124, 200, 255),   # bathroom - orange
            3: (230, 216, 173),   # bedroom - light blue
            4: (181, 228, 255),   # closet - bisque
            5: (224, 255, 255),   # room - light yellow
            6: (45, 82, 160),     # door - brown
            7: (240, 255, 240),   # entrance - honeydew
            8: (144, 238, 144),   # kitchen - light green
            9: (152, 251, 152),   # outdoor - pale green
            10: (169, 169, 169),  # sliding_door - gray
            11: (211, 211, 211),  # stairs - light gray
            12: (250, 206, 135),  # window - sky blue
            13: (230, 224, 176),  # balcony - powder blue
        }
        
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in colors.items():
            colored[mask == class_id] = color
        
        return colored


# ============================================================================
# ADVANCED: RECTANGULAR FITTING
# ============================================================================

class RectangularFitter:
    """
    Fits rectangular approximations to room polygons.
    
    Japanese floor plans typically have very rectangular rooms.
    This class attempts to find the best-fitting rectangle or
    union of rectangles for each room.
    """
    
    @staticmethod
    def fit_minimum_bounding_rect(contour: np.ndarray) -> np.ndarray:
        """
        Fit minimum area rotated rectangle.
        """
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return np.int32(box)
    
    @staticmethod
    def fit_axis_aligned_rect(contour: np.ndarray) -> np.ndarray:
        """
        Fit axis-aligned bounding rectangle.
        """
        x, y, w, h = cv2.boundingRect(contour)
        return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
    
    @staticmethod
    def fit_rectilinear_polygon(binary: np.ndarray, 
                                 simplification: float = 0.02) -> np.ndarray:
        """
        Fit a rectilinear (axis-aligned edges only) polygon.
        
        This is useful for L-shaped or T-shaped rooms where a single
        rectangle doesn't fit well.
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return binary
        
        result = np.zeros_like(binary)
        
        for contour in contours:
            # Approximate polygon
            perimeter = cv2.arcLength(contour, True)
            epsilon = simplification * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Snap all vertices to axis-aligned grid
            snapped = RectangularFitter._snap_to_rectilinear(approx)
            
            cv2.drawContours(result, [snapped], -1, 1, -1)
        
        return result
    
    @staticmethod
    def _snap_to_rectilinear(polygon: np.ndarray) -> np.ndarray:
        """
        Snap polygon vertices to create only horizontal/vertical edges.
        """
        points = polygon.squeeze()
        if len(points) < 3:
            return polygon
        
        n = len(points)
        result = []
        
        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            
            result.append(p1.tolist())
            
            # If edge is diagonal, add intermediate point to make it rectilinear
            dx = abs(p2[0] - p1[0])
            dy = abs(p2[1] - p1[1])
            
            if dx > 5 and dy > 5:  # Diagonal edge
                # Add corner point (horizontal then vertical)
                result.append([p2[0], p1[1]])
        
        return np.array(result, dtype=np.int32).reshape(-1, 1, 2)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def postprocess_mask(mask: np.ndarray,
                     min_region_area: int = 2000,
                     enable_straightening: bool = True) -> np.ndarray:
    """
    Convenience function to post-process a segmentation mask.
    
    Args:
        mask: Input mask (H, W) with class values 0-13
        min_region_area: Minimum area for valid regions
        enable_straightening: Whether to straighten edges
    
    Returns:
        Cleaned mask
    """
    processor = SegmentationPostProcessor(
        min_region_area=min_region_area,
        enable_straightening=enable_straightening
    )
    return processor.process(mask)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Post-process segmentation masks")
    parser.add_argument("--input", "-i", required=True, help="Input mask image")
    parser.add_argument("--output", "-o", required=True, help="Output mask image")
    parser.add_argument("--min-area", type=int, default=2000, help="Minimum region area")
    parser.add_argument("--no-straighten", action="store_true", help="Disable edge straightening")
    parser.add_argument("--compare", "-c", help="Output comparison image path")
    
    args = parser.parse_args()
    
    # Load mask
    mask = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Cannot load {args.input}")
        exit(1)
    
    # Process
    processor = SegmentationPostProcessor(
        min_region_area=args.min_area,
        enable_straightening=not args.no_straighten
    )
    clean_mask = processor.process(mask)
    
    # Save
    cv2.imwrite(args.output, clean_mask)
    print(f"✓ Saved cleaned mask to {args.output}")
    
    # Optional comparison
    if args.compare:
        processor.visualize_comparison(mask, clean_mask, args.compare)
