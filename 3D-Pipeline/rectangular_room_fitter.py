"""
Rectangular Room Fitter
=======================
Converts blob-like room segmentations into clean rectangular shapes.

Key principle: CONSTRAINED FITTING
- Only simplify shapes, never expand
- Find largest rectangle that fits INSIDE each room
- Guarantees no overlaps between rooms

Supports:
- Simple rectangles (most rooms)
- L-shapes (corner rooms, merged spaces)
- Rectilinear polygons (only H/V edges)

Author: Asheleyine's Master thesis project
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FittedRoom:
    """Result of rectangular fitting for one room."""
    class_id: int
    original_area: int
    fitted_area: int
    coverage: float  # fitted_area / original_area
    fit_type: str  # 'rectangle', 'l_shape', 'rectilinear', 'original'
    polygon: np.ndarray  # The fitted polygon vertices


# ============================================================================
# LARGEST INSCRIBED RECTANGLE FINDER
# ============================================================================

class LargestInscribedRectangle:
    """
    Finds the largest axis-aligned rectangle that fits inside a binary mask.
    
    Uses the histogram method for O(n*m) complexity.
    """
    
    @staticmethod
    def find(binary_mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Find largest inscribed axis-aligned rectangle.
        
        Args:
            binary_mask: 2D binary array (1 = inside room, 0 = outside)
        
        Returns:
            (x, y, width, height) of the largest rectangle
        """
        if binary_mask.sum() == 0:
            return (0, 0, 0, 0)
        
        rows, cols = binary_mask.shape
        
        # Build histogram of consecutive 1s in each column
        heights = np.zeros((rows, cols), dtype=np.int32)
        heights[0] = binary_mask[0]
        
        for i in range(1, rows):
            for j in range(cols):
                if binary_mask[i, j] == 1:
                    heights[i, j] = heights[i-1, j] + 1
                else:
                    heights[i, j] = 0
        
        # Find max rectangle in histogram for each row
        max_area = 0
        best_rect = (0, 0, 0, 0)
        
        for i in range(rows):
            rect = LargestInscribedRectangle._max_rect_in_histogram(heights[i], i)
            x, y, w, h = rect
            area = w * h
            
            if area > max_area:
                max_area = area
                best_rect = rect
        
        return best_rect
    
    @staticmethod
    def _max_rect_in_histogram(heights: np.ndarray, row_idx: int) -> Tuple[int, int, int, int]:
        """
        Find largest rectangle in histogram using stack method.
        
        Returns:
            (x, y, width, height) where y is the TOP of the rectangle
        """
        n = len(heights)
        stack = []  # Stack of (index, height)
        max_area = 0
        best_rect = (0, 0, 0, 0)
        
        for i in range(n + 1):
            h = heights[i] if i < n else 0
            start = i
            
            while stack and stack[-1][1] > h:
                idx, height = stack.pop()
                width = i - idx
                area = width * height
                
                if area > max_area:
                    max_area = area
                    # y is the top of rectangle (row_idx - height + 1)
                    best_rect = (idx, row_idx - height + 1, width, height)
                
                start = idx
            
            stack.append((start, h))
        
        return best_rect


# ============================================================================
# L-SHAPE FITTER
# ============================================================================

class LShapeFitter:
    """
    Fits L-shaped (or rectilinear) polygons to non-rectangular rooms.
    
    For rooms that don't fit well into a single rectangle (e.g., L-shaped
    rooms, corner rooms), this finds the best rectilinear approximation.
    """
    
    @staticmethod
    def fit(binary_mask: np.ndarray, min_coverage: float = 0.7) -> Optional[np.ndarray]:
        """
        Try to fit an L-shape to the binary mask.
        
        Strategy: Find two rectangles that together cover most of the area.
        
        Args:
            binary_mask: Binary room mask
            min_coverage: Minimum coverage required
        
        Returns:
            Polygon vertices if successful, None otherwise
        """
        # Find the largest inscribed rectangle first
        x1, y1, w1, h1 = LargestInscribedRectangle.find(binary_mask)
        
        if w1 == 0 or h1 == 0:
            return None
        
        rect1_area = w1 * h1
        total_area = binary_mask.sum()
        
        # If single rectangle covers enough, use it
        if rect1_area / total_area >= min_coverage:
            return np.array([
                [x1, y1],
                [x1 + w1, y1],
                [x1 + w1, y1 + h1],
                [x1, y1 + h1]
            ], dtype=np.int32)
        
        # Create mask without first rectangle
        remaining = binary_mask.copy()
        remaining[y1:y1+h1, x1:x1+w1] = 0
        
        # Find second largest rectangle in remaining area
        x2, y2, w2, h2 = LargestInscribedRectangle.find(remaining)
        
        if w2 == 0 or h2 == 0:
            # No good second rectangle, return first one if decent
            if rect1_area / total_area >= 0.5:
                return np.array([
                    [x1, y1],
                    [x1 + w1, y1],
                    [x1 + w1, y1 + h1],
                    [x1, y1 + h1]
                ], dtype=np.int32)
            return None
        
        rect2_area = w2 * h2
        combined_coverage = (rect1_area + rect2_area) / total_area
        
        if combined_coverage < min_coverage:
            # Even two rectangles don't cover enough
            return None
        
        # Merge two rectangles into L-shape polygon
        return LShapeFitter._merge_rectangles(x1, y1, w1, h1, x2, y2, w2, h2)
    
    @staticmethod
    def _merge_rectangles(x1, y1, w1, h1, x2, y2, w2, h2) -> np.ndarray:
        """
        Merge two rectangles into a single rectilinear polygon.
        
        Creates the outline of the union of both rectangles.
        """
        # Create a small binary image containing both rectangles
        min_x = min(x1, x2)
        min_y = min(y1, y2)
        max_x = max(x1 + w1, x2 + w2)
        max_y = max(y1 + h1, y2 + h2)
        
        temp = np.zeros((max_y - min_y + 2, max_x - min_x + 2), dtype=np.uint8)
        
        # Draw both rectangles
        cv2.rectangle(temp, (x1 - min_x, y1 - min_y), 
                     (x1 + w1 - min_x, y1 + h1 - min_y), 1, -1)
        cv2.rectangle(temp, (x2 - min_x, y2 - min_y),
                     (x2 + w2 - min_x, y2 + h2 - min_y), 1, -1)
        
        # Find contour of merged shape
        contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback: return first rectangle
            return np.array([
                [x1, y1], [x1 + w1, y1], [x1 + w1, y1 + h1], [x1, y1 + h1]
            ], dtype=np.int32)
        
        # Get the merged contour and shift back to original coordinates
        contour = contours[0].squeeze()
        if len(contour.shape) == 1:
            contour = contour.reshape(-1, 2)
        
        contour[:, 0] += min_x
        contour[:, 1] += min_y
        
        # Simplify to rectilinear
        contour = LShapeFitter._make_rectilinear(contour)
        
        return contour
    
    @staticmethod
    def _make_rectilinear(polygon: np.ndarray) -> np.ndarray:
        """
        Force polygon to have only horizontal and vertical edges.
        """
        if len(polygon) < 3:
            return polygon
        
        result = [polygon[0].tolist()]
        
        for i in range(1, len(polygon)):
            prev = result[-1]
            curr = polygon[i].tolist()
            
            dx = abs(curr[0] - prev[0])
            dy = abs(curr[1] - prev[1])
            
            # If diagonal, add corner point
            if dx > 2 and dy > 2:
                # Add horizontal then vertical
                result.append([curr[0], prev[1]])
            
            result.append(curr)
        
        return np.array(result, dtype=np.int32)


# ============================================================================
# MAIN RECTANGULAR FITTER
# ============================================================================

class RectangularRoomFitter:
    """
    Main class for fitting rectangular shapes to room segmentations.
    
    Pipeline:
    1. For each room class, extract binary mask
    2. Try to fit largest inscribed rectangle
    3. If coverage is low, try L-shape fitting
    4. Apply fitted shapes back to create clean mask
    
    Key guarantee: Never expands rooms, only simplifies within original bounds.
    """
    
    def __init__(self,
                 min_rectangle_coverage: float = 0.70,
                 min_lshape_coverage: float = 0.60,
                 min_room_area: int = 1000,
                 enable_lshape: bool = True):
        """
        Initialize the fitter.
        
        Args:
            min_rectangle_coverage: Minimum coverage to accept rectangle fit
            min_lshape_coverage: Minimum coverage to accept L-shape fit
            min_room_area: Minimum area to process (skip tiny regions)
            enable_lshape: Whether to try L-shape fitting for complex rooms
        """
        self.min_rectangle_coverage = min_rectangle_coverage
        self.min_lshape_coverage = min_lshape_coverage
        self.min_room_area = min_room_area
        self.enable_lshape = enable_lshape
        
        # Results
        self.fitted_rooms: List[FittedRoom] = []
        self.stats = {
            "total_rooms": 0,
            "rectangle_fits": 0,
            "lshape_fits": 0,
            "kept_original": 0,
            "skipped_small": 0,
        }
    
    def fit_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Process entire segmentation mask and fit rectangles to all rooms.
        
        Args:
            mask: Segmentation mask (H, W) with class values 0-13
        
        Returns:
            New mask with rectangular room shapes
        """
        logger.info("Fitting rectangular shapes to rooms...")
        
        self.fitted_rooms = []
        self.stats = {k: 0 for k in self.stats}
        
        result_mask = np.zeros_like(mask)
        
        # Process each class separately
        unique_classes = np.unique(mask)
        
        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue
            
            # Get binary mask for this class
            class_mask = (mask == class_id).astype(np.uint8)
            
            # Find connected components (separate rooms of same class)
            num_labels, labels, stats_cc, centroids = cv2.connectedComponentsWithStats(
                class_mask, connectivity=8
            )
            
            for label_id in range(1, num_labels):
                # Extract this room's mask
                room_mask = (labels == label_id).astype(np.uint8)
                area = stats_cc[label_id, cv2.CC_STAT_AREA]
                
                # Skip tiny rooms
                if area < self.min_room_area:
                    self.stats["skipped_small"] += 1
                    # Still keep them in result (don't delete)
                    result_mask[room_mask == 1] = class_id
                    continue
                
                self.stats["total_rooms"] += 1
                
                # Fit rectangle to this room
                fitted = self._fit_single_room(room_mask, class_id)
                self.fitted_rooms.append(fitted)
                
                # Apply fitted shape to result mask
                if fitted.fit_type == 'rectangle' or fitted.fit_type == 'l_shape':
                    # Draw the fitted polygon (ensure class_id is Python int for OpenCV)
                    cv2.fillPoly(result_mask, [fitted.polygon], int(class_id))
                else:
                    # Keep original
                    result_mask[room_mask == 1] = class_id
        
        # Log results
        logger.info(f"  ✓ Rectangular fitting complete")
        logger.info(f"    Total rooms: {self.stats['total_rooms']}")
        logger.info(f"    Rectangle fits: {self.stats['rectangle_fits']}")
        logger.info(f"    L-shape fits: {self.stats['lshape_fits']}")
        logger.info(f"    Kept original: {self.stats['kept_original']}")
        
        return result_mask
    
    def _fit_single_room(self, room_mask: np.ndarray, class_id: int) -> FittedRoom:
        """
        Fit rectangular shape to a single room.
        
        Strategy:
        1. Try largest inscribed rectangle
        2. If coverage too low, try L-shape
        3. If still too low, keep original
        """
        original_area = room_mask.sum()
        
        # Try rectangle fit
        x, y, w, h = LargestInscribedRectangle.find(room_mask)
        rect_area = w * h
        rect_coverage = rect_area / original_area if original_area > 0 else 0
        
        # Check if rectangle is good enough
        if rect_coverage >= self.min_rectangle_coverage:
            self.stats["rectangle_fits"] += 1
            polygon = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype=np.int32)
            
            return FittedRoom(
                class_id=class_id,
                original_area=original_area,
                fitted_area=rect_area,
                coverage=rect_coverage,
                fit_type='rectangle',
                polygon=polygon
            )
        
        # Try L-shape fit
        if self.enable_lshape:
            l_polygon = LShapeFitter.fit(room_mask, self.min_lshape_coverage)
            
            if l_polygon is not None:
                # Calculate L-shape area
                l_mask = np.zeros_like(room_mask)
                cv2.fillPoly(l_mask, [l_polygon], 1)
                l_area = l_mask.sum()
                l_coverage = l_area / original_area
                
                if l_coverage >= self.min_lshape_coverage:
                    self.stats["lshape_fits"] += 1
                    
                    return FittedRoom(
                        class_id=class_id,
                        original_area=original_area,
                        fitted_area=l_area,
                        coverage=l_coverage,
                        fit_type='l_shape',
                        polygon=l_polygon
                    )
        
        # Keep original shape
        self.stats["kept_original"] += 1
        
        # Get contour of original
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            polygon = contours[0].squeeze()
            if len(polygon.shape) == 1:
                polygon = polygon.reshape(-1, 2)
        else:
            polygon = np.array([[0, 0]], dtype=np.int32)
        
        return FittedRoom(
            class_id=class_id,
            original_area=original_area,
            fitted_area=original_area,
            coverage=1.0,
            fit_type='original',
            polygon=polygon
        )
    
    def visualize_comparison(self,
                             original_mask: np.ndarray,
                             fitted_mask: np.ndarray,
                             output_path: str):
        """Create before/after visualization."""
        # Colorize masks
        original_color = self._colorize_mask(original_mask)
        fitted_color = self._colorize_mask(fitted_mask)
        
        # Difference
        diff = (original_mask != fitted_mask).astype(np.uint8) * 255
        diff_color = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        diff_color[diff > 0] = [0, 0, 255]  # Red
        
        # Stack
        h, w = original_mask.shape
        comparison = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
        comparison[:, :w] = original_color
        comparison[:, w+10:w*2+10] = fitted_color
        comparison[:, w*2+20:] = diff_color
        
        # Labels
        cv2.putText(comparison, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Rectangular", (w + 20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Changes", (w * 2 + 30, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, comparison)
        logger.info(f"Saved comparison to {output_path}")
    
    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create colored visualization."""
        colors = {
            0: (50, 50, 50),
            1: (193, 182, 255),
            2: (124, 200, 255),
            3: (230, 216, 173),
            4: (181, 228, 255),
            5: (224, 255, 255),
            6: (45, 82, 160),
            7: (240, 255, 240),
            8: (144, 238, 144),
            9: (152, 251, 152),
            10: (169, 169, 169),
            11: (211, 211, 211),
            12: (250, 206, 135),
            13: (230, 224, 176),
        }
        
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in colors.items():
            colored[mask == class_id] = color
        
        return colored
    
    def get_summary(self) -> Dict:
        """Get fitting summary."""
        return {
            "stats": self.stats,
            "rooms": [
                {
                    "class_id": r.class_id,
                    "original_area": r.original_area,
                    "fitted_area": r.fitted_area,
                    "coverage": round(r.coverage, 3),
                    "fit_type": r.fit_type,
                }
                for r in self.fitted_rooms
            ]
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def fit_rectangular_rooms(mask: np.ndarray,
                          min_coverage: float = 0.70,
                          enable_lshape: bool = True) -> np.ndarray:
    """
    Convenience function to fit rectangular shapes to room mask.
    
    Args:
        mask: Segmentation mask
        min_coverage: Minimum coverage to accept fit
        enable_lshape: Enable L-shape fitting
    
    Returns:
        Mask with rectangular rooms
    """
    fitter = RectangularRoomFitter(
        min_rectangle_coverage=min_coverage,
        enable_lshape=enable_lshape
    )
    return fitter.fit_mask(mask)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fit rectangular shapes to room segmentation")
    parser.add_argument("--input", "-i", required=True, help="Input mask")
    parser.add_argument("--output", "-o", required=True, help="Output mask")
    parser.add_argument("--compare", "-c", help="Comparison image output")
    parser.add_argument("--min-coverage", type=float, default=0.70, help="Minimum coverage")
    parser.add_argument("--no-lshape", action="store_true", help="Disable L-shape fitting")
    
    args = parser.parse_args()
    
    # Load
    mask = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Cannot load {args.input}")
        exit(1)
    
    # Fit
    fitter = RectangularRoomFitter(
        min_rectangle_coverage=args.min_coverage,
        enable_lshape=not args.no_lshape
    )
    result = fitter.fit_mask(mask)
    
    # Save
    cv2.imwrite(args.output, result)
    print(f"✓ Saved to {args.output}")
    
    # Comparison
    if args.compare:
        fitter.visualize_comparison(mask, result, args.compare)
    
    # Summary
    summary = fitter.get_summary()
    print(f"\nSummary:")
    print(f"  Rectangle fits: {summary['stats']['rectangle_fits']}")
    print(f"  L-shape fits: {summary['stats']['lshape_fits']}")
    print(f"  Kept original: {summary['stats']['kept_original']}")
