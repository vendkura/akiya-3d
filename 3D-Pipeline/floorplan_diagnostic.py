"""
Floorplan 3D Pipeline Diagnostic Tool
=====================================
Analyzes polygon extraction quality to identify issues before 3D extrusion.

Usage:
    python floorplan_diagnostic.py --image path/to/floorplan.png --mask path/to/mask.png --output ./diagnostics
    
    Or programmatically:
        from floorplan_diagnostic import PipelineDiagnostic
        diag = PipelineDiagnostic()
        diag.run_full_diagnostic(image_path, mask_path, output_dir)

Author: Diagnostic tool for Asheleyine's Master thesis project
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


# ============================================================================
# CONFIGURATION - Same as your pipeline
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

# Distinct colors for visualization (BGR format for OpenCV)
VIZ_COLORS = {
    "dining_area": (193, 182, 255),
    "bathroom": (124, 200, 255),
    "bedroom": (230, 216, 173),
    "closet": (181, 228, 255),
    "room": (224, 255, 255),
    "door": (45, 82, 160),
    "entrance": (240, 255, 240),
    "kitchen": (144, 238, 144),
    "outdoor_space": (152, 251, 152),
    "sliding_door": (169, 169, 169),
    "stairs": (211, 211, 211),
    "window": (250, 206, 135),
    "balcony": (230, 224, 176),
}

# Diagnostic thresholds
THRESHOLDS = {
    "min_area_noise": 500,        # Below this = definitely noise
    "min_area_warning": 2000,     # Below this = suspicious
    "min_area_recommended": 5000, # Recommended minimum for real rooms
    "max_vertices_simple": 8,     # Simple polygon
    "max_vertices_complex": 20,   # Complex but acceptable
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PolygonStats:
    """Statistics for a single extracted polygon."""
    polygon_id: int
    class_name: str
    class_id: int
    area: float
    perimeter: float
    vertex_count: int
    is_convex: bool
    convexity_ratio: float  # area / convex_hull_area (1.0 = perfectly convex)
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    aspect_ratio: float
    
    # Quality flags
    is_noise: bool
    is_suspicious: bool
    is_complex: bool
    has_self_intersection: bool
    quality_score: float  # 0-1, higher = better
    issues: List[str]


@dataclass 
class DiagnosticReport:
    """Full diagnostic report for one image."""
    image_name: str
    timestamp: str
    mask_shape: Tuple[int, int]
    
    # Summary stats
    total_polygons: int
    noise_polygons: int
    suspicious_polygons: int
    valid_polygons: int
    
    # Per-class breakdown
    class_counts: Dict[str, int]
    class_avg_areas: Dict[str, float]
    
    # All polygon details
    polygons: List[dict]
    
    # Recommendations
    recommendations: List[str]


# ============================================================================
# DIAGNOSTIC CLASS
# ============================================================================

class PipelineDiagnostic:
    """
    Diagnostic tool to analyze floorplan segmentation and polygon extraction.
    """
    
    def __init__(self, 
                 min_contour_area: int = 100,
                 contour_epsilon: float = 0.02):
        """
        Initialize with same parameters as your BoundaryExtractor.
        
        Args:
            min_contour_area: Current minimum area threshold
            contour_epsilon: Douglas-Peucker approximation epsilon
        """
        self.min_contour_area = min_contour_area
        self.contour_epsilon = contour_epsilon
        
        self.mask = None
        self.original_image = None
        self.polygons: List[PolygonStats] = []
        
    def load_inputs(self, 
                    mask_path: Optional[str] = None,
                    image_path: Optional[str] = None,
                    mask_array: Optional[np.ndarray] = None):
        """
        Load mask and optionally original image.
        
        Args:
            mask_path: Path to segmentation mask (grayscale, values 0-13)
            image_path: Path to original floorplan image (optional)
            mask_array: Numpy array of mask (alternative to mask_path)
        """
        # Load mask
        if mask_array is not None:
            self.mask = mask_array.astype(np.uint8)
        elif mask_path:
            self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if self.mask is None:
                raise FileNotFoundError(f"Cannot load mask: {mask_path}")
        else:
            raise ValueError("Must provide either mask_path or mask_array")
        
        # Load original image
        if image_path:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                print(f"Warning: Cannot load image: {image_path}")
                self.original_image = cv2.cvtColor(self.mask * 18, cv2.COLOR_GRAY2BGR)
        else:
            # Create colored version of mask for visualization
            self.original_image = self._colorize_mask(self.mask)
        
        print(f"Loaded mask: {self.mask.shape}")
        print(f"Unique classes in mask: {np.unique(self.mask)}")
        
    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create colored visualization of mask."""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        colored[:] = (50, 50, 50)  # Dark gray background
        
        for class_id in range(1, 14):
            class_name = CLASS_NAMES.get(class_id, "unknown")
            color = VIZ_COLORS.get(class_name, (200, 200, 200))
            colored[mask == class_id] = color
            
        return colored
    
    def extract_and_analyze_polygons(self) -> List[PolygonStats]:
        """
        Extract polygons and compute detailed statistics for each.
        
        Returns:
            List of PolygonStats for all extracted polygons
        """
        if self.mask is None:
            raise ValueError("Mask not loaded. Call load_inputs() first.")
        
        self.polygons = []
        polygon_id = 0
        
        for class_id in range(1, 14):
            class_name = CLASS_NAMES.get(class_id, f"unknown_{class_id}")
            
            # Binary mask for this class
            binary = (self.mask == class_id).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Skip if below current threshold (but we'll still flag it)
                if area < 10:  # Absolute minimum
                    continue
                
                # Analyze this polygon
                stats = self._analyze_polygon(polygon_id, class_name, class_id, contour)
                self.polygons.append(stats)
                polygon_id += 1
        
        print(f"Extracted {len(self.polygons)} polygons")
        return self.polygons
    
    def _analyze_polygon(self, 
                         polygon_id: int,
                         class_name: str,
                         class_id: int,
                         contour: np.ndarray) -> PolygonStats:
        """Compute detailed statistics for a single polygon."""
        
        # Basic metrics
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Simplify contour (same as your pipeline)
        epsilon = self.contour_epsilon * perimeter
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        vertex_count = len(simplified)
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        
        # Convexity analysis
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        is_convex = cv2.isContourConvex(simplified)
        convexity_ratio = area / (hull_area + 1e-6)
        
        # Self-intersection check (approximate)
        has_self_intersection = self._check_self_intersection(simplified)
        
        # Quality assessment
        issues = []
        
        # Check for noise
        is_noise = area < THRESHOLDS["min_area_noise"]
        if is_noise:
            issues.append(f"NOISE: area={area:.0f} < {THRESHOLDS['min_area_noise']}")
        
        # Check for suspicious size
        is_suspicious = THRESHOLDS["min_area_noise"] <= area < THRESHOLDS["min_area_warning"]
        if is_suspicious:
            issues.append(f"SUSPICIOUS: small area={area:.0f}")
        
        # Check complexity
        is_complex = vertex_count > THRESHOLDS["max_vertices_complex"]
        if is_complex:
            issues.append(f"COMPLEX: {vertex_count} vertices")
        
        # Check convexity
        if convexity_ratio < 0.7:
            issues.append(f"HIGHLY_NON_CONVEX: ratio={convexity_ratio:.2f}")
        elif convexity_ratio < 0.85:
            issues.append(f"NON_CONVEX: ratio={convexity_ratio:.2f}")
        
        # Check self-intersection
        if has_self_intersection:
            issues.append("SELF_INTERSECTING")
        
        # Check aspect ratio (very elongated shapes are suspicious)
        if aspect_ratio > 10:
            issues.append(f"ELONGATED: aspect_ratio={aspect_ratio:.1f}")
        
        # Compute quality score (0-1)
        quality_score = self._compute_quality_score(
            area, vertex_count, convexity_ratio, 
            has_self_intersection, aspect_ratio
        )
        
        return PolygonStats(
            polygon_id=polygon_id,
            class_name=class_name,
            class_id=class_id,
            area=area,
            perimeter=perimeter,
            vertex_count=vertex_count,
            is_convex=is_convex,
            convexity_ratio=convexity_ratio,
            bounding_box=(x, y, w, h),
            aspect_ratio=aspect_ratio,
            is_noise=is_noise,
            is_suspicious=is_suspicious,
            is_complex=is_complex,
            has_self_intersection=has_self_intersection,
            quality_score=quality_score,
            issues=issues
        )
    
    def _check_self_intersection(self, contour: np.ndarray) -> bool:
        """
        Approximate check for self-intersecting polygon.
        Uses line segment intersection test on simplified contour.
        """
        if len(contour) < 4:
            return False
        
        points = contour.squeeze()
        if len(points.shape) == 1:
            return False
        
        n = len(points)
        
        # Check non-adjacent edges for intersection
        for i in range(n):
            p1, p2 = points[i], points[(i + 1) % n]
            
            for j in range(i + 2, n):
                if j == (i - 1) % n or j == (i + 1) % n:
                    continue
                    
                p3, p4 = points[j], points[(j + 1) % n]
                
                # Skip adjacent edges
                if (j + 1) % n == i:
                    continue
                
                if self._segments_intersect(p1, p2, p3, p4):
                    return True
        
        return False
    
    def _segments_intersect(self, p1, p2, p3, p4) -> bool:
        """Check if line segments (p1,p2) and (p3,p4) intersect."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))
    
    def _compute_quality_score(self,
                               area: float,
                               vertex_count: int,
                               convexity_ratio: float,
                               has_self_intersection: bool,
                               aspect_ratio: float) -> float:
        """Compute overall quality score (0-1)."""
        score = 1.0
        
        # Penalize small areas
        if area < THRESHOLDS["min_area_noise"]:
            score *= 0.1
        elif area < THRESHOLDS["min_area_warning"]:
            score *= 0.5
        elif area < THRESHOLDS["min_area_recommended"]:
            score *= 0.8
        
        # Penalize high vertex count
        if vertex_count > THRESHOLDS["max_vertices_complex"]:
            score *= 0.6
        elif vertex_count > THRESHOLDS["max_vertices_simple"]:
            score *= 0.9
        
        # Penalize non-convexity
        score *= (0.5 + 0.5 * convexity_ratio)
        
        # Heavy penalty for self-intersection
        if has_self_intersection:
            score *= 0.3
        
        # Penalize extreme aspect ratios
        if aspect_ratio > 10:
            score *= 0.5
        elif aspect_ratio > 5:
            score *= 0.8
        
        return max(0.0, min(1.0, score))
    
    def generate_report(self, image_name: str = "unknown") -> DiagnosticReport:
        """Generate comprehensive diagnostic report."""
        
        if not self.polygons:
            self.extract_and_analyze_polygons()
        
        # Count categories
        noise_count = sum(1 for p in self.polygons if p.is_noise)
        suspicious_count = sum(1 for p in self.polygons if p.is_suspicious)
        valid_count = len(self.polygons) - noise_count - suspicious_count
        
        # Per-class breakdown
        class_counts = {}
        class_areas = {}
        for p in self.polygons:
            if p.class_name not in class_counts:
                class_counts[p.class_name] = 0
                class_areas[p.class_name] = []
            class_counts[p.class_name] += 1
            class_areas[p.class_name].append(p.area)
        
        class_avg_areas = {k: np.mean(v) for k, v in class_areas.items()}
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return DiagnosticReport(
            image_name=image_name,
            timestamp=datetime.now().isoformat(),
            mask_shape=tuple(self.mask.shape),
            total_polygons=len(self.polygons),
            noise_polygons=noise_count,
            suspicious_polygons=suspicious_count,
            valid_polygons=valid_count,
            class_counts=class_counts,
            class_avg_areas=class_avg_areas,
            polygons=[asdict(p) for p in self.polygons],
            recommendations=recommendations
        )
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        noise_count = sum(1 for p in self.polygons if p.is_noise)
        suspicious_count = sum(1 for p in self.polygons if p.is_suspicious)
        non_convex_count = sum(1 for p in self.polygons if p.convexity_ratio < 0.7)
        self_intersect_count = sum(1 for p in self.polygons if p.has_self_intersection)
        
        if noise_count > 0:
            recommendations.append(
                f"CRITICAL: {noise_count} noise polygons detected. "
                f"Increase min_contour_area from {self.min_contour_area} to at least {THRESHOLDS['min_area_warning']}"
            )
        
        if suspicious_count > 0:
            recommendations.append(
                f"WARNING: {suspicious_count} suspicious small polygons. "
                f"Consider increasing min_contour_area to {THRESHOLDS['min_area_recommended']}"
            )
        
        if non_convex_count > 0:
            recommendations.append(
                f"CRITICAL: {non_convex_count} highly non-convex polygons. "
                f"Fan triangulation will fail. Use ear-clipping or constrained Delaunay triangulation."
            )
        
        if self_intersect_count > 0:
            recommendations.append(
                f"CRITICAL: {self_intersect_count} self-intersecting polygons detected. "
                f"Add polygon validation/repair step before extrusion."
            )
        
        # Check for class imbalance
        if self.polygons:
            areas = [p.area for p in self.polygons if not p.is_noise]
            if areas:
                max_area = max(areas)
                min_area = min(areas)
                if max_area / (min_area + 1) > 100:
                    recommendations.append(
                        f"INFO: Large area variance (max/min = {max_area/min_area:.0f}x). "
                        f"Some rooms may be fragmented or merged."
                    )
        
        if not recommendations:
            recommendations.append("All polygons look reasonable. Issue may be in extrusion logic.")
        
        return recommendations
    
    def visualize_polygons(self, 
                           output_path: str,
                           show_labels: bool = True,
                           highlight_issues: bool = True) -> np.ndarray:
        """
        Create visualization with all polygons color-coded by quality.
        
        Colors:
            - Green: Valid polygons (quality > 0.7)
            - Orange: Suspicious polygons (quality 0.4-0.7)
            - Red: Noise/problematic polygons (quality < 0.4)
        
        Args:
            output_path: Where to save the visualization
            show_labels: Whether to show polygon IDs and stats
            highlight_issues: Whether to use color coding
        
        Returns:
            Visualization image (BGR)
        """
        if not self.polygons:
            self.extract_and_analyze_polygons()
        
        # Start with colored mask
        viz = self._colorize_mask(self.mask).copy()
        
        for p in self.polygons:
            # Get contour for this polygon
            binary = (self.mask == p.class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find matching contour by bounding box
            x, y, w, h = p.bounding_box
            matching_contour = None
            for c in contours:
                cx, cy, cw, ch = cv2.boundingRect(c)
                if abs(cx - x) < 5 and abs(cy - y) < 5:
                    matching_contour = c
                    break
            
            if matching_contour is None:
                continue
            
            # Color based on quality
            if highlight_issues:
                if p.quality_score >= 0.7:
                    color = (0, 200, 0)  # Green - good
                    thickness = 2
                elif p.quality_score >= 0.4:
                    color = (0, 165, 255)  # Orange - suspicious
                    thickness = 2
                else:
                    color = (0, 0, 255)  # Red - problematic
                    thickness = 3
            else:
                color = VIZ_COLORS.get(p.class_name, (200, 200, 200))
                thickness = 2
            
            # Draw contour
            cv2.drawContours(viz, [matching_contour], -1, color, thickness)
            
            # Draw label
            if show_labels:
                label = f"#{p.polygon_id}"
                if p.is_noise:
                    label += " NOISE"
                elif p.is_suspicious:
                    label += " ?"
                
                # Position label at centroid
                M = cv2.moments(matching_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Background rectangle for readability
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(viz, (cx-2, cy-th-2), (cx+tw+2, cy+2), (0, 0, 0), -1)
                    cv2.putText(viz, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add legend
        viz = self._add_legend(viz)
        
        cv2.imwrite(output_path, viz)
        print(f"Saved visualization to {output_path}")
        
        return viz
    
    def _add_legend(self, img: np.ndarray) -> np.ndarray:
        """Add legend to visualization."""
        h, w = img.shape[:2]
        
        # Create legend panel
        legend_h = 100
        legend = np.zeros((legend_h, w, 3), dtype=np.uint8)
        legend[:] = (40, 40, 40)
        
        # Add legend items
        items = [
            ((0, 200, 0), "Valid (quality >= 0.7)"),
            ((0, 165, 255), "Suspicious (quality 0.4-0.7)"),
            ((0, 0, 255), "Problematic (quality < 0.4)"),
        ]
        
        x_offset = 20
        for color, text in items:
            cv2.rectangle(legend, (x_offset, 10), (x_offset + 20, 30), color, -1)
            cv2.putText(legend, text, (x_offset + 30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            x_offset += 250
        
        # Add summary stats
        if self.polygons:
            noise = sum(1 for p in self.polygons if p.is_noise)
            valid = sum(1 for p in self.polygons if p.quality_score >= 0.7)
            summary = f"Total: {len(self.polygons)} | Valid: {valid} | Noise: {noise}"
            cv2.putText(legend, summary, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Stack legend below image
        result = np.vstack([img, legend])
        return result
    
    def visualize_individual_polygons(self, output_dir: str, max_polygons: int = 50):
        """
        Save individual visualization for each polygon (useful for debugging).
        
        Args:
            output_dir: Directory to save individual images
            max_polygons: Maximum number of individual images to save
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, p in enumerate(self.polygons[:max_polygons]):
            # Create small image showing just this polygon
            viz = self._colorize_mask(self.mask).copy()
            
            # Get contour
            binary = (self.mask == p.class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            x, y, w, h = p.bounding_box
            for c in contours:
                cx, cy, cw, ch = cv2.boundingRect(c)
                if abs(cx - x) < 5 and abs(cy - y) < 5:
                    # Highlight this polygon
                    color = (0, 0, 255) if p.is_noise else (0, 255, 0)
                    cv2.drawContours(viz, [c], -1, color, 3)
                    
                    # Crop to region of interest with padding
                    pad = 50
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(viz.shape[1], x + w + pad)
                    y2 = min(viz.shape[0], y + h + pad)
                    
                    crop = viz[y1:y2, x1:x2]
                    
                    # Add info text
                    info_h = 80
                    info_panel = np.zeros((info_h, crop.shape[1], 3), dtype=np.uint8)
                    info_panel[:] = (30, 30, 30)
                    
                    lines = [
                        f"#{p.polygon_id} {p.class_name}",
                        f"Area: {p.area:.0f}px | Vertices: {p.vertex_count}",
                        f"Quality: {p.quality_score:.2f} | Convex: {p.convexity_ratio:.2f}",
                        f"Issues: {', '.join(p.issues) if p.issues else 'None'}"
                    ]
                    
                    for j, line in enumerate(lines):
                        cv2.putText(info_panel, line, (10, 18 + j*18), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    
                    result = np.vstack([crop, info_panel])
                    
                    filename = f"polygon_{p.polygon_id:03d}_{p.class_name}.png"
                    cv2.imwrite(str(output_dir / filename), result)
                    break
        
        print(f"Saved {min(len(self.polygons), max_polygons)} individual polygon images to {output_dir}")
    
    def print_summary(self):
        """Print summary to console."""
        if not self.polygons:
            print("No polygons analyzed yet. Call extract_and_analyze_polygons() first.")
            return
        
        print("\n" + "="*70)
        print("DIAGNOSTIC SUMMARY")
        print("="*70)
        
        noise = [p for p in self.polygons if p.is_noise]
        suspicious = [p for p in self.polygons if p.is_suspicious]
        valid = [p for p in self.polygons if not p.is_noise and not p.is_suspicious]
        non_convex = [p for p in self.polygons if p.convexity_ratio < 0.7]
        self_intersect = [p for p in self.polygons if p.has_self_intersection]
        
        print(f"\nTotal polygons extracted: {len(self.polygons)}")
        print(f"  ✓ Valid:       {len(valid)}")
        print(f"  ? Suspicious:  {len(suspicious)}")
        print(f"  ✗ Noise:       {len(noise)}")
        print(f"  ⚠ Non-convex:  {len(non_convex)}")
        print(f"  ⚠ Self-intersecting: {len(self_intersect)}")
        
        print("\n" + "-"*70)
        print("NOISE POLYGONS (should be filtered):")
        print("-"*70)
        for p in noise[:10]:  # Show first 10
            print(f"  #{p.polygon_id} {p.class_name}: area={p.area:.0f}, vertices={p.vertex_count}")
        if len(noise) > 10:
            print(f"  ... and {len(noise) - 10} more")
        
        print("\n" + "-"*70)
        print("NON-CONVEX POLYGONS (fan triangulation will fail):")
        print("-"*70)
        for p in non_convex[:10]:
            print(f"  #{p.polygon_id} {p.class_name}: convexity={p.convexity_ratio:.2f}, area={p.area:.0f}")
        if len(non_convex) > 10:
            print(f"  ... and {len(non_convex) - 10} more")
        
        print("\n" + "-"*70)
        print("RECOMMENDATIONS:")
        print("-"*70)
        for rec in self._generate_recommendations():
            print(f"  • {rec}")
        
        print("\n" + "="*70)
    
    def run_full_diagnostic(self,
                            image_path: Optional[str],
                            mask_path: str,
                            output_dir: str,
                            image_name: Optional[str] = None) -> DiagnosticReport:
        """
        Run complete diagnostic pipeline.
        
        Args:
            image_path: Path to original floorplan (optional)
            mask_path: Path to segmentation mask
            output_dir: Directory for all outputs
            image_name: Name for this image (defaults to mask filename)
        
        Returns:
            DiagnosticReport with all findings
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if image_name is None:
            image_name = Path(mask_path).stem
        
        print(f"\n{'='*70}")
        print(f"RUNNING DIAGNOSTIC: {image_name}")
        print(f"{'='*70}")
        
        # Load inputs
        self.load_inputs(mask_path=mask_path, image_path=image_path)
        
        # Extract and analyze
        self.extract_and_analyze_polygons()
        
        # Print summary
        self.print_summary()
        
        # Generate visualizations
        viz_path = output_dir / f"{image_name}_diagnostic.png"
        self.visualize_polygons(str(viz_path))
        
        # Save individual problematic polygons
        problem_dir = output_dir / f"{image_name}_polygons"
        self.visualize_individual_polygons(str(problem_dir))
        
        # Generate and save report
        report = self.generate_report(image_name)
        report_path = output_dir / f"{image_name}_report.json"
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Saved report to {report_path}")
        
        return report


# ============================================================================
# COMPARISON TOOL (for multiple images)
# ============================================================================

def compare_diagnostics(reports: List[DiagnosticReport], output_path: str):
    """
    Compare diagnostic reports across multiple images.
    
    Args:
        reports: List of DiagnosticReport objects
        output_path: Where to save comparison JSON
    """
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "num_images": len(reports),
        "images": [],
        "summary": {
            "total_polygons": 0,
            "total_noise": 0,
            "total_suspicious": 0,
            "avg_quality": 0,
        },
        "common_issues": [],
    }
    
    all_recommendations = []
    
    for report in reports:
        comparison["images"].append({
            "name": report.image_name,
            "total_polygons": report.total_polygons,
            "noise_polygons": report.noise_polygons,
            "valid_polygons": report.valid_polygons,
            "class_counts": report.class_counts,
        })
        
        comparison["summary"]["total_polygons"] += report.total_polygons
        comparison["summary"]["total_noise"] += report.noise_polygons
        comparison["summary"]["total_suspicious"] += report.suspicious_polygons
        
        all_recommendations.extend(report.recommendations)
    
    # Find common issues
    from collections import Counter
    rec_counts = Counter(all_recommendations)
    comparison["common_issues"] = [
        {"issue": issue, "occurrences": count}
        for issue, count in rec_counts.most_common(10)
    ]
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison saved to {output_path}")
    return comparison


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose floorplan segmentation and polygon extraction quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image diagnostic
  python floorplan_diagnostic.py --mask mask.png --output ./diagnostics
  
  # With original image overlay
  python floorplan_diagnostic.py --mask mask.png --image floorplan.png --output ./diagnostics
  
  # Multiple images
  python floorplan_diagnostic.py --mask mask1.png mask2.png --output ./diagnostics
        """
    )
    
    parser.add_argument("--mask", "-m", nargs="+", required=True,
                        help="Path(s) to segmentation mask(s)")
    parser.add_argument("--image", "-i", nargs="*", default=[],
                        help="Path(s) to original floorplan image(s)")
    parser.add_argument("--output", "-o", default="./diagnostics",
                        help="Output directory for results")
    parser.add_argument("--min-area", type=int, default=100,
                        help="Current min_contour_area setting (for comparison)")
    
    args = parser.parse_args()
    
    # Run diagnostics
    reports = []
    
    for i, mask_path in enumerate(args.mask):
        image_path = args.image[i] if i < len(args.image) else None
        
        diag = PipelineDiagnostic(min_contour_area=args.min_area)
        report = diag.run_full_diagnostic(
            image_path=image_path,
            mask_path=mask_path,
            output_dir=args.output
        )
        reports.append(report)
    
    # Compare if multiple images
    if len(reports) > 1:
        compare_diagnostics(reports, f"{args.output}/comparison.json")
    
    print("\n✓ Diagnostic complete!")


# ============================================================================
# DEFAULT FLOORPLAN IMAGES FOR TESTING
# ============================================================================
# Use glob pattern to find files (avoids encoding issues with special characters)
def get_default_floorplan_images():
    """Get default floorplan images using glob to avoid encoding issues."""
    floorplan_dir = Path(__file__).parent.parent / "U-NET" / "data" / "floorplan"
    
    if not floorplan_dir.exists():
        print(f"Warning: Floorplan directory not found: {floorplan_dir}")
        return []
    
    # Get all PNG files and sort them
    all_images = sorted(floorplan_dir.glob("*.png"))
    
    # Return first 2 images
    return [str(img) for img in all_images[:2]]


# FPN Model path for generating masks
def get_default_model_path():
    """Get the default FPN model path."""
    return Path(__file__).parent.parent / "U-NET" / "scripts" / "model_output" / "fpn_62images" / "best_model.pth"


def generate_mask_from_image(image_path: str, model_path: str, output_dir: str) -> str:
    """
    Generate segmentation mask from floorplan image using FPN model.
    
    Args:
        image_path: Path to floorplan image
        model_path: Path to FPN model
        output_dir: Directory to save generated mask
        
    Returns:
        Path to generated mask
    """
    from fpn_inference import FPNInference
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(image_path).stem
    mask_path = output_dir / f"{image_name}_mask.png"
    
    print(f"\nGenerating mask for: {image_name}")
    
    # Load model and run inference
    fpn = FPNInference(model_path)
    mask, original_shape, classes_present = fpn.infer(image_path)
    
    print(f"  ✓ Detected classes:")
    for class_name, count in sorted(classes_present.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    - {class_name}: {count} pixels")
    
    # Save mask
    fpn.save_mask(mask, str(mask_path))
    
    return str(mask_path)


if __name__ == "__main__":
    import sys
    
    # If no command-line arguments provided, use default floorplan images
    if len(sys.argv) == 1:
        # Get default images using glob (avoids encoding issues)
        default_images = get_default_floorplan_images()
        
        if not default_images:
            print("❌ ERROR: No floorplan images found!")
            sys.exit(1)
        
        print("No arguments provided. Using default floorplan images...")
        for i, img in enumerate(default_images, 1):
            print(f"  {i}. {Path(img).name}")
        
        # Check if model exists
        model_path = get_default_model_path()
        if not model_path.exists():
            print(f"\n❌ ERROR: FPN model not found at: {model_path}")
            print("Please update the model path.")
            sys.exit(1)
        
        # Run diagnostics on default images
        output_dir = "./pipeline_output/diagnostic"
        reports = []
        
        for image_path in default_images:
            # Check if image exists
            if not Path(image_path).exists():
                print(f"\n❌ ERROR: Image not found: {image_path}")
                continue
                
            # Generate mask first
            try:
                mask_path = generate_mask_from_image(image_path, str(model_path), output_dir)
            except Exception as e:
                print(f"\n❌ ERROR generating mask: {e}")
                continue
            
            # Run diagnostic
            diag = PipelineDiagnostic(min_contour_area=100)
            report = diag.run_full_diagnostic(
                image_path=image_path,
                mask_path=mask_path,
                output_dir=output_dir
            )
            if report:
                reports.append(report)
        
        if len(reports) > 1:
            compare_diagnostics(reports, f"{output_dir}/comparison.json")
        
        print("\n✓ Diagnostic complete!")
    else:
        main()