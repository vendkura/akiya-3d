"""
Quick diagnostic: Run pipeline, analyze output, compare original vs 3D
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from main_pipeline import Pipeline3D
from boundary_extraction import BoundaryExtractor

def run_full_diagnostic():
    """Run full pipeline and analyze results."""
    
    logger.info("="*60)
    logger.info("DIAGNOSTIC: Boundary Extraction & Wall Extrusion Quality")
    logger.info("="*60)
    
    # Get first test image
    test_image_dir = Path("../U-NET/data/floorplan/")
    test_images = list(test_image_dir.glob('*.png'))
    
    if not test_images:
        logger.error("No test images found!")
        return
    
    test_image = test_images[0]
    logger.info(f"\nUsing test image: {test_image.name}")
    
    # Run pipeline
    pipeline = Pipeline3D()
    result = pipeline.process_image(test_image, output_name="diagnostic")
    
    if result['status'] != 'success':
        logger.error(f"Pipeline failed: {result['error']}")
        return
    
    # ====================  DIAGNOSTIC 1: Boundary Extraction ====================
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSTIC 1: BOUNDARY EXTRACTION QUALITY")
    logger.info("="*60)
    
    boundaries_path = Path(result['output_dir']) / "diagnostic_boundaries.json"
    with open(boundaries_path, 'r') as f:
        boundaries = json.load(f)
    
    logger.info(f"\nRooms extracted: {len(boundaries['rooms'])}")
    logger.info(f"Walls detected: {len(boundaries['walls'])}")
    logger.info(f"Scale factor: {boundaries['scale_factor']:.6f} m/pixel")
    
    # Analyze room polygons
    logger.info("\n--- Room Polygon Analysis ---")
    vertex_counts = [len(room['vertices']) for room in boundaries['rooms']]
    areas = [room['area'] for room in boundaries['rooms']]
    
    logger.info(f"Vertex count - Min: {min(vertex_counts)}, Max: {max(vertex_counts)}, Avg: {np.mean(vertex_counts):.1f}")
    logger.info(f"Room areas - Min: {min(areas):.1f}, Max: {max(areas):.1f}, Avg: {np.mean(areas):.1f}")
    
    # Check for small rooms (might indicate over-simplification)
    small_rooms = [r for r in boundaries['rooms'] if r['area'] < 50]
    if small_rooms:
        logger.warning(f"⚠️  Found {len(small_rooms)} very small rooms (area < 50 px²) - possible noise")
    
    # ====================  DIAGNOSTIC 2: Scale Factor ====================
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSTIC 2: SCALE FACTOR VALIDATION")
    logger.info("="*60)
    
    scale = boundaries['scale_factor']
    logger.info(f"\nCalculated scale: {scale:.6f} m/pixel")
    logger.info(f"This means: 1 pixel = {scale*100:.4f} cm")
    logger.info(f"Average room width in pixels: ~{4.0/scale:.1f} px")
    
    # Expected room sizes
    logger.info("\nExpected vs Actual room sizes:")
    for i, room in enumerate(boundaries['rooms'][:5]):  # Check first 5 rooms
        vertices = np.array(room['vertices'])
        width = vertices[:, 0].max() - vertices[:, 0].min()
        height = vertices[:, 1].max() - vertices[:, 1].min()
        width_m = width * scale
        height_m = height * scale
        logger.info(f"  Room {i}: {width:.0f}×{height:.0f} px → {width_m:.2f}×{height_m:.2f} m")
    
    # ====================  DIAGNOSTIC 3: Geometry ====================
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSTIC 3: 3D GEOMETRY ANALYSIS")
    logger.info("="*60)
    
    geometry = result['geometry']
    vertices = geometry['vertices']
    faces = geometry['faces']
    
    logger.info(f"\nVertices: {len(vertices)}")
    logger.info(f"Faces: {len(faces)}")
    
    # Calculate bounding box
    verts_array = np.array(vertices)
    bbox_min = verts_array.min(axis=0)
    bbox_max = verts_array.max(axis=0)
    bbox_size = bbox_max - bbox_min
    
    logger.info(f"\nBounding box:")
    logger.info(f"  Min: {bbox_min}")
    logger.info(f"  Max: {bbox_max}")
    logger.info(f"  Size: {bbox_size} m")
    
    # Check if rooms are adjacent or separated
    logger.info("\n--- Room Connectivity Check ---")
    
    # For each face, check if vertices are close to other vertices
    vertex_distances = {}
    for i, v1 in enumerate(vertices):
        min_dist = float('inf')
        for j, v2 in enumerate(vertices):
            if i != j:
                dist = np.linalg.norm(np.array(v1) - np.array(v2))
                if dist < min_dist and dist > 0.001:
                    min_dist = dist
        vertex_distances[i] = min_dist
    
    avg_min_distance = np.mean(list(vertex_distances.values()))
    logger.info(f"Average minimum distance between vertices: {avg_min_distance:.4f} m")
    
    if avg_min_distance > 0.15:  # More than 15cm
        logger.warning(f"⚠️  Large gaps detected! (avg min distance = {avg_min_distance:.2f}m)")
        logger.warning("   → This suggests rooms are separated instead of adjacent")
    else:
        logger.info(f"✓ Vertices are close together (good room connectivity)")
    
    # ====================  DIAGNOSTIC 4: SUMMARY ====================
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("="*60)
    
    issues = []
    
    if len(small_rooms) > 0:
        issues.append(f"Over-simplification: {len(small_rooms)} tiny rooms detected")
    
    if avg_min_distance > 0.15:
        issues.append(f"Room separation: Large gaps between rooms ({avg_min_distance:.2f}m)")
    
    if np.mean(vertex_counts) < 4:
        issues.append("Extreme simplification: Average <4 vertices per room")
    
    if issues:
        logger.warning("\n❌ ISSUES FOUND:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("\n✓ No critical issues detected!")
    
    logger.info(f"\nFiles generated:")
    logger.info(f"  OBJ: {result['obj_file']}")
    logger.info(f"  MTL: {result['mtl_file']}")
    logger.info(f"\nOpen {result['obj_file']} in Blender to visually inspect")

if __name__ == "__main__":
    run_full_diagnostic()
