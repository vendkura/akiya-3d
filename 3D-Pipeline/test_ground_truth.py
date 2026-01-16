"""
Test 3D pipeline with GROUND TRUTH masks instead of FPN predictions.

This isolates whether the problem is:
1. FPN segmentation (if ground truth produces good 3D) 
2. Boundary extraction (if still has gaps)
3. Wall extrusion (if geometry is wrong)

Usage:
    python test_ground_truth.py
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import logging

from boundary_extraction import BoundaryExtractor
from wall_extrusion_v4 import WallExtrusion4
from export_obj import OBJExporter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_ground_truth_pipeline():
    """Test 3D generation using ground truth masks."""
    
    # Find first ground truth mask
    gt_dir = Path("../U-NET/data/floorplan_masks_13classes")
    gt_masks = sorted(list(gt_dir.glob("*_mask.png")))
    
    if not gt_masks:
        logger.error(f"No ground truth masks found in {gt_dir}")
        return
    
    gt_mask_path = gt_masks[0]
    test_image_name = gt_mask_path.stem.replace("_mask", "")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING WITH GROUND TRUTH MASK")
    logger.info(f"{'='*60}")
    logger.info(f"Mask: {gt_mask_path.name}")
    
    # Load ground truth mask
    logger.info("\n[1/3] Loading ground truth mask...")
    gt_mask = np.array(Image.open(gt_mask_path))
    logger.info(f"✓ Loaded mask shape: {gt_mask.shape}")
    logger.info(f"  Unique class IDs: {np.unique(gt_mask)}")
    logger.info(f"  Class ID range: {gt_mask.min()} - {gt_mask.max()}")
    
    # Boundary extraction
    logger.info("\n[2/3] Boundary extraction on ground truth...")
    extractor = BoundaryExtractor()
    extractor.load_mask_array(gt_mask)
    extractor.extract_room_polygons()
    extractor.detect_walls()
    extractor.calculate_scale_factor()
    
    logger.info(f"✓ Extracted {len(extractor.rooms)} rooms")
    logger.info(f"  Detected {len(extractor.walls)} walls")
    logger.info(f"  Scale factor: {extractor.scale_factor:.6f} m/pixel")
    
    # Create boundaries dict
    boundaries_json = {
        'rooms': extractor.rooms,
        'walls': extractor.walls,
        'scale_factor': extractor.scale_factor
    }
    
    # Wall extrusion
    logger.info("\n[3/3] Wall extrusion...")
    extrusion = WallExtrusion4(wall_thickness=0.1)
    extrusion.load_boundaries_json(boundaries_json)
    extrusion.extrude_all_rooms()
    geometry = extrusion.get_geometry()
    
    logger.info(f"✓ Created {len(geometry['vertices'])} vertices")
    logger.info(f"  Created {len(geometry['faces'])} faces")
    
    # Export
    logger.info("\nExporting OBJ...")
    output_dir = Path("pipeline_output/ground_truth_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exporter = OBJExporter()
    exporter.load_geometry(geometry)
    exporter.create_materials_from_colors()
    
    obj_path = output_dir / f"{test_image_name}_gt.obj"
    mtl_path = output_dir / f"{test_image_name}_gt.mtl"
    
    exporter.export_obj(str(obj_path), mtl_filename=f"{test_image_name}_gt.mtl")
    exporter.export_mtl(str(mtl_path))
    
    logger.info(f"✓ Exported OBJ: {obj_path.name} ({obj_path.stat().st_size / 1024:.1f} KB)")
    logger.info(f"✓ Exported MTL: {mtl_path.name}")
    
    # Analysis
    logger.info(f"\n{'='*60}")
    logger.info("GROUND TRUTH TEST COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"\nOpen this file in Blender to see if it looks correct:")
    logger.info(f"  {obj_path.absolute()}")
    logger.info(f"\nIf this looks good:")
    logger.info(f"  → Problem is in FPN segmentation (not boundary/extrusion)")
    logger.info(f"\nIf this still looks bad:")
    logger.info(f"  → Problem is in boundary extraction or wall extrusion")
    
    return {
        'obj_path': str(obj_path),
        'num_vertices': len(geometry['vertices']),
        'num_faces': len(geometry['faces']),
        'num_rooms': len(extractor.rooms)
    }


if __name__ == "__main__":
    result = test_ground_truth_pipeline()
    if result:
        print(f"\n✓ Test completed")
        print(f"  OBJ: {result['obj_path']}")
        print(f"  Geometry: {result['num_vertices']} vertices, {result['num_faces']} faces")
