"""
Main 3D Pipeline v3 - With Segmentation Post-Processing
========================================================

Pipeline:
1. FPN Inference: Image → Raw segmentation mask
2. **NEW** Post-Processing: Raw mask → Clean mask (removes noise, straightens edges)
3. Boundary Extraction v2: Clean mask → Room polygons
4. Wall Extrusion v6: Polygons → 3D geometry
5. OBJ Export: Geometry → OBJ + MTL files

The post-processing step significantly improves output quality by:
- Removing small noise regions
- Filling holes inside rooms
- Smoothing jagged boundaries
- Straightening edges to be horizontal/vertical

Author: Asheleyine's Master thesis project
"""

import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import cv2
import logging
from datetime import datetime

# Import pipeline components
from fpn_inference import FPNInference
from segmentation_postprocess import SegmentationPostProcessor
from boundary_extraction2 import BoundaryExtractorV2
from wall_extrusion_v6 import WallExtrusionV6, OBJExporter


# ============================================================================
# PATH HELPERS (avoid encoding issues with special characters in filenames)
# ============================================================================

def get_default_model_path():
    """Get the default FPN model path relative to this script."""
    return Path(__file__).parent.parent / "U-NET" / "scripts" / "model_output" / "fpn_62images" / "best_model.pth"


def get_default_floorplan_images(count=2):
    """Get default floorplan images using glob to avoid encoding issues."""
    floorplan_dir = Path(__file__).parent.parent / "U-NET" / "data" / "floorplan"
    
    if not floorplan_dir.exists():
        return []
    
    # Get all PNG files and sort them
    all_images = sorted(floorplan_dir.glob("*.png"))
    return [str(img) for img in all_images[:count]]


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Pipeline3DV3:
    """
    Complete 3D pipeline with segmentation post-processing.
    
    This is the recommended pipeline for production use.
    """
    
    def __init__(self,
                 fpn_model_path: str,
                 output_dir: str = "pipeline_output_v3",
                 # Post-processing params
                 postprocess_min_area: int = 2000,
                 enable_straightening: bool = True,
                 # Boundary extraction params
                 min_contour_area: int = 3000,
                 # Extrusion params
                 room_height: float = 2.5):
        """
        Initialize the pipeline.
        
        Args:
            fpn_model_path: Path to trained FPN model
            output_dir: Base directory for outputs
            postprocess_min_area: Min region area for post-processing
            enable_straightening: Whether to straighten edges
            min_contour_area: Min area for polygon extraction
            room_height: Room height in meters
        """
        self.fpn_model_path = Path(fpn_model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store params
        self.postprocess_min_area = postprocess_min_area
        self.enable_straightening = enable_straightening
        self.min_contour_area = min_contour_area
        self.room_height = room_height
        
        # Initialize components
        logger.info("=" * 60)
        logger.info("INITIALIZING PIPELINE v3 (with post-processing)")
        logger.info("=" * 60)
        
        logger.info(f"Loading FPN model...")
        self.fpn = FPNInference(str(self.fpn_model_path))
        
        logger.info(f"Post-processor: min_area={postprocess_min_area}, straightening={enable_straightening}")
        logger.info(f"Boundary extractor: min_contour_area={min_contour_area}")
        logger.info(f"Wall extrusion: height={room_height}m")
        
        logger.info("✓ Pipeline v3 initialized")
    
    def process_image(self,
                      image_path: str,
                      output_name: Optional[str] = None,
                      save_intermediate: bool = True,
                      save_comparison: bool = True) -> Dict:
        """
        Process a single floor plan image.
        
        Args:
            image_path: Path to input floor plan image
            output_name: Name for output files
            save_intermediate: Save intermediate outputs
            save_comparison: Save before/after post-processing comparison
        
        Returns:
            Result dict
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            return {"status": "error", "error": f"Image not found: {image_path}"}
        
        if output_name is None:
            output_name = image_path.stem
        
        task_dir = self.output_dir / output_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"PROCESSING: {image_path.name}")
        logger.info("=" * 60)
        
        metadata = {
            "image_name": output_name,
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": "v3",
            "steps": {}
        }
        
        try:
            # ==================== STEP 1: FPN INFERENCE ====================
            logger.info("\n[1/5] FPN Inference")
            
            raw_mask, original_shape, classes_present = self.fpn.infer(image_path)
            
            # Save raw mask
            if save_intermediate:
                raw_mask_path = task_dir / f"{output_name}_mask_raw.png"
                self.fpn.save_mask(raw_mask, raw_mask_path)
            
            metadata["steps"]["fpn_inference"] = {
                "mask_shape": list(raw_mask.shape),
                "classes_detected": list(classes_present.keys()),
            }
            
            logger.info(f"  ✓ Raw mask: {raw_mask.shape}")
            logger.info(f"  ✓ Classes: {', '.join(classes_present.keys())}")
            
            # ==================== STEP 2: POST-PROCESSING ====================
            logger.info("\n[2/5] Segmentation Post-Processing")
            
            processor = SegmentationPostProcessor(
                min_region_area=self.postprocess_min_area,
                enable_straightening=self.enable_straightening
            )
            clean_mask = processor.process(raw_mask)
            
            # Save cleaned mask
            clean_mask_path = task_dir / f"{output_name}_mask_clean.png"
            cv2.imwrite(str(clean_mask_path), clean_mask)
            
            # Save comparison
            if save_comparison:
                comparison_path = task_dir / f"{output_name}_mask_comparison.png"
                processor.visualize_comparison(raw_mask, clean_mask, str(comparison_path))
            
            metadata["steps"]["postprocessing"] = {
                "regions_removed": processor.stats["regions_removed"],
                "holes_filled": processor.stats["holes_filled"],
                "pixels_changed": processor.stats["pixels_changed"],
                "change_percent": round(100 * processor.stats["pixels_changed"] / raw_mask.size, 2),
                "clean_mask_path": str(clean_mask_path),
            }
            
            logger.info(f"  ✓ Cleaned mask saved")
            logger.info(f"  ✓ Changes: {metadata['steps']['postprocessing']['change_percent']}% pixels")
            
            # ==================== STEP 3: BOUNDARY EXTRACTION ====================
            logger.info("\n[3/5] Boundary Extraction (v2)")
            
            extractor = BoundaryExtractorV2(
                min_contour_area=self.min_contour_area,
                enable_morphology=False  # Already done in post-processing
            )
            extractor.load_mask_array(clean_mask)
            polygons = extractor.extract_polygons()
            extractor.detect_walls()
            extractor.calculate_scale_factor()
            
            boundaries_data = {
                "rooms": extractor.get_rooms_for_extrusion(),
                "walls": extractor.walls,
                "scale_factor": extractor.scale_factor
            }
            
            # Save boundaries
            boundaries_path = task_dir / f"{output_name}_boundaries.json"
            with open(boundaries_path, 'w') as f:
                json.dump(boundaries_data, f, indent=2)
            
            # Save visualization
            if save_intermediate:
                viz_path = task_dir / f"{output_name}_boundaries.png"
                extractor.visualize(str(viz_path))
            
            metadata["steps"]["boundary_extraction"] = {
                "total_rooms": len(boundaries_data["rooms"]),
                "scale_factor": boundaries_data["scale_factor"],
                "convex_rooms": extractor.stats["convex_polygons"],
                "non_convex_rooms": extractor.stats["non_convex_polygons"],
            }
            
            logger.info(f"  ✓ Rooms: {len(boundaries_data['rooms'])}")
            logger.info(f"  ✓ Convex: {extractor.stats['convex_polygons']}, Non-convex: {extractor.stats['non_convex_polygons']}")
            
            # ==================== STEP 4: WALL EXTRUSION ====================
            logger.info("\n[4/5] Wall Extrusion (v6)")
            
            extruder = WallExtrusionV6(
                room_height=self.room_height,
                scale_factor=boundaries_data["scale_factor"]
            )
            extruder.load_boundaries(boundaries_data)
            extruder.extrude_all_rooms()
            
            geometry = extruder.get_geometry()
            bounds = extruder.get_bounds()
            
            metadata["steps"]["wall_extrusion"] = {
                "vertices": len(geometry["vertices"]),
                "triangles": geometry["stats"]["triangles_created"],
                "rooms_processed": geometry["stats"]["rooms_processed"],
                "rooms_skipped": geometry["stats"]["rooms_skipped"],
                "bounds": bounds,
            }
            
            logger.info(f"  ✓ Vertices: {len(geometry['vertices'])}")
            logger.info(f"  ✓ Triangles: {geometry['stats']['triangles_created']}")
            
            # ==================== STEP 5: OBJ EXPORT ====================
            logger.info("\n[5/5] OBJ Export")
            
            exporter = OBJExporter()
            exporter.load_geometry(geometry)
            exporter.create_materials()
            
            obj_path = task_dir / f"{output_name}.obj"
            mtl_path = task_dir / f"{output_name}.mtl"
            
            exporter.export_obj(str(obj_path), f"{output_name}.mtl")
            exporter.export_mtl(str(mtl_path))
            
            metadata["steps"]["obj_export"] = {
                "obj_file": str(obj_path),
                "mtl_file": str(mtl_path),
                "obj_size_kb": round(obj_path.stat().st_size / 1024, 2),
            }
            
            logger.info(f"  ✓ OBJ: {obj_path.name}")
            
            # ==================== SUMMARY ====================
            logger.info("")
            logger.info("=" * 60)
            logger.info("✓ PIPELINE COMPLETE")
            logger.info("=" * 60)
            logger.info(f"  Output: {task_dir}/")
            logger.info(f"  Model size: {bounds['size'][0]:.1f} x {bounds['size'][2]:.1f} x {bounds['size'][1]:.1f} m")
            
            # Save metadata
            metadata_path = task_dir / f"{output_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                "status": "success",
                "image_path": str(image_path),
                "output_dir": str(task_dir),
                "obj_file": str(obj_path),
                "mtl_file": str(mtl_path),
                "clean_mask_path": str(clean_mask_path),
                "metadata": metadata,
                "bounds": bounds,
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {
                "status": "error",
                "image_path": str(image_path),
                "error": str(e),
                "metadata": metadata
            }
    
    def process_mask_only(self,
                          mask_path: str,
                          output_name: Optional[str] = None,
                          save_intermediate: bool = True) -> Dict:
        """
        Process starting from an existing mask (skip FPN inference).
        
        Useful for testing post-processing on pre-generated masks.
        """
        mask_path = Path(mask_path)
        
        if not mask_path.exists():
            return {"status": "error", "error": f"Mask not found: {mask_path}"}
        
        if output_name is None:
            output_name = mask_path.stem.replace("_mask", "")
        
        task_dir = self.output_dir / output_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Load mask
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            return {"status": "error", "error": f"Cannot read mask: {mask_path}"}
        
        logger.info(f"\nProcessing mask: {mask_path.name}")
        
        # Post-process
        processor = SegmentationPostProcessor(
            min_region_area=self.postprocess_min_area,
            enable_straightening=self.enable_straightening
        )
        clean_mask = processor.process(raw_mask)
        
        # Save comparison
        comparison_path = task_dir / f"{output_name}_comparison.png"
        processor.visualize_comparison(raw_mask, clean_mask, str(comparison_path))
        
        # Continue with extraction and extrusion...
        extractor = BoundaryExtractorV2(min_contour_area=self.min_contour_area)
        extractor.load_mask_array(clean_mask)
        extractor.extract_polygons()
        extractor.calculate_scale_factor()
        
        boundaries_data = {
            "rooms": extractor.get_rooms_for_extrusion(),
            "walls": [],
            "scale_factor": extractor.scale_factor
        }
        
        extruder = WallExtrusionV6(room_height=self.room_height)
        extruder.load_boundaries(boundaries_data)
        extruder.extrude_all_rooms()
        
        geometry = extruder.get_geometry()
        
        exporter = OBJExporter()
        exporter.load_geometry(geometry)
        
        obj_path = task_dir / f"{output_name}.obj"
        mtl_path = task_dir / f"{output_name}.mtl"
        
        exporter.export_obj(str(obj_path), f"{output_name}.mtl")
        exporter.export_mtl(str(mtl_path))
        
        logger.info(f"✓ Output: {obj_path}")
        
        return {
            "status": "success",
            "obj_file": str(obj_path),
            "comparison_path": str(comparison_path),
        }


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    # Get default model path using helper
    default_model = str(get_default_model_path())
    
    parser = argparse.ArgumentParser(description="3D Floorplan Pipeline v3")
    parser.add_argument("--image", "-i", help="Input floorplan image")
    parser.add_argument("--mask", "-m", help="Input mask (skip FPN inference)")
    parser.add_argument("--model", 
                        default=default_model,
                        help="Path to FPN model")
    parser.add_argument("--output", "-o", default="pipeline_output_v3",
                        help="Output directory")
    parser.add_argument("--height", type=float, default=2.5,
                        help="Room height in meters")
    parser.add_argument("--no-straighten", action="store_true",
                        help="Disable edge straightening")
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ ERROR: FPN model not found at: {model_path}")
        print("Please specify --model with a valid path.")
        return
    
    pipeline = Pipeline3DV3(
        fpn_model_path=args.model,
        output_dir=args.output,
        enable_straightening=not args.no_straighten,
        room_height=args.height
    )
    
    if args.mask:
        result = pipeline.process_mask_only(args.mask)
    elif args.image:
        result = pipeline.process_image(args.image)
    else:
        # Use default floorplan images (first 2 from the folder)
        test_images = get_default_floorplan_images(count=2)
        
        if test_images:
            print(f"No --image provided. Using {len(test_images)} default floorplan images:")
            for i, img in enumerate(test_images, 1):
                print(f"  {i}. {Path(img).name}")
            print()
            
            for img in test_images:
                result = pipeline.process_image(str(img))
                if result["status"] == "success":
                    print(f"\n✓ Success! OBJ: {result['obj_file']}")
                else:
                    print(f"\n✗ Failed: {result.get('error')}")
            return
        else:
            print("No test images found. Use --image or --mask to specify input.")
            return
    
    if result["status"] == "success":
        print(f"\n✓ Success! OBJ: {result['obj_file']}")
    else:
        print(f"\n✗ Failed: {result.get('error')}")


if __name__ == "__main__":
    main()
