"""
Main 3D Pipeline v4 - With Rectangular Room Fitting
====================================================

Pipeline:
1. FPN Inference: Image → Raw segmentation mask
2. Post-Processing: Raw mask → Clean mask
3. **NEW** Rectangular Fitting: Clean mask → Rectangular rooms
4. Boundary Extraction: Rectangular mask → Room polygons  
5. Wall Extrusion: Polygons → 3D geometry
6. OBJ Export: Geometry → OBJ + MTL files

The rectangular fitting step converts blob-like room shapes into
clean rectangles or L-shapes, making the 3D output look like
actual architectural floor plans.

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
from rectangular_room_fitter import RectangularRoomFitter
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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Pipeline3DV4:
    """
    Complete 3D pipeline with rectangular room fitting.
    
    This produces the cleanest output by:
    1. Filtering noise (post-processing)
    2. Making rooms rectangular (fitting)
    3. Proper triangulation (ear-clipping)
    """
    
    def __init__(self,
                 fpn_model_path: str,
                 output_dir: str = "pipeline_output_v4",
                 # Post-processing params
                 postprocess_min_area: int = 2000,
                 enable_straightening: bool = True,
                 # Rectangular fitting params
                 enable_rectangular_fitting: bool = True,
                 min_rectangle_coverage: float = 0.70,
                 enable_lshape: bool = True,
                 # Boundary extraction params
                 min_contour_area: int = 2000,
                 # Extrusion params
                 room_height: float = 2.5):
        """
        Initialize the pipeline.
        
        Args:
            fpn_model_path: Path to trained FPN model
            output_dir: Base directory for outputs
            postprocess_min_area: Min region area for post-processing
            enable_straightening: Straighten edges in post-processing
            enable_rectangular_fitting: Enable rectangular room fitting
            min_rectangle_coverage: Min coverage for rectangle fit acceptance
            enable_lshape: Enable L-shape fitting for complex rooms
            min_contour_area: Min area for polygon extraction
            room_height: Room height in meters
        """
        self.fpn_model_path = Path(fpn_model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store params
        self.postprocess_min_area = postprocess_min_area
        self.enable_straightening = enable_straightening
        self.enable_rectangular_fitting = enable_rectangular_fitting
        self.min_rectangle_coverage = min_rectangle_coverage
        self.enable_lshape = enable_lshape
        self.min_contour_area = min_contour_area
        self.room_height = room_height
        
        # Initialize FPN
        logger.info("=" * 60)
        logger.info("INITIALIZING PIPELINE v4 (with rectangular fitting)")
        logger.info("=" * 60)
        
        logger.info(f"Loading FPN model...")
        self.fpn = FPNInference(str(self.fpn_model_path))
        
        logger.info(f"Post-processor: min_area={postprocess_min_area}")
        logger.info(f"Rectangular fitting: enabled={enable_rectangular_fitting}, "
                   f"min_coverage={min_rectangle_coverage}")
        logger.info(f"Room height: {room_height}m")
        
        logger.info("✓ Pipeline v4 initialized")
    
    def process_image(self,
                      image_path: str,
                      output_name: Optional[str] = None,
                      save_intermediate: bool = True) -> Dict:
        """
        Process a single floor plan image.
        
        Args:
            image_path: Path to input floor plan image
            output_name: Name for output files
            save_intermediate: Save intermediate outputs
        
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
            "pipeline_version": "v4",
            "steps": {}
        }
        
        try:
            # ==================== STEP 1: FPN INFERENCE ====================
            logger.info("\n[1/6] FPN Inference")
            
            raw_mask, original_shape, classes_present = self.fpn.infer(image_path)
            
            if save_intermediate:
                raw_mask_path = task_dir / f"{output_name}_1_mask_raw.png"
                self.fpn.save_mask(raw_mask, raw_mask_path)
            
            metadata["steps"]["fpn_inference"] = {
                "mask_shape": list(raw_mask.shape),
                "classes_detected": list(classes_present.keys()),
            }
            
            logger.info(f"  ✓ Classes: {', '.join(classes_present.keys())}")
            
            # ==================== STEP 2: POST-PROCESSING ====================
            logger.info("\n[2/6] Post-Processing")
            
            postprocessor = SegmentationPostProcessor(
                min_region_area=self.postprocess_min_area,
                enable_straightening=self.enable_straightening
            )
            clean_mask = postprocessor.process(raw_mask)
            
            if save_intermediate:
                clean_mask_path = task_dir / f"{output_name}_2_mask_clean.png"
                cv2.imwrite(str(clean_mask_path), clean_mask)
                
                # Comparison
                comparison_path = task_dir / f"{output_name}_2_postprocess_comparison.png"
                postprocessor.visualize_comparison(raw_mask, clean_mask, str(comparison_path))
            
            metadata["steps"]["postprocessing"] = {
                "regions_removed": postprocessor.stats["regions_removed"],
                "holes_filled": postprocessor.stats["holes_filled"],
                "pixels_changed": postprocessor.stats["pixels_changed"],
            }
            
            logger.info(f"  ✓ Removed {postprocessor.stats['regions_removed']} noise regions")
            
            # ==================== STEP 3: RECTANGULAR FITTING ====================
            if self.enable_rectangular_fitting:
                logger.info("\n[3/6] Rectangular Room Fitting")
                
                fitter = RectangularRoomFitter(
                    min_rectangle_coverage=self.min_rectangle_coverage,
                    enable_lshape=self.enable_lshape,
                    min_room_area=self.min_contour_area
                )
                rect_mask = fitter.fit_mask(clean_mask)
                
                if save_intermediate:
                    rect_mask_path = task_dir / f"{output_name}_3_mask_rectangular.png"
                    cv2.imwrite(str(rect_mask_path), rect_mask)
                    
                    # Comparison
                    rect_comparison_path = task_dir / f"{output_name}_3_rectangular_comparison.png"
                    fitter.visualize_comparison(clean_mask, rect_mask, str(rect_comparison_path))
                
                fitting_summary = fitter.get_summary()
                metadata["steps"]["rectangular_fitting"] = {
                    "rectangle_fits": fitting_summary["stats"]["rectangle_fits"],
                    "lshape_fits": fitting_summary["stats"]["lshape_fits"],
                    "kept_original": fitting_summary["stats"]["kept_original"],
                    "total_rooms": fitting_summary["stats"]["total_rooms"],
                }
                
                logger.info(f"  ✓ Rectangle fits: {fitting_summary['stats']['rectangle_fits']}")
                logger.info(f"  ✓ L-shape fits: {fitting_summary['stats']['lshape_fits']}")
                logger.info(f"  ✓ Kept original: {fitting_summary['stats']['kept_original']}")
                
                final_mask = rect_mask
            else:
                logger.info("\n[3/6] Rectangular Fitting (SKIPPED)")
                final_mask = clean_mask
                metadata["steps"]["rectangular_fitting"] = {"enabled": False}
            
            # ==================== STEP 4: BOUNDARY EXTRACTION ====================
            logger.info("\n[4/6] Boundary Extraction")
            
            extractor = BoundaryExtractorV2(
                min_contour_area=self.min_contour_area,
                enable_morphology=False  # Already done
            )
            extractor.load_mask_array(final_mask)
            extractor.extract_polygons()
            extractor.calculate_scale_factor()
            
            boundaries_data = {
                "rooms": extractor.get_rooms_for_extrusion(),
                "walls": [],
                "scale_factor": extractor.scale_factor
            }
            
            if save_intermediate:
                boundaries_path = task_dir / f"{output_name}_4_boundaries.json"
                with open(boundaries_path, 'w') as f:
                    json.dump(boundaries_data, f, indent=2)
                
                viz_path = task_dir / f"{output_name}_4_boundaries.png"
                extractor.visualize(str(viz_path))
            
            metadata["steps"]["boundary_extraction"] = {
                "total_rooms": len(boundaries_data["rooms"]),
                "scale_factor": boundaries_data["scale_factor"],
                "convex_rooms": extractor.stats["convex_polygons"],
                "non_convex_rooms": extractor.stats["non_convex_polygons"],
            }
            
            logger.info(f"  ✓ Rooms: {len(boundaries_data['rooms'])}")
            logger.info(f"  ✓ Convex: {extractor.stats['convex_polygons']}, "
                       f"Non-convex: {extractor.stats['non_convex_polygons']}")
            
            # ==================== STEP 5: WALL EXTRUSION ====================
            logger.info("\n[5/6] Wall Extrusion")
            
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
            
            # ==================== STEP 6: OBJ EXPORT ====================
            logger.info("\n[6/6] OBJ Export")
            
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
            logger.info("✓ PIPELINE V4 COMPLETE")
            logger.info("=" * 60)
            logger.info(f"  Output: {task_dir}/")
            logger.info(f"  Size: {bounds['size'][0]:.1f} x {bounds['size'][2]:.1f} x {bounds['size'][1]:.1f} m")
            
            # Save metadata
            metadata_path = task_dir / f"{output_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save final colored mask for reference
            final_mask_colored = self._colorize_mask(final_mask)
            cv2.imwrite(str(task_dir / f"{output_name}_final_mask.png"), final_mask_colored)
            
            return {
                "status": "success",
                "image_path": str(image_path),
                "output_dir": str(task_dir),
                "obj_file": str(obj_path),
                "mtl_file": str(mtl_path),
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
    
    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create colored visualization of mask."""
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


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    # Get default model path using helper
    default_model = str(get_default_model_path())
    
    parser = argparse.ArgumentParser(description="3D Floorplan Pipeline v4")
    parser.add_argument("--image", "-i", help="Input floorplan image")
    parser.add_argument("--model", "-m",
                        default=default_model,
                        help="Path to FPN model")
    parser.add_argument("--output", "-o", default="pipeline_output_v4",
                        help="Output directory")
    parser.add_argument("--height", type=float, default=2.5,
                        help="Room height in meters")
    parser.add_argument("--no-rectangular", action="store_true",
                        help="Disable rectangular fitting")
    parser.add_argument("--min-coverage", type=float, default=0.70,
                        help="Minimum coverage for rectangle fit")
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ ERROR: FPN model not found at: {model_path}")
        print("Please specify --model with a valid path.")
        return
    
    pipeline = Pipeline3DV4(
        fpn_model_path=args.model,
        output_dir=args.output,
        enable_rectangular_fitting=not args.no_rectangular,
        min_rectangle_coverage=args.min_coverage,
        room_height=args.height
    )
    
    if args.image:
        result = pipeline.process_image(args.image)
        
        if result["status"] == "success":
            print(f"\n✓ Success!")
            print(f"  OBJ: {result['obj_file']}")
            print(f"  Size: {result['bounds']['size']}")
        else:
            print(f"\n✗ Failed: {result.get('error')}")
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
        else:
            print("No test images found. Use --image to specify input.")


if __name__ == "__main__":
    main()
