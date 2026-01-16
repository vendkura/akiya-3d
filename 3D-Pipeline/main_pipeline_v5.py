"""
Main 3D Pipeline v5 - Dollhouse Style with Wall-Based Extrusion
================================================================

Creates architectural 3D floor plans where:
- Walls are solid geometry with thickness
- Rooms are empty spaces you can see into
- Floor is a solid plane at the bottom
- View is open from top (dollhouse style)

Pipeline:
1. FPN Inference: Image → Raw segmentation mask
2. Post-Processing: Raw mask → Clean mask
3. Rectangular Fitting: Clean mask → Rectangular rooms
4. Wall Detection: Find walls between rooms
5. Wall Extrusion: Create 3D walls with thickness
6. Floor Generation: Add floor plane
7. OBJ Export: Save 3D model

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
from wall_based_extrusion import WallDetector, WallBasedExtruder, WallOBJExporter


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


class Pipeline3DV5:
    """
    Dollhouse-style 3D floor plan generator.
    
    Creates proper architectural models with:
    - Walls as solid geometry
    - Rooms as walkable spaces
    - Visible interior (dollhouse view)
    """
    
    def __init__(self,
                 fpn_model_path: str,
                 output_dir: str = "pipeline_output_v5",
                 # Post-processing params
                 postprocess_min_area: int = 2000,
                 # Rectangular fitting params
                 enable_rectangular_fitting: bool = True,
                 min_rectangle_coverage: float = 0.70,
                 # Wall params
                 wall_thickness: float = 0.12,
                 wall_height: float = 2.5):
        """
        Initialize the pipeline.
        
        Args:
            fpn_model_path: Path to trained FPN model
            output_dir: Base directory for outputs
            postprocess_min_area: Min region area for post-processing
            enable_rectangular_fitting: Enable rectangular room fitting
            min_rectangle_coverage: Min coverage for rectangle fit
            wall_thickness: Wall thickness in meters (default 12cm)
            wall_height: Wall height in meters (default 2.5m)
        """
        self.fpn_model_path = Path(fpn_model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store params
        self.postprocess_min_area = postprocess_min_area
        self.enable_rectangular_fitting = enable_rectangular_fitting
        self.min_rectangle_coverage = min_rectangle_coverage
        self.wall_thickness = wall_thickness
        self.wall_height = wall_height
        
        # Initialize FPN
        logger.info("=" * 60)
        logger.info("INITIALIZING PIPELINE v5 (Dollhouse Style)")
        logger.info("=" * 60)
        
        logger.info(f"Loading FPN model...")
        self.fpn = FPNInference(str(self.fpn_model_path))
        
        logger.info(f"Wall thickness: {wall_thickness}m ({wall_thickness*100:.0f}cm)")
        logger.info(f"Wall height: {wall_height}m")
        logger.info(f"Style: Dollhouse (open top, visible interior)")
        
        logger.info("✓ Pipeline v5 initialized")
    
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
            "pipeline_version": "v5_dollhouse",
            "params": {
                "wall_thickness": self.wall_thickness,
                "wall_height": self.wall_height,
            },
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
                enable_straightening=True
            )
            clean_mask = postprocessor.process(raw_mask)
            
            if save_intermediate:
                cv2.imwrite(str(task_dir / f"{output_name}_2_mask_clean.png"), clean_mask)
            
            metadata["steps"]["postprocessing"] = {
                "regions_removed": postprocessor.stats["regions_removed"],
                "pixels_changed": postprocessor.stats["pixels_changed"],
            }
            
            logger.info(f"  ✓ Removed {postprocessor.stats['regions_removed']} noise regions")
            
            # ==================== STEP 3: RECTANGULAR FITTING ====================
            if self.enable_rectangular_fitting:
                logger.info("\n[3/6] Rectangular Fitting")
                
                fitter = RectangularRoomFitter(
                    min_rectangle_coverage=self.min_rectangle_coverage,
                    enable_lshape=True
                )
                rect_mask = fitter.fit_mask(clean_mask)
                
                if save_intermediate:
                    cv2.imwrite(str(task_dir / f"{output_name}_3_mask_rectangular.png"), rect_mask)
                
                summary = fitter.get_summary()
                metadata["steps"]["rectangular_fitting"] = {
                    "rectangle_fits": summary["stats"]["rectangle_fits"],
                    "lshape_fits": summary["stats"]["lshape_fits"],
                    "kept_original": summary["stats"]["kept_original"],
                }
                
                logger.info(f"  ✓ Fitted {summary['stats']['rectangle_fits']} rectangles, "
                           f"{summary['stats']['lshape_fits']} L-shapes")
                
                final_mask = rect_mask
            else:
                logger.info("\n[3/6] Rectangular Fitting (SKIPPED)")
                final_mask = clean_mask
            
            # ==================== STEP 4: CALCULATE SCALE ====================
            logger.info("\n[4/6] Calculate Scale")
            
            # Calculate scale factor based on mask size
            # Assuming average room width of ~4 meters
            non_zero = np.argwhere(final_mask > 0)
            if len(non_zero) > 0:
                min_y, min_x = non_zero.min(axis=0)
                max_y, max_x = non_zero.max(axis=0)
                width_px = max_x - min_x
                # Assume total width is about 12-15 meters for typical Japanese house
                scale_factor = 12.0 / width_px
            else:
                scale_factor = 0.01
            
            metadata["steps"]["scale"] = {
                "scale_factor": scale_factor,
                "meters_per_pixel": scale_factor
            }
            
            logger.info(f"  ✓ Scale: {scale_factor:.6f} m/px")
            
            # ==================== STEP 5: WALL DETECTION & EXTRUSION ====================
            logger.info("\n[5/6] Wall Detection & Extrusion")
            
            # Detect walls (filter small fragments with min_wall_length)
            detector = WallDetector(
                wall_thickness=self.wall_thickness,
                min_wall_length=0.3  # Filter walls shorter than 30cm
            )
            walls = detector.detect_from_mask(final_mask, scale_factor)
            
            # Extrude walls
            extruder = WallBasedExtruder(
                wall_thickness=self.wall_thickness,
                wall_height=self.wall_height
            )
            extruder.extrude_walls(walls)
            
            # Add colored floor (each room type has its own color)
            extruder.add_colored_floor(final_mask, scale_factor)
            
            geometry = extruder.get_geometry()
            bounds = extruder.get_bounds()
            
            metadata["steps"]["wall_extrusion"] = {
                "num_walls": len(walls),
                "num_vertices": len(geometry["vertices"]),
                "num_faces": len(geometry["faces"]),
                "bounds": bounds,
            }
            
            logger.info(f"  ✓ Walls: {len(walls)}")
            logger.info(f"  ✓ Vertices: {len(geometry['vertices'])}")
            logger.info(f"  ✓ Faces: {len(geometry['faces'])}")
            
            # ==================== STEP 6: OBJ EXPORT ====================
            logger.info("\n[6/6] OBJ Export")
            
            exporter = WallOBJExporter()
            exporter.load_geometry(geometry)
            
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
            
            # ==================== SAVE FINAL MASK ====================
            final_colored = self._colorize_mask(final_mask)
            cv2.imwrite(str(task_dir / f"{output_name}_final_mask.png"), final_colored)
            
            # ==================== SUMMARY ====================
            logger.info("")
            logger.info("=" * 60)
            logger.info("✓ PIPELINE V5 COMPLETE (Dollhouse Style)")
            logger.info("=" * 60)
            logger.info(f"  Output: {task_dir}/")
            logger.info(f"  Size: {bounds['size'][0]:.1f} x {bounds['size'][2]:.1f} x {bounds['size'][1]:.1f} m")
            logger.info(f"  Walls: {len(walls)} segments")
            
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
    
    parser = argparse.ArgumentParser(description="3D Floorplan Pipeline v5 (Dollhouse)")
    parser.add_argument("--image", "-i", help="Input floorplan image")
    parser.add_argument("--model", "-m",
                        default=default_model,
                        help="Path to FPN model")
    parser.add_argument("--output", "-o", default="pipeline_output_v5",
                        help="Output directory")
    parser.add_argument("--wall-thickness", type=float, default=0.12,
                        help="Wall thickness in meters")
    parser.add_argument("--wall-height", type=float, default=2.5,
                        help="Wall height in meters")
    parser.add_argument("--no-rectangular", action="store_true",
                        help="Disable rectangular fitting")
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ ERROR: FPN model not found at: {model_path}")
        print("Please specify --model with a valid path.")
        return
    
    pipeline = Pipeline3DV5(
        fpn_model_path=args.model,
        output_dir=args.output,
        enable_rectangular_fitting=not args.no_rectangular,
        wall_thickness=args.wall_thickness,
        wall_height=args.wall_height
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