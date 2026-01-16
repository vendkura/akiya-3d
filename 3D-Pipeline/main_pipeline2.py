"""
Main 3D Pipeline v2 - Updated with robust polygon extraction and triangulation

Changes from v1:
- Uses BoundaryExtractorV2 (min_area=3000, morphology cleanup)
- Uses WallExtrusionV6 (ear-clipping triangulation)
- Better error handling and logging
- Intermediate output saving for debugging

Pipeline:
1. FPN Inference: Image → Segmentation mask
2. Boundary Extraction v2: Mask → Clean room polygons
3. Wall Extrusion v6: Polygons → 3D geometry (with ear-clipping)
4. OBJ Export: Geometry → OBJ + MTL files

Author: Asheleyine's Master thesis project
"""

import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from PIL import Image
import logging
from datetime import datetime

# Import pipeline components
from fpn_inference import FPNInference
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


class Pipeline3DV2:
    """
    End-to-end 3D pipeline with improved polygon handling.
    
    Key improvements:
    - Filters noise polygons (min_area=3000)
    - Handles non-convex rooms correctly (ear-clipping)
    - Better validation and error recovery
    """
    
    def __init__(self,
                 fpn_model_path: str,
                 output_dir: str = "pipeline_output_v2",
                 min_contour_area: int = 3000,
                 room_height: float = 2.5):
        """
        Initialize the pipeline.
        
        Args:
            fpn_model_path: Path to trained FPN model (.pth)
            output_dir: Base directory for outputs
            min_contour_area: Minimum polygon area (filters noise)
            room_height: Room height in meters for extrusion
        """
        self.fpn_model_path = Path(fpn_model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_contour_area = min_contour_area
        self.room_height = room_height
        
        # Initialize components
        logger.info("="*60)
        logger.info("INITIALIZING PIPELINE v2")
        logger.info("="*60)
        
        logger.info(f"Loading FPN model from {fpn_model_path}...")
        self.fpn = FPNInference(str(self.fpn_model_path))
        
        logger.info(f"Boundary extractor: min_area={min_contour_area}")
        logger.info(f"Wall extrusion: height={room_height}m, ear-clipping enabled")
        
        logger.info("✓ Pipeline v2 initialized")
    
    def process_image(self,
                      image_path: str,
                      output_name: Optional[str] = None,
                      save_intermediate: bool = True) -> Dict:
        """
        Process a single floor plan image.
        
        Args:
            image_path: Path to input floor plan image
            output_name: Name for output files (default: image basename)
            save_intermediate: Save intermediate outputs for debugging
        
        Returns:
            Result dict with status, paths, metadata
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            return {"status": "error", "error": f"Image not found: {image_path}"}
        
        # Setup output
        if output_name is None:
            output_name = image_path.stem
        
        task_dir = self.output_dir / output_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("")
        logger.info("="*60)
        logger.info(f"PROCESSING: {image_path.name}")
        logger.info("="*60)
        
        metadata = {
            "image_name": output_name,
            "timestamp": datetime.now().isoformat(),
            "steps": {}
        }
        
        try:
            # ==================== STEP 1: FPN INFERENCE ====================
            logger.info("\n[1/4] FPN Inference")
            
            mask, original_shape, classes_present = self.fpn.infer(image_path)
            
            mask_path = task_dir / f"{output_name}_mask.png"
            self.fpn.save_mask(mask, mask_path)
            
            metadata["steps"]["fpn_inference"] = {
                "mask_shape": list(mask.shape),
                "original_shape": list(original_shape),
                "classes_detected": list(classes_present.keys()),
                "mask_path": str(mask_path)
            }
            
            logger.info(f"  ✓ Mask: {mask.shape}")
            logger.info(f"  ✓ Classes: {', '.join(classes_present.keys())}")
            
            # ==================== STEP 2: BOUNDARY EXTRACTION ====================
            logger.info("\n[2/4] Boundary Extraction (v2)")
            
            extractor = BoundaryExtractorV2(
                min_contour_area=self.min_contour_area,
                enable_morphology=True
            )
            extractor.load_mask_array(mask)
            polygons = extractor.extract_polygons()
            extractor.detect_walls()
            extractor.calculate_scale_factor()
            
            # Get data for extrusion
            boundaries_data = {
                "rooms": extractor.get_rooms_for_extrusion(),
                "walls": extractor.walls,
                "scale_factor": extractor.scale_factor
            }
            
            # Save boundaries
            boundaries_path = task_dir / f"{output_name}_boundaries_v2.json"
            with open(boundaries_path, 'w') as f:
                json.dump(boundaries_data, f, indent=2)
            
            # Save visualization
            if save_intermediate:
                viz_path = task_dir / f"{output_name}_boundaries_v2.png"
                extractor.visualize(str(viz_path))
            
            metadata["steps"]["boundary_extraction"] = {
                "total_rooms": len(boundaries_data["rooms"]),
                "total_walls": len(boundaries_data["walls"]),
                "scale_factor": boundaries_data["scale_factor"],
                "stats": extractor.stats,
                "boundaries_path": str(boundaries_path)
            }
            
            logger.info(f"  ✓ Rooms: {len(boundaries_data['rooms'])}")
            logger.info(f"  ✓ Filtered: {extractor.stats['contours_filtered_by_area']} noise polygons")
            logger.info(f"  ✓ Scale: {extractor.scale_factor:.6f} m/px")
            
            # ==================== STEP 3: WALL EXTRUSION ====================
            logger.info("\n[3/4] Wall Extrusion (v6 - ear clipping)")
            
            extruder = WallExtrusionV6(
                room_height=self.room_height,
                scale_factor=boundaries_data["scale_factor"]
            )
            extruder.load_boundaries(boundaries_data)
            extruder.extrude_all_rooms()
            
            geometry = extruder.get_geometry()
            bounds = extruder.get_bounds()
            
            # Save geometry
            if save_intermediate:
                geometry_path = task_dir / f"{output_name}_geometry.json"
                geometry_json = {
                    "vertices": geometry["vertices"].tolist(),
                    "faces": geometry["faces"].tolist(),
                    "colors": geometry["colors"].tolist(),
                    "stats": geometry["stats"],
                    "bounds": bounds
                }
                with open(geometry_path, 'w') as f:
                    json.dump(geometry_json, f, indent=2)
            
            metadata["steps"]["wall_extrusion"] = {
                "vertices": len(geometry["vertices"]),
                "faces": len(geometry["faces"]),
                "triangles": geometry["stats"]["triangles_created"],
                "rooms_processed": geometry["stats"]["rooms_processed"],
                "rooms_skipped": geometry["stats"]["rooms_skipped"],
                "bounds": bounds
            }
            
            logger.info(f"  ✓ Vertices: {len(geometry['vertices'])}")
            logger.info(f"  ✓ Triangles: {geometry['stats']['triangles_created']}")
            logger.info(f"  ✓ Rooms: {geometry['stats']['rooms_processed']} ok, "
                       f"{geometry['stats']['rooms_skipped']} skipped")
            
            # ==================== STEP 4: OBJ EXPORT ====================
            logger.info("\n[4/4] OBJ Export")
            
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
                "obj_size_kb": obj_path.stat().st_size / 1024,
                "mtl_size_kb": mtl_path.stat().st_size / 1024,
            }
            
            logger.info(f"  ✓ OBJ: {obj_path.name} ({metadata['steps']['obj_export']['obj_size_kb']:.1f} KB)")
            logger.info(f"  ✓ MTL: {mtl_path.name}")
            
            # ==================== SUMMARY ====================
            logger.info("")
            logger.info("="*60)
            logger.info("✓ PIPELINE COMPLETE")
            logger.info("="*60)
            logger.info(f"  Output: {task_dir}/")
            logger.info(f"  Model:  {obj_path.name}")
            logger.info(f"  Size:   {bounds['size'][0]:.1f} x {bounds['size'][2]:.1f} x {bounds['size'][1]:.1f} m")
            
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
                "geometry": geometry,
                "bounds": bounds
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {
                "status": "error",
                "image_path": str(image_path),
                "error": str(e),
                "metadata": metadata
            }
    
    def process_batch(self,
                      image_paths: list,
                      save_intermediate: bool = False) -> list:
        """
        Process multiple floor plan images.
        
        Args:
            image_paths: List of image paths
            save_intermediate: Save intermediate files
        
        Returns:
            List of result dicts
        """
        results = []
        
        logger.info(f"\nBatch processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"\n--- Image {i}/{len(image_paths)} ---")
            result = self.process_image(
                image_path,
                save_intermediate=save_intermediate
            )
            results.append(result)
        
        # Summary
        success = sum(1 for r in results if r["status"] == "success")
        logger.info(f"\n=== BATCH COMPLETE ===")
        logger.info(f"Success: {success}/{len(results)}")
        
        return results


# ============================================================================
# CLI
# ============================================================================

def main():
    """Run pipeline on test images."""
    import argparse
    
    # Get default model path using helper
    default_model = str(get_default_model_path())
    
    parser = argparse.ArgumentParser(description="3D Floorplan Pipeline v2")
    parser.add_argument("--image", "-i", help="Input floorplan image")
    parser.add_argument("--model", "-m", 
                        default=default_model,
                        help="Path to FPN model")
    parser.add_argument("--output", "-o", default="pipeline_output_v2",
                        help="Output directory")
    parser.add_argument("--min-area", type=int, default=3000,
                        help="Minimum polygon area")
    parser.add_argument("--height", type=float, default=2.5,
                        help="Room height in meters")
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ ERROR: FPN model not found at: {model_path}")
        print("Please specify --model with a valid path.")
        return
    
    # Initialize pipeline
    pipeline = Pipeline3DV2(
        fpn_model_path=args.model,
        output_dir=args.output,
        min_contour_area=args.min_area,
        room_height=args.height
    )
    
    if args.image:
        # Process single image
        result = pipeline.process_image(args.image)
        
        if result["status"] == "success":
            print(f"\n✓ Success!")
            print(f"  OBJ: {result['obj_file']}")
            print(f"  Size: {result['bounds']['size']}")
        else:
            print(f"\n✗ Failed: {result.get('error', 'Unknown error')}")
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
        else:
            print("No test images found. Use --image to specify input.")


if __name__ == "__main__":
    main()