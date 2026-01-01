"""
Main 3D Pipeline Orchestrator
Chains: FPN inference → boundary extraction → wall extrusion → OBJ export
Input: Floor plan PNG image
Output: OBJ + MTL files ready for 3D viewer
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image
import logging

# Import pipeline components
from fpn_inference import FPNInference
from boundary_extraction import BoundaryExtractor
from wall_extrusion_v3 import WallExtrusion3D  # Use improved v3
from export_obj import OBJExporter


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Pipeline3D:
    """
    End-to-end 3D pipeline: Image → OBJ
    
    Workflow:
    1. FPN Inference: PNG floor plan → 13-class segmentation mask
    2. Boundary Extraction: Segmentation mask → room polygons & walls
    3. Wall Extrusion: Room boundaries → 3D geometry (vertices, faces, normals)
    4. OBJ Export: 3D geometry → OBJ + MTL files
    
    Attributes:
        fpn_model_path (Path): Path to FPN best_model.pth
        output_dir (Path): Directory for all outputs
        fpn: FPNInference instance
        extractor: BoundaryExtractor instance
        extrusion: SimpleWallExtrusion instance
        exporter: OBJExporter instance
    """
    
    def __init__(
        self,
        fpn_model_path: str = "../U-NET/scripts/model_output/fpn_62images/best_model.pth",
        output_dir: str = "pipeline_output"
    ):
        """
        Initialize the 3D pipeline.
        
        Args:
            fpn_model_path (str): Path to trained FPN model
            output_dir (str): Base directory for all outputs
        """
        self.fpn_model_path = Path(fpn_model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing FPN inference...")
        self.fpn = FPNInference(self.fpn_model_path)
        
        logger.info("Initializing boundary extractor...")
        self.extractor = BoundaryExtractor()
        
        logger.info("Initializing wall extrusion (v3 - improved)...")
        self.extrusion = WallExtrusion3D()
        
        logger.info("Initializing OBJ exporter...")
        self.exporter = OBJExporter()
        
        logger.info("✓ Pipeline initialized successfully")
    
    def process_image(
        self,
        image_path: str,
        output_name: Optional[str] = None,
        save_intermediate: bool = True
    ) -> Dict:
        """
        Process a single floor plan image through the full pipeline.
        
        Args:
            image_path (str): Path to input floor plan PNG/JPG
            output_name (str, optional): Name for output files (default: image basename)
            save_intermediate (bool): Save intermediate outputs (masks, boundaries, geometry)
        
        Returns:
            result (dict): Pipeline results with keys:
                - 'status': 'success' or 'error'
                - 'image_path': Input image path
                - 'output_dir': Directory containing OBJ/MTL files
                - 'obj_file': Path to OBJ file
                - 'mtl_file': Path to MTL file
                - 'metadata': Dict with step-by-step info
                - 'error': Error message if failed
        
        Example:
            >>> pipeline = Pipeline3D()
            >>> result = pipeline.process_image("floor_plan.png")
            >>> print(result['obj_file'])  # Path to generated OBJ
        """
        image_path = Path(image_path)
        
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Setup output name
            if output_name is None:
                output_name = image_path.stem
            
            # Create output subdirectory
            task_dir = self.output_dir / output_name
            task_dir.mkdir(parents=True, exist_ok=True)
            
            metadata = {'steps': {}}
            
            # ===================== STEP 1: FPN INFERENCE =====================
            logger.info(f"\n[1/4] FPN Inference: {image_path.name}")
            step_start = 'fpn_inference'
            metadata['steps'][step_start] = {}
            
            mask, original_shape, classes_present = self.fpn.infer(image_path)
            
            # Save segmentation mask
            mask_path = task_dir / f"{output_name}_mask.png"
            self.fpn.save_mask(mask, mask_path)
            
            metadata['steps'][step_start].update({
                'mask_shape': list(mask.shape),
                'original_shape': list(original_shape),
                'classes_detected': classes_present,
                'mask_path': str(mask_path)
            })
            
            logger.info(f"  ✓ Mask generated: {mask_path.name}")
            logger.info(f"    Classes: {', '.join(classes_present.keys())}")
            
            # ===================== STEP 2: BOUNDARY EXTRACTION =====================
            logger.info(f"\n[2/4] Boundary Extraction")
            step_name = 'boundary_extraction'
            metadata['steps'][step_name] = {}
            
            # Load mask into extractor and process
            self.extractor.load_mask_array(mask)
            self.extractor.extract_room_polygons()
            self.extractor.detect_walls()
            self.extractor.calculate_scale_factor()
            
            # Create boundaries JSON
            boundaries_json = {
                'rooms': self.extractor.rooms,
                'walls': self.extractor.walls,
                'scale_factor': self.extractor.scale_factor
            }
            
            # Save boundaries
            boundaries_path = task_dir / f"{output_name}_boundaries.json"
            with open(boundaries_path, 'w') as f:
                json.dump(boundaries_json, f, indent=2)
            
            metadata['steps'][step_name].update({
                'num_rooms': len(boundaries_json.get('rooms', [])),
                'num_walls': len(boundaries_json.get('walls', [])),
                'scale_factor': boundaries_json.get('scale_factor'),
                'boundaries_path': str(boundaries_path)
            })
            
            logger.info(f"  ✓ Boundaries extracted")
            logger.info(f"    Rooms: {metadata['steps'][step_name]['num_rooms']}")
            logger.info(f"    Walls: {metadata['steps'][step_name]['num_walls']}")
            
            # ===================== STEP 3: WALL EXTRUSION =====================
            logger.info(f"\n[3/4] Wall Extrusion (v2)")
            step_name = 'wall_extrusion'
            metadata['steps'][step_name] = {}
            
            # Load and extrude
            self.extrusion.load_boundaries_json(boundaries_json)
            self.extrusion.extrude_all_rooms()
            geometry = self.extrusion.get_geometry()
            
            metadata['steps'][step_name].update({
                'num_vertices': len(geometry['vertices']),
                'num_faces': len(geometry['faces']),
                'room_height': 2.5,  # meters
                'wall_thickness': 'removed (v2 simplified)'
            })
            
            logger.info(f"  ✓ Walls extruded")
            logger.info(f"    Vertices: {metadata['steps'][step_name]['num_vertices']}")
            logger.info(f"    Faces: {metadata['steps'][step_name]['num_faces']}")
            
            # Save geometry JSON
            if save_intermediate:
                geometry_path = task_dir / f"{output_name}_geometry.json"
                # Convert numpy types to Python native types for JSON serialization
                geometry_for_json = {
                    'vertices': [[float(x) for x in v] for v in geometry['vertices']],
                    'faces': [[int(x) for x in f] for f in geometry['faces']],
                    'normals': [[float(x) for x in n] for n in geometry['normals']],
                    'colors': [[float(x) for x in c] for c in geometry['colors']],
                    'room_mapping': {str(k): [int(v) for v in vs] for k, vs in geometry['room_mapping'].items()}
                }
                with open(geometry_path, 'w') as f:
                    json.dump(geometry_for_json, f, indent=2)
                
                metadata['steps'][step_name]['geometry_path'] = str(geometry_path)
            
            # ===================== STEP 4: OBJ EXPORT =====================
            logger.info(f"\n[4/4] OBJ Export")
            step_name = 'obj_export'
            metadata['steps'][step_name] = {}
            
            # Export
            self.exporter.load_geometry(geometry)
            self.exporter.create_materials_from_colors()
            
            obj_path = task_dir / f"{output_name}.obj"
            mtl_path = task_dir / f"{output_name}.mtl"
            
            self.exporter.export_obj(str(obj_path), mtl_filename=f"{output_name}.mtl")
            self.exporter.export_mtl(str(mtl_path))
            
            metadata['steps'][step_name].update({
                'obj_file': str(obj_path),
                'mtl_file': str(mtl_path),
                'obj_size_kb': obj_path.stat().st_size / 1024,
                'mtl_size_kb': mtl_path.stat().st_size / 1024,
                'num_materials': len(self.exporter.materials)
            })
            
            logger.info(f"  ✓ Files exported")
            logger.info(f"    OBJ: {obj_path.name} ({metadata['steps'][step_name]['obj_size_kb']:.1f} KB)")
            logger.info(f"    MTL: {mtl_path.name} ({metadata['steps'][step_name]['mtl_size_kb']:.1f} KB)")
            
            # ===================== SUMMARY =====================
            logger.info(f"\n✓ PIPELINE COMPLETE")
            logger.info(f"  Output: {task_dir}/")
            logger.info(f"  Ready for 3D viewer: {obj_path.name}")
            
            # Save metadata
            metadata_path = task_dir / f"{output_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                'status': 'success',
                'image_path': str(image_path),
                'output_dir': str(task_dir),
                'obj_file': str(obj_path),
                'mtl_file': str(mtl_path),
                'metadata': metadata,
                'geometry': geometry
            }
        
        except Exception as e:
            logger.error(f"✗ Pipeline failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'image_path': str(image_path),
                'error': str(e),
                'metadata': metadata if 'metadata' in locals() else {}
            }
    
    def process_batch(
        self,
        image_dir: str,
        output_prefix: str = "batch",
        save_intermediate: bool = False
    ) -> list:
        """
        Process multiple floor plan images.
        
        Args:
            image_dir (str): Directory containing floor plan images
            output_prefix (str): Prefix for output subdirectories
            save_intermediate (bool): Save intermediate outputs
        
        Returns:
            results (list): List of result dicts from process_image()
        """
        image_dir = Path(image_dir)
        image_paths = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
        
        logger.info(f"\nProcessing {len(image_paths)} images from {image_dir}")
        
        results = []
        for idx, image_path in enumerate(image_paths, 1):
            logger.info(f"\n--- Image {idx}/{len(image_paths)} ---")
            
            output_name = f"{output_prefix}_{idx:03d}"
            result = self.process_image(
                image_path,
                output_name=output_name,
                save_intermediate=save_intermediate
            )
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"\n=== BATCH SUMMARY ===")
        logger.info(f"Total: {len(results)} | Success: {successful} | Failed: {len(results) - successful}")
        
        return results


def main():
    """Test the pipeline on sample images."""
    # Initialize pipeline
    pipeline = Pipeline3D(
        fpn_model_path="../U-NET/scripts/model_output/fpn_62images/best_model.pth",
        output_dir="pipeline_output"
    )
    
    # Process test images
    test_image_dir = Path("../U-NET/data/floorplan/")
    
    if not test_image_dir.exists():
        logger.error(f"Test image directory not found: {test_image_dir}")
        logger.info("Update test_image_dir in main() to correct location")
        return
    
    # Get first test image
    test_images = list(test_image_dir.glob('*.png'))[:1]  # Process just 1 for testing
    
    if not test_images:
        logger.error(f"No PNG images found in {test_image_dir}")
        return
    
    # Process each test image
    logger.info(f"Processing {len(test_images)} test image(s)...")
    for test_image in test_images:
        result = pipeline.process_image(test_image)
        
        if result['status'] == 'success':
            print(f"\n✓ OBJ file: {result['obj_file']}")
            print(f"✓ MTL file: {result['mtl_file']}")
        else:
            print(f"\n✗ Error: {result['error']}")


if __name__ == "__main__":
    main()
