# Archive - Experiment 05 Artifacts

This directory contains experimental files from Experiment 05 (Dataset Size Impact Study).

## Archived on: December 16, 2025

### Why These Files Were Archived

Experiment 05 tested whether adding 42 newly annotated images (104 total) would improve FPN performance over the 62-image baseline. Results showed the opposite: the 62-image model outperformed both 104-image variants.

**Key Finding:** Quality > Quantity
- 62 images: 53.5% mean IoU ✅ **PRODUCTION MODEL**
- 104 complete: 47.8% mean IoU

### Archived Contents

#### `data/` - Experimental Datasets
- `floorplan_104/` - First 104-image consolidation (incomplete annotations)
- `floorplan_104_final/` - Second 104-image consolidation (complete annotations)
- `floorplan_masks_104_13classes_final/` - Masks for 104 images
- `floorplan_new_42/` - New 42 images after re-annotation
- `floorplan_masks/` - Old U-Net masks
- `floorplan_masks_viz/` - Old mask visualizations

#### `scripts/` - Consolidation Scripts
- `simple_consolidate_104.py` - First merge attempt
- `final_consolidate_104.py` - Final merge with complete annotations
- `consolidate_104_dataset.py` - Alternative consolidation approach
- `merge_and_prepare_104_images.py` - Additional merge script

#### `model_output/` - Experimental Models
- `fpn_104images_final/` - FPN trained on 104 complete images (47.8% mIoU)
- `30-images-experiment/` - Early U-Net experiment
- `60-images-experiment/` - Early U-Net experiment
- `unet_best_model.pth` - Original U-Net baseline (27.6% mIoU)
- `unet_history.json` - U-Net training history
- `predictions.png` - U-Net predictions
- `training_curves.png` - U-Net training curves

### Production Files (Active)

**Model:** `U-NET/scripts/model_output/fpn_62images/best_model.pth`
- Architecture: FPN with ResNet34 encoder
- Performance: 53.5% mean IoU, 38.6% val IoU
- Dataset: 62 images, 13 classes

**Data:** 
- `U-NET/data/floorplan/` (62 images)
- `U-NET/data/floorplan_masks_13classes/` (62 masks)

### Reference

See `U-NET/experiment_05_dataset_size.md` for full analysis and findings.
