# Experiment 03: Class Consolidation (16 → 13 Classes)

**Date:** December 5, 2025  
**Dataset:** 62 annotated Japanese floorplan images  
**Objective:** Reduce class confusion by consolidating similar categories

---

## Motivation

Per-class analysis of the 16-class model revealed severe confusion between similar classes:
- Living room → room (96% confusion)
- Toilet/washroom confusion (63%)
- DK/LDK/dining room overlap
- Overall IoU: 23.85%

**Hypothesis:** Merging visually similar classes will reduce confusion and improve performance.

---

## Methodology

### Class Mapping (16 → 13)
```
OLD CLASSES (16)              NEW CLASSES (13)
─────────────────────────────────────────────
DK, LDK, dinning room    →    dining_area
toilet, washroom         →    bathroom
living room              →    room
[other 10 unchanged]
```

### Process
1. Remapped COCO annotations using `remap_classes.py` (2,135 annotations updated)
2. Regenerated mask images with 13 classes (pixel values 0-12)
3. Updated training configuration (NUM_CLASSES=13)
4. Trained for 50 epochs (same hyperparameters as 16-class baseline)

---

## Results

### Overall Performance
| Metric | 16 Classes | 13 Classes | Improvement |
|--------|-----------|-----------|-------------|
| **Val IoU (final)** | 19.3% | 24.7% | **+5.4%** |
| **Val IoU (peak)** | 19.7% | 26.0% | **+6.3%** |
| **Train-Val Gap** | 16.8% | 9.8% | **+7.0%** |
| **Convergence** | Epoch 19: 13% | Epoch 19: 20% | Faster |

### Per-Class Performance
| Class | IoU | Status | Notes |
|-------|-----|--------|-------|
| **dining_area** | 90.6% | ✅ Excellent | Massive improvement from consolidation |
| **bedroom** | 82.3% | ✅ Good | Clear visual patterns |
| **closet** | 54.4% | ⚠️ Moderate | Confused with small features |
| **room** | 53.4% | ⚠️ Moderate | Generic spaces |
| **bathroom** | 29.7% | ❌ Poor | Still struggling despite merge |
| **kitchen** | 18.5% | ❌ Poor | Needs more examples |
| **window** | 1.6% | ❌ Failed | 42% → dining_area |
| **door** | 0.0% | ❌ Failed | 71% → closet |
| **sliding_door** | 0.2% | ❌ Failed | 37% → bedroom |
| **entrance** | 0.04% | ❌ Failed | 48% → kitchen |

### Key Confusions
- **Small features → large rooms:** door/window/entrance misclassified as closet/bedroom/dining_area
- **Closet as confusion sink:** Absorbing 70% of outdoor_space, 71% of doors, 32% of stairs

---

## Analysis

### ✅ Successes
1. **Class consolidation highly effective:** dining_area jumped from 3.9% → 90.6% IoU
2. **Better generalization:** 7% smaller train-val gap indicates more robust features
3. **Stable training:** Smoother validation curve, consistent improvement
4. **Overall improvement:** +28% relative gain in validation IoU

### ❌ Limitations
1. **Small feature detection fails:** doors/windows (~10-20px) lost at 512×512 resolution
2. **Scale mismatch:** Single model struggles with features spanning 100× size difference
3. **Architectural specificity:** Japanese door types (sliding/hinged/entrance) need preservation

---

## Future Work

### 1. Multi-Scale Decoder Enhancement
Current U-Net uses single-scale prediction at 512×512. Proposed improvements:
- **FPN (Feature Pyramid Network):** Multi-scale feature fusion for better small object detection
- **DeepLabV3+ with ASPP:** Atrous spatial pyramid pooling for multi-scale context
- **Higher resolution training:** 1024×1024 to double pixel density of small features

### 2. Two-Stage Detection Pipeline for Japanese Houses
Separate room segmentation from opening detection:

**Stage 1 - Room Segmentation (U-Net):**
- 9 classes: dining_area, bathroom, bedroom, closet, room, kitchen, outdoor_space, stairs, balcony
- Remove door/window classes from segmentation

**Stage 2 - Opening Detection (MMDetection):**
- Object detection for: door, sliding_door, entrance, window
- Preserves Japanese architectural distinctions (sliding vs hinged)
- Better suited for thin line features (10-20px)
- Overlay bounding boxes onto room segmentation map

**Rationale:** Rooms (filled polygons) and openings (thin lines) require different detection strategies. Two-stage approach addresses scale mismatch and maintains Japanese house specificity.

---

## Conclusion

Class consolidation successfully improved model performance by **+28% relative gain** (19.3% → 24.7% IoU). The strategy of merging confused similar classes (DK/LDK/dining → dining_area) proved highly effective.

However, small feature detection remains a fundamental limitation of single-stage semantic segmentation. Next steps should focus on either multi-scale architectures or a two-stage pipeline separating room segmentation from opening detection.

**Key Takeaway:** Consolidation works for large features with visual similarity, but architectural elements at different scales require specialized approaches.
