# Experiment 03: CRF Post-Processing Evaluation

**Date:** November 20, 2025  
**Goal:** Evaluate if Conditional Random Fields (CRF) post-processing improves U-Net segmentation quality on Japanese floorplans

---

## 1. Dataset Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Total images | 62 | From previous experiments |
| Test images evaluated | 10 | Representative samples |
| Number of classes | 16 | Japanese-specific categories |
| Image dimensions | 512x512 | Resized during inference |

**Dataset notes:**
- Using ground truth masks from `floorplan_masks/` directory
- Masks have `_mask.png` suffix naming convention

---

## 2. Model Architecture

| Component | Specification |
|-----------|---------------|
| Base model | U-Net (from Experiment 02) |
| Encoder | ResNet34 |
| Checkpoint | best_model.pth |
| Post-processing | CRF with various parameter configurations |

**Post-processing approach:**
- Applied Dense CRF on U-Net softmax probabilities
- Tested 4 different parameter configurations
- Compared against raw U-Net predictions

---

## 3. CRF Configurations Tested

### Configuration 1: Original (Strong)
```python
sxy_gaussian=3, compat_gaussian=3
sxy_bilateral=80, srgb_bilateral=13, compat_bilateral=10
iterations=5
```

### Configuration 2: Light Smoothing
```python
sxy_gaussian=2, compat_gaussian=2
sxy_bilateral=5, srgb_bilateral=3, compat_bilateral=5
iterations=3
```

### Configuration 3: Minimal CRF
```python
sxy_gaussian=1, compat_gaussian=1
sxy_bilateral=3, srgb_bilateral=2, compat_bilateral=3
iterations=2
```

### Configuration 4: Edge Preserving
```python
sxy_gaussian=2, compat_gaussian=1
sxy_bilateral=4, srgb_bilateral=5, compat_bilateral=3
iterations=3
```

---

## 4. Results

### Visual Comparison

| Configuration | Boundary Quality | Detail Preservation | Overall Quality |
|---------------|------------------|---------------------|-----------------|
| U-Net Only (No CRF) | ✓ Sharp, clean | ✓ Excellent | ✓ Best |
| Original (Strong) | ✗ Over-smoothed | ✗ Lost details | ✗ Worst |
| Light Smoothing | ✗ Blurred edges | ✗ Reduced accuracy | ✗ Degraded |
| Minimal CRF | ✗ Slight blur | ~ Mostly preserved | ✗ Slightly worse |
| Edge Preserving | ✗ Artifacts | ✗ Room distortion | ✗ Poor |

### Key Observations
- **All CRF configurations degraded segmentation quality**
- CRF introduced artifacts and destroyed sharp architectural boundaries
- Room boundaries became blurred or disappeared
- Small features (doors, windows) were completely lost
- Background areas incorrectly filled with room predictions

---

## 5. Analysis

### What Didn't Work
- CRF smoothing assumptions are incompatible with floorplan geometry
- Architectural drawings have sharp, straight edges that CRF destroys
- High-contrast boundaries (walls) don't need smoothing refinement
- Even minimal CRF parameters caused degradation

### Why CRF Failed for Floorplans
- **Natural images vs Architectural drawings**: CRF designed for soft boundaries in photos
- **Edge characteristics**: Floorplans have binary edges (wall/no wall), not gradual transitions
- **Feature scale**: Sharp corners and straight lines are architectural features, not noise
- **U-Net already optimal**: Model predictions already match ground truth geometry

### Observations
- U-Net predictions are already high quality for this task
- No visible noise or artifacts that would benefit from smoothing
- Floorplan segmentation is fundamentally different from natural image segmentation

---

## 6. Conclusion & Recommendation

**Recommendation: Do NOT use CRF post-processing for Japanese floorplan segmentation**

- U-Net predictions should be used directly without post-processing
- CRF is counterproductive for architectural drawings
- Focus should be on improving model training rather than post-processing

---

## 7. Next Steps

1. Document U-Net architecture as final pipeline (no post-processing)
2. Focus on data augmentation strategies to improve edge cases
3. Investigate class-specific performance improvements
4. Consider alternative architectures if needed (DeepLabV3+, SegFormer)

---

## 8. Files & Artifacts

| File | Location | Description |
|------|----------|-------------|
| CRF script | `scripts/crf.py` | Main evaluation script |
| Parameter tuning | `scripts/crf_parameter_tuning.py` | Multi-config comparison |
| Results | `data/floorplan_crf_results/` | Visual comparisons |
| Tuning results | `data/crf_tuning_results/` | Parameter comparison grids |

---

## 9. Comparison to Previous Experiments

| Experiment | Approach | Result |
|------------|----------|--------|
| Exp 02 (62 images) | U-Net baseline | Good segmentation quality |
| **Exp 03 (CRF)** | U-Net + CRF post-processing | **Degraded quality - CRF rejected** |

**Key Finding:** Post-processing is unnecessary for floorplan segmentation. The U-Net model already produces optimal results for sharp-edged architectural features.

---

## Notes & Observations

- CRF parameters that work well for natural images (e.g., semantic segmentation on Cityscapes) are completely inappropriate for architectural drawings
- This experiment validates that task-specific characteristics (sharp vs. soft boundaries) significantly impact which post-processing techniques are applicable
- Saved ~5 inference iterations by skipping CRF (faster deployment)
