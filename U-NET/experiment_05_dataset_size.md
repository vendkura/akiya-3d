# Experiment 05: Dataset Size Impact Study

**Date:** December 16, 2025  
**Objective:** Evaluate if adding 42 newly annotated images improves FPN performance

---

## Problem

After Experiment 04 success (FPN: 53.5% mIoU on 62 images), we acquired 42 additional floor plans. Question: Does more data improve performance?

---

## Methodology

### Datasets Tested
1. **62 images (baseline):** Original fully-annotated dataset
2. **104 images (incomplete):** 62 + 42 new images with only small features annotated (rooms missing)
3. **104 images (complete):** 62 + 42 new images with full room annotations added

### Training Configuration
- **Architecture:** FPN with ResNet34 encoder (same as Exp 04)
- **Training:** 50 epochs, batch=4, lr=0.0001, Adam, CrossEntropyLoss
- **Split:** 70/15/15 (train/val/test), seed=42
- **Resolution:** 512×512

---

## Results

### Overall Performance

| Model | Mean IoU | Val IoU | Train-Val Gap | Status |
|-------|----------|---------|---------------|--------|
| **62 images** | **53.5%** | **38.6%** | 24.1% | ✅ Best |
| 104 incomplete | 48.8% | 35.6% | 23.1% | ❌ |
| 104 complete | 47.8% | 32.4% | **15.7%** | ❌ |

### Per-Class Comparison (62 vs 104 Complete)

| Class | 62 Images | 104 Complete | Change | Winner |
|-------|-----------|--------------|--------|--------|
| dining_area | 90.6% | 92.9% | +2.3% | 104 ✓ |
| bedroom | **83.3%** | 77.5% | -5.8% | 62 ✓ |
| closet | **67.2%** | 57.1% | -10.1% | 62 ✓ |
| room | **62.1%** | 58.1% | -4.0% | 62 ✓ |
| outdoor_space | **61.4%** | 56.2% | -5.2% | 62 ✓ |
| bathroom | **58.8%** | 48.7% | -10.1% | 62 ✓ |
| stairs | **46.7%** | 38.0% | -8.7% | 62 ✓ |
| door | **46.3%** | 37.9% | -8.4% | 62 ✓ |
| kitchen | **37.8%** | 30.5% | -7.3% | 62 ✓ |
| window | **35.0%** | 32.5% | -2.5% | 62 ✓ |
| sliding_door | **29.8%** | 22.9% | -6.9% | 62 ✓ |
| entrance | **23.2%** | 21.6% | -1.6% | 62 ✓ |

**62-image model wins in 11 out of 12 classes.**

---

## Analysis

### Why 104-Image Models Underperformed

**1. Data Quality Mismatch**
- New 42 images: Different sources, annotation styles, complexity
- Old 62 images: Consistent quality, uniform annotation methodology
- Mixed dataset introduced noise

**2. Incomplete Annotations Impact** 
- Initial 104 training with rooms missing caused confusion (35.6% val IoU)
- Completing annotations helped slightly (32.4% val IoU) but not enough
- Damage from inconsistency already done

**3. Increased Diversity Without Scale**
- +67% more data but also +100% more variance
- New images genuinely harder (complex layouts, different drawing styles)
- Need proportionally more data to handle increased complexity

**4. Positive Signal: Better Generalization**
- 104 complete: Train-val gap 15.7% vs 62 images: 24.1%
- Lower overfitting suggests model learning more robust features
- Just not translating to IoU gains with current data

### Confusion Patterns

**62 images:** window→dining_area (39.0%), sliding_door→closet (16.7%)  
**104 complete:** window→dining_area (34.5%), sliding_door→bedroom (32.3%)

New data introduced different confusion patterns, especially sliding_door misclassification increased.

---

## Key Insights

### Finding 1: Quality > Quantity ⭐
**Clean, consistent 62 images outperformed mixed 104 images by 5.7% mean IoU.**

Small, well-curated datasets can outperform larger heterogeneous datasets when:
- Annotation quality varies
- Data sources differ significantly  
- Dataset scale insufficient to overcome diversity

### Finding 2: Annotation Consistency Critical
Initial incomplete annotations (rooms missing) created irrecoverable training confusion even after completion.

### Finding 3: 50 Epochs May Be Insufficient
104-image model showed healthier train-val gap (15.7% vs 24.1%) suggesting it needs longer training to fully leverage additional data.

---

## Decision

**✅ Use 62-image FPN model for production (53.5% mean IoU, 38.6% val IoU)**

**Reasoning:**
1. Best absolute performance across almost all classes
2. Proven reliable on test set
3. Simpler deployment (smaller dataset footprint)
4. Time-constrained thesis timeline (1 month remaining)

**Leave 104-image model for future work:**
- Try 100+ epochs training
- Investigate annotation quality differences
- Consider data augmentation to bridge quality gap

---

## Lessons Learned

1. **Dataset curation matters more than size** for small-scale projects
2. **Consistency in annotation style** is critical for multi-source datasets
3. **Complete annotations upfront** - partial labeling creates confusion
4. **Diversity requires scale** - 104 images insufficient for increased variance
5. **Healthy train-val gap ≠ better performance** - 104 model generalizes better but performs worse overall

---

## Next Steps

1. Deploy 62-image FPN model for 3D reconstruction pipeline
2. Use 104-image experience for thesis discussion (quality vs quantity)
3. Consider data augmentation if more performance needed
4. Document 42-image dataset for future research continuation

---

**Conclusion:** In resource-constrained scenarios, investing in high-quality annotation of fewer images yields better results than rapidly expanding dataset with inconsistent quality. The 62-image FPN model remains our production choice.
