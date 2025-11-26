# Experiment 02: 62-Image Baseline - Per-Class Analysis

**Date:** November 26, 2025  
**Dataset:** 62 annotated Japanese floorplans  
**Model:** U-Net + ResNet34 encoder

---

## 📊 Overall Performance

| Metric | Value |
|--------|-------|
| **Overall Mean IoU** | **23.85%** |
| Classes Evaluated | 16/16 |
| Best Performing Class | DK (90.41%) |
| Worst Performing Class | Multiple at 0% |

---

## 🎯 Per-Class Performance

| Rank | Class | IoU | Samples | Status | Notes |
|------|-------|-----|---------|--------|-------|
| 1 🥇 | DK | **90.41%** | 62 | ✅ Excellent | Best class |
| 2 🥈 | bedroom | **81.07%** | 62 | ✅ Very Good | Strong performance |
| 3 🥉 | room | **54.13%** | 62 | ✅ Good | Generic catch-all |
| 4 | closet | **53.78%** | 62 | ✅ Good | Acceptable |
| 5 | stairs | **42.97%** | 61 | 🟡 Moderate | Borderline |
| 6 | washroom | **31.20%** | 46 | 🟡 Moderate | Needs work |
| 7 | outdoor space | **16.17%** | 61 | ❌ Poor | Low samples |
| 8 | LDK | **6.73%** | 40 | ❌ Poor | Despite 40 samples! |
| 9 | sliding door | **4.86%** | 59 | ❌ Poor | Small features |
| 10 | windows | **0.10%** | 57 | ❌ Critical | Failed |
| 11 | door | **0.02%** | 54 | ❌ Critical | Failed |
| 12 | toilet | **0.09%** | 44 | ❌ Critical | Failed |
| 13 | dinning room | **0.00%** | 9 | ❌ Critical | Too few samples |
| 14 | entrance | **0.00%** | 34 | ❌ Critical | Failed |
| 15 | kitchen | **0.00%** | 22 | ❌ Critical | Too few samples |
| 16 | living room | **0.00%** | 3 | ❌ Critical | Too few samples |

**Legend:**  
✅ Good (IoU > 50%) | 🟡 Moderate (30-50%) | ❌ Poor/Critical (< 30%)

---

## 🔄 Top Class Confusions

| True Class | Predicted As | Confusion Rate | Issue |
|------------|--------------|----------------|-------|
| living room | **room** | 95.9% | Generic class dominates |
| toilet | **room** | 62.7% | Specific → Generic |
| door | **room** | 55.9% | Features missed |
| dinning room | **DK** | 50.7% | Similar dining spaces |
| kitchen | **bedroom** | 50.7% | Misclassification |
| entrance | **closet** | 46.5% | Spatial confusion |
| windows | **DK** | 45.8% | Features missed |
| sliding door | **bedroom** | 41.2% | Features missed |

**Key Pattern:** Model heavily biased toward predicting generic **"room"** class.

---

## 📉 Data Distribution Issues

### Classes with Insufficient Samples (<30)

| Class | Current Samples | Status | Target |
|-------|----------------|--------|--------|
| **living room** | 3 | 🔴 Critical | +27 samples |
| **dinning room** | 9 | 🔴 Critical | +21 samples |
| **kitchen** | 22 | 🔴 Critical | +8 samples |

### Classes with Poor Performance Despite Data

| Class | Samples | IoU | Issue |
|-------|---------|-----|-------|
| **LDK** | 40 | 6.73% | Confused with DK/dining |
| **door** | 54 | 0.02% | Small feature detection |
| **windows** | 57 | 0.10% | Small feature detection |

---

## 🔍 Key Findings

### ✅ What Works
- **Large room spaces** (DK, bedroom, room, closet) → IoU 54-90%
- **Well-represented classes** (62 samples) → Better performance
- **Model architecture is capable** → Proven by DK (90%) and bedroom (81%)

### ❌ What Doesn't Work
- **Small architectural features** (doors, windows) → IoU < 0.1%
- **Underrepresented classes** (<10 samples) → 0% IoU
- **Overlapping/ambiguous categories** → High confusion (LDK vs DK)
- **Generic "room" bias** → Dominates predictions for specific types

### 🎯 Root Causes
1. **Data imbalance** - 10/16 classes have <50 samples or poor representation
2. **Class taxonomy issues** - Too many overlapping categories (DK/LDK/dining)
3. **Small feature scale** - U-Net struggles with thin walls, doors, windows
4. **Ambiguous labeling** - "room" vs specific room types unclear

---

## 📋 Action Items

### Priority 1: Data Collection (Immediate)
- [ ] **living room**: Collect +27 samples (target: 30 total)
- [ ] **dinning room**: Collect +21 samples (target: 30 total)  
- [ ] **kitchen**: Collect +8 samples (target: 30 total)
- [ ] **entrance**: Review annotations + collect +16 samples (target: 50 total)
- [ ] **toilet**: Review annotations + collect +16 samples (target: 60 total)

**Estimated effort:** 88 new annotations needed

### Priority 2: Class Consolidation (Consider)
Merge similar/confused classes to reduce ambiguity:
- **Option A:** Merge `living room` → `room` (already 96% confused)
- **Option B:** Merge `DK` + `LDK` + `dinning room` → `dining area`
- **Option C:** Merge `toilet` + `washroom` → `bathroom`

**Trade-off:** Fewer classes = better performance, but less granular output

### Priority 3: Annotation Review
- [ ] Review "room" vs specific types (bedroom, living room) - ensure consistency
- [ ] Clarify DK vs LDK vs dinning room definitions
- [ ] Check entrance vs closet annotations (46.5% confusion)

### Priority 4: Technical Improvements (After data fixes)
- Implement **focal loss** for class imbalance
- Add **edge-preserving augmentations** for doors/windows
- Try **multi-scale feature fusion** for small objects
- Consider **separate detection head** for architectural elements

---

## 📈 Expected Impact After Fixes

### With Data Collection Only
- Estimated improvement: **+10-15%** overall IoU
- Classes likely to improve: living room, kitchen, dining room, entrance

### With Data Collection + Class Merging
- Estimated improvement: **+15-25%** overall IoU
- Reduced confusion between similar categories
- More stable training

### With All Improvements
- Target overall IoU: **50-60%**
- Most classes above 50% threshold
- Architectural features still challenging (need specialized approach)

---

## 📁 Files & Artifacts

| File | Location | Description |
|------|----------|-------------|
| Analysis script | `scripts/per_class_analysis.py` | Automated per-class IoU analysis |
| IoU chart | `model_output/per_class_analysis/iou_per_class.png` | Bar chart visualization |
| Confusion matrix | `model_output/per_class_analysis/confusion_matrix.png` | Heatmap of class confusions |
| Class examples | `model_output/per_class_analysis/class_examples/` | Best/worst visual examples per class |
| Full report | `model_output/per_class_analysis/class_statistics.txt` | Detailed text statistics |

---

## 💡 Conclusion

**The model architecture is capable (proven by 90% IoU on DK), but data quality and quantity are the primary bottlenecks.**

**Recommended Path:**
1. Focus on collecting **88 new samples** for underrepresented classes
2. Review and fix annotation inconsistencies
3. Consider class merging to reduce confusion
4. Retrain with balanced dataset
5. Only then pursue architectural improvements

**Timeline Estimate:**
- Data collection: 2-3 weeks (3-4 annotations/day)
- Retraining: 2-3 days
- Evaluation: 1 day

---

## 📊 Visual Results

See folder: `model_output/per_class_analysis/class_examples/`

**Best Performers:**
- `00_DK_examples.png` - Excellent boundary detection
- `02_bedroom_examples.png` - Strong room segmentation

**Worst Performers:**
- `05_door_examples.png` - Complete detection failure
- `13_toilet_examples.png` - Heavy confusion with "room"
- `08_living_room_examples.png` - Only 3 samples, all misclassified
