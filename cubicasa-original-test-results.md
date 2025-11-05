# CubiCasa5k Original Implementation - Test Results

**Branch:** `cubicasa-original-implementation`  
**Date:** November 6, 2025  
**Hardware:** NVIDIA RTX 3060 (6GB VRAM) | PyTorch 2.7.1 + CUDA 11.8

---

## 1. Test Overview

**Objective:** Evaluate CubiCasa5k pretrained model performance on akiya floorplan images.

**Implementation:** Original CubiCasa5k (PyTorch)  
**Repository:** https://github.com/CubiCasa/CubiCasa5k  
**Test Images:** 3 floorplan images (PNG)

---

## 2. Setup & Installation

**Installation Steps:**
```bash
git clone https://github.com/CubiCasa/CubiCasa5k.git
pip install lmdb Pillow svgpathtools matplotlib scipy scikit-image tqdm jupyter
# Download weights: model_best_val_loss_var.pkl from Google Drive
```

**Setup Time:** ~1 hour  
**Status:** ✅ Success

---

## 3. Model Information

| Property | Details |
|----------|---------|
| **Architecture** | Hourglass Network (hg_furukawa_original) |
| **Training Data** | CubiCasa5k dataset (5000 floorplans, Western architecture) |
| **Preprocessing** | Resize to 256x256, normalize (mean=0.5, std=0.5) |
| **Test-Time Augmentation** | 4 rotations (0°, 90°, 180°, 270°) with averaging |
| **Output Classes** | 12 room types, 11 icon types (doors/windows/fixtures) |
| **Inference Device** | GPU (CUDA) |

---

## 4. Test Results

### Image 1 (Traditional Japanese House - 1F)
**Inference Time:** 0.256 seconds

| Aspect | Quality | Notes |
|--------|---------|-------|
| **Room Detection** | ⚠️ Fair | Detected walls and room boundaries, but classified all as generic (Undefined/Storage/Entry). Failed to recognize tatami rooms, DK area as Kitchen/Living. |
| **Icon Detection** | ❌ Poor | Barely detected any doors or windows. Missing most door openings and all window locations. Very low confidence. |
| **Overall** | ❌ Poor | Struggles with Japanese architecture. 256x256 resolution may be too low. Better at structural walls than functional elements. |

### Image 2 (Traditional Japanese House with Garage - 1階)
**Inference Time:** 0.220 seconds

| Aspect | Quality | Notes |
|--------|---------|-------|
| **Room Detection** | ⚠️ Fair | ✅ Correctly identified garage area. Detected room boundaries well. ❌ Misclassified most rooms as Undefined/Storage/Entry. DK area not recognized. |
| **Icon Detection** | ❌ Poor | Extremely poor detection. Missing sliding doors (fusuma/shoji), garage door, bathroom fixtures. Almost no visible marks. |
| **Overall** | ⚠️ Fair | Slightly better than Image 1 due to garage detection. Still fundamentally unsuitable for Japanese layouts. |

### Image 3 (Complex Japanese House - 1階)
**Inference Time:** 0.404 seconds

| Aspect | Quality | Notes |
|--------|---------|-------|
| **Room Detection** | ⚠️ Fair | Good room boundary separation. Multiple distinct zones detected. ❌ All generic classifications. Tatami rooms not identified as bedrooms. |
| **Icon Detection** | ⚠️ Fair | Slightly better than previous images with some faint orange marks visible. ❌ Still missing most doors/windows. No bathroom fixture detection. |
| **Overall** | ⚠️ Fair | Best icon detection of the 3 images. Slower inference (2x) due to complex layout. Cleaner linear layout helped slightly but still unsuitable overall. |

---

## 5. Summary

| Metric | Value |
|--------|-------|
| **Success Rate** | 0/3 images properly analyzed |
| **Room Detection Quality** | ⚠️ Fair (structure detected, classifications wrong) |
| **Icon Detection Quality** | ❌ Poor (minimal detection) |
| **Avg Inference Time** | 0.293 seconds (~0.3s per image) |

### Strengths ✅
- **Fast inference:** ~0.3s per image on RTX 3060
- **Wall detection:** Successfully detects structural walls and room boundaries
- **GPU acceleration:** Efficient CUDA implementation
- **Easy setup:** Simple installation, pretrained weights available

### Weaknesses ❌
- **Wrong room classifications:** Cannot identify Japanese room types (tatami rooms, washitsu, genkan, DK areas)
- **Poor door/window detection:** Fails catastrophically on traditional sliding doors (fusuma/shoji)
- **Training data mismatch:** Trained on Western architecture (CubiCasa5k), doesn't generalize to Japanese layouts
- **Low resolution:** 256x256 input may lose important details in complex Japanese floorplans
- **Text interference:** Japanese text labels appear to confuse the model
- **No fixture detection:** Misses bathroom fixtures (toilet, sink, bathtub) shown in originals

---

## 6. Conclusion

**Suitability for Akiya:** ❌ **Not Suitable** (without fine-tuning)

**Recommendation:** While the CubiCasa5k model successfully detects structural elements like walls and room boundaries with fast inference (~0.3s), it is fundamentally unsuitable for Japanese akiya floorplans in its pretrained state. The model was trained on Western architecture and cannot recognize Japanese room types or traditional architectural elements (sliding doors, tatami layouts). **Requires fine-tuning on Japanese floorplan dataset (100+ labeled akiya images) to be viable for this thesis project.**

**Next Steps:**
1. Test alternative implementations (MMDetection-based, other architectures)
2. If no pretrained model works well, proceed with fine-tuning CubiCasa5k or similar on akiya dataset
3. Consider higher resolution inputs (512x512) for better detail preservation

---

**Repository:** https://github.com/CubiCasa/CubiCasa5k  
**Paper:** https://arxiv.org/abs/1904.01920
