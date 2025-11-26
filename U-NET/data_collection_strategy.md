# Data Collection Strategy for Underrepresented Classes

**Date:** November 26, 2025  
**Context:** Addressing data scarcity for small architectural features (doors, windows, entrances, etc.)

---

## 🎯 Problem Statement

Based on per-class analysis of 62-image baseline:
- **10/16 classes** have poor performance (IoU < 30%)
- **Critical gaps**: doors (0.02%), windows (0.10%), entrances (0%), toilet (0.09%), kitchen (0%)
- **Root cause**: Insufficient training samples + difficulty finding complete akiya floorplans

---

## 💡 Proposed Solution: Targeted Crop Collection

Instead of requiring **88 complete floorplans**, collect **targeted crops** containing specific elements with spatial context.

### Key Principle
**Collect focused regions that include the target element + surrounding context, not isolated features.**

---

## ✅ Collection Guidelines

### **Rule 1: Maintain Spatial Context**

**DO ✅**
- Include target element (door/window/entrance) **with surrounding walls**
- Show adjacent room spaces (even if partial)
- Preserve floor boundaries and spatial relationships
- Minimum crop: 300x300px, recommended: 512x512px

**DON'T ❌**
- Isolated door/window symbols on white background
- Heavily zoomed-in detail without context
- Features without adjacent room information

### **Rule 2: Source Diversity**

**Acceptable Sources:**
- ✅ Sections cropped from larger floorplans
- ✅ Generic Japanese residential plans (not limited to akiya)
- ✅ Real estate listing floorplans
- ✅ Architectural drawing databases
- ✅ Online floorplan repositories

**Key insight:** The model learns spatial patterns, not akiya-specific styles. Any Japanese residential floorplan is valid.

### **Rule 3: Data Mix Ratio**

| Approach | Target % | Purpose |
|----------|----------|---------|
| **Full floorplans** | 60-70% | Global context, room relationships |
| **Targeted crops** | 30-40% | Boost underrepresented classes |

---

## 📋 Collection Targets

### Priority Classes for Targeted Collection

| Class | Current Samples | Target | Need | Strategy |
|-------|----------------|--------|------|----------|
| **door** | 54 | 80+ | +26 | Crop door regions with hallway/room context |
| **windows** | 57 | 80+ | +23 | Crop window sections with room + outdoor space |
| **entrance** | 34 | 60+ | +26 | Crop entrance areas with outdoor connection |
| **toilet** | 44 | 60+ | +16 | Crop toilet rooms (often small, easy to isolate) |
| **kitchen** | 22 | 50+ | +28 | Crop kitchen sections with adjacent dining/LDK |
| **living room** | 3 | 30+ | +27 | Full or large crops showing living spaces |
| **dinning room** | 9 | 30+ | +21 | Crop dining areas with table/space indicators |

**Total needed:** ~167 targeted samples (much faster than 88 full floorplans!)

---

## 🔧 Annotation Workflow

### Step 1: Source Collection
1. Find Japanese residential floorplans (any source)
2. Identify plans with target elements clearly marked
3. Can use same source image multiple times for different crops

### Step 2: Strategic Cropping
```
For a floorplan with 3 doors:
  ├── Crop 1: Door + hallway + adjacent bedroom (512x512)
  ├── Crop 2: Door + entrance + outdoor space (512x512)
  └── Crop 3: Door + bathroom + hallway (512x512)

Result: 3 training samples from 1 source image
```

### Step 3: Annotation
- Annotate only the cropped region
- Maintain same 16-class taxonomy
- Keep consistent labeling standards
- Use 512x512 resolution for consistency with existing dataset

### Step 4: Integration
- Add to existing dataset structure
- Treat as regular training samples
- No special preprocessing needed

---

## 📊 Expected Impact

### With Targeted Collection (60/40 mix)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Overall IoU** | 23.85% | ~35-40% | +11-16% |
| **door IoU** | 0.02% | ~20-30% | Significant |
| **windows IoU** | 0.10% | ~15-25% | Significant |
| **entrance IoU** | 0.00% | ~25-35% | Critical fix |
| **toilet IoU** | 0.09% | ~30-40% | Moderate fix |
| **kitchen IoU** | 0.00% | ~35-45% | Critical fix |

---

## 🛠️ Implementation Example

### Example: Collecting 26 Door Samples

**Efficient approach:**
1. Find 10 diverse Japanese residential floorplans
2. Identify 2-3 doors per plan with good context
3. Crop 300x300 or 512x512 regions around each door
4. Annotate: door + walls + adjacent rooms
5. Result: 20-30 door samples from 10 source images

**Time estimate:** 2-3 doors/hour → ~10 hours for 26 samples  
**vs. finding 26 complete akiya plans:** Weeks of searching

---

## ⚠️ Quality Checklist

Before adding a crop to the dataset, verify:

- [ ] **Minimum size:** 300x300px (512x512 recommended)
- [ ] **Context included:** Target element + walls + adjacent spaces
- [ ] **Clear labels:** All elements in crop are annotatable
- [ ] **Spatial logic:** Realistic room layout (door between rooms, window on exterior wall)
- [ ] **Consistent style:** Japanese residential floorplan conventions
- [ ] **Annotation complete:** All visible elements labeled (not just target class)

---

## 🎯 Collection Priority Order

### Phase 1: Critical Classes (Week 1-2)
Focus on complete failures first:
1. **kitchen** (0% → need +28 samples)
2. **entrance** (0% → need +26 samples)
3. **living room** (0% → need +27 samples)
4. **dinning room** (0% → need +21 samples)

### Phase 2: Small Features (Week 3)
Address architectural elements:
5. **door** (0.02% → need +26 samples)
6. **windows** (0.10% → need +23 samples)

### Phase 3: Moderate Performers (Week 4)
Boost borderline classes:
7. **toilet** (0.09% → need +16 samples)
8. **outdoor space** (16.17% → optional, collect opportunistically)

---

## 📁 Dataset Organization

### Recommended Structure
```
U-NET/data/
├── floorplan/                    # Full images (keep existing)
│   ├── [existing 62 images]
│   └── [new crops: prefix with 'crop_']
│
├── floorplan_masks/              # Annotations
│   ├── [existing 62 masks]
│   └── [new crop masks: crop_*_mask.png]
│
└── metadata.json                 # NEW: Track source info
    └── {
          "crop_door_001.png": {
            "source": "real_estate_site_XYZ",
            "type": "crop",
            "target_classes": ["door", "hallway", "bedroom"]
          }
        }
```

### Naming Convention
- Full floorplans: Original names
- Targeted crops: `crop_[class]_[number].png`
  - Example: `crop_door_001.png`, `crop_window_015.png`

---

## 🚀 Quick Start Guide

### Immediate Actions

1. **Find sources** (30 minutes)
   - Bookmark 5-10 Japanese real estate sites
   - Search for "間取り図" (madori-zu = floorplan)
   - Look for downloadable/screenshot-able plans

2. **Start with easiest targets** (Day 1)
   - Kitchen/dining areas (usually large, easy to crop)
   - Entrances (distinctive, usually near edges)

3. **Batch annotate** (Days 2-3)
   - Collect 10 crops → annotate together
   - Maintain consistency in labeling
   - Use Label Studio or existing tool

4. **Test incrementally** (Week 2)
   - Add 20-30 new samples
   - Retrain model
   - Check if target classes improve
   - Adjust strategy if needed

---

## 📈 Success Metrics

Track progress weekly:

| Week | New Samples | Target Classes | Expected IoU Gain |
|------|-------------|----------------|-------------------|
| 1 | +30 | kitchen, entrance, living room | +5-8% overall |
| 2 | +30 | dinning room, toilet, door | +3-5% overall |
| 3 | +30 | windows, door (continued) | +2-4% overall |
| 4 | +20 | balance remaining gaps | +1-2% overall |

**Goal:** 80-110 new targeted samples over 4 weeks → Overall IoU 35-40%

---

## 💡 Alternative Strategies (If Needed)

### Plan B: Class Consolidation
If collection proves too time-consuming, consider merging:
- `living room` → `room` (already 96% confused)
- `DK` + `LDK` + `dinning room` → `dining area`
- `toilet` + `washroom` → `bathroom`

**Trade-off:** Fewer classes, better performance, less granularity

### Plan C: Synthetic Augmentation
- Extract door/window patterns from existing annotations
- Programmatically paste into varied contexts
- Requires careful implementation to maintain realism

---

## 📚 Resources

### Useful Search Terms (Japanese)
- 間取り図 (madori-zu) - floorplan
- 一戸建て (ikkodate) - detached house
- 2LDK, 3LDK - room configurations
- 平面図 (heimen-zu) - floor plan

### Potential Sources
- Real estate sites: SUUMO, Homes.co.jp, Yahoo Real Estate
- Architecture databases: Architectural Institute of Japan
- Government housing resources
- University architectural archives

---

## ✅ Summary

**Key Takeaway:** Don't need 88 complete akiya floorplans. Collect 100-150 targeted crops with context from any Japanese residential plans.

**Advantages:**
- ✅ 3-5x faster than finding complete floorplans
- ✅ Directly addresses weak classes
- ✅ Practical given akiya scarcity
- ✅ Maintains model learning effectiveness

**Critical Success Factors:**
1. Always include spatial context (walls, adjacent rooms)
2. Mix 60% full plans + 40% targeted crops
3. Maintain annotation quality and consistency
4. Test incrementally (don't wait for all 100+ samples)

---

**Next Step:** Start with Phase 1 targets (kitchen, entrance, living room, dining room) → Collect 10 samples this week and evaluate approach.
