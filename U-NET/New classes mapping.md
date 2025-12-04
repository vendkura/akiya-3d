OLD_TO_NEW_MAPPING = {
    # Merger 1: Eliminate "living room" → absorb into "room"
    "living room": "room",
    
    # Merger 2: Consolidate dining spaces
    "dinning room": "dining_area",  # Note: they spelled it "dinning" 
    "DK": "dining_area",
    "LDK": "dining_area",
    
    # Merger 3: Bathroom consolidation
    "toilet": "bathroom",
    "washroom": "bathroom",
    
    # Keep as-is:
    "bedroom": "bedroom",
    "room": "room",
    "closet": "closet",
    "entrance": "entrance",
    "kitchen": "kitchen",
    "outdoor space": "outdoor_space",
    "stairs": "stairs",
    "sliding door": "sliding_door",
    "door": "door",
    "windows": "windows",
}
```

**New class count: 13 classes**

---

### Impact on Annotation Targets (REVISED)

| Old Plan | New Plan | Savings |
|----------|----------|---------|
| living room: +27 | ❌ Eliminated (merged to "room") | **-27 images** |
| dinning room: +21 | dining_area: +30 | Need examples of all 3 types |
| DK: already 62 | ✅ Sufficient | - |
| LDK: +20 | ✅ Included in +30 | - |
| toilet: +16 | bathroom: +20 | Need both toilet + washroom |
| washroom: already 46 | ✅ Sufficient | - |
| kitchen: +8 | kitchen: +8 | Same |
| entrance: +16 | entrance: +16 | Same |

**New total needed: ~81 images** (instead of 108)

**Revised timeline: 81 ÷ 6/day = 13.5 days ≈ 2 weeks** 🎉

---

## 🎯 ULTRA-REVISED REALISTIC PLAN (2 Weeks)

### Week 1 (Nov 27 - Dec 3): 42 images

**Priority targets:**
- Kitchen: 8 images
- Dining_area: 30 images (mix of DK, LDK, dining room examples)
  - DK examples: 10
  - LDK examples: 10  
  - Dining room examples: 10
- Entrance: 4 images (start)

**Daily:** 6 images/day × 7 days = 42 images

---

### Week 2 (Dec 4 - Dec 10): 39 images

**Priority targets:**
- Entrance: 12 more (total 16)
- Bathroom: 20 images (mix of toilet + washroom)
  - Toilet examples: 10
  - Washroom examples: 10
- Buffer: 7 images for any shortfalls

**Daily:** 6 images/day × 6.5 days = 39 images

**Total after 2 weeks: 62 + 42 + 39 = 143 images**

---

### Week 3 (Dec 11-13): Retrain & Evaluate

**3 days:**
- Day 1: Retrain with 143 images
- Day 2: Per-class analysis v2
- Day 3: Document results, compare to baseline

---

## 🌐 DATA SOURCES (Practical List)

### Primary Source (You Have Access)
1. **akiya2.com** - Japanese abandoned houses
   - Pro: Matches your domain exactly
   - Con: Limited inventory

### Secondary Sources (Free)

2. **SUUMO (suumo.jp)** 
   - 中古一戸建て (used detached houses)
   - Filter by 築年数古い順 (oldest first) to get traditional layouts
   - Many have floorplan PDFs

3. **Homes.co.jp**
   - Similar to SUUMO
   - Search: 古民家 (kominka - old folk houses)

4. **At Home (athome.co.jp)**
   - Good for regional listings
   - Filter by 間取り図あり (floorplan available)

5. **Real Estate Japan (realestate.co.jp/en)**
   - English interface
   - Many rural properties with plans

### Search Terms to Use
```
Japanese:
- 間取り図 (madori-zu) = floorplan
- 古民家 (kominka) = old folk house  
- 空き家 (akiya) = abandoned house
- 中古住宅 (chuuko juutaku) = used house
- 平面図 (heimen-zu) = floor plan diagram

English (for Real Estate Japan):
- "traditional house floorplan"
- "Japanese house layout"
- "rural property"