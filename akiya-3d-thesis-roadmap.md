# Akiya 2D-to-3D Floorplan Conversion - Master's Thesis Roadmap

**Project Duration:** 2-3 months (Implementation) + 3 weeks (Writing)  
**Partner:** akiya2.com  
**Dataset:** ~100 labeled akiya floorplans  
**Goal:** Working prototype for thesis validation + potential business application

---

## Project Overview

### Objectives
- Convert 2D akiya floorplan images to 3D structural models
- Address Japanese traditional architecture challenges (tatami, fusuma, engawa)
- Evaluate transfer learning effectiveness on limited specialized dataset
- Create working demo for thesis defense and akiya2.com

### Key Constraints
- 2-3 month timeline for implementation
- 100 labeled training images
- Japanese architectural features require custom handling
- Master's thesis scope (proof-of-concept, not production)

---

## Week-by-Week Implementation Roadmap

### **WEEK 1: Get Something Working (Minimal Viable Pipeline)**

**Goal:** 2D floorplan image → 3D mesh file (even if results are poor)

#### Day 1-2: Setup + Baseline
```bash
# Install essentials
pip install torch torchvision opencv-python pillow trimesh gradio

# Project structure
mkdir akiya-3d-thesis
cd akiya-3d-thesis
git init
mkdir data models outputs notebooks
```

**Tasks:**
1. Set up development environment
2. Pick ONE pre-trained model (CubiCasa5k or Detectron2)
3. Run model on 5 of your akiya images
4. Document initial results (expect failures - that's OK!)

#### Day 3-4: Basic 3D Extrusion
```python
# Dead simple approach:
# Segmentation mask → 3D boxes
# If model detects a room, create a 3D box for it

import trimesh
import numpy as np

# Pseudo-code concept:
# 1. Get wall pixels from segmentation
# 2. Extrude walls to height (e.g., 2.4m)
# 3. Export to .obj file
# mesh.export('akiya_model.obj')
```

**Tasks:**
1. Implement basic segmentation → 3D conversion
2. Create simple extrusion logic (walls = vertical boxes)
3. Export to .obj format
4. View in Blender or online 3D viewer

#### Day 5: Test End-to-End
- Run complete pipeline: image → segmentation → 3D mesh
- Test on 5 different akiya images
- Document what works and what fails
- Take screenshots for thesis documentation

**Week 1 Deliverable:** ✅ Working (but crude) 2D→3D pipeline

---

### **WEEK 2: Make It Actually Work on Your Data**

**Goal:** Get decent results on at least 20 akiya images

#### Day 1-2: Data Preprocessing
```python
# Data organization and preparation
# - Resize all images to consistent size (e.g., 512x512)
# - Normalize pixel values
# - Create train/val/test splits (60/20/20)
# - Implement data augmentation (rotation, flip, brightness)

# Recommended split:
# - Training: 60 images
# - Validation: 20 images  
# - Test: 20 images
```

**Tasks:**
1. Audit your 100 labeled images
2. Categorize by complexity (simple/medium/complex)
3. Create clean data splits
4. Set up data loading pipeline

#### Day 3-5: Fine-tune Pre-trained Model
```python
# Transfer learning approach:
# 1. Load pre-trained weights (CubiCasa5k/RPLAN)
# 2. Freeze early layers
# 3. Fine-tune on your 60 training images
# 4. Focus on wall detection first (most critical)
# 5. Test on 20 validation images
# 6. Iterate quickly based on results
```

**Tasks:**
1. Set up training loop
2. Fine-tune on training set
3. Monitor validation metrics
4. Adjust hyperparameters as needed
5. Save best model checkpoints

#### Weekend: Evaluation
- Run full evaluation on validation set
- Count successful wall detections
- Identify failure patterns (fusuma, tatami, weird layouts)
- Create error analysis document

**Week 2 Deliverable:** ✅ Model that works reasonably on simple akiya floorplans

---

### **WEEK 3-4: Handle Akiya-Specific Features**

**Goal:** Deal with Japanese architecture quirks

#### Week 3 Focus: Pattern Recognition

**Tatami Detection:**
```python
# Detect tatami room patterns
# Standard tatami: 91cm × 182cm
# Look for grid patterns in room layout

def detect_tatami_pattern(room_mask):
    # Check for rectangular subdivisions
    # Ratio check: width/height ≈ 1:2
    # Grid alignment detection
    pass
```

**Fusuma Handling:**
```python
# Fusuma = sliding doors (thin parallel lines)
# Often misclassified as walls

def classify_fusuma(wall_segments):
    # Check wall thickness
    # Look for parallel line patterns
    # Distinguish from solid walls
    pass
```

**Tasks:**
1. Implement post-processing rules for tatami
2. Add fusuma detection logic
3. Handle engawa (perimeter corridors)
4. Test on medium complexity akiya images

#### Week 4 Focus: 3D Refinement

**Improved 3D Extrusion:**
```python
# Better 3D model generation:
# - Proper wall thickness
# - Door and window openings
# - Room height variations
# - Basic textures (optional)

def create_3d_model(segmentation_result):
    # 1. Extract wall polygons
    # 2. Create 3D walls with proper thickness (15-20cm)
    # 3. Add door/window openings
    # 4. Generate floor and ceiling
    # 5. Export to OBJ/glTF
    pass
```

**Tasks:**
1. Refine 3D extrusion algorithm
2. Add door and window geometry
3. Improve wall thickness rendering
4. Test on full test set (20 images)

**Week 3-4 Deliverable:** ✅ System that handles common akiya layouts with Japanese features

---

### **WEEK 5-6: Polish + Demo**

**Goal:** Make it presentable for thesis and akiya2.com

#### Week 5: User Interface

**Create Simple Web Demo:**
```python
# Use Gradio or Streamlit
import gradio as gr

def convert_floorplan(image):
    # 1. Run segmentation
    # 2. Generate 3D model
    # 3. Return 3D viewer + downloadable file
    return model_3d, obj_file

demo = gr.Interface(
    fn=convert_floorplan,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Model3D(), gr.File()]
)
demo.launch()
```

**Tasks:**
1. Create web interface (Streamlit or Gradio)
2. Allow image upload
3. Display 3D result in browser
4. Provide download link for .obj file
5. Add basic error handling

#### Week 6: Final Testing

**Comprehensive Evaluation:**
1. Run on all 20 test images
2. Calculate metrics:
   - Wall detection accuracy
   - Room segmentation IoU
   - Door/window detection rate
   - Successful 3D conversion rate
3. Generate comparison visualizations
4. Document failure cases with analysis

**Demo Preparation:**
1. Create demo video (2-3 minutes)
2. Prepare slides showing results
3. Write README with setup instructions
4. Clean up code and add documentation

**Week 5-6 Deliverable:** ✅ Working demo + comprehensive evaluation results

---

### **WEEK 7-8: Buffer + Experiments (Optional)**

**Goal:** Try improvements if time permits

#### Potential Experiments:
1. **Different Architectures**
   - Try alternative backbone networks
   - Compare U-Net vs. FPN vs. DeepLab

2. **Enhanced Post-Processing**
   - More sophisticated fusuma detection
   - Better room type classification
   - Automatic tatami layout optimization

3. **Ensemble Methods**
   - Combine multiple models
   - Voting or averaging strategies

4. **Ablation Studies**
   - Test impact of each component
   - What helps most? Transfer learning? Rules? Augmentation?

**Week 7-8 Deliverable:** ✅ Additional results for thesis discussion

---

## Practical Daily Workflow

### Every Day:
```
Morning:
1. Review yesterday's results
2. Plan today's specific tasks (3-4 concrete goals)
3. Code/experiment (4-6 hours)

Afternoon:
4. Test on 5-10 images
5. Document what worked/failed (30-60 min)
6. Git commit + push changes

Evening:
7. Update progress log
8. Prepare tomorrow's tasks
```

### Every Week:
```
Friday:
1. Update akiya2.com (15 min Slack/email)
2. Review week's accomplishments
3. Identify blockers
4. Plan next week's focus

Sunday:
5. Back up everything (code, models, results)
6. Organize files and documentation
7. Mental preparation for next week
```

---

## Start TODAY - First 3 Tasks

### Task 1: Environment Setup (1 hour)
```bash
# Create project
mkdir akiya-3d-thesis
cd akiya-3d-thesis
git init

# Install core libraries
pip install torch torchvision opencv-python pillow numpy trimesh gradio

# Create basic structure
mkdir data models outputs notebooks
touch README.md
```

### Task 2: Get Pre-trained Model (2 hours)
```python
# Option A: Use Detectron2 (Facebook's framework)
pip install detectron2

# Option B: Use MMDetection (OpenMMLab)
pip install mmdet mmcv

# Option C: Find CubiCasa5k weights
# Download from GitHub and test on 1 image

# Quick test to verify installation
import torch
import cv2
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Task 3: Simple Test (1 hour)
```python
# Run model on 1 akiya image
# Save result
# View output
# Screenshot everything

# You're now coding!
```

---

## Practical Decision Tree (When Stuck)

### Problem → Solution

**"Model not working on my data"**
- → Start with simpler model
- → Use more data augmentation
- → Reduce input resolution

**"Can't get pre-trained model running"**
- → Try different framework (Detectron2 → MMDet)
- → Use simpler baseline (ResNet + FCN)
- → Train from scratch (last resort)

**"Results are bad"**
- → That's OK for thesis! Document WHY
- → Failure analysis is valuable content
- → Focus on what CAN be learned

**"Running out of time"**
- → Cut scope: wall detection only
- → Simpler 3D: just boxes
- → Document limitations clearly

**"Don't know which model to use"**
- → Pick ANY segmentation model
- → Get it running in 1 day
- → Switch later if needed
- → Perfect is the enemy of done

**"Training is too slow"**
- → Use smaller images (256x256)
- → Reduce batch size
- → Use Google Colab Pro ($10/month)
- → Train overnight

---

## Technical Stack Recommendations

### Core ML Frameworks:
```python
# Primary: PyTorch (most flexible)
pip install torch torchvision

# Segmentation: Detectron2 or MMDetection
pip install detectron2
# OR
pip install mmdet mmcv

# Computer Vision
pip install opencv-python pillow albumentations
```

### 3D Processing:
```python
# 3D mesh generation
pip install trimesh pyglet

# Alternative: Open3D
pip install open3d
```

### Visualization & Demo:
```python
# Web interface
pip install gradio streamlit

# Plotting
pip install matplotlib seaborn
```

### Development Tools:
```python
# Jupyter for experiments
pip install jupyter notebook

# Progress tracking
pip install tqdm wandb

# Code quality
pip install black flake8
```

---

## Recommended Architecture

### Pipeline Overview:
```
Input: 2D Floorplan Image (PNG/JPG)
    ↓
[1] Image Preprocessing
    - Resize to 512x512
    - Normalize
    - Augmentation (training only)
    ↓
[2] Semantic Segmentation Model
    - Pre-trained backbone (ResNet50/101)
    - Feature Pyramid Network (FPN)
    - Output: Segmentation masks
        • Walls
        • Doors
        • Windows
        • Rooms
    ↓
[3] Japanese Architecture Post-Processing
    - Detect tatami patterns (91x182cm grid)
    - Identify fusuma (thin parallel lines)
    - Handle engawa (perimeter corridors)
    - Room type classification
    ↓
[4] 3D Extrusion Module
    - Convert 2D polygons → 3D meshes
    - Add wall thickness (15-20cm)
    - Create door/window openings
    - Standard heights (2.4m rooms)
    ↓
[5] 3D Model Export
    - Format: OBJ, glTF, or STL
    - Include basic materials/textures
    ↓
Output: 3D Model File + Visualization
```

### Model Architecture Recommendation:
```python
# Transfer Learning + Rule-Based Hybrid

Base Model:
- Backbone: ResNet50 (pre-trained on ImageNet)
- Neck: Feature Pyramid Network (FPN)
- Head: Semantic Segmentation (4 classes: wall, door, window, background)

Fine-tuning:
- Freeze: First 2 layers of ResNet
- Train: FPN + Segmentation head
- Data: 60 labeled akiya images
- Augmentation: Heavy (rotation, flip, brightness, contrast)

Post-Processing:
- Rule-based Japanese architecture detection
- Morphological operations
- Polygon simplification
```

---

## Success Criteria

### Minimum Success (To Graduate):
- ✅ Working pipeline: image in → 3D mesh out
- ✅ Tested on 20 test images
- ✅ 50%+ success rate on simple layouts
- ✅ Clear documentation of failures
- ✅ Runnable demo

### Good Success:
- ✅ 60-70% success rate on test set
- ✅ Handles basic akiya features (tatami, basic fusuma)
- ✅ Web demo with good UX
- ✅ akiya2.com expresses interest

### Great Success:
- ✅ 70%+ success rate
- ✅ Handles complex Japanese architecture
- ✅ Publication-ready results
- ✅ akiya2.com ready to integrate
- ✅ Can extend to business after thesis

---

## Thesis Writing Phase (After Implementation)

### Week 1-3 Schedule:

**Week 1: Structure + Methods**
- Introduction (2 pages)
- Related Work (5 pages - leverage your existing paper)
- Methodology (10 pages - describe what you built)

**Week 2: Results + Discussion**
- Experiments (8 pages - results, tables, figures)
- Discussion (5 pages - what worked, what didn't, why)
- Limitations (2 pages)

**Week 3: Polish**
- Conclusion (2 pages)
- Abstract (1 page)
- References
- Proofread everything
- Format according to university guidelines

### Thesis Sections:
1. **Abstract** (1 page)
2. **Introduction** (2-3 pages)
   - Problem statement
   - Akiya context
   - Thesis objectives
3. **Related Work** (5-7 pages)
   - Use your existing paper
   - Focus on 2D-3D conversion methods
   - Japanese architecture challenges
4. **Methodology** (10-15 pages)
   - System architecture
   - Model selection and training
   - Japanese architecture handling
   - 3D generation pipeline
5. **Experiments** (8-10 pages)
   - Dataset description
   - Evaluation metrics
   - Results and comparisons
   - Ablation studies
6. **Discussion** (5-7 pages)
   - Analysis of results
   - Failure case analysis
   - Limitations
   - Future work
7. **Conclusion** (2-3 pages)
8. **References**
9. **Appendix** (code, additional figures)

---

## Key Thesis Defense Points

### Expected Questions + Answers:

**Q: "Why only 100 images? That's a small dataset."**  
**A:** "This reflects real-world constraints in specialized domains. My work demonstrates transfer learning's effectiveness with limited data, which is more practical than methods requiring thousands of annotated examples. This is particularly relevant for akiya, where large-scale annotated datasets don't exist."

**Q: "Your accuracy is only 65%, why so low?"**  
**A:** "This is actually a main contribution - I identify and quantify specific architectural features of Japanese traditional housing that cause state-of-the-art methods to fail. I provide detailed failure mode analysis and propose targeted solutions. The gap between Western and Japanese architecture performance (95% vs 65%) quantifies an important research challenge."

**Q: "Why not use more complex models like Transformers?"**  
**A:** "Given the limited data and need for interpretability in production settings, I prioritized approaches that generalize well and provide explainable outputs. My hybrid rule-based approach achieves comparable performance with better interpretability and lower computational cost."

**Q: "What's the novel contribution?"**  
**A:** 
1. First systematic evaluation of 2D-3D conversion on Japanese traditional architecture
2. Curated and annotated dataset of 100 akiya floorplans
3. Quantitative analysis of architectural style transfer challenges
4. Hybrid ML + rule-based approach incorporating architectural domain knowledge
5. Industry partnership validation with akiya2.com

**Q: "How is this different from existing work?"**  
**A:** "Existing work focuses on Western or modern Asian architecture with rigid wall structures. My work addresses traditional Japanese architecture with flexible spaces (fusuma), modular layouts (tatami), and unique spatial concepts (engawa). This represents a significant architectural diversity challenge not addressed in prior work."

---

## Collaboration with akiya2.com

### During Thesis (2-3 months):

**What They Provide:**
- Access to floorplan images
- Feedback on prototype usability
- Real use cases and requirements
- Potential user study participants
- Letter of support/collaboration for thesis

**What You Provide:**
- Bi-weekly progress updates
- Demo of working prototype
- Technical documentation
- Results analysis and findings

**Communication Schedule:**
- Week 1: Kickoff meeting
- Week 3: First demo
- Week 5: Mid-point review
- Week 7: Final demo
- Week 9: Results presentation

### After Thesis (Business Phase):

**Potential Arrangements:**
- Technology licensing to akiya2.com
- Revenue sharing model
- Full-time position offer
- Continued collaboration on production system
- Expansion to other akiya platforms

---

## Budget Estimate (Thesis Phase)

| Item | Cost | Notes |
|------|------|-------|
| Cloud GPU (Colab Pro) | $10/month × 3 = $30 | Or use free tier |
| Demo hosting (optional) | $5-15/month | Vercel/Netlify free tier available |
| Domain name (optional) | $10/year | For demo |
| **Total** | **~$40-60** | Very affordable! |

**Cost Saving Tips:**
- Use Google Colab free tier initially
- Host demo on free platforms (HuggingFace Spaces, Streamlit Cloud)
- Use university computing resources if available

---

## Risk Management

### Potential Risks + Mitigation:

**Risk 1: Pre-trained model doesn't work on akiya images**
- Mitigation: Have backup models ready (3 options)
- Plan B: Train simpler model from scratch
- Plan C: Focus on subset of simpler floorplans

**Risk 2: 100 images insufficient for good results**
- Mitigation: Heavy data augmentation
- Document as limitation
- Request more images from akiya2.com if needed

**Risk 3: Japanese architecture too challenging**
- Mitigation: Start with simpler layouts
- Document challenges as contribution
- Focus on what CAN be learned

**Risk 4: Running out of time**
- Mitigation: Weekly milestone tracking
- Cut scope if needed (walls only)
- Buffer weeks 7-8 for contingency

**Risk 5: Technical blockers (library issues, etc.)**
- Mitigation: Have alternative tech stack ready
- Active community support (PyTorch/MMDet forums)
- Document workarounds

---

## Resources & References

### Key Papers (You've Read These):
1. 3DPlanNet (Park & Kim, 2021)
2. CubiCasa5k (Kalervo et al., 2019)
3. Plan2Scene (Vidanapathirana et al., 2021)
4. Raster-to-Vector (Liu et al., 2017)
5. FloorNet (Liu et al., 2019)

### Useful GitHub Repositories:
- Detectron2: facebook/detectron2
- MMDetection: open-mmlab/mmdetection
- CubiCasa5k: CubiCasa/CubiCasa5k
- Trimesh: mikedh/trimesh

### Tools & Platforms:
- Label Studio (already using)
- Weights & Biases (experiment tracking)
- Gradio (web demos)
- Google Colab (GPU access)

### Online Communities:
- PyTorch Forums
- Reddit: r/MachineLearning
- Papers With Code
- Computer Vision Discord servers

---

## Weekly Checklist Template

### Week X: [Focus Area]

**Monday:**
- [ ] Review last week's results
- [ ] Plan week's goals (3-5 specific outcomes)
- [ ] Set up this week's experiments

**Tuesday-Thursday:**
- [ ] Implement core functionality
- [ ] Run experiments
- [ ] Test on validation images
- [ ] Document results

**Friday:**
- [ ] Complete weekly deliverable
- [ ] Test on new images
- [ ] Update akiya2.com
- [ ] Git commit and push

**Weekend:**
- [ ] Review week's progress
- [ ] Plan next week
- [ ] Backup everything
- [ ] Optional: explore new ideas

**Key Metrics to Track:**
- Lines of code written
- Models tested
- Images processed
- Accuracy metrics
- Bugs fixed
- Documentation updated

---

## Final Thoughts

### Remember:
- **Perfect is the enemy of done** - get something working first
- **Document everything** - failures are learning opportunities
- **Iterate quickly** - test often, fail fast
- **Ask for help** - use forums, communities, and me!
- **Celebrate small wins** - every working component counts

### You Have:
✅ Real industry partner (akiya2.com)  
✅ Real data (100 labeled images)  
✅ Real problem to solve  
✅ Technical skills (Python, ML, software engineering)  
✅ Clear timeline (2-3 months)  

### You Don't Need:
❌ Perfect results (60-70% is fine)  
❌ Production-ready system  
❌ 1000s of training images  
❌ Real-time performance  
❌ Beautiful UI  

### This is Achievable! 🚀

Most thesis projects are purely theoretical. You have:
- Real partner
- Real data
- Real problem
- Clear path forward

**Now go build it!**

---

## Contact & Support

**When You Need Help:**
1. Check this roadmap first
2. Review error messages carefully
3. Search GitHub issues
4. Ask in relevant forums
5. Come back to me with specific questions

**Good Luck!** 🎓

---

*Last Updated: [Current Date]*  
*Version: 1.0*
