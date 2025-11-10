# Experiment [NUMBER]: [SHORT DESCRIPTION]

**Date:** [DATE]  
**Goal:** [What are you testing/improving in this experiment?]

---

## 1. Dataset Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Total images | | |
| Train/Val/Test split | | e.g., 70/15/15 |
| Number of classes | 16 | Japanese-specific categories |
| Image dimensions | | e.g., 512x512 |
| Annotation tool | Label Studio | |

**Dataset notes:**
- Any special considerations about this dataset batch
- Data quality issues noticed
- Class distribution if relevant

---

## 2. Model Architecture

| Component | Specification |
|-----------|---------------|
| Base architecture | U-Net |
| Encoder | ResNet34 |
| Pre-trained weights | ImageNet |
| Library | segmentation-models-pytorch |

**Architecture notes:**
- Why you chose this architecture
- Any modifications from standard U-Net

---

## 3. Training Configuration

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Epochs | |
| Batch size | |
| Learning rate | |
| Optimizer | |
| Loss function | |
| Device | GPU/CPU |

### Data Augmentation
```python
# List your augmentation pipeline here
- HorizontalFlip: p=0.5
- VerticalFlip: p=0.5
- Rotation: degrees=15
# etc.
```

### Other Settings
- Early stopping: Yes/No (patience=X)
- Learning rate scheduler: Yes/No (type)
- Class weights: Yes/No

---

## 4. Results

### Quantitative Metrics

| Metric | Final Value | Best Epoch |
|--------|-------------|------------|
| Train Loss | | |
| Val Loss | | |
| Train IoU | | |
| Val IoU | | |
| Test IoU (if evaluated) | | |

### Training Characteristics
- Total training time: 
- Convergence: Did it plateau? Stabilize?
- Overfitting signs: Gap between train/val IoU

### Visual Results
- [Attach or reference prediction visualizations]
- Best cases: [Which floorplans worked well?]
- Failure cases: [Which floorplans failed?]

---

## 5. Analysis

### What Worked
- 
- 
- 

### What Didn't Work
- 
- 
- 

### Observations
- Specific patterns about Japanese architectural features
- Boundary quality
- Small feature detection (doors, windows, stairs)
- Hallway/corridor performance
- Room type confusion (which classes get mixed up?)

---

## 6. Next Steps

Based on this experiment:
1. 
2. 
3. 

---

## 7. Files & Artifacts

| File | Location | Description |
|------|----------|-------------|
| Training script | | |
| Model checkpoint | | Best model weights |
| Training logs | | JSON/CSV with epoch data |
| Predictions | | Visualization images |
| Config file | | Hyperparameters used |

---

## 8. Comparison to Previous Experiments

| Experiment | Dataset Size | Val IoU | Key Difference |
|------------|--------------|---------|----------------|
| Baseline (Exp 1) | 30 | 0.18 | |
| This experiment | | | |

**Relative improvement:** [Better/worse/similar and why]

---

## Notes & Observations

Any other relevant information:
- Unexpected behaviors
- Technical issues encountered
- Ideas that came up during training
- Questions to investigate