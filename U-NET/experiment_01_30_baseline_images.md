# Experiment 1: Baseline U-Net with 30 Images

**Date:** November 10, 2025  
**Goal:** Establish baseline performance for Japanese akiya floorplan segmentation using minimal dataset.

---

## 1. Dataset Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Total images | 30 | From akiya2.com database |
| Train/Val/Test split | 70/15/15 | 21 train / 4-5 val / 4-5 test |
| Number of classes | 16 | Japanese-specific categories (DK, LDK, tatami rooms, etc.) |
| Image dimensions | 512×512 | Resized during training (various original sizes) |
| Annotation tool | Label Studio | Manual annotation |

**Dataset notes:**
- First batch of labeled akiya floorplans
- Includes traditional Japanese architectural features: tatami layouts, fusuma panels, engawa
- Quality control: All annotations verified

---

## 2. Model Architecture

| Component | Specification |
|-----------|---------------|
| Base architecture | U-Net |
| Encoder | ResNet34 |
| Pre-trained weights | ImageNet |
| Library | segmentation-models-pytorch |

**Architecture notes:**
- ResNet34 chosen based on systematic literature review
- Best balance for small dataset: boundary precision + computational efficiency
- ImageNet pre-training provides general feature extraction despite domain gap

---

## 3. Training Configuration

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch size | 4 |
| Learning rate | 0.0001 |
| Optimizer | Adam |
| Loss function | Cross-Entropy |
| Device | CUDA (GPU) |

### Data Augmentation
```python
# Training augmentation pipeline:
- Resize: 512×512
- HorizontalFlip: p=0.5
- VerticalFlip: p=0.3
- RandomRotate90: p=0.5
- ShiftScaleRotate: scale_limit=0.1, rotate_limit=15°, shift_limit=0.1, p=0.5
- RandomBrightnessContrast: p=0.3
- Normalize: ImageNet mean/std (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)
```

### Other Settings
- Early stopping: No (fixed 50 epochs)
- Learning rate scheduler: Yes (ReduceLROnPlateau, patience=5, factor=0.5)
- Class weights: No

---

## 4. Results

### Quantitative Metrics

| Metric | Final Value (Epoch 50) | Best Epoch |
|--------|------------------------|------------|
| Train Loss | 0.618 | Epoch 50 |
| Val Loss | 0.978 | Epoch 48 (0.962) |
| Train IoU | 0.271 (27.1%) | Epoch 50 |
| Val IoU | 0.177 (17.7%) | Epoch 49 (18.1%) |
| Test IoU | Not yet evaluated | - |

### Training Characteristics
- Total training time: ~50 epochs (time depends on GPU, typically 1-2 hours for this dataset)
- Convergence: Both loss and IoU showing continued improvement, not fully plateaued
- Overfitting signs: Moderate gap between train IoU (27%) and val IoU (18%) - approximately 9 percentage points
- Loss decreased steadily from ~3.0 to ~0.6 (train) and ~3.1 to ~1.0 (val)

### Visual Results
- See: predictions.png, training_curves.png
- Best cases: Large open rooms (DK, LDK areas) show reasonable segmentation
- Failure cases: 
  - Boundaries are very noisy and imprecise
  - Hallways/corridors heavily fragmented
  - Small features (doors, stairs) not captured
  - Room boundaries bleed into adjacent spaces

---

## 5. Analysis

### What Worked
- Model successfully learns general room layout structure
- Large room regions (orange areas in predictions) are approximately correct in location
- Training is stable - no divergence or collapse
- Proof that U-Net can extract meaningful patterns from Japanese floorplans

### What Didn't Work
- Boundary precision is poor - very noisy edges
- Small architectural features not detected
- Hallway segmentation particularly problematic
- Class confusion evident in multi-colored noise at boundaries

### Observations
- IoU of 0.18 is ~3x better than random guessing (1/16 ≈ 0.06) but far from production quality (0.70+)
- The model identifies "where rooms are" but struggles with "what type" and "exact boundaries"
- For 30 images with 16 classes, this is actually reasonable performance
- Japanese architectural features (tatami mats, sliding doors) not specifically recognized yet
- Need more data to reduce overfitting and improve generalization

---

## 6. Next Steps

Based on this experiment:
1. **Scale dataset to ~60 images** 
2. Add more aggressive data augmentation to combat overfitting
3. Consider class weighting if some room types are underrepresented
4. Eventually test transfer learning from Western floorplan models
5. Implement post-processing (CRF) to clean up boundaries

---

## 7. Files & Artifacts

| File | Location | Description |
|------|----------|-------------|
| Training script | U-NET/scripts/unet-training.py | Main training loop |
| Model checkpoint | U-NET/scripts/model_output/best_model.pth | Best model weights (epoch ~48-49) |
| Training logs | U-NET/scripts/model_output/history.json | Epoch-by-epoch metrics |
| Predictions | U-NET/scripts/model_output/predictions.png | 3 example visualizations |
| Training curves | U-NET/scripts/model_output/training_curves.png | Loss and IoU plots |
| Config file | In unet-training.py (Config class) | Hyperparameters embedded in script |

---

## 8. Comparison to Previous Experiments

| Experiment | Dataset Size | Val IoU | Key Difference |
|------------|--------------|---------|----------------|
| Baseline (Exp 1) | 30 | 0.18 | Initial baseline - this experiment |
| Future Exp 2 | 60 | TBD | 2x data |
| Future Exp 3 | 100 | TBD | Full dataset |

**Relative improvement:** This is the baseline - all future experiments compare against Val IoU = 0.18

---

## Notes & Observations

- Dataset size of 30 is very small for 16-class segmentation task
- Expected performance limitations due to data scarcity
- Model architecture and training pipeline work correctly
- This baseline validates the technical approach before scaling up
- For thesis: This demonstrates "standard approach limitations" before proposing improvements
- Japanese architectural features present unique challenge not addressed by Western-trained models