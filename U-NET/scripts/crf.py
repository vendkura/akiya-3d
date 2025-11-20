"""
CRF Post-Processing for U-Net Predictions
==========================================
This script applies Conditional Random Fields (CRF) to clean up noisy 
segmentation boundaries from U-Net predictions.

Usage:
    python crf_postprocess.py

The script will:
1. Load your trained U-Net model
2. Make predictions on test images
3. Apply CRF post-processing
4. Save before/after comparisons
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


# CONFIG
MODEL_PATH = "model_output/best_model.pth"
TEST_DIR = "../data/floorplan"
MASK_DIR = "../data/floorplan_masks"  # Ground truth masks directory
OUTPUT_DIR = "../data/floorplan_crf_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NOTE: Based on experiments, CRF degrades floorplan segmentation quality!
# U-Net predictions are already excellent for architectural drawings.
# Set USE_CRF=False to skip CRF post-processing.
USE_CRF = True  # Toggle this to compare with/without CRF

def apply_crf(image, probs, sxy_gaussian=2, compat_gaussian=2, 
               sxy_bilateral=5, srgb_bilateral=3, compat_bilateral=5, iterations=3):
    """
    Apply CRF with tunable parameters.
    
    For floorplans, we want:
    - Lower sxy values = less smoothing, preserve sharp edges
    - Lower compat values = trust the model more, smooth less
    - Fewer iterations = less aggressive smoothing
    """
    h, w = image.shape[:2]
    d = dcrf.DenseCRF2D(w, h, probs.shape[0])
    
    # Unary potential from the neural network
    d.setUnaryEnergy(unary_from_softmax(probs))
    
    # Smoothness kernel (position only) - for general smoothness
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)
    
    # Appearance kernel (position + color) - for edge-aware smoothing
    d.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb_bilateral, 
                          rgbim=image, compat=compat_bilateral)
    
    Q = d.inference(iterations)
    return np.argmax(Q, axis=0).reshape((h, w))

def load_model():
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=16)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def preprocess(img_path):
    transform = Compose([
        Resize(512, 512),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    return transform(image=img)['image'].unsqueeze(0), cv2.resize(img, (512, 512))

def predict(model, img_path):
    input_tensor, original = preprocess(img_path)
    with torch.no_grad():
        output = model(input_tensor.to(DEVICE))
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred_raw = np.argmax(probs, axis=0)
        
        # Apply CRF only if enabled (not recommended for floorplans)
        if USE_CRF:
            pred_crf = apply_crf(original, probs.astype(np.float32))
        else:
            pred_crf = pred_raw  # Use U-Net prediction directly
            
    return pred_raw, pred_crf, original

def colorize(mask):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(16, 3), dtype=np.uint8)
    return colors[mask]

def save_comparison(original, pred_raw, pred_crf, path, gt_mask=None):
    if gt_mask is not None:
        if USE_CRF:
            # Show all 4: Original, GT, U-Net, U-Net+CRF
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(original); axes[0].set_title('Original Image'); axes[0].axis('off')
            axes[1].imshow(colorize(gt_mask)); axes[1].set_title('Ground Truth'); axes[1].axis('off')
            axes[2].imshow(colorize(pred_raw)); axes[2].set_title('U-Net Only'); axes[2].axis('off')
            axes[3].imshow(colorize(pred_crf)); axes[3].set_title('U-Net + CRF'); axes[3].axis('off')
        else:
            # Show only 3: Original, GT, U-Net (skip CRF since it's identical)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(original); axes[0].set_title('Original Image'); axes[0].axis('off')
            axes[1].imshow(colorize(gt_mask)); axes[1].set_title('Ground Truth'); axes[1].axis('off')
            axes[2].imshow(colorize(pred_raw)); axes[2].set_title('U-Net Prediction'); axes[2].axis('off')
    else:
        if USE_CRF:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(original); axes[0].set_title('Original Image'); axes[0].axis('off')
            axes[1].imshow(colorize(pred_raw)); axes[1].set_title('U-Net Only'); axes[1].axis('off')
            axes[2].imshow(colorize(pred_crf)); axes[2].set_title('U-Net + CRF'); axes[2].axis('off')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(original); axes[0].set_title('Original Image'); axes[0].axis('off')
            axes[1].imshow(colorize(pred_raw)); axes[1].set_title('U-Net Prediction'); axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

# MAIN
Path(OUTPUT_DIR).mkdir(exist_ok=True)
model = load_model()
test_images = list(Path(TEST_DIR).glob("*.png"))[:10]

for img_path in test_images:
    pred_raw, pred_crf, original = predict(model, img_path)
    
    # Try to load ground truth (mask files have "_mask" suffix)
    gt_mask = None
    mask_filename = img_path.stem + "_mask.png"
    mask_path = Path(MASK_DIR) / mask_filename
    if mask_path.exists():
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    save_comparison(original, pred_raw, pred_crf, f"{OUTPUT_DIR}/{img_path.stem}.png", gt_mask)
    print(f"Processed: {img_path.name}")

status = "WITH CRF" if USE_CRF else "WITHOUT CRF (recommended)"
print(f"\nDone! Check {OUTPUT_DIR}")
print(f"Status: Running {status}")
if not USE_CRF:
    print("\n✓ CRF is disabled. U-Net predictions are used directly.")
    print("  (CRF degrades quality for floorplan segmentation)")




