"""
CRF Parameter Tuning for Floorplan Segmentation
===============================================
This script helps you find the best CRF parameters by comparing
different configurations side-by-side.

Usage:
    python crf_parameter_tuning.py
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

# CONFIG
MODEL_PATH = "model_output/best_model.pth"
TEST_DIR = "../data/floorplan"
OUTPUT_DIR = "../data/crf_tuning_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Different CRF parameter configurations to test
CRF_CONFIGS = {
    'Original (Too Strong)': {
        'sxy_gaussian': 3, 'compat_gaussian': 3,
        'sxy_bilateral': 80, 'srgb_bilateral': 13, 'compat_bilateral': 10,
        'iterations': 5
    },
    'Light Smoothing': {
        'sxy_gaussian': 2, 'compat_gaussian': 2,
        'sxy_bilateral': 5, 'srgb_bilateral': 3, 'compat_bilateral': 5,
        'iterations': 3
    },
    'Minimal CRF': {
        'sxy_gaussian': 1, 'compat_gaussian': 1,
        'sxy_bilateral': 3, 'srgb_bilateral': 2, 'compat_bilateral': 3,
        'iterations': 2
    },
    'Edge Preserving': {
        'sxy_gaussian': 2, 'compat_gaussian': 1,
        'sxy_bilateral': 4, 'srgb_bilateral': 5, 'compat_bilateral': 3,
        'iterations': 3
    }
}

def apply_crf(image, probs, **params):
    h, w = image.shape[:2]
    d = dcrf.DenseCRF2D(w, h, probs.shape[0])
    d.setUnaryEnergy(unary_from_softmax(probs))
    d.addPairwiseGaussian(sxy=params['sxy_gaussian'], compat=params['compat_gaussian'])
    d.addPairwiseBilateral(sxy=params['sxy_bilateral'], srgb=params['srgb_bilateral'], 
                          rgbim=image, compat=params['compat_bilateral'])
    Q = d.inference(params['iterations'])
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
    return pred_raw, probs, original

def colorize(mask):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(16, 3), dtype=np.uint8)
    return colors[mask]

def compare_configs(original, pred_raw, probs, configs, save_path):
    n_configs = len(configs) + 2  # +2 for original image and U-Net only
    fig, axes = plt.subplots(2, (n_configs + 1) // 2, figsize=(5 * ((n_configs + 1) // 2), 10))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # U-Net only (no CRF)
    axes[1].imshow(colorize(pred_raw))
    axes[1].set_title('U-Net Only\n(No CRF)', fontsize=10, fontweight='bold')
    axes[1].axis('off')
    
    # Different CRF configurations
    for idx, (name, params) in enumerate(configs.items(), start=2):
        pred_crf = apply_crf(original, probs.astype(np.float32), **params)
        axes[idx].imshow(colorize(pred_crf))
        
        # Format parameter info
        param_text = f"sxy_g={params['sxy_gaussian']}, sxy_b={params['sxy_bilateral']}\niter={params['iterations']}"
        axes[idx].set_title(f'{name}\n{param_text}', fontsize=9)
        axes[idx].axis('off')
    
    # Hide any unused subplots
    for idx in range(n_configs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# MAIN
Path(OUTPUT_DIR).mkdir(exist_ok=True)
model = load_model()
test_images = list(Path(TEST_DIR).glob("*.png"))[:5]  # Test on 5 images

print("Testing different CRF configurations...")
print("Configurations being tested:")
for name, params in CRF_CONFIGS.items():
    print(f"  - {name}: {params}")
print()

for img_path in test_images:
    print(f"Processing: {img_path.name}")
    pred_raw, probs, original = predict(model, img_path)
    
    output_path = f"{OUTPUT_DIR}/{img_path.stem}_comparison.png"
    compare_configs(original, pred_raw, probs, CRF_CONFIGS, output_path)

print(f"\n✓ Done! Check {OUTPUT_DIR} to compare different CRF settings.")
print("\nRecommendations:")
print("  - If CRF makes it worse: Use U-Net predictions directly (no CRF)")
print("  - If boundaries are too jagged: Try 'Light Smoothing' or 'Edge Preserving'")
print("  - If losing too much detail: Try 'Minimal CRF'")
