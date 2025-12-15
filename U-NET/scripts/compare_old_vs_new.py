"""
Visual comparison: Old 62 images vs New 42 images predictions
Shows why 104-image model underperforms compared to 62-image model
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==================== CONFIGURATION ====================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'

# Model and data paths
MODEL_PATH = SCRIPT_DIR / 'model_output' / 'fpn_104images' / 'best_model.pth'
IMAGES_DIR = DATA_DIR / 'floorplan_104'
MASKS_DIR = DATA_DIR / 'floorplan_masks_104_13classes'
OUTPUT_DIR = SCRIPT_DIR / 'model_output' / 'fpn_104images' / 'old_vs_new_comparison'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

NUM_CLASSES = 13
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = [
    "dining_area", "bathroom", "bedroom", "closet", "room",
    "door", "entrance", "kitchen", "outdoor_space",
    "sliding_door", "stairs", "window", "balcony"
]

# ==================== TRANSFORMS ====================
def get_transform():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ==================== LOAD MODEL ====================
print("Loading FPN model...")
model = smp.FPN(
    encoder_name='resnet34',
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print(f"✓ Model loaded from {MODEL_PATH}")

# ==================== GET IMAGES ====================
all_images = sorted(list(IMAGES_DIR.glob('*')))
old_images = [img for img in all_images if img.stem.startswith('old_')]
new_images = [img for img in all_images if img.stem.startswith('new_')]

print(f"\n✓ Found {len(old_images)} old images, {len(new_images)} new images")

# ==================== PREDICT ====================
transform = get_transform()

def predict_image(image_path):
    """Predict mask for a single image"""
    # Load image
    image = np.array(Image.open(image_path).convert('RGB'))
    original_size = image.shape[:2]
    
    # Transform
    augmented = transform(image=image)
    img_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1)[0].cpu().numpy()
    
    # Resize back to original
    pred = Image.fromarray(pred.astype(np.uint8))
    pred = pred.resize((original_size[1], original_size[0]), Image.NEAREST)
    pred = np.array(pred)
    
    return image, pred

def load_mask(image_path):
    """Load corresponding ground truth mask"""
    mask_name = image_path.stem + "_mask.png"
    mask_path = MASKS_DIR / mask_name
    
    if mask_path.exists():
        return np.array(Image.open(mask_path))
    return None

# ==================== VISUALIZE COMPARISON ====================
def create_comparison_grid(num_samples=3):
    """Create side-by-side comparison of old vs new images"""
    
    # Sample random images
    np.random.seed(42)
    old_samples = np.random.choice(old_images, min(num_samples, len(old_images)), replace=False)
    new_samples = np.random.choice(new_images, min(num_samples, len(new_images)), replace=False)
    
    # Create figure: 2 columns (old, new) x num_samples rows
    # Each row shows: Image | Ground Truth | Prediction
    fig, axes = plt.subplots(num_samples, 6, figsize=(24, num_samples * 4))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    print("\nGenerating comparison visualizations...")
    
    for i in range(num_samples):
        # Process OLD image
        old_img_path = old_samples[i]
        old_img, old_pred = predict_image(old_img_path)
        old_mask = load_mask(old_img_path)
        
        # Plot OLD
        axes[i, 0].imshow(old_img)
        axes[i, 0].set_title(f'OLD Image\n{old_img_path.stem}', fontsize=10)
        axes[i, 0].axis('off')
        
        if old_mask is not None:
            axes[i, 1].imshow(old_mask, cmap='tab20', vmin=0, vmax=NUM_CLASSES-1)
            axes[i, 1].set_title('Ground Truth', fontsize=10)
            axes[i, 1].axis('off')
        
        axes[i, 2].imshow(old_pred, cmap='tab20', vmin=0, vmax=NUM_CLASSES-1)
        axes[i, 2].set_title('Prediction', fontsize=10)
        axes[i, 2].axis('off')
        
        # Process NEW image
        new_img_path = new_samples[i]
        new_img, new_pred = predict_image(new_img_path)
        new_mask = load_mask(new_img_path)
        
        # Plot NEW
        axes[i, 3].imshow(new_img)
        axes[i, 3].set_title(f'NEW Image\n{new_img_path.stem}', fontsize=10)
        axes[i, 3].axis('off')
        
        if new_mask is not None:
            axes[i, 4].imshow(new_mask, cmap='tab20', vmin=0, vmax=NUM_CLASSES-1)
            axes[i, 4].set_title('Ground Truth', fontsize=10)
            axes[i, 4].axis('off')
        
        axes[i, 5].imshow(new_pred, cmap='tab20', vmin=0, vmax=NUM_CLASSES-1)
        axes[i, 5].set_title('Prediction', fontsize=10)
        axes[i, 5].axis('off')
    
    plt.suptitle('OLD Dataset (62 images) vs NEW Dataset (42 images)\nFPN Model Predictions Comparison', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / 'old_vs_new_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison to: {save_path}")
    plt.close()

# ==================== CALCULATE SEPARATE IoUs ====================
def calculate_iou_by_dataset():
    """Calculate separate IoU for old vs new images"""
    print("\nCalculating IoU separately for old and new datasets...")
    
    def calc_iou_for_images(image_list, dataset_name):
        total_iou = 0
        count = 0
        
        for img_path in image_list:
            _, pred = predict_image(img_path)
            mask = load_mask(img_path)
            
            if mask is None:
                continue
            
            # Calculate IoU
            ious = []
            for cls in range(NUM_CLASSES):
                pred_mask = (pred == cls)
                true_mask = (mask == cls)
                intersection = np.logical_and(pred_mask, true_mask).sum()
                union = np.logical_or(pred_mask, true_mask).sum()
                
                if union > 0:
                    ious.append(intersection / union)
            
            if ious:
                total_iou += np.mean(ious)
                count += 1
        
        return total_iou / count if count > 0 else 0
    
    old_iou = calc_iou_for_images(old_images, "OLD")
    new_iou = calc_iou_for_images(new_images, "NEW")
    
    print(f"\n{'='*60}")
    print(f"IoU BY DATASET:")
    print(f"{'='*60}")
    print(f"OLD 62 images IoU: {old_iou:.4f} ({old_iou*100:.2f}%)")
    print(f"NEW 42 images IoU: {new_iou:.4f} ({new_iou*100:.2f}%)")
    print(f"Difference: {(old_iou - new_iou)*100:.2f}% (old - new)")
    print(f"{'='*60}")
    
    return old_iou, new_iou

# ==================== MAIN ====================
print("="*70)
print("OLD vs NEW DATASET COMPARISON")
print("="*70)

# Create visual comparison
create_comparison_grid(num_samples=4)

# Calculate separate IoUs
old_iou, new_iou = calculate_iou_by_dataset()

print("\n✓ Analysis complete!")
print(f"\nResults saved to: {OUTPUT_DIR}")
