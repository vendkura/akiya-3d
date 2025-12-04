import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# CONFIG
MODEL_PATH = "model_output/best_model.pth"
TEST_IMAGES_DIR = "../data/floorplan"
TEST_MASKS_DIR = "../data/floorplan_masks_13classes"
OUTPUT_DIR = "model_output/per_class_analysis/13_classes"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CLASS NAMES - 13 consolidated classes
CLASS_NAMES = [
    "dining_area", "bathroom", "bedroom", "closet", "room",
    "door", "entrance", "kitchen", "outdoor_space",
    "sliding_door", "stairs", "window", "balcony"
]

def load_model():
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=13)
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
    return transform(image=img)['image'].unsqueeze(0)

def predict(model, img_path):
    input_tensor = preprocess(img_path)
    with torch.no_grad():
        output = model(input_tensor.to(DEVICE))
        pred = torch.argmax(output, dim=1)[0].cpu().numpy()
    return pred

def calculate_iou_per_class(pred, target, num_classes=13):
    """Calculate IoU for each class."""
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        if union == 0:
            iou = np.nan
        else:
            iou = intersection / union
        ious.append(iou)
    return np.array(ious)

def build_confusion_matrix(pred, target, num_classes=13):
    """Build confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_cls in range(num_classes):
        mask = (target == true_cls)
        if mask.sum() > 0:
            for pred_cls in range(num_classes):
                cm[true_cls, pred_cls] = np.logical_and(mask, pred == pred_cls).sum()
    return cm

def plot_iou_per_class(class_ious, class_counts, output_path):
    """Bar chart of IoU per class."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Filter out classes with no samples
    valid_classes = ~np.isnan(class_ious)
    names = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES)) if valid_classes[i]]
    ious = class_ious[valid_classes]
    counts = class_counts[valid_classes]
    
    # Create bars
    colors = ['green' if iou > 0.5 else 'orange' if iou > 0.3 else 'red' for iou in ious]
    bars = ax.bar(range(len(names)), ious, color=colors, alpha=0.7)
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'n={int(count)}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title('Per-Class IoU Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='IoU=0.5')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, output_path):
    """Heatmap of confusion matrix."""
    # Normalize by row (true class)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Proportion'}, ax=ax)
    
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title('Confusion Matrix (Normalized by True Class)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def colorize_mask(mask, num_classes=16):
    """Colorize segmentation mask with consistent colors."""
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return colors[mask]

def save_class_examples(predictions_dict, image_data_dict, output_dir, top_k=3):
    """Save visual examples (best and worst) for each class."""
    examples_dir = output_dir / "class_examples"
    examples_dir.mkdir(exist_ok=True)
    
    print("\nGenerating visual examples for each class...")
    
    for cls_id in range(len(CLASS_NAMES)):
        cls_name = CLASS_NAMES[cls_id]
        
        # Get examples for this class
        examples = [(img_name, ious[cls_id]) 
                   for img_name, ious in predictions_dict.items() 
                   if not np.isnan(ious[cls_id])]
        
        if len(examples) == 0:
            print(f"  Skipping {cls_name}: no examples found")
            continue
        
        # Sort by IoU
        examples.sort(key=lambda x: x[1], reverse=True)
        
        # Determine how many to show (at least 1 best and 1 worst if we have 2+ examples)
        n_show = min(top_k, len(examples))
        
        if len(examples) >= 2:
            # Show best and worst
            best_examples = examples[:n_show]
            worst_examples = examples[-n_show:][::-1]
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Best example
            best_img_name, best_iou = best_examples[0]
            data = image_data_dict[best_img_name]
            
            axes[0, 0].imshow(data['original_image'])
            axes[0, 0].set_title(f'BEST Example\nIoU = {best_iou:.4f}', fontweight='bold')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(colorize_mask(data['mask']))
            axes[0, 1].set_title('Ground Truth')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(colorize_mask(data['pred']))
            axes[0, 2].set_title('Prediction')
            axes[0, 2].axis('off')
            
            # Worst example
            worst_img_name, worst_iou = worst_examples[0]
            data = image_data_dict[worst_img_name]
            
            axes[1, 0].imshow(data['original_image'])
            axes[1, 0].set_title(f'WORST Example\nIoU = {worst_iou:.4f}', fontweight='bold')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(colorize_mask(data['mask']))
            axes[1, 1].set_title('Ground Truth')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(colorize_mask(data['pred']))
            axes[1, 2].set_title('Prediction')
            axes[1, 2].axis('off')
            
            fig.suptitle(f'Class: {cls_name} (ID={cls_id})', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Clean filename
            safe_name = cls_name.replace(' ', '_').replace('/', '-')
            plt.savefig(examples_dir / f'{cls_id:02d}_{safe_name}_examples.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved examples for: {cls_name} ({len(examples)} samples)")
        else:
            print(f"  Skipping {cls_name}: only 1 example (need ≥2 for best/worst)")

def generate_report(class_ious, class_counts, cm, output_path):
    """Generate text report with statistics."""
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PER-CLASS IoU ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Overall stats
        valid_ious = class_ious[~np.isnan(class_ious)]
        f.write(f"Overall Mean IoU: {valid_ious.mean():.4f}\n")
        f.write(f"Overall Std IoU:  {valid_ious.std():.4f}\n")
        f.write(f"Classes evaluated: {len(valid_ious)}/{len(CLASS_NAMES)}\n\n")
        
        # Per-class stats
        f.write("-"*70 + "\n")
        f.write("PER-CLASS STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Class':<20} {'IoU':<10} {'Pixels':<15} {'Status':<15}\n")
        f.write("-"*70 + "\n")
        
        for i, (name, iou, count) in enumerate(zip(CLASS_NAMES, class_ious, class_counts)):
            if np.isnan(iou):
                status = "NOT PRESENT"
                iou_str = "N/A"
            elif iou > 0.5:
                status = "GOOD"
                iou_str = f"{iou:.4f}"
            elif iou > 0.3:
                status = "MODERATE"
                iou_str = f"{iou:.4f}"
            else:
                status = "POOR"
                iou_str = f"{iou:.4f}"
            
            f.write(f"{name:<20} {iou_str:<10} {int(count):<15} {status:<15}\n")
        
        # Top confusions
        f.write("\n" + "-"*70 + "\n")
        f.write("TOP CLASS CONFUSIONS (excluding diagonal)\n")
        f.write("-"*70 + "\n")
        
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        np.fill_diagonal(cm_norm, 0)  # Ignore correct predictions
        
        # Find top 10 confusions
        confusions = []
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                if i != j and cm_norm[i, j] > 0.05:  # At least 5% confusion
                    confusions.append((CLASS_NAMES[i], CLASS_NAMES[j], cm_norm[i, j]))
        
        confusions.sort(key=lambda x: x[2], reverse=True)
        
        for true_cls, pred_cls, proportion in confusions[:10]:
            f.write(f"{true_cls:>20} → {pred_cls:<20} : {proportion*100:.1f}%\n")
        
        f.write("\n" + "="*70 + "\n")

# MAIN
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(exist_ok=True)

print("Loading model...")
model = load_model()

print("Finding test images...")
test_images = list(Path(TEST_IMAGES_DIR).glob("*.png")) + list(Path(TEST_IMAGES_DIR).glob("*.jpg"))

# Storage
all_ious = []
all_cm = np.zeros((13, 13), dtype=np.int64)
predictions_dict = {}
image_data_dict = {}  # Store image data for visualization

print(f"Processing {len(test_images)} images...")
for i, img_path in enumerate(test_images, 1):
    # Find mask
    mask_path = Path(TEST_MASKS_DIR) / img_path.name
    if not mask_path.exists():
        mask_path = Path(TEST_MASKS_DIR) / img_path.name.replace('.png', '_mask.png')
    if not mask_path.exists():
        print(f"  Skipping {img_path.name} (no mask found)")
        continue
    
    # Load original image for visualization
    original_image = cv2.imread(str(img_path))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (512, 512))
    
    # Predict
    pred = predict(model, img_path)
    target = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Calculate metrics
    ious = calculate_iou_per_class(pred, target)
    cm = build_confusion_matrix(pred, target)
    
    all_ious.append(ious)
    all_cm += cm
    predictions_dict[img_path.name] = ious
    
    # Store image data for visualization
    image_data_dict[img_path.name] = {
        'original_image': original_image,
        'pred': pred,
        'mask': target
    }
    
    print(f"  [{i}/{len(test_images)}] {img_path.name} - Mean IoU: {np.nanmean(ious):.4f}")

# Aggregate results
all_ious = np.array(all_ious)
mean_ious = np.nanmean(all_ious, axis=0)
class_counts = np.nansum(~np.isnan(all_ious), axis=0)

print("\nGenerating visualizations...")
plot_iou_per_class(mean_ious, class_counts, output_dir / "iou_per_class.png")
plot_confusion_matrix(all_cm, output_dir / "confusion_matrix.png")
save_class_examples(predictions_dict, image_data_dict, output_dir)
generate_report(mean_ious, class_counts, all_cm, output_dir / "class_statistics.txt")

print(f"\n✓ Done! Results saved to {output_dir}")
print(f"\nQuick Summary:")
print(f"  Overall Mean IoU: {np.nanmean(mean_ious):.4f}")
print(f"  Best class:  {CLASS_NAMES[np.nanargmax(mean_ious)]} ({np.nanmax(mean_ious):.4f})")
print(f"  Worst class: {CLASS_NAMES[np.nanargmin(mean_ious)]} ({np.nanmin(mean_ious):.4f})")