"""
FPN Training Script for 104-image dataset with 13 classes.
Experiment 05 FINAL: Training on fully-annotated 62+42 images (rooms included).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# ==================== CONFIGURATION ====================
class Config:
    # Paths - USE FINAL 104-IMAGE DATASET WITH COMPLETE ANNOTATIONS
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR.parent / 'data'
    IMAGES_DIR = DATA_DIR / 'floorplan_104_final'
    MASKS_DIR = DATA_DIR / 'floorplan_masks_104_13classes_final'
    
    # Model - CHANGED TO FPN
    MODEL_ARCH = 'FPN'  # Feature Pyramid Network
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    NUM_CLASSES = 13
    
    # Class names (13 classes)
    CLASS_NAMES = [
        "dining_area", "bathroom", "bedroom", "closet", "room",
        "door", "entrance", "kitchen", "outdoor_space",
        "sliding_door", "stairs", "window", "balcony"
    ]
    
    # Training
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data split
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Output
    OUTPUT_DIR = Path("model_output/fpn_104images_final")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ==================== DATASET ====================
class FloorplanDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        
        # Load mask
        mask = np.array(Image.open(self.mask_paths[idx]))
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()

# ==================== TRANSFORMS ====================
def get_train_transform():
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ==================== METRICS ====================
def calculate_iou(pred, target, num_classes):
    """Calculate mean IoU across all classes."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    # Return mean IoU (ignoring NaN)
    ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(ious) if ious else 0.0

# ==================== TRAINING ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_iou = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        preds = torch.argmax(outputs, dim=1)
        iou = calculate_iou(preds, masks, Config.NUM_CLASSES)
        
        total_loss += loss.item()
        total_iou += iou
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})
    
    return total_loss / len(dataloader), total_iou / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            preds = torch.argmax(outputs, dim=1)
            iou = calculate_iou(preds, masks, Config.NUM_CLASSES)
            
            total_loss += loss.item()
            total_iou += iou
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})
    
    return total_loss / len(dataloader), total_iou / len(dataloader)

# ==================== VISUALIZATION ====================
def plot_training_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # IoU
    ax2.plot(history['train_iou'], label='Train IoU', marker='o')
    ax2.plot(history['val_iou'], label='Val IoU', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Training and Validation IoU')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_predictions(model, dataset, device, num_samples=4, save_path=None):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 4))
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            
            # Predict
            pred = model(image.unsqueeze(0).to(device))
            pred = torch.argmax(pred, dim=1)[0].cpu().numpy()
            
            # Denormalize image for display
            img_display = image.permute(1, 2, 0).numpy()
            img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_display = np.clip(img_display, 0, 1)
            
            # Plot
            axes[i, 0].imshow(img_display)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask.numpy(), cmap='tab20', vmin=0, vmax=Config.NUM_CLASSES-1)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='tab20', vmin=0, vmax=Config.NUM_CLASSES-1)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ==================== MAIN ====================
def main():
    print("="*70)
    print("FPN TRAINING - 104 IMAGES FINAL, 13 CLASSES (Experiment 05)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {Config.MODEL_ARCH}")
    print(f"  Encoder: {Config.ENCODER}")
    print(f"  Images: {Config.IMAGES_DIR}")
    print(f"  Masks: {Config.MASKS_DIR}")
    print(f"  Classes: {Config.NUM_CLASSES}")
    print(f"  Device: {Config.DEVICE}")
    
    # Get image and mask paths
    image_paths = sorted(list(Config.IMAGES_DIR.glob("*.png")) + list(Config.IMAGES_DIR.glob("*.jpg")))
    
    # Masks have "_mask" suffix
    mask_paths = []
    for img in image_paths:
        mask_name = img.stem + "_mask.png"
        mask_paths.append(Config.MASKS_DIR / mask_name)
    
    # Filter valid pairs
    valid_pairs = [(img, mask) for img, mask in zip(image_paths, mask_paths) if mask.exists()]
    print(f"\n✓ Found {len(valid_pairs)} valid image-mask pairs")
    
    image_paths, mask_paths = zip(*valid_pairs)
    
    # Split dataset
    np.random.seed(42)
    indices = np.random.permutation(len(image_paths))
    
    train_size = int(Config.TRAIN_SPLIT * len(indices))
    val_size = int(Config.VAL_SPLIT * len(indices))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_images = [image_paths[i] for i in train_indices]
    train_masks = [mask_paths[i] for i in train_indices]
    val_images = [image_paths[i] for i in val_indices]
    val_masks = [mask_paths[i] for i in val_indices]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_images)}")
    print(f"  Val: {len(val_images)}")
    print(f"  Test: {len(test_indices)}")
    
    # Create datasets
    train_dataset = FloorplanDataset(train_images, train_masks, get_train_transform())
    val_dataset = FloorplanDataset(val_images, val_masks, get_val_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create FPN model
    print(f"\n✓ Creating {Config.MODEL_ARCH} model...")
    model = smp.FPN(
        encoder_name=Config.ENCODER,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=Config.NUM_CLASSES
    )
    model = model.to(Config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': []
    }
    
    best_val_iou = 0.0
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{Config.NUM_EPOCHS}]")
        
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, val_iou = validate(model, val_loader, criterion, Config.DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        print(f"  Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val IoU:   {val_iou:.4f}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), Config.OUTPUT_DIR / 'best_model.pth')
            print(f"  ✓ Saved best model (Val IoU: {val_iou:.4f})")
    
    # Save history
    with open(Config.OUTPUT_DIR / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot curves
    plot_training_curves(history, Config.OUTPUT_DIR / 'training_curves.png')
    
    # Visualize predictions
    visualize_predictions(model, val_dataset, Config.DEVICE, 
                         save_path=Config.OUTPUT_DIR / 'predictions.png')
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest Validation IoU: {best_val_iou:.4f}")
    print(f"Results saved to: {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
