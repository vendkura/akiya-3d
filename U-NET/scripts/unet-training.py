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
    # Paths
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR.parent / 'data'
    IMAGES_DIR = DATA_DIR / 'floorplan'          # Your images folder
    MASKS_DIR = DATA_DIR / 'floorplan_masks_13classes'     # Training masks (13 classes)
    
    # Model
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    NUM_CLASSES = 13  # Remapped from 16 to 13 categories
    
    # Class names (13 classes after consolidation)
    CLASS_NAMES = [
        "dining_area",    # 0 - merged: DK, LDK, dinning room
        "bathroom",       # 1 - merged: toilet, washroom
        "bedroom",        # 2
        "room",           # 3 - merged: includes living room
        "closet",         # 4
        "entrance",       # 5
        "kitchen",        # 6
        "outdoor_space",  # 7
        "stairs",         # 8
        "sliding_door",   # 9
        "door",           # 10
        "windows",        # 11
        "background",     # 12
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
    OUTPUT_DIR = Path("model_output")
    OUTPUT_DIR.mkdir(exist_ok=True)

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
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()

# ==================== DATA AUGMENTATION ====================
def get_training_augmentation():
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_validation_augmentation():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ==================== METRICS ====================
def calculate_iou(pred, target, num_classes):
    """Calculate IoU per class"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    return np.nanmean(ious)

# ==================== TRAINING ====================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_iou = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        pred = torch.argmax(outputs, dim=1)
        iou = calculate_iou(pred, masks, Config.NUM_CLASSES)
        
        total_loss += loss.item()
        total_iou += iou
        
        pbar.set_postfix({'loss': loss.item(), 'iou': iou})
    
    return total_loss / len(loader), total_iou / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            pred = torch.argmax(outputs, dim=1)
            iou = calculate_iou(pred, masks, Config.NUM_CLASSES)
            
            total_loss += loss.item()
            total_iou += iou
            
            pbar.set_postfix({'loss': loss.item(), 'iou': iou})
    
    return total_loss / len(loader), total_iou / len(loader)

# ==================== VISUALIZATION ====================
def visualize_predictions(model, dataset, device, num_samples=3):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        image, mask = dataset[i]
        
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device))
            pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
        
        # Denormalize image for display
        img_display = image.permute(1, 2, 0).numpy()
        img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_display = np.clip(img_display, 0, 1)
        
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask.numpy(), cmap='tab20')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred, cmap='tab20')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / 'predictions.png', dpi=150)
    plt.close()

# ==================== MAIN TRAINING ====================
def main():
    print("=" * 60)
    print("🚀 U-NET TRAINING FOR AKIYA FLOORPLANS")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Classes: {Config.NUM_CLASSES}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print("=" * 60)
    
    # Get all image and mask paths
    image_files = sorted(list(Config.IMAGES_DIR.glob("*.png")) + list(Config.IMAGES_DIR.glob("*.jpg")))
    mask_files = []
    all_masks = list(Config.MASKS_DIR.glob("*_mask.png"))

    for img_path in image_files:
        # Normalize filename: remove extra spaces
        img_stem_normalized = ' '.join(img_path.stem.split())
        
        # Find matching mask
        matched = False
        for mask_path in all_masks:
            mask_stem_normalized = ' '.join(mask_path.stem.replace('_mask', '').split())
            
            if img_stem_normalized == mask_stem_normalized:
                mask_files.append(mask_path)
                matched = True
                break
        
        if not matched:
            print(f"⚠️  No mask found for {img_path.name}")
    
    # Split dataset
    n = len(image_files)
    n_train = int(n * Config.TRAIN_SPLIT)
    n_val = int(n * Config.VAL_SPLIT)
    
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    train_images = [image_files[i] for i in train_idx]
    train_masks = [mask_files[i] for i in train_idx]
    val_images = [image_files[i] for i in val_idx]
    val_masks = [mask_files[i] for i in val_idx]
    test_images = [image_files[i] for i in test_idx]
    test_masks = [mask_files[i] for i in test_idx]
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Create datasets
    train_dataset = FloorplanDataset(train_images, train_masks, get_training_augmentation())
    val_dataset = FloorplanDataset(val_images, val_masks, get_validation_augmentation())
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model
    print("\n🏗️  Building U-Net model...")
    model = smp.Unet(
        encoder_name=Config.ENCODER,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=Config.NUM_CLASSES,
    )
    model = model.to(Config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': []
    }
    
    best_val_iou = 0
    
    # Training loop
    print("\n🏋️  Starting training...\n")
    for epoch in range(Config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 60)
        
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        val_loss, val_iou = validate_epoch(model, val_loader, criterion, Config.DEVICE)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), Config.OUTPUT_DIR / 'best_model.pth')
            print(f"✅ Saved best model (IoU: {val_iou:.4f})")
        
        print()
    
    # Save training history
    with open(Config.OUTPUT_DIR / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.plot(history['val_iou'], label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('Training and Validation IoU')
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / 'training_curves.png', dpi=150)
    
    # Visualize predictions
    print("🎨 Generating prediction visualizations...")
    visualize_predictions(model, val_dataset, Config.DEVICE)
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"✅ Best validation IoU: {best_val_iou:.4f}")
    print(f"📂 Model saved: {Config.OUTPUT_DIR / 'best_model.pth'}")
    print(f"📊 History saved: {Config.OUTPUT_DIR / 'history.json'}")
    print(f"📈 Plots saved: {Config.OUTPUT_DIR / 'training_curves.png'}")
    print("=" * 60)

if __name__ == "__main__":
    main()