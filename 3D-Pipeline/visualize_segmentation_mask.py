"""
Visualize FPN segmentation mask with colors for each class.
Shows what the model is actually predicting.
"""

import cv2
import numpy as np
from pathlib import Path
from fpn_inference import FPNInference
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Class names and colors
CLASS_NAMES = {
    0: "background",
    1: "dining_area",
    2: "bathroom",
    3: "bedroom",
    4: "closet",
    5: "room",
    6: "door",
    7: "entrance",
    8: "kitchen",
    9: "outdoor_space",
    10: "sliding_door",
    11: "stairs",
    12: "window",
    13: "balcony"
}

# Distinct colors for each class (BGR for OpenCV)
CLASS_COLORS = {
    0: (50, 50, 50),        # background - dark gray
    1: (255, 182, 193),     # dining_area - light pink
    2: (255, 200, 124),     # bathroom - light orange
    3: (173, 216, 230),     # bedroom - light blue
    4: (255, 228, 181),     # closet - bisque
    5: (255, 255, 224),     # room - light yellow
    6: (160, 82, 45),       # door - brown
    7: (240, 255, 240),     # entrance - honeydew
    8: (144, 238, 144),     # kitchen - light green
    9: (152, 251, 152),     # outdoor_space - pale green
    10: (169, 169, 169),    # sliding_door - dark gray
    11: (211, 211, 211),    # stairs - light gray
    12: (135, 206, 250),    # window - sky blue
    13: (176, 224, 230),    # balcony - powder blue
}

def visualize_mask_colored(mask_array, output_path="mask_colored.png"):
    """
    Convert grayscale mask (class IDs) to colored visualization.
    
    Args:
        mask_array: Numpy array with shape (H, W), values 0-13 (class IDs)
        output_path: Path to save colored visualization
    """
    height, width = mask_array.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Assign colors based on class ID
    for class_id, color in CLASS_COLORS.items():
        mask = (mask_array == class_id)
        colored_mask[mask] = color
    
    # Save
    cv2.imwrite(output_path, colored_mask)
    print(f"✓ Saved colored mask: {output_path}")
    
    return colored_mask

def create_legend(output_path="mask_legend.png"):
    """Create a legend showing all classes and their colors."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create patches for legend
    patches = []
    for class_id in sorted(CLASS_NAMES.keys()):
        color_bgr = CLASS_COLORS[class_id]
        # Convert BGR to RGB for matplotlib
        color_rgb = (color_bgr[2]/255, color_bgr[1]/255, color_bgr[0]/255)
        class_name = CLASS_NAMES[class_id]
        patches.append(mpatches.Patch(color=color_rgb, label=f"{class_id:2d}: {class_name}"))
    
    ax.legend(handles=patches, loc='center', fontsize=12, frameon=True)
    ax.axis('off')
    ax.set_title('FPN Segmentation Classes', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved legend: {output_path}")
    plt.close()

def main():
    """Generate colored segmentation mask."""
    print("=" * 60)
    print("SEGMENTATION MASK VISUALIZATION")
    print("=" * 60)
    
    # Initialize FPN
    fpn_model_path = "../U-NET/scripts/model_output/fpn_62images/best_model.pth"
    print(f"\nLoading FPN model...")
    fpn = FPNInference(fpn_model_path)
    
    # Find test image
    image_dir = Path("../U-NET/data/floorplan/")
    test_images = list(image_dir.glob("*.png"))
    
    if not test_images:
        print("❌ No test images found!")
        return
    
    test_image = test_images[0]
    print(f"Using image: {test_image.name}")
    
    # Run inference
    print(f"\nRunning FPN inference...")
    mask, original_shape, classes_present = fpn.infer(test_image)
    
    print(f"✓ Mask generated: {mask.shape}")
    print(f"✓ Original shape: {original_shape}")
    print(f"✓ Classes detected: {', '.join(classes_present.keys())}")
    
    # Visualize
    print(f"\nGenerating colored visualization...")
    colored_mask = visualize_mask_colored(mask, "mask_colored.png")
    
    # Create legend
    print(f"Creating legend...")
    create_legend("mask_legend.png")
    
    # Statistics
    print(f"\n{'=' * 60}")
    print("CLASS STATISTICS")
    print(f"{'=' * 60}")
    
    unique_classes, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    
    for class_id, count in zip(unique_classes, counts):
        class_name = CLASS_NAMES.get(class_id, f"unknown_{class_id}")
        percentage = (count / total_pixels) * 100
        print(f"{class_id:2d} {class_name:20s}: {count:8d} px ({percentage:5.2f}%)")
    
    print(f"\n✓ Visualization complete!")
    print(f"  Colored mask: mask_colored.png")
    print(f"  Legend: mask_legend.png")

if __name__ == "__main__":
    main()
