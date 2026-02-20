# Akiya-3D: Automatic Conversion of 2D Japanese Floorplans to 3D Models

> 🚧 **Preliminary Documentation** — This README provides a research overview. Comprehensive documentation including reproducibility instructions, dataset specifications, and detailed methodology will be released upon project completion.

## Abstract

This repository presents an end-to-end pipeline for automatically converting 2D Japanese floorplan images into 3D architectural models. Unlike existing approaches (CubiCasa5k, MMDetection-based methods) that target Western architectural conventions, this work addresses the unique characteristics of traditional Japanese residential architecture—including tatami rooms, DK/LDK layouts, fusuma sliding panels, and engawa verandas.

The pipeline combines semantic segmentation using U-Net with ResNet34 encoder (achieving 53.5% mIoU on a custom 13-class taxonomy) with geometric post-processing and wall-based 3D extrusion to produce "dollhouse style" navigable models.

Developed in partnership with [akiya2.com](https://akiya2.com) to address the visualization needs of foreign buyers acquiring *akiya* (abandoned traditional Japanese houses).

## Repository Structure

| Branch | Description |
|--------|-------------|
| `main` | Project documentation and overview |
| `u-net_implementation` | **Core segmentation pipeline** — U-Net with ResNet34 encoder, training scripts, inference |
| `3d-pipeline-integration` | **3D reconstruction** — Wall extrusion, boundary extraction, OBJ generation |
| `cubicasa-original-implementation` | Baseline experiments with CubiCasa5k approach |
| `mmdetection-implementation` | Baseline experiments with MMDetection framework |
| `media_documentation` | Figures, visualizations, and supplementary materials |

## Methodology

### Semantic Segmentation

The segmentation model employs a U-Net architecture with ResNet34 encoder pretrained on ImageNet. Training configuration:

- Input resolution: 512×512
- Optimizer: Adam (lr=1e-4)
- Batch size: 4
- Augmentation: rotation, flipping, elastic deformation, color jittering

### Taxonomy Design

A custom 13-class taxonomy was developed for Japanese residential architecture:

| Category | Classes |
|----------|---------|
| Living spaces | LDK, DK, Bedroom, Japanese-style room (和室) |
| Functional | Bathroom, Toilet, Kitchen, Entrance (玄関), Storage |
| Structural | Wall, Door, Window |
| Exterior | Balcony/Veranda |

> **Key finding**: Taxonomy consolidation (16 → 13 classes) yielded greater mIoU improvement than doubling dataset size, demonstrating the importance of label design in low-resource specialized domains.

### 3D Reconstruction

The reconstruction pipeline uses wall-based extrusion rather than room-based extrusion:

```
Input Mask → Boundary Extraction → Morphological Filtering → Wall Extrusion (12cm) → OBJ Export
```

This approach produces architecturally realistic models where walls are solid geometry and rooms are walkable voids.

## Experimental Results

### Segmentation Performance Progression

| Experiment | mIoU | Δ | Configuration |
|------------|------|---|---------------|
| E1: Baseline | 17.7% | — | 16 classes, 50 images, light augmentation |
| E2: Augmentation | 19.3% | +1.6 | Heavy augmentation strategy |
| E3: Taxonomy | 26.0% | +6.7 | Consolidated to 13 classes |
| E4: Data scaling | 42.2% | +16.2 | Extended to 100 images |
| E5: Fine-tuning | **53.5%** | +11.3 | Extended training, hyperparameter optimization |

### Baseline Comparisons

Experiments with Western-focused approaches revealed significant domain gap:

- **CubiCasa5k**: Pretrained models failed on Japanese architectural elements
- **MMDetection**: Required extensive adaptation for non-Western room types

These negative results validated the need for a domain-specific approach.

## Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Input Image    │────▶│  U-Net Encoder   │────▶│  Segmentation   │
│  (Japanese FP)  │     │  (ResNet34)      │     │  Mask (13 cls)  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  3D OBJ Model   │◀────│  Wall Extrusion  │◀────│  Boundary       │
│  (Dollhouse)    │     │  (12cm walls)    │     │  Extraction     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (12GB+ VRAM recommended)
- Dependencies: segmentation-models-pytorch, albumentations, OpenCV, trimesh

*Detailed requirements and installation instructions will be provided in branch-specific documentation.*

## Dataset

The model was trained on a custom dataset of ~100 manually annotated Japanese floorplan images sourced through partnership with akiya2.com. Annotations were created using Label Studio following the 13-class taxonomy.

> Dataset availability details to be announced.

## Acknowledgments

- **akiya2.com** — Industry partnership and data provision

## License

*To be specified upon release.*

---

<p align="center">
  <i>Full documentation, pretrained weights, and reproducibility materials forthcoming.</i>
</p>
