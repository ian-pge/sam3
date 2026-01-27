## Installation with Pixi (Recommended)

This project uses [Pixi](https://pixi.sh/) for dependency management, ensuring a reproducible environment with **automatic GPU support**.

### 1. Install Environment
This command installs Python, PyTorch (CUDA enabled), and all dependencies.

```bash
pixi install
```

### 2. Authenticate with Hugging Face
SAM 3 is a gated model. You must have access approved on [Hugging Face](https://huggingface.co/facebook/sam3).
Authenticate to download the model:

```bash
pixi run huggingface-cli login
```
(Paste your User Access Token when prompted).

---

## Object Segmentation Task (SAM3 Batched)

This script is highly optimized to segment **Cars** and **Windows** from a dataset. It uses batched processing for maximum speed.

### Specialized Logic
- **Car**: Keeps the mask with the **largest surface area**.
- **Window**: Keeps the **union** of all detected window masks.

### Command
```bash
pixi run mask-cars <dataset_path> [options]
```

### Arguments
- `dataset_path`: Path to the dataset (searches recursively).
- `--batch_size`: Number of images to process in parallel (Default: 4). Increase for faster speed if GPU RAM allows.
- `--output_dir`: Output root directory (Default: `masks`).
- `--threshold`: Detection confidence (Default: `0.15`).
- `--viz`: Generate visualizations.

### Features
- **Batched Processing**: ~10x faster than single-mode.
- **Specific Logic**: Handles "Car" and "Window" differently as requested.
- **Recursive Search**: Automatically finds images in subfolders.

### Example
```bash
pixi run mask-cars /workspace/datasets/voiture_dure --batch_size 4 --viz
```

---
