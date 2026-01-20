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

## Car Segmentation Task

We provide a specialized task to mask objects (e.g., cars) in a dataset.

### Command
```bash
pixi run mask-cars <dataset_path> [options]
```

### Arguments
- `dataset_path`: Path to the directory containing images (Required).
- `--output_dir`: Subdirectory to save masks (Default: `masks`).
- `--prompt`: Text prompt for segmentation (Default: `"car"`).
- `--threshold`: Confidence threshold for detection (Default: `0.15`). Lower this if cars are missed.

### Example
```bash
pixi run mask-cars /workspace/datasets/voiture_random/mixed/images --prompt "car"
```

**Note:** The script is configured to save the **single largest mask** detected (most pixels), which typically corresponds to the main object of interest.

---
