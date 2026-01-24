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

## Car Segmentation Task (Video Mode)

We provide a specialized task to mask objects (e.g., cars) in a dataset using **SAM 3 Video Mode**. This leverages temporal tracking to ensure consistent masks across sequential video frames.

### File Naming Requirement (CRITICAL)
Your image files **MUST** follow this specific naming convention to be recognized and grouped correctly:
- `frame_{frame_index}_video_{video_id}.png`

Example:
- `frame_0_video_1.png`
- `frame_1_video_1.png`
- `frame_0_video_2.png` 

Files not matching this pattern will be ignored.

### Command
```bash
pixi run mask-cars <dataset_path> [options]
```

### Arguments
- `dataset_path`: Path to the directory containing images (Required).
- `--output_dir`: Subdirectory to save masks (Default: `masks`).
- `--prompt`: Text prompt for segmentation (Default: `"car"`).
- `--threshold`: Confidence threshold for detection (Default: `0.15`). Lower this if cars are missed.
- `--viz`: Enable visualization of crops (cutouts) on real images (Optional). Saved in `viz` folder.
- `--chunk_size`: Number of frames to process at once (Default: `60`). Lower this if OOM occurs.

### Features
- **Video Mode**: Automatically groups frames by video ID and uses SAM 3's temporal tracker for consistent segmentation.
- **Progress Bar**: Tracks processed videos and frames.
- **Visualizations**: Optional (`--viz`). Saves `_cutout.png` showing the isolated object.
- **Analytics**: Displays a summary of processed videos, total frames, and errors.

### Example
```bash
pixi run mask-cars /workspace/datasets/voiture_random/mixed/images --prompt "car"
```

**Note:** The script creates temporary working directories to link sequential frames for the model, maximizing storage efficiency. These are cleaned up automatically.

---
