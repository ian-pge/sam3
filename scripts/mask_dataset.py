import os
import argparse
import glob
import torch
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def main():
    parser = argparse.ArgumentParser(description="Mask cars in a dataset using SAM3")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory containing images")
    parser.add_argument("--output_dir", type=str, default="masks", help="Subdirectory name for saving masks")
    parser.add_argument("--prompt", type=str, default="car", help="Text prompt for segmentation")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    # Create output directory
    output_path = os.path.join(dataset_path, args.output_dir)
    os.makedirs(output_path, exist_ok=True)

    print("Loading SAM3 model...")
    # NOTE: SAM3 implementation might require CUDA. 
    # If running on CPU-only machine, this might fail or need explicit mapping if supported by library.
    try:
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have a CUDA-compatible GPU and PyTorch compiled with CUDA support.")
        # Attempt to proceed if it's just a warning, but likely it will fail later if model is None
        # returning here to avoid cascade errors if model build failed
        return

    # Get images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(dataset_path, ext)))
        # Also try case-insensitive uppercase
        image_files.extend(glob.glob(os.path.join(dataset_path, ext.upper())))
    
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"No images found in {dataset_path}")
        return

    print(f"Found {len(image_files)} images. Starting processing...")

    for idx, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        print(f"Processing {idx+1}/{len(image_files)}: {filename}")
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Inference
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(state=inference_state, prompt=args.prompt)
            
            # Output contains masks, boxes, scores. 
            # masks shape is presumably [N, H, W] or similar.
            masks = output["masks"]
            
            if masks is None or len(masks) == 0:
                print(f"  No {args.prompt} detected.")
                continue

            # Combine all masks for the prompt (logical OR) if multiple instances found
            if isinstance(masks, list):
                masks = torch.stack(masks)
            
            # We want to select the single mask with the largest area (most pixels)
            # First, reshape to [-1, H, W] to handle arbitrary batch/channel dims
            if masks.ndim >= 2:
                H, W = masks.shape[-2:]
                candidates = masks.reshape(-1, H, W) # Flatten all leading dims
                
                # Calculate area for each candidate mask
                # Assuming masks are boolean or 0/1 (if logits, we might need threshold, but usually they are masks)
                # Ensure float for summation if bool
                areas = candidates.float().sum(dim=(1, 2))
                
                # Find index of largest mask
                best_idx = torch.argmax(areas)
                final_mask = candidates[best_idx]
            else:
                # Should not happen if shapes are correct, but fallback
                final_mask = masks

            # Convert to numpy uint8
            if isinstance(final_mask, torch.Tensor):
                final_mask = final_mask.cpu().numpy()
            
            final_mask_uint8 = (final_mask * 255).astype(np.uint8)
            
            # Save
            save_name = os.path.splitext(filename)[0] + "_mask.png"
            save_path = os.path.join(output_path, save_name)
            Image.fromarray(final_mask_uint8).save(save_path)
            print(f"  Saved mask to {save_path} (largest detected area)")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    print("Processing complete.")

if __name__ == "__main__":
    main()
