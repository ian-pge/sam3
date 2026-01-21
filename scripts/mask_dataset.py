import os
import argparse
import glob
import torch
import numpy as np
import time
from PIL import Image
from tqdm import tqdm
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def main():
    parser = argparse.ArgumentParser(description="Mask cars in a dataset using SAM3")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory containing images")
    parser.add_argument("--output_dir", type=str, default="masks", help="Subdirectory name for saving masks")
    parser.add_argument("--prompt", type=str, default="car", help="Text prompt for segmentation")
    parser.add_argument("--threshold", type=float, default=0.15, help="Confidence threshold for detection")
    parser.add_argument("--fp16", action="store_true", help="Use half-precision (fp16/bf16) to save VRAM")
    args = parser.parse_args()

    print("\n🚀 Starting SAM3 Masking Pipeline")
    print("=" * 50)
    
    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path '{dataset_path}' does not exist.")
        return

    # Create output directory
    output_path = os.path.join(dataset_path, args.output_dir)
    os.makedirs(output_path, exist_ok=True)

    # Recap
    print("📋 Configuration Recap:")
    print(f"  ├── 📂 Dataset Path: {dataset_path}")
    print(f"  ├── 📂 Output Dir:   {output_path}")
    print(f"  ├── 📝 Prompt:       '{args.prompt}'")
    print(f"  ├── 🎚️  Threshold:    {args.threshold}")
    print(f"  └── 💾 FP16 Mode:    {'Enabled ✅' if args.fp16 else 'Disabled ❌'}")
    print("=" * 50)
    print("\n📦 Loading SAM3 model...")

    try:
        model = build_sam3_image_model()
        
        # Optimization: Use half-precision to save VRAM
        if args.fp16 and torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"  ⚡ Moving model to {dtype} for VRAM optimization...")
            model = model.to(dtype=dtype)
        
        processor = Sam3Processor(model, confidence_threshold=args.threshold)
        print("  ✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Ensure you have a CUDA-compatible GPU and PyTorch compiled with CUDA support.")
        return

    # Get images
    print("\n🔍 Scanning for images...")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(dataset_path, ext)))
        # Also try case-insensitive uppercase
        image_files.extend(glob.glob(os.path.join(dataset_path, ext.upper())))
    
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"❌ No images found in {dataset_path}")
        return

    print(f"  ✅ Found {len(image_files)} images.")
    print("\n▶️  Starting processing loop...")

    # Statistics tracking
    stats = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "total_area_pixels": 0
    }
    
    start_time = time.time()

    # Progress bar
    pbar = tqdm(image_files, desc="Processing 🚗", unit="img", ncols=100)
    
    for img_path in pbar:
        filename = os.path.basename(img_path)
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Inference
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(state=inference_state, prompt=args.prompt)
            
            masks = output["masks"]
            
            if masks is None or len(masks) == 0:
                stats["skipped"] += 1
                pbar.write(f"  ⚠️  No {args.prompt} detected in {filename}. Skipping.")
                continue

            # Combine all masks for the prompt (logical OR) if multiple instances found
            if isinstance(masks, list):
                masks = torch.stack(masks)
            
            # Select largest mask
            if masks.ndim >= 2:
                H, W = masks.shape[-2:]
                candidates = masks.reshape(-1, H, W) 
                areas = candidates.float().sum(dim=(1, 2))
                best_idx = torch.argmax(areas)
                final_mask = candidates[best_idx]
            else:
                final_mask = masks

            # Convert to numpy uint8
            if isinstance(final_mask, torch.Tensor):
                final_mask = final_mask.cpu().numpy()
            
            final_mask_uint8 = (final_mask * 255).astype(np.uint8)
            
            # Update stats
            mask_area = np.sum(final_mask_uint8 > 0)
            stats["total_area_pixels"] += mask_area
            
            # Save
            save_name = os.path.splitext(filename)[0] + "_mask.png"
            save_path = os.path.join(output_path, save_name)
            Image.fromarray(final_mask_uint8).save(save_path)
            
            stats["processed"] += 1

        except Exception as e:
            stats["errors"] += 1
            pbar.write(f"  ❌ Error processing {filename}: {e}")
        
        # Clear VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    end_time = time.time()
    total_duration = end_time - start_time
    avg_time = total_duration / len(image_files) if len(image_files) > 0 else 0

    print("\n" + "=" * 50)
    print("📊 Final Analytics")
    print("=" * 50)
    print(f"⏱️  Total Duration:      {total_duration:.2f} seconds")
    print(f"⚡ Average Time/Image:  {avg_time:.2f} seconds")
    print("-" * 30)
    print(f"🔢 Total Images:        {len(image_files)}")
    print(f"✅ Successfully Masked: {stats['processed']} ({stats['processed']/len(image_files)*100:.1f}%)")
    print(f"⚠️  Skipped (No detection): {stats['skipped']}")
    print(f"❌ Errors:              {stats['errors']}")
    print("-" * 30)
    print(f"💾 Output Directory:    {output_path}")
    print("=" * 50)
    print("🏁 Done!")

if __name__ == "__main__":
    main()
