import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model import box_ops
from sam3.model.data_misc import FindStage

def main():
    parser = argparse.ArgumentParser(
        description="Mask objects using SAM3 Image Mode (High Performance Batched)"
    )
    parser.add_argument(
        "dataset_path", type=str, help="Path to the dataset directory containing images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="masks",
        help="Subdirectory name for saving masks",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Confidence threshold for detection",
    )
    parser.add_argument(
        "--viz", action="store_true", help="Enable visualization of masks on images"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to a local model checkpoint"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Number of images to process at once"
    )
    args = parser.parse_args()

    print(f"🚀 SAM3 Batched Masking | Batch Size: {args.batch_size} | Car: Largest | Window: Union")

    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path '{dataset_path}' does not exist.")
        return

    # Create output directories
    output_path = args.output_dir
    if not os.path.isabs(output_path):
        output_path = os.path.join(dataset_path, args.output_dir)

    car_output_path = os.path.join(output_path, "car")
    window_output_path = os.path.join(output_path, "window")
    os.makedirs(car_output_path, exist_ok=True)
    os.makedirs(window_output_path, exist_ok=True)

    if args.viz:
        os.makedirs(os.path.join(car_output_path, "viz"), exist_ok=True)
        os.makedirs(os.path.join(window_output_path, "viz"), exist_ok=True)

    # 1. Setup Model
    print("📦 Loading SAM3 Image model...", end="", flush=True)
    try:
        model = build_sam3_image_model(
            checkpoint_path=args.checkpoint,
        )
        processor = Sam3Processor(model, confidence_threshold=args.threshold)
        print(" Done!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Pre-compute Text Prompts
    print("🧠 Pre-computing features...", end="", flush=True)
    try:
        with torch.inference_mode():
            car_text_features = model.backbone.forward_text(["car"], device=processor.device)
            window_text_features = model.backbone.forward_text(["window"], device=processor.device)
    except Exception as e:
        print(f"❌ Error encoding text: {e}")
        return
    print(" Done!")

    # 3. Find Images
    print("\n🔍 Scanning images...")
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    
    search_path = dataset_path
    if os.path.exists(os.path.join(dataset_path, "images")):
        search_path = os.path.join(dataset_path, "images")
        print(f"👉 Found 'images' subdirectory, using: {search_path}")

    # Recursive search
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(search_path, "**", ext), recursive=True))
        image_files.extend(glob.glob(os.path.join(search_path, "**", ext.upper()), recursive=True))
    
    image_files.sort()

    if not image_files:
        print(f"❌ No valid images found in {search_path}")
        return

    print(f"🔍 Found {len(image_files)} images. Starting processing...")

    stats = {"processed": 0, "errors": 0}
    start_time = time.time()

    # Batch Processing Loop
    for i in tqdm(range(0, len(image_files), args.batch_size), desc="Processing Batches 📦"):
        batch_paths = image_files[i : i + args.batch_size]
        current_batch_size = len(batch_paths)
        
        try:
            # Load images
            images = []
            for img_path in batch_paths:
                images.append(Image.open(img_path).convert("RGB"))
            
            # Set Image Batch
            inference_state = processor.set_image_batch(images)
            
            # Setup batched prompt input
            find_stage = FindStage(
                img_ids=torch.arange(current_batch_size, device=processor.device, dtype=torch.long),
                text_ids=torch.zeros(current_batch_size, device=processor.device, dtype=torch.long),
                input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                input_points=None, input_points_mask=None
            )
            dummy_geometric = model._get_dummy_prompt(num_prompts=current_batch_size)

            # --- Define Helper for Decoding ---
            def run_batch_inference(features_update):
                with torch.inference_mode():
                    # Update features
                    inference_state["backbone_out"].update(features_update)
                    
                    # Forward
                    outputs = model.forward_grounding(
                        backbone_out=inference_state["backbone_out"],
                        find_input=find_stage,
                        geometric_prompt=dummy_geometric,
                        find_target=None
                    )
                    
                    out_logits = outputs["pred_logits"] # [B, N, 1]
                    out_masks = outputs["pred_masks"]   # [B, N, H, W] (low res)
                    
                    # Probabilities
                    out_probs = out_logits.sigmoid()
                    presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
                    out_probs = (out_probs * presence_score).squeeze(-1) # [B, N]
                    
                    return out_masks, out_probs

            # === CAR PASS ===
            car_masks_lowres, car_probs = run_batch_inference(car_text_features)
            
            # === WINDOW PASS ===
            window_masks_lowres, window_probs = run_batch_inference(window_text_features)

            # === Post-Processing per Image ===
            for b_idx in range(current_batch_size):
                img_path = batch_paths[b_idx]
                w, h = images[b_idx].size
                
                # Interpolate function
                def get_hires_masks(lowres_masks):
                    # lowres_masks: [N, H_small, W_small]
                    # Resize to [N, h, w]
                    if lowres_masks.numel() == 0: return lowres_masks
                    masks = F.interpolate(
                        lowres_masks.unsqueeze(0), # [1, N, H, W]
                        size=(1008, 1008), # Model resolution
                        mode="bilinear", 
                        align_corners=False
                    ).squeeze(0)
                    
                    # Crop/Resize to original image aspect if needed? 
                    # Processor usually resizes 1008x1008 back to original.
                    # Simple resize to (h, w)
                    masks = F.interpolate(
                        masks.unsqueeze(0),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)
                    return masks.sigmoid()

                # -- CAR (Largest) --
                c_probs = car_probs[b_idx] # [N]
                c_masks = car_masks_lowres[b_idx] # [N, h, w]
                
                keep = c_probs > args.threshold
                if keep.any():
                    valid_masks = c_masks[keep]
                    # Get High Res
                    hires = get_hires_masks(valid_masks)
                    binary = (hires > 0.5)
                    # Largest
                    areas = binary.float().sum(dim=(1,2))
                    largest = torch.argmax(areas)
                    final_car = (binary[largest].float().cpu().numpy() * 255).astype(np.uint8)
                else:
                    final_car = np.zeros((h, w), dtype=np.uint8)

                # -- WINDOW (Union) --
                w_probs = window_probs[b_idx]
                w_masks = window_masks_lowres[b_idx]
                
                keep = w_probs > args.threshold
                if keep.any():
                    valid_masks = w_masks[keep]
                    hires = get_hires_masks(valid_masks)
                    binary = (hires > 0.5)
                    # Union
                    union = torch.any(binary, dim=0)
                    final_window = (union.float().cpu().numpy() * 255).astype(np.uint8)
                else:
                    final_window = np.zeros((h, w), dtype=np.uint8) # Or Car mask?

                # Save
                base_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
                Image.fromarray(final_car).save(os.path.join(car_output_path, base_name))
                Image.fromarray(final_window).save(os.path.join(window_output_path, base_name))

                # Viz
                if args.viz:
                    image_viz = images[b_idx].copy().convert("RGBA")
                    viz_name = os.path.splitext(os.path.basename(img_path))[0] + "_cutout.png"
                    
                    # Car
                    img_car = image_viz.copy()
                    img_car.putalpha(Image.fromarray(final_car).convert("L"))
                    img_car.save(os.path.join(car_output_path, "viz", viz_name))
                    
                    # Window
                    img_win = image_viz.copy()
                    img_win.putalpha(Image.fromarray(final_window).convert("L"))
                    img_win.save(os.path.join(window_output_path, "viz", viz_name))

            stats["processed"] += current_batch_size

        except Exception as e:
            stats["errors"] += 1
            print(f"  ❌ Error processing batch starting {os.path.basename(batch_paths[0])}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_duration = time.time() - start_time
    print(f"\n🏁 Finished {stats['processed']} images in {total_duration:.1f}s")

if __name__ == "__main__":
    main()
