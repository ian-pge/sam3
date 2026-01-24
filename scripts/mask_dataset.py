import os
import argparse
import glob
import torch
import numpy as np
import time
import gc
import re
import tempfile
import shutil
from PIL import Image
from tqdm import tqdm
from sam3.model_builder import build_sam3_video_predictor

def parse_video_info(filename):
    """
    Parses filenames of format: frame_{frame_idx}_video_{video_idx}.png
    Returns: (video_idx, frame_idx, full_path)
    """
    basename = os.path.basename(filename)
    # Regex to match frame_*_video_*.ext
    match = re.search(r'frame_(\d+)_video_(\d+)', basename)
    if match:
        frame_idx = int(match.group(1))
        video_idx = int(match.group(2))
        return video_idx, frame_idx, filename
    return None

def main():
    parser = argparse.ArgumentParser(description="Mask cars in a dataset using SAM3 Video Mode")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory containing images")
    parser.add_argument("--output_dir", type=str, default="masks", help="Subdirectory name for saving masks")
    parser.add_argument("--prompt", type=str, default="car", help="Text prompt for segmentation")
    parser.add_argument("--threshold", type=float, default=0.15, help="Confidence threshold for detection")
    parser.add_argument("--viz", action="store_true", help="Enable visualization of masks on images")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a local model checkpoint")
    parser.add_argument("--cpu-offload", action="store_true", help="Offload video frames to CPU to save GPU memory")
    args = parser.parse_args()

    print("\n🚀 Starting SAM3 Video Masking Pipeline")
    print("=" * 50)
    
    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path '{dataset_path}' does not exist.")
        return

    # Create output directory
    output_path = args.output_dir # Use absolute path if provided, or handle relative
    if not os.path.isabs(output_path):
        output_path = os.path.join(dataset_path, args.output_dir)
        
    os.makedirs(output_path, exist_ok=True)
    
    # Create visualization directory
    viz_path = os.path.join(output_path, "viz")
    if args.viz:
        os.makedirs(viz_path, exist_ok=True)

    # 1. Setup
    print("📦 Loading SAM3 Video model...")
    # Use all available GPUs
    gpus_to_use = range(torch.cuda.device_count()) if torch.cuda.is_available() else []
    
    predictor = build_sam3_video_predictor(
        gpus_to_use=gpus_to_use,
        checkpoint_path=args.checkpoint,
        offload_video_to_cpu=args.cpu_offload
    )

    print("\n🔍 Scanning and grouping images...")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(dataset_path, ext)))
        image_files.extend(glob.glob(os.path.join(dataset_path, ext.upper())))
    
    video_groups = {} # video_idx -> [(frame_idx, filepath)]
    
    for img_path in image_files:
        info = parse_video_info(img_path)
        if info:
            vid_idx, frame_idx, path = info
            if vid_idx not in video_groups:
                video_groups[vid_idx] = []
            video_groups[vid_idx].append((frame_idx, path))
    
    if not video_groups:
        print(f"❌ No valid video frames found in {dataset_path} matching pattern 'frame_*_video_*'")
        return

    # Sort frames within each video
    for vid_idx in video_groups:
        video_groups[vid_idx].sort(key=lambda x: x[0])

    print(f"  ✅ Found {len(video_groups)} unique videos with total {sum(len(v) for v in video_groups.values())} frames.")

    print("\n📦 Loading SAM3 Video model...")
    try:
        # Use all available GPUs
        gpus_to_use = range(torch.cuda.device_count()) if torch.cuda.is_available() else []
        predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use, checkpoint_path=args.checkpoint)

        print("  ✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("\n▶️  Starting video processing loop...")
    
    stats = {
        "videos_processed": 0,
        "frames_processed": 0,
        "errors": 0
    }
    
    start_time = time.time()
    
    # Process each video
    for vid_idx, frames in tqdm(video_groups.items(), desc="Processing Videos 🎥"):
        # Create a temporary directory for this video's frames
        with tempfile.TemporaryDirectory() as temp_video_dir:
            try:
                # 1. Prepare symlinks: 00000.jpg, 00001.jpg, etc.
                # Map temp_idx -> original_path to save masks later
                temp_to_original_map = {} 
                
                for i, (original_frame_idx, original_path) in enumerate(frames):
                    ext = os.path.splitext(original_path)[1]
                    symlink_name = f"{i:05d}{ext}"
                    symlink_path = os.path.join(temp_video_dir, symlink_name)
                    os.symlink(original_path, symlink_path)
                    temp_to_original_map[i] = original_path

                # 2. Start Session
                with torch.inference_mode():
                    response = predictor.handle_request({
                        "type": "start_session",
                        "resource_path": temp_video_dir
                    })
                    session_id = response["session_id"]
                    
                    # 3. Add text prompt to the FIRST frame (frame 0 in temp dir)
                    # We assume the object is present in the first frame.
                    prompt_response = predictor.handle_request({
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": args.prompt
                    })

                    target_obj_id = None # Default to None (keep all if no decision made, or handle accordingly)

                    # 3b. Filter for the largest object (Main Car)
                    if "outputs" in prompt_response and "out_binary_masks" in prompt_response["outputs"]:
                        masks = prompt_response["outputs"]["out_binary_masks"] # (N, H, W)
                        obj_ids = prompt_response["outputs"]["out_obj_ids"]      # (N,)
                        
                        if len(obj_ids) > 1:
                            # Calculate area for each object
                            areas = [np.sum(mask) for mask in masks]
                            # Find index of largest area
                            largest_idx = np.argmax(areas)
                            target_obj_id = obj_ids[largest_idx]
                            
                            print(f"  🔍 Detected {len(obj_ids)} objects:")
                            for i, obj_id in enumerate(obj_ids):
                                is_target = "✅ (Main)" if i == largest_idx else "❌ (Ignored)"
                                print(f"    - ID {obj_id}: Area={areas[i]} {is_target}")

                            print(f"  🎯 Keeping Main Car (ID {target_obj_id}) for OUTPUT (Tracking all internal)")
                            
                            # Do NOT remove objects from session to maintain tracking stability
                        elif len(obj_ids) == 1:
                            print(f"  🎯 Found 1 object (ID {obj_ids[0]}). Keeping it.")
                            target_obj_id = obj_ids[0]
                        else:
                            print(f"  ⚠️  No objects detected in frame 0.")
                    
                    # 4. Propagate
                    total_frames = len(frames)
                    output_generator = predictor.handle_stream_request({
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "start_frame_idx": 0,
                    })

                    # 5. Process propagation results
                    for frame_out in tqdm(output_generator, desc="propagate_in_video", total=total_frames):
                        temp_frame_idx = frame_out["frame_index"]
                        outputs = frame_out["outputs"]

                         # If we have a target object, filter the output
                        if target_obj_id is not None:
                            current_obj_ids = outputs["out_obj_ids"]
                            if target_obj_id in current_obj_ids:
                                # Find index of target
                                # current_obj_ids is a numpy array or list
                                idx = np.where(current_obj_ids == target_obj_id)[0][0]
                                
                                # Filter masks to just this object
                                outputs["out_binary_masks"] = outputs["out_binary_masks"][idx : idx+1]
                                # We don't strictly need to update obj_ids/probs for saving, just the mask
                            else:
                                # Target lost in this frame
                                outputs["out_binary_masks"] = np.zeros((0, outputs["out_binary_masks"].shape[1], outputs["out_binary_masks"].shape[2]), dtype=bool)

                        
                        original_path = temp_to_original_map[temp_frame_idx]
                        original_filename = os.path.basename(original_path)
                        
                        # Extract mask
                        # outputs["out_binary_masks"] is (N, H, W) bool, N=number of objects
                        masks = outputs["out_binary_masks"]
                        
                        if len(masks) > 0:
                            # Combine masks if multiple objects found (e.g. multiple cars)
                            final_mask = np.any(masks, axis=0).astype(np.uint8) * 255
                        else:
                            final_mask = np.zeros((outputs["out_binary_masks"].shape[1], outputs["out_binary_masks"].shape[2]), dtype=np.uint8)

                        # Save Mask
                        save_name = os.path.splitext(original_filename)[0] + "_mask.png"
                        save_full_path = os.path.join(output_path, save_name)
                        Image.fromarray(final_mask).save(save_full_path)
                        
                        # Clean up memory for this frame output
                        del outputs
                        
                        stats["frames_processed"] += 1

                    # 5. Close session
                    predictor.handle_request({
                        "type": "close_session",
                        "session_id": session_id
                    })
                
                stats["videos_processed"] += 1
                
            except Exception as e:
                stats["errors"] += 1
                print(f"  ❌ Error processing video {vid_idx}: {e}")
                
            finally:
                # Force cleanup after each video
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    end_time = time.time()
    total_duration = end_time - start_time

    print("\n" + "=" * 50)
    print("📊 Final Analytics")
    print("=" * 50)
    print(f"⏱️  Total Duration:      {total_duration:.2f} seconds")
    print(f"🔢 Total Videos:        {len(video_groups)}")
    print(f"🎞️  Total Frames:        {stats['frames_processed']}")
    print(f"❌ Errors:              {stats['errors']}")
    print("=" * 50)
    print("🏁 Done!")

if __name__ == "__main__":
    main()
