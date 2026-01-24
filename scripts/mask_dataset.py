import argparse
import gc
import glob
import os
import re
import shutil
import tempfile
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_video_predictor


def parse_video_info(filename):
    basename = os.path.basename(filename)
    # Regex to match frame_*_video_*.ext
    match = re.search(r"frame_(\d+)_video_(\d+)", basename)
    if match:
        frame_idx = int(match.group(1))
        video_idx = int(match.group(2))
        return video_idx, frame_idx, filename
    return None


def compute_iou(mask1, mask2):
    """Computes Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def main():
    parser = argparse.ArgumentParser(
        description="Mask cars in a dataset using SAM3 Video Mode (Chunked)"
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
        "--prompt", type=str, default="car", help="Text prompt for segmentation"
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
        "--cpu-offload",
        action="store_true",
        help="Offload video frames to CPU to save GPU memory",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=60,
        help="Number of frames to process at once (lower this if OOM occurs)",
    )
    args = parser.parse_args()

    print(f"🚀 SAM3 Video Masking | Chunk Size: {args.chunk_size}")

    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path '{dataset_path}' does not exist.")
        return

    # Create output directory
    output_path = args.output_dir
    if not os.path.isabs(output_path):
        output_path = os.path.join(dataset_path, args.output_dir)

    os.makedirs(output_path, exist_ok=True)

    # Create visualization directory
    if args.viz:
        os.makedirs(os.path.join(output_path, "viz"), exist_ok=True)

    # 1. Setup
    print("📦 Loading SAM3 Video model...", end="", flush=True)
    try:
        gpus_to_use = (
            range(torch.cuda.device_count()) if torch.cuda.is_available() else []
        )
        predictor = build_sam3_video_predictor(
            gpus_to_use=gpus_to_use,
            checkpoint_path=args.checkpoint,
            offload_video_to_cpu=args.cpu_offload,
        )
        print(" Done!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("\n🔍 Scanning and grouping images...")
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(dataset_path, ext)))
        image_files.extend(glob.glob(os.path.join(dataset_path, ext.upper())))

    video_groups = {}  # video_idx -> [(frame_idx, filepath)]

    for img_path in image_files:
        info = parse_video_info(img_path)
        if info:
            vid_idx, frame_idx, path = info
            if vid_idx not in video_groups:
                video_groups[vid_idx] = []
            video_groups[vid_idx].append((frame_idx, path))

    if not video_groups:
        print(f"❌ No valid video frames found in {dataset_path}")
        return

    # Sort frames within each video
    for vid_idx in video_groups:
        video_groups[vid_idx].sort(key=lambda x: x[0])

    print(f"🔍 Found {len(video_groups)} videos. Starting processing...")

    stats = {"videos_processed": 0, "frames_processed": 0, "errors": 0}

    start_time = time.time()

    # Process each video
    for vid_idx, frames in tqdm(video_groups.items(), desc="Processing Videos 🎥"):
        # State tracking for continuity between chunks
        previous_chunk_last_mask = None

        # Calculate chunks
        total_frames = len(frames)
        chunk_size = args.chunk_size
        num_chunks = (total_frames + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_frames)
            current_chunk_frames = frames[start_idx:end_idx]

            if not current_chunk_frames:
                continue

            # Force cleanup before every chunk
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with tempfile.TemporaryDirectory() as temp_video_dir:
                try:
                    # 1. Prepare symlinks for this chunk
                    temp_to_original_map = {}

                    for i, (original_frame_idx, original_path) in enumerate(
                        current_chunk_frames
                    ):
                        ext = os.path.splitext(original_path)[1]
                        symlink_name = f"{i:05d}{ext}"
                        symlink_path = os.path.join(temp_video_dir, symlink_name)
                        os.symlink(original_path, symlink_path)
                        temp_to_original_map[i] = original_path

                    # 2. Start Session
                    with torch.inference_mode():
                        response = predictor.handle_request(
                            {"type": "start_session", "resource_path": temp_video_dir}
                        )
                        session_id = response["session_id"]

                        # 3. Add text prompt to frame 0 of THIS chunk
                        prompt_response = predictor.handle_request(
                            {
                                "type": "add_prompt",
                                "session_id": session_id,
                                "frame_index": 0,
                                "text": args.prompt,
                            }
                        )

                        target_obj_id = None

                        if (
                            "outputs" in prompt_response
                            and "out_binary_masks" in prompt_response["outputs"]
                        ):
                            masks = prompt_response["outputs"]["out_binary_masks"]
                            obj_ids = prompt_response["outputs"]["out_obj_ids"]

                            if len(obj_ids) > 0:
                                if chunk_idx == 0:
                                    # First chunk: Pick largest object
                                    areas = [np.sum(mask) for mask in masks]
                                    target_obj_id = obj_ids[np.argmax(areas)]
                                else:
                                    # Subsequent chunks: Pick object matching previous mask
                                    if previous_chunk_last_mask is not None:
                                        best_iou = -1
                                        best_id = None

                                        for i, obj_id in enumerate(obj_ids):
                                            iou = compute_iou(
                                                masks[i], previous_chunk_last_mask
                                            )
                                            if iou > best_iou:
                                                best_iou = iou
                                                best_id = obj_id

                                        # If overlap is decent, keep it. Else fallback to largest.
                                        if best_iou > 0.01:
                                            target_obj_id = best_id
                                        else:
                                            areas = [np.sum(mask) for mask in masks]
                                            target_obj_id = obj_ids[np.argmax(areas)]
                                    else:
                                        # Fallback
                                        areas = [np.sum(mask) for mask in masks]
                                        target_obj_id = obj_ids[np.argmax(areas)]

                        # 4. Propagate
                        output_generator = predictor.handle_stream_request(
                            {
                                "type": "propagate_in_video",
                                "session_id": session_id,
                                "start_frame_idx": 0,
                            }
                        )

                        chunk_last_mask = None

                        # 5. Process results
                        for frame_out in output_generator:
                            temp_idx = frame_out["frame_index"]
                            outputs = frame_out["outputs"]

                            final_mask = None

                            # Filter for the specific object ID
                            if (
                                target_obj_id is not None
                                and target_obj_id in outputs["out_obj_ids"]
                            ):
                                idx = np.where(outputs["out_obj_ids"] == target_obj_id)[
                                    0
                                ][0]
                                raw_mask = outputs["out_binary_masks"][idx]
                                final_mask = raw_mask.astype(np.uint8) * 255
                                chunk_last_mask = raw_mask  # Save for continuity
                            else:
                                shape = outputs["out_binary_masks"].shape
                                final_mask = np.zeros(
                                    (shape[1], shape[2]), dtype=np.uint8
                                )
                                chunk_last_mask = None

                            # Save Mask
                            original_path = temp_to_original_map[temp_idx]
                            save_name = (
                                os.path.splitext(os.path.basename(original_path))[0]
                                + "_mask.png"
                            )
                            save_full_path = os.path.join(output_path, save_name)
                            Image.fromarray(final_mask).save(save_full_path)

                            # Save Viz (Optional)
                            if args.viz:
                                # Save as cutout with transparency
                                viz_name = (
                                    os.path.splitext(os.path.basename(original_path))[0]
                                    + "_cutout.png"
                                )
                                viz_full_path = os.path.join(
                                    output_path, "viz", viz_name
                                )
                                img = Image.open(original_path).convert("RGBA")
                                mask_img = Image.fromarray(final_mask).convert("L")
                                img.putalpha(mask_img)
                                img.save(viz_full_path)

                            stats["frames_processed"] += 1

                        # Store mask for next chunk
                        previous_chunk_last_mask = chunk_last_mask

                        # 5. Close session immediately to free VRAM
                        predictor.handle_request(
                            {"type": "close_session", "session_id": session_id}
                        )

                except Exception as e:
                    stats["errors"] += 1
                    print(
                        f"  ❌ Error processing video {vid_idx} (chunk {chunk_idx}): {e}"
                    )
                    break  # Skip rest of this video on error

        stats["videos_processed"] += 1

    end_time = time.time()
    total_duration = end_time - start_time

    print("\n" + "=" * 30)
    print(
        f"🏁 Done! {stats['frames_processed']} frames from {len(video_groups)} videos in {total_duration:.1f}s. Errors: {stats['errors']}"
    )
    print("=" * 30)


if __name__ == "__main__":
    main()
