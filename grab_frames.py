# -*- coding: utf-8 -*-
"""
=====================================================================
Standalone frame-grabbing script: start one VLABenchEnv, reset a scene, save
all three camera views (image / second_image / wrist_image) as PNGs, then exit.

Purpose:
  - Get real test screenshots without running the full Gradio UI main loop.
  - Save all three camera keys so you can visually compare which one is the main view.
  - Fully standalone; does not modify or affect test_sim_env.py.

Usage:
    cd /home/aijia/Desktop/DISS_Aijia_Wang
    python grab_frames.py

    # Optional: output directory (default ./debug_frames)
    python grab_frames.py --out debug_frames

    # Optional: random seed to reproduce a specific scene layout
    python grab_frames.py --seed 42

After running, debug_frames/ will contain:
    image.png
    second_image.png
    wrist_image.png
Each image also has a text label overlaid to show which camera it is.
=====================================================================
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Same path config as test_sim_env.py so it finds the same VLABench resources
sys.path.append("/home/aijia/Desktop/DISS_Aijia_Wang/rrt-algorithms")
os.environ["VLABENCH_ROOT"] = "/home/aijia/Desktop/DISS_Aijia_Wang/VLABench/VLABench"


def _save_labeled_frame(frame: np.ndarray, label: str, save_path: str):
    """Save a frame and overlay a text label (top-left) to identify the camera."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame.astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # Try a larger font; fall back to the default if unavailable
    font = None
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, 22)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    # Semi-transparent black bar so white text stays readable on any background
    text = f"  {label}  "
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = len(text) * 12, 22

    draw.rectangle([0, 0, text_w + 10, text_h + 14], fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 0), font=font)

    img.save(save_path)


def main():
    parser = argparse.ArgumentParser(description="Grab the three camera views of a VLABench simulation scene")
    parser.add_argument("--out", default="debug_frames", help="Output directory (default debug_frames)")
    parser.add_argument("--seed", type=int, default=42, help="Scene random seed (default 42, same as test_sim_env.py)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Importing VLABenchEnv (this loads VLABench/MuJoCo dependencies)...")
    from lerobot.envs.vlabench import VLABenchEnv

    print("Starting the simulation environment...")
    env = VLABenchEnv(render_mode="human")

    print(f"Resetting the scene (seed={args.seed})...")
    try:
        observation, info = env.reset(seed=args.seed)
    except TypeError:
        observation, info = env.reset()

    pixels = observation.get("pixels", {})
    if not pixels:
        print("[Error] observation has no 'pixels' field; cannot grab frames.")
        print(f"   observation keys: {list(observation.keys())}")
        sys.exit(1)

    print(f"\nDetected camera keys: {list(pixels.keys())}\n")

    saved_paths = []
    for cam_key, frame in pixels.items():
        save_path = os.path.join(args.out, f"{cam_key}.png")
        _save_labeled_frame(frame, cam_key, save_path)
        saved_paths.append(save_path)
        print(f"  Saved: {save_path}  (shape={frame.shape})")

    print(f"\nDone. Saved {len(saved_paths)} images to '{args.out}/'.")
    print("You can now test visual_perception.py with these images, e.g.:")
    print(
        f"  python visual_perception.py --selftest images "
        f"--images {args.out}/wrist_image.png {args.out}/image.png {args.out}/second_image.png "
        f"--hints apple banana pear plate"
    )

    env.close()


if __name__ == "__main__":
    main()