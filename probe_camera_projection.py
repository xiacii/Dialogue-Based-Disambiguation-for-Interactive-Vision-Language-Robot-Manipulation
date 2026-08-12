# -*- coding: utf-8 -*-
"""
=====================================================================
Diagnostic script: verifies whether we can use dm_control's camera
projection matrix to map each object's 3D world coordinates
(site_xpos) into 2D pixel coordinates, and thus obtain a bounding box.
This is a prerequisite for the "compute colors directly with classical
CV" approach, since the CV method needs to know each object's pixel
region in the image to sample its color.

Principle: dm_control's Camera object exposes a `.matrix` attribute
(a 3x4 projection matrix) that maps homogeneous world coordinates
[x, y, z, 1] into homogeneous pixel coordinates [u*w, v*w, w].

Usage:
    cd /home/aijia/Desktop/DISS_Aijia_Wang
    python probe_camera_projection.py

Outputs:
  - Each object's world coordinates (from site_xpos)
  - The projected pixel coordinates (u, v) in the wrist_image camera
  - An annotated image with projected points + a sample bounding box
    so you can visually verify the projection accuracy.
=====================================================================
"""

import os
import sys

import numpy as np

sys.path.append("/home/aijia/Desktop/DISS_Aijia_Wang/rrt-algorithms")
os.environ["VLABENCH_ROOT"] = "/home/aijia/Desktop/DISS_Aijia_Wang/VLABench/VLABench"


def world_to_pixel(physics, camera_name: str, world_pos: np.ndarray, render_resolution: tuple[int, int]):
    """Project world coordinates onto the specified camera's pixel plane.

    Args:
        physics: dm_control physics object
        camera_name: camera name as defined in the scene XML
        world_pos: (3,) world coordinate
        render_resolution: (height, width) render resolution, must match actual rendering

    Returns:
        (u, v): pixel coordinates (u=horizontal, v=vertical); may be
        negative/invalid if the object is behind the camera.
    """
    from dm_control.mujoco.engine import Camera

    h, w = render_resolution
    camera = Camera(physics, height=h, width=w, camera_id=camera_name)

    # camera.matrix is a 3x4 projection matrix: pixel_homogeneous = matrix @ [x, y, z, 1]
    proj_matrix = camera.matrix  # shape (3, 4)

    world_homogeneous = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0])
    pixel_homogeneous = proj_matrix @ world_homogeneous  # shape (3,)

    if abs(pixel_homogeneous[2]) < 1e-8:
        return None  # projection failed (division by zero risk)

    u = pixel_homogeneous[0] / pixel_homogeneous[2]
    v = pixel_homogeneous[1] / pixel_homogeneous[2]
    return u, v


def main():
    print("Importing VLABenchEnv...")
    from lerobot.envs.vlabench import VLABenchEnv

    print("Starting simulation and resetting scene (seed=42, matches earlier frame-grab script)...")
    env = VLABenchEnv(render_mode="human")
    try:
        observation, info = env.reset(seed=42)
    except TypeError:
        observation, info = env.reset()

    physics = env._env.physics
    render_resolution = (480, 480)  # must match actual render resolution

    # ---- Step 1: list all available camera names in the scene ----
    print("\n=== [DEBUG] Cameras defined in the scene ===")
    try:
        camera_names = list(physics.named.data.cam_xpos.axes.row.names)
        print(camera_names)
    except Exception as e:
        print(f"Failed to list camera names: {e}")
        camera_names = []

    # ---- Step 2: list object site coordinates ----
    print("\n=== [DEBUG] Object site coordinates (world frame) ===")
    site_names = list(physics.named.data.site_xpos.axes.row.names)
    object_sites = {}
    for name in site_names:
        if name.endswith("/bottom_site") and "plate" not in name.lower():
            obj_name = name.split("/")[0]
            pos = physics.named.data.site_xpos[name].copy()
            object_sites[obj_name] = pos
            print(f"  {obj_name}: world_pos={np.round(pos, 4)}")

    if not object_sites:
        print("No object bottom_site found; cannot proceed with projection test.")
        env.close()
        return

    # ---- Step 3: try projection with each candidate camera name ----
    # Common naming guesses: based on VLABenchEnv's image_keys ordering, camera names
    # may be 'image' / 'second_image' / 'wrist_image', or prefixed variants.
    # We try all detected camera names and see which one yields pixel coordinates
    # within [0, width] x [0, height].
    candidate_camera_names = camera_names if camera_names else [
        "image", "second_image", "wrist_image",
        "front", "side", "wrist",
    ]

    print(f"\n=== Trying projection test for each candidate camera ===")
    working_camera = None
    for cam_name in candidate_camera_names:
        print(f"\n--- Camera: '{cam_name}' ---")
        try:
            success_count = 0
            for obj_name, world_pos in object_sites.items():
                result = world_to_pixel(physics, cam_name, world_pos, render_resolution)
                if result is None:
                    print(f"  {obj_name}: projection failed (division by zero)")
                    continue
                u, v = result
                in_bounds = 0 <= u <= render_resolution[1] and 0 <= v <= render_resolution[0]
                status = "OK" if in_bounds else "out of frame"
                print(f"  {obj_name}: pixel=({u:.1f}, {v:.1f}) {status}")
                if in_bounds:
                    success_count += 1

            if success_count == len(object_sites):
                working_camera = cam_name
                print(f"  -> All object projections are within frame; this is the correct camera name")
        except Exception as e:
            print(f"  Call failed: {type(e).__name__}: {e}")

    # ---- Step 4: if a working camera was found, render its real view and annotate ----
    # Important fix: previously this section guessed the frame from observation["pixels"]
    # (keys like "wrist_image"/"image"), but those belong to a different camera-naming
    # scheme and may not correspond to working_camera (the true MuJoCo camera name,
    # e.g. 'forward'). We must render working_camera directly via dm_control to guarantee
    # the image content and the projected coordinates share the exact same viewpoint.
    if working_camera:
        print(f"\nFound working camera: '{working_camera}', rendering its real view...")
        try:
            from dm_control.mujoco.engine import Camera
            from PIL import Image, ImageDraw

            render_camera = Camera(
                physics,
                height=render_resolution[0],
                width=render_resolution[1],
                camera_id=working_camera,
            )
            base_frame = render_camera.render()  # (H, W, 3) uint8, matches working_camera exactly

            os.makedirs("debug_frames", exist_ok=True)

            # Save an unannotated raw frame for later color-extraction tests, so the image
            # and the coordinates from world_to_pixel() come from the same camera with
            # no risk of misalignment.
            raw_path = f"debug_frames/{working_camera.replace('/', '_')}_raw.png"
            Image.fromarray(base_frame.astype(np.uint8)).save(raw_path)
            print(f"Saved unannotated raw frame: {raw_path} (use this for color extraction tests)")

            # Also produce an annotated version for visual verification.
            img = Image.fromarray(base_frame.astype(np.uint8))
            draw = ImageDraw.Draw(img)
            for obj_name, world_pos in object_sites.items():
                result = world_to_pixel(physics, working_camera, world_pos, render_resolution)
                if result is None:
                    continue
                u, v = result
                r = 15  # sample bounding-box radius (pixels); calibrate per actual object size later
                draw.ellipse([u - r, v - r, u + r, v + r], outline=(255, 0, 0), width=3)
                draw.text((u + r + 2, v - r), obj_name, fill=(255, 0, 0))

            out_path = "debug_frames/projection_test.png"
            img.save(out_path)
            print(f"Saved visualization: {out_path}")
            print("   Open this image and check that the red circles line up with each object.")
        except Exception as e:
            print(f"Visualization failed: {type(e).__name__}: {e}")
    else:
        print("\nNo camera name places all objects within the frame.")
        print("   Possible causes: camera naming doesn't match guesses, or the projection")
        print("   matrix needs to be obtained differently. Please share the full output for diagnosis.")

    env.close()


if __name__ == "__main__":
    main()