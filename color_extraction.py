# Independent colour extraction module: Using traditional computer vision methods 
# 3D-to-2D camera projection + HSV colour statistics
# it directly calculates the dominant colour of each object in the scene, 
# without relying on any VLM or AI models.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np


# HSV Colour Refraction Chart

@dataclass
class ColorBand:
    # The H (hue) range is expressed in degrees from 0 to 360. 
    # S (saturation) and V (value) are expressed on a scale from 0 to 1.

    name: str
    # There may be multiple segments, such as a red line spanning the vicinity of 0 degrees
    hue_ranges: list[tuple[float, float]]


_COLOR_BANDS = [
    ColorBand("red",          [(345, 360), (0, 12)]),
    ColorBand("pink-red",     [(12, 20)]),
    ColorBand("orange",       [(20, 45)]),
    ColorBand("yellow",       [(45, 65)]),
    ColorBand("yellow-green", [(65, 90)]),
    ColorBand("green",        [(90, 165)]),
    ColorBand("cyan",         [(165, 190)]),
    ColorBand("blue",         [(190, 250)]),
    ColorBand("purple",       [(250, 290)]),
    ColorBand("pink",         [(290, 345)]),
]

# Neutral colour detection threshold
_WHITE_VALUE_THRESHOLD = 0.85
_WHITE_SATURATION_THRESHOLD = 0.12
_BLACK_VALUE_THRESHOLD = 0.20
_GRAY_SATURATION_THRESHOLD = 0.10

_BROWN_HUE_LO, _BROWN_HUE_HI = 20, 45
_BROWN_MAX_VALUE = 0.45
_BROWN_MIN_SATURATION = 0.30

_PINK_HUE_LO, _PINK_HUE_HI = 345, 30
_PINK_MIN_VALUE = 0.70
_PINK_MAX_SATURATION = 0.45


def _rgb_to_hsv_single(r: float, g: float, b: float) -> tuple[float, float, float]:
    # Converting a single RGB pixel to HSV
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    maxc, minc = max(r, g, b), min(r, g, b)
    v = maxc
    delta = maxc - minc

    if delta < 1e-6:
        return 0.0, 0.0, v

    s = delta / maxc if maxc > 0 else 0.0

    if maxc == r:
        h = 60 * (((g - b) / delta) % 6)
    elif maxc == g:
        h = 60 * (((b - r) / delta) + 2)
    else:
        h = 60 * (((r - g) / delta) + 4)

    return h, s, v


def classify_hsv_color(hue: float, saturation: float, value: float) -> str:
    # Map a single HSV value to a predefined colour term
    if value < _BLACK_VALUE_THRESHOLD:
        return "black"
    if saturation < _GRAY_SATURATION_THRESHOLD:
        return "white" if value > _WHITE_VALUE_THRESHOLD else "gray"
    if value > _WHITE_VALUE_THRESHOLD and saturation < _WHITE_SATURATION_THRESHOLD:
        return "white"

    hue_in_pink = (hue >= _PINK_HUE_LO or hue < _PINK_HUE_HI)
    if (hue_in_pink
            and value >= _PINK_MIN_VALUE
            and saturation < _PINK_MAX_SATURATION):
        return "pink"

    if (_BROWN_HUE_LO <= hue < _BROWN_HUE_HI
            and value < _BROWN_MAX_VALUE
            and saturation > _BROWN_MIN_SATURATION):
        return "brown"

    for band in _COLOR_BANDS:
        for lo, hi in band.hue_ranges:
            if lo <= hue < hi:
                return band.name

    return "unknown"


# Background colour filtering
@dataclass
class BackgroundColorRef:
    # Identify and exclude background pixels when sampling colours
    name: str
    hue: float
    saturation: float
    value: float
    hue_tolerance: float = 20.0
    sat_tolerance: float = 0.20
    val_tolerance: float = 0.20


DEFAULT_BACKGROUND_REFS = [
    BackgroundColorRef(
        "wood_table", hue=35, saturation=0.30, value=0.85,
        hue_tolerance=25, sat_tolerance=0.25, val_tolerance=0.20,
    ),
    BackgroundColorRef(
        "gray_wall", hue=0, saturation=0.05, value=0.70,
        hue_tolerance=360, sat_tolerance=0.08, val_tolerance=0.25,
    ),
]


def _is_background_pixel(
    hue: float, saturation: float, value: float, background_refs: list[BackgroundColorRef]
) -> bool:
    # Determine whether an individual HSV pixel falls within the tolerance range of any of the background reference colours
    for ref in background_refs:
        hue_diff = min(abs(hue - ref.hue), 360 - abs(hue - ref.hue))
        if (
            hue_diff <= ref.hue_tolerance
            and abs(saturation - ref.saturation) <= ref.sat_tolerance
            and abs(value - ref.value) <= ref.val_tolerance
        ):
            return True
    return False


# Pixel Window Colour Statistics
def extract_dominant_color(
    frame: np.ndarray,
    center_uv: tuple[float, float],
    window_radius: int = 15,
    background_refs: Optional[list[BackgroundColorRef]] = None,
) -> Optional[str]:
    # The dominant hue is calculated from the window surrounding a given pixel coordinate 
    # in the image and converted into a colour term.
    if background_refs is None:
        background_refs = DEFAULT_BACKGROUND_REFS

    h, w = frame.shape[:2]
    u, v = center_uv

    x0 = max(0, int(round(u - window_radius)))
    x1 = min(w, int(round(u + window_radius)))
    y0 = max(0, int(round(v - window_radius)))
    y1 = min(h, int(round(v + window_radius)))

    if x1 <= x0 or y1 <= y0:
        return None

    window = frame[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
    if window.shape[0] == 0:
        return None

    # Convert each pixel within the window to HSV; first filter out the pixels that match the background reference colour, 
    # then perform colour classification voting on the remaining pixels.
    votes: dict[str, int] = {}
    total_saturation = 0.0
    total_value = 0.0
    kept_count = 0
    for px in window:
        h_, s_, v_ = _rgb_to_hsv_single(*px)
        if _is_background_pixel(h_, s_, v_, background_refs):
            continue
        color_name = classify_hsv_color(h_, s_, v_)
        votes[color_name] = votes.get(color_name, 0) + 1
        total_saturation += s_
        total_value += v_
        kept_count += 1

    if not votes:
        # All pixels within the window are treated as background
        return None

    best_color = max(votes.items(), key=lambda kv: kv[1])[0]

    if best_color == "orange":
        yellow_votes = votes.get("yellow", 0)
        if kept_count > 0 and yellow_votes / kept_count >= 0.25:
            best_color = "yellow"

    if best_color == "yellow":
        yg_votes = votes.get("yellow-green", 0)
        if kept_count > 0 and yg_votes / kept_count >= 0.25:
            best_color = "yellow-green"

    avg_value = total_value / kept_count
    avg_saturation = total_saturation / kept_count

    # These colour terms already convey connotations of lightness and darkness in themselves
    if best_color in ("white", "black", "gray", "unknown", "brown", "pink"):
        return best_color

    if avg_value < 0.55:
        modifier = "dark"
    elif avg_value > 0.75 and avg_saturation < 0.5:
        modifier = "pale"
    elif avg_value > 0.75:
        modifier = "bright"
    else:
        modifier = None

    result = f"{modifier} {best_color}" if modifier else best_color

    # Simplify compound words to expressions that are more intuitive for users
    _remap = {
        "dark yellow-green": "dark green",
        "pale yellow-green": "light green",
        "bright yellow-green": "green",
        "dark yellow": "olive green",
    }
    return _remap.get(result, result)


# Projection of world coordinates onto the pixel coordinates of a specified camera
def world_to_pixel(
    physics, camera_name: str, world_pos: np.ndarray, height: int = 480, width: int = 480
) -> Optional[tuple[float, float]]:
    model = physics.model
    data = physics.data

    cam_id = model.camera(camera_name).id

    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].copy().reshape(3, 3)

    rel_pos = world_pos - cam_pos
    cam_frame_pos = cam_mat.T @ rel_pos  # (x_cam, y_cam, z_cam)

    z_cam = cam_frame_pos[2]
    if z_cam >= -1e-8:
        return None  # Behind the camera or exactly on the focal plane; cannot be projected

    fovy_deg = model.camera(camera_name).fovy[0] if hasattr(model.camera(camera_name), "fovy") else model.cam_fovy[cam_id]
    fovy_rad = np.deg2rad(fovy_deg)
    focal_px = (height / 2.0) / np.tan(fovy_rad / 2.0)

    x_screen = -focal_px * cam_frame_pos[0] / z_cam
    y_screen = -focal_px * cam_frame_pos[1] / z_cam

    u = width / 2.0 + x_screen
    v = height / 2.0 - y_screen

    return u, v


def extract_colors_for_scene(
    physics,
    frame: np.ndarray,
    object_world_positions: dict[str, np.ndarray],
    camera_name: str = "forward",
    window_radius: int = 15,
    background_refs: Optional[list[BackgroundColorRef]] = None,
    height: int = 480,
    width: int = 480,
) -> dict[str, str]:
    # Batch projection and colour extraction for multiple objects in a scene
    results: dict[str, str] = {}
    for obj_name, world_pos in object_world_positions.items():
        pixel = world_to_pixel(physics, camera_name, world_pos, height, width)
        if pixel is None:
            continue
        color = extract_dominant_color(frame, pixel, window_radius, background_refs)
        if color:
            results[obj_name] = color
    return results


# Test
def _selftest_hsv_logic():
    test_cases = [
        ((220, 20, 20), "red"),
        ((255, 220, 30), "yellow"),
        ((40, 160, 40), "green"),
        ((230, 230, 235), "white"),
        ((20, 20, 25), "black"),
        ((255, 140, 0), "orange"),
        ((200, 200, 100), "yellow"),
    ]

    for (r, g, b), expected_keyword in test_cases:
        h, s, v = _rgb_to_hsv_single(r, g, b)
        result = classify_hsv_color(h, s, v)
        status = "" if expected_keyword in result else ""
        print(f" {status} RGB({r},{g},{b}) -> HSV(h={h:.0f},s={s:.2f},v={v:.2f}) -> '{result}' (Expects to contain'{expected_keyword}')")

    print("\nWindow Colour Statistics")
    h, w = 480, 480
    frame = np.full((h, w, 3), (240, 240, 245), dtype=np.uint8)
    frame[200:260, 200:260] = (200, 30, 30)

    color = extract_dominant_color(frame, center_uv=(230, 230), window_radius=15)
    print(f" The window centre is within the red square: Extract colour='{color}'")
    assert color is not None and "red" in color
    print(" The window colour statistics correctly identified red")

    print("\nFinish test\n")


def _selftest_with_image(image_path: str, pixel_coords: list[tuple[float, float]], window_radius: int = 15):
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    frame = np.array(img)
    print(f" Loaded: {image_path}  Size={img.size}\n")

    for i, (u, v) in enumerate(pixel_coords, 1):
        color_no_filter = extract_dominant_color(frame, (u, v), window_radius, background_refs=[])
        color_with_filter = extract_dominant_color(frame, (u, v), window_radius)
        marker = " <- Diff" if color_no_filter != color_with_filter else ""
        print(f" Position #{i} ({u:.0f}, {v:.0f}):")
        print(f" Turn off background filtering: '{color_no_filter}'")
        print(f" Enable background filtering: '{color_with_filter}'{marker}")

    print("\nFinish test\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="color_extraction Test")
    parser.add_argument("--selftest", choices=["hsv", "image"], default="hsv")
    parser.add_argument("--image", default=None, help="(For --selftest image only) Image path")
    parser.add_argument(
        "--positions",
        nargs="+",
        default=None,
        help='(--selftest image only) A list of pixel coordinates in the format ‘u,v’, for example --positions "239,292" "104,327"',
    )
    parser.add_argument("--window-radius", type=int, default=15)
    args = parser.parse_args()

    if args.selftest == "hsv":
        _selftest_hsv_logic()
    elif args.selftest == "image":
        if not args.image or not args.positions:
            print("False: --selftest image need --image and --positions")
            raise SystemExit(1)
        coords = []
        for p in args.positions:
            u_str, v_str = p.split(",")
            coords.append((float(u_str), float(v_str)))
        _selftest_with_image(args.image, coords, args.window_radius)