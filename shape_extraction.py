# -*- coding: utf-8 -*-
"""
shape_extraction.py
=====================================================================
Standalone shape extraction module, parallel in design to
color_extraction.py. After each reset, it computes an extra "shape"
feature (round / oval / elongated / flat / ...) for every object on
the table, without relying on any VLM/AI model.

Core idea (dual signal, geometry primary, 2D silhouette secondary):

  1) Geometric signal (primary, most reliable):
     VLABench attaches three standard sites to every object:
         {obj}/bottom_site        object bottom
         {obj}/top_site           object top
         {obj}/horizontal_radius_site   a point at the horizontal radius
     From these we read the object's true 3D size:
         height  = |top.z - bottom.z|
         radius  = ||radius_site.xy - center.xy||
         aspect  = height / (2 * radius)
     aspect ~= 1 -> round; aspect >> 1 -> oval; aspect very small -> flat.
     Pure coordinate reading, no rendering or projection; most stable.

  2) 2D silhouette signal (secondary, fallback for cases where the
     geometric assumption breaks):
     horizontal_radius_site assumes horizontal circular symmetry, but
     long curved objects like bananas violate this — their "radius"
     gets computed as roughly half the length, so the aspect ratio
     comes out too small and gets misclassified as flat/round. We
     reuse the color module's camera projection to project the object
     onto the pixel plane, take a window around the projected point,
     mask out the background, and run PCA on the silhouette pixels to
     get elongation = sqrt(lambda1/lambda2). When elongation is clearly
     high, we override the geometric verdict with "elongated".

This module fully reuses color_extraction.py's camera projection
(world_to_pixel) and background filtering (_is_background_pixel /
_rgb_to_hsv_single), so shape and color always use the same camera
and the same background calibration — no coordinate misalignment.

Standalone test:
    python shape_extraction.py --selftest synthetic
=====================================================================
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Reuse the camera projection and background filtering already validated in the
# color module, avoiding duplication and ensuring shape/color features use the
# exact same projection and background calibration.
from color_extraction import (
    world_to_pixel,
    DEFAULT_BACKGROUND_REFS,
    BackgroundColorRef,
    _is_background_pixel,
    _rgb_to_hsv_single,
)


# ======================================================================
# Shape classification thresholds (centralized here for easy tuning,
# like the ColorBand thresholds in the color module)
# ======================================================================

@dataclass
class ShapeThresholds:
    """All thresholds used for shape classification, centralized for tuning.

    Geometric aspect = height / (2 * radius):
        aspect < flat_max              -> flat   (plates, disks)
        round_lo <= aspect <= round_hi -> round  (apple, orange, peach)
        round_hi < aspect <= oval_hi   -> oval   (pear, mango, lemon)
        aspect > oval_hi               -> elongated (geometrically slender)

    2D silhouette elongation = sqrt(major-axis variance / minor-axis variance):
        elongation >= elong_2d_thresh  -> force elongated (overrides geometry
                                          verdict, covers curved-slender
                                          objects like bananas where the
                                          horizontal-radius assumption fails)
    """

    flat_max: float       = 0.45    # aspect below this -> flat
    round_lo: float       = 0.70    # lower bound of round range
    round_hi: float       = 1.20    # upper bound of round range
    oval_hi: float        = 2.20    # upper bound of oval range; above -> elongated
    elong_2d_thresh: float = 2.30   # 2D silhouette elongation threshold; above -> elongated
    # Pure-2D fallback thresholds (used when the object has no geometric sites):
    round_2d_max_elong: float = 1.45  # silhouette near-circular -> round
    round_2d_min_extent: float = 0.62 # and fill ratio (silhouette / bbox) high enough

    # ---- Geom AABB three-axis half-size ratio thresholds (primary, most reliable) ----
    # Sort a geom's three half-sizes in descending order [s0>=s1>=s2] and look at ratios:
    #   r1 = s1/s0 (second-longest / longest), r2 = s2/s0 (shortest / longest)
    # These ratios are independent of the object's orientation on the table and can
    # cleanly distinguish a "horizontally lying stick (banana)" from a "flat disk" —
    # something the site-based aspect ratio cannot do.
    ext_rod_r1_max: float   = 0.55  # second axis much smaller than the first -> elongated
    ext_flat_r1_min: float  = 0.70  # first two axes close ...
    ext_flat_r2_max: float  = 0.45  # ... but the third is clearly small -> flat (disk)
    ext_round_r1_min: float = 0.78  # all three axes close -> round (sphere)
    ext_round_r2_min: float = 0.62


DEFAULT_SHAPE_THRESHOLDS = ShapeThresholds()


# ======================================================================
# Shape classification core
# ======================================================================

def classify_shape(
    height: Optional[float],
    radius: Optional[float],
    elongation_2d: Optional[float] = None,
    extent_2d: Optional[float] = None,
    thr: ShapeThresholds = DEFAULT_SHAPE_THRESHOLDS,
) -> str:
    """Map (geometric size + 2D silhouette features) to a shape word.

    Args:
        height:        object vertical height (meters); pass None if no geometric sites
        radius:        object horizontal radius (meters); pass None if no geometric sites
        elongation_2d: 2D silhouette elongation sqrt(lambda1/lambda2); pass None if no image signal
        extent_2d:     2D silhouette fill ratio = silhouette area / bbox area; may be None
        thr:           threshold table

    Returns:
        Shape word: "round" / "oval" / "elongated" / "flat" / "unknown"
    """
    # 2D elongation takes precedence: an obviously elongated silhouette (e.g. banana)
    # is classified as "elongated" directly, not misled by the horizontal-symmetry assumption.
    if elongation_2d is not None and elongation_2d >= thr.elong_2d_thresh:
        return "elongated"

    # Have geometric size: classify by aspect ratio (most reliable)
    if height is not None and radius is not None and radius > 1e-6:
        aspect = height / (2.0 * radius)
        if aspect < thr.flat_max:
            return "flat"
        if aspect <= thr.round_hi:
            # Aspect below round_lo (flatter than a sphere but not yet flat) still looks
            # roughly round, so we group it into round to avoid mislabelling slightly
            # flat apples.
            return "round"
        if aspect <= thr.oval_hi:
            return "oval"
        return "elongated"

    # No geometric sites: fall back to pure 2D silhouette
    if elongation_2d is not None:
        if elongation_2d >= thr.elong_2d_thresh:
            return "elongated"
        if (elongation_2d <= thr.round_2d_max_elong
                and (extent_2d is None or extent_2d >= thr.round_2d_min_extent)):
            return "round"
        return "oval"

    return "unknown"


# ======================================================================
# Geometric signal: read 3D size from the three standard sites
# ======================================================================

def read_object_geometry(physics, obj_name: str):
    """Read (height, radius, center) from {obj}/top_site, {obj}/bottom_site,
    and {obj}/horizontal_radius_site.

    All three sites must exist; otherwise returns None (caller falls back to
    pure 2D silhouette classification).

    Returns:
        (height, radius, center_xyz) or None
    """
    try:
        site_xpos = physics.named.data.site_xpos
        available = set(site_xpos.axes.row.names)

        bottom = f"{obj_name}/bottom_site"
        top = f"{obj_name}/top_site"
        rad = f"{obj_name}/horizontal_radius_site"
        if not (bottom in available and top in available and rad in available):
            return None

        bp = site_xpos[bottom].copy().astype(np.float64)
        tp = site_xpos[top].copy().astype(np.float64)
        rp = site_xpos[rad].copy().astype(np.float64)

        center = (bp + tp) / 2.0
        height = float(abs(tp[2] - bp[2]))
        radius = float(np.linalg.norm(rp[:2] - center[:2]))
        # Radius degeneracy guard: if radius_site coincides with the center xy, give a
        # tiny positive value to avoid division by zero when computing aspect ratio.
        if radius < 1e-4:
            radius = 1e-4
        return height, radius, center
    except Exception:
        return None


def read_object_center(physics, obj_name: str):
    """Return the object's world-frame center xyz, or None if sites unavailable.

    Thin wrapper around read_object_geometry for callers that only need the center.
    """
    geom = read_object_geometry(physics, obj_name)
    return None if geom is None else geom[2]


def read_object_geom_extents(physics, obj_name: str):
    """Read the local AABB of all geoms belonging to the object, merge them, and return
    the three half-sizes sorted in descending order [s0>=s1>=s2] (meters). This signal
    is independent of camera and object orientation, and can distinguish a "horizontally
    lying stick" from a "flat disk" — something a single horizontal_radius_site cannot.

    Implementation: iterate all geoms in the model, pick those whose body name starts
    with obj_name + '/', read the last three entries of model.geom_aabb (half-sizes in
    the local frame), take the per-axis max across multiple geoms (a rough union), then
    return the three half-sizes sorted in descending order.

    Returns:
        np.ndarray([s0, s1, s2]) or None (falls back gracefully if geom_aabb unavailable).
    """
    try:
        model = physics.model

        # body name -> id: xpos row-name order equals body id order (consistent throughout code)
        body_names = list(physics.named.data.xpos.axes.row.names)
        target_body_ids = set()
        for bid, bname in enumerate(body_names):
            if bname == obj_name + "/" or bname.startswith(obj_name + "/"):
                target_body_ids.add(bid)
        if not target_body_ids:
            return None

        aabb = np.asarray(model.geom_aabb, dtype=np.float64).reshape(-1, 6)
        geom_bodyid = np.asarray(model.geom_bodyid).reshape(-1)

        halves = []
        for gid in range(int(model.ngeom)):
            if int(geom_bodyid[gid]) in target_body_ids:
                half = np.abs(aabb[gid, 3:6]).astype(np.float64)
                if np.all(half > 1e-9):
                    halves.append(half)
        if not halves:
            return None

        merged = np.max(np.stack(halves, axis=0), axis=0)  # per-axis max across geoms
        return np.sort(merged)[::-1]                        # descending [s0>=s1>=s2]
    except Exception:
        return None


def classify_from_extents(half_sizes, thr: ShapeThresholds = DEFAULT_SHAPE_THRESHOLDS) -> str:
    """Classify shape using the ratios of the geom's three half-sizes (most reliable path).

    s = [s0>=s1>=s2], r1 = s1/s0, r2 = s2/s0:
        small r1 (one axis dominates)   -> elongated (banana, rod)
        large r1 but small r2 (two big, one small) -> flat (plate, disk)
        both r1 and r2 large (all three close) -> round (sphere: apple, orange, peach)
        otherwise (moderately elongated ellipsoid) -> oval (pear, mango, lemon)
    """
    s = np.sort(np.asarray(half_sizes, dtype=np.float64))[::-1]
    s0 = max(float(s[0]), 1e-9)
    r1 = float(s[1]) / s0
    r2 = float(s[2]) / s0

    if r1 <= thr.ext_rod_r1_max:
        return "elongated"
    if r1 >= thr.ext_flat_r1_min and r2 <= thr.ext_flat_r2_max:
        return "flat"
    if r1 >= thr.ext_round_r1_min and r2 >= thr.ext_round_r2_min:
        return "round"
    return "oval"


# ======================================================================
# 2D signal: extract silhouette around the projected point + PCA elongation
# ======================================================================

def _foreground_mask(
    frame: np.ndarray,
    center_uv: tuple[float, float],
    window_radius: int,
    background_refs: list[BackgroundColorRef],
) -> Optional[tuple[np.ndarray, int, int]]:
    """In a square window around the projected point, use background filtering to
    extract a boolean foreground (object) mask.

    Returns:
        (mask, cu, cv) — mask is an (h,w) boolean foreground mask inside the window;
        cu/cv are the window center's local coordinates. Returns None if out of bounds
        or the window is empty.
    """
    h, w = frame.shape[:2]
    u, v = center_uv
    x0 = max(0, int(round(u - window_radius)))
    x1 = min(w, int(round(u + window_radius)))
    y0 = max(0, int(round(v - window_radius)))
    y1 = min(h, int(round(v + window_radius)))
    if x1 <= x0 or y1 <= y0:
        return None

    window = frame[y0:y1, x0:x1].astype(np.float64)
    # Vectorized: compute HSV + background mask for the whole window at once, avoiding
    # per-pixel Python loops. This lets us use larger windows (to fit long objects like
    # a whole banana) without slowing down the scan.
    H, S, V = _rgb_to_hsv_array(window)
    bg = _is_background_array(H, S, V, background_refs)
    mask = ~bg

    cu = int(round(u)) - x0
    cv = int(round(v)) - y0
    return mask, cu, cv


def _rgb_to_hsv_array(rgb: np.ndarray):
    """Vectorized RGB->HSV (same formula as color_extraction._rgb_to_hsv_single).

    Args:
        rgb: (..., 3) float, values in 0-255
    Returns:
        (H, S, V), H in 0-360, S/V in 0-1, shape equals input minus the last dim.
    """
    r = rgb[..., 0] / 255.0
    g = rgb[..., 1] / 255.0
    b = rgb[..., 2] / 255.0
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc

    s = np.where(maxc > 1e-12, delta / np.maximum(maxc, 1e-12), 0.0)

    has_delta = delta > 1e-6
    deltad = np.where(has_delta, delta, 1.0)  # avoid division by zero
    is_r = has_delta & (maxc == r)
    is_g = has_delta & (maxc == g) & ~is_r
    is_b = has_delta & (maxc == b) & ~is_r & ~is_g

    hpre = np.zeros_like(maxc)
    hpre = np.where(is_r, ((g - b) / deltad) % 6.0, hpre)
    hpre = np.where(is_g, ((b - r) / deltad) + 2.0, hpre)
    hpre = np.where(is_b, ((r - g) / deltad) + 4.0, hpre)
    hue = (hpre * 60.0) % 360.0
    hue = np.where(has_delta, hue, 0.0)
    return hue, s, v


def _is_background_array(H: np.ndarray, S: np.ndarray, V: np.ndarray,
                         background_refs: list[BackgroundColorRef]) -> np.ndarray:
    """Vectorized background detection (same logic as color_extraction._is_background_pixel).

    Returns a boolean array with the same shape as H; True means the pixel falls within
    some background reference color's tolerance.
    """
    bg = np.zeros(H.shape, dtype=bool)
    for ref in background_refs:
        hue_diff = np.minimum(np.abs(H - ref.hue), 360.0 - np.abs(H - ref.hue))
        m = ((hue_diff <= ref.hue_tolerance)
             & (np.abs(S - ref.saturation) <= ref.sat_tolerance)
             & (np.abs(V - ref.value) <= ref.val_tolerance))
        bg |= m
    return bg


def _connected_blob(mask: np.ndarray, seed_xy: tuple[int, int]) -> np.ndarray:
    """Starting from the seed pixel, flood-fill with 4-neighbour BFS on the foreground
    mask, keeping only the blob connected to the seed (so foreground pixels of adjacent
    objects/plates leaking into the window don't distort the shape statistics). If the
    seed itself isn't foreground, search a small nearby area for the closest foreground
    pixel and use that as the seed.

    Returns:
        Boolean mask (same shape as input) of foreground connected to the seed.
    """
    h, w = mask.shape
    cu, cv = seed_xy  # note: seed is given as (column, row)

    # Seed correction: if the center pixel isn't foreground, spiral outward for the nearest one
    if not (0 <= cv < h and 0 <= cu < w and mask[cv, cu]):
        found = None
        for rad in range(1, max(h, w)):
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    ny, nx = cv + dy, cu + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                        found = (nx, ny)
                        break
                if found:
                    break
            if found:
                break
        if found is None:
            return np.zeros_like(mask)
        cu, cv = found

    blob = np.zeros_like(mask)
    q = deque([(cv, cu)])
    blob[cv, cu] = True
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not blob[ny, nx]:
                blob[ny, nx] = True
                q.append((ny, nx))
    return blob


def silhouette_features(
    frame: np.ndarray,
    center_uv: tuple[float, float],
    window_radius: int = 60,
    background_refs: Optional[list[BackgroundColorRef]] = None,
    min_blob_pixels: int = 25,
):
    """Compute shape statistics for the silhouette at the projected point; returns
    (elongation, extent).

    elongation = sqrt(major-axis variance / minor-axis variance): ~1 means close to
    circle/square; higher means more elongated.
    extent     = silhouette pixels / bounding-box area: high for solid/round objects,
    low for thin/hollow ones.

    Returns (None, None) when the silhouette is too small (< min_blob_pixels) or no
    foreground can be extracted.
    """
    if background_refs is None:
        background_refs = DEFAULT_BACKGROUND_REFS

    fg = _foreground_mask(frame, center_uv, window_radius, background_refs)
    if fg is None:
        return None, None
    mask, cu, cv = fg

    blob = _connected_blob(mask, (cu, cv))
    coords = np.argwhere(blob)  # (N,2), each row (row_y, col_x)
    if coords.shape[0] < min_blob_pixels:
        return None, None

    ys = coords[:, 0].astype(np.float64)
    xs = coords[:, 1].astype(np.float64)
    pts = np.stack([xs, ys], axis=1)  # (N,2) -> (x,y)

    # PCA: covariance eigenvalues give the variance along the two principal axes;
    # the square root of their ratio is the elongation.
    cov = np.cov(pts.T)
    eigvals = np.linalg.eigvalsh(cov)  # ascending
    lam_min, lam_max = float(eigvals[0]), float(eigvals[-1])
    if lam_min <= 1e-9:
        elongation = float("inf")
    else:
        elongation = float(np.sqrt(lam_max / lam_min))

    # Fill ratio: silhouette area / bounding-box area
    bbox_h = ys.max() - ys.min() + 1.0
    bbox_w = xs.max() - xs.min() + 1.0
    extent = float(coords.shape[0] / (bbox_h * bbox_w))

    return elongation, extent


# ======================================================================
# Scene-level convenience wrapper
# (interface mirrors color_extraction.extract_colors_for_scene)
# ======================================================================

def extract_shapes_for_scene(
    physics,
    frame: np.ndarray,
    object_world_positions: dict[str, np.ndarray],
    camera_name: str = "forward",
    window_radius: int = 60,
    background_refs: Optional[list[BackgroundColorRef]] = None,
    height: int = 480,
    width: int = 480,
    thr: ShapeThresholds = DEFAULT_SHAPE_THRESHOLDS,
    return_details: bool = False,
) -> dict[str, str]:
    """Batch shape extraction for multiple objects in a scene.

    The interface is intentionally kept aligned with
    color_extraction.extract_colors_for_scene, so it can be called from the
    main program just like the color scan.

    Args:
        physics:                dm_control physics object
        frame:                  (H,W,3) uint8 image rendered from camera_name
                                (in the main program, observation["pixels"]["wrist_image"],
                                which actually corresponds to the 'forward' camera —
                                the same frame used for color extraction)
        object_world_positions: {object_name: world_pos}, from
                                _get_scene_object_world_positions()
        camera_name:            real MuJoCo camera name ('forward' verified to work)
        window_radius:          2D silhouette sampling window radius (pixels), larger
                                than the color window to fit the full object outline
        background_refs:        background reference colors; defaults to the color
                                module's calibration
        height, width:          render resolution; must match the actual frame size
        thr:                    shape threshold table
        return_details:         if True, values become dicts containing shape + geometric
                                + 2D intermediates (useful for tuning); default False
                                returns only the shape word

    Returns:
        {object_name: shape_word} (values are detail dicts when return_details=True)
    """
    results: dict[str, str] = {}
    for obj_name, world_pos in object_world_positions.items():
        # 1) Primary signal: geom AABB three-axis half-sizes (independent of camera and
        #    orientation, most reliable, distinguishes lying stick vs flat disk)
        extents = read_object_geom_extents(physics, obj_name)

        # 2) Backup signal: height/radius from the three sites (used with 2D when geom unavailable)
        geom = read_object_geometry(physics, obj_name)
        if geom is not None:
            g_height, g_radius, g_center = geom
        else:
            g_height, g_radius, g_center = None, None, None

        # 3) Auxiliary signal: 2D silhouette elongation (projection + connected blob + PCA).
        #    Serves as fallback when geom/sites are missing, and as an override for extreme
        #    elongation cases.
        elong, extent = None, None
        pixel = world_to_pixel(physics, camera_name, world_pos, height, width)
        if pixel is not None:
            elong, extent = silhouette_features(
                frame, pixel, window_radius, background_refs
            )

        # ---- Combined decision ----
        if extents is not None:
            # Have geom three-axis sizes: trust them (banana correctly ends up as elongated)
            shape = classify_from_extents(extents, thr)
        else:
            # Fall back to site aspect ratio + 2D
            shape = classify_shape(g_height, g_radius, elong, extent, thr)

        # 2D elongation strong override: regardless of path, an obviously elongated
        # silhouette forces "elongated" as a safety net (in case an object has weird
        # geom sizes but a clearly long silhouette).
        if elong is not None and elong >= thr.elong_2d_thresh:
            shape = "elongated"

        if return_details:
            results[obj_name] = {
                "shape": shape,
                "extents_half": (np.round(extents, 4).tolist() if extents is not None else None),
                "ext_ratios": ([round(float(extents[1] / extents[0]), 3),
                                round(float(extents[2] / extents[0]), 3)]
                               if extents is not None and extents[0] > 0 else None),
                "height": g_height,
                "radius": g_radius,
                "aspect": (g_height / (2 * g_radius)) if (g_height and g_radius) else None,
                "elongation_2d": elong,
                "extent_2d": extent,
            }
        else:
            results[obj_name] = shape

    return results


# ======================================================================
# Standalone self-test entry point
# ======================================================================

def _selftest_synthetic():
    """Synthetic silhouette test: no simulation needed; verifies 2D elongation + classification logic."""
    print("=== Self-test: synthetic silhouette PCA elongation ===")
    h, w = 480, 480

    # Use a neutral gray that the default background filter can hit (value~=0.70, within
    # gray_wall's tolerance) so the silhouette gets extracted correctly and we actually
    # test the real PCA elongation logic.
    BG = (180, 180, 180)

    # 1) Circular silhouette (mimics apple/orange) -> expect elongation ~= 1, class round
    frame_circle = np.full((h, w, 3), BG, dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    circle = (xx - 240) ** 2 + (yy - 240) ** 2 <= 30 ** 2
    frame_circle[circle] = (200, 30, 30)  # red circle
    elong_c, extent_c = silhouette_features(frame_circle, (240, 240), window_radius=45)
    shape_c = classify_shape(0.07, 0.035, elong_c, extent_c)  # aspect ~= 1
    print(f"  Circle: elongation={elong_c:.2f} extent={extent_c:.2f} -> geom round, combined='{shape_c}'")

    # 2) Long thin silhouette (mimics banana) -> expect large elongation, class elongated
    frame_bar = np.full((h, w, 3), BG, dtype=np.uint8)
    frame_bar[235:245, 170:310] = (230, 220, 40)  # horizontal yellow bar
    elong_b, extent_b = silhouette_features(frame_bar, (240, 240), window_radius=80)
    # Assume the banana geometry was misread as flat (small aspect); check whether 2D
    # can override the verdict to elongated.
    shape_b = classify_shape(0.03, 0.07, elong_b, extent_b)
    print(f"  Bar: elongation={elong_b:.2f} extent={extent_b:.2f} -> geom flat-ish, combined='{shape_b}'")

    print("\n=== Self-test: pure geometric aspect classification (no image) ===")
    cases = [
        ("apple(sphere)",  0.072, 0.036, None),   # aspect ~= 1.0 -> round
        ("pear",           0.110, 0.040, None),   # aspect ~= 1.38 -> oval
        ("plate",          0.020, 0.120, None),   # aspect ~= 0.08 -> flat
        ("rod",            0.250, 0.030, None),   # aspect ~= 4.2 -> elongated
    ]
    for name, hgt, rad, e2d in cases:
        s = classify_shape(hgt, rad, e2d)
        print(f"  {name}: aspect={hgt/(2*rad):.2f} -> '{s}'")

    print("\n=== Self-test: geom three-axis half-size classification (primary signal, distinguishes stick vs disk) ===")
    # For each object, provide three descending half-sizes [s0,s1,s2] (meters)
    ext_cases = [
        ("banana(lying stick)", [0.090, 0.018, 0.016]),  # one axis dominates -> elongated
        ("plate(flat disk)",    [0.120, 0.118, 0.020]),  # two big, one small -> flat
        ("apple(sphere)",       [0.036, 0.035, 0.034]),  # all three close -> round
        ("pear(ellipsoid)",     [0.055, 0.040, 0.038]),  # moderately long -> oval
    ]
    for name, half in ext_cases:
        s = classify_from_extents(half)
        r1 = half[1] / half[0]; r2 = half[2] / half[0]
        print(f"  {name}: r1={r1:.2f} r2={r2:.2f} -> '{s}'")

    print("\nSelf-test complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="shape_extraction module self-test")
    parser.add_argument("--selftest", choices=["synthetic"], default="synthetic")
    args = parser.parse_args()
    if args.selftest == "synthetic":
        _selftest_synthetic()