# -*- coding: utf-8 -*-
"""
spatial_relations.py
=====================================================================
Object spatial relations + table region module. Parallel to
color_extraction / shape_extraction, this scans the scene layout after
each reset and provides two capabilities for downstream instructions:

  A) Relative direction between objects (left/right/above/below):
     Judged in the monitor view's (forward camera) pixel coordinates.
     Reason: when the user says "pick up the one to the left of the
     apple", "left" means what they see on screen. Comparing pixel
     column (u) / row (v) directly gives a result consistent with what
     the user sees, without needing to guess the world x/y axis
     orientation.
        smaller u -> more to the left on screen; larger u -> right
        smaller v -> higher on screen; larger v -> lower
     Typical use: "pick up the item to the left of the apple"
        -> find_object_toward(..., 'left')

  B) One side of the table (used for placement):
     Uses world coordinates, but the left/right direction is
     camera-calibrated. Two points ("table center +x half-step" and
     "-x half-step") are projected onto the image; whichever projects
     further to the left becomes the "left" world direction. This
     guarantees "put on the left side of the table" actually appears
     on the left of the monitor, regardless of camera placement.
     Front/back (near/far) are determined by the robot base y: the
     side closer to the base is front/near, the far side is back/far.
     Typical use: "put the banana on the right side of the table"
        -> table_side_point(..., 'right')

Reuses color_extraction.world_to_pixel for projection, so this module
shares the same camera and projection as color/shape — no coordinate
misalignment.

Standalone test:
    python spatial_relations.py --selftest
=====================================================================
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from color_extraction import world_to_pixel


# ======================================================================
# Thresholds and direction-word normalization
# ======================================================================

@dataclass
class RelationThresholds:
    """Thresholds for spatial relation judgement."""
    margin_px: float = 12.0     # If two objects' pixel difference along an axis is below
                                # this, treat them as roughly aligned on that axis and
                                # don't report a relation on it (avoids forcing left/right
                                # on nearly co-linear objects).
    diagonal_min_ratio: float = 0.40  # Threshold for calling a relation "diagonal": after
                                      # both horizontal and vertical components exceed
                                      # margin, the weaker axis must reach this ratio of
                                      # the stronger axis to form a diagonal (upper-left /
                                      # lower-right / ...). Otherwise report only the
                                      # stronger single axis (avoids calling a nearly
                                      # horizontal relation "diagonal").
    center_frac: float = 0.25   # Table zoning: within this fraction of the half-size
                                # from the table center counts as "center".
    side_inset: float = 0.60    # Table-side placement point: offset from center along
                                # the side = half-size * side_inset.


DEFAULT_RELATION_THRESHOLDS = RelationThresholds()


# Direction-word normalization: unifies synonyms into canonical directions
# for downstream instruction parsing.
#   4 cardinal: left/right/above/below (inter-object image relations); front/back (table)
#   4 diagonal: upper-left/upper-right/lower-left/lower-right
_DIRECTION_ALIASES = {
    "left": "left",
    "right": "right",
    "above": "above", "up": "above", "top": "above",
    "back": "back", "far": "back",
    "below": "below", "down": "below", "bottom": "below",
    "front": "front", "near": "front",
    # ---- Diagonal directions ----
    "upper-left": "upper-left", "upper left": "upper-left", "top-left": "upper-left",
    "top left": "upper-left", "ul": "upper-left",
    "upper-right": "upper-right", "upper right": "upper-right", "top-right": "upper-right",
    "top right": "upper-right", "ur": "upper-right",
    "lower-left": "lower-left", "lower left": "lower-left", "bottom-left": "lower-left",
    "bottom left": "lower-left", "ll": "lower-left",
    "lower-right": "lower-right", "lower right": "lower-right", "bottom-right": "lower-right",
    "bottom right": "lower-right", "lr": "lower-right",
}

# The "base components" each direction must hit (used by find_object_toward for
# candidate filtering): diagonal directions require both a horizontal and vertical hit.
_DIRECTION_COMPONENTS = {
    "left": ("left",), "right": ("right",),
    "above": ("above",), "below": ("below",),
    "upper-left": ("above", "left"), "upper-right": ("above", "right"),
    "lower-left": ("below", "left"), "lower-right": ("below", "right"),
}


def normalize_direction(word: str) -> Optional[str]:
    """Normalize any direction word to left/right/above/below/front/back, or None if unrecognized.

    Note: 'above'/'below' are image-space (used for inter-object relations); 'front'/'back'
    are table-space (used for table sides).
    """
    if word is None:
        return None
    return _DIRECTION_ALIASES.get(str(word).strip().lower())


# ======================================================================
# A) Inter-object relative direction (image pixel space) — pure functions, offline testable
# ======================================================================

def image_relation(a_uv, b_uv, thr: RelationThresholds = DEFAULT_RELATION_THRESHOLDS):
    """Judge a's direction relative to b in the monitor image.

    Args:
        a_uv, b_uv: (u, v) pixel coordinates
    Returns:
        (words, primary):
          words   — list of hit direction words, subset of {left,right,above,below}
                    (may contain one horizontal + one vertical, e.g. ["left","above"])
          primary — the dominant direction (axis with larger pixel displacement);
                    None if neither axis exceeds the threshold
    """
    du = a_uv[0] - b_uv[0]   # +: a is to the right of b (larger u)
    dv = a_uv[1] - b_uv[1]   # +: a is below b (larger v)

    words = []
    if du <= -thr.margin_px:
        words.append("left")
    elif du >= thr.margin_px:
        words.append("right")
    if dv <= -thr.margin_px:
        words.append("above")
    elif dv >= thr.margin_px:
        words.append("below")

    primary = None
    if abs(du) >= abs(dv) and abs(du) >= thr.margin_px:
        primary = "right" if du > 0 else "left"
    elif abs(dv) > abs(du) and abs(dv) >= thr.margin_px:
        primary = "below" if dv > 0 else "above"

    return words, primary


def combined_direction(a_uv, b_uv, thr: RelationThresholds = DEFAULT_RELATION_THRESHOLDS) -> Optional[str]:
    """Judge a's 8-way direction (including diagonals) relative to b.

    When both horizontal and vertical components exceed margin, if the weaker axis
    reaches at least diagonal_min_ratio of the stronger, we synthesize a diagonal
    (upper-left / upper-right / lower-left / lower-right); otherwise only the stronger
    single axis is reported. If only one component exceeds the margin, that single
    direction is reported. Returns None if neither does.
    """
    du = a_uv[0] - b_uv[0]   # +: a is right of b
    dv = a_uv[1] - b_uv[1]   # +: a is below b

    h = "right" if du >= thr.margin_px else ("left" if du <= -thr.margin_px else None)
    v = "below" if dv >= thr.margin_px else ("above" if dv <= -thr.margin_px else None)

    if h and v:
        big = max(abs(du), abs(dv))
        small = min(abs(du), abs(dv))
        if big > 0 and small >= thr.diagonal_min_ratio * big:
            vert = "upper" if v == "above" else "lower"
            return f"{vert}-{h}"          # upper-left / upper-right / lower-left / lower-right
        # Diagonal not "diagonal enough" — fall back to stronger single axis
        return h if abs(du) >= abs(dv) else v
    return h or v


def pairwise_relations_from_pixels(pixels: dict, thr: RelationThresholds = DEFAULT_RELATION_THRESHOLDS):
    """For a dict of {object_name: (u,v)}, compute relations for every ordered pair (a,b): a relative to b.

    Returns:
        {(a, b): {"words": [...], "primary": str|None}}
    """
    rel = {}
    names = list(pixels.keys())
    for a in names:
        for b in names:
            if a == b:
                continue
            words, primary = image_relation(pixels[a], pixels[b], thr)
            rel[(a, b)] = {"words": words, "primary": primary}
    return rel


def find_toward_from_pixels(
    pixels: dict, ref_name: str, direction: str,
    thr: RelationThresholds = DEFAULT_RELATION_THRESHOLDS,
) -> Optional[str]:
    """In the pixel layout, find the object that lies in the given direction from ref
    and is closest to ref.

    Supports 4 cardinal + 4 diagonal directions (left/right/above/below/upper-left/.../
    lower-right), and accepts common synonyms. Diagonal directions require both horizontal
    and vertical components to hit. Candidates whose 8-way direction exactly equals the
    target direction are preferred; ties break on pixel distance.

    Returns:
        Object name, or None (ref missing / no object in that direction).
    """
    norm = normalize_direction(direction) or direction
    components = _DIRECTION_COMPONENTS.get(norm)
    if components is None or ref_name not in pixels:
        return None
    ref = pixels[ref_name]

    candidates = []
    for name, uv in pixels.items():
        if name == ref_name:
            continue
        words, _ = image_relation(uv, ref, thr)
        # Every base component required by this direction must hit (diagonals need both axes)
        if all(c in words for c in components):
            comb = combined_direction(uv, ref, thr)
            dist = float(np.hypot(uv[0] - ref[0], uv[1] - ref[1]))
            # 8-way direction exactly matching target ranks first (0), otherwise (1); ties by distance
            candidates.append(((0 if comb == norm else 1, dist), name))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# ======================================================================
# A) Physics wrappers: projection + the pure functions above
# ======================================================================

def project_objects_to_pixels(
    physics, object_world_positions: dict, camera_name: str = "forward",
    height: int = 480, width: int = 480,
) -> dict:
    """Project each object's world coordinate onto the image; returns {object_name: (u,v)} (failures skipped)."""
    out = {}
    for name, pos in object_world_positions.items():
        uv = world_to_pixel(physics, camera_name, np.asarray(pos, dtype=np.float64), height, width)
        if uv is not None:
            out[name] = uv
    return out


def compute_pairwise_relations(
    physics, object_world_positions: dict, camera_name: str = "forward",
    thr: RelationThresholds = DEFAULT_RELATION_THRESHOLDS, height: int = 480, width: int = 480,
):
    """Scene-level: compute pairwise (image) directional relations between all objects."""
    pix = project_objects_to_pixels(physics, object_world_positions, camera_name, height, width)
    return pairwise_relations_from_pixels(pix, thr)


def find_object_toward(
    physics, object_world_positions: dict, ref_name: str, direction: str,
    camera_name: str = "forward", thr: RelationThresholds = DEFAULT_RELATION_THRESHOLDS,
    height: int = 480, width: int = 480,
) -> Optional[str]:
    """Scene-level: find the object nearest to ref_name in the given direction (for relational pick commands)."""
    pix = project_objects_to_pixels(physics, object_world_positions, camera_name, height, width)
    return find_toward_from_pixels(pix, ref_name, direction, thr)


def describe_scene_layout(
    physics, object_world_positions: dict, camera_name: str = "forward",
    thr: RelationThresholds = DEFAULT_RELATION_THRESHOLDS, height: int = 480, width: int = 480,
) -> dict:
    """Scene-level: generate a concise, readable layout description for logging/caching after reset.

    For each unordered pair, keeps only the primary relation, e.g.
    {"apple vs pear": "apple is left of pear"}.
    """
    pix = project_objects_to_pixels(physics, object_world_positions, camera_name, height, width)
    names = list(pix.keys())
    layout = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            comb = combined_direction(pix[a], pix[b], thr)
            if comb is not None:
                layout[f"{a} vs {b}"] = f"{a} is {comb} of {b}"
    return layout


# ======================================================================
# B) Table zoning / table-side placement point (world coords, left/right camera-calibrated)
# ======================================================================

def read_table_bounds(physics, table_keyword: str = "table"):
    """Read the table (surface) world center, xy half-sizes, and top-face height.

    Iterates geoms whose names contain table_keyword, picks the one with the largest
    horizontal footprint as the tabletop, and uses geom_xpos (world center) plus the
    last three entries of geom_aabb (local half-sizes) to estimate the table extent.
    Assumes the table is not rotated (axis-aligned), which holds in this scene.

    Returns:
        (center_xyz(3,), half_xyz(3,), top_z) or None.
    """
    try:
        model = physics.model
        geom_names = list(physics.named.data.geom_xpos.axes.row.names)
        aabb = np.asarray(model.geom_aabb, dtype=np.float64).reshape(-1, 6)

        best = None  # (area, center, half)
        for gid, gname in enumerate(geom_names):
            if table_keyword in gname.lower():
                half = np.abs(aabb[gid, 3:6]).astype(np.float64)
                area = float(half[0] * half[1])
                if best is None or area > best[0]:
                    center = physics.named.data.geom_xpos[gname].copy().astype(np.float64)
                    best = (area, center, half)
        if best is None:
            return None
        _, center, half = best
        top_z = float(center[2] + half[2])
        return center, half, top_z
    except Exception:
        return None


def _neg_x_is_image_left(physics, center, half, camera_name, height, width) -> bool:
    """Camera self-calibration: check whether world -x maps to image "left" (smaller pixel u).

    Projects "table center +/- x half-step" onto the image and compares u; if -x projects
    further left, returns True. Falls back to True on projection failure (typical layout).
    """
    p_plus = center.copy();  p_plus[0] += half[0] * 0.5
    p_minus = center.copy(); p_minus[0] -= half[0] * 0.5
    uv_plus = world_to_pixel(physics, camera_name, p_plus, height, width)
    uv_minus = world_to_pixel(physics, camera_name, p_minus, height, width)
    if uv_plus is None or uv_minus is None:
        return True
    return uv_minus[0] < uv_plus[0]


def _pos_y_is_image_front(physics, center, half, camera_name, height, width) -> bool:
    """Camera self-calibration: check whether world +y maps to image "front"
    (larger pixel v, i.e. lower in the image / closer to the viewer).

    Projects "table center +/- y half-step" onto the image and compares v; if +y
    projects further down (larger v), returns True. Falls back to True on
    projection failure.
    """
    p_plus = center.copy();  p_plus[1] += half[1] * 0.5
    p_minus = center.copy(); p_minus[1] -= half[1] * 0.5
    uv_plus = world_to_pixel(physics, camera_name, p_plus, height, width)
    uv_minus = world_to_pixel(physics, camera_name, p_minus, height, width)
    if uv_plus is None or uv_minus is None:
        return True
    return uv_plus[1] > uv_minus[1]


def classify_table_region(
    world_pos, center, half, thr: RelationThresholds = DEFAULT_RELATION_THRESHOLDS,
    neg_x_is_left: bool = True, robot_base_y: float = -0.4,
    pos_y_is_front: Optional[bool] = None,
) -> str:
    """Judge which table region a world coordinate falls into.

    Returns "left"/"right"/"front"/"back"/"center" (aligned with image / robot-arm
    intuition: left/right are camera-calibrated, front = closer to the robot base,
    back = further). Reports whichever axis's deviation from center is larger; both
    within center_frac -> "center".

    If pos_y_is_front is provided (from camera self-calibration), it overrides the
    robot_base_y-based heuristic for front/back direction.
    """
    p = np.asarray(world_pos, dtype=np.float64)
    dx = (p[0] - center[0]) / max(half[0], 1e-9)   # -1..1
    dy = (p[1] - center[1]) / max(half[1], 1e-9)

    # x direction -> left/right (respecting calibrated orientation)
    if dx > 0:
        x_word = "left" if not neg_x_is_left else "right"
    else:
        x_word = "left" if neg_x_is_left else "right"

    # y direction -> front/back. Prefer camera-calibrated pos_y_is_front when given;
    # otherwise fall back to the robot_base_y heuristic (front = closer to base).
    if pos_y_is_front is not None:
        toward_robot_is_neg_y = not pos_y_is_front
    else:
        toward_robot_is_neg_y = robot_base_y < center[1]
    if dy < 0:
        y_word = "front" if toward_robot_is_neg_y else "back"
    else:
        y_word = "back" if toward_robot_is_neg_y else "front"

    ax, ay = abs(dx), abs(dy)
    if ax < thr.center_frac and ay < thr.center_frac:
        return "center"
    return x_word if ax >= ay else y_word


def table_side_point(
    physics, side: str, camera_name: str = "forward",
    thr: RelationThresholds = DEFAULT_RELATION_THRESHOLDS,
    robot_base_y: float = -0.4, z_offset: float = 0.02,
    height: int = 480, width: int = 480,
):
    """Return a world placement coordinate on the specified side of the table.

    Args:
        side: 'left'/'right'/'front'/'back' (synonyms accepted, auto-normalized)
        z_offset: extra height above the table top (meters), for placement clearance
    Returns:
        (3,) world coordinate, or None (table extent unavailable / invalid direction).
    """
    tb = read_table_bounds(physics)
    if tb is None:
        return None
    center, half, top_z = tb

    side_n = normalize_direction(side)
    if side_n in ("above", "back"):   # semantic "above/back" for an object == "back" for the table
        side_n = "back"
    if side_n in ("below", "front"):
        side_n = "front"
    if side_n not in ("left", "right", "front", "back"):
        return None

    neg_x_is_left = _neg_x_is_image_left(physics, center, half, camera_name, height, width)
    p = center.copy()

    if side_n in ("left", "right"):
        # If we want "left", pick the world x direction that maps to image-left
        want_neg_x = (side_n == "left") == neg_x_is_left  # left AND -x is left -> use -x
        sign = -1.0 if want_neg_x else 1.0
        p[0] = center[0] + sign * half[0] * thr.side_inset
    else:
        toward_robot_is_neg_y = robot_base_y < center[1]
        if side_n == "front":
            sign = -1.0 if toward_robot_is_neg_y else 1.0
        else:  # back
            sign = 1.0 if toward_robot_is_neg_y else -1.0
        p[1] = center[1] + sign * half[1] * thr.side_inset

    p[2] = top_z + z_offset
    return p.astype(np.float64)


# ======================================================================
# Natural-language placement instruction parsing
# ======================================================================

# Side words we recognize in placement instructions (all normalized via
# normalize_direction to canonical left/right/front/back).
_PLACEMENT_SIDE_WORDS = (
    "left", "right", "front", "back", "near", "far",
    "above", "below", "up", "down", "top", "bottom",
)


def parse_placement(text: str, scene_names: Optional[list] = None) -> Optional[dict]:
    """Parse a natural-language placement instruction like "put the apple on the left side".

    Returns a dict:
        {"type": "table_side", "moved": <name or None>, "side": <"left"/"right"/"front"/"back">}
    when a side is detected, or:
        {"type": "unknown", "moved": <name or None>, "side": None}
    when the text mentions no recognizable side.

    Returns None only if the input text is empty/None.

    Args:
        text: raw user utterance
        scene_names: optional list of known object names; if provided, the first one
                     that appears in the text (case-insensitive) is returned as "moved".
    """
    if not text:
        return None
    lower = str(text).strip().lower()
    if not lower:
        return None

    # 1) Extract the object being moved (first scene name that appears in the text)
    moved = None
    if scene_names:
        # Prefer longer names first so "apple_1" wins over "apple" when both are in the scene
        for name in sorted((n for n in scene_names if n), key=len, reverse=True):
            if re.search(r"\b" + re.escape(name.lower()) + r"\b", lower):
                moved = name
                break

    # 2) Extract the side.
    # First try the strong pattern "on/to/at the <side> [side]" for high-confidence matches.
    side_pattern = (
        r"\b(?:on|to|onto|at|in|toward|towards|put(?:\s+it)?)\s+"
        r"(?:the\s+)?"
        r"(upper[-\s]?left|upper[-\s]?right|lower[-\s]?left|lower[-\s]?right|"
        r"top[-\s]?left|top[-\s]?right|bottom[-\s]?left|bottom[-\s]?right|"
        r"left|right|front|back|near|far|above|below|up|down|top|bottom)"
        r"(?:\s+side)?\b"
    )
    side = None
    m = re.search(side_pattern, lower)
    if m:
        side = normalize_direction(m.group(1))

    # Fallback: look for "<side> side" bare phrase anywhere in the text
    if side is None:
        for w in _PLACEMENT_SIDE_WORDS:
            if re.search(r"\b" + w + r"\s+side\b", lower):
                side = normalize_direction(w)
                break

    # Fallback: bare side word as its own token (weakest signal). Only trust it when a
    # real place verb is present, otherwise "pick UP" would match "up"->above->back.
    _has_place_verb = any(v in lower for v in
        ("put", "place", "move", "drop", "set ", "leave", "position", "放", "摆", "挪", "搓"))
    if side is None and _has_place_verb:
        for w in _PLACEMENT_SIDE_WORDS:
            if re.search(r"\b" + w + r"\b", lower):
                side = normalize_direction(w)
                break

    # Map image-space "above/below" to table-space "back/front" for placement
    # (same convention used inside table_side_point).
    if side == "above":
        side = "back"
    elif side == "below":
        side = "front"

    if side in ("left", "right", "front", "back"):
        return {"type": "table_side", "moved": moved, "side": side}
    return {"type": "unknown", "moved": moved, "side": None}


# ======================================================================
# Standalone self-test
# ======================================================================

def _selftest():
    print("=== Self-test: inter-object 8-way image directions ===")
    # Construct a pixel layout (u increases right, v increases down)
    pix = {
        "apple":  (100, 300),   # lower-left
        "pear":   (260, 110),   # upper-right
        "banana": (110, 120),   # upper-left
        "orange": (300, 310),   # lower-right
        "kiwi":   (200, 205),   # middle
    }
    for i, a in enumerate(pix):
        for b in list(pix)[i + 1:]:
            comb = combined_direction(pix[a], pix[b])
            print(f"  {a} relative to {b}: {comb}")

    print("\n=== Self-test: find object in given (diagonal) direction from a reference ===")
    for ref, d in [("kiwi", "upper-left"), ("kiwi", "lower-right"),
                   ("kiwi", "upper right"), ("kiwi", "bottom-left"), ("kiwi", "left")]:
        got = find_toward_from_pixels(pix, ref, d)
        print(f"  Object nearest to {ref} on side {d:11s} -> {got}")

    print("\n=== Self-test: direction-word normalization (incl. diagonals) ===")
    for w in ["left", "right", "up", "bottom", "near", "far", "lower-right", "upper left", "ul", "?"]:
        print(f"  '{w}' -> {normalize_direction(w)}")

    print("\n=== Self-test: table zoning (pure logic, given center/half) ===")
    center = np.array([0.0, 0.05, 0.78]); half = np.array([0.30, 0.30, 0.02])
    samples = {
        "far_right_pt": np.array([0.25, 0.30, 0.80]),
        "near_left_pt": np.array([-0.25, -0.20, 0.80]),
        "center_pt":    np.array([0.02, 0.06, 0.80]),
    }
    for nm, p in samples.items():
        region = classify_table_region(p, center, half, neg_x_is_left=True, robot_base_y=-0.4)
        print(f"  {nm} {p[:2]} -> table region '{region}'")

    print("\nSelf-test complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="spatial_relations module self-test")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    _selftest()