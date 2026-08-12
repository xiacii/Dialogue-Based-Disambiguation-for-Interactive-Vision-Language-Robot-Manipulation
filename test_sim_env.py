import os
import re
import sys
import time
import queue
import threading
import asyncio
import torch
import mujoco
import numpy as np
import gradio as gr

sys.path.append("/home/aijia/Desktop/DISS_Aijia_Wang/rrt-algorithms")
os.environ["VLABENCH_ROOT"] = "/home/aijia/Desktop/DISS_Aijia_Wang/VLABench/VLABench"

from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies import make_pre_post_processors
from lerobot.envs.vlabench import VLABenchEnv

from color_extraction import extract_colors_for_scene
from shape_extraction import extract_shapes_for_scene, read_object_center, read_object_geom_extents
from spatial_relations import (
    describe_scene_layout,
    find_object_toward,
    table_side_point,
    read_table_bounds,
    classify_table_region,
    _neg_x_is_image_left,
    _pos_y_is_image_front,
    project_objects_to_pixels,
    combined_direction,
    parse_placement,
)

# Dialogue Intermediary Layer (disambiguation/dual output between user and robotic arm). ‘dialogue’ is a package located in the project’s root directory.
from dialogue.state_manager import StateManager
from dialogue.dialogue_manager import DialogueManager

# Thread safe communication queue
frame_queue = queue.Queue(maxsize=1)  # Simulation to UI, frame-by-frame. Hold at most 1 element(only the most recent value)
task_command_queue = queue.Queue()   # UI to Simulation, immediate/control commands (reset/stop/shutdown)
action_command_queue = queue.Queue()  # UI to Simulation, sequential ACTION commands (pick/place/...),
                                       # drained one-at-a-time only when the arm is idle → Action Queue
camera_selection_queue = queue.Queue()  # UI to Simulation, Camera selected by the user
response_message_queue = queue.Queue(maxsize=4) # Simulation to UI, Feedback message
web_ui_ready = threading.Event()   # UI to Simulation, Event Signal: Webpage Opened


class MainThreadSimulationEngine:
    # A fixed tolerance (in meters) added to the plate radius multiplier (1.3x) 
    # when determining whether an object is considered to be on the plate
    PLATE_CHECK_RADIUS_MARGIN = 0.05

    def __init__(self, env, model, preprocess, postprocess, device):
        self.env = env  # Simulation Environment Objects
        self.model = model  # AI model
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.device = device    # cuda

        # Initializing the scene with a fixed seed
        self._scene_seed = 42
        # If the env version does not support the seed parameter, use the version without parameters
        try:
            self.observation, self.info = self.env.reset(seed=self._scene_seed)
        except TypeError:
            self.observation, self.info = self.env.reset()

        # State machine variables
        self.current_task = None
        self.is_active = False
        self.stage = 0   # 0=Standby 1=Hover 2=Descend 3=Clamp 4=Lift 5=Place 6=Return to Home Position
        self.stage_step = 0
        self.step_count = 0
        self.done = False

        # terminated delayed processing flag
        self._pending_terminated = False
        self._env_needs_reset = False

        # Camera Button
        self.active_camera_key = "image"

        # Set Robot Parameters
        self.robot_base = np.array([0.0, -0.4, 0.78], dtype=np.float64)
        self.default_euler = np.array([-3.1324, -0.0564, -1.6066], dtype=np.float64)
        self._active_euler = self.default_euler.copy()   # Current grasping orientation (for elliptical objects, yaw is adjusted according to the major axis)
        self.home_pos = np.array([0.0, -0.17, 1.21], dtype=np.float64)

        # Record the initial joint angles to accurately restore the pose after a silent reset
        self._home_qpos = None
        try:
            self._home_qpos = self.env._env.physics.data.qpos[:7].copy()
            print(f"[Init] Record the initial joint angles: {np.round(self._home_qpos, 4)}")
        except Exception as e:
            print(f"[Init] Unable to record initial joint angles: {e}")

        # Coordinates Related to the Mission
        self.target_pos = None
        self.plate_pos = np.array([0.0, 0.15, 0.87], dtype=np.float64)
        self._last_pickup_pos = None
        self._place_target = None
        self._grasp_obj_name = None      # Name of the object currently being gripped (used to calculate the gripper jaw opening based on dimensions)
        self._release_gripper = 1.0      # The ‘release claw opening’ (0–1), calculated based on the object’s dimensions, is fully open by default
        self._holding = False            # Is the gripper currently gripping an object? (Used during the ‘stop’ safety retraction)
        self._origin_pos = {}            # {Object name: Coordinates at the start of the Primordial World}, use ‘Return to original position’
        self._aliases = {}               # User alias {term: object name}, e.g. helen->apple (persists across resets)
        self._pending_alias_key = None
        self._pending_correction_wrong = None  # Items taken by mistake that are to be returned (cross-round correction)
        self._displaced = set()          # Objects that have been moved (more than 4 cm from their original position); main loop maintenance; read-only UI thread
        self._stop_put_xy = None         # stop: Records the current x/y coordinates when the device is set down
        self._last_action_object = None  # The name of the object picked up or put down most recently (for use as a reference for ‘the one just now’)

        # Sub-state machine for Stage 5 (Placement Stage): hover， descend，release
        # Resolve the issue of objects being placed crookedly or falling due to letting go before they are fully in place
        self._place_sub_stage = "hover"
        self._place_sub_step  = 0

        # Scene Object Color Cache: {Object Name: Color Term}
        self.scene_colors: dict[str, str] = {}
        self._color_scan_camera_name = "forward"

        # Scene Object Shape Cache: {Object Name: Shape Term}
        # It is entirely on a par with the colour cache and is repopulated by _scan_scene_shapes after every reset.
        self.scene_shapes: dict[str, str] = {}

        # Scene Layout Cache: {"a vs b": "a is left of b", ...}
        # A snapshot of the relative positions of objects after a reset (for printing and debugging purposes only
        # positions are always recalculated using real-time coordinates during command parsing, as object positions change after being picked up and placed).
        self.scene_layout: dict[str, str] = {}

        # Scene Region Cache: {Object Name: 'left'/'right'/'front'/'back'/'center'}
        # To be read safely by the dialogue intermediary when constructing the item list (do not access physics from other threads to avoid data races).
        self.scene_regions: dict[str, str] = {}

        # Scene Relations Cache: {(a, b): 'left'/'upper-right'/...}
        self.scene_relations: dict[tuple, str] = {}

        # ---- Interface Layer: Ambiguity Resolution between User and Robotic Arm + Dual Output ----
        # `state` holds the conversation state and the ‘real-time scene list’, whilst `dm` is responsible 
        # for resolving ambiguous commands into {`user_response` (for human viewing) and `robot_signal` (for the robotic arm)}.
        self.state = StateManager(current_task=self.current_task or "Pick an object")
        self.dm = DialogueManager(self.state)

        # Print the name of the site/body scene
        try:
            physics = self.env._env.physics
            print("=== [DEBUG] site names ===")
            print(list(physics.named.data.site_xpos.axes.row.names))
            print("=== [DEBUG] body names ===")
            print(list(physics.named.data.xpos.axes.row.names))
        except Exception as e:
            print(f"[DEBUG] {e}")

        # Initial Scene Color Scan
        self._scan_scene_colors()
        # Initial Scene Shape Scan
        self._scan_scene_shapes()
        # Initial Scene Layout Scan
        self._scan_scene_layout()


    # Tools functions
    def _physics(self):
        return self.env._env.physics

    # Read the position of the robotic arm's end effector
    def get_eef_pos(self):
        try:
            # Copy to prevent overwriting
            return self._physics().named.data.site_xpos[
                "franka/end_effector"].copy().astype(np.float64)
        except Exception:
            return self.observation["agent_pos"][0:3].astype(np.float64) + self.robot_base

    # Constructing a 7 dimensional motion vector, coordinate system transformation
    def make_action(self, target_world, gripper):   # World coordinates to robot body
        target_base = np.array(target_world, dtype=np.float64) - self.robot_base    # Robot controller receives coordinates relative to its own base
        act = np.zeros(7, dtype=np.float32)
        act[0:3] = target_base.astype(np.float32)   # XYZ position
        act[3:6] = self._active_euler.astype(np.float32)    # Gripper Posture (Orientation after alignment along the object’s long axis)
        act[6] = float(gripper)   # Gripper: 1.0 = open, 0.0 = closed
        return act

    # Unified object list: enumerate objects from BODY names (every "X/" prefix is
    # an object), position from bottom_site/unnamed_site if present, else geom centroid.
    def _get_scene_object_world_positions(self) -> dict[str, np.ndarray]:
        physics = self._physics()
        site_names = list(physics.named.data.site_xpos.axes.row.names)
        body_names = list(physics.named.data.xpos.axes.row.names)

        # Non-object prefix
        NON_OBJECT = ("world", "franka", "table", "plate", "default")

        roster = []
        for b in body_names:
            prefix = b.split("/")[0]
            if not prefix or prefix in roster:
                continue
            if any(k in prefix.lower() for k in NON_OBJECT):
                continue
            roster.append(prefix)

        # Collect available site locations
        site_pos: dict[str, np.ndarray] = {}
        for name in site_names:
            nl = name.lower()
            if "plate" in nl or "franka" in nl:
                continue
            is_std = name.endswith("/bottom_site")
            is_unnamed = "unnamed_site" in nl
            if not (is_std or is_unnamed):
                continue
            obj = name.split("/")[0]
            if not obj:
                continue
            if obj in site_pos and is_unnamed and not is_std:
                continue
            site_pos[obj] = physics.named.data.site_xpos[name].copy()

        # Determine the position of each object: use ‘site’ where available
        # otherwise, use the centre of mass of the ‘geom’ as a fallback.
        positions: dict[str, np.ndarray] = {}
        for obj in roster:
            if obj in site_pos:
                positions[obj] = site_pos[obj]
            else:
                c = read_object_center(physics, obj)
                if c is not None:
                    positions[obj] = c.astype(np.float64)
        return positions

    def _grasp_point_for(self, obj_name):
        # Given an object name, return the grasping point (preferably the true centre of mass
        # if this cannot be determined, return `bottom_site + 0.04`)
        # GRASP_Z_BIAS: Vertical fine-tuning relative to the centre of mass (metres), default 0.
        GRASP_Z_BIAS = 0.0
        try:
            center = read_object_center(self._physics(), obj_name)
        except Exception:
            center = None
        if center is not None:
            grasp = center.copy().astype(np.float64)
            grasp[2] += GRASP_Z_BIAS
            print(f"[Target] Get '{obj_name}' (centroid): {np.round(grasp, 4)}")
            return grasp
        positions = self._get_scene_object_world_positions()
        if obj_name in positions:
            pos = positions[obj_name].copy()
            pos[2] += 0.04
            print(f"[Target] Get '{obj_name}' (bottom_site+0.04 fallback): {np.round(pos, 4)}")
            return pos.astype(np.float64)
        return None

    def _release_gripper_for(self, obj_name):
        # Calculate the ‘release gap’ (0..1) based on the object’s actual lateral dimensions
        GRIPPER_MAX_GAP = 0.08  # Maximum finger spacing of the Franka gripper (metres)
        # An additional allowance (in metres) added to the width of the object to ensure 
        # that fingers can clear the object and the object is truly released from the hand
        EXTRA_CLEARANCE = 0.025
        FLOOR = 0.60    # Set the lower limit for the jaw opening to prevent the jaws from lifting an object whilst still clamped, even if the opening is too small.
        if not obj_name:
            return 1.0
        try:
            ext = read_object_geom_extents(self._physics(), obj_name)
            if ext is None:
                return 1.0
            width = 2.0 * float(ext[1]) # Median shaft diameter ≈ width of the gripping cross-section
            gap = width + EXTRA_CLEARANCE
            val = gap / GRIPPER_MAX_GAP
            val = float(min(max(val, FLOOR), 1.0))
            print(f"[Release] '{obj_name}' size(half)={np.round(ext,4)} "
                  f"-> gap≈{gap:.3f}m -> gripper_open={val:.2f}")
            return val
        except Exception:
            return 1.0

    # Posture Alignment for the Grasping of Elliptical Objects
    ALIGN_ENABLE = True
    ELONGATION_MIN = 1.30
    YAW_SIGN = 1.0
    YAW_ALIGN_K = 0.0

    # Calculate the Euler angles of the gripper for an elliptical object, with the gripper ‘clamping the short axis’.
    # Return the default orientation in the event of a circular error or an anomaly.
    def _grasp_euler_for(self, obj_name):
        base = self.default_euler.copy()
        if not (self.ALIGN_ENABLE and obj_name):
            return base
        try:
            physics = self._physics()
            model = physics.model
            body_names = list(physics.named.data.xpos.axes.row.names)
            target_ids = {i for i, b in enumerate(body_names)
                          if b == obj_name + "/" or b.startswith(obj_name + "/")}
            if not target_ids:
                return base
            aabb = np.asarray(model.geom_aabb, dtype=np.float64).reshape(-1, 6)
            geom_bodyid = np.asarray(model.geom_bodyid).reshape(-1)
            geom_xmat = np.asarray(physics.data.geom_xmat, dtype=np.float64).reshape(-1, 9)

            best_dir, best_score, halves = None, -1.0, []
            for gid in range(int(model.ngeom)):
                if int(geom_bodyid[gid]) not in target_ids:
                    continue
                half = np.abs(aabb[gid, 3:6])
                halves.append(half)
                R = geom_xmat[gid].reshape(3, 3)
                for k in range(3):
                    world_axis = R[:, k]
                    horiz = float(np.hypot(world_axis[0], world_axis[1]))
                    score = float(half[k]) * horiz  # A long, horizontal shaft
                    if score > best_score:
                        best_score, best_dir = score, world_axis
            if best_dir is None or not halves:
                return base
            merged = np.max(np.stack(halves, axis=0), axis=0)
            s = np.sort(merged)[::-1]
            ratio = float(s[0] / max(s[1], 1e-9))
            if ratio < self.ELONGATION_MIN:
                return base
            theta = float(np.arctan2(best_dir[1], best_dir[0])) # Long-axis orientation of the world
            base[2] = self.YAW_SIGN * theta + self.YAW_ALIGN_K
            print(f"[GraspYaw] '{obj_name}' elong={ratio:.2f} "
                  f"long-axis={np.degrees(theta):.0f}° -> yaw={np.degrees(base[2]):.0f}°")
            return base
        except Exception as e:
            print(f"[GraspYaw] failed ({e}); use default posture")
            return base

    # Match the target object from the scene and return its world coordinates
    def get_target_pos_from_scene(self, task_name):
        try:
            object_positions = self._get_scene_object_world_positions()
            task_lower = task_name.lower()
            for obj_name in object_positions:
                if obj_name in task_lower:
                    return self._grasp_point_for(obj_name)
            print(f"[Target] No scene objects were found for task '{task_name}'")
        except Exception as e:
            print(f"[Target] Read failed: {e}")
        return None

    # Calculate the world placement point in a given direction relative to a reference object
    def _relative_place_point(self, ref_name, direction, offset=0.13):
        try:
            physics = self._physics()
            positions = self._get_scene_object_world_positions()
            if ref_name not in positions:
                return None
            ref_xy = positions[ref_name][:2].astype(np.float64).copy()
            tb = read_table_bounds(physics)
            if tb is None:
                return None
            center, half, top_z = tb
            dest = ref_xy.copy()
            if direction in ("left", "right"):
                neg_left = _neg_x_is_image_left(
                    physics, center, half, self._color_scan_camera_name, 480, 480)
                right_is_plus_x = neg_left
                if direction == "right":
                    dest[0] += offset if right_is_plus_x else -offset
                else:
                    dest[0] += -offset if right_is_plus_x else offset
            elif direction in ("front", "back"):
                pos_y_front = _pos_y_is_image_front(
                    physics, center, half, self._color_scan_camera_name, 480, 480)
                if direction == "front":
                    dest[1] += offset if pos_y_front else -offset
                else:  # back
                    dest[1] += -offset if pos_y_front else offset
            else:
                return None
            # Within the safe working area on the worktop
            m = 0.05
            dest[0] = float(np.clip(dest[0], center[0]-half[0]+m, center[0]+half[0]-m))
            dest[1] = float(np.clip(dest[1], center[1]-half[1]+m, center[1]+half[1]-m))
            return np.array([dest[0], dest[1], float(top_z) + 0.03], dtype=np.float64)
        except Exception as e:
            print(f"[RelPlace] failed: {e}")
            return None

    # Use traditional CV methods to scan the colors of all objects in the current scene 
    # and store the results in self.scene_colors. 
    # This should be called once after each reset
    def _scan_scene_colors(self):
        try:
            physics = self._physics()
            object_world_positions = self._get_scene_object_world_positions()

            if not object_world_positions:
                print("[ColorScan] No recognizable objects were found in the scene, skipping the color scan")
                self.scene_colors = {}
                return

            # Reuse pre-rendered frames directly
            frame = self.observation.get("pixels", {}).get("wrist_image")
            if frame is None:
                print("[ColorScan] There is no wrist_image frame in the observation, skip the color scan")
                self.scene_colors = {}
                return

            self.scene_colors = extract_colors_for_scene(
                physics, frame, object_world_positions, camera_name=self._color_scan_camera_name
            )
            print(f"[ColorScan] Color scan complete: {self.scene_colors}")
        except Exception as e:
            print(f"[ColorScan] Color scan failed, set scene_colors to empty: {e}")
            self.scene_colors = {}

    # Use geometry sites (+ a 2D silhouette cue) to read the shape of every
    # object in the current scene and store the results in self.scene_shapes.
    # This should be called once after each reset, right next to the color scan.
    def _scan_scene_shapes(self):
        try:
            physics = self._physics()
            object_world_positions = self._get_scene_object_world_positions()

            if not object_world_positions:
                print("[ShapeScan] No recognizable objects were found in the scene, skipping the shape scan")
                self.scene_shapes = {}
                return

            # Reusing the same rendered image for both re-use and colour scanning
            # Only 2D silhouette signals can be aligned with geometric signals and colour signal coordinates.
            frame = self.observation.get("pixels", {}).get("wrist_image")
            if frame is None:
                print("[ShapeScan] There is no wrist_image frame in the observation, skip the shape scan")
                self.scene_shapes = {}
                return

            self.scene_shapes = extract_shapes_for_scene(
                physics, frame, object_world_positions, camera_name=self._color_scan_camera_name
            )
            print(f"[ShapeScan] Shape scan complete: {self.scene_shapes}")
        except Exception as e:
            print(f"[ShapeScan] Shape scan failed, set scene_shapes to empty: {e}")
            self.scene_shapes = {}

    # Scan the relative layout (left/right/above/below in the monitor view) of
    # all objects and cache a readable snapshot in self.scene_layout.
    # Call once after each reset, right after the color/shape scans.
    def _scan_scene_layout(self):
        try:
            physics = self._physics()
            object_world_positions = self._get_scene_object_world_positions()
            if not object_world_positions:
                print("[LayoutScan] No recognizable objects were found, skipping the layout scan")
                self.scene_layout = {}
                return
            self.scene_layout = describe_scene_layout(
                physics, object_world_positions, camera_name=self._color_scan_camera_name
            )
            print(f"[LayoutScan] Layout scan complete: {self.scene_layout}")

            # Caching complete and symmetric pairs of positions
            self.scene_relations = {}
            try:
                pix = project_objects_to_pixels(
                    physics, object_world_positions,
                    camera_name=self._color_scan_camera_name,
                )
                for a in pix:
                    for b in pix:
                        if a == b:
                            continue
                        d = combined_direction(pix[a], pix[b])
                        if d:
                            self.scene_relations[(a, b)] = d
            except Exception as e:
                print(f"[LayoutScan] full-relations build failed: {e}")
                self.scene_relations = {}

            # Cache the desktop partition for each object
            self.scene_regions = {}
            try:
                tb = read_table_bounds(physics)
                if tb is not None:
                    center, half, _top_z = tb
                    neg_left = _neg_x_is_image_left(
                        physics, center, half, self._color_scan_camera_name, 480, 480
                    )
                    pos_y_front = _pos_y_is_image_front(
                        physics, center, half, self._color_scan_camera_name, 480, 480
                    )
                    for name, pos in object_world_positions.items():
                        self.scene_regions[name] = classify_table_region(
                            pos, center, half,
                            neg_x_is_left=neg_left, pos_y_is_front=pos_y_front,
                        )
                    print(f"[RegionScan] Region scan complete: {self.scene_regions}")
            except Exception as e:
                print(f"[RegionScan] Region scan failed: {e}")
                self.scene_regions = {}

            # Record the initial position of each object in this scene, 
            # where they land at the start of the game or after a reset.
            self._origin_pos = {n: p.copy().astype(np.float64)
                                for n, p in object_world_positions.items()}
            self._displaced = set()
        except Exception as e:
            print(f"[LayoutScan] Layout scan failed, set scene_layout to empty: {e}")
            self.scene_layout = {}

    # Real-time query interface for command parsing
    def resolve_relational_target(self, ref_name: str, direction: str):
        try:
            physics = self._physics()
            positions = self._get_scene_object_world_positions()
            return find_object_toward(
                physics, positions, ref_name, direction,
                camera_name=self._color_scan_camera_name,
            )
        except Exception as e:
            print(f"[Relational] Parsing failed: {e}")
            return None

    # Parses a specific side of the table and returns a world placement coordinate for that side, 
    # to be used during the placement phase
    def get_table_side_world_point(self, side: str):
        try:
            physics = self._physics()
            robot_base_y = float(self.robot_base[1])
            return table_side_point(
                physics, side, camera_name=self._color_scan_camera_name,
                robot_base_y=robot_base_y,
            )
        except Exception as e:
            print(f"[TableSide] Parsing failed: {e}")
            return None

    # Dialogue Broker: Scenario List Assembly + Processing User Messages
    def build_objects_manifest(self):
        roster = set(self.scene_colors) | set(self.scene_shapes) | set(self.scene_regions)
        manifest = {}
        for name in sorted(roster):
            attrs = {}
            if self.scene_colors.get(name):
                attrs["color"] = self.scene_colors[name]
            if self.scene_shapes.get(name):
                attrs["shape"] = self.scene_shapes[name]
            if self.scene_regions.get(name):
                attrs["table_region"] = self.scene_regions[name]
            rel = self._relations_for(name)
            if rel:
                attrs["relative_position"] = rel
            manifest[name] = attrs
        return manifest

    # Provide the complete orientation of the object relative to all other objects
    def _relations_for(self, name: str):
        parts = []
        for (a, b), d in self.scene_relations.items():
            if a == name:
                parts.append(f"{d} of {b}")
        return "; ".join(parts) if parts else None

    # Process a user input and produce an output (reply, commands)
    # commands is a list of commands
    def handle_user_message(self, user_input: str):
        # Inject into the current scene; read-only cache dictionary
        self.state.set_objects_manifest(self.build_objects_manifest())
        scene_names = list(self.scene_colors) or list(self.scene_shapes) or list(self.scene_regions)

        # Batch Pickup
        filtered = self._resolve_by_attributes(user_input)  # Matches a subset of colours/shapes; may be empty
        is_all = self._looks_like_all_items(user_input)
        is_attr_plural = self._looks_like_plural_select(user_input) and len(filtered) > 1
        if is_all or is_attr_plural:
            if filtered:
                targets = [o for o in scene_names if o in filtered]
            elif is_all:
                targets = scene_names[:]
            else:
                targets = []
            if targets:
                # Estimating plate capacity
                try:
                    count, capacity = self.count_objects_on_plate()
                    available = max(int(capacity) - int(count), 0)
                except Exception:
                    available, capacity = len(targets), len(targets)
                placed = targets if available >= len(targets) else targets[:available]
                leftover = [] if available >= len(targets) else targets[available:]

                cmds = [f"pick up {o}" for o in placed]
                for o in placed:
                    self.state.update_target(o)
                if leftover:
                    kept = ", ".join(placed) if placed else "nothing (it's already full)"
                    reply = (f"Agent: The plate only holds about {capacity} item(s), so I'll place "
                             f"{kept} and leave {', '.join(leftover)} on the table.")
                else:
                    qual = " matching" if filtered else ""
                    reply = (f"Agent: Okay, I'll place all {len(placed)}{qual} items on the plate "
                             f"one by one: {', '.join(placed)}.")
                print(f"[Mediator] BATCH-PICK filtered={bool(filtered)} "
                      f"placed={placed} leftover={leftover} (capacity={capacity})")
                return reply, cmds
            if is_all and not scene_names:
                return "Agent: I don't see any objects on the table.", []

        # Chained multi-commands
        segments = self._split_into_segments(user_input)
        if len(segments) > 1:
            pieces, cmds = [], []
            for seg in segments:
                r, c = self._resolve_segment(seg)
                pieces.append(r)
                if c:
                    cmds.append(c)
            reply = "Agent: Okay — " + "; ".join(pieces) + "."
            print(f"[Mediator] CHAINED {segments} -> {cmds}")
            return reply, cmds

        # Single command
        if any(k in user_input.lower() for k in ("wait","mistake","actually","instead","the other",
                "not that","not this","wrong")) \
                and self._last_action_object and not self._pending_correction_wrong:
            self._pending_correction_wrong=self._last_action_object
        _prev = self._last_action_object
        reply, command = self._resolve_single(user_input)
        if command and command.startswith("pick up"):
            low=user_input.lower().strip()
            aff=any(low.startswith(k) for k in ("ok","okay","yes","yeah","yep","sure"))
            if not aff and not self._has_anchor(user_input):
                nm=list(self.scene_colors) or list(self.scene_shapes) or list(self.scene_regions)
                ask=(reply if (reply and "?" in reply) else ("Agent: I'm not sure which one you mean — I see "+", ".join(nm)+". Which would you like?") if nm else "Agent: Which one do you mean?")
                print(f"[Mediator] HARD-GATE suppressed {command!r}")
                return ask,[]
            # Correct the error and put it back first
            _newt=self._match_scene_name(command[len("pick up"):].strip())
            _wrong=self._match_scene_name(self._pending_correction_wrong)
            if _wrong and _newt and _wrong!=_newt:
                self._pending_correction_wrong=None
                print(f"[Mediator] CORRECT: placeback {_wrong} then pick {_newt}")
                return (f"Agent: Putting {_wrong} back first, then picking up the {_newt}.",
                        [f"placeback {_wrong}", command])
            if _newt==_wrong: self._pending_correction_wrong=None
            return reply, [command]
        return reply, ([command] if command else [])

    # Determine whether it is a batch command such as ‘Take all items’
    def _looks_like_all_items(self, text: str) -> bool:
        low = " " + text.lower().strip() + " "
        has_all = any(k in low for k in (" all ", " every ", "everything", "all of them", "each "))
        has_action = any(v in low for v in ("pick", "grab", "take", "collect", "put", "place", "move", "clear"))
        return has_all and has_action

    # Determine whether multiple items have been selected, such as when selecting several items at once by colour or shape.
    def _looks_like_plural_select(self, text: str) -> bool:
        low = " " + text.lower().strip() + " "
        return any(p in low for p in (" ones ", " items ", " Items ".lower(),
                                      " them ", " both ", " these ", " those "))

    # Split multiple sub-commands within a single command into their constituent parts using explicit conjunctions
    def _split_into_segments(self, text: str):
        parts = re.split(r'\s*(?:\band then\b|\bthen\b|\bafter that\b|\bnext\b|;|and|then)\s*',
                         text, flags=re.IGNORECASE)
        segs = [p for p in (s.strip() for s in parts) if p]
        # A command is considered to consist of multiple commands only if it contains at least two segments,
        # each of which resembles an independent clause
        # otherwise, it is treated as a single command.
        return segs if len(segs) >= 2 else [text.strip()]

    # Return the name of the scene object that first appears in the text
    def _first_named_object(self, text: str):
        low = text.lower()
        best = None
        scene_names = list(self.scene_colors) or list(self.scene_shapes) or list(self.scene_regions)
        for n in scene_names:
            pos = low.find(n.lower())
            if pos != -1 and (best is None or pos < best[0]):
                best = (pos, n)
        return best[1] if best else None

    # Deterministic Analysis
    def _resolve_segment(self, seg: str):
        scene_names = list(self.scene_colors) or list(self.scene_shapes) or list(self.scene_regions)
        # put back
        if self._is_place_back_request(seg):
            obj = self._place_back_object(seg)
            if obj:
                return (f"put {obj} back to its spot", f"placeback {obj}")
            return ("put the last item back", "place it back")
        # Objects covered in this section
        named = self._first_named_object(seg)
        attr = self._resolve_by_attributes(seg)
        obj = named or (attr[0] if len(attr) == 1 else None)
        # Set destination
        place = parse_placement(seg, scene_names)
        if place is not None and place.get("type") != "unknown":
            moved = (self._match_scene_name(place.get("moved")) or obj
                     or self._match_scene_name(self.state.current_target))
            if not moved:
                return (f"couldn't tell which object to move for '{seg}'", None)
            if place["type"] == "table_side":
                self.state.update_target(moved)
                return (f"place {moved} on the {place['side']} side",
                        f"placeside {moved} {place['side']}")
            ref = self._match_scene_name(place.get("ref"))
            if not ref:
                return (f"place {moved} — next to which object?", None)
            self.state.update_target(moved)
            return (f"place {moved} to the {place['direction']} of {ref}",
                    f"placerel {moved} {place['direction']} {ref}")
        # Otherwise, when picking up
        if obj:
            self.state.update_target(obj)
            return (f"pick up {obj}", f"pick up {obj}")
        return (f"couldn't resolve '{seg}'", None)

    _ALIAS_STOP={"the","a","an","one","ones","item","items","thing","things","is","are","it","this",
                "that","these","those","i","you","me","my","please","pick","up","grab","take","get","think",
                "want","like","put","place","move","back","to","of","on","and","or","no","yes","ok","okay",
                "sure","fruit","color","colour","bright","dark","long","longer","round","oval","left","right",
                "front","closer","near","which","other","green","red","yellow","orange","blue","just"}

    def _unknown_tokens(self,text):
        sc=set(n.lower() for n in (list(self.scene_colors) or list(self.scene_shapes) or list(self.scene_regions)))
        at=set()
        for d in (self.scene_colors,self.scene_shapes,self.scene_regions):
            for v in d.values():
                if v: at|=set(str(v).replace("-"," ").split())
        return [w for w in re.findall(r"[a-z0-9]+",text.lower())
                if len(w)>=2 and w not in self._ALIAS_STOP and w not in sc and w not in at]

    def _apply_alias(self,text):
        low=" "+text.lower()+" "
        for k,v in self._aliases.items():
            if (f" {k} " in low or low.rstrip().endswith(" "+k)) and self._match_scene_name(v):
                return self._match_scene_name(v)
        return None

    def _capture_alias(self,text):
        low=text.lower()
        m=(re.search(r"^\s*by\s+(.+?)\s+i\s+mean\s+(.+)$",low)
           or re.search(r"^(.*?)\s+(?:is|means|refers to|=)\s+(.+)$",low))
        if not m: return None
        L=re.sub(r"^(the|a|an)\s+","",m.group(1).strip())
        R=re.sub(r"^(the|a|an)\s+","",m.group(2).strip())
        def objs(desc):
            n=self._first_named_object(desc)
            if n: return [n]
            a=list(dict.fromkeys(self._resolve_by_attributes(desc)))
            return a if a else self._resolve_by_reference(desc)
        def is_unknown_word(w):
            return bool(re.fullmatch(r"[a-z0-9]+",w)) and not self._match_scene_name(w) \
                   and not self._resolve_by_attributes(w) and w not in self._ALIAS_STOP
        def tok_key(side):
            for w in re.findall(r"[a-z0-9]+",side):
                if w in self._aliases: return w # Aliases that already exist can be re-bound
                if is_unknown_word(w): return w
            return None
        Ld,Rd=objs(L),objs(R)
        if Rd and not Ld: desc,other=R,L
        elif Ld and not Rd: desc,other=L,R
        else: desc,other=(R,L)
        key=tok_key(other) or tok_key(desc) or self._pending_alias_key
        if not key: return None
        o=objs(desc)
        if len(o)==1:
            self._aliases[key]=o[0]
            if self._pending_alias_key==key: self._pending_alias_key=None
            self.state.update_target(o[0])
            print(f"[Mediator] alias: {key}->{o[0]}"); return ("ok",key,o[0])
        if len(o)>1:
            self._pending_alias_key=key; return ("ambiguous",key,o)
        return None

    def _has_anchor(self,text):
        if self._first_named_object(text): return True
        if self._resolve_by_attributes(text): return True
        if self._resolve_by_reference(text): return True
        if self._apply_alias(text): return True
        low=" "+text.lower()+" "
        return any(k in low for k in (" any "," whatever ","anything"))

    def _resolve_single(self, user_input: str):
        _lw=user_input.lower().strip()
        if any(_lw==k or _lw.startswith(k+" ") or _lw.startswith(k+",") for k in
               ("yes","yeah","yep","ok","okay","sure","correct")) \
                and not self._first_named_object(user_input):
            _t=self._match_scene_name(self.state.current_target)
            if _t:
                self._last_action_object=_t
                print(f"[Mediator] affirm -> pick {_t}")
                return (f"Agent: Sure — picking up the {_t}.", f"pick up {_t}")
        # Explicitly naming an object triggers locking
        # locking only occurs when the object’s name is a ‘direct object’.
        # If a sentence contains relational, referential or locative words, 
        # locking does not occur; the sentence is parsed as normal.
        _named = self._first_named_object(user_input)
        _lowu = user_input.lower()
        _REL = ("next to", "left of", "right of", " near ", "beside", "above", "below",
                "in front", "behind", "between", " other ", "closer", "farther",
                "put ", "place ", "move ")
        _has_rel = any(r in _lowu for r in _REL)
        _pickv = any(v in _lowu for v in ("pick", "grab", "take", "fetch", " get "))
        _bare = len(_lowu.split()) <= 2 # "apple" / "the apple"
        _conf_is = bool(_named and re.search(r"\bis\s+(the\s+)?" + re.escape(_named.lower()), _lowu))
        if _named and not _has_rel and not self._looks_like_question(user_input) \
                and not any(n in _lowu for n in ("not ", "n't")) \
                and (_pickv or _bare or _conf_is):
            self.state.update_target(_named)
            self._last_action_object = _named
            print(f"[Mediator] explicit-name lock -> pick up {_named}")
            return (f"Agent: Sure — picking up the {_named}.", f"pick up {_named}")
        # Intermediary disambiguation + dual output
        # the large model is responsible for resolving ambiguous references into object names
        _cap=self._capture_alias(user_input)
        if _cap and _cap[0]=="ambiguous":
            ds=[f"{o}"+((" ("+(self.scene_colors.get(o) or self.scene_shapes.get(o) or "")+")") if (self.scene_colors.get(o) or self.scene_shapes.get(o)) else "") for o in _cap[2]]
            return (f"Agent: More than one matches — {', '.join(ds)}. Which one is '{_cap[1]}'?",None)
        if _cap and _cap[0]=="ok":
            _cue=any(v in user_input.lower() for v in ("pick","grab","take",
                     "wait","mistake","actually","instead","not that","wrong"))
            if _cue:
                self.state.update_target(_cap[2]); self._last_action_object=_cap[2]
                return (f"Agent: Going with the {_cap[2]}.",f"pick up {_cap[2]}")
            return (f"Agent: Got it — '{_cap[1]}' means the {_cap[2]}. Want me to pick it up?",None)
        if not self._apply_alias(user_input):
            _u=[w for w in self._unknown_tokens(user_input) if w not in self._aliases]
            if _u: self._pending_alias_key=_u[0]
        _al=self._apply_alias(user_input)
        if _al and any(v in user_input.lower() for v in ("pick","grab","take","fetch","get ")) and not self._looks_like_question(user_input):
            self.state.update_target(_al); self._last_action_object=_al
            return (f"Agent: Going with the {_al}.",f"pick up {_al}")
        result = self.dm.process(user_input)
        reply = result.get("user_response", "") or ""
        signal = result.get("robot_signal", {}) or {}

        # Analysing ‘put it where’ using a deterministic spatial lexicon
        scene_names = list(self.scene_colors) or list(self.scene_shapes) or list(self.scene_regions)
        # Guard: if this is clearly a pick intent (pick/grab/take and no real place verb),
        # skip placement parsing so "actually pick up the yellow one" is never read as a table side.
        _lp = user_input.lower()
        _is_pick_intent = (any(v in _lp for v in ("pick", "grab", "take", "fetch", "\u62ff", "\u6293"))
                           and not any(v in _lp for v in ("put ", "place ", "move ", "drop ",
                                                          "set ", "leave ", "\u653e", "\u6446", "\u632a")))
        place = None if _is_pick_intent else parse_placement(user_input, scene_names)
        if place is not None and place.get("type") != "unknown":
            moved = (self._match_scene_name(place.get("moved"))
                     or self._match_scene_name(signal.get("target"))
                     or self._match_scene_name(self.state.current_target))
            if not moved:
                return ("Agent: Sure — which object should I move?", None)
            if place["type"] == "table_side":
                side = place["side"]
                cmd = f"placeside {moved} {side}"
                reply = f"Agent: Okay, placing the {moved} on the {side} side of the table."
            else:  # relative
                ref = self._match_scene_name(place.get("ref"))
                direction = place["direction"]
                if not ref:
                    return (f"Agent: I can place the {moved}, but next to which object?", None)
                cmd = f"placerel {moved} {direction} {ref}"
                reply = f"Agent: Okay, placing the {moved} to the {direction} of the {ref}."
            self.state.update_target(moved)
            print(f"[Mediator] PLACE reply={reply!r} | place={place} | command={cmd!r}")
            return reply, cmd

        if place is not None and place.get("type") == "unknown":
            # There is an intention to place something, but the destination is unclear
            r = (reply or "Agent: Where should I put it — a table side (left/right/front/back), "
                          "or next to another object?")
            print(f"[Mediator] PLACE unknown -> ask. reply={r!r}")
            return r, None

        command = self._robot_signal_to_command(signal)

        # The large model failed to identify the target
        # Loose exact match
        if command is None:
            low = user_input.lower()
            neg = any(n in low for n in ("not ", "n't", "don't", "dont", "no "))
            act = any(v in low for v in ("pick", "grab", "take", "want", "give",
                                         "bring", "fetch", "get "))
            short_select = len(low.split()) <= 3
            affirm = any(low.strip().startswith(k) for k in
                         ("ok", "okay", "yes", "yeah", "yep", "sure"))
            is_correction = any(k in low for k in ("actually", "i mean", "i meant", "no,", "no it"))
            
            cand = self._resolve_by_attributes(user_input)
            if len(cand) != 1:
                ref = self._resolve_by_reference(user_input)
                if len(ref) == 1:
                    cand = ref
            has_ref_cue = len(cand) == 1 and cand == self._resolve_by_reference(user_input)
            # Referential selection
            allow = has_ref_cue or is_correction or affirm or (not neg and (act or short_select))
            gate = (not self._looks_like_question(user_input) and allow)
            if gate and len(cand) == 1:
                obj = cand[0]
                self.state.update_target(obj)
                desc = self.scene_colors.get(obj) or self.scene_shapes.get(obj) or ""
                reply = (f"Agent: Going with the {obj}"
                         + (f" — the {desc} one." if desc else "."))
                command = f"pick up {obj}"
                print(f"[Mediator] fallback matched '{obj}' (ref_cue={has_ref_cue}) | command={command!r}")

        _low=user_input.lower().strip()
        _aff=any(_low.startswith(k) for k in ("ok","okay","yes","yeah","yep","sure"))
        _ct=None
        if command and command.startswith("pick up"): _ct=command[len("pick up"):].strip()
        elif _aff: _ct=self._match_scene_name(self.state.current_target) or self._match_scene_name(signal.get("target"))
        elif str(signal.get("action_modifier","")).lower() in ("keep","update"): _ct=self._match_scene_name(signal.get("target"))
        if _ct and self._pending_alias_key and self._pending_alias_key not in self._aliases:
            self._aliases[self._pending_alias_key]=_ct; print(f"[Mediator] alias(confirm): {self._pending_alias_key}->{_ct}"); self._pending_alias_key=None
        if not (reply and reply.strip()):
            nm=", ".join(list(self.scene_colors) or list(self.scene_shapes) or list(self.scene_regions))
            reply=("Agent: Sorry, I didn't catch which item you mean"+(f" — I see {nm}. Which one?" if nm else ". Could you clarify?")); command=None
        print(f"[Mediator] reply={reply!r} | robot_signal={signal} | command={command!r}")
        return reply, command

    # Determine whether it is a case of ‘putting it back in its original place’, and distinguish this from ‘placing it at the back of the table’
    def _is_place_back_request(self, text: str) -> bool:
        low = " " + text.lower().strip() + " "
        if "back to" in low and any(w in low for w in
                                    ("original", "place", "position", "spot", "where", "initial")):
            return True
        back_as_region = any(p in low for p in
                             ("back side", "the back", "to back", "on back",
                              "back of the table", "back region"))
        if " return " in low and not back_as_region and " side " not in low:
            return True
        has_verb = any(v in low for v in (" put ", " place ", " move ", " take ",
                                          " bring ", " set ", " return "))
        if has_verb and (" back " in low) and not back_as_region:
            return True
        return False

    def _place_back_object(self, text: str):
        named = self._first_named_object(text)
        if named:
            return named
        attr = self._resolve_by_attributes(text)
        return attr[0] if len(attr) == 1 else None

    # Refresh the set of ‘objects that have been moved’
    def _update_displaced(self):
        try:
            if not self._origin_pos:
                return
            cur = self._get_scene_object_world_positions()
            disp = set()
            for name, o in self._origin_pos.items():
                c = cur.get(name)
                if c is not None and float(np.linalg.norm(c[:2] - np.asarray(o)[:2])) > 0.04:
                    disp.add(name)
            self._displaced = disp
        except Exception:
            pass

    # Divide the items to be returned into those that need to be moved back 
    # and those that are already in their original positions
    def _partition_place_back(self, objs):
        to_move, already = [], []
        for o in objs:
            if (o in self._origin_pos) and (o not in self._displaced):
                already.append(o)
            else:
                to_move.append(o)
        return to_move, already

    def _place_back_targets(self, text: str):
        batch = self._looks_like_all_items(text) or self._looks_like_plural_select(text)
        if batch:
            filtered = self._resolve_by_attributes(text)
            scene = list(self.scene_colors) or list(self.scene_shapes) or list(self.scene_regions)
            base = [o for o in scene if o in filtered] if filtered else scene[:]
            targets = [o for o in base if self._origin_pos.get(o) is not None]
            if targets:
                return targets
        single = self._place_back_object(text)
        return [single] if single else None

    # Roughly determine whether it is a question
    def _looks_like_question(self, text: str) -> bool:
        low = text.lower().strip()
        if any(c in low for c in "?？"):
            return True
        words = low.split()
        first = words[0] if words else ""
        if first in ("what", "which", "why", "how", "where", "who", "is", "are",
                     "do", "does", "can", "could", "would", "should", "will", "whose"):
            return True
        return any(z in low for z in ("what", "which", "why", "how", "where", "who"))

    # Use the colours and shapes obtained from the scan to perform a loose match against the user’s text
    def _resolve_by_attributes(self, text: str):
        low = text.lower()
        matches = []
        roster = set(self.scene_colors) | set(self.scene_shapes)
        for name in roster:
            tokens = []
            c = self.scene_colors.get(name)
            s = self.scene_shapes.get(name)
            if c:
                tokens += c.replace("-", " ").split()
            if s:
                tokens += s.replace("-", " ").split()
            if any(len(t) >= 3 and t in low for t in tokens):
                matches.append(name)
        return matches

    # Deterministic anaphoric resolution: cases other than complement attribute matching
    def _resolve_by_reference(self, text: str):
        low = " " + text.lower().strip() + " "

        # The one that was recently modified
        last_cue = any(k in low for k in (
            "just put", "just placed", "just moved", "just picked", "put back",
            "placed back", "you moved", "you placed", "you put", "same one"))
        if last_cue:
            m = self._match_scene_name(self._last_action_object)
            if m:
                return [m]

        # Regional terms
        region_map = {"left": "left", "right": "right", "front": "front", "back": "back"}
        want = None
        for w, r in region_map.items():
            if w in low and f"not the {w}" not in low and f"not {w}" not in low:
                want = r
                break
        if want:
            hits = [n for n, reg in self.scene_regions.items() if reg == want]
            if len(hits) == 1:
                return hits
        return []

    # Clear the action commands in the queue
    def _drain_action_queue(self):
        try:
            while True:
                action_command_queue.get_nowait()
        except queue.Empty:
            pass

    # Match a potentially vague name to the exact name of an object in the scene
    def _match_scene_name(self, name):
        if not name:
            return None
        t = str(name).strip().lower()
        scene_names = list(self.scene_colors) or list(self.scene_shapes) or list(self.scene_regions)
        for n in scene_names:
            nl = n.lower()
            if t == nl or t in nl or nl in t:
                return n
        return None

    # Translate into a command string that the simulator can execute
    def _robot_signal_to_command(self, signal: dict):
        if not signal:
            return None

        target = signal.get("target")
        if not target:
            return None

        # A clear ‘Do Not Move’ sign
        modifier = str(signal.get("action_modifier", "")).strip().lower()
        action = str(signal.get("action", "")).strip().lower()
        if modifier in ("keep", "none", "noop") or action in ("none", "keep", "noop"):
            return None

        # Verification/alignment using real-world object names
        t = str(target).strip().lower()
        scene_names = list(self.scene_colors) or list(self.scene_shapes) or list(self.scene_regions)
        for n in scene_names:
            nl = n.lower()
            if t == nl or t in nl or nl in t:
                return f"pick up {n}"
        return None

    # Get position of the plate
    def get_plate_pos_from_scene(self):
        try:
            physics = self._physics()
            site_names = list(physics.named.data.site_xpos.axes.row.names)

            bottom_match = None
            unnamed_sites = []
            for name in site_names:
                nl = name.lower()
                if "plate" not in nl:
                    continue
                if "horizontal_radius" in nl or "top" in nl:
                    continue
                if "bottom" in nl and bottom_match is None:
                    bottom_match = name
                elif "unnamed_site" in nl:
                    unnamed_sites.append(name)

            # Use ‘bottom_site’ directly as the centre of the disc
            if bottom_match is not None:
                pos = physics.named.data.site_xpos[bottom_match].copy()
                pos[2] = max(pos[2] + 0.04, 0.80)
                print(f"[Plate] Use Standard Site '{bottom_match}': {np.round(pos, 4)}")
                return pos.astype(np.float64)

            # Where possible, use the centre of mass of the plate as the centre of the plate
            center = read_object_center(physics, "plate_seen")
            if center is not None:
                center = center.astype(np.float64)
                # Raise the top surface of the geometric shape slightly to ensure it sits slightly above the plate.
                if unnamed_sites:
                    zmax = float(np.array([physics.named.data.site_xpos[n][2]
                                           for n in unnamed_sites]).max())
                else:
                    zmax = float(center[2])
                center[2] = max(zmax + 0.04, 0.80)
                print(f"[Plate] Center from plate geom centroid: {np.round(center, 4)}")
                return center

            if unnamed_sites:
                pts = np.array([physics.named.data.site_xpos[n] for n in unnamed_sites],
                               dtype=np.float64)
                center = pts.mean(axis=0)
                center[2] = max(float(pts[:, 2].max()) + 0.04, 0.80)
                print(f"[Plate] Center from {len(unnamed_sites)} unnamed sites "
                      f"(mean): {np.round(center, 4)}")
                return center.astype(np.float64)
        except Exception:
            pass
        try:
            physics = self._physics()
            body_names = list(physics.named.data.xpos.axes.row.names)
            for name in body_names:
                if "plate" in name.lower() and name != "plate_seen/":
                    pos = physics.named.data.xpos[name].copy()
                    pos[2] = max(pos[2] + 0.06, 0.85)
                    print(f"[Plate] Use body '{name}': {np.round(pos, 4)}")
                    return pos.astype(np.float64)
        except Exception:
            pass
        print("[Plate] Use fallback position")
        return np.array([0.0, 0.15, 0.87], dtype=np.float64)

    # Calculate the radius of the plate
    def get_plate_radius_from_scene(self):
        try:
            physics = self._physics()
            site_names = list(physics.named.data.site_xpos.axes.row.names)

            # Reuse the unified entry point that is already compatible with unnamed_site
            plate_pos = self.get_plate_pos_from_scene()
            if plate_pos is None:
                return 0.12
            # The z value returned by get_plate_pos_from_scene, here only the x and y values are needed
            bottom_xy = plate_pos[:2]

            radius_pos = None
            for name in site_names:
                if "plate" in name.lower() and "horizontal_radius" in name.lower():
                    radius_pos = physics.named.data.site_xpos[name].copy()
                    break

            if radius_pos is not None:
                return max(np.linalg.norm(radius_pos[:2] - bottom_xy), 0.06)

            rim = []
            for name in site_names:
                nl = name.lower()
                if "plate" in nl and "unnamed_site" in nl:
                    rim.append(physics.named.data.site_xpos[name][:2].copy())
            if rim:
                dists = [float(np.linalg.norm(p - bottom_xy)) for p in rim]
                dists = [d for d in dists if d > 0.02]  # Exclude points that lie exactly at the centre of the disc
                if dists:
                    return max(float(np.median(dists)), 0.06)
        except Exception:
            pass
        return 0.12

    # Standardize the radius used to determine whether an 'object is considered to be on the plate'
    def _get_plate_check_radius(self, plate_radius: float) -> float:
        return plate_radius * 1.3 + self.PLATE_CHECK_RADIUS_MARGIN

    # Count the number of objects on the tray
    def count_objects_on_plate(self):
        try:
            plate_pos = self.get_plate_pos_from_scene()
            if plate_pos is None:
                return 0, 3
            plate_xy = plate_pos[:2]

            radius = self.get_plate_radius_from_scene()
            check_radius = self._get_plate_check_radius(radius)
            # Count the number of objects whose bottom_site falls within the plate's boundaries
            count = 0
            for obj_name, obj_pos in self._get_scene_object_world_positions().items():
                # Whether it's on the plate or not depends solely on the horizontal distance as seen from a top down perspective
                if np.linalg.norm(obj_pos[:2] - plate_xy) <= check_radius:
                    count += 1

            # Calculating the theoretical capacity of a Plate
            obj_area = np.pi * (0.04 ** 2)  # The area occupied by an object, a circle with a radius of 4cm
            capacity = max(2, min(int(np.pi * radius ** 2 / obj_area), 5))  # Plate area / Object area
            return count, capacity
        except Exception as e:
            print(f"[PlateCheck] Capacity Check Failed: {e}")
            return 0, 3

    # Dynamically calculate the placement coordinates within the disk
    def _compute_plate_placement_point(self):
        # Approximate radius of the object (meters), 
        # used to maintain a safe distance between the object's edge and the edge of the plate
        OBJECT_HALF_SIZE = 0.06

        try:
            plate_pos = self.get_plate_pos_from_scene()
            if plate_pos is None:
                return self.plate_pos

            plate_xy = plate_pos[:2]
            plate_radius = self.get_plate_radius_from_scene()

            # Maximum offset radius available for candidate points
            safe_radius = max(plate_radius - OBJECT_HALF_SIZE, 0.0)

            # Find the coordinates of objects that are already within the plate's range
            check_radius = self._get_plate_check_radius(plate_radius)
            existing_xy_list = []
            all_positions = self._get_scene_object_world_positions()
            print(f"[PlacePoint][DEBUG] plate_xy={np.round(plate_xy,4)} "
                  f"plate_radius={plate_radius:.4f} safe_radius={safe_radius:.4f} "
                  f"check_radius={check_radius:.4f}")
            for obj_name, obj_pos in all_positions.items():
                dist = np.linalg.norm(obj_pos[:2] - plate_xy)
                in_range = dist <= check_radius
                print(f"[PlacePoint][DEBUG] {obj_name}: xy={np.round(obj_pos[:2],4)} "
                      f"dist_to_plate_center={dist:.4f} Ruled in-bounds={in_range}")
                if in_range:
                    existing_xy_list.append(obj_pos[:2])

            if not existing_xy_list:
                # The plate is empty, place it in the center
                print("[PlacePoint][DEBUG] Determine if the plate is empty, using the center of the plate")
                return plate_pos

            # Candidate points include the center of the plate and 
            # grid points arranged in multiple concentric circles with varying radii and angles.
            # These points are strictly limited to within the safe_radius 
            # to ensure that the object's edges do not extend beyond the plate.
            candidate_points = [plate_xy]
            if safe_radius > 0.01:  # An outer ring is generated only if the available area is large enough
                n_rings = 3
                n_angles = 12
                for ring in range(1, n_rings + 1):
                    r = safe_radius * ring / n_rings
                    for a in range(n_angles):
                        theta = 2 * np.pi * a / n_angles
                        cand = plate_xy + r * np.array([np.cos(theta), np.sin(theta)])
                        if np.linalg.norm(cand - plate_xy) <= safe_radius + 1e-6:
                            candidate_points.append(cand)

            best_point = plate_xy
            best_min_dist = -1.0
            for cand in candidate_points:
                # The distance from a candidate point to the center of the disk must be <= safe_radius
                if np.linalg.norm(cand - plate_xy) > safe_radius + 1e-6:
                    continue
                min_dist = min(np.linalg.norm(cand - exist_xy) for exist_xy in existing_xy_list)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_point = cand

            result = plate_pos.copy()
            result[:2] = best_point
            print(f"[PlacePoint] There are already {len(existing_xy_list)} objects in the array, "
                  f"Dynamic Selection of Placement Points: {np.round(result, 4)} (Nearest Neighbor Distance={best_min_dist:.3f})")
            return result
        except Exception as e:
            print(f"[PlacePoint] Failed to calculate the dynamic placement point, reverting to the center of the plate: {e}")
            return self.plate_pos

    # Determines whether the plate is full, and returns (is_full, count, capacity, ui_message)
    def is_plate_full(self):
        count, capacity = self.count_objects_on_plate()
        is_full = count >= capacity
        if is_full:
            msg = (f"[System]: The plate is full! There are already {count} objects on the plate (the limit is approximately {capacity} objects)."
                   f"Click the “Reset New Scene” button to generate a new scene, or send 'place it back' to return the object to the desktop.")
        else:
            msg = f"[PlateCheck] Currently {count} out of {capacity} are in use; {capacity - count} more can be added."
        print(msg)
        return is_full, count, capacity, msg

    # Update the frames in frame_queue
    def _push_frame(self, frame=None):
        if frame is None:
            frame = self.observation["pixels"].get(self.active_camera_key)
        if frame is not None:
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except queue.Empty: pass
            frame_queue.put(frame)

    # Reset the scene using the same seed, then immediately reset the robot arm’s joint angles 
    # to their initial values and clear the velocity state to ensure the display does not flicker 
    # and that the IK starts from a clean state.
    def _silent_reset(self):
        print(f"[Env] Silent reset (seed={self._scene_seed})...")
        try:
            self.observation, self.info = self.env.reset(seed=self._scene_seed)
        except TypeError:
            self.observation, self.info = self.env.reset()

        if self._home_qpos is not None:
            try:
                physics = self._physics()
                physics.data.qpos[:] = 0.0
                physics.data.qpos[:7] = self._home_qpos
                physics.data.qvel[:] = 0.0
                physics.data.qacc[:] = 0.0
                physics.data.ctrl[:] = 0.0
                physics.forward()
                print("[Env] Silent reset complete, joint angles have been restored, and speed has been reset to zero.")
            except Exception as e:
                print(f"[Env] Silent reset complete (Failed to restore joint angle: {e})")
        else:
            print("[Env] Silent reset complete")

        # The scene has stabilized, rescan the object's color, shape, relative layout
        self._scan_scene_colors()
        self._scan_scene_shapes()
        self._scan_scene_layout()

        try:
            self._push_frame(self.observation["pixels"].get(self.active_camera_key))
        except Exception:
            pass
        self.env.render()

    # State Machine Main Loop, check the status every frame and then decide what action to take
    def run_pure_main_thread_loop(self):
        print("[MainThread Engine] The native MuJoCo 3D interactive environment is ready, waiting for the browser page to open...")
        web_ui_ready.wait()
        print("[MainThread Engine] A browser connection has been detected! The system has entered a secure standby mode...")

        while not self.done:
            # Camera Switch
            try:
                self.active_camera_key = camera_selection_queue.get_nowait()
            except queue.Empty:
                pass

            # New Instruction Processing
            try:
                new_task = task_command_queue.get_nowait()
            except queue.Empty:
                new_task = None
            if new_task is None and not self.is_active:
                try:
                    new_task = action_command_queue.get_nowait()
                except queue.Empty:
                    new_task = None
            if new_task is not None:
                cmd = new_task.lower().strip()

                if cmd in ("shutdown", "exit", "quit", "x"):
                    print("[Shutdown] Exiting...")
                    self.done = True
                    break

                if cmd in ("reset", "reload", "restart", "new scene", "new world"):
                    print("[Reset] Generating a new scene...")
                    self._drain_action_queue()  # Discard actions in the queue to prevent old commands from being executed in the new scene
                    self.is_active = False
                    self.stage = 0
                    self.stage_step = 0
                    self.current_task = None
                    self.target_pos = None
                    self._last_pickup_pos = None
                    self._place_target = None
                    self._holding = False
                    self._place_sub_stage = "hover"
                    self._place_sub_step = 0
                    self._active_euler = self.default_euler.copy()  # Reset the gripper orientation
                    self._pending_terminated = False
                    self._env_needs_reset = False
                    self._scene_seed = int(time.time()) % 10000 # Set a new scene seed
                    try:
                        self.observation, self.info = self.env.reset(seed=self._scene_seed)
                    except TypeError:
                        self.observation, self.info = self.env.reset()
                    print(f"[Reset] A new scene has been generated (seed={self._scene_seed})")
                    self._scan_scene_colors()
                    self._scan_scene_shapes()
                    self._scan_scene_layout()
                    self.state.reset_session()
                    if self._aliases:
                        valid=set(self.scene_colors)|set(self.scene_shapes)|set(self.scene_regions)
                        self._aliases={k:v for k,v in self._aliases.items() if v in valid}
                        self._pending_alias_key=None
                        print(f"[Mediator] aliases kept: {self._aliases}")
                    self._push_frame()
                    # Notification: UI Reset Complete
                    try:
                        response_message_queue.put_nowait(("reset_done", ""))
                    except queue.Full:
                        pass
                    continue

                if cmd in ("stop", "halt", "cancel", "discontinue", "end"):
                    self._drain_action_queue()
                    self.is_active = True
                    if self._holding:
                        # Set it down gently where it is
                        self._stop_put_xy = self.get_eef_pos()[:2].copy()
                        self.stage = 7
                        self.stage_step = 0
                        self._place_sub_stage = "down"
                        self._place_sub_step = 0
                        print("[Stop] Halting; placing held item down in place, then retracting.")
                    else:
                        self.stage = 6
                        self.stage_step = 0
                        print("[Stop] Halting; returning home (empty gripper).")
                    continue

                # Normal Task Commands
                self.current_task = new_task
                self.is_active = True
                self.stage = 1
                self.stage_step = 0
                self.target_pos = None
                self._place_target = None

                if self.env._robot_base_xyz is not None:
                    self.robot_base = np.array(self.env._robot_base_xyz, dtype=np.float64)
                print(f"[Init] robot_base={self.robot_base}")

                place_back_kw = ("place it back", "put it back", "put back", "take it back", "bring it back")
                is_place_back = any(k in cmd for k in place_back_kw) or cmd.startswith("placeback ")

                # Structured Placement Commands:
                #   placeside <obj> <side>  Place it on one side of the table
                #   placerel  <obj> <dir> <ref> Place it in a certain direction relative to an object
                #   placeback <obj> Put the specified object back in its proper place
                is_place_side = cmd.startswith("placeside ")
                is_place_rel = cmd.startswith("placerel ")

                if not is_place_back and not is_place_side and not is_place_rel:
                    # Dynamically calculate placement points within the disk in advance
                    self._place_target = self._compute_plate_placement_point()

                if is_place_side or is_place_rel:
                    try:
                        parts = new_task.strip().split()
                        obj = parts[1] if len(parts) > 1 else None
                        self._grasp_obj_name = obj
                        self.target_pos = self._grasp_point_for(obj) if obj else None
                        if is_place_side and len(parts) >= 3:
                            self._place_target = self.get_table_side_world_point(parts[2])
                        elif is_place_rel and len(parts) >= 4:
                            self._place_target = self._relative_place_point(parts[3], parts[2])
                        else:
                            self._place_target = None
                        if self.target_pos is None or self._place_target is None:
                            print(f"[Place] Could not resolve grasp/place target for '{new_task}'. Cancel.")
                            self.is_active = False
                        else:
                            self._last_pickup_pos = self.target_pos.copy()
                            print(f"[Place] Grab '{obj}': {np.round(self.target_pos,4)} "
                                  f"-> Put: {np.round(self._place_target,4)}")
                    except Exception as e:
                        print(f"[Place] Failed: {e}")
                        self.is_active = False
                elif is_place_back:
                    try:
                        # Determine the object to be returned: if specified in the command, use the specified one
                        # otherwise, take the one closest to the centre of the disc.
                        named = new_task.strip().split()[1] if cmd.startswith("placeback ") \
                            and len(new_task.strip().split()) > 1 else None
                        obj_name = self._match_scene_name(named) if named else None
                        if obj_name is None:
                            plate_xy = self.plate_pos[:2]
                            best_d = 99.0
                            for nm, pos in self._get_scene_object_world_positions().items():
                                d = np.linalg.norm(pos[:2] - plate_xy)
                                if d < best_d:
                                    best_d, obj_name = d, nm

                        # Give priority to the object’s own starting position
                        # if none exists, revert to the last pick-up point.
                        origin = self._origin_pos.get(obj_name) if obj_name else None
                        if origin is None:
                            origin = self._last_pickup_pos

                        grab = self._grasp_point_for(obj_name) if obj_name else None
                        if grab is not None and origin is not None:
                            origin_arr = np.asarray(origin, dtype=np.float64)
                            # Already in place
                            if np.linalg.norm(grab[:2] - origin_arr[:2]) < 0.04:
                                print(f"[PlaceBack] '{obj_name}' already at its original place; skip.")
                                try:
                                    response_message_queue.put_nowait(
                                        ("place_back_noop", f"{obj_name} is already in its original place."))
                                except queue.Full:
                                    pass
                                self.is_active = False
                            else:
                                self._grasp_obj_name = obj_name
                                self.target_pos = grab.astype(np.float64)
                                self._place_target = origin_arr.copy()
                                print(f"[PlaceBack] '{obj_name}' Grab: {np.round(self.target_pos,4)} "
                                      f"-> Origin: {np.round(self._place_target,4)}")
                        else:
                            print(f"[PlaceBack] Could not resolve object/origin "
                                  f"(obj={obj_name}). Cancel")
                            self.is_active = False
                    except Exception as e:
                        print(f"[PlaceBack] Failed: {e}")
                        self.is_active = False
                else:
                    # Standard Pickup
                    toks = self.current_task.strip().split()
                    self._grasp_obj_name = toks[-1] if toks else None
                    self.target_pos = self.get_target_pos_from_scene(self.current_task)
                    if self.target_pos is not None:
                        print(f"[Target] Final Destination Coordinates: {np.round(self.target_pos, 4)}")
                    else:
                        print("[Target] Unable to locate the target object; cancel this mission.")
                        self.is_active = False

                # Calculate the ‘release gap’ for this placement based on the actual dimensions of the object currently being gripped.
                if self.is_active:
                    self._release_gripper = self._release_gripper_for(self._grasp_obj_name)
                    # Aligning an elliptical object with the gripper along its major axis
                    self._active_euler = self._grasp_euler_for(self._grasp_obj_name)
                    if self._grasp_obj_name:
                        self._last_action_object = self._grasp_obj_name

                # Plate capacity check, standard pickup tasks only
                if self.is_active and not is_place_back and not is_place_side and not is_place_rel:
                    is_full, count, capacity, plate_msg = self.is_plate_full()
                    if is_full:
                        self.is_active = False
                        self.stage = 0
                        print(f"[PlateCheck] Plate is full. Refuse to carry out the task.")
                        try:
                            response_message_queue.put_nowait(("plate_full", plate_msg))
                        except queue.Full:
                            pass

                # Read the coordinates of the plate
                self.plate_pos = self._compute_plate_placement_point()
                print(f"[Plate] World Coordinates={np.round(self.plate_pos, 4)}")

            # In standby mode, only render frames, not step through
            if not self.is_active:
                self._push_frame()
                self.env.render()
                time.sleep(0.05)
                continue

            # Read EEF
            eef = self.get_eef_pos()
            tgt = self.target_pos if self.target_pos is not None else eef.copy()

            def reached(goal, tol=0.03):
                return np.linalg.norm(eef - np.array(goal)) < tol

            if self.step_count % 30 == 0:
                print(f"st{self.stage} step{self.step_count} "
                      f"eef={np.round(eef,3)} err={np.linalg.norm(eef-tgt):.3f}")

            # State machine
            action_np    = np.zeros(7, dtype=np.float32)
            action_np[6] = 1.0

            if self.stage == 1:
                # 15cm directly above the target, avoid knocking objects over
                hover = tgt + np.array([0, 0, 0.15])
                action_np = self.make_action(hover, gripper=1.0)    # The Claws Open
                self.stage_step += 1
                if reached(hover, tol=0.04) or self.stage_step >= 60:
                    self.stage = 2; self.stage_step = 0
                    print(f"[1 to 2] hover err={np.linalg.norm(eef-hover):.3f}")

            # Slowly lower it to the gripping position, keeping the claws open and aligned with the object
            elif self.stage == 2:
                grasp = tgt.copy()
                action_np = self.make_action(grasp, gripper=1.0)
                self.stage_step += 1
                if reached(grasp, tol=0.05) or self.stage_step >= 60:
                    self.stage = 3; self.stage_step = 0
                    print(f"[2 to 3] grasp err={np.linalg.norm(eef-grasp):.3f}")

            # Keep the position unchanged, close the gripper (0..45), then hold still a bit
            # longer (45..70) so the object seats firmly and its rotation damps before we
            # lift/transport it — otherwise oval fruit keeps spinning in the jaws and smashes the plate.
            elif self.stage == 3:
                grasp = tgt.copy()
                action_np = self.make_action(grasp, gripper=0.0)
                self.stage_step += 1
                if self.stage_step == 45:
                    self._last_pickup_pos = tgt.copy().astype(np.float64)
                    self._holding = True
                    print("[3] Gripper Closed; settling grasp...")
                if self.stage_step >= 70:
                    self.stage = 4; self.stage_step = 0
                    print("[3 to 4] Grasp settled.")

            # After clamping the object, lift it upward to remove it from the tabletop.
            elif self.stage == 4:
                if self.stage_step < 22:
                    lift = tgt + np.array([0, 0, 0.05])
                else:
                    lift = tgt + np.array([0, 0, 0.25])
                action_np = self.make_action(lift, gripper=0.0)
                self.stage_step += 1
                full_lift = tgt + np.array([0, 0, 0.25])
                if (self.stage_step >= 22 and reached(full_lift, tol=0.05)) or self.stage_step >= 90:
                    self.stage = 5; self.stage_step = 0
                    self._place_sub_stage = "hover"  # Explicit Reset of the Placement Sub-State Machine
                    self._place_sub_step = 0
                    print(f"[4 to 5] lift err={np.linalg.norm(eef-full_lift):.3f}")

            elif self.stage == 5:
                dest_base = np.array(self._place_target if self._place_target is not None
                                      else self.plate_pos, dtype=np.float64)
                dest = dest_base + np.array([0, 0, 0.025])
                dest_hover = dest_base + np.array([0, 0, 0.10])
                dest_clear = dest_base + np.array([0, 0, 0.12])

                if self._place_sub_stage == "hover":
                    action_np = self.make_action(dest_hover, gripper=0.0)
                    self._place_sub_step += 1
                    if reached(dest_hover, tol=0.04) or self._place_sub_step >= 50:
                        print(f"  [5.hover to hold] err={np.linalg.norm(eef-dest_hover):.3f} "
                              f"step={self._place_sub_step}")
                        self._place_sub_stage = "hover_hold"
                        self._place_sub_step = 0

                elif self._place_sub_stage == "hover_hold":
                    # Once you are directly above the drop-off point, 
                    # hold your position until the object’s swaying or rotation within the claw has subsided.
                    action_np = self.make_action(dest_hover, gripper=0.0)
                    self._place_sub_step += 1
                    if self._place_sub_step >= 15:
                        self._place_sub_stage = "descend"
                        self._place_sub_step = 0

                elif self._place_sub_stage == "descend":
                    # Descend vertically until you are close to the surface
                    action_np = self.make_action(dest, gripper=0.0)
                    self._place_sub_step += 1
                    if reached(dest, tol=0.015) or self._place_sub_step >= 80:
                        print(f"  [5.descend to settle] err={np.linalg.norm(eef-dest):.3f} "
                              f"step={self._place_sub_step}")
                        self._place_sub_stage = "settle"
                        self._place_sub_step = 0

                elif self._place_sub_stage == "settle":
                    # Keep it closed, hold it steady for a few frames, and eliminate the residual descent velocity.
                    action_np = self.make_action(dest, gripper=0.0)
                    self._place_sub_step += 1
                    if self._place_sub_step >= 10:
                        self._place_sub_stage = "release_soft"
                        self._place_sub_step = 0

                elif self._place_sub_stage == "release_soft":
                    # When an object falls onto the turntable and is completely released from the grip, 
                    # hold the camera steady for a few extra frames to ensure it has landed safely before raising it.
                    action_np = self.make_action(dest, gripper=self._release_gripper)
                    self._place_sub_step += 1
                    if self._place_sub_step >= 16:
                        self._place_sub_stage = "lift_clear"
                        self._place_sub_step = 0

                elif self._place_sub_stage == "lift_clear":
                    # Keep the door open at this angle and raise the lift arms vertically above the level of the items
                    action_np = self.make_action(dest_clear, gripper=self._release_gripper)
                    self._place_sub_step += 1
                    if reached(dest_clear, tol=0.03) or self._place_sub_step >= 60:
                        print(f"  [5.lift clear] err={np.linalg.norm(eef-dest_clear):.3f} "
                              f"step={self._place_sub_step}")
                        self._place_sub_stage = "release_full"
                        self._place_sub_step = 0

                # Reach the highest point and then open it fully, so that it does not touch anything else on the plate
                else:
                    action_np = self.make_action(dest_clear, gripper=1.0)
                    self._place_sub_step += 1
                    if self._place_sub_step >= 20:
                        self._place_sub_stage = "hover"  # Reset to default values for reuse in the next task
                        self._place_sub_step = 0
                        self._place_target = None
                        self._holding = False
                        self.stage = 6; self.stage_step = 0
                        print("[5 to 6] Put it down when you're done, and return it to its original place.")

            elif self.stage == 6:
                SAFE_Z = 1.10
                lifting = (eef[2] < SAFE_Z - 0.03) and (self.stage_step < 60)
                if lifting:
                    safe_up = np.array([eef[0], eef[1], SAFE_Z])
                    action_np = self.make_action(safe_up, gripper=(0.0 if self._holding else 1.0))
                else:
                    action_np = self.make_action(self.home_pos, gripper=1.0)
                    self._holding = False
                self.stage_step += 1
                home_err = np.linalg.norm(eef - self.home_pos)
                if (self.stage_step >= 60 and home_err < 0.08) or self.stage_step >= 240:
                    self.is_active = False
                    self.stage = 0
                    self.stage_step = 0
                    self._active_euler = self.default_euler.copy()
                    self._update_displaced()
                    print(f"Done! The robotic arm has returned to its original position (err={home_err:.3f})")

            elif self.stage == 7:
                # stop and set it down gently where it is
                put_xy = self._stop_put_xy if self._stop_put_xy is not None else eef[:2]
                PUT_Z = 0.82    # Release height relative to the desk
                down_target = np.array([put_xy[0], put_xy[1], PUT_Z], dtype=np.float64)
                if self._place_sub_stage == "down":
                    action_np = self.make_action(down_target, gripper=0.0)
                    self._place_sub_step += 1
                    if reached(down_target, tol=0.02) or self._place_sub_step >= 80:
                        self._place_sub_stage = "open"
                        self._place_sub_step = 0
                else:
                    action_np = self.make_action(down_target, gripper=self._release_gripper)
                    self._place_sub_step += 1
                    if self._place_sub_step >= 18:
                        self._holding = False
                        self._stop_put_xy = None
                        self._place_sub_stage = "hover"
                        self._place_sub_step = 0
                        self.stage = 6
                        self.stage_step = 0
                        print("[Stop] Item placed down in place; retracting home.")

            # Physical Stepping
            _orig_should_terminate = None
            try:
                _task = self.env._env.task
                _orig_should_terminate = _task.should_terminate_episode
                _task.should_terminate_episode = lambda physics: False
            except Exception:
                pass

            raw_obs, reward, raw_terminated, raw_truncated, info = self.env.step(action_np)

            # Restore the Original Method
            if _orig_should_terminate is not None:
                try:
                    self.env._env.task.should_terminate_episode = _orig_should_terminate
                except Exception:
                    pass

            self.observation = raw_obs
            self._push_frame()
            self.env.render()

            if raw_terminated or raw_truncated:
                print(f"[Env] VLABench terminated={raw_terminated}(Mission Success Determination), "
                      f"Ignored—reset triggered manually by the user or by a “disk full” prompt")

            self.step_count += 1
            time.sleep(0.01)

        print("[MainThread Engine] The main loop has exited, and the system has shut down safely.")
        self.env.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Currently Running Devices: {device}")

    print("Loading local open-source pre-trained models: lerobot/smolvla_base...")
    model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device)
    preprocess, postprocess = make_pre_post_processors(
        model.config, "lerobot/smolvla_base",
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    print("Launching the 3D VLABench multi-object simulation environment...")
    env = VLABenchEnv(render_mode="human")
    engine = MainThreadSimulationEngine(env, model, preprocess, postprocess, device)

    custom_css = """
    footer {visibility: hidden !important; height: 0px !important; padding: 0px !important; margin: 0px !important;}
    .gradio-container {background-color: #f7f9fa !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;}
    #send_btn   {background: linear-gradient(135deg, #007acc, #005999) !important; border: none !important; transition: all 0.2s ease;}
    #send_btn:hover {background: linear-gradient(135deg, #005999, #003f66) !important; transform: scale(1.02);}
    #reset_btn  {background: linear-gradient(135deg, #e07b00, #b35f00) !important; border: none !important; transition: all 0.2s ease; color: white !important;}
    #reset_btn:hover {background: linear-gradient(135deg, #b35f00, #8c4900) !important; transform: scale(1.02);}
    .gr-textbox input, .gr-textbox textarea {border-radius: 6px !important; font-size: 14px !important;}
    """

    with gr.Blocks(title="MSc Dissertation Platform") as demo:
        gr.HTML("<style>" + custom_css + "</style>")
        gr.Markdown(
            "# MSc Dissertation: Interactive Task Learning Platform\n"
            "### **3D Multi-Object Embodied Simulation Verification System (VLABench + SmolVLA + NLI Intention Disambiguation Console)**"
        )

        with gr.Row():
            # Left: Switch between video and camera
            with gr.Column(scale=2):
                video_output = gr.Image(
                    label="Web-Based Monitor (Current View)", height=380)
                camera_selector = gr.Radio(
                    choices=[
                        ("Right-side camera", "image"),
                        ("Left-side camera", "second_image"),
                        ("Main camera", "wrist_image"),
                    ],
                    value="image",
                    label="Switch between monitor and render views",
                )

            # Right side: Conversation area + Input + Buttons
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    value=[
                        {"role": "assistant",
                         "content": "Agent: The 3D multi-object simulation environment is ready and standing by."},
                        {"role": "assistant",
                         "content": "Agent: Tell me what you'd like in natural language (e.g. \"pick up the red one\", \"grab any fruit\", or just \"apple\"). "
                                    "I'll work out which object you mean, reply here, and send the robot a clear command. Type \"reset\" for a new scene."},
                    ],
                    label="Record of Human-Computer Interaction Experiment (Swipe to view)",
                    height=240,
                )

                with gr.Row():
                    user_input_box = gr.Textbox(
                        label="Enter a command (press Enter or click “Send” on the right)",
                        placeholder="Example: 'pick up banana', 'pick up apple'...",
                        lines=1,
                        scale=4,
                    )
                    send_btn = gr.Button(
                        "Send the control", variant="primary", scale=1, elem_id="send_btn")

                # Reset Button
                reset_btn = gr.Button(
                    "Reset to New Scene", variant="secondary", elem_id="reset_btn")

                status_display = gr.Markdown(
                    "**State**: The robotic arm stands at the ready, waiting for a command from a human to activate it...")

        # Camera Switch Callback
        def on_camera_changed(selected_cam):
            camera_selection_queue.put(selected_cam)

        camera_selector.change(fn=on_camera_changed, inputs=camera_selector, outputs=None)

        # Video Streaming Delivery
        async def refresh_video_stream():
            web_ui_ready.set()
            last_frame = None
            while True:
                if not frame_queue.empty():
                    last_frame = frame_queue.get_nowait()
                if engine.done:
                    yield last_frame
                    break
                yield last_frame
                await asyncio.sleep(0.05)

        # User input first passes through the dialogue intermediary, 
        # after which a decision is made as to whether to issue a command to the robotic arm
        def handle_intervention(user_feedback, chat_history):
            if not user_feedback.strip():
                yield "", chat_history, gr.update()
                return

            history = chat_history + [{"role": "user", "content": f"User: {user_feedback}"}]
            lowered = user_feedback.lower().strip()

            try:
                while True:
                    mtype, mtext = response_message_queue.get_nowait()
                    if mtype == "plate_full" and mtext:
                        history = history + [{"role": "assistant", "content": f"Agent: {mtext}"}]
            except queue.Empty:
                pass

            # Control-type commands are allowed through directly, without entering the dialogue interface.
            if lowered in ("reset", "reload", "restart", "new scene", "new world"):
                history = history + [{"role": "assistant",
                                      "content": "Agent: Reset command received. Generating a new scene..."}]
                task_command_queue.put(user_feedback)
                yield "", history, "**State**: Resetting the scene..."
                return

            if lowered in ("shutdown", "exit", "quit", "x"):
                history = history + [{"role": "assistant",
                                      "content": "Agent: Shutdown command received; the system is shutting down..."}]
                task_command_queue.put(user_feedback)
                yield "", history, "**State**: The system is shutting down..."
                return

            if lowered in ("stop", "halt", "cancel"):
                history = history + [{"role": "assistant",
                                      "content": "Agent: Stopping. The arm is returning to its home position."}]
                task_command_queue.put(user_feedback)
                yield "", history, "**State**: Stopped."
                return

            # Put it back where it belongs
            if engine._is_place_back_request(user_feedback):
                targets = engine._place_back_targets(user_feedback)
                if targets:
                    to_move, already = engine._partition_place_back(targets)
                    for o in to_move:
                        action_command_queue.put(f"placeback {o}")
                    parts = []
                    if to_move:
                        poss = "its" if len(to_move) == 1 else "their"
                        parts.append(f"returning {', '.join(to_move)} to {poss} original "
                                     + ("spot" if len(to_move) == 1 else "spots"))
                    if already:
                        be = "is" if len(already) == 1 else "are"
                        parts.append(f"{', '.join(already)} {be} already in place")
                    msg = ("Agent: Okay — " + "; ".join(parts) + ".") if parts \
                        else "Agent: Everything is already in its original place."
                    st = ("**State**: Placing the item(s) back..." if to_move
                          else "**State**: Already in place — nothing to move.")
                    print(f"[Mediator] PLACE-BACK move={to_move} already={already}")
                else:
                    action_command_queue.put("place it back")
                    msg = "Agent: Okay, returning the last item to its original position."
                    st = "**State**: Placing the item(s) back..."
                history = history + [{"role": "assistant", "content": msg}]
                yield "", history, st
                return

            thinking = history + [{"role": "assistant", "content": "Agent: Thinking..."}]
            yield "", thinking, "**State**: Thinking..."

            try:
                reply, commands = engine.handle_user_message(user_feedback)
            except Exception as e:
                reply, commands = (f"Agent: Sorry, I hit an internal error ({e}).", [])

            final = history + [{"role": "assistant", "content": reply or "Agent: (no response)"}]

            if commands:
                # Error correction and on-the-spot redirection
                if commands and commands[0].startswith("placeback "):
                    _w = engine._match_scene_name(commands[0].split()[1])
                    if (_w and engine.is_active
                            and engine._match_scene_name(engine._grasp_obj_name) == _w):
                        _org = engine._origin_pos.get(_w)
                        if _org is not None:
                            engine._place_target = np.asarray(_org, dtype=np.float64).copy()
                            print(f"[Mediator] REDIRECT in-flight {_w} -> its origin (skip plate)")
                            commands = commands[1:]
                # Add items to the action queue one by one; when the main loop is idle, execute them in order, one by one.
                for c in commands:
                    action_command_queue.put(c)
                if len(commands) == 1:
                    status = f"**State**: Executing → `{commands[0]}`"
                else:
                    status = f"**State**: Queued {len(commands)} actions → " + " → ".join(f"`{c}`" for c in commands)
            else:
                status = "**State**: Standing by (no robot action this turn)."

            yield "", final, status

        # Reset Button Callback
        def handle_reset(chat_history):
            chat_history.append(
                {"role": "assistant", "content": "[System]: Generating a new scene. Please wait..."})
            task_command_queue.put("reset")

            # Wait for the main loop to confirm that the reset is complete (up to 5 seconds)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                try:
                    msg_type, _ = response_message_queue.get_nowait()
                    if msg_type == "reset_done":
                        chat_history.append(
                            {"role": "assistant",
                             "content": "[System]: A new scene has been generated! The objects have been randomly rearranged, and you can begin a new experiment."})
                        return chat_history, "**State**: New scene ready, awaiting instructions..."
                except queue.Empty:
                    time.sleep(0.1)

            # Timeout, reset is still in progress
            chat_history.append(
                {"role": "assistant",
                 "content": "[System]: The scene is being reset. Please wait a moment before sending a command."})
            return chat_history, "**State**: Scene is reloading. Please wait..."

        # Event Binding
        user_input_box.submit(
            fn=handle_intervention,
            inputs=[user_input_box, chatbot],
            outputs=[user_input_box, chatbot, status_display],
        )
        send_btn.click(
            fn=handle_intervention,
            inputs=[user_input_box, chatbot],
            outputs=[user_input_box, chatbot, status_display],
        )

        reset_btn.click(
            fn=handle_reset,
            inputs=[chatbot],
            outputs=[chatbot, status_display],
        )

        demo.load(fn=refresh_video_stream, inputs=None, outputs=video_output)

    threading.Thread(
        target=demo.launch,
        kwargs={"server_name": "127.0.0.1", "server_port": 7860, "share": False},
        daemon=True,
    ).start()
    engine.run_pure_main_thread_loop()


if __name__ == "__main__":
    main()