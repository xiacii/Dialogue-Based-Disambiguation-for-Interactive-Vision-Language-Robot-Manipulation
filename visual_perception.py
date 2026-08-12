# -*- coding: utf-8 -*-
"""
=====================================================================
Standalone visual perception module: uses a local open-source VLM
(SmolVLM2-500M-Video-Instruct) to generate structured visual feature
descriptions (color/shape/...) for objects in the simulated scene.

Core design
-----------
1. Fully isolated from SmolVLA's internal visual encoder; loads its
   own model weights.
2. VRAM friendly: loads on demand by default (moves to GPU right
   before inference, back to CPU and clears the cache right after).
3. Outputs structured JSON for easy parsing by the disambiguation
   interface.
4. On JSON parse failure, retries with backoff; falls back to an
   empty dict when retries are exhausted.
5. Accepts a ground-truth object list from the simulator (ObjectHint),
   forcing the VLM to fill in every entry — no missed detections, no
   hallucinated objects. Color/shape are still judged by the VLM,
   but "how many objects, what they are called, roughly where" comes
   from the caller's precise coordinates, not the VLM's free discovery.

Camera naming
-------------
Correspondence between VLABenchEnv's observation["pixels"] keys and
actual camera positions:
    - "wrist_image"  -> main camera / center viewpoint (clearest overview)
    - "image"        -> left camera
    - "second_image" -> right camera
Recommended call order: [wrist_image, image, second_image].

Standalone tests
----------------
This file can run standalone, no VLABench/lerobot/mujoco needed:
    python visual_perception.py --selftest json     # pure logic test, seconds, no model
    python visual_perception.py --selftest images --images frame.png --hints apple banana

Integration examples are at the bottom of this file.
=====================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("visual_perception")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


# Configuration
@dataclass
class VisualPerceptionConfig:
    """Module configuration; centralizes all tunable parameters."""

    model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    # On-demand loading strategy: True = move model back to CPU right after inference
    # (recommended for 8GB GPUs).
    offload_after_inference: bool = True

    # Retries on JSON parse failure (excluding the first attempt)
    max_retries: int = 2
    retry_backoff_seconds: float = 0.3

    max_new_tokens: int = 256
    device: str = "auto"           # "auto" / "cuda" / "cpu"
    torch_dtype: str = "float16"   # fp16 recommended for 8GB GPUs

    # Image preprocessing: crop the table region + upscale, boosting effective
    # resolution for small objects. (In the raw 480x480 frame the robot arm covers
    # a large portion of pixels, and small tabletop objects may be only ~20-30 pixels
    # across — nearly invisible to a 500M-scale VLM.)
    enable_crop_and_upscale: bool = True

    # Crop box (top, bottom, left, right), values in 0.0~1.0.
    # Defaults are calibrated for wrist_image (main/center overhead view); image/
    # second_image have different arm occlusion and can be given a per-frame box
    # via scan_scene()'s crop_boxes parameter.
    default_crop_box: tuple[float, float, float, float] = (0.18, 1.0, 0.0, 1.0)
    upscale_target_size: int = 640


# Object hint data structure (carries ground-truth coordinate info to
# strongly constrain the VLM's output)
@dataclass
class ObjectHint:
    """
    Metadata for a known object in the scene, sourced from simulator
    ground-truth coordinates (e.g. MuJoCo's site_xpos) rather than VLM guesses.

    Attributes:
        name: Object identifier; recommended to match the simulator's
              internal site/body name, e.g. "banana", "apple_1" (use
              suffixes to disambiguate multiple instances of the same type).
        relative_position: Optional short description of relative position,
              e.g. "front-left of the plate". Recommended to be generated
              by the caller based on site_xpos world coordinates relative
              to other objects / robot base.
    """

    name: str
    relative_position: Optional[str] = None

    def to_prompt_line(self) -> str:
        if self.relative_position:
            return f"- {self.name} (located: {self.relative_position})"
        return f"- {self.name}"


def _normalize_object_hints(object_hints: Optional[list] = None) -> list[ObjectHint]:
    """Normalize object_hints into list[ObjectHint].

    Backward compatible: accepts list[str] (names only) or list[ObjectHint].
    """
    if not object_hints:
        return []
    normalized: list[ObjectHint] = []
    for item in object_hints:
        if isinstance(item, ObjectHint):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append(ObjectHint(name=item))
        else:
            raise TypeError(f"object_hints elements must be str or ObjectHint, got {type(item)}")
    return normalized


# ======================================================================
# Prompt templates
# Design evolution notes (explains why the current form looks like this):
#   v1: ask for color+shape of all objects at once, require JSON output
#       -> model verbatim-copied the few-shot examples
#   v2: dynamic list + placeholder skeleton, still one shot for all
#       -> long prompt overloaded attention; model dropped the shape field
#          and color judgements degraded to defaults
#   v3 (current): drop the JSON requirement; ask about one object at a time
#       with minimal natural-language Q&A (e.g. "What color is the banana?
#       Answer in 1-3 words."). Prompt length is roughly constant regardless
#       of object count. Diagnostic tests showed the 500M model returns
#       answers close to reality with this simple Q&A. Structured extraction
#       is handled deterministically in Python (see _parse_color_answer /
#       _parse_shape_answer below); we no longer rely on the model to emit
#       valid JSON. Cost: 2N calls (N objects x color/shape), but each call
#       is faster (short prompt + short output), so overall latency is fine.

_SINGLE_OBJECT_COLOR_PROMPT = (
    "Look at the photo. What is the dominant color of the {object_name}"
    "{position_hint}? Answer in 1 to 3 words only, e.g. 'dark red' or "
    "'pale yellow'. Do not write a full sentence."
)

_SINGLE_OBJECT_SHAPE_PROMPT = (
    "Look at the photo. What is the shape of the {object_name}"
    "{position_hint}? Answer in 1 to 3 words only, e.g. 'round' or "
    "'long and curved'. Do not write a full sentence."
)

# Color/shape keyword library for deterministic extraction from the VLM's free-text
# answer. Order matters — more specific terms come before more general ones (e.g.
# "yellow-green" must match before plain "yellow"/"green", otherwise it'd get
# truncated to an inaccurate result).
_COLOR_KEYWORDS = [
    "yellow-green", "yellow green",
    "dark red", "bright red", "red",
    "dark green", "bright green", "green",
    "bright yellow", "pale yellow", "yellow",
    "orange",
    "pink",
    "purple",
    "brown",
    "white",
    "black",
    "gray", "grey",
    "beige", "tan",
]

_SHAPE_KEYWORDS = [
    "long and curved", "long, curved", "curved",
    "cut open", "half-circle", "half circle",
    "round", "circular", "spherical",
    "oval", "elongated",
    "square", "rectangular",
    "flat",
    "irregular",
]


def _build_single_object_prompt(
    object_name: str, position_hint: Optional[str], attribute: str
) -> str:
    """Build a minimal natural-language question about one attribute of one object.

    Args:
        object_name: object name
        position_hint: optional relative-position description to help the model locate the object
        attribute: "color" or "shape"
    """
    hint_str = f" (located {position_hint})" if position_hint else ""
    template = _SINGLE_OBJECT_COLOR_PROMPT if attribute == "color" else _SINGLE_OBJECT_SHAPE_PROMPT
    return template.format(object_name=object_name, position_hint=hint_str)


def _parse_attribute_answer(raw_text: str, attribute: str) -> Optional[str]:
    """Extract a short attribute descriptor from the VLM's free-text answer via keyword match.

    First tries the predefined keyword library (more reliable and consistent); on miss
    falls back to taking the first few words of the answer (better than returning None
    and leaving the object with no description at all).

    Returns:
        A matched keyword, or the fallback first-few-words, or None (empty/unusable input).
    """
    if not raw_text:
        return None

    text_lower = raw_text.strip().lower()
    keywords = _COLOR_KEYWORDS if attribute == "color" else _SHAPE_KEYWORDS

    for kw in keywords:
        if kw in text_lower:
            return kw

    # Keyword library missed: fallback = first 3 words (strip common leading filler)
    cleaned = re.sub(r"^(it'?s|the .+? is|i see|this is)\s+", "", text_lower)
    words = cleaned.strip(" .,!?").split()
    if words:
        return " ".join(words[:3])
    return None


_DISAMBIGUATION_QA_PROMPT = (
    "You are a JSON-only visual perception API for a robot, not a chatbot, "
    "helping to resolve a referential ambiguity. You will be shown one to "
    "three photos of a tabletop scene. The human gave an ambiguous "
    "instruction and a clarification follow-up. Based on the visual "
    "appearance of the objects, decide which candidate object is meant.\n\n"
    "STRICT OUTPUT RULE: Output ONLY a single JSON object, nothing else. "
    "Your entire response must start with '{' and end with '}'.\n\n"
    'The JSON must have exactly this shape: {"best_match": "<object_name>", '
    '"confidence": <a number between 0.0 and 1.0>, "reasoning": "<short '
    'reason, one sentence>"}.\n'
    "<object_name> must be exactly one of the candidate object names given "
    "to you below — copy it verbatim, do not invent a new name.\n\n"
    "Remember: output ONLY the JSON object, nothing else."
)


# Utility functions
def crop_and_upscale_frame(
    frame: np.ndarray,
    crop_box: tuple[float, float, float, float],
    target_size: int,
) -> np.ndarray:
    """Crop a fractional region of the image and upscale it to the target resolution (LANCZOS).

    Args:
        frame: (H, W, 3) uint8 numpy array
        crop_box: (top, bottom, left, right), values in 0.0~1.0
        target_size: target side length after upscaling (square output)

    Returns:
        (target_size, target_size, 3) uint8 numpy array
    """
    from PIL import Image

    h, w = frame.shape[:2]
    top, bottom, left, right = crop_box

    y0 = max(0, min(h - 1, int(round(top * h))))
    y1 = max(y0 + 1, min(h, int(round(bottom * h))))
    x0 = max(0, min(w - 1, int(round(left * w))))
    x1 = max(x0 + 1, min(w, int(round(right * w))))

    cropped = frame[y0:y1, x0:x1]
    img = Image.fromarray(cropped.astype(np.uint8))
    img_resized = img.resize((target_size, target_size), Image.LANCZOS)
    return np.array(img_resized)


def _extract_json_block(raw_text: str) -> Optional[dict]:
    """Robustly extract a JSON object from the VLM's raw text output.

    Tries in order: parse the whole string -> strip markdown code fences and parse ->
    take the substring from the first "{" to the last "}" and parse. Returns None if all fail.
    """
    if not raw_text:
        return None

    candidates = [raw_text.strip()]

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw_text, re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(raw_text[first_brace : last_brace + 1])

    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    return None


# Core class
class VisualPerceptionModule:
    """
    Wraps SmolVLM2 loading, on-demand GPU transfer, inference, JSON parsing + retry.

    Usage:
        vp = VisualPerceptionModule(VisualPerceptionConfig())

        desc = vp.scan_scene(
            frames=[frame_wrist, frame_left],
            object_hint_names=[
                ObjectHint("banana", "left side"),
                ObjectHint("apple_1", "between banana and plate"),
            ],
        )
        # desc looks like {"banana": "bright yellow, long and curved", ...}

        answer = vp.disambiguate(
            frames=[frame_wrist],
            human_utterance="not that one, the other apple",
            candidate_objects=["apple_1", "apple_2"],
        )
        # answer looks like {"best_match": "apple_2", "confidence": 0.8, "reasoning": "..."}
    """

    def __init__(self, config: Optional[VisualPerceptionConfig] = None):
        self.config = config or VisualPerceptionConfig()
        self._model = None
        self._processor = None
        self._torch = None
        self._is_on_gpu = False
        self._target_device = None
        self._dtype = None

    # Load / unload
    def _lazy_import_deps(self):
        """Lazy-import heavy dependencies so this file can be imported without torch/transformers."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        return torch, AutoModelForImageTextToText, AutoProcessor

    def load(self):
        """Explicitly load the model onto CPU (moves to GPU on demand during inference)."""
        if self._model is not None:
            return

        torch, AutoModelForImageTextToText, AutoProcessor = self._lazy_import_deps()

        self._target_device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if self.config.device == "auto"
            else self.config.device
        )
        self._dtype = getattr(torch, self.config.torch_dtype, torch.float32)

        logger.info(f"Loading visual perception model: {self.config.model_id} ...")
        self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id, dtype=self._dtype
        )
        self._model.to("cpu")
        self._is_on_gpu = False
        logger.info("Visual perception model loaded (currently on CPU, moved to GPU on demand).")

    def _ensure_on_device_for_inference(self):
        if self._model is None:
            self.load()
        if self._target_device == "cuda" and not self._is_on_gpu:
            self._model.to("cuda")
            self._is_on_gpu = True

    def _offload_after_inference(self):
        if not self.config.offload_after_inference:
            return
        if self._is_on_gpu:
            self._model.to("cpu")
            self._is_on_gpu = False
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()

    # Low-level inference (with JSON parsing + retry)
    def _run_vlm_once(
        self,
        frames: list[np.ndarray],
        system_prompt: str,
        user_prompt: str,
        crop_boxes: Optional[list[Optional[tuple[float, float, float, float]]]] = None,
    ) -> str:
        """Run one VLM inference and return the raw text output (no JSON parsing).

        Args:
            crop_boxes: optional list of crop boxes paired with frames. None entries
                use config.default_crop_box. The overall behavior is gated by
                config.enable_crop_and_upscale.
        """
        # Must ensure the model is loaded (which sets self._torch) before reading self._torch.
        self._ensure_on_device_for_inference()
        torch = self._torch

        if self.config.enable_crop_and_upscale:
            processed_frames = []
            for i, f in enumerate(frames):
                box = crop_boxes[i] if crop_boxes is not None and i < len(crop_boxes) else None
                box = box or self.config.default_crop_box
                processed_frames.append(
                    crop_and_upscale_frame(f, box, self.config.upscale_target_size)
                )
        else:
            processed_frames = frames

        from PIL import Image

        pil_images = [Image.fromarray(f.astype(np.uint8)) for f in processed_frames]

        # Concatenate the system prompt into the user message text (rather than using
        # a separate system-role message): SmolVLM2's chat template does not reliably
        # support a standalone system role — the content gets dropped by the template,
        # and the format constraints never reach the model.
        combined_text = f"{system_prompt}\n\n---\n\n{user_prompt}" if system_prompt else user_prompt

        content = [{"type": "image"} for _ in pil_images]
        content.append({"type": "text", "text": combined_text})
        messages = [{"role": "user", "content": content}]

        # Two-step: first generate the plain-text template, then explicitly pass images
        # to the processor. (The "one-shot" apply_chat_template(tokenize=True) fails to
        # bind image data on some version combinations, throwing "tokens but no images".)
        prompt_text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(text=prompt_text, images=pil_images, return_tensors="pt")

        device = "cuda" if self._is_on_gpu else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,  # disable sampling for stable structured output
            )

        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        new_tokens = generated_ids[:, input_len:]
        output_text = self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return output_text

    def _run_vlm_with_json_retry(
        self,
        frames: list[np.ndarray],
        system_prompt: str,
        user_prompt: str,
        crop_boxes: Optional[list[Optional[tuple[float, float, float, float]]]] = None,
    ) -> tuple[Optional[dict], str]:
        """Run VLM inference, parse JSON, retry per config on failure."""
        last_raw_text = ""
        attempts = self.config.max_retries + 1

        for attempt in range(1, attempts + 1):
            try:
                last_raw_text = self._run_vlm_once(frames, system_prompt, user_prompt, crop_boxes)
            except Exception as e:
                logger.warning(f"[VLM] Inference call failed (attempt {attempt}/{attempts}): {e}")
                time.sleep(self.config.retry_backoff_seconds)
                continue

            parsed = _extract_json_block(last_raw_text)
            if parsed is not None:
                if attempt > 1:
                    logger.info(f"[VLM] JSON parsed successfully on attempt {attempt}")
                return parsed, last_raw_text

            logger.warning(
                f"[VLM] Attempt {attempt}/{attempts} produced non-JSON output; "
                f"raw snippet: {last_raw_text[:120]!r}"
            )
            if attempt < attempts:
                time.sleep(self.config.retry_backoff_seconds)

        logger.error(f"[VLM] Still unable to parse JSON after {attempts} attempts; using fallback")
        return None, last_raw_text

    # Public API 1: scene scan (called once after reset)
    def _ask_single_object_attribute(
        self,
        frames: list[np.ndarray],
        object_name: str,
        position_hint: Optional[str],
        attribute: str,
        crop_boxes: Optional[list[Optional[tuple[float, float, float, float]]]] = None,
    ) -> Optional[str]:
        """Ask a minimal single-attribute question about one object and return the keyword-matched short result.

        Does NOT ask the model for JSON — just asks one sentence expecting a 1-3 word
        answer, then extracts the result deterministically in Python. On failure (nothing
        extractable) retries per config.max_retries.
        """
        prompt = _build_single_object_prompt(object_name, position_hint, attribute)

        last_raw = ""
        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                last_raw = self._run_vlm_once(frames, system_prompt="", user_prompt=prompt, crop_boxes=crop_boxes)
            except Exception as e:
                logger.warning(
                    f"[VLM] {object_name}.{attribute} inference failed (attempt {attempt}/{attempts}): {e}"
                )
                time.sleep(self.config.retry_backoff_seconds)
                continue

            result = _parse_attribute_answer(last_raw, attribute)
            if result:
                return result

            if attempt < attempts:
                time.sleep(self.config.retry_backoff_seconds)

        logger.warning(
            f"[VLM] {object_name}.{attribute} produced no valid result after {attempts} attempts; "
            f"last raw output: {last_raw[:80]!r}"
        )
        return None

    def scan_scene(
        self,
        frames: list[np.ndarray],
        object_hint_names: Optional[list] = None,
        crop_boxes: Optional[list[Optional[tuple[float, float, float, float]]]] = None,
    ) -> dict[str, str]:
        """
        Produce a structured visual description for the current scene.

        Implementation: for each object in the list, ask two minimal questions
        ("what color" and "what shape") separately, rather than asking the model
        for a JSON covering all objects at once. Each prompt is constant-length,
        does not grow with object count, and combined with deterministic keyword
        matching this avoids the instability of asking a small model to emit
        complex JSON.

        Cost: 2N calls (N objects), slower than a single JSON call, but each call
        is faster (short prompt + short output) and there are no missing entries
        or format collapses.

        Args:
            frames: 1~3 (H,W,3) uint8 numpy images, recommended order
                [wrist_image main view, image left, second_image right].
            object_hint_names: known objects in the scene, from simulator
                ground-truth coordinates (not VLM guesses). Accepts list[str]
                (names) or list[ObjectHint] (name + relative position;
                recommended, helps the model locate small objects via textual
                hints — though the current implementation mainly relies on the
                simplicity of one-object-per-question). If not provided, this
                function cannot work (it needs to know which objects to ask about).
            crop_boxes: optional per-frame crop boxes.

        Returns:
            dict, e.g. {"banana": "yellow, long and curved", ...}. An object
            appears in the result as long as at least one of color/shape was
            recognized; objects with neither are omitted.
        """
        hints = _normalize_object_hints(object_hint_names)
        if not hints:
            logger.warning("[VLM] scan_scene received no object_hint_names; cannot ask per object; returning empty dict")
            return {}

        merged: dict[str, str] = {}
        for hint in hints:
            color = self._ask_single_object_attribute(
                frames, hint.name, hint.relative_position, "color", crop_boxes
            )
            shape = self._ask_single_object_attribute(
                frames, hint.name, hint.relative_position, "shape", crop_boxes
            )
            if color and shape:
                merged[hint.name] = f"{color}, {shape}"
            elif color:
                merged[hint.name] = color
            elif shape:
                merged[hint.name] = shape
            else:
                logger.warning(f"[VLM] Object '{hint.name}': neither color nor shape recognized")

        self._offload_after_inference()
        return merged

    # ------------------------------------------------------------------
    # Public API 2: disambiguation Q&A (called on demand when an instruction is ambiguous)
    # ------------------------------------------------------------------

    def disambiguate(
        self,
        frames: list[np.ndarray],
        human_utterance: str,
        candidate_objects: list[str],
        scene_description: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """
        Given a potentially ambiguous human utterance, decide the referent based
        on the current frame(s) and a candidate object list.

        Args:
            frames: 1~3 current frames; recommended order same as scan_scene
            human_utterance: the raw human instruction / clarification
            candidate_objects: candidate object names
            scene_description: optional cached scan_scene() result providing
                color/shape priors to reduce redundant recognition

        Returns:
            dict, e.g. {"best_match": "apple_2", "confidence": 0.8, "reasoning": "..."}.
            On failure returns {"best_match": None, "confidence": 0.0, "reasoning": "parse_failed"}.
        """
        user_prompt = (
            f'Human said: "{human_utterance}"\n'
            f"Candidate objects (choose exactly one): {candidate_objects}\n"
        )
        if scene_description:
            relevant = {k: v for k, v in scene_description.items() if k in candidate_objects}
            if relevant:
                user_prompt += f"Known visual descriptions: {json.dumps(relevant, ensure_ascii=False)}\n"

        parsed, raw_text = self._run_vlm_with_json_retry(frames, _DISAMBIGUATION_QA_PROMPT, user_prompt)
        self._offload_after_inference()

        if parsed is None or "best_match" not in parsed:
            logger.warning("[VLM] Disambiguation failed; caller should fall back to a default strategy (e.g. nearest candidate)")
            return {"best_match": None, "confidence": 0.0, "reasoning": "parse_failed"}

        conf = parsed.get("confidence", 0.0)
        try:
            conf = max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            conf = 0.0
        parsed["confidence"] = conf

        if parsed.get("best_match") not in candidate_objects:
            logger.warning(
                f"[VLM] Returned best_match='{parsed.get('best_match')}' "
                f"is not in the candidate list {candidate_objects}; treating as parse failure"
            )
            return {"best_match": None, "confidence": 0.0, "reasoning": "invalid_candidate"}

        return parsed


# ======================================================================
# Standalone self-test entry
# ======================================================================

def _make_placeholder_frame(color: tuple[int, int, int] = (200, 150, 100)) -> np.ndarray:
    """Generate a solid-color placeholder image for pipeline self-tests without a real simulator frame."""
    h, w = 384, 384
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def _selftest_json_parsing_only():
    """No torch/transformers dependency; tests only the JSON extraction fault tolerance."""
    print("=== Self-test: JSON extraction fault tolerance (no GPU/model needed) ===")
    cases = [
        '{"banana": "yellow, long"}',
        'Sure! Here is the result:\n{"banana": "yellow, long"}\nHope that helps.',
        '```json\n{"banana": "yellow, long"}\n```',
        'garbage with no json at all',
        '{"banana": "yellow, long", }',  # trailing comma; standard json should fail
    ]
    for i, raw in enumerate(cases, 1):
        result = _extract_json_block(raw)
        status = "parse OK" if result is not None else "parse FAILED (expected; tests fallback path)"
        print(f"  case {i}: {status} -> {result}")
    print("Self-test complete.\n")


def _selftest_full_pipeline():
    """Full-pipeline self-test (solid-color placeholder): verifies load/infer/parse chain runs without errors."""
    print("=== Self-test: full VLM load + inference pipeline (needs torch/transformers, will download weights) ===")
    config = VisualPerceptionConfig(max_retries=2)
    vp = VisualPerceptionModule(config)

    frame_wrist = _make_placeholder_frame((200, 60, 60))
    frame_left = _make_placeholder_frame((230, 200, 50))

    print("Scanning placeholder scene (solid-color patches, just to verify the call chain and JSON parsing)...")
    desc = vp.scan_scene([frame_wrist, frame_left], object_hint_names=["apple", "banana"])
    print(f"Scene description result: {desc}")
    print("Self-test complete.\n")


def _selftest_from_real_images(
    image_paths: list[str],
    object_hints: Optional[list[str]] = None,
    object_positions: Optional[list[str]] = None,
    crop_box: Optional[tuple[float, float, float, float]] = None,
    no_crop: bool = False,
):
    """Test scene scanning on real screenshot files (the key verification step).

    Examples:
        python visual_perception.py --selftest images --images frame.png --hints apple banana

        # With relative-position info (strong-constraint mode, recommended):
        # --positions must be paired one-to-one with --hints (equal lengths)
        python visual_perception.py --selftest images --images wrist_image.png \\
            --hints banana apple_1 --positions "left side" "right side, dark red"

        python visual_perception.py --selftest images --images wrist_image.png --no-crop
    """
    from PIL import Image as PILImage

    print(f"=== Self-test: scene scan from real screenshots ({len(image_paths)} image(s)) ===")
    for p in image_paths:
        print(f"  - {p}")

    frames = []
    for p in image_paths:
        img = PILImage.open(p).convert("RGB")
        frames.append(np.array(img))
        print(f"  loaded: {p}  size={img.size}")

    hints_for_scan: Optional[list] = None
    if object_hints:
        if object_positions:
            if len(object_positions) != len(object_hints):
                print(
                    f"Error: --positions provided {len(object_positions)} item(s), "
                    f"but --hints has {len(object_hints)}; counts must match."
                )
                raise SystemExit(1)
            hints_for_scan = [
                ObjectHint(name=n, relative_position=p) for n, p in zip(object_hints, object_positions)
            ]
            print("\nUsing object list with position info (strong-constraint mode):")
            for h in hints_for_scan:
                print(f"  {h.to_prompt_line()}")
        else:
            hints_for_scan = object_hints
            print(f"\nUsing name-only object list (no position info): {object_hints}")

    config = VisualPerceptionConfig(max_retries=2)
    if no_crop:
        config.enable_crop_and_upscale = False
        print("\nCrop + upscale preprocessing disabled; running inference on the raw image")
    elif crop_box is not None:
        config.default_crop_box = crop_box
        print(f"\nUsing custom crop box: {crop_box}")
    else:
        print(f"\nUsing default crop box: {config.default_crop_box}, upscaled to {config.upscale_target_size}px")

    vp = VisualPerceptionModule(config)

    print("\nCalling VLM to analyze the scene (first call loads the model to GPU; may take seconds to tens of seconds)...")
    desc = vp.scan_scene(frames, object_hint_names=hints_for_scan)

    print("\n========== Scene visual feature scan result ==========")
    if not desc:
        print("Parse failed; empty dict returned. Check the [VLM] logs above for the raw output.")
    else:
        for obj_name, obj_desc in desc.items():
            print(f"  {obj_name}: {obj_desc}")
    print("======================================================\n")

    return desc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visual_perception module self-test")
    parser.add_argument(
        "--selftest",
        choices=["json", "full", "images"],
        default="json",
        help=(
            "json = JSON parsing fault tolerance only (fast, no model needed); "
            "full = full pipeline with solid-color placeholder (slow, needs torch/transformers, downloads model); "
            "images = test recognition on real screenshot files (recommended, use with --images)"
        ),
    )
    parser.add_argument(
        "--images", nargs="+", default=None,
        help="(--selftest images only) 1~3 image paths, recommended order: wrist_image image second_image",
    )
    parser.add_argument(
        "--hints", nargs="+", default=None,
        help="(--selftest images only) object name list, e.g. apple banana pear plate",
    )
    parser.add_argument(
        "--positions", nargs="+", default=None,
        help='(--selftest images only) relative-position descriptions paired one-to-one with --hints, e.g. '
        '--positions "left side" "center, between banana and plate"',
    )
    parser.add_argument(
        "--crop-box", nargs=4, type=float, default=None,
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        help="(--selftest images only) custom crop box, 4 values in 0~1",
    )
    parser.add_argument(
        "--no-crop", action="store_true",
        help="(--selftest images only) disable crop + upscale preprocessing; use raw image",
    )
    args = parser.parse_args()

    if args.selftest == "json":
        _selftest_json_parsing_only()
    elif args.selftest == "full":
        _selftest_json_parsing_only()
        _selftest_full_pipeline()
    elif args.selftest == "images":
        if not args.images:
            print("Error: --selftest images mode needs at least one image path via --images.")
            raise SystemExit(1)
        crop_box_tuple = tuple(args.crop_box) if args.crop_box else None
        _selftest_from_real_images(
            args.images,
            object_hints=args.hints,
            object_positions=args.positions,
            crop_box=crop_box_tuple,
            no_crop=args.no_crop,
        )

