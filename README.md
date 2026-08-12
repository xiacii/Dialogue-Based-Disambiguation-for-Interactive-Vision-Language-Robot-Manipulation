# Interactive Task Learning: Dialogue-Based Disambiguation for Vision-Language Robot Manipulation

MSc Dissertation project materials.

**Project title:** Human-robot Interfaces for Interactive Task Learning
**Author (UID):** S2792628

This archive contains the source code for a human-robot interface that lets a
user command a **simulated Franka arm** (VLABench + MuJoCo) in natural language
to pick and place objects. A **local-LLM dialogue mediator** turns vague or
spoken-style instructions into a human-readable reply plus a clear robot command,
and handles disambiguation, clarification, and error correction through dialogue.

---

## 1. What is in this archive

```
.
├── README.md                     # this file
├── test_sim_env.py               # MAIN program: simulation + Gradio web UI + dialogue bridge
├── dialogue/                     # dialogue mediator package
│   ├── __init__.py
│   ├── llm_interface.py          # HTTP calls to a local Ollama LLM
│   ├── llm_reasoner.py           # prompt + JSON parsing (utterance -> reply + robot_signal)
│   ├── dialogue_manager.py       # orchestrates a dialogue turn
│   └── state_manager.py          # dialogue state + live object manifest
├── color_extraction.py           # perception: object colours (world->pixel projection + HSV)
├── shape_extraction.py           # perception: object shapes (geometry-first + silhouette)
├── spatial_relations.py          # perception: inter-object directions, table regions, placement parsing
├── visual_perception.py          # optional VLM-based perception module (SmolVLM2, self-test only)
├── grab_frames.py                # debug: save the three camera views of a scene as PNGs
└── probe_camera_projection.py    # debug: verify world->pixel camera projection
```

> Note: the `dialogue/` package files are listed above; if your extracted copy
> shows them at the top level instead, place them inside a `dialogue/` folder so
> `from dialogue.state_manager import ...` resolves.

---

## 2. External dependencies (NOT included in this archive)

The following are large third-party frameworks and are **deliberately excluded**
from the archive. They must be installed / obtained separately:

- **LeRobot** — provides `SmolVLAPolicy`, `make_pre_post_processors`, and the
  `VLABenchEnv` wrapper (`from lerobot...` in `test_sim_env.py`).
- **VLABench** — the benchmark/scene assets and MuJoCo tasks used by `VLABenchEnv`.
- **RRT-algorithms** — added to `sys.path` at the top of `test_sim_env.py`.
- **Ollama** — local LLM server used by the dialogue mediator (see Section 5).

Python packages used directly by the code:
`torch`, `mujoco`, `dm_control`, `numpy`, `gradio`, `requests`, `opencv-python`,
`Pillow`. `visual_perception.py` additionally uses `transformers` (only if you
run that optional module).

---

## 3. Installation

1. Create and activate a Python environment (the code was developed under a
   `lerobot`-style environment, Python 3.12):

   ```bash
   conda create -n lerobot312 python=3.12 -y
   conda activate lerobot312
   ```

2. Install LeRobot, VLABench, and RRT-algorithms following their own
   instructions (these are the external dependencies above).

3. Install the remaining Python packages:

   ```bash
   pip install torch mujoco dm_control numpy gradio requests opencv-python Pillow
   # only if you want to run visual_perception.py:
   pip install transformers
   ```

4. **Fix the hard-coded paths.** `test_sim_env.py` contains absolute paths from
   the development machine that you must edit to match your setup.

---

## 4. Running the main program

The main deliverable is `test_sim_env.py`. It launches the MuJoCo simulation
(main thread) and a Gradio web UI (background thread):

```bash
conda activate lerobot312
python test_sim_env.py
```

Then open the Gradio URL printed in the terminal (default
`http://127.0.0.1:7860`). In the web UI you can:

- watch the live camera view and switch between cameras;
- type natural-language commands, e.g. `pick up the red one`, `grab any fruit`,
  `put the apple on the left side`, `put it back`, `pick up all the items`;
- teach an alias, e.g. `the green one is helen`, then later say `pick up helen`;
- correct a wrong pick, e.g. `wait, actually pick up the yellow one`;
- type `reset` (or press the button) for a new random scene.

Make sure the Ollama server is running first (Section 5), otherwise the dialogue
mediator cannot respond.

---

## 5. Ollama / LLM configuration

The dialogue mediator talks to a **local Ollama server**. The endpoint and model
are set in `dialogue/llm_interface.py`:

```python
OLLAMA_URL = "http://localhost:11434/api/generate"
# model: "mistral", request timeout: 20s
```

To configure:

1. Install Ollama and start the server (it listens on `localhost:11434`):
   ```bash
   ollama serve
   ```
2. Pull the model used by the code:
   ```bash
   ollama pull mistral
   ```
3. To use a different model or host, edit `OLLAMA_URL` / the `"model"` field in
   `dialogue/llm_interface.py`.

No API key is required; everything runs locally.

---

## 6. Which script does what

**Simulation + interface (the system):**
- `test_sim_env.py` — main entry point. State-machine pick-and-place control of
  the Franka arm, the Gradio UI, and the bridge to the dialogue mediator.
- `dialogue/` — the mediator: `llm_interface` (Ollama HTTP), `llm_reasoner`
  (prompt + JSON parse), `dialogue_manager` (turn orchestration),
  `state_manager` (state + live object manifest).

**Perception (used by the main program at runtime):**
- `color_extraction.py` — per-object colour via camera projection + HSV sampling.
- `shape_extraction.py` — per-object shape (geometry-first, plus silhouette cue).
- `spatial_relations.py` — inter-object directions, table-region classification,
  and natural-language placement parsing (`put X on the left`, etc.).

**Optional / experimental perception:**
- `visual_perception.py` — a VLM-based perception module (SmolVLM2). Not required
  by the main program; run only via its self-test (below).

**Debugging utilities (standalone, do not affect the main program):**
- `grab_frames.py` — start one scene and save all three camera views as PNGs.
- `probe_camera_projection.py` — verify that world coordinates project correctly
  to pixel coordinates for the working camera.

---

## 7. Running the perception / debugging scripts

Several modules have a `--selftest` entry point so a marker can check them in
isolation (the pure-logic ones need no simulator or model):

```bash
# Pure-logic self-tests (fast, no simulator, no model):
python spatial_relations.py --selftest
python shape_extraction.py --selftest synthetic
python color_extraction.py --selftest hsv           # pure HSV logic test (default)

# Debug utilities (need VLABench/MuJoCo installed):
python grab_frames.py --out debug_frames --seed 42
python probe_camera_projection.py

# Optional VLM module self-tests (transformers required; 'images' needs real frames):
python visual_perception.py --selftest json
python visual_perception.py --selftest images --images frame.png --hints apple banana
```

Use `python <script>.py --help` for the full set of arguments where available.

---

## 8. Notes for the marker

- The main program (`test_sim_env.py`) requires all four external dependencies
  (LeRobot, VLABench, RRT, Ollama) to be installed and the absolute paths in
  Section 3 to be corrected.
- If only a quick check is needed without the full stack, the `--selftest`
  commands for `spatial_relations.py` and `shape_extraction.py` run standalone.
- The dialogue mediator uses a local LLM, so its exact wording can vary between
  runs; the deterministic layer (name/colour/shape/alias matching plus scene
  validation) is what guarantees the robot never acts on an unresolved target.
