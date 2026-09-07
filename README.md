# STL Rescale Toolkit

A Qt GUI (plus reusable Python library) for aligning and non-uniformly
rescaling brain-region STLs (e.g. the Allen CCF whole brain and its
region parts) into a mouse skull STL, using a 3-point landmark method
(Bregma, Lambda, Ventral). It's a GUI wrapper around the workflow
originally implemented in
`objective_functionnality/skull_addapt.py`.

## Installation (uv only, no conda)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10+ (uv will
fetch a matching Python automatically if needed).

```bash
cd 3_Combine/rescale_stl
uv sync
```

This creates a local `.venv` and installs everything from
`pyproject.toml`: `numpy`, `trimesh`, `pyvista`, `pyvistaqt`, `PySide6`.

## Running

```bash
uv run rescale-stl          # installed console script
# or
uv run app.py                # convenience launcher script
# or
uv run python -m rescale_app
```

## Functionality

The app has two tabs:

### 1. Generate Transform

Pick landmarks on a skull and a brain STL and compute/save the
rescaling transform as a JSON file:

1. Select the **skull STL**, **brain STL**, and **output JSON** paths,
   and optionally adjust the **cavity scale** (fraction of the skull's
   bounding box the brain is scaled to fill; default `0.90`).
2. Click **Start Landmark Picking**. The skull is loaded into the
   embedded 3D view; click, in order, on:
   - **Bregma** — coronal/sagittal suture junction (dorsal, midline, anterior)
   - **Lambda** — lambdoid/sagittal suture junction (dorsal, midline, posterior)
   - **Ventral** — a ventral midline point (e.g. base of brainstem / basisphenoid)

   Once the 3 skull points are placed, the brain STL loads
   automatically and the same 3 landmarks are picked on it.
   **Undo Last Point** / **Reset Picking** are available throughout.
3. Click **Compute Transform** to build the 4x4 rotation + non-uniform
   scale + translation matrix that maps the brain onto the skull. The
   resulting scale/translation are shown in the summary box.
4. **Preview Result** shows the skull and the transformed brain
   together in the same embedded view.
5. **Save JSON** writes the transform to the chosen output path.

The saved JSON schema matches the original CLI script's
`align_config.json` / `brain_to_skull_transform.json`, so files are
interchangeable between the two tools.

### 2. Batch Rescale

Apply a previously saved transform JSON to every `.stl` file in a
folder:

1. Select the **input folder** (containing the STLs to transform) and
   the **transform JSON** (from step 1, or from the original CLI
   script).
2. Optionally set an **output folder** — if left blank, it defaults to
   `<input folder name>_rescaled` next to the input folder.
3. Click **Run Rescale**. Processing runs on a background thread with
   a progress bar and a per-file log, so the UI stays responsive.
4. **View Result** opens a 3D preview window with the skull (if its
   path from the transform JSON still resolves) and all the
   transformed output meshes.

## Library layout

```
rescale_app/
  core.py        # pure geometry/transform math (no UI, no picking)
  io_utils.py     # transform JSON load/save, folder listing, batch apply
  gui/
    pick_viewer.py    # embedded PyVista/Qt widget with ordered point picking
    generate_tab.py   # Tab 1
    batch_tab.py       # Tab 2
    main_window.py     # tab container
```

`core.py` and `io_utils.py` have no Qt dependency and can be reused
from scripts or notebooks, e.g.:

```python
import trimesh
from rescale_app import core, io_utils

skull = trimesh.load("skull.stl")
brain = trimesh.load("brain.stl")
lm = core.Landmarks(
    bregma_skull=..., lambda_skull=..., ventral_skull=...,
    bregma_brain=..., lambda_brain=..., ventral_brain=...,
)
transform = core.compute_transform(skull, brain, lm, skull_path="skull.stl", brain_path="brain.stl")
io_utils.save_transform(transform, "align_config.json")
io_utils.apply_transform_to_folder(transform, "brain_parts/", "brain_parts_rescaled/")
```
