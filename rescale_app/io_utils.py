"""File-system helpers for loading/saving transforms and batch-applying them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from . import core

RESCALED_SUFFIX = "_rescaled"


def load_transform(json_path: str | Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def save_transform(transform: dict, json_path: str | Path) -> None:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(transform, f, indent=2)


def list_stl_files(folder: str | Path) -> list[Path]:
    return sorted(Path(folder).glob("*.stl"))


def default_output_folder(input_folder: str | Path) -> Path:
    input_folder = Path(input_folder)
    return input_folder.parent / (input_folder.name + RESCALED_SUFFIX)


def default_output_file(input_file: str | Path) -> Path:
    input_file = Path(input_file)
    return input_file.with_name(f"{input_file.stem}{RESCALED_SUFFIX}{input_file.suffix}")


def apply_transform_to_folder(
    transform: dict,
    input_folder: str | Path,
    output_folder: str | Path,
    progress_cb: Optional[Callable[[int, int, Path], None]] = None,
) -> list[Path]:
    """Apply ``transform`` to every .stl in ``input_folder``.

    Writes outputs (same filenames) into ``output_folder`` and returns
    the list of output paths. ``progress_cb(index, total, stl_path)`` is
    called before processing each file, if given.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    stl_files = list_stl_files(input_folder)
    if not stl_files:
        raise FileNotFoundError(f"No .stl files found in '{input_folder}'")

    output_folder.mkdir(parents=True, exist_ok=True)

    outputs = []
    total = len(stl_files)
    for i, stl in enumerate(stl_files):
        if progress_cb:
            progress_cb(i, total, stl)
        out_path = output_folder / stl.name
        core.apply_transform(transform, stl, out_path)
        outputs.append(out_path)
    return outputs
