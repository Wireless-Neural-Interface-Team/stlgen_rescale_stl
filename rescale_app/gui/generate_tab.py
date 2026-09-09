"""T1 — pick landmarks on skull + brain and save the rescaling transform."""

from __future__ import annotations

import numpy as np
import trimesh
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import core, io_utils
from .pick_viewer import (
    BONE_COLOR,
    BONE_OPACITY_PICK,
    BONE_OPACITY_VIEW,
    BRAIN_COLOR,
    BRAIN_OPACITY_TRANSPARENT,
    BRAIN_OPACITY_VIEW,
    PickViewer,
    trimesh_to_pv,
)

LANDMARK_LABELS = ["Bregma", "Lambda", "Ventral"]


def _path_row(label_text: str, browse_slot, filter_: str = "STL files (*.stl)") -> tuple[QHBoxLayout, QLineEdit]:
    row = QHBoxLayout()
    row.addWidget(QLabel(label_text))
    line = QLineEdit()
    row.addWidget(line, 1)
    btn = QPushButton("Browse…")
    btn.clicked.connect(lambda: browse_slot(line, filter_))
    row.addWidget(btn)
    return row, line


class AxisControl(QWidget):
    """A slider + precise spinbox pair for one coordinate axis, plus a
    Reset button. Resetting is the caller's responsibility (via
    `resetRequested`) since only the caller knows what "reset" means."""

    valueChanged = Signal(float)
    resetRequested = Signal()

    _SLIDER_STEPS = 2000

    def __init__(self, axis_label: str, parent=None):
        super().__init__(parent)
        self._lo, self._hi = -1.0, 1.0
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(axis_label)
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self._SLIDER_STEPS)
        layout.addWidget(self.slider, 1)

        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(3)
        self.spin.setFixedWidth(90)
        layout.addWidget(self.spin)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setFixedWidth(55)
        self.reset_btn.clicked.connect(self.resetRequested.emit)
        layout.addWidget(self.reset_btn)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin.valueChanged.connect(self._on_spin_changed)

    def set_range(self, lo: float, hi: float) -> None:
        if hi <= lo:
            hi = lo + 1.0
        self._lo, self._hi = lo, hi
        self._updating = True
        self.spin.setRange(lo, hi)
        self._updating = False

    def set_value(self, value: float) -> None:
        value = min(max(value, self._lo), self._hi)
        self._updating = True
        self.spin.setValue(value)
        frac = (value - self._lo) / (self._hi - self._lo)
        self.slider.setValue(int(round(frac * self._SLIDER_STEPS)))
        self._updating = False

    def value(self) -> float:
        return self.spin.value()

    def set_axis_label(self, text: str) -> None:
        self.label.setText(text)

    def _on_slider_changed(self, raw: int) -> None:
        if self._updating:
            return
        frac = raw / self._SLIDER_STEPS
        value = self._lo + frac * (self._hi - self._lo)
        self._updating = True
        self.spin.setValue(value)
        self._updating = False
        self.valueChanged.emit(value)

    def _on_spin_changed(self, value: float) -> None:
        if self._updating:
            return
        frac = (value - self._lo) / (self._hi - self._lo)
        self._updating = True
        self.slider.setValue(int(round(frac * self._SLIDER_STEPS)))
        self._updating = False
        self.valueChanged.emit(value)


class GenerateTab(QWidget):
    """Pick 3 paired landmarks (Bregma, Lambda, Ventral) and save the transform JSON."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._transform: dict | None = None
        self._transform_dirty = False  # a landmark was edited since the last compute
        self._last_committed_points: list[np.ndarray] | None = None  # brain points as of self._transform
        self._undo_stack: list[tuple[dict, list[np.ndarray]]] = []
        self._redo_stack: list[tuple[dict, list[np.ndarray]]] = []
        self._current_target: str | None = None  # "skull" | "brain"
        self._skull_points: list[np.ndarray] = []
        self._brain_points: list[np.ndarray] = []
        self._preview_camera_set = False  # only set a default camera view once per picking session
        self._vector_mode = False  # axis controls set a displacement vector instead of a new position
        self._skull_bounds: np.ndarray | None = None  # set whenever the preview is (re)shown

        self.viewer = PickViewer()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        self.viewer.point_picked.connect(self._on_point_picked)
        self.viewer.point_undone.connect(self._on_point_undone)
        self.viewer.picking_finished.connect(self._on_stage_finished)

        self._refresh_buttons()

    # ------------------------------------------------------------------ UI

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(380)
        layout = QVBoxLayout(panel)

        files_box = QGroupBox("Input / output files")
        files_layout = QVBoxLayout(files_box)
        row, self.skull_edit = _path_row("Skull STL", self._browse_file)
        files_layout.addLayout(row)
        row, self.brain_edit = _path_row("Brain STL", self._browse_file)
        files_layout.addLayout(row)
        row, self.output_edit = _path_row("Output JSON", self._browse_save_json,
                                          filter_="JSON files (*.json)")
        files_layout.addLayout(row)

        cavity_tooltip = (
            "Antero-posterior and dorso-ventral scale always come from the exact\n"
            "Bregma/Lambda/Ventral landmark distances. Those 3 points are all on\n"
            "the midline, so they carry no medio-lateral information -- this value\n"
            "is what sets the medio-lateral (left-right) scale instead: the brain's\n"
            "medio-lateral width becomes this fraction of the skull's own."
        )
        scale_row = QHBoxLayout()
        scale_label = QLabel("Cavity scale (medio-lateral)")
        scale_label.setToolTip(cavity_tooltip)
        scale_row.addWidget(scale_label)
        self.cavity_spin = QDoubleSpinBox()
        self.cavity_spin.setRange(0.10, 2.00)
        self.cavity_spin.setSingleStep(0.01)
        self.cavity_spin.setValue(core.CAVITY_SCALE_DEFAULT)
        self.cavity_spin.setToolTip(cavity_tooltip)
        scale_row.addWidget(self.cavity_spin)
        scale_row.addStretch(1)
        files_layout.addLayout(scale_row)

        layout.addWidget(files_box)

        picking_box = QGroupBox("Landmark picking")
        picking_layout = QVBoxLayout(picking_box)

        self.instructions_label = QLabel("Set the skull and brain STL paths, then start picking.")
        self.instructions_label.setWordWrap(True)
        picking_layout.addWidget(self.instructions_label)

        self.start_btn = QPushButton("Start Landmark Picking")
        self.start_btn.clicked.connect(self._on_start_picking_clicked)
        picking_layout.addWidget(self.start_btn)

        load_row = QHBoxLayout()
        self.load_skull_landmarks_btn = QPushButton("Load Skull Landmarks…")
        self.load_skull_landmarks_btn.clicked.connect(self._on_load_skull_landmarks)
        load_row.addWidget(self.load_skull_landmarks_btn)
        self.load_brain_landmarks_btn = QPushButton("Load Brain Landmarks…")
        self.load_brain_landmarks_btn.clicked.connect(self._on_load_brain_landmarks)
        load_row.addWidget(self.load_brain_landmarks_btn)
        picking_layout.addLayout(load_row)

        undo_row = QHBoxLayout()
        self.undo_btn = QPushButton("Undo Last Point")
        self.undo_btn.clicked.connect(self.viewer.undo)
        undo_row.addWidget(self.undo_btn)
        self.reset_btn = QPushButton("Reset Picking")
        self.reset_btn.clicked.connect(self._on_reset_picking)
        undo_row.addWidget(self.reset_btn)
        picking_layout.addLayout(undo_row)

        layout.addWidget(picking_box)

        refine_box = QGroupBox("Refine Brain Landmark (in preview)")
        refine_layout = QVBoxLayout(refine_box)

        self.refine_combo = QComboBox()
        self.refine_combo.addItems(LANDMARK_LABELS)
        self.refine_combo.currentIndexChanged.connect(self._on_refine_point_selected)
        refine_layout.addWidget(self.refine_combo)

        transparency_row = QHBoxLayout()
        self.skull_transparency_check = QCheckBox("Skull transparency")
        self.skull_transparency_check.setChecked(True)
        self.skull_transparency_check.toggled.connect(self._on_transparency_toggled)
        transparency_row.addWidget(self.skull_transparency_check)

        self.brain_transparency_check = QCheckBox("Brain transparency")
        self.brain_transparency_check.setChecked(False)
        self.brain_transparency_check.toggled.connect(self._on_transparency_toggled)
        transparency_row.addWidget(self.brain_transparency_check)
        refine_layout.addLayout(transparency_row)

        self.vector_mode_check = QCheckBox("Vector")
        self.vector_mode_check.setToolTip(
            "When checked, the X/Y/Z controls below set a displacement\n"
            "(dx, dy, dz) applied to the landmark's position as of the last\n"
            "Compute Transform, instead of setting its absolute new\n"
            "coordinates: the new position is (x0-dx, y0-dy, z0-dz). An\n"
            "arrow is drawn from the landmark's old position to the new one."
        )
        self.vector_mode_check.toggled.connect(self._on_vector_mode_toggled)
        refine_layout.addWidget(self.vector_mode_check)

        self.axis_x = AxisControl("X")
        self.axis_y = AxisControl("Y")
        self.axis_z = AxisControl("Z")
        for i, axis_ctrl in enumerate((self.axis_x, self.axis_y, self.axis_z)):
            axis_ctrl.valueChanged.connect(self._on_axis_value_changed)
            axis_ctrl.resetRequested.connect(lambda axis=i: self._on_reset_axis(axis))
            refine_layout.addWidget(axis_ctrl)

        self.refine_box = refine_box
        self.refine_box.setEnabled(False)
        layout.addWidget(refine_box)

        transform_box = QGroupBox("Transform")
        transform_layout = QVBoxLayout(transform_box)

        compute_row = QHBoxLayout()
        self.compute_btn = QPushButton("Compute Transform")
        self.compute_btn.clicked.connect(self._on_compute)
        compute_row.addWidget(self.compute_btn, 1)
        self.transform_undo_btn = QPushButton("Undo")
        self.transform_undo_btn.clicked.connect(self._on_undo_transform)
        compute_row.addWidget(self.transform_undo_btn)
        self.transform_redo_btn = QPushButton("Redo")
        self.transform_redo_btn.clicked.connect(self._on_redo_transform)
        compute_row.addWidget(self.transform_redo_btn)
        transform_layout.addLayout(compute_row)

        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(140)
        transform_layout.addWidget(self.summary_text)

        self.save_btn = QPushButton("Save JSON")
        self.save_btn.clicked.connect(self._on_save)
        transform_layout.addWidget(self.save_btn)

        layout.addWidget(transform_box)
        layout.addStretch(1)
        return panel

    # --------------------------------------------------------------- browse

    def _browse_file(self, line: QLineEdit, filter_: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", filter_)
        if path:
            line.setText(path)

    def _browse_save_json(self, line: QLineEdit, filter_: str) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save transform as", "", filter_)
        if path:
            if not path.lower().endswith(".json"):
                path += ".json"
            line.setText(path)

    # -------------------------------------------------------------- picking

    def _on_start_picking_clicked(self) -> None:
        """'Start Landmark Picking' button handler: confirms before
        discarding an already-computed transform."""
        if self._transform is not None:
            reply = QMessageBox.warning(
                self, "Discard current results?",
                "A transform has already been computed. Starting landmark "
                "picking again will discard it, along with the current "
                "picked landmarks. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._on_start_picking()

    def _on_start_picking(self) -> None:
        skull_path = self.skull_edit.text().strip()
        brain_path = self.brain_edit.text().strip()
        if not skull_path or not brain_path:
            QMessageBox.warning(self, "Missing files", "Select both a skull and a brain STL first.")
            return

        self._transform = None
        self._transform_dirty = False
        self._last_committed_points = None
        self._undo_stack = []
        self._redo_stack = []
        self._skull_points = []
        self._brain_points = []
        self._current_target = "skull"
        self._preview_camera_set = False

        self.viewer.clear()
        try:
            self.viewer.show_mesh(skull_path, color=BONE_COLOR, opacity=BONE_OPACITY_PICK)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load skull", str(exc))
            self._current_target = None
            return
        self.viewer.start_picking(LANDMARK_LABELS)
        self._update_instructions()
        self._refresh_buttons()

    def _on_reset_picking(self) -> None:
        if self._current_target is None:
            return
        self._on_start_picking()

    def _load_landmarks_from_file(self, keys: tuple[str, str, str], side: str) -> list[np.ndarray] | None:
        """Prompt for a transform JSON and return `side`'s 3 landmark
        points (in `keys` order). Shows an error and returns None on
        failure. Does NOT touch the skull/brain STL path fields -- the
        points are meant to be applied to whatever mesh is currently set,
        not the mesh the JSON happened to be saved against."""
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load {side} landmarks from transform JSON", "", "JSON files (*.json)")
        if not path:
            return None
        try:
            transform = io_utils.load_transform(path)
            lm_json = transform["landmarks"]
            values = [lm_json[k] for k in keys]
            if any(v is None for v in values):
                raise ValueError(f"one or more of {side}'s Bregma/Lambda/Ventral is missing")
            return [np.array(v, dtype=float) for v in values]
        except Exception as exc:
            QMessageBox.critical(
                self, "Failed to load landmarks",
                f"Could not read {side} Bregma/Lambda/Ventral landmarks from:\n{path}\n\n{exc}",
            )
            return None

    def _on_load_skull_landmarks(self) -> None:
        points = self._load_landmarks_from_file(
            ("bregma_skull", "lambda_skull", "ventral_skull"), "skull")
        if points is not None:
            self._apply_loaded_points("skull", points)

    def _on_load_brain_landmarks(self) -> None:
        points = self._load_landmarks_from_file(
            ("bregma_brain", "lambda_brain", "ventral_brain"), "brain")
        if points is not None:
            self._apply_loaded_points("brain", points)

    def _apply_loaded_points(self, side: str, points: list[np.ndarray]) -> None:
        """Install `points` as the current Bregma/Lambda/Ventral for
        `side` ("skull" or "brain"), applied to whatever STL is currently
        set for that side. Any existing transform is now stale, so it's
        cleared. If the other side isn't filled yet (3 points, from a
        prior pick or a prior load), start interactive picking on it;
        otherwise both sides are ready for Compute Transform."""
        self.viewer.clear()
        self._transform = None
        self._transform_dirty = False
        self._last_committed_points = None
        self._undo_stack = []
        self._redo_stack = []
        self._preview_camera_set = False

        if side == "skull":
            self._skull_points = points
            other_side, other_edit = "brain", self.brain_edit
            other_ready = len(self._brain_points) == 3
        else:
            self._brain_points = points
            other_side, other_edit = "skull", self.skull_edit
            other_ready = len(self._skull_points) == 3

        if other_ready:
            self._current_target = "done"
        else:
            if other_side == "skull":
                self._skull_points = []
            else:
                self._brain_points = []
            if not other_edit.text().strip():
                QMessageBox.warning(
                    self, "Missing file",
                    f"Loaded the {side} landmarks. Set the {other_side} STL path, "
                    f"then pick its landmarks (or load them too).",
                )
                self._current_target = None
            elif not self._begin_picking_side(other_side, other_edit):
                self._current_target = None

        self._update_instructions()
        self._refresh_buttons()

    def _on_point_picked(self, idx: int, label: str, point) -> None:
        target_points = self._skull_points if self._current_target == "skull" else self._brain_points
        # keep list in sync with the viewer's own picked-point buffer
        target_points.clear()
        target_points.extend(self.viewer.picked_points())
        self._update_instructions()

    def _on_point_undone(self, idx: int, label: str) -> None:
        target_points = self._skull_points if self._current_target == "skull" else self._brain_points
        target_points.clear()
        target_points.extend(self.viewer.picked_points())
        self._update_instructions()
        self._refresh_buttons()

    def _on_stage_finished(self) -> None:
        # Whichever side just finished, the other side might already be
        # filled in (loaded from JSON rather than picked) -- don't assume
        # skull always comes before brain.
        if self._current_target == "skull":
            other_side, other_ready = "brain", len(self._brain_points) == 3
        else:
            other_side, other_ready = "skull", len(self._skull_points) == 3

        if other_ready:
            self._current_target = "done"
        else:
            self.viewer.clear()
            other_edit = self.brain_edit if other_side == "brain" else self.skull_edit
            if not self._begin_picking_side(other_side, other_edit):
                self._current_target = None
        self._update_instructions()
        self._refresh_buttons()

    def _begin_picking_side(self, side: str, path_edit: QLineEdit) -> bool:
        """Load `path_edit`'s mesh and start picking on it, setting
        `_current_target = side`. Returns False (after showing an error)
        on failure; the caller is responsible for `viewer.clear()`."""
        path = path_edit.text().strip()
        if not path:
            return False
        try:
            self.viewer.show_mesh(path, color=BONE_COLOR, opacity=BONE_OPACITY_PICK)
        except Exception as exc:
            QMessageBox.critical(self, f"Failed to load {side}", str(exc))
            return False
        self.viewer.start_picking(LANDMARK_LABELS)
        self._current_target = side
        return True

    def _update_instructions(self) -> None:
        if self._current_target in (None, "done"):
            if self._current_target == "done":
                self.instructions_label.setText(
                    "All 6 landmarks picked (skull + brain). Click 'Compute Transform'."
                )
            return
        next_label = self.viewer.next_label()
        target_name = "SKULL" if self._current_target == "skull" else "BRAIN"
        if next_label:
            self.instructions_label.setText(f"Picking on {target_name}: click '{next_label}'.")
        else:
            self.instructions_label.setText(f"{target_name} landmarks complete.")

    def _refresh_buttons(self) -> None:
        picking_active = self._current_target in ("skull", "brain")
        self.undo_btn.setEnabled(picking_active or self._current_target == "done")
        self.reset_btn.setEnabled(self._current_target is not None)
        have_all = (
            len(self._skull_points) == 3 and len(self._brain_points) == 3
            and self._current_target == "done"
        )
        self.compute_btn.setEnabled(have_all)
        self.transform_undo_btn.setEnabled(bool(self._undo_stack))
        self.transform_redo_btn.setEnabled(bool(self._redo_stack))
        self.save_btn.setEnabled(self._transform is not None and not self._transform_dirty)
        self.refine_box.setEnabled(self._transform is not None)

    # --------------------------------------------------------- refine points

    def _local_to_display(self, local_point: np.ndarray) -> np.ndarray:
        """Map a brain-local (picked) point into the current preview/skull space."""
        return core.xform_pt(np.asarray(local_point), np.array(self._transform["full_4x4"]))

    def _display_to_local(self, display_point: np.ndarray) -> np.ndarray:
        """Inverse of `_local_to_display`, used when a slider edit is applied."""
        inv = np.linalg.inv(np.array(self._transform["full_4x4"]))
        return core.xform_pt(np.asarray(display_point), inv)

    def _set_axis_ranges(self, bounds: np.ndarray) -> None:
        if self._vector_mode:
            # Displacement mode: symmetric range around 0. Sized to the full
            # extent of the bounds along each axis (rather than half of it)
            # so that, whatever the landmark's baseline position within the
            # skull, the whole bounds remain reachable in either direction.
            extents = bounds[1] - bounds[0]
            for axis_ctrl, extent in zip((self.axis_x, self.axis_y, self.axis_z), extents):
                axis_ctrl.set_range(float(-extent), float(extent))
        else:
            pads = (bounds[1] - bounds[0]) * 0.15
            for axis_ctrl, lo, hi, pad in zip(
                (self.axis_x, self.axis_y, self.axis_z), bounds[0], bounds[1], pads
            ):
                axis_ctrl.set_range(float(lo - pad), float(hi + pad))

    def _on_refine_point_selected(self, idx: int) -> None:
        self._load_axis_values(idx)

    def _baseline_display_point(self, idx: int) -> np.ndarray | None:
        """The selected landmark's position as of the last Compute
        Transform, in display space -- the reference point displacement
        mode's (0, 0, 0) means "no change" relative to."""
        if self._last_committed_points is None or not (0 <= idx < len(self._last_committed_points)):
            return None
        return self._local_to_display(self._last_committed_points[idx])

    def _load_axis_values(self, idx: int) -> None:
        if not (0 <= idx < len(self._brain_points)) or self._transform is None:
            return
        point = self._local_to_display(self._brain_points[idx])
        if self._vector_mode:
            # The landmark sits at baseline - vector (see `_on_axis_value_changed`),
            # so the vector that produced the current point is baseline - point.
            baseline = self._baseline_display_point(idx)
            point = point if baseline is None else baseline - point
        self.axis_x.set_value(float(point[0]))
        self.axis_y.set_value(float(point[1]))
        self.axis_z.set_value(float(point[2]))

    def _on_axis_value_changed(self, _value: float) -> None:
        idx = self.refine_combo.currentIndex()
        if not (0 <= idx < len(self._brain_points)) or self._transform is None:
            return
        values = np.array([self.axis_x.value(), self.axis_y.value(), self.axis_z.value()])
        baseline = self._baseline_display_point(idx)
        if self._vector_mode:
            # Plain coordinate displacement to (baseline - vector): moving
            # dX/dY/dZ by (dx, dy, dz) displaces the landmark by -(dx, dy, dz).
            display_point = values if baseline is None else baseline - values
        else:
            display_point = values
        # Store back in the brain's own frame, so "Compute Transform" keeps
        # working from the same raw-landmark inputs as before; only the
        # marker moves live here, in the space the preview is shown in.
        self._brain_points[idx] = self._display_to_local(display_point)
        self._transform_dirty = True
        # In vector mode, draw an arrow whose fixed tail is the landmark's
        # old (baseline) position and whose tip tracks the new, displaced
        # position exactly, so the displacement is visible in the view.
        origin = baseline if self._vector_mode else None
        self.viewer.move_marker(idx, display_point, origin=origin)
        self._refresh_buttons()

    def _on_reset_axis(self, axis_index: int) -> None:
        """Revert one axis of the selected point to its value as of the
        last Compute Transform (not to whatever it was a moment ago)."""
        idx = self.refine_combo.currentIndex()
        if (
            self._transform is None
            or self._last_committed_points is None
            or not (0 <= idx < len(self._last_committed_points))
        ):
            return
        axis_ctrl = (self.axis_x, self.axis_y, self.axis_z)[axis_index]
        if self._vector_mode:
            # Baseline itself is the (0, 0, 0) displacement.
            axis_ctrl.set_value(0.0)
        else:
            baseline_display = self._baseline_display_point(idx)
            axis_ctrl.set_value(float(baseline_display[axis_index]))
        # `set_value` doesn't emit `valueChanged` (it's used for silent,
        # programmatic updates), so drive the usual edit path by hand,
        # using whatever the three spinboxes now read.
        self._on_axis_value_changed(0.0)

    def _on_transparency_toggled(self, _checked: bool) -> None:
        self._show_preview_with_markers()

    def _on_vector_mode_toggled(self, checked: bool) -> None:
        self._vector_mode = checked
        for axis_ctrl, base in zip((self.axis_x, self.axis_y, self.axis_z), ("X", "Y", "Z")):
            axis_ctrl.set_axis_label(f"d{base}" if checked else base)
        self.viewer.set_point_radius_scale(0.25 if checked else 1.0)
        if self._transform is None:
            return
        if self._skull_bounds is not None:
            self._set_axis_ranges(self._skull_bounds)
        idx = self.refine_combo.currentIndex()
        self._load_axis_values(idx)
        # Immediately reflect the mode switch on the currently selected
        # point's marker (add/remove its origin ghost + arrow); other,
        # unselected points keep whatever they last showed.
        if 0 <= idx < len(self._brain_points):
            point = self._local_to_display(self._brain_points[idx])
            origin = self._baseline_display_point(idx) if checked else None
            self.viewer.move_marker(idx, point, origin=origin)

    # -------------------------------------------------------------- compute

    def _on_compute(self) -> None:
        skull_path = self.skull_edit.text().strip()
        brain_path = self.brain_edit.text().strip()
        try:
            skull_mesh = trimesh.load(skull_path)
            brain_mesh = trimesh.load(brain_path)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load meshes", str(exc))
            return

        lm = core.Landmarks(
            bregma_skull=self._skull_points[0],
            lambda_skull=self._skull_points[1],
            ventral_skull=self._skull_points[2],
            bregma_brain=self._brain_points[0],
            lambda_brain=self._brain_points[1],
            ventral_brain=self._brain_points[2],
        )
        transform = core.compute_transform(
            skull_mesh, brain_mesh, lm,
            skull_path=skull_path, brain_path=brain_path,
            cavity_scale=self.cavity_spin.value(),
        )

        if self._transform is not None:
            self._undo_stack.append((self._transform, self._last_committed_points))
            self._redo_stack.clear()

        self._transform = transform
        self._last_committed_points = [p.copy() for p in self._brain_points]
        self._transform_dirty = False

        self._update_summary_text(transform)
        self._show_preview_with_markers()
        self._refresh_buttons()

    def _update_summary_text(self, transform: dict) -> None:
        scale = transform["scale_xyz"]
        translation = transform["translation_xyz"]
        self.summary_text.setPlainText(
            "Scale (antero-post., dorso-vent., medio-lat.):\n"
            f"  {scale[0]:.4f}, {scale[1]:.4f}, {scale[2]:.4f}\n\n"
            "Translation (X, Y, Z):\n"
            f"  {translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}"
        )

    def _on_undo_transform(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append((self._transform, self._last_committed_points))
        self._transform, self._last_committed_points = self._undo_stack.pop()
        self._brain_points = [p.copy() for p in self._last_committed_points]
        self._transform_dirty = False
        self._update_summary_text(self._transform)
        self._show_preview_with_markers()
        self._refresh_buttons()

    def _on_redo_transform(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append((self._transform, self._last_committed_points))
        self._transform, self._last_committed_points = self._redo_stack.pop()
        self._brain_points = [p.copy() for p in self._last_committed_points]
        self._transform_dirty = False
        self._update_summary_text(self._transform)
        self._show_preview_with_markers()
        self._refresh_buttons()

    def _show_preview_with_markers(self) -> None:
        """Show skull + aligned brain, with the brain landmarks overlaid and movable."""
        if self._transform is None:
            return
        skull_path = self.skull_edit.text().strip()
        brain_path = self.brain_edit.text().strip()
        try:
            skull_mesh = trimesh.load(skull_path)
            skull_mesh.apply_transform(np.array(self._transform["skull_canonical_4x4"]))
            brain_mesh = trimesh.load(brain_path)
            brain_mesh.apply_transform(np.array(self._transform["full_4x4"]))
        except Exception as exc:
            QMessageBox.critical(self, "Preview failed", str(exc))
            return

        skull_opacity = BONE_OPACITY_VIEW if self.skull_transparency_check.isChecked() else 1.0
        brain_opacity = BRAIN_OPACITY_TRANSPARENT if self.brain_transparency_check.isChecked() else BRAIN_OPACITY_VIEW

        self.viewer.clear()
        self.viewer.show_mesh_object(trimesh_to_pv(skull_mesh), "preview_skull",
                                     BONE_COLOR, skull_opacity)
        self.viewer.show_mesh_object(trimesh_to_pv(brain_mesh), "preview_brain",
                                     BRAIN_COLOR, brain_opacity)
        if not self._preview_camera_set:
            # Only frame the scene the first time it's shown for this
            # picking session; later refreshes (recompute, undo/redo,
            # toggling skull transparency) must not disturb a view the
            # user has since rotated to.
            self.viewer.plotter.reset_camera()
            self.viewer.plotter.view_xz()
            self._preview_camera_set = True

        self._skull_bounds = skull_mesh.bounds
        self._set_axis_ranges(self._skull_bounds)
        display_points = [self._local_to_display(p) for p in self._brain_points]
        self.viewer.set_markers(LANDMARK_LABELS, display_points)
        self._load_axis_values(self.refine_combo.currentIndex())

    def _on_save(self) -> None:
        if self._transform is None or self._transform_dirty:
            return
        path = self.output_edit.text().strip()
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save transform as", "", "JSON files (*.json)")
            if not path:
                return
            if not path.lower().endswith(".json"):
                path += ".json"
            self.output_edit.setText(path)
        io_utils.save_transform(self._transform, path)
        QMessageBox.information(self, "Saved", f"Transform saved to:\n{path}")
