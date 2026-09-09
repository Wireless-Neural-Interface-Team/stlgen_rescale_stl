"""Embedded PyVista viewer widget with ordered landmark picking.

Wraps a ``pyvistaqt.QtInteractor`` so it can sit directly inside a Qt
layout (as opposed to the original CLI script, which opened a separate
blocking ``pyvista.Plotter`` window per pick).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyvista as pv
import vtk
from pyvistaqt import QtInteractor
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

BONE_COLOR = "#6f6a60"
BONE_OPACITY_PICK = 0.55
BONE_OPACITY_VIEW = .55
BRAIN_COLOR = "#e87060"
BRAIN_OPACITY_VIEW = 1.0
BRAIN_OPACITY_TRANSPARENT = 0.35
REGION_COLOR = "#8b0000"
REGION_OPACITY_VIEW = 1.0
WHOLE_BRAIN_FILENAME = "whole_brain.stl"

VIEWER_BACKGROUND = "#3a3a3a"  # dark neutral bg: contrasts with pale meshes better than white

PICK_COLORS = ["#ff3b30", "#34c759", "#0a84ff", "#ffcc00", "#ff2d92"]

# A press+release pair is treated as a landmark click only if the cursor
# stayed within this many pixels between the two events; anything further
# is a camera rotate/pan/zoom drag and must not place a landmark.
CLICK_DRAG_THRESHOLD_PX = 4


def trimesh_to_pv(mesh) -> pv.PolyData:
    """Convert a trimesh.Trimesh into a pyvista.PolyData."""
    faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).astype(np.int64)
    return pv.PolyData(np.asarray(mesh.vertices), faces)


class PickViewer(QWidget):
    """A 3D view embedded in the Qt layout that supports ordered point picking."""

    point_picked = Signal(int, str, object)  # index, label, np.ndarray point
    point_undone = Signal(int, str)  # index, label
    picking_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        self.plotter.set_background(VIEWER_BACKGROUND)
        try:
            # Correct blending for overlapping translucent surfaces (e.g. the
            # front and back of the skull) instead of painter's-algorithm
            # sorting artifacts.
            self.plotter.enable_depth_peeling()
        except Exception:
            pass
        try:
            # Screen-space depth shading: darkens concave surface detail
            # (suture lines, small crevices) that flat/PBR lighting on a
            # pale mesh over a white background otherwise washes out.
            self.plotter.enable_eye_dome_lighting()
        except Exception:
            pass

        self._labels: list[str] = []
        self._picked: list[np.ndarray] = []
        self._picking_enabled = False
        self._mesh_length = 1.0
        self._point_radius_scale = 1.0  # shrinks `set_markers`/`move_marker` spheres (e.g. in vector mode)

        # Custom press/release picking (see `_enable_picking`) instead of
        # pyvista's built-in click-picking, which fires on button *press*
        # and has no way to tell a click apart from the start of a
        # rotate/pan/zoom drag.
        self._vtk_picker = vtk.vtkCellPicker()
        self._vtk_picker.SetTolerance(0.0005)
        self._press_pos: Optional[tuple[int, int]] = None
        self._press_observer = None
        self._release_observer = None

        # Right-click (anywhere) undoes the last placed point, mirroring
        # the "Undo Last Point" button. Same click-vs-drag distinction as
        # left-click, so a right-drag (camera zoom) doesn't trigger it.
        self._right_press_pos: Optional[tuple[int, int]] = None
        self._right_press_observer = None
        self._right_release_observer = None

    # ------------------------------------------------------------------ mesh

    def show_mesh(self, mesh_path: str, color: str = BONE_COLOR,
                  opacity: float = BONE_OPACITY_PICK, name: str = "active_mesh") -> None:
        # No `pbr=True` here: VTK's PBR shading path does not respect
        # opacity/alpha blending, which makes the mesh render fully solid
        # regardless of the opacity value. Instead, standard Phong shading
        # with a stronger specular term gives visible highlights along
        # ridges/sutures (helps spot small crevices) while still honoring
        # opacity correctly.
        mesh = pv.read(str(mesh_path))
        self.plotter.add_mesh(mesh, color=color, smooth_shading=True, name=name)
        self._mesh_length = mesh.length or 1.0
        self.plotter.reset_camera()
        self.plotter.view_xz()
        self.plotter.add_axes()

    def show_mesh_object(self, mesh: pv.PolyData, name: str, color: str, opacity: float) -> None:
        # No camera move here: this is called on every preview refresh
        # (Compute Transform, Preview Result, toggling skull transparency),
        # and the caller decides once whether this is a first-time view
        # that needs a default camera position.
        self.plotter.add_mesh(mesh, color=color, opacity=opacity,
                              smooth_shading=True, name=name)
        self._mesh_length = max(self._mesh_length, mesh.length or 0.0)

    def clear(self) -> None:
        self.stop_picking()
        self.plotter.clear()

    # --------------------------------------------------------------- picking

    def start_picking(self, labels: list[str]) -> None:
        """Begin ordered picking for the given landmark labels."""
        self._labels = labels
        self._picked = []
        self._enable_picking()
        self._update_overlay()

    def stop_picking(self) -> None:
        self._picking_enabled = False
        self._press_pos = None
        self._right_press_pos = None
        for attr in ("_press_observer", "_release_observer",
                     "_right_press_observer", "_right_release_observer"):
            observer = getattr(self, attr)
            if observer is not None:
                try:
                    self.plotter.iren.remove_observer(observer)
                except Exception:
                    pass
                setattr(self, attr, None)

    def undo(self) -> None:
        if not self._picked:
            return
        idx = len(self._picked) - 1
        label = self._labels[idx]
        self._picked.pop()
        self.plotter.remove_actor(f"pick_{idx}")
        self.plotter.remove_actor(f"label_{idx}")
        if not self._picking_enabled:
            self._enable_picking()
        self._update_overlay()
        self.point_undone.emit(idx, label)

    def picked_points(self) -> list[np.ndarray]:
        return list(self._picked)

    def next_label(self) -> Optional[str]:
        if len(self._picked) < len(self._labels):
            return self._labels[len(self._picked)]
        return None

    def set_point_radius_scale(self, scale: float) -> None:
        """Scale factor applied to the sphere radius in `set_markers` and
        `move_marker` (e.g. shrink to a quarter size in vector mode, where
        the arrow is the primary visual and the point is just its tail)."""
        self._point_radius_scale = scale

    def set_markers(self, labels: list[str], points: list[np.ndarray]) -> None:
        """Show a fixed set of markers, not tied to click-based picking.

        Used to overlay landmarks (e.g. on a computed preview) so they can
        be nudged afterwards via `move_marker`.
        """
        self._labels = list(labels)
        self._picked = [np.asarray(p) for p in points]
        radius = self._mesh_length * 0.012 * self._point_radius_scale
        for idx, point in enumerate(self._picked):
            color = PICK_COLORS[idx % len(PICK_COLORS)]
            self.plotter.add_mesh(pv.Sphere(radius=radius, center=point), color=color,
                                  name=f"pick_{idx}", pickable=False)
            self.plotter.add_point_labels(
                [point], [f"  {labels[idx]}"], font_size=11, text_color=color,
                name=f"label_{idx}", always_visible=True,
            )

    def move_marker(self, idx: int, point: np.ndarray, origin: Optional[np.ndarray] = None) -> None:
        """Reposition an already-placed landmark marker/label in place.

        Cheap (repositions one small sphere + label, plus an optional
        arrow), unlike recomputing the transform or the transformed mesh,
        so callers can use this for live feedback while the point
        coordinate is still being edited.

        `point` is always the landmark's actual (current) position -- the
        solid, labeled sphere is drawn there (its radius scaled by
        `set_point_radius_scale`). If `origin` is given and distinct from
        `point`, an arrow is drawn from a fixed, dimmed "ghost" sphere at
        `origin`, with its tip mirroring `point` through `origin` --
        i.e. the tip sits at ``2*origin - point``, moving in the opposite
        direction from `point` itself.
        """
        if idx >= len(self._picked):
            return
        point = np.asarray(point)
        self._picked[idx] = point
        color = PICK_COLORS[idx % len(PICK_COLORS)]
        radius = self._mesh_length * 0.012 * self._point_radius_scale

        self.plotter.remove_actor(f"pick_origin_{idx}")
        self.plotter.remove_actor(f"pick_vector_{idx}")
        if origin is not None:
            origin = np.asarray(origin)
            vec = origin - point  # mirrored: tip = origin + vec = 2*origin - point
            length = float(np.linalg.norm(vec))
            if length > 1e-9:
                self.plotter.add_mesh(
                    pv.Sphere(radius=radius * 0.7, center=origin), color=color,
                    opacity=0.35, name=f"pick_origin_{idx}", pickable=False,
                )
                arrow = pv.Arrow(start=origin, direction=vec / length, scale=length)
                self.plotter.add_mesh(arrow, color=color, name=f"pick_vector_{idx}", pickable=False)

        self.plotter.add_mesh(pv.Sphere(radius=radius, center=point), color=color,
                              name=f"pick_{idx}", pickable=False)
        self.plotter.add_point_labels(
            [point], [f"  {self._labels[idx]}"], font_size=11, text_color=color,
            name=f"label_{idx}", always_visible=True,
        )

    def _enable_picking(self) -> None:
        self._picking_enabled = True
        if self._press_observer is None:
            self._press_observer = self.plotter.iren.add_observer(
                "LeftButtonPressEvent", self._on_left_press)
            self._release_observer = self.plotter.iren.add_observer(
                "LeftButtonReleaseEvent", self._on_left_release)
        if self._right_press_observer is None:
            self._right_press_observer = self.plotter.iren.add_observer(
                "RightButtonPressEvent", self._on_right_press)
            self._right_release_observer = self.plotter.iren.add_observer(
                "RightButtonReleaseEvent", self._on_right_release)

    def _on_left_press(self, _obj, _event) -> None:
        self._press_pos = self.plotter.iren.get_event_position()

    def _on_left_release(self, _obj, _event) -> None:
        if not self._picking_enabled or self._press_pos is None:
            return
        press_pos = self._press_pos
        self._press_pos = None
        release_pos = self.plotter.iren.get_event_position()
        dx = release_pos[0] - press_pos[0]
        dy = release_pos[1] - press_pos[1]
        if dx * dx + dy * dy > CLICK_DRAG_THRESHOLD_PX ** 2:
            return  # the camera was rotated/panned/zoomed, not a landmark click
        self._do_pick(release_pos)

    def _on_right_press(self, _obj, _event) -> None:
        self._right_press_pos = self.plotter.iren.get_event_position()

    def _on_right_release(self, _obj, _event) -> None:
        if not self._picking_enabled or self._right_press_pos is None:
            return
        press_pos = self._right_press_pos
        self._right_press_pos = None
        release_pos = self.plotter.iren.get_event_position()
        dx = release_pos[0] - press_pos[0]
        dy = release_pos[1] - press_pos[1]
        if dx * dx + dy * dy > CLICK_DRAG_THRESHOLD_PX ** 2:
            return  # the camera was zoomed (right-drag), not an undo click
        self.undo()

    def _do_pick(self, screen_pos) -> None:
        if len(self._picked) >= len(self._labels):
            return
        renderer = self.plotter.iren.get_poked_renderer()
        self._vtk_picker.Pick(screen_pos[0], screen_pos[1], 0, renderer)
        if self._vtk_picker.GetActor() is None:
            return  # clicked on empty space, not on the mesh surface
        self._on_pick(self._vtk_picker.GetPickPosition(), self._vtk_picker)

    def _update_overlay(self) -> None:
        idx = len(self._picked)
        if idx < len(self._labels):
            msg = f"({idx + 1}/{len(self._labels)}) Click to place:  {self._labels[idx]}"
        else:
            msg = f"All {len(self._labels)} points picked."
        self.plotter.remove_actor("instructions")
        self.plotter.add_text(msg, position="upper_left", font_size=11,
                              color="white", name="instructions")

    def _on_pick(self, point, _picker) -> None:
        if not self._picking_enabled:
            return
        idx = len(self._picked)
        if idx >= len(self._labels):
            return
        pt = np.array(point)
        self._picked.append(pt)
        color = PICK_COLORS[idx % len(PICK_COLORS)]
        radius = self._mesh_length * 0.012
        self.plotter.add_mesh(pv.Sphere(radius=radius, center=pt), color=color,
                              name=f"pick_{idx}", pickable=False)
        self.plotter.add_point_labels(
            [pt], [f"  {self._labels[idx]}"], font_size=11, text_color=color,
            name=f"label_{idx}", always_visible=True,
        )
        label = self._labels[idx]
        finished = len(self._picked) >= len(self._labels)
        if finished:
            self.stop_picking()
        self._update_overlay()
        self.point_picked.emit(idx, label, pt)
        if finished:
            self.picking_finished.emit()
