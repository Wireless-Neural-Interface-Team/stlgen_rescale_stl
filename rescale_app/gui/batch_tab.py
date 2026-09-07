"""T2 — apply a saved rescaling transform to a folder of STLs, or to a single STL."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import BackgroundPlotter

from .. import core, io_utils
from .pick_viewer import (
    BONE_COLOR,
    BONE_OPACITY_VIEW,
    BRAIN_COLOR,
    BRAIN_OPACITY_VIEW,
    REGION_COLOR,
    REGION_OPACITY_VIEW,
    WHOLE_BRAIN_FILENAME,
    trimesh_to_pv,
)


class BatchWorker(QThread):
    progress = Signal(int, int, str)
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(self, transform: dict, input_folder: Path, output_folder: Path, parent=None):
        super().__init__(parent)
        self.transform = transform
        self.input_folder = input_folder
        self.output_folder = output_folder

    def run(self) -> None:
        try:
            def cb(i, total, path):
                self.progress.emit(i, total, path.name)

            outputs = io_utils.apply_transform_to_folder(
                self.transform, self.input_folder, self.output_folder, progress_cb=cb
            )
            self.finished_ok.emit([str(p) for p in outputs])
        except Exception as exc:
            self.failed.emit(str(exc))


class BatchTab(QWidget):
    """Batch-apply a saved transform JSON to a folder of STLs, or a single STL."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: BatchWorker | None = None
        self._last_outputs: list[str] = []
        self._background_plotter: BackgroundPlotter | None = None

        layout = QVBoxLayout(self)

        files_box = QGroupBox("Input / output")
        files_layout = QVBoxLayout(files_box)

        row = QHBoxLayout()
        row.addWidget(QLabel("Input data"))
        self.input_edit = QLineEdit()
        self.input_edit.textChanged.connect(self._update_output_placeholder)
        row.addWidget(self.input_edit, 1)
        self.browse_input_btn = QPushButton("Browse…")
        browse_input_menu = QMenu(self.browse_input_btn)
        folder_action = QAction("Folder…", self.browse_input_btn)
        folder_action.triggered.connect(self._browse_input_folder)
        browse_input_menu.addAction(folder_action)
        file_action = QAction("STL File…", self.browse_input_btn)
        file_action.triggered.connect(self._browse_input_file)
        browse_input_menu.addAction(file_action)
        self.browse_input_btn.setMenu(browse_input_menu)
        row.addWidget(self.browse_input_btn)
        files_layout.addLayout(row)

        self.skull_only_check = QCheckBox("Skull only")
        self.skull_only_check.setToolTip(
            "Apply only the skull's own rigid repositioning (the transform\n"
            "JSON's 'skull_canonical_4x4') instead of the full brain-to-skull\n"
            "transform. Use this when the input data is skull geometry, not brain."
        )
        files_layout.addWidget(self.skull_only_check)

        row = QHBoxLayout()
        row.addWidget(QLabel("Transform JSON"))
        self.json_edit = QLineEdit()
        row.addWidget(self.json_edit, 1)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse_json)
        row.addWidget(btn)
        files_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Output (optional)"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("<input folder>_rescaled")
        row.addWidget(self.output_edit, 1)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse_output)
        row.addWidget(btn)
        files_layout.addLayout(row)

        layout.addWidget(files_box)

        self.run_btn = QPushButton("Run Rescale")
        self.run_btn.clicked.connect(self._on_run)
        layout.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text, 1)

        self.view_result_btn = QPushButton("View Result")
        self.view_result_btn.setEnabled(False)
        self.view_result_btn.clicked.connect(self._on_view_result)
        layout.addWidget(self.view_result_btn)

    # --------------------------------------------------------------- browse

    def _browse_input_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select input folder")
        if path:
            self.input_edit.setText(path)

    def _browse_input_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select input STL file", "", "STL files (*.stl)")
        if path:
            self.input_edit.setText(path)

    def _browse_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select transform JSON", "", "JSON files (*.json)")
        if path:
            self.json_edit.setText(path)

    def _browse_output(self) -> None:
        # Match the browse dialog to whichever input mode is currently set.
        if Path(self.input_edit.text().strip() or ".").is_file():
            path, _ = QFileDialog.getSaveFileName(self, "Select output STL file", "", "STL files (*.stl)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.output_edit.setText(path)

    def _update_output_placeholder(self, text: str) -> None:
        text = text.strip()
        if not text:
            self.output_edit.setPlaceholderText("<input folder>_rescaled")
            return
        path = Path(text)
        if path.is_file() or (not path.exists() and path.suffix):
            self.output_edit.setPlaceholderText(str(io_utils.default_output_file(path)))
        else:
            self.output_edit.setPlaceholderText(str(io_utils.default_output_folder(path)))

    # ----------------------------------------------------------------- run

    def _on_run(self) -> None:
        input_text = self.input_edit.text().strip()
        json_path = self.json_edit.text().strip()
        output_text = self.output_edit.text().strip()

        input_path = Path(input_text) if input_text else None
        if not input_path or not (input_path.is_dir() or input_path.is_file()):
            QMessageBox.warning(self, "Invalid input", "Select a valid input folder or STL file.")
            return
        if not json_path or not Path(json_path).is_file():
            QMessageBox.warning(self, "Invalid input", "Select a valid transform JSON file.")
            return

        try:
            transform = io_utils.load_transform(json_path)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load transform", str(exc))
            return

        if self.skull_only_check.isChecked():
            if "skull_canonical_4x4" not in transform:
                QMessageBox.critical(
                    self, "Missing skull transform",
                    "This transform JSON has no 'skull_canonical_4x4' entry. "
                    "Only transform JSONs saved by this app's Compute Transform include it.",
                )
                return
            transform = {**transform, "full_4x4": transform["skull_canonical_4x4"]}

        if input_path.is_file():
            self._run_single_file(transform, input_path, output_text)
        else:
            self._run_folder(transform, input_path, output_text)

    def _run_folder(self, transform: dict, input_folder: Path, output_text: str) -> None:
        output_folder = Path(output_text) if output_text else io_utils.default_output_folder(input_folder)

        stl_files = io_utils.list_stl_files(input_folder)
        if not stl_files:
            QMessageBox.warning(self, "No files found", f"No .stl files found in:\n{input_folder}")
            return

        self.log_text.clear()
        self.log_text.appendPlainText(f"Input:  {input_folder}")
        self.log_text.appendPlainText(f"Output: {output_folder}")
        self.log_text.appendPlainText(f"Transform: {self.json_edit.text().strip()}")
        self.log_text.appendPlainText(f"Found {len(stl_files)} STL file(s).\n")

        self.progress_bar.setRange(0, len(stl_files))
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.view_result_btn.setEnabled(False)

        self._worker = BatchWorker(transform, input_folder, output_folder, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _run_single_file(self, transform: dict, input_file: Path, output_text: str) -> None:
        if output_text:
            output_path = Path(output_text)
            if output_path.suffix.lower() != ".stl":
                output_path = output_path / input_file.name
        else:
            output_path = io_utils.default_output_file(input_file)

        self.log_text.clear()
        self.log_text.appendPlainText(f"Input:  {input_file}")
        self.log_text.appendPlainText(f"Output: {output_path}")
        self.log_text.appendPlainText(f"Transform: {self.json_edit.text().strip()}\n")

        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.view_result_btn.setEnabled(False)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            core.apply_transform(transform, input_file, output_path)
        except Exception as exc:
            self.run_btn.setEnabled(True)
            QMessageBox.critical(self, "Rescale failed", str(exc))
            return

        self.progress_bar.setValue(1)
        self.log_text.appendPlainText("Done. 1 file written.")
        self._last_outputs = [str(output_path)]
        self.run_btn.setEnabled(True)
        self.view_result_btn.setEnabled(True)

    def _on_progress(self, index: int, total: int, filename: str) -> None:
        self.progress_bar.setValue(index)
        self.log_text.appendPlainText(f"[{index + 1}/{total}] {filename}")

    def _on_finished(self, outputs: list[str]) -> None:
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.log_text.appendPlainText(f"\nDone. {len(outputs)} file(s) written.")
        self._last_outputs = outputs
        self.run_btn.setEnabled(True)
        self.view_result_btn.setEnabled(bool(outputs))

    def _on_failed(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Batch rescale failed", message)

    # -------------------------------------------------------------- viewer

    def _on_view_result(self) -> None:
        if not self._last_outputs:
            return
        import trimesh

        transform_path = self.json_edit.text().strip()
        transform = io_utils.load_transform(transform_path) if transform_path else {}
        skull_path = transform.get("skull_stl")

        plotter = BackgroundPlotter(title="Batch rescale result")
        if skull_path and Path(skull_path).is_file():
            plotter.add_mesh(trimesh_to_pv(trimesh.load(skull_path)), color=BONE_COLOR,
                             opacity=BONE_OPACITY_VIEW, smooth_shading=True, label="skull")
        for out_path in self._last_outputs:
            is_whole = Path(out_path).name == WHOLE_BRAIN_FILENAME
            color = BRAIN_COLOR if is_whole else REGION_COLOR
            opacity = BRAIN_OPACITY_VIEW if is_whole else REGION_OPACITY_VIEW
            plotter.add_mesh(trimesh_to_pv(trimesh.load(out_path)), color=color,
                             opacity=opacity, smooth_shading=True,
                             label=Path(out_path).stem)
        plotter.add_legend()
        # plotter.view_isometric()
        self._background_plotter = plotter
