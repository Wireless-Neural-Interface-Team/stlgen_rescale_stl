"""Main window: tabs for T1 (generate transform) and T2 (batch rescale)."""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget

from .batch_tab import BatchTab
from .generate_tab import GenerateTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STL Rescale Toolkit")
        self.resize(1280, 820)

        tabs = QTabWidget()
        tabs.addTab(GenerateTab(), "1. Generate Transform")
        tabs.addTab(BatchTab(), "2. Batch Rescale")
        self.setCentralWidget(tabs)
