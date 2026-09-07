"""Entry point for the STL Rescale Toolkit Qt GUI.

Launch it with:

    uv run rescale-stl          # installed console script
    uv run python -m rescale_app
"""

import sys

from PySide6.QtWidgets import QApplication

from .gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
