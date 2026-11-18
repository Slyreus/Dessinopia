from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication

import dessinopia
from dessinopia.ui.main_window import MainWindow
from dessinopia.ui.styles import APP_STYLESHEET


def main() -> int:
    QCoreApplication.setOrganizationName("Slyreus")
    QCoreApplication.setApplicationName("Dessinopia")
    QCoreApplication.setApplicationVersion(dessinopia.__version__)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(Path(dessinopia.__file__).parent / "assets" / "palette.png")))
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#111918"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#edf4f3"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#101816"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#edf4f3"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#263632"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#edf4f3"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#37c8ad"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#09211d"))
    app.setPalette(palette)
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
