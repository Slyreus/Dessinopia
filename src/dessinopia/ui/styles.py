APP_STYLESHEET = r"""
QWidget {
    color: #edf4f3;
    font-family: "Segoe UI Variable", "Segoe UI";
    font-size: 13px;
}
QMainWindow, QDialog { background: #111918; }
QFrame#card {
    background: #192422;
    border: 1px solid #293a37;
    border-radius: 14px;
}
QFrame#heroCard {
    background: #162926;
    border: 1px solid #31564f;
    border-radius: 16px;
}
QLabel#title { font-size: 28px; font-weight: 750; color: #ffffff; }
QLabel#subtitle { color: #9db4af; font-size: 13px; }
QLabel#sectionTitle { font-size: 16px; font-weight: 700; color: #f7fbfa; }
QLabel#muted { color: #93a9a4; }
QLabel#statusPill {
    background: #203632;
    border: 1px solid #31554e;
    border-radius: 9px;
    padding: 4px 9px;
    color: #9ce8d9;
    font-weight: 650;
}
QPushButton {
    background: #263632;
    border: 1px solid #38514b;
    border-radius: 9px;
    padding: 8px 13px;
    font-weight: 650;
}
QPushButton:hover { background: #304742; border-color: #4d756c; }
QPushButton:pressed { background: #1e302c; }
QPushButton:disabled { color: #687975; background: #1b2523; border-color: #26322f; }
QPushButton#primary {
    background: #37c8ad;
    border-color: #62dcc5;
    color: #09211d;
    font-weight: 750;
}
QPushButton#primary:hover { background: #55d9c1; }
QPushButton#primary:disabled {
    color: #66817b;
    background: #1d302c;
    border-color: #2a413c;
}
QPushButton#danger { background: #4a262b; border-color: #7d3f49; color: #ffb8c1; }
QPushButton#danger:disabled { color: #806168; background: #2c2224; border-color: #473034; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #101816;
    border: 1px solid #32433f;
    border-radius: 8px;
    padding: 7px 9px;
    min-height: 20px;
    selection-background-color: #37c8ad;
}
QComboBox:hover, QSpinBox:hover, QLineEdit:hover { border-color: #52726a; }
QComboBox::drop-down { border: 0; width: 28px; }
QComboBox QAbstractItemView {
    background: #1a2523;
    border: 1px solid #3c514c;
    selection-background-color: #2b5f55;
}
QSlider::groove:horizontal { height: 5px; background: #31413e; border-radius: 2px; }
QSlider::handle:horizontal { width: 16px; margin: -6px 0; background: #55d9c1; border-radius: 8px; }
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #4d6761;
    border-radius: 4px;
    background: #101816;
}
QCheckBox::indicator:hover { border-color: #65a99b; }
QCheckBox::indicator:checked { background: #37c8ad; border-color: #62dcc5; }
QTabWidget::pane { border: 0; }
QTabBar::tab { color: #91a7a2; padding: 8px 16px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #dffff8; border-bottom-color: #42d0b4; }
QListWidget {
    background: #111a18;
    border: 1px solid #2b3a37;
    border-radius: 9px;
    padding: 4px;
}
QListWidget::item { padding: 5px; border-radius: 5px; }
QListWidget::item:selected { background: #294c45; }
QProgressBar {
    background: #101816;
    border: 1px solid #30413d;
    border-radius: 7px;
    text-align: center;
    min-height: 15px;
}
QProgressBar::chunk { background: #37c8ad; border-radius: 6px; }
QToolTip { background: #263632; color: white; border: 1px solid #52726a; padding: 5px; }
"""
