from __future__ import annotations

from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication


DARK_QSS = """
QWidget {
    background: #0b1020;
    color: #e7ecf7;
    font-family: "Segoe UI";
    font-size: 10.5pt;
}
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget { background: #0b1020; }
QLabel, QCheckBox, QRadioButton { background: transparent; }
#Sidebar { background: #10172a; border-right: 1px solid #27324a; }
#Brand { font-size: 19pt; font-weight: 700; color: #78a8ff; }
#BrandCaption, [muted="true"] { color: #8f9bb3; }
#PageTitle { font-size: 21pt; font-weight: 700; }
#PageSubtitle { color: #94a2bd; font-size: 10pt; }
#Card {
    background: #121b30;
    border: 1px solid #263553;
    border-radius: 14px;
}
#VideoSurface {
    background: #050811;
    border: 2px solid #2d4775;
    border-radius: 18px;
}
#HandSurface {
    background: #080d18;
    border: 1px solid #304261;
    border-radius: 12px;
}
#StatusGood { color: #6ee7a8; font-weight: 600; }
#StatusNeutral { color: #9fb0cc; font-weight: 600; }
#StatusWarn { color: #ffc96b; font-weight: 600; }
#StatusBad { color: #ff7b88; font-weight: 600; }
QPushButton {
    background: #2563eb;
    color: white;
    border: 0;
    border-radius: 9px;
    padding: 9px 14px;
    font-weight: 600;
}
QPushButton:hover { background: #3475f5; }
QPushButton:pressed { background: #1d4ed8; }
QPushButton:checked { background: #245dcc; border: 1px solid #78a8ff; }
QPushButton:disabled { background: #303a50; color: #7f899d; }
QPushButton[secondary="true"] {
    background: #1b2740;
    border: 1px solid #34476b;
}
QPushButton[accent="true"] {
    background: #167b58;
    border: 1px solid #55dca6;
}
QPushButton[danger="true"] { background: #9f3042; }
QLineEdit, QTextEdit, QTextBrowser, QPlainTextEdit, QComboBox,
QSpinBox, QDoubleSpinBox, QTableWidget {
    background: #0c1426;
    color: #ecf1fb;
    border: 1px solid #334565;
    border-radius: 8px;
    padding: 7px;
    selection-background-color: #275bb9;
}
QComboBox QAbstractItemView { background: #111a2d; color: #ecf1fb; }
QHeaderView::section {
    background: #18233b;
    color: #b8c5dc;
    padding: 7px;
    border: 0;
    border-bottom: 1px solid #334565;
}
QTableWidget { gridline-color: #273653; }
QGroupBox {
    border: 1px solid #2b3a58;
    border-radius: 11px;
    margin-top: 13px;
    padding-top: 13px;
    font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QSlider::groove:horizontal { background: #27334a; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #67a0ff; width: 18px; margin: -6px 0; border-radius: 9px;
}
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 18px; height: 18px; }
QSplitter::handle { background: #202d47; width: 2px; }
QToolTip { background: #18233b; color: white; border: 1px solid #405476; }
"""


LIGHT_QSS = """
QWidget {
    background: #f4f7fb;
    color: #172033;
    font-family: "Segoe UI";
    font-size: 10.5pt;
}
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget { background: #f4f7fb; }
QLabel, QCheckBox, QRadioButton { background: transparent; }
#Sidebar { background: #ffffff; border-right: 1px solid #d9e2ef; }
#Brand { font-size: 19pt; font-weight: 700; color: #245ec7; }
#BrandCaption, [muted="true"] { color: #64748b; }
#PageTitle { font-size: 21pt; font-weight: 700; }
#PageSubtitle { color: #65748c; font-size: 10pt; }
#Card {
    background: #ffffff;
    border: 1px solid #d9e2ef;
    border-radius: 14px;
}
#VideoSurface {
    background: #0d1422;
    border: 2px solid #7fa5e5;
    border-radius: 18px;
}
#HandSurface {
    background: #e9eef6;
    border: 1px solid #bdcbe0;
    border-radius: 12px;
}
#StatusGood { color: #14804a; font-weight: 600; }
#StatusNeutral { color: #64748b; font-weight: 600; }
#StatusWarn { color: #a66700; font-weight: 600; }
#StatusBad { color: #b42335; font-weight: 600; }
QPushButton {
    background: #2864d7;
    color: white;
    border: 0;
    border-radius: 9px;
    padding: 9px 14px;
    font-weight: 600;
}
QPushButton:hover { background: #3975e6; }
QPushButton:checked { background: #dbe9ff; color: #174b9d; border: 1px solid #4d83dc; }
QPushButton:disabled { background: #c7d0df; color: #78869a; }
QPushButton[secondary="true"] {
    background: #eaf0f9;
    color: #24344e;
    border: 1px solid #c2cee0;
}
QPushButton[accent="true"] {
    background: #d9f6e8;
    color: #11623f;
    border: 1px solid #4fbd8c;
}
QPushButton[danger="true"] { background: #bd3b4e; }
QLineEdit, QTextEdit, QTextBrowser, QPlainTextEdit, QComboBox,
QSpinBox, QDoubleSpinBox, QTableWidget {
    background: #ffffff;
    color: #172033;
    border: 1px solid #c7d2e3;
    border-radius: 8px;
    padding: 7px;
    selection-background-color: #a8c6f4;
}
QComboBox QAbstractItemView { background: white; color: #172033; }
QHeaderView::section {
    background: #edf2f8;
    color: #46556d;
    padding: 7px;
    border: 0;
    border-bottom: 1px solid #cbd6e5;
}
QTableWidget { gridline-color: #dce4ef; }
QGroupBox {
    border: 1px solid #d2ddea;
    border-radius: 11px;
    margin-top: 13px;
    padding-top: 13px;
    font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QSlider::groove:horizontal { background: #d5deea; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #3977dc; width: 18px; margin: -6px 0; border-radius: 9px;
}
QSplitter::handle { background: #d3ddea; width: 2px; }
"""


def apply_theme(app: QApplication, theme: str) -> None:
    dark = (theme or "dark").lower() == "dark"
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#0b1020" if dark else "#f4f7fb"))
    palette.setColor(QPalette.WindowText, QColor("#e7ecf7" if dark else "#172033"))
    palette.setColor(QPalette.Base, QColor("#0c1426" if dark else "#ffffff"))
    palette.setColor(QPalette.Text, QColor("#ecf1fb" if dark else "#172033"))
    palette.setColor(QPalette.Button, QColor("#1b2740" if dark else "#eaf0f9"))
    palette.setColor(QPalette.ButtonText, QColor("#ecf1fb" if dark else "#172033"))
    app.setPalette(palette)
    app.setStyleSheet(DARK_QSS if dark else LIGHT_QSS)
