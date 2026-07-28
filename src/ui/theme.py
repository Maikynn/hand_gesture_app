from __future__ import annotations

from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication


DARK_QSS = """
QWidget {
    background: #050b11;
    color: #d9f8fb;
    font-family: "Segoe UI Variable", "Segoe UI";
    font-size: 10.5pt;
}
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget { background: #050b11; }
QLabel, QCheckBox, QRadioButton { background: transparent; }
#Sidebar {
    background: #07121b;
    border-right: 1px solid #1c6673;
}
#Brand { font-size: 18pt; font-weight: 700; color: #73efff; letter-spacing: 3px; }
#BrandCaption, [muted="true"] { color: #668690; }
#CoreStatus { color: #55ebbd; font-size: 9pt; font-weight: 700; letter-spacing: 2px; }
#HardwareBadge {
    color: #78cdd7;
    background: #081b24;
    border: 1px solid #1d5360;
    border-radius: 3px;
    padding: 7px;
    font-family: "Cascadia Mono";
    font-size: 8.5pt;
}
#HudClock { color: #73cbd6; font-family: "Cascadia Mono"; font-size: 9pt; }
#PageTitle {
    color: #bdf8ff;
    font-size: 19pt;
    font-weight: 650;
    letter-spacing: 2px;
}
#PageSubtitle { color: #648690; font-size: 9.5pt; }
#SectionTitle { color: #8eefff; font-size: 13pt; font-weight: 650; }
#Card {
    background: #091620;
    border: 1px solid #194754;
    border-radius: 5px;
}
#VideoSurface {
    background: #010407;
    border: 1px solid #38d4e8;
    border-radius: 3px;
}
#HandSurface {
    background: #02090d;
    border: 1px solid #26707d;
    border-radius: 2px;
}
#TelemetryGraph {
    border: 1px solid #20515d;
    border-radius: 3px;
}
#StatusGood { color: #55e8b6; font-weight: 600; }
#StatusNeutral { color: #76b9c3; font-weight: 600; }
#StatusWarn { color: #ffd166; font-weight: 600; }
#StatusBad { color: #ff7188; font-weight: 600; }
QPushButton {
    background: #0c7788;
    color: #edfdff;
    border: 1px solid #2ec8dc;
    border-radius: 3px;
    padding: 9px 14px;
    font-weight: 600;
}
QPushButton:hover { background: #119aae; border-color: #88f3ff; }
QPushButton:pressed { background: #075a67; }
QPushButton:checked { background: #115566; border: 1px solid #7af1ff; }
QPushButton:disabled { background: #15242b; color: #597079; border-color: #273f47; }
QPushButton[secondary="true"] {
    background: #0b202b;
    border: 1px solid #285666;
}
QPushButton[accent="true"] {
    background: #0a6d57;
    border: 1px solid #58e7bb;
}
QPushButton[danger="true"] { background: #77293a; border-color: #c9586c; }
QPushButton#NavButton {
    background: transparent;
    color: #6e919b;
    border: 0;
    border-left: 2px solid transparent;
    border-radius: 0;
    padding: 10px 12px;
    text-align: left;
    font-family: "Cascadia Mono", "Segoe UI";
    font-size: 9.5pt;
}
QPushButton#NavButton:hover {
    color: #b8f8ff;
    background: #0b202a;
    border-left: 2px solid #317c89;
}
QPushButton#NavButton:checked {
    color: #c9fbff;
    background: #0c2934;
    border-left: 2px solid #4fe8f8;
}
QLineEdit, QTextEdit, QTextBrowser, QPlainTextEdit, QComboBox,
QSpinBox, QDoubleSpinBox, QTableWidget {
    background: #061018;
    color: #dffcff;
    border: 1px solid #28525e;
    border-radius: 3px;
    padding: 7px;
    selection-background-color: #14687a;
}
QLineEdit:focus, QTextEdit:focus, QTextBrowser:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus { border-color: #4ce1f1; }
QComboBox QAbstractItemView {
    background: #091923;
    color: #dffcff;
    selection-background-color: #155467;
}
QHeaderView::section {
    background: #0c202b;
    color: #8ac7d0;
    padding: 7px;
    border: 0;
    border-bottom: 1px solid #27606c;
}
QTableWidget { gridline-color: #173a44; }
QGroupBox {
    background: #08151e;
    border: 1px solid #204a56;
    border-radius: 4px;
    margin-top: 13px;
    padding-top: 13px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 7px;
    color: #91eefa;
}
QSlider::groove:horizontal { background: #18313a; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #5ceafa;
    border: 1px solid #bffaff;
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
}
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; }
QScrollBar:vertical {
    background: #050b11;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #225865;
    min-height: 28px;
    border-radius: 4px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { background: #0b202b; color: #dffcff; border: 1px solid #3a8997; }
"""


LIGHT_QSS = """
QWidget {
    background: #edf7f8;
    color: #102b32;
    font-family: "Segoe UI Variable", "Segoe UI";
    font-size: 10.5pt;
}
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget { background: #edf7f8; }
QLabel, QCheckBox, QRadioButton { background: transparent; }
#Sidebar { background: #f8ffff; border-right: 1px solid #8fc7ce; }
#Brand { font-size: 18pt; font-weight: 700; color: #08798a; letter-spacing: 3px; }
#BrandCaption, [muted="true"] { color: #5a7b82; }
#CoreStatus { color: #0c8a68; font-size: 9pt; font-weight: 700; letter-spacing: 2px; }
#HardwareBadge {
    color: #246d78;
    background: #e6f5f6;
    border: 1px solid #9bcbd1;
    border-radius: 3px;
    padding: 7px;
    font-family: "Cascadia Mono";
    font-size: 8.5pt;
}
#HudClock { color: #287581; font-family: "Cascadia Mono"; font-size: 9pt; }
#PageTitle {
    color: #0c5966;
    font-size: 19pt;
    font-weight: 650;
    letter-spacing: 2px;
}
#PageSubtitle { color: #5a7b82; font-size: 9.5pt; }
#SectionTitle { color: #08798a; font-size: 13pt; font-weight: 650; }
#Card {
    background: #ffffff;
    border: 1px solid #a9d0d5;
    border-radius: 5px;
}
#VideoSurface {
    background: #031015;
    border: 1px solid #1593a4;
    border-radius: 3px;
}
#HandSurface {
    background: #e7f2f3;
    border: 1px solid #93c6cd;
    border-radius: 2px;
}
#StatusGood { color: #14804a; font-weight: 600; }
#StatusNeutral { color: #64748b; font-weight: 600; }
#StatusWarn { color: #a66700; font-weight: 600; }
#StatusBad { color: #b42335; font-weight: 600; }
QPushButton {
    background: #087d8e;
    color: white;
    border: 1px solid #0c7180;
    border-radius: 3px;
    padding: 9px 14px;
    font-weight: 600;
}
QPushButton:hover { background: #0b96aa; }
QPushButton:checked { background: #d5f4f7; color: #075d6a; border: 1px solid #1593a4; }
QPushButton:disabled { background: #c7d0df; color: #78869a; }
QPushButton[secondary="true"] {
    background: #e2f1f3;
    color: #153a42;
    border: 1px solid #9ac7cd;
}
QPushButton[accent="true"] {
    background: #d9f6e8;
    color: #11623f;
    border: 1px solid #4fbd8c;
}
QPushButton[danger="true"] { background: #bd3b4e; }
QPushButton#NavButton {
    background: transparent;
    color: #52767d;
    border: 0;
    border-left: 2px solid transparent;
    border-radius: 0;
    padding: 10px 12px;
    text-align: left;
    font-family: "Cascadia Mono", "Segoe UI";
    font-size: 9.5pt;
}
QPushButton#NavButton:hover {
    color: #0b6876;
    background: #e7f7f8;
    border-left: 2px solid #7ebfc7;
}
QPushButton#NavButton:checked {
    color: #064d58;
    background: #d8f1f3;
    border-left: 2px solid #078a9c;
}
QLineEdit, QTextEdit, QTextBrowser, QPlainTextEdit, QComboBox,
QSpinBox, QDoubleSpinBox, QTableWidget {
    background: #ffffff;
    color: #172033;
    border: 1px solid #a8cbd0;
    border-radius: 3px;
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
    background: #faffff;
    border: 1px solid #afd1d5;
    border-radius: 4px;
    margin-top: 13px;
    padding-top: 13px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #116d79;
}
QSlider::groove:horizontal { background: #b9d9dd; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #168fa0; width: 14px; margin: -6px 0; border-radius: 7px;
}
QScrollBar:vertical { background: #edf7f8; width: 8px; margin: 0; }
QScrollBar::handle:vertical {
    background: #84bbc2; min-height: 28px; border-radius: 4px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


def apply_theme(app: QApplication, theme: str) -> None:
    dark = (theme or "dark").lower() == "dark"
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#050b11" if dark else "#edf7f8"))
    palette.setColor(QPalette.WindowText, QColor("#d9f8fb" if dark else "#102b32"))
    palette.setColor(QPalette.Base, QColor("#061018" if dark else "#ffffff"))
    palette.setColor(QPalette.Text, QColor("#dffcff" if dark else "#102b32"))
    palette.setColor(QPalette.Button, QColor("#0b202b" if dark else "#e2f1f3"))
    palette.setColor(QPalette.ButtonText, QColor("#dffcff" if dark else "#102b32"))
    app.setPalette(palette)
    app.setStyleSheet(DARK_QSS if dark else LIGHT_QSS)
