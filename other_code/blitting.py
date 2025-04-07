import sys
import os
import pickle
from PyQt5.QtGui import QIcon, QColor
import lasio
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QMenu, QVBoxLayout, QFormLayout, QLabel, QLineEdit,
    QDialogButtonBox, QMainWindow, QDockWidget, QListWidget,
    QListWidgetItem, QWidget, QComboBox, QPushButton, QCheckBox, QSpinBox,
    QScrollArea, QAction, QColorDialog, QTabWidget, QFrame, QApplication, QToolBar
)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle

# Update the overall font size on plots
plt.rcParams.update({'font.size': 8.5})

def loadStyleSheet(fileName):
    try:
        with open(fileName, "r") as f:
            return f.read()
    except Exception as e:
        print("Failed to load stylesheet:", e)
        return ""

# --- New Widget: WellTopLinkWidget ---
class WellTopLinkWidget(QWidget):
    def __init__(self, left_well, right_well, well_tops, parent=None):
        """
        left_well, right_well: names of the two wells.
        well_tops: dictionary mapping well names to list of (top, md)
        """
        super().__init__(parent)
        self.left_well = left_well
        self.right_well = right_well
        self.well_tops = well_tops
        self.canvas = FigureCanvas(Figure(figsize=(1, 5)))
        self.ax = self.canvas.figure.add_subplot(111)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.draw_links()

    def draw_links(self):
        self.ax.clear()
        # Set a simple x-range so that we have two points: left at x=0 and right at x=1.
        self.ax.set_xlim(-0.1, 1.1)
        # Invert the y-axis since depth increases downward.
        self.ax.invert_yaxis()
        # Get tops for left and right wells
        left_tops = {top: md for top, md in self.well_tops.get(self.left_well, [])}
        right_tops = {top: md for top, md in self.well_tops.get(self.right_well, [])}
        # For every top that exists in both wells, draw a connecting line.
        for top in left_tops:
            if top in right_tops:
                md_left = left_tops[top]
                md_right = right_tops[top]
                self.ax.plot([0, 1], [md_left, md_right], color='red', linestyle='--', linewidth=1)
                # Mark the top points
                self.ax.plot(0, md_left, 'ro')
                self.ax.plot(1, md_right, 'ro')
                # Label the connection in the middle
                mid_x = 0.5
                mid_y = (md_left + md_right) / 2
                self.ax.text(mid_x, mid_y, top, fontsize=8, ha='center', va='bottom', color='red')
        # Hide axes for a cleaner look.
        self.ax.axis('off')
        self.canvas.draw()

# --- Custom QListWidget: Clicking on an item's label toggles its check state ---
class ClickableListWidget(QListWidget):
    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if item is not None:
            rect = self.visualItemRect(item)
            # Assume the checkbox is within the left 20 pixels.
            if event.pos().x() > rect.left() + 20:
                if item.checkState() == Qt.Checked:
                    item.setCheckState(Qt.Unchecked)
                else:
                    item.setCheckState(Qt.Checked)
                return
        super().mousePressEvent(event)

class FigureWidget(QWidget):
    mouse_moved = pyqtSignal(float, float)
    external_crosshair = pyqtSignal(float, float)
    zoomChanged = pyqtSignal(object)  # Signal emitted after a zoom event

    def __init__(self, well_name, parent=None):
        super().__init__(parent)
        self.well_name = well_name
        self.figure = Figure(layout="constrained")  # Use constrained layout
        self.figure.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)
        

        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
         # Now connect events
        self.canvas.mpl_connect('resize_event', self.handle_resize)  # Moved after canvas creation
        self.crosshair_hlines = []
        self.crosshair_vline = None
        self.cursor_coords = None

        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.external_crosshair.connect(self.on_external_crosshair)

        # Variables for zoom functionality
        self._dragging = False
        self._press_event = None
        self._rect = None
        self._zoom_history = []  # For undo functionality
        self._initial_limits = None
        self.current_zoom_limits = None  # Stores current zoom state
        self.zoom_mode = None

        # Connect mouse events
        self.canvas.mpl_connect("button_press_event", self.onMousePress)
        self.canvas.mpl_connect("motion_notify_event", self.onMouseMove)
        self.canvas.mpl_connect("button_release_event", self.onMouseRelease)

    def setZoomMode(self, mode):
        self.zoom_mode = mode
        if mode == "Rectangular":
            self.canvas.setCursor(Qt.CrossCursor)
        else:
            self.canvas.setCursor(Qt.ArrowCursor)

    def onMousePress(self, event):
        if event.inaxes is None or self.zoom_mode != "Rectangular":
            return

        self._dragging = True
        self._press_event = event
        ax = event.inaxes
        self._rect = Rectangle((event.xdata, event.ydata), 0, 0,
                               fill=False, edgecolor='red', linestyle='--')
        ax.add_patch(self._rect)
        self.canvas.draw()

    def onMouseMove(self, event):
        if not self._dragging or event.inaxes is None or self._press_event is None or self._rect is None:
            return

        x0, y0 = self._press_event.xdata, self._press_event.ydata
        x1, y1 = event.xdata, event.ydata
        xmin = min(x0, x1)
        ymin = min(y0, y1)
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        self._rect.set_xy((xmin, ymin))
        self._rect.set_width(width)
        self._rect.set_height(height)
        self.canvas.draw()

    def onMouseRelease(self, event):
        if not self._dragging or event.inaxes is None or self._press_event is None:
            return

        ax = event.inaxes
        if self._rect is not None:
            x0, y0 = self._press_event.xdata, self._press_event.ydata
            x1, y1 = event.xdata, event.ydata
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])

            # Store the zoom limits
            self.current_zoom_limits = (xmin, xmax, ymin, ymax)

            self._rect.remove()
            self._rect = None
            self.canvas.draw()

            # Emit signal that zoom has changed
            self.zoomChanged.emit(self)

        self._dragging = False
        self._press_event = None

    def applyZoom(self, xmin, xmax, ymin, ymax):
        """Apply zoom limits to all axes in this figure"""
        for ax in self.figure.axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymax, ymin)
        self.current_zoom_limits = (xmin, xmax, ymin, ymax)
        self.canvas.draw()

    def undoZoom(self):
        """Undo the last zoom operation"""
        if len(self._zoom_history) > 0:
            prev_limits = self._zoom_history.pop()
            self.applyZoom(*prev_limits)
            return True
        elif self._initial_limits:
            # If no zoom history,   revert to initial limits
            self.resetZoom()
        return False

    def resetZoom(self):
        """Reset zoom to initial state"""
        if self._initial_limits:
            for ax, limits in zip(self.figure.axes, self._initial_limits):
                ax.set_xlim(limits[0])
                ax.set_ylim(limits[1])
            self.current_zoom_limits = None
            self._zoom_history = []
            self.canvas.draw()

    def recordCurrentZoom(self):
        """Record current zoom state for undo functionality"""
        if self.current_zoom_limits:
            self._zoom_history.append(self.current_zoom_limits)

    def on_mouse_move(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.mouse_moved.emit(x, y)
            self.update_crosshair(event.inaxes, x, y)

    def on_external_crosshair(self, x, y):
        axes = self.figure.get_axes()
        if axes:
            self.update_crosshair(axes[0], x, y, external=True)

    def update_crosshair(self, ax, x, y, external=False):
        if not self.background_captured or not self.crosshair_hlines or (external and not self.crosshair_vline):
            return

        # Restore the background
        self.canvas.restore_region(self.background)

        # Update horizontal lines
        for hline in self.crosshair_hlines:
            hline.set_ydata([y])
            hline.set_visible(True)
            hline.axes.draw_artist(hline)

        # Update vertical line and text if not external
        if not external:
            if self.crosshair_vline:
                self.crosshair_vline.set_xdata([x])
                self.crosshair_vline.set_visible(True)
                self.crosshair_vline.axes.draw_artist(self.crosshair_vline)
            if self.cursor_coords:
                self.cursor_coords.set_text(f'x={x:.2f}, y={y:.2f}')
                self.cursor_coords.set_position((x, y))
                self.cursor_coords.set_visible(True)
                ax.draw_artist(self.cursor_coords)

        # Blit the updated regions
        self.canvas.blit(self.figure.bbox)

    def remove_crosshair(self):
        if self.background_captured:
            self.canvas.restore_region(self.background)
            self.canvas.blit(self.figure.bbox)
        # Hide crosshair elements
        for hline in self.crosshair_hlines:
            hline.set_visible(False)
        if self.crosshair_vline:
            self.crosshair_vline.set_visible(False)
        if self.cursor_coords:
            self.cursor_coords.set_visible(False)

    def update_plot(self, data, tracks, well_top_lines=None):
        self.figure.clear()
        self.figure.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)
        self.data = data
        self.tracks = tracks
        n_tracks = len(tracks)
        if n_tracks == 0:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No tracks", ha='center', va='center')
        else:
            axes = self.figure.subplots(1, n_tracks, sharey=True) if n_tracks > 1 else [self.figure.add_subplot(111)]
            depth = data['DEPT']
            for idx, (ax, track) in enumerate(zip(axes, tracks)):
                ax.set_facecolor(track.bg_color)
                if idx != 0:
                    ax.tick_params(left=False, labelleft=False)
                track.ax = ax
                valid_curves = []
                lines_list = []
                if not track.curves:
                    ax.text(0.5, 0.5, "No curves", ha='center', va='center')
                    continue
                for i, curve in enumerate(track.curves):
                    curve_name = curve.curve_box.currentText()
                    if curve_name == "Select Curve" or curve_name not in data.columns:
                        continue
                    twin_ax = ax.twiny()
                    twin_ax.xaxis.set_ticks_position('top')
                    twin_ax.xaxis.set_label_position('top')
                    twin_ax.spines['top'].set_color(curve.color)
                    twin_ax.spines['top'].set_linewidth(2)
                    twin_ax.spines['top'].set_position(('axes', 1 + i * 0.025))
                    twin_ax.tick_params(axis='x', colors=curve.color)
                    twin_ax.set_xlabel(curve_name, color=curve.color)
                    line, = twin_ax.plot(
                        data[curve_name], depth,
                        color=curve.color,
                        linewidth=curve.width.value(),
                        linestyle=curve.get_line_style(),
                        label=curve_name,
                        picker=True
                    )
                    line.set_gid(curve_name)
                    valid_curves.append(curve)
                    lines_list.append(line)
                    if curve.flip.isChecked():
                        twin_ax.invert_xaxis()
                    if curve.x_min.text():
                        try:
                            twin_ax.set_xlim(float(curve.x_min.text()), twin_ax.get_xlim()[1])
                        except ValueError:
                            pass
                    if curve.x_max.text():
                        try:
                            twin_ax.set_xlim(twin_ax.get_xlim()[0], float(curve.x_max.text()))
                        except ValueError:
                            pass
                    if curve.scale_combobox.currentText() == "Log":
                        twin_ax.set_xscale('log')
                    else:
                        twin_ax.set_xscale('linear')
                if idx == 0:
                    ax.set_ylabel("Depth")
                ax.grid(track.grid.isChecked())
                ax.set_ylim(depth.max(), depth.min())
                if track.flip_y.isChecked():
                    ax.invert_yaxis()
                if track.y_min.text():
                    try:
                        ax.set_ylim(float(track.y_min.text()), ax.get_ylim()[1])
                    except ValueError:
                        pass
                if track.y_max.text():
                    try:
                        ax.set_ylim(ax.get_ylim()[0], float(track.y_max.text()))
                    except ValueError:
                        pass
                ax.set_xticklabels([])
            self.figure.suptitle(f"Well: {self.well_name}", fontsize=11, alpha=0.6)
            if well_top_lines:
                for track in tracks:
                    for (top, md) in well_top_lines:
                        track.ax.axhline(y=md, color='red', linestyle='--', linewidth=1)
                        track.ax.text(
                            0.02, md, f"{self.well_name}: {top}",
                            transform=track.ax.get_yaxis_transform(),
                            color='red', fontsize=8, horizontalalignment='left', verticalalignment='bottom'
                        )
        self.canvas.draw()
        self._initial_limits = []
        for ax in self.figure.axes:
            self._initial_limits.append((ax.get_xlim(), ax.get_ylim()))
        if self.current_zoom_limits:
            self.applyZoom(*self.current_zoom_limits)
        
        # Create crosshair lines (initially hidden)
        self.crosshair_hlines = []
        for ax in self.figure.axes:
            hline = ax.axhline(0, color='red', linestyle='--', linewidth=1, visible=False)
            self.crosshair_hlines.append(hline)
        if self.figure.axes:
            self.crosshair_vline = self.figure.axes[0].axvline(0, color='red', linestyle='--', linewidth=1, visible=False)
            self.cursor_coords = self.figure.axes[0].text(
                0, 0, '', visible=False, transform=self.figure.axes[0].transData,
                fontsize=9, bbox=dict(boxstyle='round,pad=0.1', facecolor='yellow', alpha=0.5)
            )
        else:
            self.crosshair_vline = None
            self.cursor_coords = None

        self.canvas.draw()
        # Capture the background for blitting
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)
        self.background_captured = True

    def handle_resize(self, event):
        # Re-capture background after resize
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)
        self.background_captured = True

    def applyZoom(self, xmin, xmax, ymin, ymax):
        """Apply zoom limits to all axes in this figure"""
        for ax in self.figure.axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymax, ymin)
        self.current_zoom_limits = (xmin, xmax, ymin, ymax)
        self.canvas.draw()
        # Re-capture background after zoom
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)
        self.background_captured = True

    def resetZoom(self):
        """Reset zoom to initial state"""
        if self._initial_limits:
            for ax, limits in zip(self.figure.axes, self._initial_limits):
                ax.set_xlim(limits[0])
                ax.set_ylim(limits[1])
            self.current_zoom_limits = None
            self._zoom_history = []
            self.canvas.draw()
            # Re-capture background after reset
            self.background = self.canvas.copy_from_bbox(self.figure.bbox)
            self.background_captured = True

class CurveControl(QWidget):
    changed = pyqtSignal()
    def __init__(self, curve_number, curves, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        self.setStyleSheet("""
            border-radius: 5px;
            color: Black;
            font: 10pt;
            padding: 2px;
            height: 30px;
        """)
        self.curve_label = QLabel(f"Curve {curve_number}:")
        layout.addWidget(self.curve_label)
        self.curve_box = QComboBox()
        self.curve_box.addItem("Select Curve")
        self.curve_box.addItems(curves)
        self.curve_box.currentIndexChanged.connect(self.changed.emit)
        layout.addWidget(self.curve_box)
        self.width = QSpinBox()
        self.width.setRange(1, 5)
        self.width.setValue(1)
        self.width.valueChanged.connect(self.changed.emit)
        layout.addWidget(QLabel("Width:"))
        layout.addWidget(self.width)
        self.color = "#000000"
        self.color_btn = QPushButton("Color")
        self.color_btn.setStyleSheet(f"background-color: {self.color}; border: none;")
        self.color_btn.clicked.connect(self.select_color)
        layout.addWidget(self.color_btn)
        self.line_style_box = QComboBox()
        self.line_style_box.addItems(["Solid", "Dashed", "Dotted", "Dash-dot"])
        self.line_style_box.currentIndexChanged.connect(self.changed.emit)
        layout.addWidget(QLabel("Style:"))
        layout.addWidget(self.line_style_box)
        self.flip = QCheckBox("X-Flip")
        self.flip.stateChanged.connect(self.changed.emit)
        layout.addWidget(self.flip)
        xy_range_layout = QHBoxLayout()
        xy_range_layout.addWidget(QLabel("X-min:"))
        self.x_min = QLineEdit()
        self.x_min.setStyleSheet("background-color: White; color: blue; font: 12pt;")
        self.x_min.setFixedWidth(50)
        self.x_min.setPlaceholderText("Auto")
        self.x_min.textChanged.connect(self.changed.emit)
        xy_range_layout.addWidget(self.x_min)
        xy_range_layout.addWidget(QLabel("X-max:"))
        self.x_max = QLineEdit()
        self.x_max.setStyleSheet("background-color: White; color: blue; font: 10pt;")
        self.x_max.setFixedWidth(50)
        self.x_max.setPlaceholderText("Auto")
        self.x_max.textChanged.connect(self.changed.emit)
        xy_range_layout.addWidget(self.x_max)
        self.scale_combobox = QComboBox()
        self.scale_combobox.addItems(["Linear", "Log"])
        self.scale_combobox.currentIndexChanged.connect(self.changed.emit)
        xy_range_layout.addWidget(self.scale_combobox)
        layout.addLayout(xy_range_layout)
    def select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.color = color.name()
            self.color_btn.setStyleSheet(f"background-color: {self.color}; border: none;")
            self.curve_label.setStyleSheet(f"color: {self.color};")
            self.changed.emit()
    def get_line_style(self):
        styles = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-dot": "-."}
        return styles[self.line_style_box.currentText()]

class TrackControl(QWidget):
    changed = pyqtSignal()
    def __init__(self, number, curves, parent=None):
        super().__init__(parent)
        self.number = number
        self.curves = []
        self.bg_color = "#FFFFFF"
        self.curve_count = 0
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.setStyleSheet("""
            background-color: White;
            border-radius: 5px;
            color: #53003e;
            font: 10pt;
            padding: 5px;
            min-width: fit-content;
        """)
        layout = QVBoxLayout(self)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        range_layout = QHBoxLayout()
        self.grid = QCheckBox("Grid")
        self.grid.setFixedWidth(100)
        self.grid.stateChanged.connect(self.changed.emit)
        range_layout.addWidget(self.grid)
        self.flip_y = QCheckBox("Flip Y-Axis")
        self.flip_y.stateChanged.connect(self.changed.emit)
        self.flip_y.setFixedWidth(100)
        range_layout.addWidget(self.flip_y)
        self.bg_color_btn = QPushButton("Bg Color")
        self.bg_color_btn.setStyleSheet(f"background-color: {self.bg_color}; border: none;")
        self.bg_color_btn.setFixedWidth(100)
        self.bg_color_btn.clicked.connect(self.select_bg_color)
        range_layout.addWidget(self.bg_color_btn)
        y_min_label = QLabel("Y min:")
        y_min_label.setFixedWidth(50)
        range_layout.addWidget(y_min_label)
        self.y_min = QLineEdit()
        self.y_min.setStyleSheet("background-color: White; color: blue; font: 12pt;")
        self.y_min.setPlaceholderText("Auto")
        self.y_min.setFixedWidth(60)
        self.y_min.textChanged.connect(self.changed.emit)
        range_layout.addWidget(self.y_min)
        y_max_label = QLabel("Y max:")
        y_max_label.setFixedWidth(50)
        range_layout.addWidget(y_max_label)
        self.y_max = QLineEdit()
        self.y_max.setStyleSheet("background-color: White; color: blue; font: 12pt;")
        self.y_max.setPlaceholderText("Auto")
        self.y_max.setFixedWidth(60)
        self.y_max.textChanged.connect(self.changed.emit)
        range_layout.addWidget(self.y_max)
        layout.addLayout(range_layout)
        self.curve_tabs = QTabWidget()
        self.curve_tabs.setTabsClosable(True)
        self.curve_tabs.tabCloseRequested.connect(self.remove_curve)
        layout.addWidget(self.curve_tabs)
        add_curve_btn = QPushButton("Add Curve")
        add_curve_btn.setFixedSize(150, 30)
        add_curve_btn.setStyleSheet("""
            background-color: White;
            border-radius: 5px;
            color: blue;
            font: 12pt;
            font-weight: bold;
        """)
        add_curve_btn.clicked.connect(lambda: self.add_curve(curves))
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        btn_layout.addWidget(add_curve_btn)
        btn_layout.addStretch(1)
        layout.addLayout(btn_layout)
        self.setFixedHeight(300)
        self.add_curve(curves)
    def select_bg_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.bg_color = color.name()
            self.bg_color_btn.setStyleSheet(f"background-color: {self.bg_color}; border: none;")
            self.changed.emit()
    def add_curve(self, curves):
        self.curve_count += 1
        curve = CurveControl(self.curve_count, curves)
        curve.changed.connect(self.changed.emit)
        self.curves.append(curve)
        self.curve_tabs.addTab(curve, f"Curve {self.curve_count}")
        self.update_curve_numbers()
        self.changed.emit()
    def remove_curve(self, index):
        curve = self.curve_tabs.widget(index)
        if curve:
            self.curves.remove(curve)
            self.curve_tabs.removeTab(index)
            curve.deleteLater()
            self.update_curve_numbers()
            self.changed.emit()
    def update_curve_numbers(self):
        for i, curve in enumerate(self.curves, start=1):
            curve.curve_label.setText(f"Curve {i}:")
            self.curve_tabs.setTabText(i - 1, f"Curve {i}")

class WellLogViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.wells = {}
        self.well_tops = {}
        self.selected_top_names = set()
        self.tracks = []
        self.figure_widgets = {}
        self.show_well_tops = True
        self.sync_zoom_enabled = False
        self.current_single_zoom_well = None
        self.single_zoom_limits = None
        self.sync_zoom_limits = None
        self.share_y_axis_enabled = False
        self.link_well_tops_enabled = False
        self.link_widgets = []
        self.initUI()
        self.setWindowIcon(QIcon('images/ONGC_Logo.png'))
        self.enableSingleZoom()
    def initUI(self):
        self.setWindowTitle('Well Log Viewer')
        self.setGeometry(100, 100, 1200, 800)
        self.figure_scroll = QScrollArea()
        self.figure_container = QWidget()
        self.figure_layout = QHBoxLayout(self.figure_container)
        self.figure_scroll.setWidgetResizable(True)
        self.figure_scroll.setWidget(self.figure_container)
        self.setCentralWidget(self.figure_scroll)
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        load_files_action = QAction("Load LAS Files", self)
        load_files_action.triggered.connect(self.load_las_files)
        file_menu.addAction(load_files_action)
        load_welltops_action = QAction("Load Well Tops", self)
        load_welltops_action.triggered.connect(self.load_well_tops)
        file_menu.addAction(load_welltops_action)
        settings_menu = menubar.addMenu("Settings")
        save_template_action = QAction("Save Template", self)
        save_template_action.triggered.connect(self.save_template)
        settings_menu.addAction(save_template_action)
        load_template_action = QAction("Load Template", self)
        load_template_action.triggered.connect(self.load_template)
        settings_menu.addAction(load_template_action)
        toggle_controls_action = QAction("Toggle Controls", self)
        toggle_controls_action.triggered.connect(self.toggle_controls)
        menubar.addAction(toggle_controls_action)
        change_bg_action = QAction("Themes", self)
        change_bg_action.triggered.connect(self.change_background_color)
        menubar.addAction(change_bg_action)
        self.toggle_well_tops_action = QAction("Hide Well Tops", self)
        self.toggle_well_tops_action.triggered.connect(self.toggle_well_tops)
        menubar.addAction(self.toggle_well_tops_action)
        self.sync_zoom_action = QAction("Enable Sync Zoom", self, checkable=True)
        self.sync_zoom_action.toggled.connect(self.onSyncZoomToggled)
        menubar.addAction(self.sync_zoom_action)
        self.undo_zoom_action = QAction("Undo Zoom", self)
        self.undo_zoom_action.triggered.connect(self.undoZoom)
        menubar.addAction(self.undo_zoom_action)
        self.reset_zoom_action = QAction("Reset Zoom", self)
        self.reset_zoom_action.triggered.connect(self.resetZoom)
        menubar.addAction(self.reset_zoom_action)
        self.share_y_axis_action = QAction("Link Y Axis", self, checkable=True)
        self.share_y_axis_action.toggled.connect(self.onShareYAxisToggled)
        menubar.addAction(self.share_y_axis_action)
        self.link_well_tops_action = QAction("Link Well Top", self)
        self.link_well_tops_action.setCheckable(True)
        self.link_well_tops_action.triggered.connect(self.toggle_link_well_tops)
        menubar.addAction(self.link_well_tops_action)
        self.dock = QDockWidget("Control", self)
        self.dock.setStyleSheet("background-color: White; border-radius: 5px; color: blue; font: 12pt;")
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        dock_widget = QWidget()
        dock_layout = QVBoxLayout()
        list_layout = QHBoxLayout()
        self.well_list = ClickableListWidget()
        self.well_list.itemChanged.connect(self.update_plot)
        well_label = QLabel("Wells:")
        well_layout = QVBoxLayout()
        well_layout.addWidget(well_label)
        well_layout.addWidget(self.well_list)
        list_layout.addLayout(well_layout)
        self.well_tops_list = ClickableListWidget()
        self.well_tops_list.itemChanged.connect(self.well_top_item_changed)
        welltops_label = QLabel("Well Tops:")
        welltops_layout = QVBoxLayout()
        welltops_layout.addWidget(welltops_label)
        welltops_layout.addWidget(self.well_tops_list)
        list_layout.addLayout(welltops_layout)
        dock_layout.addLayout(list_layout)
        btn_add_track = QPushButton("Track +")
        btn_add_track.setStyleSheet("background-color: White; border-radius: 5px; color: blue; font: 15pt; font-weight: bold;")
        btn_add_track.clicked.connect(self.add_track)
        dock_layout.addWidget(btn_add_track)
        self.track_tabs = QTabWidget()
        self.track_tabs.setStyleSheet("background-color: #e2e2e2; border-radius: 5px; color: #53003e; font: 10pt;")
        self.track_tabs.setTabsClosable(True)
        self.track_tabs.tabCloseRequested.connect(self.delete_track)
        dock_layout.addWidget(self.track_tabs)
        dock_widget.setLayout(dock_layout)
        self.dock.setWidget(dock_widget)
    def toggle_link_well_tops(self, checked):
        self.link_well_tops_enabled = checked
        if self.link_well_tops_enabled:
            self.link_well_tops_action.setText("Unlink Well Top")
        else:
            self.link_well_tops_action.setText("Link Well Top")
        self.update_plot()
    def onShareYAxisToggled(self, checked):
        self.share_y_axis_enabled = checked
        if checked:
            self.share_y_axis_action.setText("Unlink Y Axis")
            self.synchronizeYAxisLimits()
        else:
            self.share_y_axis_action.setText("Link Y Axis")
            self.update_plot()
    def synchronizeYAxisLimits(self):
            """Synchronize Y-axis limits across all wells."""
            if not self.figure_widgets:
                return

            # Get the Y-axis limits from the first well
            first_widget = next(iter(self.figure_widgets.values()))
            y_min, y_max = first_widget.figure.axes[0].get_ylim()

            # Apply the Y-axis limits to all wells
            for idx, widget in enumerate(self.figure_widgets.values()):
                for ax in widget.figure.axes:
                    ax.set_ylim(y_min, y_max)
                    if idx != 0:  # Hide Y-axis tick labels for all but the first well
                        ax.tick_params(labelleft=False)
                widget.canvas.draw()
    def onSyncZoomToggled(self, checked):
        self.sync_zoom_enabled = checked
        if checked:
            self.disableSingleZoom()
            self.enableSyncZoom()
            if self.single_zoom_limits:
                self.sync_zoom_limits = self.single_zoom_limits
                for widget in self.figure_widgets.values():
                    widget.applyZoom(*self.single_zoom_limits)
                    widget.recordCurrentZoom()
            self.sync_zoom_action.setText("Disable Sync Zoom")
        else:
            self.disableSyncZoom()
            self.sync_zoom_action.setText("Enable Sync Zoom")
            self.enableSingleZoom()
    def enableSyncZoom(self):
        for widget in self.figure_widgets.values():
            widget.setZoomMode("Rectangular")
            widget.zoomChanged.connect(self.handleSyncZoom)
    def disableSyncZoom(self):
        for widget in self.figure_widgets.values():
            widget.setZoomMode("Rectangular")
            try:
                widget.zoomChanged.disconnect(self.handleSyncZoom)
            except TypeError:
                pass
            widget.zoomChanged.connect(self.handleSingleZoom)
    def enableSingleZoom(self):
        for widget in self.figure_widgets.values():
            widget.setZoomMode("Rectangular")
            widget.zoomChanged.connect(self.handleSingleZoom)
    def disableSingleZoom(self):
        for widget in self.figure_widgets.values():
            try:
                widget.zoomChanged.disconnect(self.handleSingleZoom)
            except TypeError:
                pass
    def handleSyncZoom(self, sender):
        if not sender.current_zoom_limits:
            return
        self.sync_zoom_limits = sender.current_zoom_limits
        for widget in self.figure_widgets.values():
            widget.applyZoom(*sender.current_zoom_limits)
            widget.recordCurrentZoom()
    def handleSingleZoom(self, sender):
        if not sender.current_zoom_limits:
            return
        self.current_single_zoom_well = sender.well_name
        self.single_zoom_limits = sender.current_zoom_limits
        for widget in self.figure_widgets.values():
            if widget.well_name == sender.well_name:
                widget.applyZoom(*sender.current_zoom_limits)
                widget.recordCurrentZoom()
    def undoZoom(self):
        if self.sync_zoom_enabled:
            for widget in self.figure_widgets.values():
                widget.undoZoom()
        else:
            if self.current_single_zoom_well:
                widget = self.figure_widgets[self.current_single_zoom_well]
                widget.undoZoom()
    def resetZoom(self):
        if self.sync_zoom_enabled:
            for widget in self.figure_widgets.values():
                widget.resetZoom()
            self.sync_zoom_limits = None
        else:
            if self.current_single_zoom_well:
                widget = self.figure_widgets[self.current_single_zoom_well]
                widget.resetZoom()
                self.current_single_zoom_well = None
                self.single_zoom_limits = None
    def toggle_well_tops(self):
        self.show_well_tops = not self.show_well_tops
        self.toggle_well_tops_action.setText("Hide Well Tops" if self.show_well_tops else "Show Well Tops")
        self.update_plot()
    def update_plot(self):
        selected_wells = [self.well_list.item(i).text() for i in range(self.well_list.count())
                          if self.well_list.item(i).checkState() == Qt.Checked]
        # Disconnect signals safely from existing widgets
        for widget in list(self.figure_widgets.values()):
            try:
                widget.mouse_moved.disconnect()
            except (RuntimeError, TypeError):
                pass
        # Remove unselected well widgets
        for well in list(self.figure_widgets.keys()):
            if well not in selected_wells:
                widget = self.figure_widgets[well]
                self.figure_layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
                del self.figure_widgets[well]
        # Remove existing link widgets
        for link_widget in self.link_widgets:
            self.figure_layout.removeWidget(link_widget)
            link_widget.setParent(None)
            link_widget.deleteLater()
        self.link_widgets = []
        # Clear the layout without deleting widgets that are in self.figure_widgets
        while self.figure_layout.count():
            item = self.figure_layout.takeAt(0)
            if item.widget():
                # If widget is one of our well figure widgets, just remove it from layout
                if item.widget() in self.figure_widgets.values():
                    item.widget().setParent(None)
                else:
                    item.widget().deleteLater()
        new_widgets = []
        for well in selected_wells:
            if well not in self.figure_widgets:
                self.figure_widgets[well] = FigureWidget(well)
                if self.sync_zoom_enabled:
                    self.figure_widgets[well].setZoomMode("Rectangular")
                    self.figure_widgets[well].zoomChanged.connect(self.handleSyncZoom)
                else:
                    self.figure_widgets[well].setZoomMode("Rectangular")
                    self.figure_widgets[well].zoomChanged.connect(self.handleSingleZoom)
            new_widgets.append(self.figure_widgets[well])
        # Insert linking widgets if enabled and more than one well selected
        if self.link_well_tops_enabled and len(new_widgets) > 1:
            combined_widgets = []
            for i in range(len(new_widgets) - 1):
                combined_widgets.append(new_widgets[i])
                link_widget = WellTopLinkWidget(
                    new_widgets[i].well_name,
                    new_widgets[i+1].well_name,
                    self.well_tops
                )
                link_widget.setFixedWidth(100)
                combined_widgets.append(link_widget)
                self.link_widgets.append(link_widget)
            combined_widgets.append(new_widgets[-1])
            new_widgets = combined_widgets
        for widget in new_widgets:
            self.figure_layout.addWidget(widget)
        # Update each well's figure
        for well in selected_wells:
            well_top_lines = []
            if well in self.well_tops and self.show_well_tops:
                for top, md in self.well_tops[well]:
                    if top in self.selected_top_names:
                        well_top_lines.append((top, md))
            widget = self.figure_widgets[well]
            widget.update_plot(self.wells[well]['data'], self.tracks, well_top_lines)
            if self.sync_zoom_enabled and self.sync_zoom_limits:
                widget.applyZoom(*self.sync_zoom_limits)
            elif self.current_single_zoom_well == well and self.single_zoom_limits:
                widget.applyZoom(*self.single_zoom_limits)
        if self.share_y_axis_enabled:
            self.synchronizeYAxisLimits()
    def change_background_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.setStyleSheet(f"QWidget {{ background-color: {color.name()}; }}")
    def toggle_controls(self):
        self.dock.setVisible(not self.dock.isVisible())
    def load_las_files(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Select LAS Files", "", "LAS Files (*.las);;All Files (*)", options=options)
        if files:
            for file in files:
                self.load_las_file(file)
            self.update_plot()
    def load_las_file(self, path):
        try:
            las = lasio.read(path)
            df = las.df()
            df.reset_index(inplace=True)
            df.dropna(inplace=True)
            depth_col = next((col for col in df.columns if col.upper() in ["DEPT", "DEPTH", "MD"]), None)
            if depth_col is None:
                raise ValueError("No valid depth column found.")
            df.rename(columns={depth_col: "DEPT"}, inplace=True)
            well_name = las.well.WELL.value if las.well.WELL.value else os.path.basename(path)
            if well_name in self.wells:
                return
            self.wells[well_name] = {'data': df, 'path': path}
            item = QListWidgetItem(well_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.well_list.addItem(item)
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
    def load_well_tops(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Well Tops File", "", "Text Files (*.txt *.csv)"
        )
        if not file_path:
            return
        try:
            delimiter = ','
            if file_path.endswith('.txt'):
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    has_comma = any(',' in line for line in lines)
                    if has_comma:
                        df = pd.read_csv(file_path, delimiter=',', header=None, engine='python', on_bad_lines='skip')
                    else:
                        df = pd.read_csv(file_path, delim_whitespace=True, header=None, engine='python', on_bad_lines='skip')
            num_cols = df.apply(lambda row: row.count(), axis=1).mode()[0]
            df = df.iloc[:, :num_cols]
            header_present = False
            try:
                float(df.iloc[0, 2])
            except ValueError:
                header_present = True
            if header_present:
                df = df.drop(0).reset_index(drop=True)
            if df.shape[1] > 3:
                df = df.iloc[:, :3]
            df.columns = ["well", "top", "md"]
            for idx, row in df.iterrows():
                well = str(row["well"]).strip()
                top = str(row["top"]).strip()
                try:
                    md = float(row["md"])
                except ValueError:
                    continue
                if well not in self.well_tops:
                    self.well_tops[well] = []
                self.well_tops[well].append((top, md))
            self.update_well_tops_list()
        except Exception as e:
            print(f"Error loading well tops from {file_path}: {str(e)}")
    def update_well_tops_list(self):
        self.well_tops_list.clear()
        unique_tops = set()
        for tops in self.well_tops.values():
            for top, md in tops:
                unique_tops.add(top)
        for top in sorted(unique_tops):
            item = QListWidgetItem(top)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, top)
            self.well_tops_list.addItem(item)
    def well_top_item_changed(self, item):
        top = item.data(Qt.UserRole)
        if not top:
            return
        if item.checkState() == Qt.Checked:
            self.selected_top_names.add(top)
        else:
            self.selected_top_names.discard(top)
        self.update_plot()
    def add_track(self):
        if not self.wells:
            return
        curves = sorted(set(curve for well in self.wells.values() for curve in well['data'].columns))
        track = TrackControl(len(self.tracks) + 1, curves)
        track.changed.connect(self.update_plot)
        self.tracks.append(track)
        self.track_tabs.addTab(track, f"Track {track.number}")
        self.update_plot()
    def delete_track(self, index):
        track = self.track_tabs.widget(index)
        if track:
            self.tracks.remove(track)
            self.track_tabs.removeTab(index)
            track.deleteLater()
            self.renumber_tracks()
            self.update_plot()
    def renumber_tracks(self):
        for i, track in enumerate(self.tracks, start=1):
            track.number = i
            track.update_curve_numbers()
            self.track_tabs.setTabText(i - 1, f"Track {i}")
    def save_template(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Template", "", "Template Files (*.pkl)")
        if file_path:
            file_path = file_path + ".pkl"
            template_data = {
                'tracks': [track.number for track in self.tracks],
                'track_settings': [self.get_track_settings(track) for track in self.tracks]
            }
            with open(file_path, 'wb') as f:
                pickle.dump(template_data, f)
    def load_template(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Template", "", "Template Files (*.pkl)")
        if file_path:
            with open(file_path, 'rb') as f:
                template_data = pickle.load(f)
            self.apply_template(template_data)
    def get_track_settings(self, track):
        return {
            'bg_color': track.bg_color,
            'grid': track.grid.isChecked(),
            'flip_y': track.flip_y.isChecked(),
            'y_min': track.y_min.text(),
            'y_max': track.y_max.text(),
            'curves': [self.get_curve_settings(curve) for curve in track.curves]
        }
    def get_curve_settings(self, curve):
        return {
            'width': curve.width.value(),
            'color': curve.color,
            'line_style': curve.get_line_style(),
            'flip': curve.flip.isChecked(),
            'x_min': curve.x_min.text(),
            'x_max': curve.x_max.text(),
            'scale': curve.scale_combobox.currentText()
        }
    def apply_template(self, template_data):
        while self.track_tabs.count() > 0:
            self.delete_track(0)
        for track_number, track_settings in zip(template_data['tracks'], template_data['track_settings']):
            curves = sorted(set(curve for well in self.wells.values() for curve in well['data'].columns))
            track = TrackControl(track_number, curves)
            track.bg_color = track_settings['bg_color']
            track.bg_color_btn.setStyleSheet(f"background-color: {track.bg_color}; border: none;")
            track.grid.setChecked(track_settings['grid'])
            track.flip_y.setChecked(track_settings['flip_y'])
            track.y_min.setText(track_settings['y_min'])
            track.y_max.setText(track_settings['y_max'])
            track.changed.connect(self.update_plot)
            self.tracks.append(track)
            self.track_tabs.addTab(track, f"Track {track.number}")
            while track.curve_tabs.count() > 0:
                track.remove_curve(0)
            for curve_settings in track_settings['curves']:
                curve = CurveControl(len(track.curves) + 1, curves)
                curve.width.setValue(curve_settings['width'])
                curve.color = curve_settings['color']
                curve.color_btn.setStyleSheet(f"background-color: {curve.color}; border: none;")
                curve.line_style_box.setCurrentText(curve_settings['line_style'])
                curve.flip.setChecked(curve_settings['flip'])
                curve.x_min.setText(curve_settings['x_min'])
                curve.x_max.setText(curve_settings['x_max'])
                curve.scale_combobox.setCurrentText(curve_settings['scale'])
                curve.changed.connect(track.changed.emit)
                track.curves.append(curve)
                track.curve_tabs.addTab(curve, f"Curve {track.curve_count+1}")
                track.update_curve_numbers()
        self.update_plot()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(loadStyleSheet("style/darkmode.qss"))
    viewer = WellLogViewer()
    viewer.show()
    sys.exit(app.exec_())
