import sys
import os
import pickle
from PyQt5.QtGui import QIcon
import lasio
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QMenu, QVBoxLayout, QFormLayout, QLabel, QLineEdit,
    QDialogButtonBox, QMainWindow, QDockWidget, QListWidget,
    QListWidgetItem, QWidget, QComboBox, QPushButton, QCheckBox, QSpinBox,
    QScrollArea, QSizePolicy , QAction, QColorDialog, QTabWidget, QFrame, QApplication, QToolBar
)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
from collections import defaultdict
from PyQt5.QtWidgets import QProgressDialog

# Update the overall font size on plots
plt.rcParams.update({'font.size': 8.5})

def loadStyleSheet(fileName):
    """
    Loads a stylesheet from a file.

    Parameters:
        fileName (str): The path to the stylesheet file.

    Returns:
        str: The content of the stylesheet file.
    """
    try:
        with open(fileName, "r") as f:
            return f.read()
    except Exception as e:
        print("Failed to load stylesheet:", e)
        return ""

# --- Custom QListWidget: Clicking on an item's label toggles its check state ---
class ClickableListWidget(QListWidget):
    """
    Custom QListWidget that allows toggling the check state of an item by clicking on its label.
    """
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
    """
    Widget that contains a Matplotlib figure for plotting well log data.
    """
    mouse_moved = pyqtSignal(float, float)
    external_crosshair = pyqtSignal(float, float)
    zoomChanged = pyqtSignal(object)  # Signal emitted after a zoom event

    def __init__(self, well_name, parent=None):
        super().__init__(parent)
        self.well_name = well_name
        self.figure = Figure()  # Use tight layout

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # Allow horizontal expansion

        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Set contents margins to zero
        layout.setSpacing(0)  # Set spacing to zero
        layout.addWidget(self.canvas)
        self.setLayout(layout)

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
        """
        Sets the zoom mode for the figure.

        Parameters:
            mode (str): The zoom mode to set.
        """
        self.zoom_mode = mode
        if mode == "Rectangular":
            self.canvas.setCursor(Qt.CrossCursor)
        else:
            self.canvas.setCursor(Qt.ArrowCursor)

    def onMousePress(self, event):
        """
        Handles mouse press events for zooming.
        """
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
        """
        Handles mouse move events for zooming.
        """
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
        """
        Handles mouse release events for zooming.
        """
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
        """
        Applies zoom limits to all axes in this figure.

        Parameters:
            xmin (float): Minimum x-axis limit.
            xmax (float): Maximum x-axis limit.
            ymin (float): Minimum y-axis limit.
            ymax (float): Maximum y-axis limit.
        """
        for ax in self.figure.axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymax, ymin)
        self.current_zoom_limits = (xmin, xmax, ymin, ymax)
        self.canvas.draw()

    def undoZoom(self):
        """
        Undo the last zoom operation.
        """
        try:
            if self._zoom_history:
                prev_limits = self._zoom_history.pop()
                self.applyZoom(*prev_limits)
                return True
            elif self._initial_limits:
                # If no zoom history, revert to initial limits
                self.resetZoom()
            return False
        except Exception as e:
            print(f"Error while undoing zoom: {e}")
            return False

    def resetZoom(self):
        """
        Resets zoom to the initial state.
        """
        if self._initial_limits:
            for ax, limits in zip(self.figure.axes, self._initial_limits):
                ax.set_xlim(limits[0])
                ax.set_ylim(limits[1])
            self.current_zoom_limits = None
            self._zoom_history = []
            self.canvas.draw()

    def recordCurrentZoom(self):
        """
        Records the current zoom state for undo functionality.
        """
        if self.current_zoom_limits:
            self._zoom_history.append(self.current_zoom_limits)

    def on_mouse_move(self, event):
        """
        Handles mouse move events for crosshair.
        """
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.mouse_moved.emit(x, y)
            self.update_crosshair(event.inaxes, x, y)

    def on_external_crosshair(self, x, y):
        """
        Handles external crosshair signals.
        """
        # Find the first axes to use for positioning
        axes = self.figure.get_axes()
        if axes:
            self.update_crosshair(axes[0], x, y, external=True)

    def update_crosshair(self, ax, x, y, external=False):
        """
        Updates the crosshair position.

        Parameters:
            ax (Axes): The axes to update the crosshair on.
            x (float): The x-coordinate of the crosshair.
            y (float): The y-coordinate of the crosshair.
            external (bool): Whether the crosshair update is from an external signal.
        """
        self.remove_crosshair()

        # Add horizontal crosshair lines to all subplots in the current figure
        for sub_ax in self.figure.get_axes():
            hline = sub_ax.axhline(y, color='red', linestyle='--', linewidth=1)
            self.crosshair_hlines.append(hline)

        # Add vertical crosshair line only to the current axis if not an external signal
        if ax and not external:
            self.crosshair_vline = ax.axvline(x, color='red', linestyle='--', linewidth=1)

        if not external:
            self.cursor_coords = ax.text(x, y, f'x={x:.2f}, y={y:.2f}',
                                         transform=ax.transData, fontsize=9,
                                         verticalalignment='bottom', horizontalalignment='left',
                                         bbox=dict(boxstyle='round,pad=0.1', facecolor='yellow', alpha=0.5))
        self.canvas.draw()

    def remove_crosshair(self):
        """
        Removes the crosshair from the figure.
        """
        if self.crosshair_vline:
            self.crosshair_vline.remove()
            self.crosshair_vline = None
        for hline in self.crosshair_hlines:
            hline.remove()
        self.crosshair_hlines = []
        if self.cursor_coords:
            self.cursor_coords.remove()
            self.cursor_coords = None
        self.canvas.draw()

    def update_plot(self, data, tracks, well_top_lines=None):
        """
        Updates the plot with the given data and tracks.

        Parameters:
            data (DataFrame): The well log data.
            tracks (list): The list of tracks to plot.
            well_top_lines (list): The list of well top lines to plot.
        """
        self.figure.clear()

        self.data = data
        self.tracks = tracks
        n_tracks = len(tracks)
        if n_tracks == 0:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No tracks", ha='center', va='center')
        else:
            # Modify subplots creation to remove gaps
            if n_tracks > 1:
                axes = self.figure.subplots(1, n_tracks, sharey=True, gridspec_kw={'wspace': 0})
            else:
                axes = [self.figure.add_subplot(111)]

            depth = data['DEPT']

            for idx, (ax, track) in enumerate(zip(axes, tracks)):
                ax.set_facecolor(track.bg_color)  # Apply Background Color
                if idx != 0:
                    ax.tick_params(left=False, labelleft=False)
                track.ax = ax  # Store the axis for later reference
                valid_curves = []
                lines_list = []
                if not track.curves:
                    ax.text(0.5, 0.5, "No curves", ha='center', va='center')
                    continue

                for i, curve in enumerate(track.curves):
                    curve_name = curve.curve_box.currentText()
                    if curve_name == "Select Curve" or curve_name not in data.columns:
                        continue

                    # Create a new axis for each curve to manage individual x-axis limits
                    twin_ax = ax.twiny()
                    twin_ax.xaxis.set_ticks_position('top')
                    twin_ax.xaxis.set_label_position('top')
                    twin_ax.spines['top'].set_color(curve.color)
                    twin_ax.spines['top'].set_linewidth(2)
                    twin_ax.spines['top'].set_position(('axes', 1 + i * 0.025))  # Adjust the gap here
                    twin_ax.tick_params(axis='x', colors=curve.color)
                    twin_ax.set_xlabel(curve_name, color=curve.color)

                    line, = twin_ax.plot(
                        data[curve_name], depth,
                        color=curve.color,
                        linewidth=curve.width.value(),
                        linestyle=curve.get_line_style(),
                        label=curve_name,  # Add curve name as label for legend
                        picker=True  # Enable picking on the line
                    )
                    line.set_gid(curve_name)  # Set an ID for the line
                    valid_curves.append(curve)
                    lines_list.append(line)

                    if curve.flip.isChecked():
                        twin_ax.invert_xaxis()

                    # Apply individual x-axis limits for each curve
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

                    # Apply individual scale setting for each curve
                    if curve.scale_combobox.currentText() == "Log":
                        twin_ax.set_xscale('log')
                    else:
                        twin_ax.set_xscale('linear')

                if idx == 0:
                    ax.set_ylabel("Depth")
                ax.grid(track.grid.isChecked())

                ax.set_ylim(depth.max(), depth.min())
                if track.flip_y.isChecked():  # Flip Y-axis if checked
                    ax.invert_yaxis()

                # Apply Y min/max if values are provided
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

                # Remove x-axis labels for the primary axis
                ax.set_xticks([])
                ax.set_xticklabels([])

            # Add a title to the figure using the well name in a box
            self.figure.suptitle(f"Well: {self.well_name}", fontsize=11, alpha=0.6)

            # --- Plot Well Tops ---
            # well_top_lines is expected to be a list of tuples (top, md) for this well.
            if well_top_lines:
                for track in tracks:
                    for (top, md) in well_top_lines:
                        track.ax.axhline(y=md, color='red', linestyle='--', linewidth=1)
                        track.ax.text(
                            0.0005, md, f"{top}",  # Adjust x-coordinate to 0.02 for left alignment
                            transform=track.ax.get_yaxis_transform(),
                            color='red', fontsize=8, horizontalalignment='left', verticalalignment='bottom'
                        )

        self.figure.tight_layout()  # Apply tight layout
        self.canvas.draw()

        # Store initial limits
        self._initial_limits = []
        for ax in self.figure.axes:
            self._initial_limits.append((ax.get_xlim(), ax.get_ylim()))

        # If we have a current zoom state, reapply it
        if self.current_zoom_limits:
            self.applyZoom(*self.current_zoom_limits)

class CurveControl(QWidget):
    """
    Widget for controlling the properties of a curve.
    """
    changed = pyqtSignal()

    def __init__(self, curve_number, curves, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)

        # **Apply StyleSheet to the entire TrackControl Widget**
        self.setStyleSheet("""
            border-radius: 5px;
            color: Black;
            font: 10pt;
            padding: 2px;
            height: 30px;
        """)

        # **Curve Number Label**
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

        # Initial curve color set to black.
        self.color = "#000000"
        self.color_btn = QPushButton("Color")
        self.color_btn.setStyleSheet(f"background-color: {self.color}; border: none;")
        self.color_btn.clicked.connect(self.select_color)
        layout.addWidget(self.color_btn)

        # **Line Style Selection**
        self.line_style_box = QComboBox()
        self.line_style_box.addItems(["Solid", "Dashed", "Dotted", "Dash-dot"])
        self.line_style_box.currentIndexChanged.connect(self.changed.emit)
        layout.addWidget(QLabel("Style:"))
        layout.addWidget(self.line_style_box)

        self.flip = QCheckBox("X-Flip")
        self.flip.stateChanged.connect(self.changed.emit)
        layout.addWidget(self.flip)

        # X min and X max input fields
        xy_range_layout = QHBoxLayout()
        xy_range_layout.addWidget(QLabel("X-min:"))
        self.x_min = QLineEdit()
        self.x_min.setStyleSheet("background-color: White; color: blue; font: 12pt;")
        self.x_min.setFixedWidth(50)
        self.x_min.setPlaceholderText("Auto")
        self.x_min.textChanged.connect(self.changed.emit)  # Connect to changed signal
        xy_range_layout.addWidget(self.x_min)

        xy_range_layout.addWidget(QLabel("X-max:"))
        self.x_max = QLineEdit()
        self.x_max.setStyleSheet("background-color: White; color: blue; font: 12pt;")
        self.x_max.setFixedWidth(50)
        self.x_max.setPlaceholderText("Auto")
        self.x_max.textChanged.connect(self.changed.emit)  # Connect to changed signal
        xy_range_layout.addWidget(self.x_max)

        # **Scale Selection**
        self.scale_combobox = QComboBox()
        self.scale_combobox.addItems(["Linear", "Log"])
        self.scale_combobox.currentIndexChanged.connect(self.changed.emit)
        xy_range_layout.addWidget(self.scale_combobox)

        layout.addLayout(xy_range_layout)

    def select_color(self):
        """
        Opens a color picker to change the curve color.
        """
        color = QColorDialog.getColor()
        if color.isValid():
            self.color = color.name()
            # Update both the color button and the curve label to match the chosen color.
            self.color_btn.setStyleSheet(f"background-color: {self.color}; border: none;")
            self.curve_label.setStyleSheet(f"color: {self.color};")
            self.changed.emit()

    def get_line_style(self):
        """
        Returns the Matplotlib line style based on selection.

        Returns:
            str: The Matplotlib line style.
        """
        styles = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-dot": "-."}
        return styles[self.line_style_box.currentText()]

class TrackControl(QWidget):
    """
    Widget for controlling the properties of a track.
    """
    changed = pyqtSignal()

    def __init__(self, number, curves, parent=None):
        super().__init__(parent)
        self.number = number
        self.curves = []
        self.bg_color = "#FFFFFF"  # Default background color (white)
        self.curve_count = 0  # Track number of added curves
        self.setContextMenuPolicy(Qt.CustomContextMenu)

        # **Apply StyleSheet to the entire TrackControl Widget**
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
        self.grid.setFixedWidth(100)  # Set fixed width
        self.grid.stateChanged.connect(self.changed.emit)
        range_layout.addWidget(self.grid)

        self.flip_y = QCheckBox("Flip Y-Axis")  # New checkbox for flipping Y-axis
        self.flip_y.stateChanged.connect(self.changed.emit)
        self.flip_y.setFixedWidth(100)  # Set fixed width
        range_layout.addWidget(self.flip_y)

        # Background Color Selection Button
        self.bg_color_btn = QPushButton("Bg Color")
        self.bg_color_btn.setStyleSheet(f"background-color: {self.bg_color}; border: none;")
        self.bg_color_btn.setFixedWidth(100)  # Set fixed width
        self.bg_color_btn.clicked.connect(self.select_bg_color)
        range_layout.addWidget(self.bg_color_btn)

        # Y min and Y max input fields with fixed width labels
        y_min_label = QLabel("Y min:")
        y_min_label.setFixedWidth(50)  # Set fixed width for the label
        range_layout.addWidget(y_min_label)

        self.y_min = QLineEdit()
        self.y_min.setStyleSheet("background-color: White; color: blue; font: 12pt;")
        self.y_min.setPlaceholderText("Auto")
        self.y_min.setFixedWidth(60)  # Set fixed width for the input field
        self.y_min.textChanged.connect(self.changed.emit)  # Connect to changed signal
        range_layout.addWidget(self.y_min)

        y_max_label = QLabel("Y max:")
        y_max_label.setFixedWidth(50)  # Set fixed width for the label
        range_layout.addWidget(y_max_label)

        self.y_max = QLineEdit()
        self.y_max.setStyleSheet("background-color: White; color: blue; font: 12pt;")
        self.y_max.setPlaceholderText("Auto")
        self.y_max.setFixedWidth(60)  # Set fixed width for the input field
        self.y_max.textChanged.connect(self.changed.emit)  # Connect to changed signal
        range_layout.addWidget(self.y_max)

        layout.addLayout(range_layout)

        # Curve Tabs
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

        # Center the button within its layout
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        btn_layout.addWidget(add_curve_btn)
        btn_layout.addStretch(1)
        layout.addLayout(btn_layout)

        # Set a fixed height for the TrackControl widget
        self.setFixedHeight(300)

        self.add_curve(curves)  # Start with one curve

    def select_bg_color(self):
        """
        Opens a color picker to change the background color and update the button.
        """
        color = QColorDialog.getColor()
        if color.isValid():
            self.bg_color = color.name()
            self.bg_color_btn.setStyleSheet(f"background-color: {self.bg_color}; border: none;")
            self.changed.emit()

    def add_curve(self, curves):
        """
        Adds a new curve to the track.

        Parameters:
            curves (list): The list of available curves.
        """
        self.curve_count += 1  # Increment curve number
        curve = CurveControl(self.curve_count, curves)  # Pass curve_number
        curve.changed.connect(self.changed.emit)
        self.curves.append(curve)
        self.curve_tabs.addTab(curve, f"Curve {self.curve_count}")
        self.update_curve_numbers()
        self.changed.emit()

    def remove_curve(self, index):
        """
        Removes a curve from the track.

        Parameters:
            index (int): The index of the curve to remove.
        """
        curve = self.curve_tabs.widget(index)
        if curve:
            self.curves.remove(curve)
            self.curve_tabs.removeTab(index)
            curve.deleteLater()
            self.update_curve_numbers()  # Renumber remaining curves
            self.changed.emit()

    def update_curve_numbers(self):
        """
        Renumbers curves after a deletion or addition.
        """
        for i, curve in enumerate(self.curves, start=1):
            curve.curve_label.setText(f"Curve {i}:")
            self.curve_tabs.setTabText(i - 1, f"Curve {i}")

class WellLogViewer(QMainWindow):
    """
    Main window for the Well Log Viewer application.
    """
    def __init__(self):
        super().__init__()
        self.wells = {}
        self.well_tops = {}
        self.selected_top_names = set()
        self.tracks = []
        self.figure_widgets = {}
        self.show_well_tops = True  # New attribute to track well top visibility
        self.sync_zoom_enabled = True
        self.current_single_zoom_well = None  # Track which well has single zoom
        self.single_zoom_limits = None  # Store single zoom limits
        self.sync_zoom_limits = None  # Store sync zoom limits
        self.share_y_axis_enabled = True  # New attribute to track shared Y-axis state
        self.link_well_tops_enabled = False  # New attribute for well top connections
        self.initUI()
        self.setWindowIcon(QIcon('images/ONGC_Logo.png'))
        # Enable single zoom by default
        self.enableSingleZoom()

    def initUI(self):
        """
        Initializes the user interface.
        """
        self.setWindowTitle('Well Log Viewer')
        self.setGeometry(100, 100, 1200, 800)

        self.figure_scroll = QScrollArea()
        self.figure_container = QWidget()
        self.figure_layout = QHBoxLayout(self.figure_container)
        self.figure_layout.setContentsMargins(0, 0, 0, 0)  # Set contents margins to zero
        self.figure_layout.setSpacing(0)  # Set spacing to zero

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

        # New action: Toggle Well Tops Visibility
        self.toggle_well_tops_action = QAction("Hide Well Tops", self)
        self.toggle_well_tops_action.triggered.connect(self.toggle_well_tops)
        menubar.addAction(self.toggle_well_tops_action)

        # New action: Toggle Well Top Connections
        self.link_well_tops_action = QAction("Link Well Tops", self, checkable=True)
        self.link_well_tops_action.toggled.connect(self.toggle_link_well_tops)
        menubar.addAction(self.link_well_tops_action)

        # New actions for zoom functionality
        self.sync_zoom_action = QAction("Disable Sync Zoom", self, checkable=True)
        self.sync_zoom_action.toggled.connect(self.onSyncZoomToggled)
        menubar.addAction(self.sync_zoom_action)

        self.undo_zoom_action = QAction("Undo Zoom", self)
        self.undo_zoom_action.triggered.connect(self.undoZoom)
        menubar.addAction(self.undo_zoom_action)

        self.reset_zoom_action = QAction("Reset Zoom", self)
        self.reset_zoom_action.triggered.connect(self.resetZoom)
        menubar.addAction(self.reset_zoom_action)

        # New action for sharing Y-axis limits
        self.share_y_axis_action = QAction("Disable Link Y Axis", self, checkable=True)
        self.share_y_axis_action.toggled.connect(self.onShareYAxisToggled)
        menubar.addAction(self.share_y_axis_action)

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
        """
        Toggle the well top connections feature.

        Parameters:
            checked (bool): Whether the feature is enabled.
        """
        self.link_well_tops_enabled = checked
        self.link_well_tops_action.setText("Unlink Well Tops" if checked else "Link Well Tops")
        self.update_plot()

    def onShareYAxisToggled(self, checked):
        """
        Handle share Y-axis toggle.

        Parameters:
            checked (bool): Whether the feature is enabled.
        """
        self.share_y_axis_enabled = checked
        if checked:
            self.share_y_axis_action.setText("Disable Link Y Axis")
            self.synchronizeYAxisLimits()
        else:
            self.share_y_axis_action.setText("Enable Link Y Axis")
            self.update_plot()

    def synchronizeYAxisLimits(self):
        """
        Synchronize Y-axis limits across all wells.
        """
        if not self.figure_widgets:
            return

        # Get the Y-axis limits from the selected wells
        selected_wells = [
            self.well_list.item(i).text()
            for i in range(self.well_list.count())
            if self.well_list.item(i).checkState() == Qt.Checked
        ]

        y_min = float('inf')
        y_max = float('-inf')

        for well in selected_wells:
            data = self.wells[well]['data']
            depth = data['DEPT']
            y_min = min(y_min, depth.min())
            y_max = max(y_max, depth.max())

        # Apply the Y-axis limits to all wells
        for idx, widget in enumerate(self.figure_widgets.values()):
            for ax in widget.figure.axes:
                ax.set_ylim(y_max, y_min)  # Inverted to suit depth plots if needed
                if idx != 0:  # Hide ticks and tick labels for all but the first well
                    ax.tick_params(left=False, labelleft=False)
                    ax.set_ylabel(None)
            widget.canvas.draw()

    def onSyncZoomToggled(self, checked):
        """
        Handle sync zoom toggle.

        Parameters:
            checked (bool): Whether the feature is enabled.
        """
        self.sync_zoom_enabled = checked
        if checked:
            self.disableSingleZoom()
            self.enableSyncZoom()
            # If a single zoom was already applied, use its limits for all wells.
            if self.single_zoom_limits:
                self.sync_zoom_limits = self.single_zoom_limits
                for widget in self.figure_widgets.values():
                    widget.applyZoom(*self.single_zoom_limits)
                    widget.recordCurrentZoom()
            self.sync_zoom_action.setText("Disable Sync Zoom")

        else:
            self.disableSyncZoom()
            self.sync_zoom_action.setText("Sync Zoom")
            self.enableSingleZoom()

    def enableSyncZoom(self):
        """
        Enable sync zoom mode.
        """
        for widget in self.figure_widgets.values():
            widget.setZoomMode("Rectangular")
            widget.zoomChanged.connect(self.handleSyncZoom)

    def disableSyncZoom(self):
        """
        Disable sync zoom mode.
        """
        for widget in self.figure_widgets.values():
            widget.setZoomMode("Rectangular")
            try:
                widget.zoomChanged.disconnect(self.handleSyncZoom)
            except TypeError:
                pass
            widget.zoomChanged.connect(self.handleSingleZoom)

    def enableSingleZoom(self):
        """
        Enable single zoom mode (default).
        """
        for widget in self.figure_widgets.values():
            widget.setZoomMode("Rectangular")
            widget.zoomChanged.connect(self.handleSingleZoom)

    def disableSingleZoom(self):
        """
        Disable single zoom mode.
        """
        for widget in self.figure_widgets.values():
            try:
                widget.zoomChanged.disconnect(self.handleSingleZoom)
            except TypeError:
                pass

    def handleSyncZoom(self, sender):
        """
        Handle sync zoom event - apply to all wells.

        Parameters:
            sender (FigureWidget): The widget that triggered the zoom event.
        """
        if not sender.current_zoom_limits:
            return

        self.sync_zoom_limits = sender.current_zoom_limits

        # Apply to all wells
        for widget in self.figure_widgets.values():
            widget.applyZoom(*sender.current_zoom_limits)
            widget.recordCurrentZoom()

    def handleSingleZoom(self, sender):
        """
        Handle single zoom event - apply to the selected well only.

        Parameters:
            sender (FigureWidget): The widget that triggered the zoom event.
        """
        if not sender.current_zoom_limits:
            return

        # Store which well has the single zoom
        self.current_single_zoom_well = sender.well_name
        self.single_zoom_limits = sender.current_zoom_limits

        # Apply to the selected well only
        for widget in self.figure_widgets.values():
            if widget.well_name == sender.well_name:
                widget.applyZoom(*sender.current_zoom_limits)
                widget.recordCurrentZoom()

    def undoZoom(self):
        """
        Undo the last zoom operation.
        """
        if self.sync_zoom_enabled:
            for widget in self.figure_widgets.values():
                widget.undoZoom()
        else:
            if self.current_single_zoom_well:
                widget = self.figure_widgets[self.current_single_zoom_well]
                widget.undoZoom()

    def resetZoom(self):
        """
        Reset zoom for all wells.
        """
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
        """
        Toggle the visibility of well tops.
        """
        self.show_well_tops = not self.show_well_tops
        self.toggle_well_tops_action.setText("Hide Well Tops" if self.show_well_tops else "Show Well Tops")
        self.update_plot()

    def update_plot(self):
        """
        Main update method that handles well selection/deselection.
        """
        selected_wells = [self.well_list.item(i).text() for i in range(self.well_list.count())
                        if self.well_list.item(i).checkState() == Qt.Checked]

        # First remove all connection subplots
        self.remove_connection_subplots()

        # Update main well plots
        #self.update_well_plots(selected_wells)

        # Draw connections if enabled
        if self.link_well_tops_enabled and len(selected_wells) > 1:
            self.draw_well_top_connections()

        # First remove all connection subplots (anything that's not a main well widget)
        for i in reversed(range(self.figure_layout.count())):
            widget = self.figure_layout.itemAt(i).widget()
            if widget and widget not in self.figure_widgets.values():
                widget.setParent(None)
                widget.deleteLater()

        # Disconnect existing signals to prevent multiple connections
        for widget in self.figure_widgets.values():
            try:
                widget.mouse_moved.disconnect()
            except TypeError:
                pass

        # Connect signals for crosshair synchronization and create widgets if needed.
        for well in selected_wells:
            if well not in self.figure_widgets:
                self.figure_widgets[well] = FigureWidget(well)
                self.figure_layout.addWidget(self.figure_widgets[well])
                # Set the appropriate zoom mode based on current mode.
                if self.sync_zoom_enabled:
                    self.figure_widgets[well].setZoomMode("Rectangular")
                    self.figure_widgets[well].zoomChanged.connect(self.handleSyncZoom)
                else:
                    self.figure_widgets[well].setZoomMode("Rectangular")
                    self.figure_widgets[well].zoomChanged.connect(self.handleSingleZoom)

            # Connect the mouse_moved signal to all other widgets
            for other_well, other_widget in self.figure_widgets.items():
                if other_well != well:
                    self.figure_widgets[well].mouse_moved.connect(other_widget.external_crosshair)

            well_top_lines = []
            if well in self.well_tops and self.show_well_tops:
                for top, md in self.well_tops[well]:
                    if top in self.selected_top_names:
                        well_top_lines.append((top, md))

            # Apply any existing zoom states
            widget = self.figure_widgets[well]
            widget.update_plot(self.wells[well]['data'], self.tracks, well_top_lines)

            # Reapply zoom states if they exist
            if self.sync_zoom_enabled and self.sync_zoom_limits:
                widget.applyZoom(*self.sync_zoom_limits)
            elif self.current_single_zoom_well == well and self.single_zoom_limits:
                widget.applyZoom(*self.single_zoom_limits)

        # Remove widgets for unselected wells
        for well in list(self.figure_widgets.keys()):
            if well not in selected_wells:
                widget = self.figure_widgets[well]
                self.figure_layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
                del self.figure_widgets[well]

        # Synchronize Y-axis limits if enabled
        if self.share_y_axis_enabled:
            self.synchronizeYAxisLimits()

        # Draw well top connections if enabled
        if self.link_well_tops_enabled and len(selected_wells) > 1:
            self.draw_well_top_connections()

    def remove_connection_subplots(self):
        """
        Remove all connection subplots while preserving well plots.
        """
        for i in reversed(range(self.figure_layout.count())):
            widget = self.figure_layout.itemAt(i).widget()
            if widget and not hasattr(widget, 'well_name'):
                widget.setParent(None)
                widget.deleteLater()

    def draw_well_top_connections(self):
        """
        Draw perfectly aligned connections between well tops.
        """
        selected_wells = self.get_ordered_visible_wells()
        if len(selected_wells) < 2 or not self.selected_top_names:
            return

        # Get reference dimensions from first well plot
        ref_well = self.figure_widgets[selected_wells[0]]
        ref_fig = ref_well.figure
        plt_bbox = ref_fig.axes[-1].get_position()  # Get position of last track's axes

        for i in range(len(selected_wells)-1):
            well1, well2 = selected_wells[i], selected_wells[i+1]
            widget1 = self.figure_widgets[well1]
            widget2 = self.figure_widgets[well2]

            # Create connection figure with identical layout parameters
            conn_fig = Figure()  # Use tight layout
            conn_fig.set_tight_layout(True)
            canvas = FigureCanvas(conn_fig)
            canvas.setFixedSize(60, widget1.height())

            # Create connection axes with matching spine positions
            ax = conn_fig.add_subplot(111)
            ax.set_position(plt_bbox)  # Match main plot's axes position

            # Configure spines to match well plot styling
            ax.spines['top'].set_visible(True)
            ax.spines['top'].set_position(('axes', 1.0))  # Align with well plot's top spine
            ax.spines['top'].set_color('#404040')
            ax.spines['top'].set_linewidth(0.8)
            for spine in ['left', 'right', 'bottom']:
                ax.spines[spine].set_visible(False)

            # Match axis limits with well plots
            ymin = max(widget1.figure.axes[0].get_ylim()[0],
                    widget2.figure.axes[0].get_ylim()[0])
            ymax = min(widget1.figure.axes[0].get_ylim()[1],
                    widget2.figure.axes[0].get_ylim()[1])
            ax.set_ylim(ymin, ymax)
            ax.set_axis_off()

            # Get common well tops between adjacent wells
            tops1 = {t:md for t,md in self.well_tops.get(well1, [])
                    if t in self.selected_top_names}
            tops2 = {t:md for t,md in self.well_tops.get(well2, [])
                    if t in self.selected_top_names}
            common_tops = set(tops1.keys()) & set(tops2.keys())

            # Draw connection lines with precise positioning
            for top in common_tops:
                y1 = tops1[top]
                y2 = tops2[top]

                # Draw line using figure coordinates for perfect alignment
                ax.plot([0.05, 0.95],  # Use 5% padding on both sides
                    [y1, y2],
                    color='#2b70b7',
                    linewidth=1.2,
                    alpha=0.8,
                    solid_capstyle='round')

                # Add label at midpoint
                ax.text(0.5, (y1+y2)/2, top,
                    fontsize=8,
                    ha='center',
                    va='center',
                    rotation=90,
                    bbox=dict(facecolor='white', alpha=0.85,
                                edgecolor='none', pad=0))

            # Create container widget and insert in layout
            conn_widget = QWidget()
            conn_widget.setFixedHeight(widget1.height())
            layout = QVBoxLayout(conn_widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(canvas)

            insert_position = self.figure_layout.indexOf(widget1) + 1
            self.figure_layout.insertWidget(insert_position, conn_widget)
            canvas.draw()

    def get_top_axis_position(self, well_widget):
        """
        Get the top axis position from a well widget.

        Parameters:
            well_widget (FigureWidget): The well widget to get the top axis position from.

        Returns:
            float: The top axis position.
        """
        for ax in well_widget.figure.axes:
            if ax.xaxis.get_ticks_position() == 'top':
                return ax.get_position().y1
        return 2#0.95  # Default if not found

    def draw_connection_lines(self, ax, top_dict, well1, well2):
        """
        Draw connection lines between matching well tops.

        Parameters:
            ax (Axes): The axes to draw the connection lines on.
            top_dict (dict): The dictionary containing well top information.
            well1 (str): The first well name.
            well2 (str): The second well name.
        """
        for top_name in self.selected_top_names:
            if well1 in top_dict[top_name] and well2 in top_dict[top_name]:
                md1 = top_dict[top_name][well1]
                md2 = top_dict[top_name][well2]

                ax.plot([0.1, 0.9], [md1, md2],
                    color='blue', linestyle='-', linewidth=1.5, alpha=0.8)

                ax.text(0.5, (md1 + md2)/2, top_name,
                    ha='center', va='center',
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    def get_ordered_visible_wells(self):
        """
        Get wells in their current display order.

        Returns:
            list: The list of wells in their current display order.
        """
        wells = []
        for i in range(self.figure_layout.count()):
            widget = self.figure_layout.itemAt(i).widget()
            if widget and hasattr(widget, 'well_name'):
                wells.append(widget.well_name)
        return wells

    def get_widget_position(self, well_name):
        """
        Get layout position of a well widget.

        Parameters:
            well_name (str): The name of the well.

        Returns:
            int: The layout position of the well widget.
        """
        for i in range(self.figure_layout.count()):
            widget = self.figure_layout.itemAt(i).widget()
            if widget and hasattr(widget, 'well_name') and widget.well_name == well_name:
                return i
        return None

    def change_background_color(self):
        """
        Opens a color picker to change the background color.
        """
        color = QColorDialog.getColor()
        if color.isValid():
            self.setStyleSheet(f"QWidget {{ background-color: {color.name()}; }}")

    def toggle_controls(self):
        """
        Toggles the visibility of the controls dock.
        """
        self.dock.setVisible(not self.dock.isVisible())

    def load_las_files(self):
        """
        Loads LAS files and updates the well list.
        """
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Select LAS Files", "", "LAS Files (*.las);;All Files (*)", options=options)
        if files:
            progress = QProgressDialog("Loading LAS Files...", "Cancel", 0, len(files), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setWindowTitle("Loading")
            progress.setAutoClose(True)

            for i, file in enumerate(files):
                progress.setValue(i)
                if progress.wasCanceled():
                    break
                self.load_las_file(file)

            progress.setValue(len(files))
            self.update_plot()

    def load_las_file(self, path):
        """
        Loads a LAS file and adds it to the well list.

        Parameters:
            path (str): The path to the LAS file.
        """
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
        """
        Loads well tops from a file and updates the well tops list.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Well Tops File", "", "Text Files (*.txt *.csv)"
        )
        if not file_path:
            return

        progress = QProgressDialog("Loading Well Tops...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Loading")
        progress.setAutoClose(True)

        try:
            # Determine the delimiter based on file content
            delimiter = ','
            if file_path.endswith('.txt'):
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    has_comma = any(',' in line for line in lines)

                    if has_comma:
                        df = pd.read_csv(
                            file_path,
                            delimiter=',',
                            header=None,
                            engine='python',
                            on_bad_lines='skip')
                    else:
                        df = pd.read_csv(
                            file_path,
                            delim_whitespace=True,
                            header=None,
                            engine='python',
                            on_bad_lines='skip')

            # Determine common number of columns.
            num_cols = df.apply(lambda row: row.count(), axis=1).mode()[0]
            df = df.iloc[:, :num_cols]

            # Check for header by trying to convert third column to float.
            header_present = False
            try:
                float(df.iloc[0, 2])
            except ValueError:
                header_present = True

            if header_present:
                df = df.drop(0).reset_index(drop=True)

            # Trim to the first three columns if necessary
            if df.shape[1] > 3:
                df = df.iloc[:, :3]

            df.columns = ["well", "top", "md"]

            # Process well tops
            progress.setMaximum(len(df))
            for idx, row in df.iterrows():
                progress.setValue(idx)
                if progress.wasCanceled():
                    break
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
        finally:
            progress.setValue(len(df))

    def update_well_tops_list(self):
        """
        Updates the well tops list with unique top names.
        """
        self.well_tops_list.clear()
        unique_tops = set()
        # Aggregate unique top names from all wells.
        for tops in self.well_tops.values():
            for top, md in tops:
                unique_tops.add(top)
        # Create one list item per unique top name.
        for top in sorted(unique_tops):
            item = QListWidgetItem(top)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            # Store only the top name.
            item.setData(Qt.UserRole, top)
            self.well_tops_list.addItem(item)

    def well_top_item_changed(self, item):
        """
        Handles the change in well top item selection.

        Parameters:
            item (QListWidgetItem): The well top item that was changed.
        """
        top = item.data(Qt.UserRole)
        if not top:
            return
        if item.checkState() == Qt.Checked:
            self.selected_top_names.add(top)
        else:
            self.selected_top_names.discard(top)
        self.update_plot()

    def add_track(self):
        """
        Adds a new track to the plot.
        """
        if not self.wells:
            return

        curves = sorted(set(curve for well in self.wells.values() for curve in well['data'].columns))
        track = TrackControl(len(self.tracks) + 1, curves)
        track.changed.connect(self.update_plot)
        self.tracks.append(track)
        self.track_tabs.addTab(track, f"Track {track.number}")

        self.update_plot()

    def delete_track(self, index):
        """
        Deletes a track from the plot.

        Parameters:
            index (int): The index of the track to delete.
        """
        track = self.track_tabs.widget(index)
        if track:
            self.tracks.remove(track)
            self.track_tabs.removeTab(index)
            track.deleteLater()
            self.renumber_tracks()
            self.update_plot()

    def renumber_tracks(self):
        """
        Renumbers tracks and updates their tab titles.
        """
        for i, track in enumerate(self.tracks, start=1):
            track.number = i
            track.update_curve_numbers()  # Renumber curves within the track
            self.track_tabs.setTabText(i - 1, f"Track {i}")

    def save_template(self):
        """
        Saves the current template settings to a .pkl file.
        """
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Template", "", "Template Files (*.pkl)")
        if file_path:
            file_path = file_path + ".pkl"
            print(file_path)
            template_data = {
                'tracks': [track.number for track in self.tracks],
                'track_settings': [self.get_track_settings(track) for track in self.tracks]

            }
            with open(file_path, 'wb') as f:
                pickle.dump(template_data, f)

    def load_template(self):
        """
        Loads template settings from a .pkl file.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Template", "", "Template Files (*.pkl)")
        if file_path:
            with open(file_path, 'rb') as f:
                template_data = pickle.load(f)
            self.apply_template(template_data)

    def get_track_settings(self, track):
        """
        Gets the settings of a track.

        Parameters:
            track (TrackControl): The track to get the settings from.

        Returns:
            dict: The settings of the track.
        """
        return {
            'bg_color': track.bg_color,
            'grid': track.grid.isChecked(),
            'flip_y': track.flip_y.isChecked(),
            'y_min': track.y_min.text(),
            'y_max': track.y_max.text(),
            'curves': [self.get_curve_settings(curve) for curve in track.curves]
        }

    def get_curve_settings(self, curve):
        """
        Gets the settings of a curve.

        Parameters:
            curve (CurveControl): The curve to get the settings from.

        Returns:
            dict: The settings of the curve.
        """
        return {
            #'curve_name': curve.curve_box.currentText(),
            'width': curve.width.value(),
            'color': curve.color,
            'line_style': curve.get_line_style(),
            'flip': curve.flip.isChecked(),
            'x_min': curve.x_min.text(),
            'x_max': curve.x_max.text(),
            'scale': curve.scale_combobox.currentText()
        }

    def apply_template(self, template_data):
        """
        Applies the template settings.

        Parameters:
            template_data (dict): The template data to apply.
        """
        # Clear existing tracks
        while self.track_tabs.count() > 0:
            self.delete_track(0)

        # Load tracks
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
            while track.curve_tabs.count()>0:
                track.remove_curve(0)
            # Load curves
            for curve_settings in track_settings['curves']:
                    curve = CurveControl(len(track.curves) + 1, curves)
                    #curve.curve_box.setCurrentText(curve_settings['curve_name'])
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
