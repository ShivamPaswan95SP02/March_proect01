import sys
import os
from PyQt5.QtGui import QIcon
import lasio
import pickle
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QMenu, QVBoxLayout, QFormLayout, QLabel, QLineEdit,
    QDialogButtonBox, QMainWindow, QDockWidget, QListWidget,
    QListWidgetItem, QWidget, QComboBox, QPushButton, QCheckBox, QSpinBox,
    QScrollArea, QAction, QColorDialog, QTabWidget, QFrame, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd

# Update the overall font size on plots
plt.rcParams.update({'font.size': 8.5})

def loadStyleSheet(fileName):
    try:
        with open(fileName, "r") as f:
            return f.read()
    except Exception as e:
        print("Failed to load stylesheet:", e)
        return ""

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
    curve_clicked = pyqtSignal(str, object)
    track_clicked = pyqtSignal(object)  # Signal for track click

    def __init__(self, well_name, parent=None):
        super().__init__(parent)
        self.well_name = well_name
        self.figure = Figure(layout="constrained")  # Use constrained layout
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Connect mouse move event
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        """Handle click events to detect which curve or track was clicked."""
        if event.inaxes:
            if event.button == 3:  # Right-click
                for track in self.tracks:
                    if event.inaxes == track.ax:
                        self.track_clicked.emit(track)
                        return

    def update_plot(self, data, tracks, well_tops):
        self.figure.clear()
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
                ax.set_facecolor(track.bg_color)  # Apply Background Color
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
                    twin_ax.spines['top'].set_position(('axes', 1 + i * 0.08))  # Adjust the gap here
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
                ax.set_xticklabels([])

            # Add well tops to the plot
            for well_top in well_tops:
                ax.axhline(y=well_top['MD [m]'], color='red', linestyle='--', label=well_top['Top'])
                ax.text(ax.get_xlim()[1], well_top['MD [m]'], well_top['Top'], va='center', ha='left', color='red')

            # Add a title to the figure using the well name in a box
            self.figure.suptitle(f"Well: {self.well_name}", fontsize=11, alpha=0.6)

        self.canvas.draw()

class CurveControl(QWidget):
    changed = pyqtSignal()
    deleteRequested = pyqtSignal(object)

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
        self.x_max.setStyleSheet("background-color: White; color: blue; font: 10pt;")
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
        color = QColorDialog.getColor()
        if color.isValid():
            self.color = color.name()
            # Update both the color button and the curve label to match the chosen color.
            self.color_btn.setStyleSheet(f"background-color: {self.color}; border: none;")
            self.curve_label.setStyleSheet(f"color: {self.color};")
            self.changed.emit()

    def get_line_style(self):
        """Returns the Matplotlib line style based on selection."""
        styles = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-dot": "-."}
        return styles[self.line_style_box.currentText()]

class TrackControl(QWidget):
    changed = pyqtSignal()
    deleteRequested = pyqtSignal(object)

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
        add_curve_btn.setFixedSize(120, 30)
        add_curve_btn.setStyleSheet("background-color: White; border-radius: 5px; color: Green; font: 12pt;")
        add_curve_btn.clicked.connect(lambda: self.add_curve(curves))
        layout.addWidget(add_curve_btn)

        self.add_curve(curves)  # Start with one curve

    def select_bg_color(self):
        """Opens a color picker to change background color and update the button."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.bg_color = color.name()
            self.bg_color_btn.setStyleSheet(f"background-color: {self.bg_color}; border: none;")
            self.changed.emit()

    def add_curve(self, curves):
        self.curve_count += 1  # Increment curve number
        curve = CurveControl(self.curve_count, curves)  # Pass curve_number
        curve.changed.connect(self.changed.emit)
        curve.deleteRequested.connect(self.remove_curve)
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
            self.update_curve_numbers()  # Renumber remaining curves
            self.changed.emit()

    def update_curve_numbers(self):
        """Renumbers curves after a deletion or addition."""
        for i, curve in enumerate(self.curves, start=1):
            curve.curve_label.setText(f"Curve {i}:")
            self.curve_tabs.setTabText(i - 1, f"Curve {i}")

class EditTrackDialog(QDialog):
    """Dialog for editing track properties."""
    def __init__(self, track, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Track Properties")
        self.setLayout(QVBoxLayout())

        form_layout = QFormLayout()
        self.layout().addLayout(form_layout)

        # Initialize properties with current track settings
        self.grid = QCheckBox("Grid")
        self.grid.setChecked(track.grid.isChecked())
        form_layout.addRow(self.grid)

        self.flip_y = QCheckBox("Flip Y-Axis")
        self.flip_y.setChecked(track.flip_y.isChecked())
        form_layout.addRow(self.flip_y)

        self.bg_color_btn = QPushButton("Bg Color")
        self.bg_color_btn.setStyleSheet(f"background-color: {track.bg_color}; border: none;")
        self.bg_color_btn.clicked.connect(self.select_bg_color)
        form_layout.addRow(self.bg_color_btn)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout().addWidget(self.buttons)

        # Store the initial background color
        self.bg_color = track.bg_color

    def select_bg_color(self):
        """Opens a color picker to change background color."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.bg_color = color.name()

    def accept(self):
        """Store the state of each property when OK is clicked."""
        self.grid_state = self.grid.isChecked()
        self.flip_y_state = self.flip_y.isChecked()
        super().accept()

class EditCurveDialog(QDialog):
    """Dialog for editing curve properties."""
    def __init__(self, curve, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Curve Properties")
        self.setLayout(QVBoxLayout())

        form_layout = QFormLayout()
        self.layout().addLayout(form_layout)

        # Initialize properties with current curve settings
        self.grid = QCheckBox("Grid")
        self.grid.setChecked(curve.grid.isChecked())
        form_layout.addRow(self.grid)

        self.flip = QCheckBox("X-Flip")
        self.flip.setChecked(curve.flip.isChecked())
        form_layout.addRow(self.flip)

        self.color_btn = QPushButton("Color")
        self.color_btn.setStyleSheet(f"background-color: {curve.color}; border: none;")
        self.color_btn.clicked.connect(self.select_color)
        form_layout.addRow(self.color_btn)

        self.width = QSpinBox()
        self.width.setRange(1, 5)
        self.width.setValue(curve.width.value())
        form_layout.addRow("Width:", self.width)

        self.line_style_box = QComboBox()
        self.line_style_box.addItems(["Solid", "Dashed", "Dotted", "Dash-dot"])
        self.line_style_box.setCurrentText(curve.line_style_box.currentText())
        form_layout.addRow("Style:", self.line_style_box)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout().addWidget(self.buttons)

        # Store the initial color
        self.color = curve.color

    def select_color(self):
        """Opens a color picker to change curve color."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.color = color.name()

    def accept(self):
        """Store the state of each property when OK is clicked."""
        self.grid_state = self.grid.isChecked()
        self.flip_state = self.flip.isChecked()
        self.width_value = self.width.value()
        self.line_style = self.line_style_box.currentText()
        super().accept()

class WellLogViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.wells = {}
        self.tracks = []
        self.figure_widgets = {}
        self.well_tops = []
        self.initUI()
        self.setWindowIcon(QIcon('images/ONGC_Logo.png'))

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

        load_folder_action = QAction("Load LAS Folder", self)
        load_folder_action.triggered.connect(self.load_las_folder)
        file_menu.addAction(load_folder_action)

        load_files_action = QAction("Load LAS Files", self)
        load_files_action.triggered.connect(self.load_las_files)
        file_menu.addAction(load_files_action)

        load_well_tops_action = QAction("Load Well Tops", self)
        load_well_tops_action.triggered.connect(self.load_well_tops)
        file_menu.addAction(load_well_tops_action)

        toggle_controls_action = QAction("Toggle Controls", self)
        toggle_controls_action.triggered.connect(self.toggle_controls)
        menubar.addAction(toggle_controls_action)

        # New action: Change Background Color
        change_bg_action = QAction("Change Background Color", self)
        change_bg_action.triggered.connect(self.change_background_color)
        menubar.addAction(change_bg_action)

        # Settings Menu
        settings_menu = menubar.addMenu("Settings")

        save_action = QAction("Save Config", self)
        save_action.triggered.connect(self.save_configuration)
        settings_menu.addAction(save_action)

        load_action = QAction("Load Config", self)
        load_action.triggered.connect(self.load_configuration)
        settings_menu.addAction(load_action)

        self.dock = QDockWidget("Control", self)
        self.dock.setStyleSheet("background-color: White; border-radius: 5px; color: blue; font: 12pt;")
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)

        dock_widget = QWidget()
        dock_layout = QVBoxLayout()  # Main layout for the dock widget

        # Horizontal layout for well_list and well_tops_list
        horizontal_layout = QHBoxLayout()

        self.well_list = ClickableListWidget()
        self.well_list.itemChanged.connect(self.update_plot)
        dock_layout.addWidget(QLabel("Loaded Wells:"))
        horizontal_layout.addWidget(self.well_list)

        # Layout for well tops label and list
        well_tops_layout = QVBoxLayout()
        well_tops_layout.addWidget(QLabel("Loaded Well Tops:"))
        self.well_tops_list = ClickableListWidget()
        self.well_tops_list.itemChanged.connect(self.update_plot)
        well_tops_layout.addWidget(self.well_tops_list)

        # Add the well tops layout to the horizontal layout
        horizontal_layout.addLayout(well_tops_layout)

        dock_layout.addLayout(horizontal_layout)

        btn_add_track = QPushButton("Track +")
        btn_add_track.setStyleSheet("background-color: White; border-radius: 5px; color: blue; font: 15pt; font-weight: bold;")
        btn_add_track.clicked.connect(self.add_track)
        dock_layout.addWidget(btn_add_track)

        self.track_tabs = QTabWidget()
        self.track_tabs.setStyleSheet("background-color: #e2e2e2; border-radius: 5px; color: #53003e; font: 10pt;")
        self.track_tabs.setTabsClosable(True)  # Enable close button on tabs
        self.track_tabs.tabCloseRequested.connect(self.delete_track)  # Connect tab close event
        dock_layout.addWidget(self.track_tabs)

        dock_widget.setLayout(dock_layout)
        self.dock.setWidget(dock_widget)

    def save_configuration(self):
        """Save well and track settings to a pickle file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "Config Files (*.pkl);;All Files (*)", options=options)
        if file_path:
            config_data = {
                "selected_wells": [self.well_list.item(i).text() for i in range(self.well_list.count()) if self.well_list.item(i).checkState() == Qt.Checked],
                "tracks": [{"curves": [curve.curve_box.currentText() for curve in track.curves], "bg_color": track.bg_color} for track in self.tracks]
            }
            with open(file_path, "wb") as f:  # Use binary write mode
                pickle.dump(config_data, f)

    def load_configuration(self):
        """Load well and track settings from a pickle file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "Config Files (*.pkl);;All Files (*)", options=options)
        if file_path:
            with open(file_path, "rb") as f:  # Use binary read mode
                config_data = pickle.load(f)

            # Restore selected wells
            for i in range(self.well_list.count()):
                item = self.well_list.item(i)
                if item.text() in config_data["selected_wells"]:
                    item.setCheckState(Qt.Checked)
                else:
                    item.setCheckState(Qt.Unchecked)

            # Restore tracks
            self.tracks.clear()
            self.track_tabs.clear()
            for track_data in config_data["tracks"]:
                track = TrackControl(len(self.tracks) + 1, [])
                track.bg_color = track_data["bg_color"]
                self.tracks.append(track)
                self.track_tabs.addTab(track, f"Track {track.number}")

            self.update_plot()

    def change_background_color(self):
        """Opens a color picker to change the background color."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.setStyleSheet(f"QWidget {{ background-color: {color.name()}; }}")

    def toggle_controls(self):
        self.dock.setVisible(not self.dock.isVisible())

    def load_las_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing LAS Files")
        if folder:
            for filename in os.listdir(folder):
                if filename.lower().endswith(".las"):
                    self.load_las_file(os.path.join(folder, filename))
            self.update_plot()

    def load_las_files(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Select LAS Files", "", "LAS Files (*.las);;All Files (*)", options=options)
        if files:
            for file in files:
                self.load_las_file(file)
            self.update_plot()

    def load_well_tops(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Well Tops", "", "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            try:
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(".txt"):
                    df = pd.read_csv(file_path, delimiter=r"\s+")
                else:
                    print("Unsupported file format. Please use CSV or TXT.")
                    return

                # Normalize column names to lowercase and strip spaces and special characters
                df.columns = df.columns.str.lower().str.replace(r'[^a-z0-9_]', '', regex=True)
                print("Columns in the file:", df.columns.tolist())  # Debug print

                # Ensure the DataFrame has the required columns
                required_columns = {'well', 'top', 'md'}
                if not required_columns.issubset(df.columns):
                    print(f"Invalid file format. Required columns: {', '.join(required_columns)}")
                    return

                self.well_tops = df.to_dict(orient="records")
                self.well_tops_list.clear()

                # Collect unique top values
                unique_tops = set(well_top['top'] for well_top in self.well_tops)

                # Add unique top values to the list widget
                for top in unique_tops:
                    item = QListWidgetItem(top)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Unchecked)
                    self.well_tops_list.addItem(item)

                self.update_plot()
            except Exception as e:
                print(f"Error loading well tops file: {str(e)}")

    def load_las_file(self, path):
        try:
            las = lasio.read(path)
            df = las.df()
            df.reset_index(inplace=True)
            df.dropna(inplace=True)  # Remove rows with NaN values
            # Find a valid depth column.

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

    def add_track(self):
        if not self.wells:
            return

        curves = sorted(set(curve for well in self.wells.values() for curve in well['data'].columns))
        track = TrackControl(len(self.tracks) + 1, curves)
        track.changed.connect(self.update_plot)
        track.deleteRequested.connect(self.delete_track)
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
        """Renumber tracks and update their tab titles."""
        for i, track in enumerate(self.tracks, start=1):
            track.number = i
            track.update_curve_numbers()  # Renumber curves within the track
            self.track_tabs.setTabText(i - 1, f"Track {i}")

    def update_plot(self):
        selected_wells = [self.well_list.item(i).text() for i in range(self.well_list.count()) if self.well_list.item(i).checkState() == Qt.Checked]
        selected_well_tops = [self.well_tops_list.item(i).text() for i in range(self.well_tops_list.count()) if self.well_tops_list.item(i).checkState() == Qt.Checked]
        well_tops_to_plot = [well_top for well_top in self.well_tops if well_top['top'] in selected_well_tops and well_top['well'] in selected_wells]

        for well in selected_wells:
            if well not in self.figure_widgets:
                self.figure_widgets[well] = FigureWidget(well)
                self.figure_layout.addWidget(self.figure_widgets[well])
                self.figure_widgets[well].curve_clicked.connect(self.open_edit_curve_dialog)
                self.figure_widgets[well].track_clicked.connect(self.open_edit_track_dialog)  # Connect track click signal
            self.figure_widgets[well].update_plot(self.wells[well]['data'], self.tracks, well_tops_to_plot)

        # Remove figure widgets for deselected wells.
        for well in list(self.figure_widgets.keys()):
            if well not in selected_wells:
                widget = self.figure_widgets[well]
                self.figure_layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
                del self.figure_widgets[well]

    def open_edit_curve_dialog(self, curve_name, curve):
        """Open the edit curve dialog for the clicked curve."""
        dialog = EditCurveDialog(curve, self)
        if dialog.exec_():
            curve.grid.setChecked(dialog.grid_state)
            curve.flip.setChecked(dialog.flip_state)
            curve.color = dialog.color
            curve.width.setValue(dialog.width_value)
            curve.line_style_box.setCurrentText(dialog.line_style)
            curve.changed.emit()
            self.update_plot()

    def open_edit_track_dialog(self, track):
        """Open the edit track dialog for the clicked track."""
        dialog = EditTrackDialog(track, self)
        if dialog.exec_():
            # Update the track properties based on dialog results
            track.grid.setChecked(dialog.grid_state)
            track.flip_y.setChecked(dialog.flip_y_state)
            track.bg_color = dialog.bg_color
            track.changed.emit()  # Emit signal to update the plot

app = QApplication(sys.argv)
app.setStyleSheet(loadStyleSheet("style/darkmode.qss"))
viewer = WellLogViewer()
viewer.show()

sys.exit(app.exec_())
