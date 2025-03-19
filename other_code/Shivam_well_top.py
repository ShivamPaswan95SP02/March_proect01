import sys
import os
import lasio
import pickle
import pandas as pd
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QMenu, QVBoxLayout, QFormLayout, QLabel, QLineEdit,
    QDialogButtonBox, QMainWindow, QDockWidget, QListWidget,
    QListWidgetItem, QWidget, QComboBox, QPushButton, QCheckBox, QSpinBox,
    QScrollArea, QAction, QColorDialog, QTabWidget, QFrame, QApplication
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def loadStyleSheet(fileName):
    try:
        with open(fileName, "r") as f:
            return f.read()
    except Exception as e:
        print("Failed to load stylesheet:", e)
        return ""

# --- Custom QListWidget ---
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

# --- FigureWidget with well tops plotting and annotation ---
class FigureWidget(QWidget):
    mouse_moved = pyqtSignal(float, float)
    curve_clicked = pyqtSignal(str, object)
    track_clicked = pyqtSignal(object)

    def __init__(self, well_name, parent=None):
        super().__init__(parent)
        self.well_name = well_name
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.active_well_tops = []
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        """Handle click events to detect which track was clicked."""
        if event.inaxes:
            if event.button == 3:  # Right-click
                for track in self.tracks:
                    if event.inaxes == track.ax:
                        self.track_clicked.emit(track)
                        return

    def update_plot(self, data, tracks, well_top_lines=None):
        self.figure.clear()
        self.data = data
        self.tracks = tracks
        self.figure.text(0.01, 0.99, f"Well: {self.well_name}", ha='left', va='top', fontsize=10, color='Grey')

        n_tracks = len(tracks)
        if n_tracks == 0:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No tracks", ha='center', va='center')
        else:
            # Determine maximum number of valid curves across all tracks.
            max_valid_curves = max([
                len([curve for curve in track.curves
                     if curve.curve_box.currentText() != "Select Curve" and curve.curve_box.currentText() in data.columns])
                for track in tracks
            ] or [1])
            # Adjust the top margin.
            top_margin = max(0.95 - (max_valid_curves - 1) * 0.03, 0.85)
            axes = self.figure.subplots(1, n_tracks, sharey=True) if n_tracks > 1 else [self.figure.add_subplot(111)]
            self.figure.subplots_adjust(wspace=0, bottom=0.005, top=top_margin)
            depth = data['DEPT']
            all_lines = []
            for idx, (ax, track) in enumerate(zip(axes, tracks)):
                ax.clear()
                ax.set_facecolor(track.bg_color)
                track.ax = ax  # Store the primary axis for the track.
                # Plot curves.
                valid_curves = []
                for curve in track.curves:
                    curve_name = curve.curve_box.currentText()
                    if curve_name == "Select Curve" or curve_name not in data.columns:
                        continue
                    valid_curves.append(curve)
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                for i, curve in enumerate(valid_curves):
                    twin_ax = ax.twiny()
                    twin_ax.tick_params(axis='both', which='major', labelsize=8, colors=curve.color)
                    base_offset = 20 if max_valid_curves <= 3 else 10
                    offset = base_offset * i
                    twin_ax.spines["top"].set_position(("outward", offset))
                    twin_ax.spines["bottom"].set_visible(False)
                    twin_ax.spines["right"].set_visible(False)
                    twin_ax.spines["left"].set_visible(False)
                    twin_ax.tick_params(axis='x', which='both', bottom=False, top=True, labeltop=True)
                    twin_ax.set_xlabel(curve.curve_box.currentText(), color=curve.color)
                    twin_ax.spines["top"].set_edgecolor(curve.color)
                    if curve.flip_x.isChecked():
                        twin_ax.invert_xaxis()
                    if curve.x_min.text():
                        try:
                            xmin_val = float(curve.x_min.text())
                            cur_xlim = twin_ax.get_xlim()
                            twin_ax.set_xlim(xmin_val, cur_xlim[1])
                        except ValueError:
                            pass
                    if curve.x_max.text():
                        try:
                            xmax_val = float(curve.x_max.text())
                            cur_xlim = twin_ax.get_xlim()
                            twin_ax.set_xlim(cur_xlim[0], xmax_val)
                        except ValueError:
                            pass
                    if curve.scale_combobox.currentText() == "Log":
                        twin_ax.set_xscale("log")
                    else:
                        twin_ax.set_xscale("linear")
                    line, = twin_ax.plot(
                        data[curve.curve_box.currentText()], depth,
                        color=curve.color,
                        linewidth=curve.width.value(),
                        linestyle=curve.get_line_style(),
                        picker=True
                    )
                    line.set_gid(curve.curve_box.currentText())
                    all_lines.append(line)
                if idx == 0:
                    ax.set_ylabel("Depth")
                ax.grid(track.grid.isChecked())
                ax.set_ylim(depth.max(), depth.min())
                if track.flip_y.isChecked():
                    ax.invert_yaxis()
            if all_lines:
                self.figure.legend(all_lines, [line.get_gid() for line in all_lines],
                                   loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=4, fontsize='small')
            # --- Plot Well Tops ---
            # well_top_lines is expected to be a list of tuples (top, md) for this well.
            if well_top_lines:
                for track in tracks:
                    for (top, md) in well_top_lines:
                        track.ax.axhline(y=md, color='red', linestyle='--', linewidth=1)
                        track.ax.text(
                            0.98, md, f"{self.well_name}: {top}",
                            transform=track.ax.get_yaxis_transform(),
                            color='red', fontsize=8, horizontalalignment='right', verticalalignment='bottom'
                        )
        self.canvas.draw()

# --- CurveControl remains unchanged ---
class CurveControl(QWidget):
    changed = pyqtSignal()
    deleteRequested = pyqtSignal(object)

    def __init__(self, curve_number, curves, default_color="#1f77b4", parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        top_row = QHBoxLayout()
        self.curve_label = QLabel(f"Curve {curve_number}:")
        top_row.addWidget(self.curve_label)
        self.curve_box = QComboBox()
        self.curve_box.addItem("Select Curve")
        self.curve_box.addItems(curves)
        self.curve_box.currentIndexChanged.connect(self.changed.emit)
        top_row.addWidget(self.curve_box)
        self.width = QSpinBox()
        self.width.setRange(1, 5)
        self.width.setValue(1)
        self.width.valueChanged.connect(self.changed.emit)
        top_row.addWidget(QLabel("Width:"))
        top_row.addWidget(self.width)
        self.color = default_color
        self.color_btn = QPushButton("Color")
        self.color_btn.setStyleSheet(f"background-color: {self.color}; border: none;")
        self.color_btn.clicked.connect(self.select_color)
        top_row.addWidget(self.color_btn)
        self.line_style_box = QComboBox()
        self.line_style_box.addItems(["Solid", "Dashed", "Dotted", "Dash-dot"])
        self.line_style_box.currentIndexChanged.connect(self.changed.emit)
        top_row.addWidget(QLabel("Line Style:"))
        top_row.addWidget(self.line_style_box)
        main_layout.addLayout(top_row)
        bottom_row = QHBoxLayout()
        self.flip_x = QCheckBox("Flip X-Axis")
        self.flip_x.stateChanged.connect(self.changed.emit)
        bottom_row.addWidget(self.flip_x)
        bottom_row.addWidget(QLabel("X min:"))
        self.x_min = QLineEdit()
        self.x_min.setPlaceholderText("Auto")
        self.x_min.textChanged.connect(self.changed.emit)
        bottom_row.addWidget(self.x_min)
        bottom_row.addWidget(QLabel("X max:"))
        self.x_max = QLineEdit()
        self.x_max.setPlaceholderText("Auto")
        self.x_max.textChanged.connect(self.changed.emit)
        bottom_row.addWidget(self.x_max)
        bottom_row.addWidget(QLabel("Scale:"))
        self.scale_combobox = QComboBox()
        self.scale_combobox.addItems(["Linear", "Log"])
        self.scale_combobox.currentIndexChanged.connect(self.changed.emit)
        bottom_row.addWidget(self.scale_combobox)
        main_layout.addLayout(bottom_row)

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

# --- TrackControl remains mostly unchanged ---
class TrackControl(QWidget):
    changed = pyqtSignal()
    deleteRequested = pyqtSignal(object)

    def __init__(self, number, curves, parent=None):
        super().__init__(parent)
        self.number = number
        self.curves = []
        self.bg_color = "#FFFFFF"
        self.curve_count = 0
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        layout = QVBoxLayout(self)
        combined_layout = QHBoxLayout()
        self.grid = QCheckBox("Grid")
        self.grid.setChecked(True)
        self.grid.stateChanged.connect(self.changed.emit)
        combined_layout.addWidget(self.grid)
        self.flip_y = QCheckBox("Flip Y-Axis")
        self.flip_y.stateChanged.connect(self.changed.emit)
        combined_layout.addWidget(self.flip_y)
        self.bg_color_btn = QPushButton("Bg Color")
        self.bg_color_btn.clicked.connect(self.select_bg_color)
        self.bg_color_btn.setStyleSheet(f"background-color: {self.bg_color};")
        combined_layout.addWidget(self.bg_color_btn)
        combined_layout.addWidget(QLabel("Y min:"))
        self.y_min = QLineEdit()
        self.y_min.setPlaceholderText("Auto")
        self.y_min.textChanged.connect(self.changed.emit)
        combined_layout.addWidget(self.y_min)
        combined_layout.addWidget(QLabel("Y max:"))
        self.y_max = QLineEdit()
        self.y_max.setPlaceholderText("Auto")
        self.y_max.textChanged.connect(self.changed.emit)
        combined_layout.addWidget(self.y_max)
        layout.addLayout(combined_layout)
        self.curve_tabs = QTabWidget()
        self.curve_tabs.setObjectName("curveTabs")
        self.curve_tabs.setTabsClosable(True)
        self.curve_tabs.tabCloseRequested.connect(self.remove_curve)
        layout.addWidget(self.curve_tabs)
        add_curve_btn = QPushButton("Add Curve")
        add_curve_btn.setIcon(QIcon("Icons/curve.png"))
        add_curve_btn.setLayoutDirection(Qt.RightToLeft)
        add_curve_btn.clicked.connect(lambda: self.add_curve(curves))
        layout.addWidget(add_curve_btn)
        self.add_curve(curves)

    def add_curve(self, curves):
        self.curve_count += 1
        curve = CurveControl(self.curve_count, curves)
        curve.changed.connect(self.changed.emit)
        curve.deleteRequested.connect(lambda: self.remove_curve_by_instance(curve))
        self.curves.append(curve)
        self.curve_tabs.addTab(curve, f"Curve {self.curve_count}")
        self.update_curve_numbers()
        self.changed.emit()

    def remove_curve(self, index):
        if 0 <= index < len(self.curves):
            curve = self.curves.pop(index)
            self.curve_tabs.removeTab(index)
            curve.deleteLater()
            self.update_curve_numbers()
            self.changed.emit()

    def remove_curve_by_instance(self, curve):
        if curve in self.curves:
            index = self.curve_tabs.indexOf(curve)
            self.remove_curve(index)

    def update_curve_numbers(self):
        for i, curve in enumerate(self.curves, start=1):
            curve.curve_label.setText(f"Curve {i}:")
            self.curve_tabs.setTabText(self.curve_tabs.indexOf(curve), f"Curve {i}")

    def select_bg_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.bg_color = color.name()
            self.bg_color_btn.setStyleSheet(f"background-color: {self.bg_color};")
            self.changed.emit()

# --- EditCurveDialog remains unchanged ---
class EditCurveDialog(QDialog):
    def __init__(self, track, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Track Properties")
        self.setLayout(QVBoxLayout())
        form_layout = QFormLayout()
        self.layout().addLayout(form_layout)
        self.grid = QCheckBox("Grid")
        self.grid.setChecked(track.grid.isChecked())
        form_layout.addRow(self.grid)
        self.flip = QCheckBox("Flip X-Axis")
        self.flip.setChecked(False)
        form_layout.addRow(self.flip)
        self.flip_y = QCheckBox("Flip Y-Axis")
        self.flip_y.setChecked(track.flip_y.isChecked())
        form_layout.addRow(self.flip_y)
        self.bg_color_btn = QPushButton("Background Color")
        self.bg_color_btn.setStyleSheet(f"background-color: {track.bg_color}; border: none;")
        self.bg_color_btn.clicked.connect(self.select_bg_color)
        form_layout.addRow(self.bg_color_btn)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout().addWidget(self.buttons)
        self.bg_color = track.bg_color

    def select_bg_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.bg_color = color.name()

    def accept(self):
        self.grid_state = self.grid.isChecked()
        self.flip_state = self.flip.isChecked()
        self.flip_y_state = self.flip_y.isChecked()
        super().accept()

# --- WellLogViewer with updated well tops functionality ---
class WellLogViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.wells = {}
        self.well_tops = {}       # {well: [(top, md), ...]}
        # Remove or ignore self.selected_well_tops if present.
        self.selected_top_names = set()  # NEW: Holds unique top names that are selected.
        self.tracks = []
        self.figure_widgets = {}
        self.initUI()
        self.setWindowIcon(QIcon('Icons/ongc.png'))


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
        load_folder_action = QAction("Load LAS Files", self)
        load_folder_action.triggered.connect(self.load_las_folder)
        file_menu.addAction(load_folder_action)


        load_files_action = QAction("Load LAS Files", self)
        load_files_action.triggered.connect(self.load_las_files)
        file_menu.addAction(load_files_action)

        load_welltops_action = QAction("Load Well Tops", self)
        load_welltops_action.triggered.connect(self.load_well_tops)
        file_menu.addAction(load_welltops_action)
        toggle_controls_action = QAction("Toggle Controls", self)
        toggle_controls_action.triggered.connect(self.toggle_controls)
        menubar.addAction(toggle_controls_action)
        theme_menu = menubar.addMenu("Themes")
        dark_mode_action = QAction("Dark Mode", self)
        dark_mode_action.triggered.connect(lambda: self.set_theme("styles/darkmode.qss"))
        light_mode_action = QAction("Light Mode", self)
        light_mode_action.triggered.connect(lambda: self.set_theme("styles/lightmode.qss"))
        theme_menu.addAction(dark_mode_action)
        theme_menu.addAction(light_mode_action)
        self.dock = QDockWidget("Control Panel", self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        dock_widget = QWidget()
        dock_layout = QVBoxLayout()
        list_layout = QHBoxLayout()
        # Loaded Wells list.
        self.well_list = ClickableListWidget()
        self.well_list.itemChanged.connect(self.update_plot)
        well_label = QLabel("Loaded Wells:")
        well_layout = QVBoxLayout()
        well_layout.addWidget(well_label)
        well_layout.addWidget(self.well_list)
        list_layout.addLayout(well_layout)
        # Loaded Well Tops list (shows only top names).
        self.well_tops_list = QListWidget()
        self.well_tops_list.itemChanged.connect(self.well_top_item_changed)
        welltops_label = QLabel("Loaded Well Tops:")
        welltops_layout = QVBoxLayout()
        welltops_layout.addWidget(welltops_label)
        welltops_layout.addWidget(self.well_tops_list)
        list_layout.addLayout(welltops_layout)
        dock_layout.addLayout(list_layout)
        btn_add_track = QPushButton("Track")
        btn_add_track.setToolTip("Add Track")
        btn_add_track.setIcon(QIcon("Icons/plus.png"))
        btn_add_track.setLayoutDirection(Qt.RightToLeft)
        btn_add_track.setObjectName("btnAddTrack")
        btn_add_track.clicked.connect(self.add_track)
        dock_layout.addWidget(btn_add_track)
        self.track_tabs = QTabWidget()
        self.track_tabs.setObjectName("trackTabs")
        self.track_tabs.setTabsClosable(True)
        self.track_tabs.tabCloseRequested.connect(self.delete_track)
        dock_layout.addWidget(self.track_tabs)
        dock_widget.setLayout(dock_layout)
        self.dock.setWidget(dock_widget)

    def set_theme(self, theme_file):
        QApplication.instance().setStyleSheet(loadStyleSheet(theme_file))

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
            # Read the file with whitespace delimiter and no header.
            df = pd.read_csv(
                file_path,
                delim_whitespace=True,
                header=None,
                engine='python',
                on_bad_lines='skip'
            )
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
        for i, track in enumerate(self.tracks, start=1):
            track.number = i
            track.update_curve_numbers()
            self.track_tabs.setTabText(i - 1, f"Track {i}")

    def update_plot(self):
        selected_wells = [self.well_list.item(i).text() for i in range(self.well_list.count())
                        if self.well_list.item(i).checkState() == Qt.Checked]
        for well in selected_wells:
            if well not in self.figure_widgets:
                self.figure_widgets[well] = FigureWidget(well)
                self.figure_layout.addWidget(self.figure_widgets[well])
                self.figure_widgets[well].curve_clicked.connect(self.open_edit_curve_dialog)
                self.figure_widgets[well].track_clicked.connect(self.open_edit_track_dialog)
            # Build list of (top, md) pairs for this well if the top is selected.
            well_top_lines = []
            if well in self.well_tops:
                for top, md in self.well_tops[well]:
                    if top in self.selected_top_names:
                        well_top_lines.append((top, md))
            self.figure_widgets[well].update_plot(self.wells[well]['data'], self.tracks, well_top_lines)
        for well in list(self.figure_widgets.keys()):
            if well not in selected_wells:
                widget = self.figure_widgets[well]
                self.figure_layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
                del self.figure_widgets[well]

    def open_edit_curve_dialog(self, curve_name, curve):
        available_curves = sorted(set(curve for well in self.wells.values() for curve in well['data'].columns))
        dialog = EditCurveDialog(curve, self)
        if dialog.exec_():
            curve.changed.emit()
            self.update_plot()

    def open_edit_track_dialog(self, track):
        dialog = EditCurveDialog(track, self)
        if dialog.exec_():
            track.grid.setChecked(dialog.grid_state)
            track.flip.setChecked(dialog.flip_state)
            track.flip_y.setChecked(dialog.flip_y_state)
            track.bg_color = dialog.bg_color
            track.changed.emit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(loadStyleSheet("style/darkmode.qss"))
    viewer = WellLogViewer()
    viewer.show()
    sys.exit(app.exec_())