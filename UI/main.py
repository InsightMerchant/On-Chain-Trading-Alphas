# ===== main.py =====
import sys, os, json, time, itertools, random, re, io, gspread, pathlib
from itertools import cycle
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QPushButton, QTabWidget, QMessageBox, QLabel, QTextEdit,
    QSplitter, QTableWidget, QTableWidgetItem, QSizePolicy, QGroupBox, QLineEdit,
    QGraphicsPixmapItem, QGraphicsView, QGraphicsScene, QListWidget, QListWidgetItem,
    QProgressBar, QAbstractItemView, QInputDialog, QCheckBox, QFileDialog, QSizePolicy,
    QRadioButton, QMenu, QShortcut,QStackedWidget, QCompleter, QStyleFactory, QSpinBox,
    QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QFileSystemWatcher
from PyQt5.QtGui import QPixmap, QPainter, QLinearGradient, QColor, QBrush, QKeySequence, QFont, QIcon, QPen
import pyqtgraph as pg
from pyqtgraph.graphicsItems.DateAxisItem import DateAxisItem
from pyqtgraph import GraphicsLayoutWidget
from functools import partial
from engines.bruteforce_engine import run_bruteforce_mode
from concurrent.futures import ProcessPoolExecutor
from utils.formula import load_formulas, parse_formula, validate_factors, merge_factor_data, evaluate_formula_on_df
from utils.formula import sanitize_for_folder
from helper.shift_helper import determine_shift_override
from utils.transformation import TRANSFORMATIONS
from google.oauth2.service_account import Credentials
from utils.formula import parse_formula
from gspread.exceptions import APIError
from helper.factor_loader import load_factor_dataframe
from bruteforce_tab import BruteforceTab, filter_factor_options
from utils.model_loader import ModelLoader
from utils.plugin_loader import PluginLoader
from utils.reporting import Reporter
from utils.data_loader import DataLoader
from config.config_manager import ConfigManager
from utils.datasource_utils import aggregate_factor_options
from utils.interval_helper import extract_interval_from_filename, get_common_intervals, format_interval
from UI.research_tab import ResearchTab

HERE         = pathlib.Path(__file__).parent
PROJECT_ROOT = HERE.parent
FORMULAS_CSV     = PROJECT_ROOT / "formulas.csv"
KEYFILE          = PROJECT_ROOT / "credentials" / "bruteforce.json"
CONFIG_DIR       = PROJECT_ROOT / "config"
CONFIG_SAVE_PATH = CONFIG_DIR / "bruteforce_tab_config.json"
THEME_PATH       = HERE / "theme.qss"
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"
# ==============================================================================================

def run_backtest_in_process(calculate_signal, df, params, logic_plugins, reporter):
    from engines.backtest_engine import BacktestEngine
    engine = BacktestEngine(reporter)
    results = engine.run(calculate_signal, df, params, logic_plugins)
    return results

def run_walkforward_in_process(calculate_signal, df, params, logic_plugins, reporter):
    from engines.walkforward_engine import WalkforwardEngine
    engine = WalkforwardEngine(reporter)
    results = engine.run(calculate_signal, df, params, logic_plugins)
    return results

def run_permutation_in_process(calculate_signal, df, params, logic_plugin, reporter):
    from engines.permutation_engine import PermutationEngine
    engine = PermutationEngine(reporter)
    results, filtered = engine.run(calculate_signal, df, params, logic_plugin)
    return {"results": results, "filtered": filtered}


def create_layout_icon(rows, cols, size=24):
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    pen = QPen(Qt.lightGray)
    pen.setWidth(1)
    p.setPen(pen)
    w, h = size/cols, size/rows
    for r in range(rows):
        for c in range(cols):
            p.drawRect(int(c*w)+1, int(r*h)+1, int(w)-2, int(h)-2)
    p.end()
    return QIcon(pix)

class MultiprocessingBacktestWorker(QThread):
    finished = pyqtSignal(dict)
    
    def __init__(self, calculate_signal, df, params, logic_plugins, reporter):
        super().__init__()
        self.calculate_signal = calculate_signal
        self.df = df
        self.params = params
        self.logic_plugins = logic_plugins
        self.reporter = reporter

    def run(self):
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_backtest_in_process,
                                     self.calculate_signal,
                                     self.df,
                                     self.params,
                                     self.logic_plugins,
                                     self.reporter)
            result = future.result()
        self.finished.emit(result)

class MultiprocessingWalkforwardWorker(QThread):
    finished = pyqtSignal(dict)
    
    def __init__(self, calculate_signal, df, params, logic_plugins, reporter):
        super().__init__()
        self.calculate_signal = calculate_signal
        self.df = df
        self.params = params
        self.logic_plugins = logic_plugins
        self.reporter = reporter

    def run(self):
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_walkforward_in_process,
                                     self.calculate_signal,
                                     self.df,
                                     self.params,
                                     self.logic_plugins,
                                     self.reporter)
            result = future.result()
        self.finished.emit(result)

class MultiprocessingPermutationWorker(QThread):
    finished = pyqtSignal(dict)
    
    def __init__(self, calculate_signal, df, params, logic_plugin, reporter):
        super().__init__()
        self.calculate_signal = calculate_signal
        self.df = df
        self.params = params
        self.logic_plugin = logic_plugin
        self.reporter = reporter

    def run(self):
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                run_permutation_in_process,
                self.calculate_signal,
                self.df,
                self.params,
                self.logic_plugin,
                self.reporter
            )
            result = future.result()
        self.finished.emit(result)

class PermutationLoaderThread(QThread):
    loaded = pyqtSignal(object, object)

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        json_data = None
        pixmap = None

        json_path = os.path.join(self.folder, "permutation_report_all.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    json_data = json.load(f)
            except Exception as e:
                print("Error loading JSON:", e)

        heatmap_path = os.path.join(self.folder, "heatmap.png")
        if os.path.exists(heatmap_path):
            pixmap = QPixmap(heatmap_path)
        self.loaded.emit(json_data, pixmap)

# ===================== Worker Classes =====================

class BacktestWorker(QThread):
    finished = pyqtSignal(dict)  

    def __init__(self, calculate_signal, df, params, logic_plugins, reporter):
        super().__init__()
        self.calculate_signal = calculate_signal
        self.df = df
        self.params = params
        self.logic_plugins = logic_plugins
        self.reporter = reporter

    def run(self):
        from engines.backtest_engine import BacktestEngine
        engine = BacktestEngine(self.reporter)
        results = engine.run(self.calculate_signal, self.df, self.params, self.logic_plugins)
        self.finished.emit(results)

class WalkforwardWorker(QThread):
    finished = pyqtSignal(dict)  

    def __init__(self, calculate_signal, df, params, logic_plugins, reporter):
        super().__init__()
        self.calculate_signal = calculate_signal
        self.df = df
        self.params = params
        self.logic_plugins = logic_plugins
        self.reporter = reporter

    def run(self):
        from engines.walkforward_engine import WalkforwardEngine
        engine = WalkforwardEngine(self.reporter)
        results = engine.run(self.calculate_signal, self.df, self.params, self.logic_plugins)
        self.finished.emit(results)

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom = 0
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
    
    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.25
            self._zoom += 1
        else:
            factor = 0.8
            self._zoom -= 1
        self.scale(factor, factor)

class ValidatorTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def contextMenuEvent(self, event):
        global_pos = self.viewport().mapToGlobal(event.pos())
        self.parent().show_validator_context_menu(global_pos)
        event.accept()
        
# ===================== Backtest Tab =====================
class BacktestTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.perPlotType = {}
        self.saved_params = {'single': {}, 'range': {}}
        self.current_permutation_row = None
        self.paramInputs = {}
        self.factorOptions = []
        self.ds_map = {}
        self.original_ds_map = {}
        self.cache = {}
        self.factor_cache = {}
        self.factorComboBoxes = []
        self.activeFilters = []
        self.current_metric_keys = []
        self.datasource_folder = os.path.join("data", "datasource")

        # build the UI and load datasource
        self.setup_ui()
        self.load_available_options()
        self.folderWatcher = QFileSystemWatcher([self.datasource_folder], self)
        self.formulaWatcher = QFileSystemWatcher([str(FORMULAS_CSV)], self)
        self.formulaWatcher.fileChanged.connect(self.on_formulas_changed)

        # load formulas for the “Formula” combobox
        formulas = load_formulas(str(FORMULAS_CSV))
        self.formula_map = {}
        self.formulaComboBox.setEditable(True)
        completer = QCompleter(self.formulaComboBox.model(), self.formulaComboBox)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setFilterMode(Qt.MatchContains)
        self.formulaComboBox.setCompleter(completer)

        self.formulaComboBox.clear()
        self.formulaComboBox.addItem("<none>")
        for raw in formulas:
            safe = sanitize_for_folder(raw)
            self.formula_map[raw] = safe
            self.formula_map[safe] = raw
            self.formulaComboBox.addItem(raw)
        
    def setup_ui(self):
        mainLayout = QVBoxLayout(self)
        self.splitter = QSplitter(Qt.Vertical)
        mainLayout.addWidget(self.splitter)
        
        # --- Chart Area ---
        pg.setConfigOption('background', '#020202')
        pg.setConfigOption('foreground', '#ffffff')
        self.chartTabWidget = QTabWidget()
        equityCurveWidget = QWidget()
        equityLayout = QVBoxLayout(equityCurveWidget)
        self.plotWidget = pg.PlotWidget(axisItems={'bottom': pg.DateAxisItem(orientation='bottom')})
        self.plotWidget.setBackground('#020202')
        self.plotWidget.getAxis('left').setPen(pg.mkPen(color='#ffffff'))
        self.plotWidget.getAxis('bottom').setPen(pg.mkPen(color='#ffffff'))
        equityLayout.addWidget(self.plotWidget)
        self.chartTabWidget.addTab(equityCurveWidget, "Chart")

        # ─── Data Visualization Tab ──────────────────────────────
        dataVizWidget = QWidget()
        dvLayout     = QVBoxLayout(dataVizWidget)

        # Controls
        controls       = QWidget()
        controlsLayout = QVBoxLayout(controls)

        controlsLayout.addWidget(QLabel("Select Columns:"))
        self.dataVizList = QListWidget()
        self.dataVizList.setSelectionMode(QAbstractItemView.MultiSelection)
        controlsLayout.addWidget(self.dataVizList)
        controlsLayout.addSpacing(5)

        controlsLayout.addWidget(QLabel("Plot Type (per plot):"))
        self.perPlotTypeContainer = QWidget()
        self.perPlotTypeLayout  = QFormLayout(self.perPlotTypeContainer)
        self.perPlotTypeLayout.setLabelAlignment(Qt.AlignRight)
        controlsLayout.addWidget(self.perPlotTypeContainer)
        controlsLayout.addSpacing(5)

        controlsLayout.addWidget(QLabel("Layout:"))
        self.dataVizLayoutCombo = QComboBox()
        # add as many presets as you like:
        preset_layouts = [
            (1,1), (1,2), (1,3),
            (2,2), (2,3), (3,3),
        ]
        for r, c in preset_layouts:
            icon = create_layout_icon(r, c, size=24)
            self.dataVizLayoutCombo.addItem(icon, f"{r}×{c}")
        self.dataVizLayoutCombo.addItem("Custom…")
        controlsLayout.addWidget(self.dataVizLayoutCombo)

        rcLayout = QHBoxLayout()
        rcLayout.addWidget(QLabel("Cols:"))
        self.dataVizColsSpin = QSpinBox()
        self.dataVizColsSpin.setRange(1, 10)
        rcLayout.addWidget(self.dataVizColsSpin)
        rcLayout.addWidget(QLabel("Rows:"))
        self.dataVizRowsSpin = QSpinBox()
        self.dataVizRowsSpin.setRange(1, 10)
        rcLayout.addWidget(self.dataVizRowsSpin)
        controlsLayout.addLayout(rcLayout)

        # Plot container
        self.dataVizContainer = QWidget()
        self.dataVizContainer.setStyleSheet("background: #020202;")
        self.dataVizLayout = QVBoxLayout(self.dataVizContainer)
        self.dataVizLayout.setContentsMargins(0,0,0,0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.dataVizContainer)
        splitter.addWidget(controls)
        controls.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        controls.setMinimumWidth(50)   
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        dvLayout.addWidget(splitter)
        # Hook signals
        self.dataVizList.itemSelectionChanged.connect(self.update_data_visualization)
        self.dataVizLayoutCombo.currentIndexChanged.connect(self.update_data_visualization)
        self.dataVizRowsSpin.valueChanged.connect(self.update_data_visualization)
        self.dataVizColsSpin.valueChanged.connect(self.update_data_visualization)

        self.chartTabWidget.insertTab(1, dataVizWidget, "Data Visualization")
        
        # --- Combined Heatmap and Permutation List ---
        combinedSplitter = QSplitter(Qt.Horizontal)
        heatmapWidget = QWidget()
        heatmapLayout = QVBoxLayout(heatmapWidget)
        self.heatmapView = ZoomableGraphicsView()
        self.heatmapScene = QGraphicsScene(self)
        self.heatmapView.setScene(self.heatmapScene)
        heatmapLayout.addWidget(self.heatmapView)
        combinedSplitter.addWidget(heatmapWidget)
        
        self.permutationListTab = QWidget()
        permListLayout = QVBoxLayout(self.permutationListTab)
        filterLayout = QHBoxLayout()
        self.metricsFilterComboBox = QComboBox()
        self.metricsFilterComboBox.setToolTip("Select a metric to filter on")
        self.metricsFilterComboBox.setMinimumWidth(200)
        self.metricsFilterLineEdit = QLineEdit()
        self.metricsFilterLineEdit.setPlaceholderText("Enter benchmark value...")
        self.operatorComboBox = QComboBox()
        self.operatorComboBox.addItems([">=", "<="])
        self.operatorComboBox.setToolTip("Select comparison operator")
        self.filterButton = QPushButton("Filter")
        self.filterButton.clicked.connect(self.add_filter)
        filterLayout.addWidget(QLabel("Metric:"))
        filterLayout.addWidget(self.metricsFilterComboBox)
        filterLayout.addWidget(QLabel("Benchmark:"))
        filterLayout.addWidget(self.metricsFilterLineEdit)
        filterLayout.addWidget(QLabel("Operator:"))
        filterLayout.addWidget(self.operatorComboBox)
        filterLayout.addWidget(self.filterButton)
        permListLayout.addLayout(filterLayout)
        
        # Active Filters Area
        self.activeFiltersWidget = QWidget()
        self.activeFiltersLayout = QHBoxLayout(self.activeFiltersWidget)
        self.activeFiltersLayout.setContentsMargins(5, 5, 5, 5)
        self.activeFiltersLayout.setSpacing(5)
        self.activeFiltersLayout.addStretch()
        permListLayout.addWidget(self.activeFiltersWidget)
        
        # Permutation List Table
        self.permutationListTable = QTableWidget()
        self.permutationListTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.permutationListTable.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.permutationListTable.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.permutationListTable.setContextMenuPolicy(Qt.CustomContextMenu)
        self.permutationListTable.customContextMenuRequested.connect(self.show_permutation_context_menu)
        permListLayout.addWidget(self.permutationListTable)
        combinedSplitter.addWidget(self.permutationListTab)
        combinedSplitter.setStretchFactor(0, 1)
        combinedSplitter.setStretchFactor(1, 1)
        self.chartTabWidget.addTab(combinedSplitter, "Permutation")
        self.splitter.addWidget(self.chartTabWidget)

        # ─── Validator tab ───────────────────────────────────────
        self.validatorWidget = QWidget()
        valL = QVBoxLayout(self.validatorWidget)
        self.importValidatorButton = QPushButton("Import File")
        self.importValidatorButton.setToolTip("Load a .csv, .xlsx or .xls into Validator")
        valL.addWidget(self.importValidatorButton)
        self.importValidatorButton.clicked.connect(self.import_validator_file)
        self.validatorTable = QTableWidget()
        self.validatorTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.validatorTable.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.current_validation_row = None
        self.validatorTable.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.validatorTable.customContextMenuRequested.connect(self.show_validator_context_menu)
        # right-click menu
        self.validatorTable.setContextMenuPolicy(Qt.CustomContextMenu)
        valL.addWidget(self.validatorTable)
        self.chartTabWidget.addTab(self.validatorWidget, "Validator")
        
        # --- Bottom Tabs (Config, Trades, Metrics) ---
        self.bottomTab = QTabWidget()
        self.configTab = QWidget()
        configLayout = QVBoxLayout(self.configTab)
        upperWidget = QWidget()
        upperLayout = QHBoxLayout(upperWidget)
        upperLayout.setContentsMargins(5, 5, 5, 5)
        upperLayout.setSpacing(5)
        
        self.settingsGroup = QGroupBox("Settings")
        self.settingsGroup.setStyleSheet("QGroupBox { font-weight: bold; }")
        settingsLayout = QFormLayout(self.settingsGroup)
        settingsLayout.setLabelAlignment(Qt.AlignRight)
        settingsLayout.setHorizontalSpacing(5)
        settingsLayout.setVerticalSpacing(5)
        self.symbolComboBox = QComboBox()
        self.modelComboBox = QComboBox()
        self.intervalComboBox = QComboBox()
        self.logicComboBox = QComboBox()
        self.symbolComboBox.currentIndexChanged.connect(self.update_factor_options)
        self.intervalComboBox.currentIndexChanged.connect(self.update_factor_options)
        settingsLayout.addRow("Symbol:", self.symbolComboBox)
        settingsLayout.addRow("Model:", self.modelComboBox)
        settingsLayout.addRow("Interval:", self.intervalComboBox)
        self.formulaComboBox = QComboBox()
        self.formulaComboBox.setToolTip("Select a saved formula")
        
        
        
        self.formulaComboBox.currentIndexChanged.connect(self.on_formula_selected)
        
        # --- Factor Columns row: split into two halves ---
        self.factorsWidget = QWidget()
        self.factorsLayout = QHBoxLayout(self.factorsWidget)
        self.factorsLayout.setContentsMargins(0, 0, 0, 0)
        self.factorsLayout.setSpacing(5)
        self.factorComboContainer = QHBoxLayout()
        self.factorComboContainer.setSpacing(10)
        
        self.factorsLayout.addLayout(self.factorComboContainer)
        self.factorsLayout.addStretch()
        
        self.selectionToggleButton = QPushButton("Use Formula")
        self.selectionToggleButton.setCheckable(True)
        self.selectionToggleButton.toggled.connect(self.toggle_selection_mode)
        settingsLayout.addRow("Mode:", self.selectionToggleButton)

        self.selectionComboBox = QComboBox()
        self.selectionComboBox.setEditable(True)
        completer = QCompleter(self.selectionComboBox.model(), self.selectionComboBox)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setFilterMode(Qt.MatchContains)
        self.selectionComboBox.setCompleter(completer)
        self.selectionComboBox.currentIndexChanged.connect(self.on_selection_changed)
        settingsLayout.addRow("Select:", self.selectionComboBox)
        
        self.transformationComboBox = QComboBox()
        for name in TRANSFORMATIONS.keys():
            self.transformationComboBox.addItem(name)
        settingsLayout.addRow("Transformation:", self.transformationComboBox)
        settingsLayout.addRow("Entry/Exit Logic:", self.logicComboBox)
        self.paramGroup = QGroupBox("Parameters")
        self.paramGroup.setStyleSheet("QGroupBox { font-weight: bold; }")
        vParamLayout = QVBoxLayout(self.paramGroup)
        self.modeWidget = QWidget()
        modeLayout = QHBoxLayout(self.modeWidget)
        modeLayout.setContentsMargins(0, 0, 0, 0)
        self.singleRadio = QRadioButton("Single")
        self.rangeRadio = QRadioButton("Range")
        self.singleRadio.setChecked(True)
        modeLayout.addWidget(QLabel("Parameter Mode:"))
        modeLayout.addWidget(self.singleRadio)
        modeLayout.addWidget(self.rangeRadio)
        modeLayout.addStretch()
        self.totalLabel = QLabel("Total combinations: 0")
        self.totalLabel.setStyleSheet("color: #555;")
        modeLayout.addWidget(self.totalLabel)
        self.totalLabel.hide()  
        vParamLayout.addWidget(self.modeWidget)
        self.paramInputWidget = QWidget()
        self.paramForm = QFormLayout(self.paramInputWidget)
        vParamLayout.addWidget(self.paramInputWidget)
        self.singleRadio.toggled.connect(self.update_model_parameters)
        self.rangeRadio.toggled.connect(self.update_model_parameters)
        self.singleRadio.toggled.connect(self.update_run_buttons)
        self.rangeRadio.toggled.connect(self.update_run_buttons)
        
        upperLayout.addWidget(self.settingsGroup, 1)
        upperLayout.addWidget(self.paramGroup, 1)
        configLayout.addWidget(upperWidget)
        
        # Lower Buttons 
        buttonLayout = QHBoxLayout()
        buttonLayout.setContentsMargins(5, 5, 5, 5)
        buttonLayout.setSpacing(5)
        self.runBacktestButton = QPushButton("Run Backtest")
        self.runWalkforwardButton = QPushButton("Run Walkforward")
        self.runBacktestButton.clicked.connect(self.run_backtest)
        self.runWalkforwardButton.clicked.connect(self.run_walkforward)
        buttonLayout.addWidget(self.runBacktestButton)
        buttonLayout.addWidget(self.runWalkforwardButton)
        self.runPermutationButton = QPushButton("Run Permutation")
        self.importPermutationButton = QPushButton("Import Permutation Export")
        self.runPermutationButton.clicked.connect(self.run_permutation)
        self.importPermutationButton.clicked.connect(self.import_permutation)
        buttonLayout.addWidget(self.runPermutationButton)
        buttonLayout.addWidget(self.importPermutationButton)
        configLayout.addLayout(buttonLayout)
        self.bottomTab.addTab(self.configTab, "Config")
        
        # Trades Tab 
        self.tradesTab = QWidget()
        tradesLayout = QVBoxLayout(self.tradesTab)
        self.tradesTable = QTableWidget()
        self.tradesTable.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tradesLayout.addWidget(self.tradesTable)
        self.bottomTab.addTab(self.tradesTab, "Trades")
        
        # Metrics Tab 
        self.metricsTab = QWidget()
        metricsLayout = QVBoxLayout(self.metricsTab)
        self.metricsTable = QTableWidget()
        self.metricsTable.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        metricsLayout.addWidget(self.metricsTable)
        self.bottomTab.addTab(self.metricsTab, "Metrics")
        
        self.splitter.addWidget(self.bottomTab)
        self.splitter.setStretchFactor(0, 7)
        self.splitter.setStretchFactor(1, 3)
        
        # Signal Connections
        self.modelComboBox.currentIndexChanged.connect(self.update_model_parameters)
        self.logicComboBox.currentIndexChanged.connect(self.update_model_parameters)
        self.intervalComboBox.currentIndexChanged.connect(self.update_factor_options)
        self.tradesTable.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.tradesTable.horizontalHeader().customContextMenuRequested.connect(self.show_header_context_menu)
        self.update_run_buttons()

    def _build_formula_map(self):
        raw_formulas = load_formulas(str(FORMULAS_CSV))
        self.formula_map.clear()
        for raw in raw_formulas:
            safe = sanitize_for_folder(raw)
            self.formula_map[raw] = safe
            self.formula_map[safe] = raw
        current = self.formulaComboBox.currentText()
        self.formulaComboBox.blockSignals(True)
        self.formulaComboBox.clear()
        self.formulaComboBox.addItem("<none>")
        for raw in raw_formulas:
            self.formulaComboBox.addItem(raw)
        idx = self.formulaComboBox.findText(current)
        self.formulaComboBox.setCurrentIndex(idx if idx >= 0 else 0)
        self.formulaComboBox.blockSignals(False)

    def on_formulas_changed(self, path):
        if path not in self.formulaWatcher.files():
            self.formulaWatcher.addPath(path)
        try:
            self._build_formula_map()
        except Exception as e:
            QMessageBox.warning(self, "Formula Reload Error", f"Could not reload formulas.csv:\n{e}")


    def update_data_visualization(self):

        def safe_clear_layout(layout):
            while True:
                try:
                    cnt = layout.count()
                except RuntimeError:
                    break
                if cnt == 0:
                    break
                try:
                    item = layout.takeAt(0)
                except RuntimeError:
                    break
                w = item.widget()
                if w:
                    w.setParent(None)

        # Reload trades DF if needed
        if hasattr(self, 'last_backtest_csv'):
            try:
                self.full_trades_df = pd.read_csv(self.last_backtest_csv)
            except:
                pass

        df = getattr(self, 'full_trades_df', None)
        if df is None or df.empty:
            safe_clear_layout(self.dataVizLayout)
            safe_clear_layout(self.perPlotTypeLayout)
            return

        # Selected columns
        cols = [it.text() for it in self.dataVizList.selectedItems()]
        if not cols:
            safe_clear_layout(self.dataVizLayout)
            safe_clear_layout(self.perPlotTypeLayout)
            return

        # Layout rows × cols
        layout = self.dataVizLayoutCombo.currentText()
        if layout == "Custom…":
            rows, cols_per = self.dataVizRowsSpin.value(), self.dataVizColsSpin.value()
        elif "×" in layout:
            try:
                rows, cols_per = map(int, layout.split("×"))
            except:
                rows, cols_per = 1, 1
        else:
            rows, cols_per = 1, 1

        cols = cols[: rows * cols_per ]

        # Remember prior settings
        for col in cols:
            old = getattr(self, f"_plotTypeCB_{col}", None)
            if old:
                try:
                    self.perPlotType[col] = old.currentText()
                except RuntimeError:
                    pass

        # Rebuild controls
        safe_clear_layout(self.perPlotTypeLayout)
        for col in cols:
            cb = QComboBox()
            cb.addItems(["Line", "Scatter", "Histogram"])
            if col == "datetime":  # never histogram datetime
                idx = cb.findText("Histogram")
                if idx >= 0:
                    cb.removeItem(idx)
            cb.setCurrentText(self.perPlotType.get(col, "Line"))
            cb.currentTextChanged.connect(lambda txt, c=col: self.perPlotType.__setitem__(c, txt))
            cb.currentTextChanged.connect(self.update_data_visualization)
            self.perPlotTypeLayout.addRow(f"{col}:", cb)
            setattr(self, f"_plotTypeCB_{col}", cb)

        # Clear existing plots
        safe_clear_layout(self.dataVizLayout)

        # Prepare X axis
        use_date = False
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            x = (df['datetime'].astype('int64') // 10**9).values
            use_date = True
        else:
            x = np.arange(len(df))

        colors = itertools.cycle(['#4E79A7','#F28E2B','#E15759','#76B7B2','#59A14F'])

        # Create a grid container
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0,0,0,0)
        grid.setSpacing(10)

        # Populate grid
        for idx, col in enumerate(cols):
            r, c = divmod(idx, cols_per)
            # Sub-widget with vertical layout
            cell = QWidget()
            vlay = QVBoxLayout(cell)
            vlay.setContentsMargins(0,0,0,0)
            # Title label
            title = QLabel(col)
            title.setAlignment(Qt.AlignHCenter)
            vlay.addWidget(title)
            # Plot widget
            plot_type = self.perPlotType.get(col, "Line")
            if plot_type == "Histogram":
                pw = pg.PlotWidget()
                arr = df[col].dropna().values
                counts, edges = np.histogram(arr, bins='auto')
                centers = (edges[:-1]+edges[1:])/2
                width = (edges[1]-edges[0]) * 0.9
                bg = pg.BarGraphItem(x=centers, height=counts, width=width)
                pw.addItem(bg)
                pw.getAxis('bottom').setLabel(text=col)
                pw.getAxis('left').setLabel(text='Count')
            else:
                axis_items = {'bottom': DateAxisItem(orientation='bottom')} if use_date else {}
                pw = pg.PlotWidget(axisItems=axis_items)
                pen = pg.mkPen(color=next(colors), width=2)
                y = df[col].values
                if plot_type == "Line":
                    pw.plot(x, y, pen=pen, name=col)
                else:
                    pw.plot(x, y, pen=None, symbol='o',
                            symbolBrush=pen.color(), symbolSize=6)
                pw.getAxis('bottom').setLabel(text='Datetime' if use_date else 'Index')
                pw.getAxis('left').setLabel(text='Value')
                pw.addLegend(offset=(10,10))
            vlay.addWidget(pw)
            grid.addWidget(cell, r, c)

        # Insert into layout
        self.dataVizLayout.addWidget(container)
        
    def import_validator_file(self):
        from PyQt5.QtWidgets import QFileDialog, QMessageBox

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Validator File",
            "",
            "Data Files (*.csv *.xlsx *.xls)"
        )
        if not path:
            return

        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            self.update_table_widget(self.validatorTable, df)
        except Exception as e:
            print(f"[Validator] Failed to import file: {e}")
            QMessageBox.warning(self, "Error", f"Failed to import file:\n{e}")


    def update_table_widget(self, table_widget: QTableWidget, df: pd.DataFrame):
        """Helper—clears & fills a QTableWidget from a DataFrame."""
        table_widget.clear()
        table_widget.setRowCount(0)
        table_widget.setColumnCount(len(df.columns))
        table_widget.setHorizontalHeaderLabels(df.columns.tolist())
        for i, row in df.iterrows():
            table_widget.insertRow(i)
            for j, col in enumerate(df.columns):
                item = QTableWidgetItem(str(row[col]))
                table_widget.setItem(i, j, item)
        table_widget.resizeColumnsToContents()
        table_widget.resizeRowsToContents()

    def validate_and_run_walkforward(self, row):
        """Apply validator config for a single row, then run walkforward."""
        self.current_validation_row = row
        self.apply_validator_config(row)
        self.run_walkforward()

    def _run_validate_walkforward_for_rows(self, rows):
        """Queue up multiple rows: load each, run walkforward, recurse."""
        self._val_wf_queue = list(rows)
        self._process_next_val_wf_row()

    def _process_next_val_wf_row(self):
        if not getattr(self, "_val_wf_queue", None):
            return
        row = self._val_wf_queue.pop(0)
        self.current_validation_row = row
        self.apply_validator_config(row)
        # trigger walkforward
        self.run_walkforward()
        worker = self.walkforwardWorker

        def on_done(results):
            try:
                worker.finished.disconnect(on_done)
            except:
                pass
            self._process_next_val_wf_row()

        worker.finished.connect(on_done)


    def show_validator_context_menu(self, pos):
        """Right-click menu on Validator table: single or multi-row queue validation."""
        idx = self.validatorTable.indexAt(pos)
        if not idx.isValid():
            return

        # ensure the clicked row is selected
        if not self.validatorTable.selectionModel().isSelected(idx):
            self.validatorTable.clearSelection()
            self.validatorTable.selectRow(idx.row())

        rows = sorted(r.row() for r in self.validatorTable.selectionModel().selectedRows())
        if not rows:
            return

        menu = QMenu(self.validatorTable)

        if len(rows) == 1:
            row = rows[0]
            menu.addAction("Load Config",
                           lambda: self.apply_validator_config(row))
            menu.addAction("Load Config → Run Backtest",
                           lambda: self.validate_and_run(row))
            menu.addAction("Load Config → Run Walkforward",
                           lambda: self.validate_and_run_walkforward(row))
        else:
            menu.addAction("Load Config",
                           lambda: [self.apply_validator_config(r) for r in rows])
            menu.addAction("Load Config → Run Backtest",
                           lambda: self._run_validate_for_rows(rows))
            menu.addAction("Load Config → Run Walkforward",
                           lambda: self._run_validate_walkforward_for_rows(rows))
        menu.addSeparator()
        menu.addAction("Stop Queue⛔", self.stop_validator_queue)

        menu.exec_(self.validatorTable.viewport().mapToGlobal(pos))


    def stop_validator_queue(self):
        if hasattr(self, '_val_queue'):
            self._val_queue.clear()
        if hasattr(self, '_val_wf_queue'):
            self._val_wf_queue.clear()
        self.runBacktestButton.setEnabled(True)
        self.runWalkforwardButton.setEnabled(True)
        print("[Validator] Queue stopped.")


    def _run_validate_for_rows(self, rows):
        self._val_queue = list(rows)
        self._process_next_val_row()


    def _process_next_val_row(self):
        if not getattr(self, "_val_queue", None):
            return

        row = self._val_queue.pop(0)
        self.current_validation_row = row
        self.apply_validator_config(row)
        self.run_backtest()
        worker = self.backtestWorker

        def on_done(results):
            try:
                worker.finished.disconnect(on_done)
            except:
                pass
            self._process_next_val_row()

        worker.finished.connect(on_done)

    def validate_and_run(self, row):
        self.current_validation_row = row
        self.apply_validator_config(row)
        self.run_backtest()

    def apply_validator_config(self, row):
        display_row = row + 1  

        # — Helpers to read cells by header name —
        headers = [
            self.validatorTable.horizontalHeaderItem(c).text().strip()
            for c in range(self.validatorTable.columnCount())
        ]
        def text(col):
            try:
                idx = headers.index(col)
                item = self.validatorTable.item(row, idx)
                return item.text().strip() if item else ""
            except ValueError:
                return ""

        # 1) Symbol
        symbol = text("data_asset")
        si = self.symbolComboBox.findText(symbol)
        if si >= 0:
            self.symbolComboBox.setCurrentIndex(si)

        # 2) Interval
        timeframe = text("timeframe")
        ii = self.intervalComboBox.findText(timeframe)
        if ii >= 0:
            self.intervalComboBox.setCurrentIndex(ii)

        # 3) Model
        model = text("model")
        mi = self.modelComboBox.findText(model)
        if mi >= 0:
            self.modelComboBox.blockSignals(True)
            self.modelComboBox.setCurrentIndex(mi)
            self.modelComboBox.blockSignals(False)
        self.update_model_parameters()

        # 4) Figure out if this is a single‐factor or multi‐token formula
        formula = text("alpha_formula")
        tokens = re.findall(r"df\[['\"]([^'\"]+)['\"]\]", formula)
        is_single = (len(tokens) == 1)

        # 5) Toggle into the correct mode and rebuild selectionComboBox
        #    (False = factor mode, True = formula mode)
        self.selectionToggleButton.blockSignals(True)
        self.selectionToggleButton.setChecked(not is_single)
        self.selectionToggleButton.blockSignals(False)
        # this will clear & repopulate the selectionComboBox appropriately
        self.toggle_selection_mode(self.selectionToggleButton.isChecked())

        # 6) Now choose the proper entry in selectionComboBox
        if is_single:
            # pick the one matching the single token
            tok = tokens[0]
            for i in range(self.selectionComboBox.count()):
                if self.selectionComboBox.itemText(i).split("(")[0].strip() == tok:
                    self.selectionComboBox.setCurrentIndex(i)
                    break
        else:
            # pick the exact formula string
            idx = self.selectionComboBox.findText(formula)
            if idx >= 0:
                self.selectionComboBox.setCurrentIndex(idx)
            else:
                # if missing, append, rebuild & then select
                try:
                    with open(FORMULAS_CSV, "a", newline="") as f:
                        f.write("\n" + formula)
                    self._build_formula_map()
                    # rebuild selection box in formula mode
                    self.toggle_selection_mode(True)
                    idx = self.selectionComboBox.findText(formula)
                    if idx >= 0:
                        self.selectionComboBox.setCurrentIndex(idx)
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Formula Save Error",
                        f"Could not save new formula to formulas.csv:\n{e}"
                    )

        # 5) Transformation
        transf = text("data_preprocessing")
        ti = self.transformationComboBox.findText(transf)
        if ti >= 0:
            self.transformationComboBox.blockSignals(True)
            self.transformationComboBox.setCurrentIndex(ti)
            self.transformationComboBox.blockSignals(False)

        # 6) Logic
        logic = text("entry_exit_logic").lower()
        li = self.logicComboBox.findText(logic)
        if li >= 0:
            self.logicComboBox.blockSignals(True)
            self.logicComboBox.setCurrentIndex(li)
            self.logicComboBox.blockSignals(False)

        # 7) Rolling window
        rw = text("rolling_window_1") or text("rolling_window_2")
        if rw and "rolling_window" in self.paramInputs:
            self.paramInputs["rolling_window"].setText(rw)

        # 8) Read raw thresholds
        raw = {
            "long_entry_threshold":    text("long_entry_threshold"),
            "long_exit_threshold":     text("long_exit_threshold"),
            "short_entry_threshold":   text("short_entry_threshold"),
            "short_exit_threshold":    text("short_exit_threshold"),
        }

        # 9) Map raw → UI long_threshold & short_threshold
        logic_map_ui = {
            "trend":                 ("long_entry_threshold",   "short_entry_threshold"),
            "trend_long":            ("long_entry_threshold",   "long_exit_threshold"),
            "trend_short":           ("short_exit_threshold",   "short_entry_threshold"),
            "trend_reverse":         ("short_entry_threshold",  "long_entry_threshold"),
            "trend_reverse_long":    ("long_entry_threshold",    "long_exit_threshold"),
            "trend_reverse_short":   ("short_entry_threshold",   "short_exit_threshold"),
            "mr":                    ("long_entry_threshold",   "short_entry_threshold"),
            "mr_long":               ("long_entry_threshold",   None),
            "mr_short":              (None,                     "short_entry_threshold"),
            "mr_reverse":            ("short_entry_threshold",  "long_entry_threshold"),
            "mr_reverse_long":       (None,                     "long_entry_threshold"),
            "mr_reverse_short":      ("short_entry_threshold",  None),
            "fast":                  ("long_entry_threshold",   "short_entry_threshold"),
            "fast_long":             ("long_entry_threshold",   None),
            "fast_short":            (None,                     "short_entry_threshold"),
            "fast_reverse":          ("short_entry_threshold",  "long_entry_threshold"),
            "fast_reverse_long":     ("short_entry_threshold",  "short_entry_threshold"),
            "fast_reverse_short":    ("long_entry_threshold",   "long_entry_threshold"),
        }
        long_col, short_col = logic_map_ui.get(logic, (None, None))
        long_val  = raw[long_col]  if long_col  in raw and raw[long_col]  else "0"
        short_val = raw[short_col] if short_col in raw and raw[short_col] else "0"

        # 10) Apply into the UI
        if "long_threshold" in self.paramInputs:
            self.paramInputs["long_threshold"].setText(long_val)
        if "short_threshold" in self.paramInputs:
            self.paramInputs["short_threshold"].setText(short_val)

        # 11) Ensure single-mode
        if not self.singleRadio.isChecked():
            self.singleRadio.setChecked(True)

    def on_formula_selected(self, index):
        formula = self.formulaComboBox.currentText()
        if not formula or formula == "<none>":
            for cb in self.factorComboBoxes:
                cb.setEnabled(True)
            self.computed_factor_df = None
            self.selected_formula = None
            return

        for cb in self.factorComboBoxes:
            cb.setEnabled(False)

        try:
            factors = parse_formula(formula)
            missing_tokens = []
            for token in factors:
                found = False
                for key in self.ds_map:
                    base = key.split("(")[0].strip() 
                    if base == token:
                        found = True
                        break
                if not found:
                    missing_tokens.append(token)
            if missing_tokens:
                raise ValueError(f"Data not available for: {', '.join(missing_tokens)}")
            self.selected_formula = formula  
            self.computed_factor_df = None
        except Exception as e:
            QMessageBox.warning(self, "Formula Error", str(e))
            self.formulaComboBox.setCurrentIndex(0)
            for cb in self.factorComboBoxes:
                cb.setEnabled(True)
            self.selected_formula = None
            return

    def update_factor_options(self):
        prev_selection = self.selectionComboBox.currentText()
        prev_formula   = self.formulaComboBox.currentText()
        prev_factors   = [cb.currentText() for cb in self.factorComboBoxes]
        selected_symbol   = self.symbolComboBox.currentText().lower()
        selected_interval = self.intervalComboBox.currentText()
        filtered_options, filtered_ds_map = filter_factor_options(
            selected_interval,
            self.factorOptions,
            self.original_ds_map
        )
        filtered_options = [
            opt for opt in filtered_options
            if selected_symbol in filtered_ds_map[opt][0].lower()
        ]
        filtered_ds_map = {
            opt: ds_info for opt, ds_info in filtered_ds_map.items()
            if selected_symbol in ds_info[0].lower()
        }
        for idx, cb in enumerate(self.factorComboBoxes):
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(filtered_options)
            if idx < len(prev_factors):
                prev = prev_factors[idx]
                i = cb.findText(prev)
                cb.setCurrentIndex(i if i >= 0 else 0)
            elif filtered_options:
                cb.setCurrentIndex(0)
            cb.blockSignals(False)
        all_formulas   = load_formulas(str(FORMULAS_CSV))
        valid_formulas = []
        for raw in all_formulas:
            try:
                tokens = parse_formula(raw)
            except:
                continue
            if all(any(key.split("(")[0].strip() == tok for key in filtered_ds_map)
                   for tok in tokens):
                valid_formulas.append(raw)
        self.selectionComboBox.blockSignals(True)
        self.selectionComboBox.clear()
        if self.selectionToggleButton.isChecked():
            self.selectionComboBox.addItems(valid_formulas)
        else:
            self.selectionComboBox.addItems(filtered_options)
        idx = self.selectionComboBox.findText(prev_selection)
        if idx >= 0:
            self.selectionComboBox.setCurrentIndex(idx)
        self.selectionComboBox.blockSignals(False)
        self.ds_map = filtered_ds_map
        self.formulaComboBox.blockSignals(True)
        self.formulaComboBox.clear()
        self.formulaComboBox.addItem("<none>")
        for f in valid_formulas:
            self.formulaComboBox.addItem(f)
        idx = self.formulaComboBox.findText(prev_formula)
        if idx >= 0:
            self.formulaComboBox.setCurrentIndex(idx)
        self.formulaComboBox.blockSignals(False)

    def stop_walkforward(self):
        if hasattr(self, 'walkforwardWorker') and self.walkforwardWorker is not None:
            self.walkforwardWorker.terminate()  
            self.walkforwardWorker = None
            self.runWalkforwardButton.setEnabled(True)
            print("Walkforward run stopped.")

    def show_permutation_context_menu(self, pos):
        index = self.permutationListTable.indexAt(pos)
        if not index.isValid():
            return

        sel = self.permutationListTable.selectionModel().selectedRows()
        rows = sorted(
            r.row()
            for r in sel
            if 0 <= r.row() < self.permutationListTable.rowCount()
               and not self.permutationListTable.isRowHidden(r.row())
        )
        if not rows:
            return

        menu = QMenu(self.permutationListTable)
        menu.addAction("Load Config Parameters",
                       lambda: self.apply_parameters_from_permutation(rows[0]))
        menu.addAction("Run Walkforward",
                       lambda: self._run_walkforward_for_rows(rows))
        menu.addAction("Stop Walkforward",
                       lambda: self.stop_walkforward())
        menu.exec_(self.permutationListTable.viewport().mapToGlobal(pos))

    def _run_walkforward_for_rows(self, rows):
        self._wf_queue = list(rows)
        self.runWalkforwardButton.setEnabled(False)
        self.skip_walkforward_plot = True
        self._process_next_wf_row()

    def _process_next_wf_row(self):
        if not self._wf_queue:
            self.runWalkforwardButton.setEnabled(True)
            return

        row = self._wf_queue.pop(0)
        self.apply_parameters_from_permutation(row)
        self.run_walkforward()

        def on_done(results, row=row):
            self.handle_walkforward_results_for_row(results, row)
            try: self.walkforwardWorker.finished.disconnect(on_done)
            except: pass
            self._process_next_wf_row()

        self.walkforwardWorker.finished.connect(on_done)


    def handle_walkforward_results_for_row(self, results, row):
        QApplication.beep()
        logic_keys = list(results.keys())
        if not logic_keys:
            QMessageBox.warning(self, "Error", "No results returned from the walkforward engine.")
            return
        first_logic = logic_keys[0]
        result = results[first_logic]

        report_json = result.get("report_json")
        if not report_json or not os.path.exists(report_json):
            QMessageBox.warning(self, "Error", "Walkforward report JSON not found.")
            return

        with open(report_json) as f:
            report_data = json.load(f)

        forward_metrics = report_data.get("out-sample", {})
        sharpe_diff = report_data.get("sharpe_ratio_diff_pct", 0.0)

        # Ensure columns exist
        headers = [
            self.permutationListTable.horizontalHeaderItem(c).text().strip()
            for c in range(self.permutationListTable.columnCount())
        ]
        def ensure_column(name):
            if name not in headers:
                idx = self.permutationListTable.columnCount()
                self.permutationListTable.insertColumn(idx)
                self.permutationListTable.setHorizontalHeaderItem(idx, QTableWidgetItem(name))
                headers.append(name)
            return headers.index(name)

        col_sharpe = ensure_column("OS SR")
        col_mdd    = ensure_column("OS MDD")
        col_diff   = ensure_column("Sharpe Ratio Diff %")

        # Out‑sample SR (fallback SR)
        os_sr = forward_metrics.get("sharpe_ratio", forward_metrics.get("SR", ""))
        try: os_sr = f"{float(os_sr):.2f}"
        except: pass
        self.permutationListTable.setItem(row, col_sharpe, QTableWidgetItem(os_sr))

        # Out‑sample MDD (fallback MDD)
        os_mdd = forward_metrics.get("max_drawdown", forward_metrics.get("MDD", ""))
        try: os_mdd = f"{float(os_mdd)*100:.2f}%"
        except: pass
        self.permutationListTable.setItem(row, col_mdd, QTableWidgetItem(os_mdd))

        # Sharpe Diff %
        diff_item = QTableWidgetItem(f"{float(sharpe_diff):.2f}%")
        self.permutationListTable.setItem(row, col_diff, diff_item)
        
    def apply_parameters_from_permutation(self, row):
        # 1) Symbol
        symbol_text = self.permutationListTable.item(row, 0).text()
        si = self.symbolComboBox.findText(symbol_text)
        if si >= 0:
            self.symbolComboBox.setCurrentIndex(si)

        # 2) Interval
        interval_text = self.permutationListTable.item(row, 1).text()
        ii = self.intervalComboBox.findText(interval_text)
        if ii >= 0:
            self.intervalComboBox.setCurrentIndex(ii)

        # 3) Model
        model_text = self.permutationListTable.item(row, 4).text()
        mi = self.modelComboBox.findText(model_text)
        if mi >= 0:
            self.modelComboBox.blockSignals(True)
            self.modelComboBox.setCurrentIndex(mi)
            self.modelComboBox.blockSignals(False)

        # Rebuild dependent UI
        self.update_model_parameters()
        self.update_factor_options()

        # 4) Raw factor/formula text
        raw_factor = self.permutationListTable.item(row, 2).text().strip()
        chosen = None

        # 5) Normalize 'm'/'d' in formulas
        tokens = re.findall(r"(df\[['\"][^'\"]+['\"]\])", raw_factor)
        if len(tokens) > 1 and re.search(r"\b[md]\b", raw_factor):
            rebuilt = raw_factor.replace(" m ", " * ").replace(" d ", " / ")
            chosen = rebuilt
            # persist if new
            with open(FORMULAS_CSV, "r") as f:
                existing = set(f.read().splitlines())
            if rebuilt not in existing:
                try:
                    with open(FORMULAS_CSV, "a", newline="") as f:
                        f.write("\n" + rebuilt)
                    self._build_formula_map()
                except Exception as e:
                    QMessageBox.warning(self, "Formula Save Error",
                                        f"Could not persist new formula:\n{e}")

        # 6) Otherwise try formula_map
        if not chosen:
            if raw_factor in self.formula_map:
                chosen = raw_factor
            else:
                safe = sanitize_for_folder(raw_factor)
                chosen = self.formula_map.get(safe)

        # 7) Determine mode: formula if chosen, else factor
        is_formula = bool(chosen)

        # 8) Force SINGLE mode always on load
        if not self.singleRadio.isChecked():
            self.singleRadio.blockSignals(True)
            self.singleRadio.setChecked(True)
            self.rangeRadio.setChecked(False)
            self.singleRadio.blockSignals(False)
            self.update_model_parameters()  # rebuild params as single

        # 9) Toggle selection mode
        self.selectionToggleButton.blockSignals(True)
        self.selectionToggleButton.setChecked(is_formula)
        self.selectionToggleButton.blockSignals(False)
        self.toggle_selection_mode(is_formula)

        # 10) Select in dropdown
        idx = -1
        if is_formula:
            idx = self.selectionComboBox.findText(chosen or "")
            if idx < 0:
                for j in range(self.selectionComboBox.count()):
                    txt = self.selectionComboBox.itemText(j)
                    if txt == chosen or txt == raw_factor:
                        idx = j
                        break
        else:
            for j in range(self.selectionComboBox.count()):
                item = self.selectionComboBox.itemText(j)
                base = item.split("(", 1)[0].strip()
                if base == raw_factor:
                    idx = j
                    break

        if idx >= 0:
            self.selectionComboBox.setCurrentIndex(idx)
        else:
            print(f"[Warning] could not restore selection: {raw_factor}")

        # 11) Transformation
        trans_text = self.permutationListTable.item(row, 3).text()
        ti = self.transformationComboBox.findText(trans_text)
        if ti >= 0:
            self.transformationComboBox.setCurrentIndex(ti)

        # 12) Logic
        logic_text = self.permutationListTable.item(row, 5).text()
        li = self.logicComboBox.findText(logic_text)
        if li >= 0:
            self.logicComboBox.blockSignals(True)
            self.logicComboBox.setCurrentIndex(li)
            self.logicComboBox.blockSignals(False)

        # 13) Other parameters (single mode only)
        headers = [
            self.permutationListTable.horizontalHeaderItem(c).text().strip()
            for c in range(self.permutationListTable.columnCount())
        ]
        if "Parameters" in headers:
            pcol = headers.index("Parameters")
            params_text = self.permutationListTable.item(row, pcol).text()
            parts = [p.strip() for p in params_text.split("|") if ":" in p]
            pdict = {k.strip(): v.strip() for k, v in (p.split(":", 1) for p in parts)}

            for key, widget in self.paramInputs.items():
                # in single mode, widget is always QLineEdit
                if hasattr(widget, "setText") and key in pdict:
                    widget.setText(pdict[key])

    def update_run_buttons(self):
        if not hasattr(self, "runPermutationButton"):
            return
        if self.singleRadio.isChecked():
            self.runPermutationButton.setEnabled(False)
            self.runBacktestButton.setEnabled(True)
            self.runWalkforwardButton.setEnabled(True)
        else:
            self.runPermutationButton.setEnabled(True)
            self.runBacktestButton.setEnabled(False)
            self.runWalkforwardButton.setEnabled(False)
    
    def add_filter(self):
        selected_metric = self.metricsFilterComboBox.currentText()
        benchmark_text = self.metricsFilterLineEdit.text().strip()
        operator = self.operatorComboBox.currentText()
        if not selected_metric or not benchmark_text:
            return
        try:
            benchmark_val = float(benchmark_text)
        except ValueError:
            QMessageBox.warning(self, "Error", "Benchmark value must be numeric.")
            return
        filter_obj = {"metric": selected_metric, "operator": operator, "benchmark": benchmark_val}
        self.activeFilters.append(filter_obj)
        chip = QWidget()
        chip_layout = QHBoxLayout(chip)
        chip_layout.setContentsMargins(5, 2, 5, 2)
        label = QLabel(f"{selected_metric} {operator} {benchmark_text}")
        close_btn = QPushButton("x")
        close_btn.setFixedSize(16, 16)
        close_btn.clicked.connect(lambda _, chip=chip, filt=filter_obj: self.remove_filter_chip(chip, filt))
        chip_layout.addWidget(label)
        chip_layout.addWidget(close_btn)
        self.activeFiltersLayout.insertWidget(self.activeFiltersLayout.count() - 1, chip)
        self.metricsFilterLineEdit.clear()
        self.apply_filters()
    
    def remove_filter_chip(self, chip, filt):
        self.activeFiltersLayout.removeWidget(chip)
        chip.deleteLater()
        if filt in self.activeFilters:
            self.activeFilters.remove(filt)
        self.apply_filters()
    
    def apply_filters(self):
        headers = [
            self.permutationListTable.horizontalHeaderItem(c).text()
            for c in range(self.permutationListTable.columnCount())
        ]

        for row in range(self.permutationListTable.rowCount()):
            row_visible = True

            for filt in self.activeFilters:
                metric = filt["metric"]
                operator = filt["operator"]
                benchmark = filt["benchmark"]

                try:
                    col_index = headers.index(metric)
                except ValueError:
                    continue

                item = self.permutationListTable.item(row, col_index)
                cell_text = item.text() if item else ""

                try:
                    cell_val = float(cell_text.replace("%", "").strip())
                except ValueError:
                    row_visible = False
                    break

                if operator == ">=" and cell_val < benchmark:
                    row_visible = False
                    break
                elif operator == "<=" and cell_val > benchmark:
                    row_visible = False
                    break

            self.permutationListTable.setRowHidden(row, not row_visible)


    def calculate_signal(self, df, rolling_window, threshold, factor_column):
        rolling_window = int(rolling_window)
        if rolling_window <= 0:
            raise ValueError("rolling_window must be a positive integer")
    
    def update_permutation_list_table(self, trades_data):
        metric_keys = set()
        for entry in trades_data:
            if "metrics" in entry:
                metric_keys.update(entry["metrics"].keys())
        default_filter_metrics = [
            "sharpe_ratio",
            "max_drawdown",
            "trades_per_interval",
            "win_rate",
            "total_returns",
            "profit_factor",
            "annualized_avg_return"
        ]
        if not metric_keys:
            metric_keys = set(default_filter_metrics)
        metric_keys = sorted(metric_keys)
        self.current_metric_keys = metric_keys[:]  
        self.metricsFilterComboBox.clear()
        self.metricsFilterComboBox.addItems(metric_keys)
    
        fixed_keys = ["symbol", "interval", "factor_column", "transformation", "model", "entry_exit_logic"]
        headers = fixed_keys + metric_keys + ["Parameters", "Sharpe Ratio Diff %"]
    
        rows = []
        for entry in trades_data:
            params = entry.get("params", {})
            symbol    = params.get("symbol", entry.get("symbol", ""))
            interval  = params.get("interval", entry.get("interval", ""))
            factor_val = entry.get("factor_column",
                          params.get("factor_column",
                          params.get("factor_columns", "")))
            if isinstance(factor_val, list):
                factor_val = "_".join(map(str, factor_val))
            transformation = entry.get("transformation", "None")
            model = params.get("model", entry.get("model", ""))
            entry_logic = (
                params.get("entry_exit_logic") or params.get("entryexit_logic") or 
                entry.get("entry_exit_logic") or entry.get("entryexit_logic", "")
            )
            row = [str(symbol), str(interval), str(factor_val), str(transformation), str(model), str(entry_logic)]
            for mkey in metric_keys:
                val = entry.get("metrics", {}).get(mkey, "")
                formatted = str(val)
                row.append(formatted)
            other_params = []
            for key, value in params.items():
                if key in fixed_keys \
                   or key in {"entry_exit_logic", "entryexit_logic", "factor_column", "factor_columns", "backtest_period"}:
                    continue

                try:
                    formatted_value = f"{float(value):.2f}"
                except (ValueError, TypeError):
                    formatted_value = str(value)
                other_params.append(f"{key}: {formatted_value}")
            row.append(" | ".join(other_params))
            row.append("")
            rows.append(row)
    
        self.permutationListTable.clear()
        self.permutationListTable.setRowCount(0)
        self.permutationListTable.setColumnCount(len(headers))
        self.permutationListTable.setHorizontalHeaderLabels(headers)
        for i, row in enumerate(rows):
            self.permutationListTable.insertRow(i)
            for j, cell in enumerate(row):
                self.permutationListTable.setItem(i, j, QTableWidgetItem(cell))
        self.permutationListTable.resizeColumnsToContents()
        self.permutationListTable.resizeRowsToContents()
        self.apply_filters()

    def run_permutation(self):
        self.runPermutationButton.setEnabled(False)
        symbol       = self.symbolComboBox.currentText()
        model_choice = self.modelComboBox.currentText()
        interval     = self.intervalComboBox.currentText()

        # and one more time, refresh factors
        self.update_factor_options()

        calculate_signal, model_config = ModelLoader().load_model(model_choice)
        config = ConfigManager("config/config.yaml").config
        data_loader = DataLoader(
            candle_path=config["data"]["candle_path"],
            datasource_path=config["data"]["datasource_path"],
            config=config,
            debug=True
        )
        reporter = Reporter()

        if getattr(self, "selected_formula", None):
            formula_name = self.selected_formula
            factor_list  = []
        else:
            formula_name = ""
            factor_list  = [self.current_factor]

        try:
            df, factor_column_str, shift_val, datasource_files = load_factor_dataframe(
                data_loader,
                config,
                self.ds_map,
                symbol,
                interval,
                formula_name,
                factor_list
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            self.runPermutationButton.setEnabled(True)
            return

        logic_choice = self.logicComboBox.currentText()
        logic_plugins = PluginLoader().load_logic_plugins()
        logic_plugin = logic_plugins.get(logic_choice)
        logic_config = getattr(type(logic_plugin), "LOGIC_CONFIG", {})

        if not logic_choice:
            QMessageBox.warning(self, "Error", "No entry/exit logic selected.")
            self.runPermutationButton.setEnabled(True)
            return

        strategy_params = {"factor_column": factor_column_str}
        permutation_params = {}
        if self.singleRadio.isChecked():
            for key, widget in self.paramInputs.items():
                val = widget.text().strip()
                default = model_config.get(key, {}).get("default")
                try:
                    if isinstance(default, int):
                        strategy_params[key] = int(val)
                    elif isinstance(default, float):
                        strategy_params[key] = float(val)
                    else:
                        strategy_params[key] = val
                except Exception:
                    strategy_params[key] = val
        else:
            full_config = {}
            full_config.update(model_config)
            full_config.update(logic_config)

            for key, fields in self.paramInputs.items():
                try:
                    default = full_config[key]["default"]
                    mn = float(fields["min"].text())
                    mx = float(fields["max"].text())
                    st = float(fields["step"].text())
                    if st <= 0 or mx < mn:
                        vals = []
                    else:
                        if isinstance(default, int):
                            vals = list(range(int(mn), int(mx) + 1, int(st)))
                        else:
                            vals = list(np.arange(mn, mx + st/2, st))
                    permutation_params[key] = vals
                    strategy_params[key] = vals[0] if vals else default
                    print(f"Parameter {key}: min={mn}, max={mx}, step={st}, values={vals}")
                except Exception as e:
                    permutation_params[key] = []
                    strategy_params[key] = model_config.get(key, {}).get("default")
                    print(f"Error processing parameter {key}: {e}")

        print("Strategy Params:", strategy_params)
        print("Permutation Params:", permutation_params)
        
        params = {
            "model": model_choice,
            "entryexit_logic": logic_choice,
            "strategy_params": strategy_params,
            "permutation_params": permutation_params,
            "symbol": symbol,
            "interval": interval,
            "shift_override": shift_val,
            "transformation": self.transformationComboBox.currentText(),
            "report_mode": "permutation"
        }
        params["logic_config"] = logic_config
        print("Final params passed to Permutation Engine:", params)

        report_dir = reporter.create_detailed_report_directory(
            "permutation", symbol, interval, factor_column_str, model_choice, logic_choice, self.transformationComboBox.currentText()
        )
        self.report_dir = report_dir

        logic_plugins = PluginLoader().load_logic_plugins()
        self.permutationWorker = MultiprocessingPermutationWorker(
            calculate_signal, df, params, {logic_choice: logic_plugins[logic_choice]}, reporter
        )
        def handle_with_dir(result):
            result["report_dir"] = report_dir
            self.handle_permutation_results(result)
        self.permutationWorker.finished.connect(handle_with_dir)
        self.permutationWorker.start()

    def handle_permutation_results(self, result):
        # get the report directory
        report_dir = result.get("report_dir") or getattr(self, "report_dir", None)
        if not report_dir:
            # nothing to do if there’s no report
            self.heatmapScene.clear()
            self.heatmapScene.addText("Permutation report not found.")
            self.runPermutationButton.setEnabled(True)
            return

        self.heatmapScene.clear()
        heatmap_path = os.path.join(report_dir, "heatmap.png")
        if os.path.exists(heatmap_path):
            pixmap = QPixmap(heatmap_path)
            if not pixmap.isNull():
                item = QGraphicsPixmapItem(pixmap)
                self.heatmapScene.addItem(item)
                self.heatmapView.fitInView(item, Qt.KeepAspectRatio)
            else:
                self.heatmapScene.addText("Failed to load heatmap image.")
        else:
            self.heatmapScene.addText("Heatmap image not found.")

        json_path = os.path.join(report_dir, "permutation_report_all.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                trades_data = json.load(f)
            self.update_permutation_list_table(trades_data)
        else:
            QMessageBox.warning(self, "Error", "Permutation JSON report not found.")

        self.runPermutationButton.setEnabled(True)
        for idx in range(self.chartTabWidget.count()):
            if self.chartTabWidget.tabText(idx) == "Permutation":
                self.chartTabWidget.setCurrentIndex(idx)
                break

    def import_permutation(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Permutation Export Folder")
        if folder:
            self.heatmapScene.clear()
            self.heatmapScene.addText("Loading heatmap and report...")
            self.loaderThread = PermutationLoaderThread(folder)
            self.loaderThread.loaded.connect(self.on_permutation_loaded)
            self.loaderThread.start()

    def on_permutation_loaded(self, json_data, pixmap):
        self.heatmapScene.clear()
        if pixmap and not pixmap.isNull():
            pixmapItem = QGraphicsPixmapItem(pixmap)
            self.heatmapScene.addItem(pixmapItem)
            self.heatmapView.fitInView(pixmapItem, Qt.KeepAspectRatio)
        else:
            self.heatmapScene.addText("Failed to load heatmap image.")
        if json_data:
            try:
                self.update_permutation_list_table(json_data)
            except Exception as e:
                QMessageBox.warning(self, "Error",
                                    f"Error loading permutation JSON: {e}")
        else:
            QMessageBox.warning(self, "Error",
                                "Permutation JSON report not found or failed to load.")
        for idx in range(self.chartTabWidget.count()):
            if self.chartTabWidget.tabText(idx) == "Permutation":
                self.chartTabWidget.setCurrentIndex(idx)
                break
        self.importPermutationButton.setEnabled(True)

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def update_model_parameters(self):
        # determine new vs old mode
        new_mode = "single" if self.singleRadio.isChecked() else "range"
        old_mode = getattr(self, 'current_param_mode', new_mode)

        # ── SAVE EXISTING FACTOR SELECTION(S) ───────────────────────────────────────
        prev_factor_selections = [cb.currentText() for cb in self.factorComboBoxes]

        # 1) save out whatever was in the old widgets
        for key, widget in self.paramInputs.items():
            if old_mode == 'single':
                self.saved_params['single'][key] = widget.text()
            else:
                self.saved_params['range'][key] = {
                    'min': widget['min'].text(),
                    'max': widget['max'].text(),
                    'step': widget['step'].text()
                }

        # 2) clear out the form
        self.current_param_mode = new_mode
        self.clear_layout(self.paramInputWidget.layout())
        self.paramInputs.clear()

        # 3) load model + logic config
        model_choice = self.modelComboBox.currentText()
        if not model_choice:
            return
        calculate_signal, model_config = ModelLoader().load_model(model_choice)
        logic_choice = self.logicComboBox.currentText()
        logic_plugins = PluginLoader().load_logic_plugins()
        logic_plugin = logic_plugins.get(logic_choice)
        logic_config = getattr(type(logic_plugin), "LOGIC_CONFIG", {}) if logic_plugin else {}
        full_config = {}
        full_config.update(model_config)
        full_config.update(logic_config)

        # ── REBUILD FACTOR UI ─────────────────────────────────────────────────────
        self.clear_layout(self.factorsLayout)
        self.factorComboBoxes = []
        factorWidget = QWidget()
        factorLayout = QHBoxLayout(factorWidget)
        factorLayout.setContentsMargins(0, 0, 0, 0)
        factorLayout.setSpacing(5)
        cb = QComboBox()
        cb.setEditable(True)
        completer = QCompleter(cb.model(), cb)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setFilterMode(Qt.MatchContains)
        cb.setCompleter(completer)
        factorLayout.addWidget(cb)
        self.factorsLayout.addWidget(factorWidget)
        self.factorComboBoxes.append(cb)

        # repopulate and (by default) select the previous choice if still available
        self.update_factor_options()

        if prev_factor_selections:
            saved = prev_factor_selections[0]
            options = [cb.itemText(i) for i in range(cb.count())]
            if saved in options:
                idx = options.index(saved)
                cb.blockSignals(True)
                cb.setCurrentIndex(idx)
                cb.blockSignals(False)

        # ── BUILD PARAMETER INPUTS ─────────────────────────────────────────────────
        if new_mode == "single":
            self.totalLabel.hide()
            for key, info in full_config.items():
                if not (isinstance(info, dict) and "default" in info):
                    continue
                default_val = info["default"]
                lineEdit = QLineEdit()
                lineEdit.setText(self.saved_params['single'].get(key, str(default_val)))
                self.paramForm.addRow(f"{key}:", lineEdit)
                self.paramInputs[key] = lineEdit

        else:  # range mode
            self.totalLabel.show()
            for key, info in full_config.items():
                if not (isinstance(info, dict) and "default" in info):
                    continue
                default_val = info["default"]

                minEdit = QLineEdit()
                maxEdit = QLineEdit()
                stepEdit = QLineEdit()

                saved = self.saved_params['range'].get(key, {})
                minEdit.setText(saved.get('min', str(default_val)))
                maxEdit.setText(saved.get('max', str(default_val)))
                stepEdit.setText(saved.get('step', '1'))

                rangeWidget = QWidget()
                hLayout = QHBoxLayout(rangeWidget)
                hLayout.setContentsMargins(0, 0, 0, 0)
                hLayout.addWidget(QLabel("Min:"));  hLayout.addWidget(minEdit)
                hLayout.addWidget(QLabel("Max:"));  hLayout.addWidget(maxEdit)
                hLayout.addWidget(QLabel("Step:")); hLayout.addWidget(stepEdit)

                self.paramForm.addRow(f"{key}:", rangeWidget)
                self.paramInputs[key] = {'min': minEdit, 'max': maxEdit, 'step': stepEdit}

                for w in (minEdit, maxEdit, stepEdit):
                    w.textChanged.connect(self.calculate_total_combinations)

            self.calculate_total_combinations()


    def calculate_total_combinations(self):
        total = 1
        for key, fields in self.paramInputs.items():
            try:
                min_val = float(fields["min"].text())
                max_val = float(fields["max"].text())
                step_val = float(fields["step"].text())
                if step_val <= 0 or max_val < min_val:
                    count = 0
                else:
                    count = int((max_val - min_val) // step_val) + 1
            except Exception:
                count = 0
            total *= count
        self.totalLabel.setText(f"Total combinations: {total}")

    def toggle_selection_mode(self, checked):
        self.selectionToggleButton.setText("Use Factors" if checked else "Use Formula")
        self.selectionComboBox.blockSignals(True)
        self.selectionComboBox.clear()
        if checked:
            valid = []
            for raw in load_formulas(str(FORMULAS_CSV)):
                try:
                    tokens = parse_formula(raw)
                    if all(any(ds_key.split("(")[0].strip() == tok for ds_key in self.ds_map)
                           for tok in tokens):
                        valid.append(raw)
                except:
                    continue
            self.selectionComboBox.addItems(valid)
        else:
            self.selectionComboBox.addItems(self.factorOptions)
        self.selectionComboBox.blockSignals(False)
        self.update_factor_options()
        self.on_selection_changed(self.selectionComboBox.currentIndex())

    def on_selection_changed(self, index):
        txt = self.selectionComboBox.currentText()
        if self.selectionToggleButton.isChecked():
            self.selected_formula = None if txt == "<none>" else txt
        else:
            self.selected_formula = None
            self.current_factor = txt
            

    def load_available_options(self):
        config_obj = ConfigManager("config/config.yaml")
        self.config = config_obj.config

        candle_path = self.config["data"]["candle_path"]
        candle_files = [f for f in os.listdir(candle_path)
                        if f.endswith(".parquet") or f.endswith(".csv")]
        availableSymbols = sorted(list({f.split("_")[0] for f in candle_files}))
        self.symbolComboBox.clear()
        self.symbolComboBox.addItems(availableSymbols)
        model_loader = ModelLoader()
        availableModels = model_loader.discover_models()
        self.modelComboBox.clear()
        self.modelComboBox.addItems(availableModels)

        if candle_files:
            selected_candle_file = candle_files[0]
            candle_native_interval = extract_interval_from_filename(selected_candle_file)
            if candle_native_interval is None:
                candle_native_interval = "1min"
        else:
            candle_native_interval = "1min"
        datasource_path = self.config["data"]["datasource_path"]
        factor_options, ds_map, ds_native_intervals = aggregate_factor_options(datasource_path)
        self.factorOptions = factor_options[:]
        self.original_ds_map = ds_map.copy()
        self.ds_map = ds_map.copy()
        self.all_factors = self.factorOptions[:]
        self.selectionToggleButton.setChecked(False)
        self.selectionComboBox.clear()
        self.selectionComboBox.addItems(self.all_factors)

        
        if ds_native_intervals:
            datasource_native_interval = format_interval(min(ds_native_intervals))
        else:
            datasource_native_interval = "1h"
        availableIntervals = get_common_intervals(candle_native_interval, datasource_native_interval)
        self.intervalComboBox.clear()
        self.intervalComboBox.addItems(availableIntervals)

        
        plugin_loader = PluginLoader()
        logic_plugins = plugin_loader.load_logic_plugins()
        available_logics = list(logic_plugins.keys())
        self.all_formulas = load_formulas(str(FORMULAS_CSV))
        self.all_factors = self.factorOptions[:]
        self.selectionToggleButton.setChecked(False)
        self.selectionComboBox.clear()
        self.selectionComboBox.addItems(self.all_factors)

        self.logicComboBox.addItems(available_logics)
        self.update_model_parameters()
    
    def export_selected_config(self, export_dir, mode="backtest"):
        config_data = {
            "symbol": self.symbolComboBox.currentText(),
            "model": self.modelComboBox.currentText(),
            "interval": self.intervalComboBox.currentText(),
            "transformation": self.transformationComboBox.currentText(),
            "logic": self.logicComboBox.currentText(),
            "parameter_mode": "single" if self.singleRadio.isChecked() else "range",
            "parameters": {}
        }
        
        # If a formula is selected (not "<none>"), export that instead of factor selections.
        selected_formula = self.formulaComboBox.currentText()
        if selected_formula and selected_formula != "<none>":
            config_data["formula"] = selected_formula
            config_data["factors"] = []
        else:
            config_data["factors"] = [cb.currentText() for cb in self.factorComboBoxes]
        
        if self.singleRadio.isChecked():
            for key, widget in self.paramInputs.items():
                config_data["parameters"][key] = widget.text().strip()
        else:
            for key, fields in self.paramInputs.items():
                config_data["parameters"][key] = fields["min"].text().strip()
        
        if mode == "walkforward":
            if hasattr(self, "config"):
                config_data["train_ratio"] = self.config.get("train_ratio", 0.65)
            export_filename = "selected_walkforward_config.json"
        else:
            export_filename = "selected_backtest_config.json"
        
        export_path = os.path.join(export_dir, export_filename)
        try:
            with open(export_path, "w") as f:
                json.dump(config_data, f, indent=4)
            print(f"Exported selected configuration to {export_path}")
        except Exception as e:
            print("Error exporting configuration:", e)

    def get_selected_config(self, mode="backtest"):
        config_data = {
            "symbol": self.symbolComboBox.currentText(),
            "model": self.modelComboBox.currentText(),
            "interval": self.intervalComboBox.currentText(),
            "transformation": self.transformationComboBox.currentText(),
            "logic": self.logicComboBox.currentText(),
            "parameter_mode": "single" if self.singleRadio.isChecked() else "range",
            "parameters": {}
        }
        selected_formula = self.formulaComboBox.currentText()
        if selected_formula and selected_formula != "<none>":
            config_data["formula"] = selected_formula
            config_data["factors"] = []
        else:
            config_data["factors"] = [cb.currentText() for cb in self.factorComboBoxes]
        if self.singleRadio.isChecked():
            for key, widget in self.paramInputs.items():
                config_data["parameters"][key] = widget.text().strip()
        else:
            for key, fields in self.paramInputs.items():
                config_data["parameters"][key] = fields["min"].text().strip()

            if mode == "walkforward" and hasattr(self, "config"):
                config_data["train_ratio"] = self.config.get("train_ratio", 0.65)

            return config_data

    def run_backtest(self):
        self.runBacktestButton.setEnabled(False)
        symbol       = self.symbolComboBox.currentText()
        model_choice = self.modelComboBox.currentText()
        interval     = self.intervalComboBox.currentText()

        # make sure our factor‐drop‐downs are up‐to‐date
        self.update_factor_options()

        calculate_signal, model_config = ModelLoader().load_model(model_choice)
        config = ConfigManager("config/config.yaml").config
        data_loader = DataLoader(
            candle_path=config["data"]["candle_path"],
            datasource_path=config["data"]["datasource_path"],
            config=config,
            debug=True
        )
        reporter = Reporter()

        if getattr(self, "selected_formula", None):
            formula_name = self.selected_formula
            factor_list  = []
        else:
            formula_name = ""
            factor_list  = [self.current_factor]

        try:
            df, factor_column_str, shift_val, datasource_files = load_factor_dataframe(
                data_loader,
                config,
                self.ds_map,
                symbol,
                interval,
                formula_name,
                factor_list
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            self.runBacktestButton.setEnabled(True)
            return

        logic_choice = self.logicComboBox.currentText()
        logic_plugins = PluginLoader().load_logic_plugins()
        logic_plugin = logic_plugins.get(logic_choice)
        logic_config = getattr(type(logic_plugin), "LOGIC_CONFIG", {})

        if not logic_choice:
            QMessageBox.warning(self, "Error", "No entry/exit logic selected.")
            self.runBacktestButton.setEnabled(True)
            return

        strategy_params = {"factor_column": factor_column_str}
        if self.singleRadio.isChecked():
            for key, widget in self.paramInputs.items():
                val = widget.text().strip()
                default = model_config.get(key, {}).get("default")
                try:
                    if key == "rolling_window":
                        strategy_params[key] = int(float(val))
                    elif isinstance(default, int):
                        strategy_params[key] = int(val)
                    elif isinstance(default, float):
                        strategy_params[key] = float(val)
                    else:
                        strategy_params[key] = val
                except:
                    strategy_params[key] = val

        params = {
            "model": model_choice,
            "entryexit_logic": logic_choice,
            "strategy_params": strategy_params,
            "permutation_params": {},
            "symbol": symbol,
            "interval": interval,
            "shift_override": shift_val,
            "transformation": self.transformationComboBox.currentText(),
        }
        params["datasource_files"] = datasource_files
        params["logic_config"] = logic_config
        config_data = self.get_selected_config(mode="backtest")
        params["selected_config"] = config_data
        params["parameter_mode"] = "single" if self.singleRadio.isChecked() else "range"

        logic_plugins = PluginLoader().load_logic_plugins()
        reporter = Reporter()
        self.backtestWorker = MultiprocessingBacktestWorker(
            calculate_signal, df, params, logic_plugins, reporter
        )
        self.backtestWorker.finished.connect(self.handle_backtest_results)
        self.backtestWorker.start()

    def append_log(self, message):
        print(message)

    def show_header_context_menu(self, pos):
        header = self.tradesTable.horizontalHeader()
        col_index = header.logicalIndexAt(pos)
        if col_index < 0:
            return
        column_name = self.tradesTable.horizontalHeaderItem(col_index).text()
        if hasattr(self, 'full_trades_df'):
            try:
                col_series = self.full_trades_df[column_name]
                if np.issubdtype(col_series.dtype, np.number):
                    col_sum = col_series.sum()
                else:
                    col_sum = "N/A"
                stats = col_series.describe()
                stats_text = stats.to_string()
                stats_text += f"\nSum: {col_sum}"
                QMessageBox.information(self, "Summary Statistics",
                                        f"Statistics for column '{column_name}':\n{stats_text}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error calculating stats: {e}")
        else:
            QMessageBox.warning(self, "Error", "Full dataset is not loaded.")
        
    def run_walkforward(self):
        self.runWalkforwardButton.setEnabled(False)
        symbol       = self.symbolComboBox.currentText()
        model_choice = self.modelComboBox.currentText()
        interval     = self.intervalComboBox.currentText()

        # refresh the factor lists
        self.update_factor_options()

        calculate_signal, model_config = ModelLoader().load_model(model_choice)
        config = ConfigManager("config/config.yaml").config
        data_loader = DataLoader(
            candle_path=config["data"]["candle_path"],
            datasource_path=config["data"]["datasource_path"],
            config=config,
            debug=True
        )
        reporter = Reporter()

        if getattr(self, "selected_formula", None):
            formula_name = self.selected_formula
            factor_list  = []
        else:
            formula_name = ""
            factor_list  = [self.current_factor]

        try:
            df, factor_column_str, shift_val, datasource_files = load_factor_dataframe(
                data_loader,
                config,
                self.ds_map,
                symbol,
                interval,
                formula_name,
                factor_list
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            self.runWalkforwardButton.setEnabled(True)
            return

        logic_choice = self.logicComboBox.currentText()
        logic_plugins = PluginLoader().load_logic_plugins()
        logic_plugin = logic_plugins.get(logic_choice)
        logic_config = getattr(type(logic_plugin), "LOGIC_CONFIG", {})

        if not logic_choice:
            QMessageBox.warning(self, "Error", "No entry/exit logic selected.")
            self.runWalkforwardButton.setEnabled(True)
            return


        strategy_params = {"factor_column": factor_column_str}
        if self.singleRadio.isChecked():
            for key, widget in self.paramInputs.items():
                val = widget.text().strip()
                default = model_config.get(key, {}).get("default")
                try:
                    if key == "rolling_window":
                        strategy_params[key] = int(float(val))
                    elif isinstance(default, int):
                        strategy_params[key] = int(val)
                    elif isinstance(default, float):
                        strategy_params[key] = float(val)
                    else:
                        strategy_params[key] = val
                except:
                    strategy_params[key] = val

        params = {
            "model": model_choice,
            "entryexit_logic": logic_choice,
            "strategy_params": strategy_params,
            "date_column": "datetime",
            "train_ratio": config.get("train_ratio", 0.65),
            "symbol": symbol,
            "interval": interval,
            "shift_override": shift_val,
            "transformation": self.transformationComboBox.currentText(),
        }
        params["datasource_files"] = datasource_files
        params["logic_config"] = logic_config
        config_data = self.get_selected_config(mode="walkforward")
        params["selected_config"] = config_data
        params["parameter_mode"] = "single" if self.singleRadio.isChecked() else "range"
        
        reporter = Reporter()
        logic_plugins = PluginLoader().load_logic_plugins()
        self.walkforwardWorker = MultiprocessingWalkforwardWorker(calculate_signal, df, params, logic_plugins, reporter)
        self.walkforwardWorker.finished.connect(self.handle_walkforward_results)
        self.walkforwardWorker.start()

    def reset_chart_view(self):
        self.plotWidget.enableAutoRange(axis=pg.ViewBox.XYAxes)
        self.plotWidget.autoRange()

    def handle_backtest_results(self, results):
        QApplication.beep()
        self.runBacktestButton.setEnabled(True)
        logic_keys = list(results.keys())
        if not logic_keys:
            QMessageBox.warning(self, "Error", "No results returned from the backtest engine.")
            return
        first_logic = logic_keys[0]
        result = results[first_logic]
        output_csv = result.get("output_csv")
        if not output_csv or not os.path.exists(output_csv):
            QMessageBox.warning(self, "Error", "Output CSV not found.")
            return

        backtest_df = pd.read_csv(output_csv)
        self.last_backtest_csv = output_csv
        if "datetime" not in backtest_df.columns or "cumpnl" not in backtest_df.columns:
            QMessageBox.warning(self, "Error", "Required columns (datetime, cumpnl) not found in CSV.")
            return

        # 4) Plot the equity curve
        self.plotWidget.clear()
        backtest_df["datetime"] = pd.to_datetime(backtest_df["datetime"])
        x = backtest_df["datetime"].astype("int64") // 10**9
        y = backtest_df["cumpnl"]
        # use a gradient fill for the curve
        from PyQt5.QtGui import QLinearGradient, QColor, QBrush
        from PyQt5.QtCore import Qt
        gradient = QLinearGradient(0, 0, 0, 1)
        gradient.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
        gradient.setColorAt(0.0, QColor(254, 116, 79, 153))
        gradient.setColorAt(1.0, QColor(254, 116, 79, 17))
        brush = QBrush(gradient)
        import pyqtgraph as pg
        self.plotWidget.plot(
            x, y,
            pen=pg.mkPen('#fe744f'),
            fillLevel=0,
            brush=brush,
            name="Equity Curve"
        )
        self.plotWidget.setLabel('left', "Cumulative PnL")
        self.plotWidget.setLabel('bottom', "Datetime")

        # 5) Populate the Trades table (show first 1000 rows)
        self.full_trades_df = backtest_df

        self.dataVizList.clear()
        for col in backtest_df.columns:
            if col == "datetime":
                continue
            self.dataVizList.addItem(col)
        trades_df_display = backtest_df.head(1000)
        self.update_table_widget(self.tradesTable, trades_df_display)

        # 6) Populate the Metrics table
        metrics = result.get("metrics", {})
        formatted = {}
        for key, val in metrics.items():
            lk = key.lower()
            if lk == "trades_per_interval":
                try:
                    formatted[key] = f"{float(val)*100:.2f}%"
                except:
                    formatted[key] = val
            elif lk == "num_of_trades":
                try:
                    formatted[key] = str(int(val))
                except:
                    formatted[key] = val
            else:
                try:
                    formatted[key] = f"{float(val):.2f}"
                except:
                    formatted[key] = val

        metrics_df = pd.DataFrame(list(formatted.items()), columns=["Metric", "Value"])
        self.update_table_widget(self.metricsTable, metrics_df)
        self.reset_chart_view()
        
    def handle_walkforward_results(self, results):

        logic_keys = list(results.keys())
        if not logic_keys:
            QMessageBox.warning(self, "Error", "No results returned from the walkforward engine.")
            return
        first_logic = logic_keys[0]
        result = results[first_logic]

        report_json = result.get("report_json")
        if not report_json or not os.path.exists(report_json):
            QMessageBox.warning(self, "Error", "Walkforward report JSON not found.")
            self.runWalkforwardButton.setEnabled(True)
            return

        with open(report_json) as f:
            report_data = json.load(f)

        backtest_metrics = report_data.get("in-sample", {})
        forward_metrics = report_data.get("out-sample", {})
        sharpe_diff = report_data.get("sharpe_ratio_diff_pct", 0.0)
        all_keys = sorted(set(backtest_metrics) | set(forward_metrics))
        rows = []
        for key in all_keys:
            in_raw = backtest_metrics.get(key, "")
            out_raw = forward_metrics.get(key, "")
            lk = key.lower()
            if lk == "num_of_trades":
                in_val = str(int(in_raw)) if in_raw != "" else ""
                out_val = str(int(out_raw)) if out_raw != "" else ""
            elif lk == "trades_per_interval" or any(tok in lk for tok in ("rate", "pct", "drawdown", "return")):
                try:
                    in_val = f"{float(in_raw) * 100:.2f}%"
                except:
                    in_val = in_raw
                try:
                    out_val = f"{float(out_raw) * 100:.2f}%"
                except:
                    out_val = out_raw
            else:
                try:
                    in_val = f"{float(in_raw):.4f}"
                except:
                    in_val = in_raw
                try:
                    out_val = f"{float(out_raw):.4f}"
                except:
                    out_val = out_raw
            rows.append({"Metric": key, "In Sample": in_val, "Out Sample": out_val})
        rows.append({
            "Metric": "Sharpe Ratio Diff %",
            "In Sample": "",
            "Out Sample": f"{float(sharpe_diff):.2f}%"
        })

        metrics_df = pd.DataFrame(rows)
        self.update_table_widget(self.metricsTable, metrics_df)

        if self.current_validation_row is not None:
            row = self.current_validation_row
            try:
                headers = [
                    self.validatorTable.horizontalHeaderItem(c).text().strip()
                    for c in range(self.validatorTable.columnCount())
                ]
                if "trade_numbers" in headers:
                    idx = headers.index("trade_numbers")
                    try:
                        expected = int(self.validatorTable.item(row, idx).text())
                    except:
                        expected = None
                    actual = backtest_metrics.get("num_of_trades")
                    actual = int(actual) if actual is not None else None
                    color = QColor(0,128,0) if expected == actual else QColor(128,0,0)
                    for c in range(self.validatorTable.columnCount()):
                        item = self.validatorTable.item(row, c)
                        if item:
                            item.setBackground(color)
            except Exception as e:
                print(f"[Validator highlight error]: {e}")
            finally:
                self.current_validation_row = None

        self.runWalkforwardButton.setEnabled(True)
        self.bottomTab.setCurrentWidget(self.metricsTab)
        headers = [self.permutationListTable.horizontalHeaderItem(c).text().strip()
                   for c in range(self.permutationListTable.columnCount())]

        def ensure_column(name):
            if name not in headers:
                idx = self.permutationListTable.columnCount()
                self.permutationListTable.insertColumn(idx)
                self.permutationListTable.setHorizontalHeaderItem(idx, QTableWidgetItem(name))
                headers.append(name)
            return headers.index(name)

        col_sharpe = ensure_column("OS SR")
        col_mdd = ensure_column("OS MDD")
        col_diff = ensure_column("Sharpe Ratio Diff %")

        row = self.current_permutation_row
        self.current_permutation_row = None
        if row is not None:
            os_sr = forward_metrics.get("sharpe_ratio", forward_metrics.get("SR", ""))
            try:
                os_sr = f"{float(os_sr):.2f}"
            except:
                pass
            self.permutationListTable.setItem(row, col_sharpe, QTableWidgetItem(os_sr))

            os_mdd = forward_metrics.get("max_drawdown", forward_metrics.get("MDD", ""))
            try:
                os_mdd = f"{float(os_mdd) * 100:.2f}%"
            except:
                pass
            self.permutationListTable.setItem(row, col_mdd, QTableWidgetItem(os_mdd))

            diff_item = QTableWidgetItem(f"{float(sharpe_diff):.2f}%")
            self.permutationListTable.setItem(row, col_diff, diff_item)

        if not getattr(self, 'skip_walkforward_plot', False):
            # Update Trades table using the out_sample CSV
            out_sample_csv = result.get("out_sample_csv")
            if out_sample_csv and os.path.exists(out_sample_csv):
                try:
                    trades_df = pd.read_csv(out_sample_csv)
                    self.update_table_widget(self.tradesTable, trades_df)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Error loading out-sample CSV: {e}")
            else:
                QMessageBox.warning(self, "Error", "Out-sample CSV not found.")

            in_sample_csv = result.get("in_sample_csv")
            if in_sample_csv and os.path.exists(in_sample_csv) and out_sample_csv and os.path.exists(out_sample_csv):
                self.update_walkforward_plot(in_sample_csv, out_sample_csv)
            else:
                print("In-sample or out-sample CSV not found for plotting.")
        else:
            print("Skipping trades table update and equity curve plot (triggered from permutation table).")

        if hasattr(self, '_wf_queue') and not self._wf_queue:
            self.skip_walkforward_plot = False

        self.runWalkforwardButton.setEnabled(True)
        self.bottomTab.setCurrentWidget(self.metricsTab)

    def update_walkforward_plot(self, in_sample_csv, out_sample_csv):
        train_df = pd.read_csv(in_sample_csv)
        test_df = pd.read_csv(out_sample_csv)
        train_df["datetime"] = pd.to_datetime(train_df["datetime"])
        test_df["datetime"] = pd.to_datetime(test_df["datetime"])
        x_train = train_df["datetime"].astype("int64") // 10**9
        x_test = test_df["datetime"].astype("int64") // 10**9
        y_train = train_df["pnl"].cumsum()
        y_test = test_df["pnl"].cumsum()
        offset = y_train.iloc[-1]
        y_test_offset = y_test + offset
        self.plotWidget.clear()
        self.plotWidget.setBackground('#020202')
        self.plotWidget.getAxis('left').setPen(pg.mkPen(color='#ffffff', width=2))
        self.plotWidget.getAxis('bottom').setPen(pg.mkPen(color='#ffffff', width=2))
        in_sample_pen = pg.mkPen('#fe744f', width=2) 
        out_sample_pen = pg.mkPen('#38b2ac', width=2)   
        self.plotWidget.plot(x_train, y_train, pen=in_sample_pen, name="In-Sample")
        self.plotWidget.plot(x_test, y_test_offset, pen=out_sample_pen, name="Out-Sample")
        self.plotWidget.addLegend()
        self.plotWidget.setLabel('left', "Cumulative PnL")
        self.plotWidget.setLabel('bottom', "Datetime")
       
    def update_table_widget(self, table_widget, df):
        table_widget.clear()
        table_widget.setRowCount(0)
        table_widget.setColumnCount(0)
        table_widget.setColumnCount(len(df.columns))
        table_widget.setHorizontalHeaderLabels(df.columns.tolist())
        for i, row in df.iterrows():
            table_widget.insertRow(i)
            for j, col in enumerate(df.columns):
                item = QTableWidgetItem(str(row[col]))
                table_widget.setItem(i, j, item)
        table_widget.resizeColumnsToContents()
        table_widget.resizeRowsToContents()

# ===================== Main Window =====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QLab")
        self.setGeometry(100, 100, 1000, 700)
        self.init_ui()
        self.create_shortcuts()

    def init_ui(self):
        self.tabWidget = QTabWidget()
        self.backtestTab = BacktestTab()
        self.bruteforceTab = BruteforceTab(parent=self)
        self.researchTab = ResearchTab(parent=self)
        self.tabWidget.addTab(self.backtestTab, "Backtest")
        self.tabWidget.addTab(self.bruteforceTab, "Bruteforce")
        self.tabWidget.addTab(self.researchTab,    "Research")
        self.setCentralWidget(self.tabWidget)

    def create_shortcuts(self):
        self.runBacktestShortcut = QShortcut(QKeySequence("Ctrl+B"), self)
        self.runBacktestShortcut.activated.connect(self.trigger_run_backtest)
        self.runWalkforwardShortcut = QShortcut(QKeySequence("Ctrl+W"), self)
        self.runWalkforwardShortcut.activated.connect(self.trigger_run_walkforward)

    def trigger_run_backtest(self):
        self.backtestTab.run_backtest()

    def trigger_run_walkforward(self):
        self.backtestTab.run_walkforward()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setFont(QFont("Arial", 11))
    theme_path = os.path.join(os.path.dirname(__file__), "theme.qss")
    if os.path.exists(theme_path):
        with open(theme_path, "r") as f:
            app.setStyleSheet(f.read())
    else:
        print("theme.qss file not found at:", theme_path)
    window = MainWindow()
    window.setUnifiedTitleAndToolBarOnMac(False)
    window.show()
    sys.exit(app.exec_())
