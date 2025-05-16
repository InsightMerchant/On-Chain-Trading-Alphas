import sys, os, json, time, itertools, random, re, io, gspread, pathlib
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QPushButton, QTabWidget, QMessageBox, QLabel, QTextEdit,
    QSplitter, QTableWidget, QTableWidgetItem, QSizePolicy, QGroupBox, QLineEdit,
    QGraphicsPixmapItem, QGraphicsView, QGraphicsScene, QListWidget, QListWidgetItem,
    QProgressBar, QAbstractItemView, QInputDialog, QCheckBox, QFileDialog, QSizePolicy,
    QRadioButton, QMenu, QShortcut,QStackedWidget, QCompleter, QStyleFactory, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QFileSystemWatcher
from PyQt5.QtGui import QPixmap, QPainter, QLinearGradient, QColor, QBrush, QKeySequence, QFont, QIcon, QPen
import pyqtgraph as pg
from datetime import datetime
from pyqtgraph.graphicsItems.DateAxisItem import DateAxisItem
from pyqtgraph import GraphicsLayoutWidget
from functools import partial
from engines.bruteforce_engine import run_bruteforce_mode
from concurrent.futures import ProcessPoolExecutor
from utils.formula import load_formulas, parse_formula, validate_factors, merge_factor_data, evaluate_formula_on_df
from utils.formula import sanitize_for_folder
from helper.shift_helper import determine_shift_override
from utils.transformation import TRANSFORMATIONS
from utils.plugin_loader import PluginLoader
from google.oauth2.service_account import Credentials
from utils.formula import parse_formula
from gspread.exceptions import APIError
from helper.factor_loader import load_factor_dataframe
from utils.model_loader import ModelLoader
from utils.plugin_loader import PluginLoader
from utils.reporting import Reporter
from utils.data_loader import DataLoader
from config.config_manager import ConfigManager
from utils.datasource_utils import aggregate_factor_options
from utils.interval_helper import extract_interval_from_filename, get_common_intervals, format_interval
from engines.bruteforce_engine import BruteforceEngine

HERE             = pathlib.Path(__file__).parent
PROJECT_ROOT     = HERE.parent
FORMULAS_CSV     = PROJECT_ROOT / "formulas.csv"
KEYFILE          = PROJECT_ROOT / "credentials" / "bruteforce.json"
CONFIG_DIR       = PROJECT_ROOT / "config"
CONFIG_SAVE_PATH = CONFIG_DIR / "bruteforce_tab_config.json"

def load_bruteforce_tab_config():
    try:
        with open(CONFIG_SAVE_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"model_configs": {}, "logic_configs": {}}

def save_bruteforce_tab_config(config_data):
    os.makedirs(os.path.dirname(CONFIG_SAVE_PATH), exist_ok=True)
    with open(CONFIG_SAVE_PATH, "w") as f:
        json.dump(config_data, f, indent=4)
        
def filter_factor_options(selected_interval, factor_options, original_ds_map):
    from utils.interval_helper import extract_interval_from_filename, parse_interval
    try:
        selected_value = parse_interval(selected_interval)
    except Exception:
        selected_value = None

    filtered_options = []
    filtered_ds_map = {}
    for opt in factor_options:
        if opt not in original_ds_map:
            continue
        ds_file, _ = original_ds_map[opt]
        file_interval = extract_interval_from_filename(ds_file)
        if file_interval is None:
            file_value = parse_interval("1h")
        else:
            try:
                file_value = parse_interval(file_interval)
            except Exception:
                file_value = parse_interval("1h")
        if selected_value is None or file_value <= selected_value:

            filtered_options.append(opt)
            filtered_ds_map[opt] = original_ds_map[opt]
    return filtered_options, filtered_ds_map

def run_bruteforce_in_process(config, data_loader, model_loader, logic_plugins, reporter, extra_params):
    from engines.bruteforce_engine import run_bruteforce_mode
    result = run_bruteforce_mode(config, data_loader, model_loader, logic_plugins, reporter, extra_params)
    return result

class MultiprocessingBruteforceWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    
    def __init__(self, config, data_loader, model_loader, logic_plugins, reporter, extra_params):
        super().__init__()
        self.config = config
        self.data_loader = data_loader
        self.model_loader = model_loader
        self.logic_plugins = logic_plugins
        self.reporter = reporter
        self.extra_params = extra_params

    def run(self):
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                run_bruteforce_in_process,
                self.config,
                self.data_loader,
                self.model_loader,
                self.logic_plugins,
                self.reporter,
                self.extra_params
            )
            result = future.result()
        self.finished.emit(result)

class WorkerLogStream(io.StringIO):
    def __init__(self, log_signal, throttle_interval=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_signal = log_signal
        self.throttle_interval = throttle_interval
        self.last_emit_time = time.time()
        self.buffer = ""

    def write(self, s):
        self.buffer += s
        current_time = time.time()
        if current_time - self.last_emit_time >= self.throttle_interval:
            self.log_signal.emit(self.buffer)
            self.buffer = ""
            self.last_emit_time = current_time

    def flush(self):
        if self.buffer:
            self.log_signal.emit(self.buffer)
            self.buffer = ""
        
class BruteforceWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, config, data_loader, model_loader, logic_plugins, reporter, extra_params):
        super().__init__()
        self.config = config
        self.data_loader = data_loader
        self.model_loader = model_loader
        self.logic_plugins = logic_plugins
        self.reporter = reporter
        self.extra_params = extra_params

    def run(self):
        old_stdout = sys.stdout
        sys.stdout = WorkerLogStream(self.log, throttle_interval=0)
        try:
            result = run_bruteforce_mode(
                self.config, self.data_loader, self.model_loader, 
                self.logic_plugins, self.reporter, self.extra_params
            )
        except Exception as e:
            self.log.emit("Error: " + str(e))
            result = {}
        finally:
            sys.stdout.flush()
            sys.stdout = old_stdout
        self.finished.emit(result)

class SimulatedBruteforceWorker(QThread):
    finished = pyqtSignal(pd.DataFrame)
    progress = pyqtSignal(int)
    
    def __init__(self, extra_params):
        super().__init__()
        self.extra_params = extra_params
        
    def run(self):
        symbols = self.extra_params.get("selected_symbols", [])
        factors = self.extra_params.get("selected_factors", [])
        intervals = self.extra_params.get("selected_intervals", [])
        models = self.extra_params.get("selected_models", [])
        logics = self.extra_params.get("selected_logics", [])
        model_params = self.extra_params.get("model_params", {})
        combinations = list(itertools.product(symbols, factors, intervals, models, logics))
        total = len(combinations)
        results = []
        for i, combo in enumerate(combinations):
            symbol, factor, interval, model, logic = combo
            params = model_params.get(model, {})
            avg_params = {}
            for param, bounds in params.items():
                if bounds:
                    avg_params[param] = (bounds["min"] + bounds["max"]) / 2.0
                else:
                    avg_params[param] = None
            dummy_metric = round(random.uniform(0, 1), 4)
            results.append({
                "Symbol": symbol,
                "Factor": factor,
                "Interval": interval,
                "Model": model,
                "Logic": logic,
                "Avg Params": str(avg_params),
                "Metric": dummy_metric
            })
            self.progress.emit(int((i+1) / total * 100))
            time.sleep(0.1)
        df = pd.DataFrame(results)
        self.finished.emit(df)

class BruteforceTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_google_sheet()
        self.results_df = None
        self.modelConfigInputs = {}  
        self.modelConfigs = {}        
        self.modelTotalLabels = {}    
        self.formula_mode = False
        self.load_available_options()
        self.original_ds_map = self.ds_map.copy()
        self.init_ui()
        self.datasource_folder = self.config["data"]["datasource_path"]
        self.folderWatcher = QFileSystemWatcher([self.datasource_folder], self)
        self.folderWatcher.directoryChanged.connect(self.on_folder_changed)

    def _init_google_sheet(self):
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_file(str(KEYFILE), scopes=scopes)
        client = gspread.authorize(creds)
        SHEET_ID = "1Hl-yF44_-Gu37bqxH-kWZyZZITyMigE0rf6RLJHoD2Q"
        try:
            self.spreadsheet = client.open_by_key(SHEET_ID)
        except Exception as e:
            raise RuntimeError(f"[GoogleSheets] could not open spreadsheet: {e}")
        for ws in self.spreadsheet.worksheets():
            if ws.id == 590014065:
                self.bruteforce_ws = ws
                break
        else:
            self.bruteforce_ws = self.spreadsheet.get_worksheet(0)
        header = [
            "Timestamp",
            "Symbols",
            "Intervals",
            "Factor / Formula",
            "Transformation",
            "Shift Override",
            "Models",
            "Model Params",
            "Entry/Exit Logic",
            "Logic Params",
            "Status",
        ]
        first_row = self.bruteforce_ws.row_values(1)
        if not first_row:
            self.bruteforce_ws.append_row(header)

    def _log_bruteforce_run(self, extra_params, results):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        temp_engine = BruteforceEngine(
            self.config,
            DataLoader(
                candle_path=self.config["data"]["candle_path"],
                datasource_path=self.config["data"]["datasource_path"],
                config=self.config,
                debug=False
            ),
            ModelLoader(),
            PluginLoader().load_logic_plugins(),
            Reporter(),
            extra_params
        )
        all_combos = temp_engine._prepare_combos()
        passed_set = set(results.keys())
        raw_mparams = extra_params.get("model_params", {})
        raw_lparams_map = extra_params.get("logic_config", {})

        rows = []
        for combo in all_combos:
            symbol, interval, model, logic, factor_combo, formula = combo
            status = "PASS✔" if combo in passed_set else "FAIL❌"
            if formula and formula not in ("<none>", None):
                fact_cell = formula
            else:
                fact_cell = factor_combo[0] if factor_combo else "<none>"
            m_cfg = {}
            for p, vals in raw_mparams.get(model, {}).items():
                if isinstance(vals, list) and vals:
                    mn, mx = vals[0], vals[-1]
                    st = round(vals[1] - vals[0], 4) if len(vals) > 1 else 0
                    m_cfg[p] = {"min": mn, "max": mx, "step": st}
            model_bounds_json = json.dumps(m_cfg, ensure_ascii=False)
            logic_vals = raw_lparams_map.get(logic, {}) 
            l_cfg = {}
            for p, vals in logic_vals.items():
                if isinstance(vals, list) and vals:
                    mn, mx = vals[0], vals[-1]
                    st = round(vals[1] - vals[0], 4) if len(vals) > 1 else 0
                    l_cfg[p] = {"min": mn, "max": mx, "step": st}
            logic_bounds_json = json.dumps(l_cfg, ensure_ascii=False)
            row = [
                ts,
                symbol,
                interval,
                fact_cell,
                extra_params.get("transformation", "None"),
                extra_params.get("shift_override", ""),
                model,
                model_bounds_json,
                logic,
                logic_bounds_json,
                status
            ]
            rows.append(row)

        try:
            self.bruteforce_ws.append_rows(rows, value_input_option='RAW')
        except APIError as e:
            print(f"[GoogleSheets] batch append failed: {e}")

    def load_available_options(self):
        cfg_path = str(CONFIG_DIR / "config.yaml")
        self.config_obj = ConfigManager(cfg_path)
        self.config     = self.config_obj.config
        candle_path = self.config["data"]["candle_path"]
        candle_files = [f for f in os.listdir(candle_path)
                        if f.endswith(".parquet") or f.endswith(".csv")]
        self.availableSymbols = sorted(list({f.split("_")[0] for f in candle_files}))
        if not self.availableSymbols:
            self.availableSymbols = ["N/A"]
        factor_options, self.ds_map, ds_native_intervals = aggregate_factor_options(
            self.config["data"]["datasource_path"]
        )
        self.availableFactors = factor_options[:]
        self.ds_native_intervals = ds_native_intervals
        if candle_files:
            first_file = candle_files[0]
            candle_native_interval = extract_interval_from_filename(first_file)
            if candle_native_interval is None:
                candle_native_interval = "1min"
        else:
            candle_native_interval = "1min"

        if self.ds_native_intervals:
            datasource_native_interval = format_interval(min(self.ds_native_intervals))
            self.availableIntervals = get_common_intervals(candle_native_interval, datasource_native_interval)
        else:
            self.availableIntervals = ["1h"]

        model_loader = ModelLoader()
        self.availableModels = model_loader.discover_models()
        if not self.availableModels:
            self.availableModels = ["N/A"]

        plugin_loader = PluginLoader()
        logic_plugins = plugin_loader.load_logic_plugins()
        self.availableLogics = list(logic_plugins.keys())
        if not self.availableLogics:
            self.availableLogics = ["N/A"]
    
    def init_ui(self):
        mainLayout = QVBoxLayout(self)

        # --- Configuration Group ---
        configGroup = QGroupBox("Configuration")
        configGroupLayout = QVBoxLayout(configGroup)
        configSplitter = QSplitter(Qt.Horizontal)

        # Symbols
        symbolsWidget = QWidget()
        symbolsLayout = QVBoxLayout(symbolsWidget)
        symbolsLabel = QLabel("Symbols:")
        self.symbolsList = QListWidget()
        self.symbolsList.setSelectionMode(QListWidget.SingleSelection)
        for symbol in self.availableSymbols:
            QListWidgetItem(symbol, self.symbolsList)
        symbolsLayout.addWidget(symbolsLabel)
        symbolsLayout.addWidget(self.symbolsList)
        configSplitter.addWidget(symbolsWidget)

        # Intervals
        intervalsWidget = QWidget()
        intervalsLayout = QVBoxLayout(intervalsWidget)
        intervalsLabel = QLabel("Intervals:")
        self.intervalsList = QListWidget()
        self.intervalsList.setSelectionMode(QListWidget.ExtendedSelection)
        for interval in self.availableIntervals:
            QListWidgetItem(interval, self.intervalsList)
        intervalsLayout.addWidget(intervalsLabel)
        intervalsLayout.addWidget(self.intervalsList)
        configSplitter.addWidget(intervalsWidget)

        # Datasource with Toggle
        factorsWidget = QWidget()
        factorsLayout = QVBoxLayout(factorsWidget)
        headerLayout = QHBoxLayout()
        factorsLabel = QLabel("Datasource")
        self.toggleButton = QPushButton("Use Formula")
        self.toggleButton.setCheckable(True)
        self.toggleButton.clicked.connect(self.toggleDatasourceMode)
        headerLayout.addWidget(factorsLabel)
        headerLayout.addWidget(self.toggleButton)
        headerLayout.addStretch()
        factorsLayout.addLayout(headerLayout)

        # ─── Search bar ───────────────────────────────────
        self.datasourceSearch = QLineEdit()
        self.datasourceSearch.setPlaceholderText("Search datasource / formula…")
        self.datasourceSearch.textChanged.connect(self.filterDatasourceList)
        factorsLayout.addWidget(self.datasourceSearch)
        # ───────────────────────────────────────────────────

        self.datasourceList = QListWidget()
        self.loadDatasourceFactors()
        factorsLayout.addWidget(self.datasourceList)
        configSplitter.addWidget(factorsWidget)

        # Transformation
        transformWidget = QWidget()
        transformLayout = QVBoxLayout(transformWidget)
        transformLabel = QLabel("Transformation:")
        self.transformList = QListWidget()
        self.transformList.setSelectionMode(QListWidget.SingleSelection)
        from utils.transformation import TRANSFORMATIONS
        for trans_name in TRANSFORMATIONS.keys():
            QListWidgetItem(trans_name, self.transformList)
        if self.transformList.count() > 0:
            self.transformList.setCurrentRow(0)
        transformLayout.addWidget(transformLabel)
        transformLayout.addWidget(self.transformList)
        configSplitter.addWidget(transformWidget)

        # Models
        modelsWidget = QWidget()
        modelsLayout = QVBoxLayout(modelsWidget)
        modelsLabel = QLabel("Models:")
        self.modelsList = QListWidget()
        self.modelsList.setSelectionMode(QListWidget.ExtendedSelection)
        for model in self.availableModels:
            QListWidgetItem(model, self.modelsList)
        modelsLayout.addWidget(modelsLabel)
        modelsLayout.addWidget(self.modelsList)
        configSplitter.addWidget(modelsWidget)

        # Entry/Exit Logic
        logicWidget = QWidget()
        logicLayout = QVBoxLayout(logicWidget)
        logicLabel = QLabel("Entry/Exit Logic:")
        self.logicList = QListWidget()
        self.logicList.setSelectionMode(QListWidget.ExtendedSelection)
        for logic in self.availableLogics:
            QListWidgetItem(logic, self.logicList)
        logicLayout.addWidget(logicLabel)
        logicLayout.addWidget(self.logicList)
        configSplitter.addWidget(logicWidget)

        # Add config splitter to the group
        configGroupLayout.addWidget(configSplitter)

        # --- Parameter Ranges Container (Model & Logic) ---
        self.parameterRangesContainer = QWidget()
        rangesLayout = QHBoxLayout(self.parameterRangesContainer)
        rangesLayout.setContentsMargins(0, 0, 0, 0)
        rangesLayout.setSpacing(10)

        # Left: Model Config Parameter Ranges
        self.modelConfigGroup = QGroupBox("Model Config Parameter Ranges")
        self.modelConfigTabs = QTabWidget()
        modelLayout = QVBoxLayout()
        modelLayout.addWidget(self.modelConfigTabs)
        self.modelConfigGroup.setLayout(modelLayout)
        rangesLayout.addWidget(self.modelConfigGroup)

        # Right: Logic Config Parameter Ranges
        self.logicConfigGroup = QGroupBox("Logic Config Parameter Ranges")
        self.logicConfigTabs = QTabWidget()
        logicLayout = QVBoxLayout()
        logicLayout.addWidget(self.logicConfigTabs)
        self.logicConfigGroup.setLayout(logicLayout)
        rangesLayout.addWidget(self.logicConfigGroup)

        # --- Vertical splitter between Configuration and Parameter Ranges ---
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(configGroup)
        splitter.addWidget(self.parameterRangesContainer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        mainLayout.addWidget(splitter)

        # --- Overall Combinations Label ---
        self.overallTotalLabel = QLabel("Overall total combinations: 0")
        mainLayout.addWidget(self.overallTotalLabel)

        # --- Save Config Button (aligned right) ---
        self.saveConfigButton = QPushButton("Save Config")
        self.saveConfigButton.setToolTip("Save current model & logic parameter ranges")
        self.saveConfigButton.clicked.connect(self.save_configs)
        saveLayout = QHBoxLayout()
        saveLayout.addStretch()
        saveLayout.addWidget(self.saveConfigButton)
        mainLayout.addLayout(saveLayout)

        # --- Apply Criteria Toggle (above Run button) ---
        self.criteriaToggle = QCheckBox("Apply Bruteforce Criteria")
        self.criteriaToggle.setChecked(True)
        mainLayout.addWidget(self.criteriaToggle)

        # --- Run Bruteforce Button ---
        self.runButton = QPushButton("Run Bruteforce")
        mainLayout.addWidget(self.runButton)

        # --- Signal Connections ---
        self.symbolsList.itemSelectionChanged.connect(self.update_bruteforce_factors)
        self.intervalsList.itemSelectionChanged.connect(self.update_bruteforce_factors)
        self.modelsList.itemSelectionChanged.connect(self.updateModelConfigPanel)
        self.logicList.itemSelectionChanged.connect(self.updateLogicConfigPanel)
        self.symbolsList.itemSelectionChanged.connect(self.calculate_overall_combinations)
        self.datasourceList.itemSelectionChanged.connect(self.calculate_overall_combinations)
        self.intervalsList.itemSelectionChanged.connect(self.calculate_overall_combinations)
        self.logicList.itemSelectionChanged.connect(self.calculate_overall_combinations)
        self.modelsList.itemSelectionChanged.connect(self.calculate_overall_combinations)
        self.runButton.clicked.connect(self.run_bruteforce)
        
    def save_configs(self):
        from PyQt5.QtWidgets import QMessageBox
        cfg = load_bruteforce_tab_config()
        for model, inputs in self.modelConfigInputs.items():
            model_cfg = {}
            for param, fields in inputs.items():
                try:
                    mn = float(fields["min"].text())
                    mx = float(fields["max"].text())
                    st = float(fields["step"].text())
                except ValueError:
                    mn = fields["min"].text()
                    mx = fields["max"].text()
                    st = fields["step"].text()
                model_cfg[param] = {"min": mn, "max": mx, "step": st}
            cfg.setdefault("model_configs", {})[model] = model_cfg

        for logic, inputs in getattr(self, "logicConfigInputs", {}).items():
            logic_cfg = {}
            for param, fields in inputs.items():
                try:
                    mn = float(fields["min"].text())
                    mx = float(fields["max"].text())
                    st = float(fields["step"].text())
                except ValueError:
                    mn = fields["min"].text()
                    mx = fields["max"].text()
                    st = fields["step"].text()
                logic_cfg[param] = {"min": mn, "max": mx, "step": st}
            cfg.setdefault("logic_configs", {})[logic] = logic_cfg
        save_bruteforce_tab_config(cfg)
        QMessageBox.information(self, "Success", "Bruteforce configuration saved.")

    def update_bruteforce_factors(self):
        symbols = [item.text().lower() for item in self.symbolsList.selectedItems()]
        intervals = [item.text() for item in self.intervalsList.selectedItems()]
        selected_interval = intervals[0] if intervals else None
        filtered_options, filtered_map = filter_factor_options(
            selected_interval,
            self.availableFactors,
            self.original_ds_map
        )
        def _normalize_filename(fn: str) -> str:
            fn = fn.lower()
            for prefix in ("cq_", "gn_"):
                if fn.startswith(prefix):
                    return fn[len(prefix):]
            return fn

        filtered_options2 = []
        filtered_map2     = {}
        for opt in filtered_options:
            fname, col = filtered_map[opt]
            norm = _normalize_filename(fname)
            if any(sym in norm for sym in symbols):
                filtered_options2.append(opt)
                filtered_map2[opt] = (fname, col)
        self.ds_map = filtered_map2
        self.datasourceList.clear()
        if not self.formula_mode:
            for opt in filtered_options2:
                QListWidgetItem(opt, self.datasourceList)
        else:
            for raw in load_formulas(str(FORMULAS_CSV)):
                try:
                    tokens = parse_formula(raw)
                except:
                    continue
                if all(any(k.split("(")[0].strip() == tok for k in filtered_map2) for tok in tokens):
                    QListWidgetItem(raw, self.datasourceList)

        self.datasourceList.setSelectionMode(QListWidget.ExtendedSelection)
    
    def loadDatasourceFactors(self):
        self.datasourceList.clear()
        self.datasourceSearch.clear()
        for factor in self.availableFactors:
            QListWidgetItem(factor, self.datasourceList)
        self.datasourceList.setSelectionMode(QListWidget.ExtendedSelection)
        self.formula_mode = False
    
    def loadFormulas(self):
        self.datasourceList.clear()
        self.datasourceSearch.clear()
        self.datasourceList.addItem("<none>")
        self.loaded_formulas = load_formulas(str(FORMULAS_CSV))
        self.formula_map = {}
        for raw in self.loaded_formulas:
            safe = sanitize_for_folder(raw)
            self.formula_map[raw] = safe
            self.formula_map[safe] = raw
            QListWidgetItem(raw, self.datasourceList)
        self.datasourceList.setSelectionMode(QListWidget.ExtendedSelection)
        self.formula_mode = True

    def filterDatasourceList(self, text: str):
        text = text.strip().lower()
        self.datasourceList.clear()
        if not self.formula_mode:
            matching = [
                opt for opt in self.ds_map.keys()
                if text in opt.lower()
            ]
        else:
            matching = []
            for raw in getattr(self, "loaded_formulas", []):
                if text not in raw.lower():
                    continue
                try:
                    tokens = parse_formula(raw)
                except:
                    continue
                if all(any(key.split("(")[0].strip() == tok for key in self.ds_map) for tok in tokens):
                    matching.append(raw)

        for item in matching:
            QListWidgetItem(item, self.datasourceList)
        self.datasourceList.setSelectionMode(QListWidget.ExtendedSelection)

    def toggleDatasourceMode(self):
        if self.formula_mode:
            self.loadDatasourceFactors()
            self.toggleButton.setText("Use Formula")
        else:
            self.loadFormulas()
            self.toggleButton.setText("Use Datasource")
        self.calculate_overall_combinations()
    
    def on_folder_changed(self, path):
        print(f"Datasource folder changed: {path}")
        self.load_available_options()
        self.loadDatasourceFactors()
        self.calculate_overall_combinations()
    
    def updateModelConfigPanel(self):
        self.modelConfigTabs.clear()
        self.modelConfigInputs = {}
        self.modelTotalLabels = {}
        selected_models = [item.text() for item in self.modelsList.selectedItems()]
        from utils.model_loader import ModelLoader
        model_loader = ModelLoader()
        for model in selected_models:
            calculate_signal, model_config = model_loader.load_model(model)
            if not model_config:
                continue

            self.modelConfigInputs[model] = {}
            tab = QWidget()
            formLayout = QFormLayout(tab)
            for param, info in model_config.items():
                if isinstance(info, dict) and "default" in info:
                    default_val = info["default"]
                    minEdit = QLineEdit(str(default_val))
                    maxEdit = QLineEdit(str(default_val))
                    stepEdit = QLineEdit("1")
                    hLayout = QHBoxLayout()
                    hLayout.addWidget(QLabel("Min:"))
                    hLayout.addWidget(minEdit)
                    hLayout.addWidget(QLabel("Max:"))
                    hLayout.addWidget(maxEdit)
                    hLayout.addWidget(QLabel("Step:"))
                    hLayout.addWidget(stepEdit)
                    formLayout.addRow(f"{param}:", hLayout)
                    self.modelConfigInputs[model][param] = {"min": minEdit, "max": maxEdit, "step": stepEdit}

                    saved = load_bruteforce_tab_config().get("model_configs", {}).get(model, {})
                    if param in saved:
                        bounds = saved[param]
                        minEdit.setText(str(bounds.get("min", default_val)))
                        maxEdit.setText(str(bounds.get("max", default_val)))
                        stepEdit.setText(str(bounds.get("step",    1)))
                    for widget in (minEdit, maxEdit, stepEdit):
                        widget.textChanged.connect(lambda _, m=model: self.calculate_model_total_combinations(m))
            total_label = QLabel("Total combinations: 0")
            formLayout.addRow(total_label)
            self.modelTotalLabels[model] = total_label
            self.modelConfigTabs.addTab(tab, model)
            self.calculate_model_total_combinations(model)

    def calculate_model_total_combinations(self, model):
        total = 1
        inputs = self.modelConfigInputs.get(model, {})
        for param, widgets in inputs.items():
            try:
                mn = float(widgets["min"].text())
                mx = float(widgets["max"].text())
                st = float(widgets["step"].text())
                if st <= 0 or mx < mn:
                    count = 0
                else:
                    count = int((mx - mn) // st) + 1
            except Exception:
                count = 0
            total *= count
        if model in self.modelTotalLabels:
            self.modelTotalLabels[model].setText(f"Total combinations: {total}")

    def calculate_logic_total_combinations(self, logic_name):
        total = 1
        inputs = self.logicConfigInputs.get(logic_name, {})
        for param, widgets in inputs.items():
            try:
                mn = float(widgets["min"].text())
                mx = float(widgets["max"].text())
                st = float(widgets["step"].text())
                if st <= 0 or mx < mn:
                    count = 0
                else:
                    count = int((mx - mn) // st) + 1
            except Exception:
                count = 0
            total *= count
        if logic_name in self.logicTotalLabels:
            self.logicTotalLabels[logic_name].setText(f"Total combinations: {total}")

    def updateLogicConfigPanel(self):
        self.logicConfigTabs.clear()
        self.logicConfigInputs = {}
        self.logicTotalLabels = {}
        selected_logics = [item.text() for item in self.logicList.selectedItems()]
        from utils.plugin_loader import PluginLoader
        plugin_loader = PluginLoader()
        logic_plugins = plugin_loader.load_logic_plugins()
        for logic_name in selected_logics:
            logic_instance = logic_plugins.get(logic_name)
            logic_config   = getattr(type(logic_instance), "LOGIC_CONFIG", {})
            if not logic_config:
                continue
            self.logicConfigInputs[logic_name] = {}
            tab = QWidget()
            formLayout = QFormLayout(tab)
            saved_logic = load_bruteforce_tab_config().get("logic_configs", {}).get(logic_name, {})
            for param, param_info in logic_config.items():
                default_val = param_info.get("default", 0)
                minEdit  = QLineEdit(str(default_val))
                maxEdit  = QLineEdit(str(default_val))
                stepEdit = QLineEdit("1")
                if param in saved_logic:
                    bounds = saved_logic[param]
                    minEdit.setText(str(bounds.get("min", default_val)))
                    maxEdit.setText(str(bounds.get("max", default_val)))
                    stepEdit.setText(str(bounds.get("step",    1)))

                hLayout = QHBoxLayout()
                hLayout.addWidget(QLabel("Min:"));  hLayout.addWidget(minEdit)
                hLayout.addWidget(QLabel("Max:"));  hLayout.addWidget(maxEdit)
                hLayout.addWidget(QLabel("Step:")); hLayout.addWidget(stepEdit)
                formLayout.addRow(f"{param}:", hLayout)
                self.logicConfigInputs[logic_name][param] = {
                    "min":  minEdit,
                    "max":  maxEdit,
                    "step": stepEdit
                }

                minEdit.textChanged.connect(lambda _, ln=logic_name: self.calculate_logic_total_combinations(ln))
                maxEdit.textChanged.connect(lambda _, ln=logic_name: self.calculate_logic_total_combinations(ln))
                stepEdit.textChanged.connect(lambda _, ln=logic_name: self.calculate_logic_total_combinations(ln))
            total_label = QLabel("Total combinations: 0")
            formLayout.addRow(total_label)
            self.logicTotalLabels[logic_name] = total_label
            self.logicConfigTabs.addTab(tab, logic_name)
            self.calculate_logic_total_combinations(logic_name)

    def calculate_overall_combinations(self):
        count_symbols    = len(self.symbolsList.selectedItems())
        count_intervals  = len(self.intervalsList.selectedItems())
        count_datasource = len(self.datasourceList.selectedItems())
        count_models     = len(self.modelsList.selectedItems())
        count_logics     = len(self.logicList.selectedItems())
        base = (
            count_symbols
            * count_intervals
            * count_datasource
            * count_models
            * count_logics
        )
        base_detail = (
            f"{base} = Symbols({count_symbols}) * "
            f"Intervals({count_intervals}) * "
            f"Datasource/Formulas({count_datasource}) * "
            f"Models({count_models}) * "
            f"Logics({count_logics})"
        )
        self.overallTotalLabel.setText(
            "Overall total combinations (Base only): " + base_detail
        )

        selected_models = [item.text() for item in self.modelsList.selectedItems()]
        for model in selected_models:
            inputs = self.modelConfigInputs.get(model, {})
            total = 1
            for param, widgets in inputs.items():
                minEdit  = widgets["min"]
                maxEdit  = widgets["max"]
                stepEdit = widgets["step"]

                try:
                    mn = float(minEdit.text())
                    mx = float(maxEdit.text())
                    st = float(stepEdit.text())
                    if st <= 0 or mx < mn:
                        count = 0
                    else:
                        count = int((mx - mn) // st) + 1
                except Exception:
                    count = 0

                total *= count
            if model in self.modelTotalLabels:
                self.modelTotalLabels[model].setText(f"Total combinations: {total}")

    
    def save_model_config(self):
        new_configs = {}
        for model, inputs in self.modelConfigInputs.items():
            model_config = {}
            for param, (minEdit, maxEdit, stepEdit) in inputs.items():
                try:
                    if "." in minEdit.text():
                        min_val = float(minEdit.text())
                        max_val = float(maxEdit.text())
                        step_val = float(stepEdit.text())
                    else:
                        min_val = int(minEdit.text())
                        max_val = int(maxEdit.text())
                        step_val = int(stepEdit.text())
                    model_config[param] = {"min": min_val, "max": max_val, "step": step_val}
                except Exception:
                    model_config[param] = {"min": minEdit.text(), "max": maxEdit.text(), "step": stepEdit.text()}
            new_configs[model] = model_config
        existing_configs = load_bruteforce_model_configs()
        existing_configs.update(new_configs)
        save_bruteforce_model_configs(existing_configs)
        QMessageBox.information(self, "Success", "Bruteforce model configuration saved.")
        
    def run_bruteforce(self):
        selected_symbols   = [item.text() for item in self.symbolsList.selectedItems()]
        selected_intervals = [item.text() for item in self.intervalsList.selectedItems()]
        selected_models    = [item.text() for item in self.modelsList.selectedItems()]
        selected_logics    = [item.text() for item in self.logicList.selectedItems()]

        if not (selected_symbols and selected_intervals and selected_models and selected_logics):
            return
        config = ConfigManager(str(CONFIG_DIR / "config.yaml")).config

        data_loader = DataLoader(
            candle_path=config["data"]["candle_path"],
            datasource_path=config["data"]["datasource_path"],
            config=config,
            debug=True
        )

        # --- Determine shift_override by calling into load_factor_dataframe ---
        if self.formula_mode and self.datasourceList.selectedItems():
            formula = self.datasourceList.selectedItems()[0].text()
            manual  = []
        else:
            formula = None
            manual  = [item.text() for item in self.datasourceList.selectedItems()]

        # We only need its 'shift' return value here:
        _, _, shift_val, _ = load_factor_dataframe(
            data_loader,
            config,
            self.ds_map,
            selected_symbols[0],
            selected_intervals[0],
            formula,
            manual
        )

        # --- Build model_params ranges from your UI panels ---
        model_params = {}
        for model, inputs in self.modelConfigInputs.items():
            params = {}
            model_config = self.modelConfigs.get(model, {})
            for param, widget_dict in inputs.items():
                try:
                    default_val = model_config.get(param, {}).get("default")
                    mn = int(float(widget_dict["min"].text()))
                    mx = int(float(widget_dict["max"].text()))
                    st = int(float(widget_dict["step"].text()))
                    params[param] = {"min": mn, "max": mx, "step": st}
                except Exception as e:
                    print(f"Error converting input for '{param}' in '{model}': {e}")
                    params[param] = None
            model_params[model] = params

        # Expand to full lists of values
        scalar_model_params = {}
        for model, bounds in model_params.items():
            scalar = {}
            for p, b in bounds.items():
                if isinstance(b, dict):
                    scalar[p] = list(range(b["min"], b["max"] + 1, b["step"]))
                else:
                    scalar[p] = None
            scalar_model_params[model] = scalar

        # --- Load logic plugins and assemble extra_params ---
        logic_plugins = PluginLoader().load_logic_plugins()

        extra_params = {
            "selected_symbols":   selected_symbols,
            "selected_intervals": selected_intervals,
            "selected_models":    selected_models,
            "selected_logics":    selected_logics,
            "model_params":       scalar_model_params,
            "shift_override":     shift_val,
            "apply_bruteforce_criteria": self.criteriaToggle.isChecked(),
            "transformation":     (self.transformList.currentItem().text()
                                   if self.transformList.currentItem() else "None")
        }
        # --- Datasource vs. formula flags ---
        if self.formula_mode:
            formulas = [item.text() for item in self.datasourceList.selectedItems()]
            extra_params["formula"] = formulas if formulas else ["<none>"]
        else:
            extra_params["formula"] = ["<none>"]
            extra_params["selected_factors"] = [item.text() for item in self.datasourceList.selectedItems()]
        logic_config_map = {}
        for logic_name in selected_logics:
            if logic_name in self.logicConfigInputs:
                cfg = {}
                for key, widgets in self.logicConfigInputs[logic_name].items():
                    mn = float(widgets["min"].text())
                    mx = float(widgets["max"].text())
                    st = float(widgets["step"].text())
                    if st > 0 and mx >= mn:
                        vals = []
                        cur = mn
                        while cur <= mx + 1e-8:
                            vals.append(round(cur,8))
                            cur += st
                    else:
                        vals = []
                    cfg[key] = vals
            else:
                plugin_cfg = getattr(logic_plugins[logic_name], "LOGIC_CONFIG", {})
                cfg = {k:[v["default"]] for k,v in plugin_cfg.items()}
            logic_config_map[logic_name] = cfg
        extra_params["logic_config"] = logic_config_map
        first_model = selected_models[0]
        strategy_params = scalar_model_params.get(first_model, {}).copy()
        for k, v in extra_params.get("logic_config", {}).items():
            strategy_params[k] = v
        print("DEBUG strategy_params after merge:", strategy_params)
        reporter    = Reporter()
        model_loader = ModelLoader()
        self.runButton.setEnabled(False)
        self.worker = MultiprocessingBruteforceWorker(
            config, data_loader, model_loader, logic_plugins, reporter, extra_params
        )
        self.worker.finished.connect(self.on_bruteforce_finished)
        self.worker.start()
    
    def on_bruteforce_finished(self, results):
        QApplication.beep()
        try:
            self._log_bruteforce_run(self.worker.extra_params, results)
        except Exception:
            pass
        self.runButton.setEnabled(True)
