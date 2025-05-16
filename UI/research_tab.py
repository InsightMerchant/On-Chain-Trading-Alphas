import os, math, re
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog,
    QTreeWidget, QTreeWidgetItem, QLabel,
    QSpinBox, QAbstractItemView, QGroupBox, QFormLayout,
    QSplitter, QPushButton, QComboBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QProgressBar,
    QPlainTextEdit, QTextEdit, QInputDialog, QMessageBox, QMenu, QScrollArea
)
from PyQt5.QtCore import (
    Qt, QTimer, QRunnable, QThreadPool,
    pyqtSlot, QObject, pyqtSignal
)
from PyQt5.QtGui import QTextCursor
import pyqtgraph as pg
from utils.timestamp_utils import detect_and_normalize_timestamp
from utils.formula import evaluate_formula_on_df, parse_formula
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns

pg.setConfigOption('background', '#121212')
pg.setConfigOption('foreground', '#ffffff')

def downsample_for_width(x, y, max_points=800):
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return x[idx], y[idx]

class WorkerSignals(QObject):
    finished = pyqtSignal(object)

class CsvLoader(QRunnable):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            df = pd.read_csv(self.path, index_col=None)
            df = detect_and_normalize_timestamp(df, canonical="datetime")
            if "datetime" in df.columns:
                df.set_index("datetime", inplace=True)
            self.signals.finished.emit((self.path, df))
        except Exception as e:
            self.signals.finished.emit((self.path, e))

class PlotWorkerSignals(QObject):
    finished = pyqtSignal(object)  

class PlotWorker(QRunnable):
    def __init__(self, smap, intervals, customs, methods):
        super().__init__()
        self.smap = smap
        self.intervals = intervals
        self.customs = customs
        self.methods = methods
        self.signals = PlotWorkerSignals()

    @pyqtSlot()
    def run(self):
        # 1) Resample/aggregate each series (keep NaNs)
        processed = {}
        for lbl, s in self.smap.items():
            interval = self.intervals.get(lbl, 'Original')
            method = self.methods.get(lbl, 'direct')
            if method == 'direct' or interval == 'Original':
                s2 = s.copy()
            else:
                iv = interval if interval != 'Custom' else self.customs.get(lbl)
                try:
                    s2 = s.resample(iv).agg(method)
                except Exception:
                    s2 = s.copy()
            processed[lbl] = s2

        # 2) Build table DataFrame
        combined = pd.concat(processed.values(), axis=1, keys=processed.keys())
        table_df = combined.reset_index().rename(columns={'index': 'datetime'})

        # 3) Compute per-series stats
        stats = []
        for lbl, s2 in processed.items():
            s_clean = s2[np.isfinite(s2)]
            stats.append({
                'Series': lbl,
                'dtype':  s2.dtype,
                'Min':    s2.min(skipna=True),
                'Max':    s2.max(skipna=True),
                'NaNs':     int(s2.isna().sum()), 
                'Inf':      int(np.isposinf(s2).sum()),
                '-Inf':     int(np.isneginf(s2).sum()),
                'Mean':   s2.mean(skipna=True),
                'Median': s2.median(skipna=True),
                'Std':    s2.std(skipna=True),
                'Skewness': s2.skew(skipna=True),
                'Kurtosis': s2.kurtosis(skipna=True)
            })
        self.signals.finished.emit((processed, table_df, stats))

class QPlainTextEditWithSelector(QPlainTextEdit):
    def __init__(self, parent=None, df_source=None):
        super().__init__(parent)
        self.df_source = df_source
        self.popup = QComboBox(self)
        self.popup.setWindowFlags(Qt.Popup)
        self.popup.setEditable(False)
        self.popup.activated.connect(self.insertSelection)
        self.selector_active = False
        self.selector_start = None

    def insertSelection(self, index):
        text = self.popup.currentText()
        tc = self.textCursor()
        if self.selector_start is not None:
            nback = tc.position() - self.selector_start
            tc.movePosition(QTextCursor.Left, QTextCursor.MoveAnchor, nback)
        else:
            tc.movePosition(QTextCursor.Left, QTextCursor.MoveAnchor, 1)
        tc.insertText(text + "']")
        self.setTextCursor(tc)
        self.popup.hidePopup()
        self.selector_active = False
        self.selector_start = None

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.text() != "'":
            return

        tc = self.textCursor()
        pos = tc.position()
        doc = self.toPlainText()
        before = doc[:pos-1]  
        if re.search(r"df\[\s*$", before):
            self.selector_active = True
            self.selector_start = pos
            cols = sorted({c for df in self.df_source.dataframes.values() for c in df.columns})
            self.popup.clear()
            self.popup.addItems(cols)
            rect = self.cursorRect()
            pt = self.mapToGlobal(rect.bottomLeft())
            size = self.popup.sizeHint()
            self.popup.setGeometry(pt.x(), pt.y(), size.width(), size.height())
            self.popup.setCurrentIndex(0)
            self.popup.showPopup()
        else:
            return

class ResearchTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataframes = {}
        self._last_series_map = {}
        self.series_plot_types = {}
        self.series_resample_intervals = {}
        self.series_custom_intervals = {}
        self.series_resample_methods = {}

        self.threadpool = QThreadPool()
        self._init_ui()

    def _init_ui(self):
        main_split = QSplitter(Qt.Horizontal, self)

        # LEFT pane: import + tree
        left = QWidget()
        left.setLayout(QVBoxLayout())
        btn_import = QPushButton("Import CSV(s)")
        btn_import.clicked.connect(self.import_csv)
        left.layout().addWidget(btn_import)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Files and Columns"])
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.dragEnterEvent = self._drag_enter
        self.tree.dropEvent = self._drop_files
        self.tree.itemSelectionChanged.connect(self._update_controls)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        left.layout().addWidget(self.tree)
        main_split.addWidget(left)

        # RIGHT pane
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(5)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Resampling…")
        self.progress_bar.setVisible(False)
        rv.addWidget(self.progress_bar)

        ctrl = QWidget()
        ctrl.setLayout(QVBoxLayout())
        self.series_box = QGroupBox("Per-Series Controls")
        self.series_box.setLayout(QFormLayout())
        ctrl.layout().addWidget(self.series_box)

        mode_row = QWidget()
        mr = QHBoxLayout(mode_row)
        mr.setContentsMargins(0, 0, 0, 0)
        mr.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Overlay", "Grid", "Correlation"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_change)
        mr.addWidget(self.mode_combo)
        mr.addStretch()
        ctrl.layout().addWidget(mode_row)

        # GRAPH/TABLE tabs
        graph_page = QWidget()
        self.graph_splitter = QSplitter(Qt.Vertical)
        self.plot_widget = pg.PlotWidget(title="Data Visualization")
        for ax in ("left", "bottom"):
            axis = self.plot_widget.getAxis(ax)
            axis.setPen(pg.mkPen("w"))
            axis.setTextPen(pg.mkPen("w"))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.graph_splitter.addWidget(self.plot_widget)
        self.stats_panel = QTextEdit()
        self.stats_panel.setReadOnly(True)
        self.graph_splitter.addWidget(self.stats_panel)
        graph_page.setLayout(QVBoxLayout())
        graph_page.layout().addWidget(self.graph_splitter)

        # Tab widget for Graph and Table
        self.view_tabs = QTabWidget()
        self.view_tabs.addTab(graph_page, "Graph")
        table_page = QWidget()
        table_page.setLayout(QVBoxLayout())
        self.table_widget = QTableWidget()
        self.table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        table_page.layout().addWidget(self.table_widget)
        self.view_tabs.addTab(table_page, "Table")

        # FORMULA panel in a styled QGroupBox
        formula_group = QGroupBox("Formula Editor")
        formula_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #777;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPlainTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
            }
        """)
        fg_layout = QVBoxLayout(formula_group)
        fg_layout.setContentsMargins(8, 20, 8, 8)
        fg_layout.addWidget(QLabel("Enter your formula below:"))
        self.formula_edit = QPlainTextEditWithSelector(self, df_source=self)
        self.formula_edit.setPlaceholderText("e.g. df['col1'] + df['col2']")
        self.formula_edit.setFixedHeight(80)
        fg_layout.addWidget(self.formula_edit)
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: #f00;")
        fg_layout.addWidget(self.error_label)
        btn_row = QHBoxLayout()
        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self._apply_formula)
        btn_help = QPushButton("Help")
        btn_help.clicked.connect(self._show_formula_help)
        btn_row.addWidget(btn_apply)
        btn_row.addWidget(btn_help)
        btn_row.addStretch()
        fg_layout.addLayout(btn_row)

        # combine tabs and formula in a vertical splitter
        content_split = QSplitter(Qt.Vertical)
        content_split.addWidget(self.view_tabs)
        content_split.addWidget(formula_group)
        content_split.setStretchFactor(0, 3)
        content_split.setStretchFactor(1, 1)

        # now split controls vs content
        vsplit = QSplitter(Qt.Vertical)
        vsplit.addWidget(ctrl)
        vsplit.addWidget(content_split)
        vsplit.setStretchFactor(0, 0)
        vsplit.setStretchFactor(1, 1)

        rv.addWidget(vsplit, stretch=1)
        main_split.addWidget(right)
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 3)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(main_split)

    def _on_tree_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if not item:
            return

        data = item.data(0, Qt.UserRole)
        if not isinstance(data, str):
            return

        menu = QMenu(self)
        act = menu.addAction("Remove Folder")
        action = menu.exec_(self.tree.viewport().mapToGlobal(pos))
        if action is act:
            self._remove_folder(data)

    def _remove_folder(self, filename):
        """
        Remove the folder named `filename` from both the tree and our dataframes.
        """
        # 1) Remove from the dict
        if filename in self.dataframes:
            del self.dataframes[filename]

        # 2) Find and remove the top-level item
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if it.data(0, Qt.UserRole) == filename:
                self.tree.takeTopLevelItem(i)
                break


    # CSV import & loading
    def import_csv(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Import CSV files", "", "CSV Files (*.csv)")
        self._load_paths(paths)

    def _drag_enter(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def _drop_files(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls()
                 if u.toLocalFile().lower().endswith('.csv')]
        self._load_paths(paths)

    def _load_paths(self, paths):
        for p in paths:
            loader = CsvLoader(p)
            loader.signals.finished.connect(self._on_csv_loaded)
            self.threadpool.start(loader)

    def _on_csv_loaded(self, result):
        path, df_or_err = result
        name = os.path.basename(path)
        if isinstance(df_or_err, Exception):
            print(f"Failed to load {path}: {df_or_err}")
            return
        self.dataframes[name] = df_or_err
        top = QTreeWidgetItem(self.tree, [name])
        top.setData(0, Qt.UserRole, name)
        for c in df_or_err.columns:
            ch = QTreeWidgetItem(top, [c])
            ch.setData(0, Qt.UserRole, (name, c))
        top.setExpanded(True)


    # Controls update
    def _update_controls(self):
        form = self.series_box.layout()
        while form.count():
            w = form.takeAt(0).widget()
            if w: w.deleteLater()
        for itm in self.tree.selectedItems():
            data = itm.data(0, Qt.UserRole)
            if not isinstance(data, tuple): continue
            fn, col = data
            df = self.dataframes.get(fn)
            if df is None or col not in df: continue
            s = df[col].dropna()
            if not pd.api.types.is_numeric_dtype(s): continue
            key = f"{fn}:{col}"

            # Plot type
            type_cb = QComboBox()
            type_cb.addItems(["Line","Scatter","Histogram"])
            type_cb.setCurrentText(self.series_plot_types.get(key, "Line"))
            type_cb.currentTextChanged.connect(lambda t,k=key: self._on_type_change(k,t))
            res_cb = QComboBox()
            res_cb.addItems(["Original","1h","2h","4h","1d","Custom…"])
            res_cb.setCurrentText(self.series_resample_intervals.get(key, "Original"))
            
            def on_res_change(txt, k=key):             
                if txt == "Custom…":
                    interval, ok = QInputDialog.getText(
                        self, "Custom Interval",
                        "Enter pandas offset alias (e.g. '30T','3H','7D'):"
                    )
                    if ok and interval:
                        self.series_custom_intervals[k] = interval
                        sel = "Custom"
                    else:
                        sel = self.series_resample_intervals.get(k, "Original")
                else:
                    sel = txt

                 # update intervals and default methods
                self.series_resample_intervals[k] = sel
                if sel == "Original":
                    self.series_resample_methods[k] = "direct"
                elif k not in self.series_resample_methods or \
                     self.series_resample_methods[k] == "direct":
                    self.series_resample_methods[k] = "mean"
                self._on_tree_selection()
            res_cb.currentTextChanged.connect(on_res_change)
                
            # Resample method
            method_cb = QComboBox()
            methods = ["direct","sum","last","mean","max","min"]
            method_cb.addItems(methods)
            default_m = self.series_resample_methods.get(
                key,
                "direct" if self.series_resample_intervals.get(key,"Original")=="Original" else "mean"
            )
            method_cb.setCurrentText(default_m)
            method_cb.currentTextChanged.connect(lambda m,k=key: self._on_method_change(k,m))

            # Assemble row
            row = QWidget()
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(QLabel("Type:"));     hl.addWidget(type_cb)
            hl.addSpacing(10)
            hl.addWidget(QLabel("Resample:")); hl.addWidget(res_cb)
            hl.addSpacing(10)
            hl.addWidget(QLabel("Method:"));   hl.addWidget(method_cb)
            form.addRow(col, row)

    def _on_type_change(self, key, t):
        self.series_plot_types[key] = t
        self._on_tree_selection()

    def _on_method_change(self, key, m):
        self.series_resample_methods[key] = m
        self._on_tree_selection()


    def _on_tree_selection(self):
        smap = {}
        for itm in self.tree.selectedItems():
            data = itm.data(0, Qt.UserRole)
            if isinstance(data, tuple):
                fn, col = data
                df = self.dataframes.get(fn)
                if df is None or col not in df: continue
                s = df[col].dropna()
                if pd.api.types.is_numeric_dtype(s):
                    smap[f"{fn}:{col}"] = s
        if not smap:
            return
        self._last_series_map = smap
        self.progress_bar.setVisible(True)
        worker = PlotWorker(
            smap,
            self.series_resample_intervals,
            self.series_custom_intervals,
            self.series_resample_methods
        )
        worker.signals.finished.connect(self._on_plot_ready)
        self.threadpool.start(worker)

    def _on_plot_ready(self, result):
        self.progress_bar.setVisible(False)
        processed, table_df, stats = result
        self._update_table(table_df)
        self._update_stats(stats)
        self._adjust_stats_splitter()
        self._draw_plot(processed)

    def _update_table(self, df: pd.DataFrame):
        tw = self.table_widget
        raw = df.columns.tolist()
        pretty = [c.split(':', 1)[-1] for c in raw]
        tw.setColumnCount(len(raw))
        tw.setRowCount(len(df))
        tw.setHorizontalHeaderLabels(pretty)
        for i in range(len(df)):
            for j, c in enumerate(raw):
                tw.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))
        tw.resizeColumnsToContents()
        tw.resizeRowsToContents()

    def _update_stats(self, stats: list):
        df = pd.DataFrame(stats)
        df['Series'] = df['Series'].str.split(pat=':', n=1).str[-1]

        cols = ['Series','dtype','Min','Max','NaNs','Inf','-Inf',
                'Mean','Median','Std','Skewness','Kurtosis']
        html = df[cols].to_html(
            index=False,
            float_format=lambda x: f"{x:.3f}" if isinstance(x, (float, np.floating)) else x
        )
        self.stats_panel.setHtml(html)

    def _adjust_stats_splitter(self):
        doc_h = self.stats_panel.document().size().height()
        stats_h = int(doc_h + 10)
        total_h = self.graph_splitter.height()
        plot_h = max(100, total_h - stats_h)
        self.graph_splitter.setSizes([plot_h, stats_h])

    def _draw_plot(self, processed: dict):
        import math
        # for rendering correlation heatmap via QGraphicsView with zoom
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import matplotlib.pyplot as plt
        import seaborn as sns
        from PyQt5.QtWidgets import QWidget, QGraphicsView, QGraphicsScene, QVBoxLayout, QSizePolicy
        from PyQt5.QtGui import QImage, QPixmap, QPainter
        from PyQt5.QtCore import Qt

        # subclass to handle wheel zooming
        class ZoomableGraphicsView(QGraphicsView):
            def wheelEvent(self, event):
                zoomInFactor = 1.25
                zoomOutFactor = 1 / zoomInFactor
                if event.angleDelta().y() > 0:
                    self.scale(zoomInFactor, zoomInFactor)
                else:
                    self.scale(zoomOutFactor, zoomOutFactor)

        mode = self.mode_combo.currentText()
        N = len(processed)

        # Teardown any previous view widget in the graph area
        for attr in ('plot_widget', '_grid_container', '_active_widget'):
            if hasattr(self, attr):
                w = getattr(self, attr)
                try:
                    if self.graph_splitter.indexOf(w) != -1:
                        w.setParent(None)
                except Exception:
                    pass

        # Overlay Mode
        if mode == "Overlay" or N <= 1:
            # create/re-create plot_widget
            self.plot_widget = pg.PlotWidget(title="Data Visualization")
            for ax_name in ("left", "bottom"):
                axis = self.plot_widget.getAxis(ax_name)
                axis.setPen(pg.mkPen("w"))
                axis.setTextPen(pg.mkPen("w"))
            self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
            self.graph_splitter.insertWidget(0, self.plot_widget)

            pw = self.plot_widget
            pw.clear()
            pw.addLegend()
            pw.showGrid(x=True, y=True, alpha=0.2)

            for lbl, s2 in processed.items():
                x = np.arange(len(s2))
                y = s2.values
                x_ds, y_ds = downsample_for_width(x, y, max_points=pw.width())
                typ = self.series_plot_types.get(lbl, "Line")
                pen = pg.mkPen(color=pg.intColor(hash(lbl) & 0xFFFFFF), width=2)
                if typ == "Line":
                    pw.plot(x_ds, y_ds, pen=pen, name=lbl.split(':',1)[-1])  # show only column name
                elif typ == "Scatter":
                    pw.plot(x_ds, y_ds, pen=pg.mkPen(None), symbol='o', symbolSize=6,
                            symbolBrush=pen.color(), name=lbl.split(':',1)[-1])
                else:
                    y_clean = y_ds[np.isfinite(y_ds)]
                    if y_clean.size > 0:
                        try:
                            cnt, ed = np.histogram(y_clean, bins='auto')
                        except ValueError:
                            cnt, ed = np.histogram(y_clean, bins=50)
                        ctr = (ed[:-1] + ed[1:]) / 2
                        bg = pg.BarGraphItem(x=ctr, height=cnt,
                                              width=(ed[1]-ed[0]) * 0.8,
                                              brush=pen.color())
                        pw.addItem(bg)
            pw.enableAutoRange()

        # Correlation Mode
        elif mode == "Correlation" and N > 1:
            # build aligned DataFrame & compute correlation
            df = pd.concat(processed.values(), axis=1, keys=processed.keys())
            corr = df.corr()
            # shorten series keys to column names only
            labels = [lbl.split(':',1)[-1] for lbl in corr.columns]
            corr.columns = labels
            corr.index = labels

            # draw heatmap
            fig = plt.Figure(figsize=(max(6, corr.shape[1]*0.6),
                                      max(6, corr.shape[0]*0.6)), dpi=100)
            ax = fig.add_subplot(111)
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='viridis', ax=ax)
            fig.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.3)
            # render to image
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            w_img, h_img = canvas.get_width_height()
            buf = canvas.buffer_rgba()
            img = QImage(buf, w_img, h_img, QImage.Format_RGBA8888)
            pix = QPixmap.fromImage(img)
            # put into scene/view
            scene = QGraphicsScene(self)
            scene.addPixmap(pix)
            rect = scene.itemsBoundingRect()
            scene.setSceneRect(rect.adjusted(-rect.width()*0.5,
                                             -rect.height()*0.5,
                                               rect.width()*1.5,
                                               rect.height()*1.5))
            view = ZoomableGraphicsView(scene)
            view.setDragMode(QGraphicsView.ScrollHandDrag)
            view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            view.viewport().setCursor(Qt.OpenHandCursor)
            view.setRenderHint(QPainter.Antialiasing)
            view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0,0,0,0)
            layout.addWidget(view)
            self._active_widget = container
            self.graph_splitter.insertWidget(0, container)

        # Grid Mode
        else:
            cols = math.ceil(math.sqrt(N))
            rows = math.ceil(N/cols)
            container = QWidget()
            self._grid_container = container
            gv = QVBoxLayout(container)
            it = iter(processed.items())
            for _ in range(rows):
                row_w = QWidget()
                hl = QHBoxLayout(row_w)
                for _ in range(cols):
                    try:
                        lbl, s2 = next(it)
                    except StopIteration:
                        break
                    w = pg.PlotWidget(title=lbl.split(':',1)[-1])  # show only column name
                    w.showGrid(x=True, y=True, alpha=0.2)
                    for ax_name in ("left", "bottom"):
                        axis = w.getAxis(ax_name)
                        axis.setPen(pg.mkPen("w"))
                        axis.setTextPen(pg.mkPen("w"))
                    x = np.arange(len(s2))
                    y = s2.values
                    x_ds, y_ds = downsample_for_width(x, y, max_points=w.width())
                    typ = self.series_plot_types.get(lbl, "Line")
                    pen = pg.mkPen(color=pg.intColor(hash(lbl)&0xFFFFFF), width=2)
                    if typ == "Line":
                        w.plot(x_ds, y_ds, pen=pen)
                    elif typ == "Scatter":
                        w.plot(x_ds, y_ds, pen=pg.mkPen(None), symbol='o', symbolSize=6,
                               symbolBrush=pen.color())
                    else:
                        y_clean = y_ds[np.isfinite(y_ds)]
                        if y_clean.size > 0:
                            try:
                                cnt, ed = np.histogram(y_clean, bins='auto')
                            except ValueError:
                                cnt, ed = np.histogram(y_clean, bins=50)
                            ctr = (ed[:-1] + ed[1:])/2
                            bg = pg.BarGraphItem(x=ctr, height=cnt,
                                                  width=(ed[1]-ed[0])*0.8,
                                                  brush=pg.mkBrush('#ffaa00'))
                            w.addItem(bg)
                    w.enableAutoRange()
                    hl.addWidget(w)
                gv.addWidget(row_w)
            self.graph_splitter.insertWidget(0, container)

        self.view_tabs.setCurrentIndex(0)

    def _show_formula_help(self):
        text = (
            "<b>Formula Help</b><br><br>"
            "• Reference columns as <code>df['col_name']</code>.<br>"
            "• Use +, -, *, /, and parentheses.<br>"
            "• Supported functions: your TRANSFORMATIONS, <code>np</code>, <code>pd</code>.<br>"
            "• Example: <code>df['price'] * df['quantity']</code>.<br>"
            "• More: https://github.com/ZENTCH-Q/QLab/blob/Release-Candidate-(RC)/README.md"
        )
        QMessageBox.information(self, "Formula Usage", text)

    def _on_mode_change(self, mode: str):
        self._on_tree_selection()

    def _on_grid_resize(self, _=None):
        if self._last_series_map:
            self._on_tree_selection()

    def _apply_formula(self):
        self.error_label.clear()
        formula = self.formula_edit.toPlainText().strip()
        if not formula:
            return
        tokens = parse_formula(formula)
        if not tokens:
            self.error_label.setText("No columns found in formula")
            return
        series_map = {}
        for tok in tokens:
            matches = [(n, df) for n, df in self.dataframes.items() if tok in df.columns]
            if not matches:
                self.error_label.setText(f"Column '{tok}' not found")
                return
            if len(matches) > 1:
                self.error_label.setText(f"Ambiguous '{tok}' in multiple files")
                return
            name, df = matches[0]
            series_map[tok] = df[tok].dropna()
        merged = pd.concat(series_map.values(), axis=1, keys=series_map.keys(), join='inner')
        if merged.empty:
            self.error_label.setText("No overlapping rows")
            return
        try:
            new_df, new_col = evaluate_formula_on_df(merged, formula)
        except Exception as e:
            self.error_label.setText(f"Formula error: {e}")
            return
        pseudo = f"Formula:{new_col}"
        self.dataframes[pseudo] = new_df[[new_col]]
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if it.data(0, Qt.UserRole) == pseudo:
                self.tree.takeTopLevelItem(i)
                break
        top = QTreeWidgetItem(self.tree, [pseudo])
        top.setData(0, Qt.UserRole, pseudo)
        child = QTreeWidgetItem(top, [new_col])
        child.setData(0, Qt.UserRole, (pseudo, new_col))
        top.setExpanded(True)
