import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import QSize, Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QFont, QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kedge_app.mpr.widgets import MPRPanel
from kedge_app.pipeline.core import (
    APP_ROOT,
    DEFAULT_OUTPUT_DIR,
    CaseRecord,
    build_case_thumbnail,
    case_id_from_path,
    discover_case_directory,
    discover_dicom_files,
    sanitize_case_id,
)
from kedge_app.ui.report_viewer import UnifiedCaseViewer
from kedge_app.worker.batch_manager import BatchManager
from kedge_app.worker.preview_loader import PreviewLoadManager

APP_ICON_PATH = APP_ROOT / "reconstructed_weight_maps" / "logo" / "Gemini_Generated_Image_v85s48v85s48v85s.png"


class KEdgeMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("K-edge 解析程序")
        self.resize(1720, 1060)
        self.setAcceptDrops(True)
        if APP_ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))

        self.output_root = DEFAULT_OUTPUT_DIR
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.case_records: Dict[str, CaseRecord] = {}
        self.volume_cache: Dict[str, np.ndarray] = {}
        self.processed_volume_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.queue_row_map: Dict[str, int] = {}
        self.thumb_item_map: Dict[str, QListWidgetItem] = {}
        self.current_case_id: Optional[str] = None
        self._selecting_case = False
        self._last_progress_percent = -1

        self.batch_manager = BatchManager(self)
        self.preview_loader = PreviewLoadManager(self)
        self._build_ui()
        self._apply_theme()
        self._bind_signals()
        self._load_initial_state()

    def _build_ui(self) -> None:
        central = QWidget()
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setObjectName("mainSplitter")
        main_splitter.setHandleWidth(1)
        root_layout.addWidget(main_splitter)
        main_splitter.addWidget(self._build_sidebar())
        main_splitter.addWidget(self._build_content_area())
        main_splitter.setSizes([410, 1300])
        self.setCentralWidget(central)

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("sidebarPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        hero = QFrame()
        hero.setObjectName("sidebarHero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(14, 14, 14, 14)
        hero_layout.setSpacing(6)

        title = QLabel("K-edge Workstation")
        title.setObjectName("appTitle")
        hero_layout.addWidget(title)

        subtitle = QLabel("苹果式浅色蓝极简界面，结合专业影像工作流。先拖拽病例，再预览 MPR，最后批处理生成报告。")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subTitle")
        hero_layout.addWidget(subtitle)
        layout.addWidget(hero)

        drop_box = QFrame()
        drop_box.setObjectName("dropBox")
        drop_layout = QVBoxLayout(drop_box)
        drop_layout.setContentsMargins(14, 14, 14, 14)
        drop_layout.setSpacing(6)
        drop_title = QLabel("拖拽导入")
        drop_title.setObjectName("cardTitle")
        drop_layout.addWidget(drop_title)
        hint = QLabel("支持拖拽单个 DICOM、多个 DICOM 或病例目录。\n双击病例缩略图可直接打开当前病例输出目录。")
        hint.setWordWrap(True)
        hint.setObjectName("cardHint")
        drop_layout.addWidget(hint)
        layout.addWidget(drop_box)

        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(8)
        self.btn_add_files = QPushButton("添加文件")
        self.btn_add_files.setObjectName("secondaryButton")
        self.btn_add_folder = QPushButton("添加目录")
        self.btn_add_folder.setObjectName("secondaryButton")
        btn_row1.addWidget(self.btn_add_files)
        btn_row1.addWidget(self.btn_add_folder)
        layout.addLayout(btn_row1)

        from PyQt5.QtWidgets import QComboBox
        backend_row = QHBoxLayout()
        backend_row.setSpacing(8)
        backend_label = QLabel("计算后端:")
        backend_label.setObjectName("cardTitle")
        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["默认 (AUTO)", "CUDA (异构混合)", "CUDA_FULL (强制全GPU硬解)", "CPU (纯软件)"])
        backend_row.addWidget(backend_label)
        backend_row.addWidget(self.combo_backend, 1)
        layout.addLayout(backend_row)

        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(8)
        self.btn_start = QPushButton("开始并发批处理")
        self.btn_start.setObjectName("primaryButton")
        self.btn_clear = QPushButton("清空队列")
        self.btn_clear.setObjectName("utilityButton")
        btn_row2.addWidget(self.btn_start)
        btn_row2.addWidget(self.btn_clear)
        layout.addLayout(btn_row2)

        btn_row3 = QHBoxLayout()
        btn_row3.setSpacing(8)
        self.btn_open_output = QPushButton("打开当前输出目录")
        self.btn_open_output.setObjectName("utilityButton")
        self.btn_export_mpr = QPushButton("导出当前 MPR")
        self.btn_export_mpr.setObjectName("utilityButton")
        self.btn_export_report = QPushButton("导出当前报告")
        self.btn_export_report.setObjectName("utilityButton")
        btn_row3.addWidget(self.btn_open_output)
        btn_row3.addWidget(self.btn_export_mpr)
        btn_row3.addWidget(self.btn_export_report)
        layout.addLayout(btn_row3)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        thumbs_group = QGroupBox("病例浏览")
        thumbs_layout = QVBoxLayout(thumbs_group)
        thumbs_layout.setContentsMargins(10, 14, 10, 10)
        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setObjectName("thumbnailList")
        self.thumbnail_list.setViewMode(QListWidget.IconMode)
        self.thumbnail_list.setResizeMode(QListWidget.Adjust)
        self.thumbnail_list.setIconSize(QSize(132, 132))
        self.thumbnail_list.setGridSize(QSize(158, 182))
        self.thumbnail_list.setSpacing(8)
        thumbs_layout.addWidget(self.thumbnail_list)
        layout.addWidget(thumbs_group, 2)

        queue_group = QGroupBox("处理队列")
        queue_layout = QVBoxLayout(queue_group)
        queue_layout.setContentsMargins(10, 14, 10, 10)
        self.queue_table = QTableWidget(0, 4)
        self.queue_table.setObjectName("queueTable")
        self.queue_table.setHorizontalHeaderLabels(["病例", "状态", "文件数", "输出目录"])
        self.queue_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.queue_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.queue_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.queue_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.queue_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.queue_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.queue_table.setAlternatingRowColors(True)
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.setShowGrid(False)
        queue_layout.addWidget(self.queue_table)
        layout.addWidget(queue_group, 2)

        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(10, 14, 10, 10)
        self.log_output = QPlainTextEdit()
        self.log_output.setObjectName("logOutput")
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)
        layout.addWidget(log_group, 2)
        return panel

    def _build_content_area(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("workspacePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 2, 2, 2)
        layout.setSpacing(12)

        status_card = QFrame()
        status_card.setObjectName("statusCard")
        status_layout = QHBoxLayout(status_card)
        status_layout.setContentsMargins(16, 12, 16, 12)
        status_layout.setSpacing(10)
        self.current_case_label = QLabel("当前病例: 未加载")
        self.current_case_label.setObjectName("sectionTitle")
        status_layout.addWidget(self.current_case_label, 1)
        self.status_label = QLabel("状态: 等待拖拽 DICOM 或病例目录")
        self.status_label.setObjectName("viewerInfoLabel")
        status_layout.addWidget(self.status_label)
        layout.addWidget(status_card)

        splitter = QSplitter(Qt.Vertical)
        splitter.setObjectName("contentSplitter")
        splitter.setHandleWidth(1)
        layout.addWidget(splitter)

        self.workspace_viewer = UnifiedCaseViewer()
        splitter.addWidget(self.workspace_viewer)

        self.mpr_panel = MPRPanel()
        splitter.addWidget(self.mpr_panel)
        splitter.setSizes([620, 360])
        return panel

    def _bind_signals(self) -> None:
        self.btn_add_files.clicked.connect(self.add_files)
        self.btn_add_folder.clicked.connect(self.add_folder)
        self.btn_start.clicked.connect(self.start_batch)
        self.btn_clear.clicked.connect(self.clear_queue)
        self.btn_open_output.clicked.connect(self.open_current_output)
        self.btn_export_mpr.clicked.connect(self.export_current_mpr)
        self.btn_export_report.clicked.connect(self.export_current_report)
        self.queue_table.itemSelectionChanged.connect(self.on_queue_selection_changed)
        self.thumbnail_list.itemClicked.connect(self.on_thumbnail_clicked)
        self.thumbnail_list.itemDoubleClicked.connect(self.on_thumbnail_double_clicked)

        self.batch_manager.log_message.connect(self.log)
        self.batch_manager.case_started.connect(self.on_case_started)
        self.batch_manager.case_stage.connect(self.on_case_stage)
        self.batch_manager.case_done.connect(self.on_case_done)
        self.batch_manager.case_failed.connect(self.on_case_failed)
        self.batch_manager.progress.connect(self.on_worker_progress)
        self.batch_manager.finished.connect(self.on_worker_finished)

        self.preview_loader.load_started.connect(self.on_preview_started)
        self.preview_loader.load_stage.connect(self.on_preview_stage)
        self.preview_loader.load_done.connect(self.on_preview_done)
        self.preview_loader.load_failed.connect(self.on_preview_failed)

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background:#f6f8fb;
                color:#1f2937;
            }
            #mainSplitter::handle, #contentSplitter::handle {
                background:#e8edf3;
            }
            #sidebarPanel {
                background:#f3f6fa;
                border-right:1px solid #e5eaf0;
                border-radius:18px;
            }
            #workspacePanel {
                background:#f7f9fc;
                border-radius:18px;
            }
            #sidebarHero, #statusCard {
                background:qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #eef6ff);
                border:1px solid #d5e4f7;
                border-radius:18px;
            }
            #appTitle {
                font-size:22px;
                font-weight:700;
                color:#111827;
                padding:0;
            }
            #subTitle {
                color:#6b7280;
                padding-bottom:0;
            }
            #dropBox {
                background:#fbfdff;
                border:1px dashed #d5dee8;
                border-radius:16px;
                padding:12px;
            }
            #cardTitle {
                color:#0f172a;
                font-size:14px;
                font-weight:700;
            }
            #cardHint {
                color:#6b7280;
            }
            #sectionTitle {
                font-size:18px;
                font-weight:600;
                color:#0f172a;
                padding-bottom:4px;
            }
            #fallbackNote {
                background:#f8fbff;
                color:#4b5563;
                border:1px solid #dde7f3;
                border-radius:12px;
                padding:10px;
            }
            #viewerInfoLabel {
                color:#6b7280;
                padding:4px;
            }
            QPushButton {
                background:#ffffff;
                border:1px solid #dde5ee;
                border-radius:12px;
                padding:8px 13px;
                color:#1f2937;
                font-weight:600;
            }
            QPushButton:hover {
                background:#f7fbff;
                border:1px solid #c7dbf5;
            }
            QPushButton:pressed {
                background:#eaf3ff;
            }
            QPushButton:disabled {
                background:#f3f4f6;
                border:1px solid #e5e7eb;
                color:#9ca3af;
            }
            #primaryButton {
                background:qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #06b6d4);
                border:1px solid #2563eb;
                color:#ffffff;
            }
            #primaryButton:hover {
                background:qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #22d3ee);
                border:1px solid #3b82f6;
            }
            #primaryButton:pressed {
                background:#1d4ed8;
            }
            #secondaryButton {
                background:#fbfdff;
            }
            #utilityButton {
                background:#f6f8fb;
                color:#4b5563;
            }
            QGroupBox {
                border:1px solid #dbe7f4;
                border-radius:18px;
                margin-top:10px;
                padding-top:14px;
                font-weight:700;
                color:#0f172a;
                background:qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(255,255,255,0.96), stop:1 rgba(241,247,255,0.96));
            }
            QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 6px; }
            QTableWidget, QPlainTextEdit, QTextEdit, QScrollArea, QListWidget {
                background:#ffffff;
                border:1px solid #e5eaf0;
                border-radius:14px;
                color:#1f2937;
            }
            QTableWidget {
                alternate-background-color:#f9fbfd;
                selection-background-color:#edf5ff;
            }
            QHeaderView::section {
                background:#f8fafc;
                color:#6b7280;
                padding:8px 6px;
                border:none;
                font-weight:600;
            }
            QTableWidget::item:selected, QListWidget::item:selected {
                background:#dbeafe;
                color:#0f172a;
            }
            QListWidget::item {
                background:#ffffff;
                border:1px solid #e5edf6;
                border-radius:14px;
                padding:6px;
                margin:3px;
            }
            QPushButton:checked {
                background:#dbeafe;
                border:1px solid #60a5fa;
                color:#0f172a;
            }
            #logOutput {
                font-family:Consolas, 'SF Mono', monospace;
                font-size:12px;
                color:#334155;
            }
            #pipelineCard {
                background:qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #f0f9ff);
                border:1px solid #bfdbfe;
                border-radius:16px;
            }
            #mprPanel {
                background:rgba(255,255,255,0.86);
                border:1px solid #e5eaf0;
                border-radius:18px;
            }
            QSlider::groove:horizontal {
                height:6px;
                background:#e8edf3;
                border-radius:3px;
            }
            QSlider::handle:horizontal {
                width:16px;
                margin:-5px 0;
                background:#ffffff;
                border:1px solid #c7d5e6;
                border-radius:8px;
            }
            QSlider::sub-page:horizontal {
                background:qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #60a5fa, stop:1 #22d3ee);
                border-radius:3px;
            }
            QProgressBar {
                border:1px solid #e5eaf0;
                border-radius:10px;
                text-align:center;
                background:#ffffff;
                color:#4b5563;
                min-height:20px;
            }
            QProgressBar::chunk {
                background:qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:0.55 #3b82f6, stop:1 #06b6d4);
                border-radius:10px;
            }
            """
        )

    def _show_pending_reports(self, case_id: Optional[str] = None) -> None:
        suffix = f"当前病例：{case_id}" if case_id else "当前还没有病例"
        self.workspace_viewer.set_placeholder(
            "统一 K-edge 工作站",
            f"尚未执行处理。{suffix}。处理完成后，这里会在同一界面顶部显示 pipeline，中部同步显示全部派生 DCM，底部保留 MPR。",
            "等待处理",
        )

    def _load_initial_state(self) -> None:
        self._show_pending_reports()

    def log(self, text: str) -> None:
        self.log_output.appendPlainText(text)

    def _array_to_icon(self, arr: np.ndarray) -> QIcon:
        arr = np.asarray(arr, dtype=np.float32)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            gray = np.zeros((256, 256), dtype=np.uint8)
        else:
            lo = float(np.percentile(valid, 1))
            hi = float(np.percentile(valid, 99))
            if hi - lo < 1e-6:
                gray = np.zeros_like(arr, dtype=np.uint8)
            else:
                gray = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
                gray = (gray * 255.0).astype(np.uint8)
        if gray.ndim != 2:
            gray = np.zeros((256, 256), dtype=np.uint8)
        h, w = gray.shape
        qimg = QImage(gray.data, w, h, gray.strides[0], QImage.Format_Grayscale8).copy()
        pixmap = QPixmap.fromImage(qimg).scaled(132, 132, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QIcon(pixmap)

    def _make_thumbnail_item(self, case_id: str, case_dir: Path, file_count: int) -> QListWidgetItem:
        try:
            thumb = build_case_thumbnail(case_dir)
            icon = self._array_to_icon(thumb)
        except Exception:
            icon = self.style().standardIcon(QApplication.style().SP_FileIcon)
        item = QListWidgetItem(icon, f"{case_id}\n{file_count} slices")
        item.setData(Qt.UserRole, case_id)
        item.setToolTip(str(case_dir))
        return item

    def add_case_dir(self, case_dir: Path) -> None:
        case_id = sanitize_case_id(case_id_from_path(case_dir))
        if case_id in self.case_records:
            self.log(f"已跳过重复病例: {case_id}")
            return

        try:
            file_count = len(discover_dicom_files(case_dir))
        except Exception:
            file_count = 0

        record = CaseRecord(case_dir=case_dir, case_id=case_id, status="待处理")
        self.case_records[case_id] = record

        row = self.queue_table.rowCount()
        self.queue_table.insertRow(row)
        self.queue_table.setItem(row, 0, QTableWidgetItem(case_id))
        self.queue_table.setItem(row, 1, QTableWidgetItem(record.status))
        self.queue_table.setItem(row, 2, QTableWidgetItem(str(file_count)))
        self.queue_table.setItem(row, 3, QTableWidgetItem(""))
        self.queue_table.item(row, 0).setData(Qt.UserRole, case_id)
        self.queue_row_map[case_id] = row

        thumb_item = self._make_thumbnail_item(case_id, case_dir, file_count)
        self.thumbnail_list.addItem(thumb_item)
        self.thumb_item_map[case_id] = thumb_item
        self.log(f"已加入病例: {case_dir}")

    def add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "选择 DICOM 文件", str(APP_ROOT), "All Files (*)")
        self.add_paths(files)

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择病例目录", str(APP_ROOT))
        if folder:
            self.add_paths([folder])

    def add_paths(self, paths: List[str]) -> None:
        first_added = None
        for path_str in paths:
            case_dir = discover_case_directory(path_str)
            if case_dir is None:
                self.log(f"无法识别为病例路径: {path_str}")
                continue
            if first_added is None:
                first_added = case_dir
            self.add_case_dir(case_dir)
        if first_added is not None:
            case_id = sanitize_case_id(case_id_from_path(first_added))
            self.select_case(case_id)

    def clear_queue(self) -> None:
        if self.batch_manager.is_running():
            QMessageBox.warning(self, "提示", "批处理正在运行，无法清空队列。")
            return
        self.case_records.clear()
        self.queue_row_map.clear()
        self.thumb_item_map.clear()
        self.queue_table.setRowCount(0)
        self.thumbnail_list.clear()
        self.progress.setValue(0)
        self._last_progress_percent = -1
        self.current_case_id = None
        self.current_case_label.setText("当前病例: 未加载")
        self.status_label.setText("状态: 等待拖拽 DICOM 或病例目录")
        self._show_pending_reports()
        self.log("已清空队列。")

    def start_batch(self) -> None:
        pending = [record.case_dir for record in self.case_records.values() if record.status != "完成"]
        if not pending:
            QMessageBox.information(self, "提示", "队列中没有待处理病例。")
            return
        if self.batch_manager.is_running():
            QMessageBox.information(self, "提示", "批处理已经在运行。")
            return
            
        backend_mapping = {
            "默认 (AUTO)": "auto",
            "CUDA (异构混合)": "cuda",
            "CUDA_FULL (强制全GPU硬解)": "cuda-full",
            "CPU (纯软件)": "cpu"
        }
        selected_text = self.combo_backend.currentText()
        backend_pref = backend_mapping.get(selected_text, "auto")
            
        self.progress.setValue(0)
        self._last_progress_percent = -1
        self.btn_start.setEnabled(False)
        self.combo_backend.setEnabled(False)
        self.status_label.setText(f"状态: 正在并发处理 {len(pending)} 个病例")
        self.log(f"开始并发批处理，共 {len(pending)} 个病例。后端: {selected_text}")
        self.batch_manager.start(pending, self.output_root, backend_preference=backend_pref)

    def on_case_started(self, case_id: str) -> None:
        record = self.case_records.get(case_id)
        if record:
            record.status = "处理中"
            record.stage = "扫描 DICOM"
            self._refresh_case_visuals(case_id)
        self.status_label.setText(f"状态: {case_id} 已提交处理")
        self.log(f"{case_id} 已提交到线程池。")

    def on_case_stage(self, case_id: str, stage: str, detail: str) -> None:
        record = self.case_records.get(case_id)
        if record:
            record.status = f"处理中 / {stage}"
            record.stage = stage
            self._refresh_case_visuals(case_id)
        if self.current_case_id == case_id:
            self.status_label.setText(f"状态: {stage}")
        self.log(detail)

    def on_worker_progress(self, current: int, total: int, text: str) -> None:
        percent = int(current / max(total, 1) * 100)
        self.progress.setValue(percent)
        self.status_label.setText(f"状态: {text}")
        if percent != self._last_progress_percent:
            self._last_progress_percent = percent
            self.log(f"{text} | 总进度 {current}/{total} ({percent}%)")

    def on_case_done(self, case_id: str, result: dict) -> None:
        record = self.case_records.get(case_id)
        if not record:
            return
        record.status = "完成"
        record.stage = "完成"
        record.output_dir = Path(result["output_dir"])
        record.result = result
        self._refresh_case_visuals(case_id)
        self.log(f"{case_id} 已输出到 {record.output_dir}")
        if self.current_case_id == case_id:
            self.workspace_viewer.load_case(
                Path(record.result["native_iodine"]),
                Path(record.result["native_compare"]),
                Path(record.result["native_suite"]),
                Path(record.result["output_dir"]),
            )
            self._load_processed_mpr_layers(case_id, Path(record.result["volume_npz"]))
            self.status_label.setText("状态: 完成")
        elif not self.current_case_id:
            self.select_case(case_id)

    def on_case_failed(self, case_id: str, error_text: str) -> None:
        record = self.case_records.get(case_id)
        if record:
            record.status = "失败"
            record.stage = "失败"
            self._refresh_case_visuals(case_id)
        self.log(f"{case_id} 处理失败:\n{error_text}")

    def on_worker_finished(self) -> None:
        self.progress.setValue(100)
        self._last_progress_percent = 100
        self.btn_start.setEnabled(True)
        self.combo_backend.setEnabled(True)
        self.status_label.setText("状态: 批处理结束")
        self.log("批处理结束。")

    def _refresh_case_visuals(self, case_id: str) -> None:
        row = self.queue_row_map.get(case_id)
        record = self.case_records.get(case_id)
        if row is not None and record:
            self.queue_table.item(row, 1).setText(record.status)
            self.queue_table.item(row, 3).setText(str(record.output_dir or ""))
        item = self.thumb_item_map.get(case_id)
        if item and record:
            text = item.text().split("\n")[0]
            suffix = record.status
            item.setText(f"{text}\n{suffix}")

    def on_queue_selection_changed(self) -> None:
        if self._selecting_case:
            return
        rows = self.queue_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        item = self.queue_table.item(row, 0)
        if item:
            case_id = item.data(Qt.UserRole)
            self.select_case(case_id)

    def on_thumbnail_clicked(self, item: QListWidgetItem) -> None:
        if self._selecting_case:
            return
        case_id = item.data(Qt.UserRole)
        self.select_case(case_id)

    def on_thumbnail_double_clicked(self, item: QListWidgetItem) -> None:
        case_id = item.data(Qt.UserRole)
        self.select_case(case_id)
        record = self.case_records.get(case_id)
        if record and record.result:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(record.result["output_dir"])))

    def select_case(self, case_id: str) -> None:
        record = self.case_records.get(case_id)
        if not record:
            return
        self.current_case_id = case_id
        self.current_case_label.setText(f"当前病例: {case_id}")
        record_status = record.status
        self.status_label.setText(f"状态: {record_status}")

        self._selecting_case = True
        row = self.queue_row_map.get(case_id)
        if row is not None:
            self.queue_table.selectRow(row)
        thumb_item = self.thumb_item_map.get(case_id)
        if thumb_item:
            self.thumbnail_list.setCurrentItem(thumb_item)
        self._selecting_case = False

        if record.result:
            self.workspace_viewer.load_case(
                Path(record.result["native_iodine"]),
                Path(record.result["native_compare"]),
                Path(record.result["native_suite"]),
                Path(record.result["output_dir"]),
            )
            self._load_processed_mpr_layers(case_id, Path(record.result["volume_npz"]))
        else:
            self._show_pending_reports(case_id)

        if case_id in self.processed_volume_cache:
            self.mpr_panel.load_layers(self.processed_volume_cache[case_id], current="原始")
            self.log(f"3D 重建体数据已就绪: {record.case_dir}")
            return

        if case_id in self.volume_cache:
            self.mpr_panel.load_layers({"原始": self.volume_cache[case_id]}, current="原始")
            self.log(f"MPR 已就绪: {record.case_dir}")
            return

        self.status_label.setText("状态: 扫描 DICOM")
        self.preview_loader.load(case_id, record.case_dir)

    def on_preview_started(self, case_id: str) -> None:
        record = self.case_records.get(case_id)
        if record and record.status == "待处理":
            record.status = "预览中 / 扫描 DICOM"
            record.stage = "扫描 DICOM"
            self._refresh_case_visuals(case_id)
        if record and self.current_case_id == case_id and not record.result:
            self.status_label.setText("状态: 扫描 DICOM")
        self.log(f"{case_id} 开始后台加载 MPR 预览。")

    def on_preview_stage(self, case_id: str, stage: str, detail: str) -> None:
        record = self.case_records.get(case_id)
        if record and record.status.startswith("预览中"):
            record.status = f"预览中 / {stage}"
            record.stage = stage
            self._refresh_case_visuals(case_id)
        if self.current_case_id == case_id:
            self.status_label.setText(f"状态: {stage}")
        self.log(detail)

    def on_preview_done(self, case_id: str, volume: np.ndarray) -> None:
        self.volume_cache[case_id] = volume
        record = self.case_records.get(case_id)
        if record:
            if record.status.startswith("预览中"):
                record.status = "待处理"
                record.stage = ""
                self._refresh_case_visuals(case_id)
            self.log(f"MPR 已加载: {record.case_dir}")
        if self.current_case_id == case_id:
            self.mpr_panel.load_layers({"原始": volume}, current="原始")
            current_status = self.case_records[case_id].status if case_id in self.case_records else "待处理"
            self.status_label.setText(f"状态: {current_status}")

    def on_preview_failed(self, case_id: str, error_text: str) -> None:
        record = self.case_records.get(case_id)
        if record and record.status.startswith("预览中"):
            record.status = "待处理"
            record.stage = ""
            self._refresh_case_visuals(case_id)
        self.log(f"{case_id} 的 MPR 后台加载失败:\n{error_text}")
        if self.current_case_id == case_id:
            self.status_label.setText("状态: MPR 加载失败")

    def open_current_output(self) -> None:
        if not self.current_case_id:
            QMessageBox.information(self, "提示", "请先选择一个病例。")
            return
        record = self.case_records.get(self.current_case_id)
        if record and record.output_dir and record.output_dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(record.output_dir)))
        else:
            QMessageBox.information(self, "提示", "该病例还没有输出目录。")

    def export_current_mpr(self) -> None:
        case_id = self.current_case_id or "default"
        record = self.case_records.get(case_id) if case_id in self.case_records else None
        base_dir = record.output_dir if record and record.output_dir else (self.output_root / sanitize_case_id(case_id))
        export_dir = base_dir / "exports"
        target = export_dir / f"{sanitize_case_id(case_id)}_mpr.png"
        ok = self.mpr_panel.export_snapshot(target)
        if ok:
            self.log(f"MPR 截图已导出: {target}")
        else:
            self.log("MPR 截图导出失败。")

    def export_current_report(self) -> None:
        case_id = self.current_case_id or "default"
        record = self.case_records.get(case_id) if case_id in self.case_records else None
        base_dir = record.output_dir if record and record.output_dir else (self.output_root / sanitize_case_id(case_id))
        export_dir = base_dir / "exports"
        saved = self.workspace_viewer.export_snapshot(export_dir)
        if saved:
            self.log(f"报告快照已导出: {saved}")
        else:
            self.log("当前没有可导出的报告。")

    def _load_processed_mpr_layers(self, case_id: str, volume_npz: Path) -> None:
        if case_id in self.processed_volume_cache:
            self.mpr_panel.load_layers(self.processed_volume_cache[case_id], current="原始")
            return
        if not volume_npz.exists():
            return
        data = np.load(str(volume_npz), allow_pickle=True)
        layers = {
            "原始": np.asarray(data["pixel"], dtype=np.float32),
            "碘图": np.asarray(data["iodine"], dtype=np.float32),
            "K-edge": np.asarray(data["kedge"], dtype=np.float32),
        }
        self.processed_volume_cache[case_id] = layers
        self.mpr_panel.load_layers(layers, current="原始")

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        if not event.mimeData().hasUrls():
            super().dropEvent(event)
            return
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        self.add_paths(paths)
        event.acceptProposedAction()


def main() -> None:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("K-edge 解析程序")
    if APP_ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    app.setFont(QFont("Microsoft YaHei", 10))
    window = KEdgeMainWindow()
    window.show()
    sys.exit(app.exec_())
