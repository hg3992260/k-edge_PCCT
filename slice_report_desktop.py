import os
import sys
from pathlib import Path
from typing import List, Optional

import pydicom
from PyQt5.QtCore import QObject, Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QPlainTextEdit,
    QProgressBar,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

try:
    from PyQt5.QtWebEngineWidgets import QWebEngineProfile, QWebEngineView
except Exception:
    QWebEngineProfile = None
    QWebEngineView = None

from build_iodine_kedge_report import ROOT, generate_single_slice_report, ordered_dicom_files


def bundle_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return ROOT


ICON_DIR = bundle_root() / "reconstructed_weight_maps" / "logo"
OUTPUT_ROOT = ROOT / "ui_slice_reports"
QTWEBENGINE_CACHE_DIR = ROOT / ".qtwebengine_cache"
QTWEBENGINE_STORAGE_DIR = ROOT / ".qtwebengine_storage"


def resolve_app_icon() -> Optional[Path]:
    if not ICON_DIR.exists():
        return None
    for pattern in ("*.png", "*.ico", "*.jpg", "*.jpeg", "*.bmp"):
        matches = sorted(ICON_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def describe_slice(file_path: Path, index: int, total: int) -> str:
    extra = []
    try:
        ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
        if hasattr(ds, "InstanceNumber"):
            extra.append(f"Instance {int(ds.InstanceNumber)}")
        if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
            extra.append(f"Z={float(ds.ImagePositionPatient[2]):.2f}")
    except Exception:
        pass
    suffix = f" | {' | '.join(extra)}" if extra else ""
    return f"第 {index + 1} / {total} 层 | {file_path.name}{suffix}"


class DropFrame(QFrame):
    pathsDropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(8)

        title = QLabel("拖拽 DICOM 目录或单个 DCM 文件到这里")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("dropTitle")
        layout.addWidget(title)

        hint = QLabel("支持识别目录路径，或直接拖入目录下任意一个 DICOM 文件")
        hint.setAlignment(Qt.AlignCenter)
        hint.setObjectName("dropHint")
        layout.addWidget(hint)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                paths.append(url.toLocalFile())
        if paths:
            self.pathsDropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class AnalyzeWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, file_path: Path, output_root: Path):
        super().__init__()
        self.file_path = file_path
        self.output_root = output_root

    def run(self):
        try:
            self.progress.emit(f"开始分析切片 {self.file_path.name}")
            result = generate_single_slice_report(self.file_path, output_root=self.output_root)
            self.progress.emit("报告已生成，可在右侧直接预览")
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class HtmlPreviewWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._current_path: Optional[Path] = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.title_label = QLabel("HTML 报告预览")
        self.title_label.setObjectName("sectionLabel")
        layout.addWidget(self.title_label)

        if QWebEngineView is not None:
            self.mode = "webengine"
            self.view = QWebEngineView()
            layout.addWidget(self.view, 1)
        else:
            self.mode = "text"
            self.info_label = QLabel("当前环境未安装 Qt WebEngine，将以简化 HTML 方式预览，交互脚本可能不可用。")
            self.info_label.setWordWrap(True)
            self.info_label.setObjectName("previewHint")
            layout.addWidget(self.info_label)
            self.view = QTextBrowser()
            layout.addWidget(self.view, 1)

    def show_placeholder(self, message: str):
        self._current_path = None
        if self.mode == "webengine":
            self.view.setHtml(
                f"""
                <html><body style="background:#f4f9ff;color:#29435c;font-family:'Microsoft YaHei';padding:32px;">
                <h2 style="color:#2d8cff;">HTML 报告预览</h2>
                <p>{message}</p>
                </body></html>
                """
            )
        else:
            self.view.setHtml(
                f"""
                <html><body style="background:#f4f9ff;color:#29435c;font-family:'Microsoft YaHei';padding:24px;">
                <h2 style="color:#2d8cff;">HTML 报告预览</h2>
                <p>{message}</p>
                </body></html>
                """
            )

    def load_html(self, html_path: Path):
        self._current_path = html_path
        if self.mode == "webengine":
            self.view.setUrl(QUrl.fromLocalFile(str(html_path)))
        else:
            self.view.setSource(QUrl.fromLocalFile(str(html_path)))


class SliceReportMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.case_dir: Optional[Path] = None
        self.slice_files: List[Path] = []
        self.current_report_path: Optional[Path] = None
        self.current_output_dir: Optional[Path] = None
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[AnalyzeWorker] = None

        self.setWindowTitle("K-edge 单切片报告生成器")
        self.resize(1180, 820)
        self.setAcceptDrops(True)
        icon_path = resolve_app_icon()
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))

        self._build_ui()
        self._apply_theme()

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        title = QLabel("Light 亮色风 DICOM 单切片报告工具")
        title.setObjectName("titleLabel")
        root.addWidget(title)

        subtitle = QLabel("拖拽目录或 DCM 文件，选择切片序数后点击分析，并在右侧直接预览对应 HTML 报告")
        subtitle.setObjectName("subtitleLabel")
        root.addWidget(subtitle)

        self.drop_frame = DropFrame()
        self.drop_frame.pathsDropped.connect(self.load_paths)
        root.addWidget(self.drop_frame)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        self.btn_pick_dir = QPushButton("选择目录")
        self.btn_pick_file = QPushButton("选择 DCM 文件")
        self.btn_open_report = QPushButton("打开报告")
        self.btn_open_output = QPushButton("打开输出目录")
        self.btn_open_report.setEnabled(False)
        self.btn_open_output.setEnabled(False)
        btn_row.addWidget(self.btn_pick_dir)
        btn_row.addWidget(self.btn_pick_file)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_open_report)
        btn_row.addWidget(self.btn_open_output)
        root.addLayout(btn_row)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(16)

        info_card = QFrame()
        info_card.setObjectName("panel")
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(18, 18, 18, 18)
        info_layout.setSpacing(10)

        self.slice_count_label = QLabel("有效 DICOM 数: 0")
        self.privacy_label = QLabel("界面中不显示 DCM 路径，仅展示切片序数与文件名。")
        self.privacy_label.setWordWrap(True)
        info_layout.addWidget(self.slice_count_label)
        info_layout.addWidget(self.privacy_label)
        left_layout.addWidget(info_card)

        action_card = QFrame()
        action_card.setObjectName("panel")
        action_layout = QVBoxLayout(action_card)
        action_layout.setContentsMargins(18, 18, 18, 18)
        action_layout.setSpacing(12)

        combo_label = QLabel("切片序数")
        combo_label.setObjectName("sectionLabel")
        self.slice_combo = QComboBox()
        self.slice_combo.setMinimumHeight(42)
        self.btn_analyze = QPushButton("分析并生成 HTML 报告")
        self.btn_analyze.setMinimumHeight(46)
        self.btn_analyze.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.status_label = QLabel("等待输入路径")
        self.status_label.setObjectName("statusLabel")

        action_layout.addWidget(combo_label)
        action_layout.addWidget(self.slice_combo)
        action_layout.addWidget(self.btn_analyze)
        action_layout.addWidget(self.progress)
        action_layout.addWidget(self.status_label)
        left_layout.addWidget(action_card)

        log_card = QFrame()
        log_card.setObjectName("panel")
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(18, 18, 18, 18)
        log_layout.setSpacing(10)
        log_title = QLabel("运行日志")
        log_title.setObjectName("sectionLabel")
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(260)
        log_layout.addWidget(log_title)
        log_layout.addWidget(self.log_edit)
        left_layout.addWidget(log_card, 1)

        preview_card = QFrame()
        preview_card.setObjectName("panel")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(18, 18, 18, 18)
        preview_layout.setSpacing(10)
        self.preview_widget = HtmlPreviewWidget()
        self.preview_widget.show_placeholder("生成完成后，这里将直接内嵌显示 iodine_kedge_report.html。")
        preview_layout.addWidget(self.preview_widget, 1)

        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(preview_card)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([420, 720])
        root.addWidget(main_splitter, 1)

        self.setCentralWidget(central)

        self.btn_pick_dir.clicked.connect(self.pick_directory)
        self.btn_pick_file.clicked.connect(self.pick_file)
        self.btn_analyze.clicked.connect(self.start_analysis)
        self.btn_open_report.clicked.connect(self.open_report)
        self.btn_open_output.clicked.connect(self.open_output_dir)

    def _apply_theme(self):
        self.setFont(QFont("Microsoft YaHei", 10))
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #eef6ff;
                color: #23405d;
            }
            QLabel#titleLabel {
                font-size: 30px;
                font-weight: 700;
                color: #248bff;
                padding: 4px 0;
            }
            QLabel#subtitleLabel {
                font-size: 13px;
                color: #5d7d9c;
                padding-bottom: 6px;
            }
            QLabel#sectionLabel {
                font-size: 15px;
                font-weight: 600;
                color: #2d8cff;
            }
            QLabel#statusLabel {
                color: #2d8cff;
                font-size: 12px;
            }
            QLabel#previewHint {
                color: #a96e00;
                font-size: 12px;
            }
            QFrame#panel, DropFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ffffff, stop:1 #f2f8ff);
                border: 1px solid #bfdbff;
                border-radius: 18px;
            }
            DropFrame {
                border: 2px dashed #77b8ff;
            }
            QLabel#dropTitle {
                font-size: 20px;
                font-weight: 700;
                color: #2d8cff;
            }
            QLabel#dropHint {
                font-size: 12px;
                color: #6e8ba8;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #3d96ff, stop:1 #6bc3ff);
                color: white;
                border: 1px solid #8cc8ff;
                border-radius: 12px;
                padding: 10px 18px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #7ed0ff;
            }
            QPushButton:disabled {
                background: #dbe7f5;
                color: #93a6bb;
                border-color: #c8d7e8;
            }
            QComboBox, QPlainTextEdit {
                background: #ffffff;
                border: 1px solid #bfd9f5;
                border-radius: 12px;
                padding: 8px 10px;
                color: #23405d;
                selection-background-color: #a8d4ff;
            }
            QTextBrowser {
                background: #ffffff;
                border: 1px solid #bfd9f5;
                border-radius: 12px;
                padding: 0;
                color: #23405d;
            }
            QSplitter::handle {
                background: #d7e8fb;
                width: 8px;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                border: none;
                width: 34px;
            }
            QProgressBar {
                background: #ffffff;
                border: 1px solid #bfd9f5;
                border-radius: 8px;
                min-height: 10px;
            }
            QProgressBar::chunk {
                border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #58a7ff, stop:1 #92dcff);
            }
            """
        )

    def append_log(self, text: str):
        self.log_edit.appendPlainText(text)

    def pick_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择 DICOM 目录", str(self.case_dir or ROOT))
        if directory:
            self.load_paths([directory])

    def pick_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 DICOM 文件", str(self.case_dir or ROOT))
        if file_path:
            self.load_paths([file_path])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                paths.append(url.toLocalFile())
        if paths:
            self.load_paths(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def load_paths(self, paths: List[str]):
        selected_file: Optional[Path] = None
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                self.load_case_directory(path)
                return
            if path.is_file():
                selected_file = path
                self.load_case_directory(path.parent, preferred_file=path)
                return
        if selected_file is None:
            QMessageBox.warning(self, "路径无效", "未识别到有效目录或 DICOM 文件。")

    def load_case_directory(self, case_dir: Path, preferred_file: Optional[Path] = None):
        try:
            files = ordered_dicom_files(case_dir)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            return

        if not files:
            QMessageBox.warning(self, "未发现切片", "当前目录中未发现包含 PixelData 和 EFE1 的 DICOM 文件。")
            return

        self.case_dir = case_dir
        self.slice_files = files
        self.slice_combo.blockSignals(True)
        self.slice_combo.clear()
        preferred_index = 0
        for idx, file_path in enumerate(files):
            self.slice_combo.addItem(describe_slice(file_path, idx, len(files)), file_path)
            if preferred_file is not None and file_path == preferred_file:
                preferred_index = idx
        self.slice_combo.setCurrentIndex(preferred_index)
        self.slice_combo.blockSignals(False)

        self.slice_count_label.setText(f"有效 DICOM 数: {len(files)}")
        self.status_label.setText("已完成路径识别，请选择切片并点击分析。")
        self.btn_analyze.setEnabled(True)
        self.current_report_path = None
        self.current_output_dir = None
        self.btn_open_report.setEnabled(False)
        self.btn_open_output.setEnabled(False)
        self.preview_widget.show_placeholder("已识别切片，请选择序数后点击分析，右侧将显示内嵌 HTML 预览。")
        self.append_log("已完成目录识别")
        self.append_log(f"识别到 {len(files)} 张可分析切片")

    def current_slice_path(self) -> Optional[Path]:
        data = self.slice_combo.currentData()
        if data is None:
            return None
        return Path(data)

    def set_busy(self, busy: bool):
        self.btn_analyze.setEnabled((not busy) and bool(self.slice_files))
        self.btn_pick_dir.setEnabled(not busy)
        self.btn_pick_file.setEnabled(not busy)
        self.slice_combo.setEnabled(not busy)
        if busy:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)

    def start_analysis(self):
        slice_path = self.current_slice_path()
        if slice_path is None:
            QMessageBox.warning(self, "未选择切片", "请先选择一个切片。")
            return

        self.set_busy(True)
        self.status_label.setText("分析中，请稍候...")
        self.append_log(f"开始分析切片: {slice_path.name}")

        self.worker_thread = QThread(self)
        self.worker = AnalyzeWorker(slice_path, OUTPUT_ROOT)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.append_log)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def on_analysis_finished(self, result):
        self.set_busy(False)
        self.current_report_path = Path(result["html_path"])
        self.current_output_dir = Path(result["output_dir"])
        self.btn_open_report.setEnabled(True)
        self.btn_open_output.setEnabled(True)
        self.status_label.setText("生成完成，右侧已更新预览。")
        self.preview_widget.load_html(self.current_report_path)
        self.append_log("报告生成完成")
        self.append_log("可使用“打开报告”或“打开输出目录”继续查看结果")
        QMessageBox.information(self, "生成完成", "HTML 报告已生成，并已在右侧完成内嵌预览。")
        self.cleanup_worker()

    def on_analysis_failed(self, error_text: str):
        self.set_busy(False)
        self.status_label.setText("生成失败")
        self.preview_widget.show_placeholder("报告生成失败，请查看左侧日志或错误信息。")
        self.append_log(f"生成失败: {error_text}")
        QMessageBox.critical(self, "生成失败", error_text)
        self.cleanup_worker()

    def cleanup_worker(self):
        self.worker = None
        self.worker_thread = None

    def open_report(self):
        if self.current_report_path and self.current_report_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_report_path)))

    def open_output_dir(self):
        if self.current_output_dir and self.current_output_dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_output_dir)))


def main():
    os.environ.setdefault("KEDGE_APP_ROOT", str(ROOT))
    app = QApplication(sys.argv)
    if QWebEngineProfile is not None:
        QTWEBENGINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        QTWEBENGINE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        profile = QWebEngineProfile.defaultProfile()
        profile.setCachePath(str(QTWEBENGINE_CACHE_DIR))
        profile.setPersistentStoragePath(str(QTWEBENGINE_STORAGE_DIR))
    icon_path = resolve_app_icon()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))
    window = SliceReportMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
