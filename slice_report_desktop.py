import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pydicom
from PyQt5.QtCore import QObject, Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QLineEdit,
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

from build_iodine_kedge_report import (
    ROOT,
    CPU_COUNT,
    CUDA_BACKEND,
    CUDA_ENABLED,
    DECODE_THREADS,
    GPU_COMPUTE_SLOTS,
    GPU_PREFETCH_WORKERS,
    KEDGE_MODEL_VERSION,
    REPORT_CONTENT_VERSION,
    REPORT_INTERACTION_VERSION,
    SLICE_WORKERS,
    generate_single_slice_report,
    ordered_dicom_files,
)


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


def build_batch_strategy(slice_count: int) -> dict:
    cpu_parallel = max(1, CPU_COUNT // max(DECODE_THREADS, 1))
    batch_workers = min(slice_count, max(1, min(4, SLICE_WORKERS, cpu_parallel)))
    strategy_name = "CPU 多线程并行"
    strategy_desc = (
        f"CPU / NumPy + CPU 切片并行 {batch_workers} 路"
        f" / 每层解码线程 {DECODE_THREADS}"
    )
    return {
        "backend_text": CUDA_BACKEND,
        "strategy_name": strategy_name,
        "strategy_desc": strategy_desc,
        "batch_workers": batch_workers,
        "gpu_compute_slots": max(0, GPU_COMPUTE_SLOTS),
        "gpu_prefetch_workers": max(0, GPU_PREFETCH_WORKERS),
        "decode_threads": DECODE_THREADS,
        "cpu_count": CPU_COUNT,
        "cuda_enabled": CUDA_ENABLED,
        "oom_fallback_cpu": True,
    }


def format_stage_timings(stage_timings: Optional[dict]) -> str:
    if not stage_timings:
        return "无阶段耗时"
    ordered_keys = [
        ("read_dicom_s", "读取"),
        ("compute_products_s", "计算"),
        ("export_dicom_s", "导出DICOM"),
        ("build_figures_s", "绘图"),
        ("write_assets_s", "写PNG"),
        ("write_html_s", "写HTML"),
        ("write_native_json_s", "写JSON"),
        ("total_s", "总计"),
    ]
    parts = []
    for key, label in ordered_keys:
        if key in stage_timings:
            parts.append(f"{label} {float(stage_timings[key]):.2f}s")
    return " | ".join(parts)


def accumulate_stage_timings(results: List[dict]) -> dict:
    totals = {}
    for item in results:
        for key, value in (item.get("stage_timings") or {}).items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def report_output_dir_for_slice(file_path: Path, output_root: Path) -> Path:
    return Path(output_root) / file_path.parent.name / file_path.stem


def report_has_required_interactive_block(html_path: Path) -> bool:
    try:
        html_text = html_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    required_markers = [
        "对比视图联动阈值截断",
        'id="primary-material-select"',
        'id="compare-material-select"',
        'id="material-threshold-slider"',
    ]
    return all(marker in html_text for marker in required_markers)


def inspect_existing_report_result(
    file_path: Path, output_root: Path, fast_mode: bool = False
) -> Tuple[Optional[dict], Optional[str]]:
    output_dir = report_output_dir_for_slice(file_path, output_root)
    html_path = output_dir / "iodine_kedge_report.html"
    if not html_path.exists():
        return None, None
    if not report_has_required_interactive_block(html_path):
        return None, "旧缓存缺少交互区，已判定失效并重新生成。"
    native_json = output_dir / "iodine_kedge_report_native.json"
    meta_path = output_dir / "report_generation_meta.json"
    meta_payload = {}
    if meta_path.exists():
        try:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta_payload = {}
    existing_interaction_version = int(meta_payload.get("report_interaction_version", 0) or 0)
    if existing_interaction_version != int(REPORT_INTERACTION_VERSION):
        return None, "旧缓存交互版本过期，已判定失效并重新生成。"
    existing_content_version = int(meta_payload.get("report_content_version", 0) or 0)
    if existing_content_version != int(REPORT_CONTENT_VERSION):
        return None, "旧缓存报告内容版本过期，已判定失效并重新生成。"
    existing_model_version = int(meta_payload.get("kedge_model_version", 0) or 0)
    if existing_model_version != int(KEDGE_MODEL_VERSION):
        return None, "旧缓存K-edge模型版本过期，已判定失效并重新生成。"
    existing_fast_mode = bool(meta_payload.get("fast_mode", False))
    dcm_exports_ready = bool(meta_payload.get("dcm_exports_ready", False))
    existing_defer_static_figures = bool(meta_payload.get("defer_static_figures", False))
    if bool(meta_payload.get("cuda_enabled", False)):
        return None, "旧缓存来自 PyTorch/GPU 计算路径，已判定失效并重新生成。"
    if not fast_mode and existing_fast_mode and not native_json.exists():
        return None, "旧快速批处理缓存缺少完整结果文件，已判定失效并重新生成。"
    if existing_fast_mode and not dcm_exports_ready:
        return None, "旧快速批处理缓存未导出权重图 DICOM，已判定失效并重新生成。"
    if existing_defer_static_figures:
        return None, "旧批处理缓存使用了延后静态绘图，已判定失效并重新生成。"
    return (
        {
            "source_path": file_path,
            "output_dir": output_dir,
            "html_path": html_path,
            "native_json": native_json if native_json.exists() else None,
            "skipped_existing": True,
            "fast_mode": existing_fast_mode,
            "defer_static_figures": bool(meta_payload.get("defer_static_figures", False)),
            "report_interaction_version": existing_interaction_version,
            "report_content_version": existing_content_version,
            "kedge_model_version": existing_model_version,
            "stage_timings": meta_payload.get("stage_timings", {}),
            "generation_meta": meta_path if meta_path.exists() else None,
        },
        None,
    )


def load_existing_report_result(file_path: Path, output_root: Path, fast_mode: bool = False) -> Optional[dict]:
    result, _ = inspect_existing_report_result(file_path, output_root, fast_mode=fast_mode)
    return result


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

    def __init__(self, file_paths: List[Path], output_root: Path, mode: str, strategy: Optional[dict] = None, fast_mode: bool = False):
        super().__init__()
        self.file_paths = list(file_paths)
        self.output_root = output_root
        self.mode = mode
        self.strategy = dict(strategy or build_batch_strategy(len(file_paths)))
        self.fast_mode = bool(fast_mode and mode == "batch")
        self.defer_static_figures = False

    def run_single_report(self, file_path: Path) -> dict:
        return generate_single_slice_report(
            file_path,
            output_root=self.output_root,
            fast_mode=self.fast_mode,
            defer_static_figures=self.defer_static_figures,
        )

    def build_cpu_fallback_command(self, file_path: Path) -> List[str]:
        common_args = [
            "--generate-single-report",
            str(file_path),
            "--output-root",
            str(self.output_root),
        ]
        if self.fast_mode:
            common_args.append("--fast-mode")
        if self.defer_static_figures:
            common_args.append("--defer-static-figures")
        if getattr(sys, "frozen", False):
            return [sys.executable, *common_args]
        return [sys.executable, str(Path(__file__)), *common_args]

    @staticmethod
    def is_gpu_oom_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "out of memory" in text or "cuda error: out of memory" in text

    def run_single_report_with_fallback(self, file_path: Path) -> dict:
        try:
            return self.run_single_report(file_path)
        except Exception as exc:
            if not (self.mode == "batch" and self.strategy.get("cuda_enabled") and self.is_gpu_oom_error(exc)):
                raise
            self.progress.emit(f"批处理回退 {file_path.name}: 检测到 GPU OOM，自动切换 CPU 重算该层。")
            env = os.environ.copy()
            env["KEDGE_DISABLE_CUDA"] = "1"
            env["KEDGE_DISABLE_TORCH"] = "1"
            env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            completed = subprocess.run(
                self.build_cpu_fallback_command(file_path),
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if completed.returncode != 0:
                stderr_tail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(f"GPU OOM 后 CPU 回退失败: {stderr_tail[-1200:]}")
            existing, invalid_reason = inspect_existing_report_result(
                file_path, self.output_root, fast_mode=self.fast_mode
            )
            if existing is None:
                raise RuntimeError(invalid_reason or "GPU OOM 后 CPU 回退完成，但未找到生成结果。")
            existing["oom_cpu_fallback"] = True
            self.progress.emit(f"批处理回退完成 {file_path.name}: 已使用 CPU 重算并写入缓存。")
            return existing

    def run(self):
        try:
            if not self.file_paths:
                raise RuntimeError("没有可处理的切片。")
            total = len(self.file_paths)
            self.progress.emit(
                f"计算模式: {self.strategy['strategy_name']} | {self.strategy['strategy_desc']}"
            )
            if self.mode == "batch":
                results_by_idx = {}
                pending_jobs = []
                skipped_count = 0
                for idx, file_path in enumerate(self.file_paths):
                    existing, invalid_reason = inspect_existing_report_result(
                        file_path, self.output_root, fast_mode=self.fast_mode
                    )
                    if existing is not None:
                        results_by_idx[idx] = existing
                        skipped_count += 1
                        self.progress.emit(f"批处理跳过 {idx + 1}/{total}: {file_path.name} | 已存在报告")
                    else:
                        if invalid_reason:
                            self.progress.emit(f"批处理重建 {idx + 1}/{total}: {file_path.name} | {invalid_reason}")
                        pending_jobs.append((idx, file_path))
                generated_count = 0
            else:
                results_by_idx = {}
                pending_jobs = []
                for idx, file_path in enumerate(self.file_paths):
                    _, invalid_reason = inspect_existing_report_result(
                        file_path, self.output_root, fast_mode=self.fast_mode
                    )
                    if invalid_reason:
                        self.progress.emit(f"开始重算 {file_path.name} | {invalid_reason}")
                    pending_jobs.append((idx, file_path))
                skipped_count = 0
                generated_count = 0

            if self.mode == "batch" and len(pending_jobs) > 1 and self.strategy["batch_workers"] > 1:
                with ThreadPoolExecutor(
                    max_workers=self.strategy["batch_workers"],
                    thread_name_prefix="desktop-batch",
                ) as executor:
                    future_map = {
                        executor.submit(self.run_single_report_with_fallback, file_path): (idx, file_path)
                        for idx, file_path in pending_jobs
                    }
                    for future in as_completed(future_map):
                        idx, file_path = future_map[future]
                        result = future.result()
                        results_by_idx[idx] = result
                        generated_count += 1
                        self.progress.emit(
                            f"批处理生成 {generated_count}/{len(pending_jobs)}: {file_path.name} 完成 | {format_stage_timings(result.get('stage_timings'))}"
                        )
                results = [results_by_idx[idx] for idx in range(total)]
            else:
                results = []
                for idx, file_path in pending_jobs:
                    if self.mode == "batch":
                        self.progress.emit(f"批处理生成 {generated_count + 1}/{max(len(pending_jobs), 1)}: {file_path.name}")
                    else:
                        self.progress.emit(f"开始分析切片 {file_path.name}")
                    result = self.run_single_report_with_fallback(file_path)
                    results_by_idx[idx] = result
                    generated_count += 1
                    if self.mode == "batch":
                        self.progress.emit(
                            f"批处理生成 {generated_count}/{max(len(pending_jobs), 1)}: {file_path.name} 完成 | {format_stage_timings(result.get('stage_timings'))}"
                        )
                if self.mode == "batch":
                    results = [results_by_idx[idx] for idx in range(total)]
                else:
                    results = [results_by_idx[idx] for idx in range(total)]
            final_result = results[-1]
            final_result["mode"] = self.mode
            final_result["processed_count"] = total
            final_result["results"] = results
            final_result["compute_strategy"] = self.strategy
            final_result["skipped_count"] = skipped_count
            final_result["generated_count"] = generated_count
            final_result["fast_mode"] = self.fast_mode
            final_result["batch_stage_totals"] = accumulate_stage_timings(
                [item for item in results if not item.get("skipped_existing")]
            )
            self.progress.emit("报告已生成，可在右侧直接预览")
            self.finished.emit(final_result)
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
        self.pending_mode = "single"
        self.report_results: Dict[Path, dict] = {}
        self.current_strategy = build_batch_strategy(0)

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

        subtitle = QLabel("拖拽目录或 DCM 文件，选择切片序数后点击分析，并在右侧直接预览包含多材料权重图的 HTML 报告")
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
        self.btn_clear_cache = QPushButton("清除报告缓存")
        self.btn_open_report.setEnabled(False)
        self.btn_open_output.setEnabled(False)
        self.btn_clear_cache.setEnabled(False)
        btn_row.addWidget(self.btn_pick_dir)
        btn_row.addWidget(self.btn_pick_file)
        btn_row.addWidget(self.btn_clear_cache)
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
        self.compute_mode_label = QLabel("计算后端: 等待加载病例")
        self.compute_mode_label.setWordWrap(True)
        self.batch_strategy_label = QLabel("批处理策略: 等待加载病例")
        self.batch_strategy_label.setWordWrap(True)
        info_layout.addWidget(self.slice_count_label)
        info_layout.addWidget(self.privacy_label)
        info_layout.addWidget(self.compute_mode_label)
        info_layout.addWidget(self.batch_strategy_label)
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
        self.fast_batch_checkbox = QCheckBox("快速批处理模式（少写 DICOM / PNG / JSON）")
        self.fast_batch_checkbox.setChecked(True)
        nav_row = QHBoxLayout()
        nav_row.setSpacing(10)
        self.btn_prev = QPushButton("上一页")
        self.btn_next = QPushButton("下一页")
        self.btn_batch = QPushButton("批处理所有切片")
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.btn_batch.setEnabled(False)
        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.btn_next)
        nav_row.addWidget(self.btn_batch)
        jump_row = QHBoxLayout()
        jump_row.setSpacing(10)
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("输入页码")
        self.jump_input.setMinimumHeight(40)
        self.btn_jump = QPushButton("跳到指定页")
        self.btn_jump.setEnabled(False)
        jump_row.addWidget(self.jump_input, 1)
        jump_row.addWidget(self.btn_jump)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.status_label = QLabel("等待输入路径")
        self.status_label.setObjectName("statusLabel")

        action_layout.addWidget(combo_label)
        action_layout.addWidget(self.slice_combo)
        action_layout.addWidget(self.btn_analyze)
        action_layout.addWidget(self.fast_batch_checkbox)
        action_layout.addLayout(nav_row)
        action_layout.addLayout(jump_row)
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
        self.btn_prev.clicked.connect(self.goto_previous_slice)
        self.btn_next.clicked.connect(self.goto_next_slice)
        self.btn_batch.clicked.connect(self.start_batch_analysis)
        self.btn_jump.clicked.connect(self.jump_to_slice)
        self.btn_open_report.clicked.connect(self.open_report)
        self.btn_open_output.clicked.connect(self.open_output_dir)
        self.btn_clear_cache.clicked.connect(self.clear_report_cache)
        self.slice_combo.currentIndexChanged.connect(self.on_slice_selection_changed)
        self.jump_input.returnPressed.connect(self.jump_to_slice)

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

    def current_case_output_root(self) -> Optional[Path]:
        if self.case_dir is None:
            return None
        return OUTPUT_ROOT / self.case_dir.name

    def refresh_compute_mode_labels(self):
        count = len(self.slice_files)
        self.current_strategy = build_batch_strategy(count)
        self.compute_mode_label.setText(f"计算后端: {self.current_strategy['backend_text']}")
        if count:
            self.batch_strategy_label.setText(
                f"批处理策略: {self.current_strategy['strategy_name']} | {self.current_strategy['strategy_desc']}"
            )
        else:
            self.batch_strategy_label.setText("批处理策略: 等待加载病例")

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
            QMessageBox.warning(self, "未发现切片", "当前路径及子目录中未发现包含 PixelData 和 EFE1 的 DICOM 文件。")
            return

        self.case_dir = case_dir
        self.slice_files = files
        self.report_results = {}
        self.refresh_compute_mode_labels()
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
        self.btn_clear_cache.setEnabled(True)
        self.current_report_path = None
        self.current_output_dir = None
        self.btn_open_report.setEnabled(False)
        self.btn_open_output.setEnabled(False)
        self.jump_input.clear()
        self.preview_widget.show_placeholder("已识别切片，请选择序数后点击分析，右侧将显示包含多材料权重图的 HTML 预览。")
        self.append_log("已完成路径识别")
        self.append_log(f"识别到 {len(files)} 张可分析切片")
        self.append_log(f"计算后端: {self.current_strategy['backend_text']}")
        self.append_log(f"批处理策略: {self.current_strategy['strategy_desc']}")
        self.update_navigation_buttons()

    def start_worker(self, file_paths: List[Path], mode: str, status_text: str, log_text: str):
        if self.case_dir is None:
            QMessageBox.warning(self, "未加载病例", "请先拖入病例目录或 DICOM 文件。")
            self.set_busy(False)
            return
        self.pending_mode = mode
        self.set_busy(True)
        self.status_label.setText(status_text)
        self.append_log(log_text)
        strategy = self.current_strategy if mode == "batch" else build_batch_strategy(len(file_paths))
        self.worker_thread = QThread(self)
        self.worker = AnalyzeWorker(
            file_paths,
            OUTPUT_ROOT,
            mode,
            strategy=strategy,
            fast_mode=self.fast_batch_checkbox.isChecked(),
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.append_log)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def current_slice_path(self) -> Optional[Path]:
        data = self.slice_combo.currentData()
        if data is None:
            return None
        return Path(data)

    def update_navigation_buttons(self):
        has_slices = bool(self.slice_files)
        index = self.slice_combo.currentIndex()
        count = self.slice_combo.count()
        is_idle = self.worker_thread is None
        self.btn_prev.setEnabled(has_slices and is_idle and index > 0)
        self.btn_next.setEnabled(has_slices and is_idle and 0 <= index < count - 1)
        self.btn_batch.setEnabled(has_slices and is_idle)
        self.btn_jump.setEnabled(has_slices and is_idle)
        self.jump_input.setEnabled(has_slices and is_idle)
        self.btn_clear_cache.setEnabled(has_slices and is_idle)

    def clear_report_cache(self):
        case_output_root = self.current_case_output_root()
        if case_output_root is None or not self.slice_files:
            QMessageBox.warning(self, "未加载病例", "请先加载病例后再清除报告缓存。")
            return
        if not case_output_root.exists():
            self.report_results = {}
            self.current_report_path = None
            self.current_output_dir = None
            self.btn_open_report.setEnabled(False)
            self.btn_open_output.setEnabled(False)
            self.preview_widget.show_placeholder("当前病例还没有已生成的 HTML 报告缓存。")
            self.status_label.setText("当前病例无报告缓存，可直接重新计算。")
            self.append_log(f"未发现报告缓存目录: {case_output_root}")
            return

        reply = QMessageBox.question(
            self,
            "确认清除缓存",
            f"将删除当前病例已生成的全部 HTML 报告相关数据：\n{case_output_root}\n\n是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.preview_widget.show_placeholder("报告缓存已清除，请重新点击分析或批处理生成。")
        self.current_report_path = None
        self.current_output_dir = None
        self.btn_open_report.setEnabled(False)
        self.btn_open_output.setEnabled(False)
        try:
            shutil.rmtree(case_output_root)
        except Exception as exc:
            QMessageBox.critical(self, "清除失败", str(exc))
            self.append_log(f"清除报告缓存失败: {exc}")
            return

        self.report_results = {}
        self.status_label.setText("已清除当前病例报告缓存，可重新计算。")
        self.append_log(f"已清除报告缓存目录: {case_output_root}")
        QMessageBox.information(self, "清除完成", "当前病例已生成的 HTML 报告相关数据已全部删除。")

    def apply_result_to_preview(self, result: dict, status_text: Optional[str] = None):
        self.current_report_path = Path(result["html_path"])
        self.current_output_dir = Path(result["output_dir"]).parent
        self.btn_open_report.setEnabled(True)
        self.btn_open_output.setEnabled(True)
        if status_text is not None:
            self.status_label.setText(status_text)
        self.preview_widget.load_html(self.current_report_path)

    def cache_result(self, result: dict):
        source_path = Path(result["source_path"])
        self.report_results[source_path] = result

    def show_cached_report_for_index(self, index: int) -> bool:
        if index < 0 or index >= len(self.slice_files):
            return False
        slice_path = self.slice_files[index]
        cached = self.report_results.get(slice_path)
        if cached is None:
            self.current_report_path = None
            self.current_output_dir = None
            self.btn_open_report.setEnabled(False)
            self.btn_open_output.setEnabled(False)
            self.preview_widget.show_placeholder("该层报告尚未生成。请先点击“分析并生成 HTML 报告”或先执行“批处理所有切片”。")
            self.status_label.setText(f"已切换到第 {index + 1} 层，该层尚未生成报告。")
            self.append_log(f"切换到第 {index + 1} 层: {slice_path.name} | 尚无缓存报告")
            return False
        self.apply_result_to_preview(cached, status_text=f"已切换到第 {index + 1} 层，直接加载已生成报告。")
        self.append_log(f"切换到第 {index + 1} 层: {slice_path.name} | 已直接加载缓存报告")
        return True

    def navigate_to_index(self, index: int):
        if index < 0 or index >= len(self.slice_files):
            return
        self.slice_combo.setCurrentIndex(index)
        self.show_cached_report_for_index(index)

    def on_slice_selection_changed(self):
        self.update_navigation_buttons()

    def set_busy(self, busy: bool):
        self.btn_analyze.setEnabled((not busy) and bool(self.slice_files))
        self.btn_pick_dir.setEnabled(not busy)
        self.btn_pick_file.setEnabled(not busy)
        self.slice_combo.setEnabled(not busy)
        self.fast_batch_checkbox.setEnabled(not busy and bool(self.slice_files))
        self.update_navigation_buttons()
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
        self.start_worker([slice_path], "single", "分析中，请稍候...", f"开始分析切片: {slice_path.name}")

    def start_batch_analysis(self):
        if not self.slice_files:
            QMessageBox.warning(self, "未加载病例", "请先拖入病例目录或 DICOM 文件。")
            return
        self.start_worker(
            self.slice_files,
            "batch",
            f"批处理进行中，共 {len(self.slice_files)} 层 | {self.current_strategy['strategy_name']}",
            f"开始批处理全部切片，共 {len(self.slice_files)} 层 | {self.current_strategy['strategy_desc']} | 快速模式 {'开' if self.fast_batch_checkbox.isChecked() else '关'}",
        )

    def goto_previous_slice(self):
        current = self.slice_combo.currentIndex()
        if current <= 0:
            return
        self.navigate_to_index(current - 1)

    def goto_next_slice(self):
        current = self.slice_combo.currentIndex()
        if current >= self.slice_combo.count() - 1:
            return
        self.navigate_to_index(current + 1)

    def jump_to_slice(self):
        if not self.slice_files:
            QMessageBox.warning(self, "未加载病例", "请先拖入病例目录或 DICOM 文件。")
            return
        text = self.jump_input.text().strip()
        if not text:
            QMessageBox.warning(self, "页码为空", "请输入要跳转的页码。")
            return
        try:
            page = int(text)
        except ValueError:
            QMessageBox.warning(self, "页码无效", "请输入有效的整数页码。")
            return
        if page < 1 or page > len(self.slice_files):
            QMessageBox.warning(self, "页码超范围", f"请输入 1 到 {len(self.slice_files)} 之间的页码。")
            return
        self.navigate_to_index(page - 1)

    def on_analysis_finished(self, result):
        self.set_busy(False)
        source_path = Path(result["source_path"])
        result_list = result.get("results", [result])
        for item in result_list:
            self.cache_result(item)
        for idx, file_path in enumerate(self.slice_files):
            if file_path == source_path:
                self.slice_combo.setCurrentIndex(idx)
                break
        mode = result.get("mode", "single")
        processed_count = int(result.get("processed_count", 1))
        skipped_count = int(result.get("skipped_count", 0))
        generated_count = int(result.get("generated_count", processed_count))
        compute_strategy = result.get("compute_strategy", self.current_strategy)
        fast_mode = bool(result.get("fast_mode", False))
        defer_static_figures = bool(result.get("defer_static_figures", False))
        batch_stage_totals = result.get("batch_stage_totals", {})
        if mode == "batch":
            self.apply_result_to_preview(
                result,
                status_text=f"批处理完成，新生成 {generated_count} 层，跳过 {skipped_count} 层，快速模式 {'开' if fast_mode else '关'}，当前已直接完成静态绘制，右侧显示最后一层。",
            )
        else:
            self.apply_result_to_preview(result, status_text="生成完成，右侧已更新预览。")
        if mode == "batch":
            self.append_log(
                f"批处理完成，共 {processed_count} 层 | 新生成 {generated_count} 层 | 跳过 {skipped_count} 层 | 快速模式 {'开' if fast_mode else '关'} | 静态绘图已即时生成"
            )
            self.append_log(
                f"本次批处理模式: {compute_strategy.get('strategy_name', '未知')} | {compute_strategy.get('strategy_desc', '')}"
            )
            if batch_stage_totals:
                self.append_log(f"批处理阶段耗时累计: {format_stage_timings(batch_stage_totals)}")
            self.append_log("右侧当前显示最后一层报告，可继续使用前后翻页或跳页直接加载已生成报告。")
            QMessageBox.information(
                self,
                "批处理完成",
                f"共 {processed_count} 层，其中新生成 {generated_count} 层，跳过已存在输出 {skipped_count} 层。快速模式：{'开' if fast_mode else '关'}；静态绘图：已即时生成。",
            )
        else:
            self.append_log("报告生成完成")
            self.append_log(f"本层阶段耗时: {format_stage_timings(result.get('stage_timings'))}")
            self.append_log("可使用“打开报告”或“打开输出目录”继续查看结果")
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
        self.update_navigation_buttons()

    def open_report(self):
        if self.current_report_path and self.current_report_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_report_path)))

    def open_output_dir(self):
        if self.current_output_dir and self.current_output_dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_output_dir)))


def main():
    os.environ.setdefault("KEDGE_APP_ROOT", str(ROOT))
    if "--generate-single-report" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--generate-single-report", dest="file_path", required=True)
        parser.add_argument("--output-root", dest="output_root", required=True)
        parser.add_argument("--fast-mode", action="store_true")
        parser.add_argument("--defer-static-figures", action="store_true")
        args = parser.parse_args()
        result = generate_single_slice_report(
            Path(args.file_path),
            output_root=Path(args.output_root),
            fast_mode=bool(args.fast_mode),
            defer_static_figures=bool(args.defer_static_figures),
        )
        print(
            json.dumps(
                {
                    "html_path": str(result["html_path"]),
                    "output_dir": str(result["output_dir"]),
                    "fast_mode": bool(result.get("fast_mode", False)),
                    "defer_static_figures": bool(result.get("defer_static_figures", False)),
                },
                ensure_ascii=False,
            )
        )
        return
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
