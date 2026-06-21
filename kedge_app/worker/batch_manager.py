import os
import subprocess
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from PyQt5.QtCore import QObject, pyqtSignal

from kedge_app.pipeline.core import case_id_from_path, discover_dicom_files, process_case, sanitize_case_id


def get_optimal_cuda_full_workers() -> int:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        mem_mb = max(int(line.strip()) for line in output.strip().split("\n") if line.strip().isdigit())
        usable_mb = mem_mb - 1500
        workers = max(2, usable_mb // 2560)
        return min(workers, os.cpu_count() or 4)
    except Exception:
        return min(6, max(2, (os.cpu_count() or 4) // 3))


class BatchManager(QObject):
    case_started = pyqtSignal(str)
    case_done = pyqtSignal(str, dict)
    case_failed = pyqtSignal(str, str)
    case_stage = pyqtSignal(str, str, str)
    progress = pyqtSignal(int, int, str)
    log_message = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.executor: Optional[ThreadPoolExecutor] = None
        self.futures: Dict[Future, str] = {}
        self.lock = Lock()
        self.total = 0
        self.completed = 0
        self.running = False
        self.case_slice_totals: Dict[str, int] = {}
        self.case_slice_processed: Dict[str, int] = {}
        self.total_slices = 0

    def is_running(self) -> bool:
        return self.running

    def start(self, case_dirs: List[Path], output_root: Path, max_workers: Optional[int] = None, backend_preference: str = "auto") -> None:
        if self.running:
            return
        if not case_dirs:
            self.finished.emit()
            return

        self.running = True
        self.total = len(case_dirs)
        self.completed = 0
        self.futures = {}
        self.case_slice_totals = {}
        self.case_slice_processed = {}
        self.total_slices = 0

        for case_dir in case_dirs:
            case_id = sanitize_case_id(case_id_from_path(case_dir))
            slice_total = len(discover_dicom_files(case_dir))
            self.case_slice_totals[case_id] = slice_total
            self.case_slice_processed[case_id] = 0
            self.total_slices += slice_total

        if max_workers is None:
            if backend_preference == "cuda-full":
                max_workers = get_optimal_cuda_full_workers()
            else:
                max_workers = min(max(2, (os.cpu_count() or 4) // 2), self.total, 4)

        self.log_message.emit(
            f"批处理线程池启动: max_workers={max_workers}, 总切片数={self.total_slices}"
        )
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="kedge-batch")

        for case_dir in case_dirs:
            case_id = sanitize_case_id(case_id_from_path(case_dir))
            self.case_started.emit(case_id)
            future = self.executor.submit(
                process_case,
                case_dir,
                output_root,
                lambda stage, detail, cid=case_id: self.case_stage.emit(cid, stage, detail),
                lambda current, total, detail, cid=case_id: self._on_case_slice_progress(cid, current, total, detail),
                backend_preference,
            )
            self.futures[future] = case_id
            future.add_done_callback(self._on_future_done)

    def _on_case_slice_progress(self, case_id: str, current: int, total: int, detail: str) -> None:
        with self.lock:
            self.case_slice_totals[case_id] = max(total, self.case_slice_totals.get(case_id, 0))
            self.case_slice_processed[case_id] = max(current, self.case_slice_processed.get(case_id, 0))
            processed = sum(self.case_slice_processed.values())
            total_slices = max(sum(self.case_slice_totals.values()), 1)
        self.progress.emit(processed, total_slices, detail)

    def _on_future_done(self, future: Future) -> None:
        case_id = self.futures.get(future, "unknown")
        try:
            result = future.result()
            self.case_done.emit(case_id, result)
        except Exception:
            self.case_failed.emit(case_id, traceback.format_exc())

        with self.lock:
            self.completed += 1
            done = self.completed
            total = self.total
            if case_id in self.case_slice_totals:
                self.case_slice_processed[case_id] = self.case_slice_totals[case_id]
            processed = sum(self.case_slice_processed.values())
            total_slices = max(sum(self.case_slice_totals.values()), 1)

        self.progress.emit(processed, total_slices, f"{case_id} 完成，病例进度 {done}/{total}")

        if done >= total:
            self._shutdown()

    def _shutdown(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=False)
            self.executor = None
        self.running = False
        self.finished.emit()
