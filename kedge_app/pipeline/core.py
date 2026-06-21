import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pydicom


APP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = APP_ROOT / "00020006"
DEFAULT_OUTPUT_DIR = APP_ROOT / "ui_batch_outputs"


@dataclass
class CaseRecord:
    case_dir: Path
    case_id: str
    status: str = "待处理"
    stage: str = ""
    output_dir: Optional[Path] = None
    result: Optional[Dict[str, Path]] = None


def case_id_from_path(path: Path) -> str:
    if path.is_dir():
        return path.name
    return path.parent.name or path.stem


def sanitize_case_id(case_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in case_id)
    return cleaned or "case"


def is_dicom_file(file_path: Path) -> bool:
    if not file_path.is_file():
        return False
    try:
        pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def discover_case_directory(path_str: str) -> Optional[Path]:
    path = Path(path_str)
    if not path.exists():
        return None
    if path.is_dir():
        return path
    if path.is_file() and is_dicom_file(path):
        return path.parent
    return None


def discover_dicom_files(case_dir: Path) -> List[Path]:
    files = [p for p in sorted(case_dir.iterdir()) if p.is_file()]
    return [p for p in files if is_dicom_file(p)]


def _read_hu_slice(file_path: Path) -> Optional[np.ndarray]:
    ds = pydicom.dcmread(str(file_path), force=True)
    if not hasattr(ds, "PixelData"):
        return None
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim != 2:
        return None
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def load_volume(
    case_dir: Path,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> Tuple[np.ndarray, List[Path]]:
    if progress_callback is not None:
        progress_callback("扫描 DICOM", f"扫描目录 {case_dir}")
    dicom_files = discover_dicom_files(case_dir)
    if not dicom_files:
        raise RuntimeError(f"目录中未发现可读取的 DICOM 文件: {case_dir}")
    if progress_callback is not None:
        progress_callback("加载体数据", f"读取 {len(dicom_files)} 个 DICOM 文件")

    slices = []
    for file_path in dicom_files:
        ds = pydicom.dcmread(str(file_path), force=True)
        if not hasattr(ds, "PixelData"):
            continue
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim != 2:
            continue

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
            order_key = float(ds.ImagePositionPatient[2])
        elif hasattr(ds, "SliceLocation"):
            order_key = float(ds.SliceLocation)
        elif hasattr(ds, "InstanceNumber"):
            order_key = float(ds.InstanceNumber)
        else:
            order_key = float(len(slices))
        slices.append((order_key, arr, file_path))

    if not slices:
        raise RuntimeError(f"目录中存在 DICOM，但没有二维像素切片: {case_dir}")

    slices.sort(key=lambda item: item[0])
    shapes = {}
    for _, arr, _ in slices:
        shapes[arr.shape] = shapes.get(arr.shape, 0) + 1
    target_shape = max(shapes.items(), key=lambda kv: kv[1])[0]
    filtered = [(k, a, p) for (k, a, p) in slices if a.shape == target_shape]

    volume = np.stack([arr for _, arr, _ in filtered], axis=0)
    paths = [file_path for _, _, file_path in filtered]
    return volume, paths


def build_case_thumbnail(case_dir: Path) -> np.ndarray:
    dicom_files = discover_dicom_files(case_dir)
    if not dicom_files:
        return np.zeros((256, 256), dtype=np.float32)
    sample_path = dicom_files[len(dicom_files) // 2]
    arr = _read_hu_slice(sample_path)
    if arr is None:
        return np.zeros((256, 256), dtype=np.float32)
    return arr


def _run_script(script_name: str, env: dict) -> None:
    _run_script_stream(script_name, env, None)


def _run_script_stream(
    script_name: str,
    env: dict,
    output_callback: Optional[Callable[[str], None]] = None,
) -> None:
    cmd = [sys.executable, script_name]
    proc = subprocess.Popen(
        cmd,
        cwd=str(APP_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    captured_lines: List[str] = []
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\r\n")
        captured_lines.append(line)
        if output_callback is not None:
            output_callback(line)
    proc.wait()
    if proc.returncode != 0:
        output_text = "\n".join(captured_lines)
        raise RuntimeError(
            f"执行失败: {script_name}\n"
            f"exit_code={proc.returncode}\n"
            f"stdout/stderr:\n{output_text}"
        )


def process_case(
    case_dir: Path,
    output_root: Path,
    stage_callback: Optional[Callable[[str, str], None]] = None,
    slice_progress_callback: Optional[Callable[[int, int, str], None]] = None,
    backend_preference: str = "auto",
) -> Dict[str, Path]:
    case_id = sanitize_case_id(case_id_from_path(case_dir))
    output_dir = output_root / case_id
    cache_dir = output_dir / "_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if stage_callback is not None:
        stage_callback("扫描 DICOM", f"{case_id}: 识别病例目录并准备输出路径")

    dicom_files = discover_dicom_files(case_dir)
    if not dicom_files:
        raise RuntimeError(f"目录中未发现可读取的 DICOM 文件: {case_dir}")
    if stage_callback is not None:
        stage_callback("加载体数据", f"{case_id}: 已发现 {len(dicom_files)} 个 DICOM 文件，准备读取基础图像数据")

    env = os.environ.copy()
    env["KEDGE_APP_ROOT"] = str(APP_ROOT)
    env["KEDGE_DATA_DIR"] = str(case_dir)
    env["KEDGE_OUT_DIR"] = str(output_dir)
    env["KEDGE_CACHE_DIR"] = str(cache_dir)
    if backend_preference:
        env["KEDGE_BACKEND"] = backend_preference
        if backend_preference == "cuda-full":
            env["KEDGE_FORCE_NVJPEG2K"] = "1"

    if stage_callback is not None:
        stage_callback("解析 EFE1", f"{case_id}: 解析频谱与重建主报告")
    def _handle_script_line(line: str) -> None:
        if not line.startswith("SLICE_PROGRESS\t"):
            return
        parts = line.split("\t", 3)
        if len(parts) < 4:
            return
        try:
            current = int(parts[1])
            total = int(parts[2])
        except ValueError:
            return
        detail = parts[3]
        if slice_progress_callback is not None:
            slice_progress_callback(current, total, f"{case_id}: 已处理切片 {current}/{total} - {detail}")

    _run_script_stream("build_iodine_kedge_report.py", env, _handle_script_line)

    if stage_callback is not None:
        stage_callback("生成报告", f"{case_id}: 生成 1:1 对比报告")
    _run_script("build_weight_map_comparison_report.py", env)

    if stage_callback is not None:
        stage_callback("生成报告", f"{case_id}: 生成候选交互报告")
    _run_script("build_weight_map_candidate_suite.py", env)

    if stage_callback is not None:
        stage_callback("写出 DICOM/HTML", f"{case_id}: 校验派生 DICOM 与 HTML 输出")

    return {
        "case_id": case_id,
        "case_dir": case_dir,
        "output_dir": output_dir,
        "html_iodine": output_dir / "iodine_kedge_report.html",
        "html_compare": output_dir / "weight_map_1to1_compare.html",
        "html_suite": output_dir / "weight_map_interactive_suite.html",
        "native_iodine": output_dir / "iodine_kedge_report_native.json",
        "native_compare": output_dir / "weight_map_1to1_compare_native.json",
        "native_suite": output_dir / "weight_map_interactive_suite_native.json",
        "volume_npz": output_dir / "reconstructed_volumes.npz",
        "volume_manifest": output_dir / "volume_manifest.json",
        "iodine_series_dir": output_dir / "iodine_series",
        "kedge_series_dir": output_dir / "kedge_series",
        "dcm_iodine": output_dir / "iodine_weight_map.dcm",
        "dcm_kedge": output_dir / "kedge_weight_map.dcm",
    }
