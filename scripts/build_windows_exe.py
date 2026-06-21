import os
import sys
import zipfile
from pathlib import Path

import PyInstaller.__main__
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRY_SCRIPT = PROJECT_ROOT / "slice_report_desktop.py"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
RELEASE_DIR = PROJECT_ROOT / "release"
APP_NAME = "KEdgeSliceReport"
LOGO_DIR = PROJECT_ROOT / "reconstructed_weight_maps" / "logo"


def add_data_arg(path: Path, dest: str) -> str:
    return f"{path}{os.pathsep}{dest}"


def resolve_logo_source() -> Path:
    for pattern in ("*.ico", "*.png", "*.jpg", "*.jpeg", "*.bmp"):
        matches = sorted(LOGO_DIR.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"未在 {LOGO_DIR} 找到可用 logo 文件。")


def ensure_windows_icon() -> Path:
    logo_path = resolve_logo_source()
    if logo_path.suffix.lower() == ".ico":
        return logo_path
    BUILD_DIR.mkdir(exist_ok=True)
    icon_path = BUILD_DIR / "app_icon.ico"
    with Image.open(logo_path) as img:
        rgba = img.convert("RGBA")
        sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
        rgba.save(icon_path, format="ICO", sizes=sizes)
    return icon_path


def find_openjp2_dll() -> Path:
    candidates = []
    env_candidates = [
        os.environ.get("OPENJP2_DLL"),
        os.environ.get("OPENJPEG_DLL"),
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        env_candidates.extend(
            [
                str(Path(conda_prefix) / "Library" / "bin" / "openjp2.dll"),
                str(Path(conda_prefix) / "bin" / "openjp2.dll"),
            ]
        )
    env_candidates.extend(
        [
            r"D:\python\Library\bin\openjp2.dll",
            r"C:\Miniconda\Library\bin\openjp2.dll",
            r"C:\ProgramData\miniconda3\Library\bin\openjp2.dll",
        ]
    )
    for raw in env_candidates:
        if raw:
            candidates.append(Path(raw))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "未找到 openjp2.dll。请先安装 OpenJPEG，并通过 OPENJP2_DLL 或 CONDA_PREFIX 提供路径。"
    )


def build():
    openjp2_dll = find_openjp2_dll()
    icon_path = ensure_windows_icon()
    DIST_DIR.mkdir(exist_ok=True)
    BUILD_DIR.mkdir(exist_ok=True)
    RELEASE_DIR.mkdir(exist_ok=True)
    exe_path = DIST_DIR / f"{APP_NAME}.exe"

    args = [
        str(ENTRY_SCRIPT),
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        APP_NAME,
        "--icon",
        str(icon_path),
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR),
        "--specpath",
        str(BUILD_DIR),
        "--hidden-import",
        "PyQt5.QtWebEngineWidgets",
        "--hidden-import",
        "PyQt5.QtWebEngineCore",
        "--hidden-import",
        "PyQt5.QtWebChannel",
        "--collect-submodules",
        "glymur",
        "--collect-submodules",
        "matplotlib",
        "--collect-data",
        "matplotlib",
        "--add-data",
        add_data_arg(LOGO_DIR, "reconstructed_weight_maps/logo"),
        "--add-data",
        add_data_arg(PROJECT_ROOT / "pure_c_kedge", "pure_c_kedge"),
        "--add-binary",
        add_data_arg(openjp2_dll, "pure_c_kedge"),
    ]
    PyInstaller.__main__.run(args)

    archive_base = RELEASE_DIR / f"{APP_NAME}-windows"
    with zipfile.ZipFile(f"{archive_base}.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(exe_path, arcname=exe_path.name)
    print(f"EXE_DIST={exe_path}")
    print(f"EXE_ZIP={archive_base}.zip")


if __name__ == "__main__":
    build()
