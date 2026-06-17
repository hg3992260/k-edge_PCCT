import base64
import copy
import io
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import glymur
import matplotlib
import numpy as np
import pydicom
from pydicom.uid import SecondaryCaptureImageStorage, generate_uid
from scipy.ndimage import zoom as cpu_zoom

try:
    import cupy as cp
    from cupyx.scipy.ndimage import zoom as gpu_zoom
except Exception:
    cp = None
    gpu_zoom = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt



def default_app_root():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def default_data_dir(root: Path) -> Path:
    preferred = [
        root / "dcm",
        root / "P10",
        root / "00020006",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate
    return root / "00020006"


ROOT = Path(os.environ.get("KEDGE_APP_ROOT", str(default_app_root())))
DATA_DIR = Path(os.environ.get("KEDGE_DATA_DIR", str(default_data_dir(ROOT))))
OUT_DIR = Path(os.environ.get("KEDGE_OUT_DIR", str(ROOT / "reconstructed_weight_maps")))
CACHE_DIR = Path(os.environ.get("KEDGE_CACHE_DIR", str(ROOT / "_pcct_cache")))


def env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

CUDA_ENABLED = False
CUDA_BACKEND = "NumPy"
CUDA_DISABLED_BY_ENV = env_flag("KEDGE_DISABLE_CUDA", False)
if not CUDA_DISABLED_BY_ENV and cp is not None and gpu_zoom is not None:
    try:
        if cp.cuda.runtime.getDeviceCount() > 0:
            CUDA_ENABLED = True
            CUDA_BACKEND = f"CuPy / {cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')}"
    except Exception:
        CUDA_ENABLED = False
        CUDA_BACKEND = "NumPy"

CPU_COUNT = os.cpu_count() or 1
DEFAULT_SLICE_WORKERS = 2 if CUDA_ENABLED else min(4, max(CPU_COUNT // 4, 1))
SLICE_WORKERS = max(1, int(os.environ.get("KEDGE_SLICE_WORKERS", str(DEFAULT_SLICE_WORKERS))))
DECODE_THREADS = max(1, int(os.environ.get("KEDGE_DECODE_THREADS", str(min(8, max(CPU_COUNT // max(SLICE_WORKERS, 1), 1))))))
FAST_VOLUME_SAVE = env_flag("KEDGE_FAST_VOLUME_SAVE", True)
BLOCK_SUPPRESS_ENABLED = env_flag("KEDGE_BLOCK_SUPPRESS", True)
BLOCK_SUPPRESS_STRENGTH = float(os.environ.get("KEDGE_BLOCK_SUPPRESS_STRENGTH", "0.70"))
PROJECTION_BLOCK_SUPPRESS_STRENGTH = float(os.environ.get("KEDGE_PROJECTION_BLOCK_SUPPRESS_STRENGTH", "0.45"))
RECON_MODE = os.environ.get("KEDGE_RECON_MODE", "spectral_prior").strip().lower()
if RECON_MODE not in {"spectral_prior", "direct_projection"}:
    RECON_MODE = "spectral_prior"
SPECTRAL_PRIOR_ROW_WINDOW = max(3, int(os.environ.get("KEDGE_SPECTRAL_PRIOR_ROW_WINDOW", "9")))
SPECTRAL_PRIOR_COL_WINDOW = max(9, int(os.environ.get("KEDGE_SPECTRAL_PRIOR_COL_WINDOW", "257")))
SPECTRAL_PRIOR_STRUCTURE_BLEND = float(os.environ.get("KEDGE_SPECTRAL_PRIOR_STRUCTURE_BLEND", "0.35"))
SPECTRAL_PRIOR_DENSITY_BLEND = float(os.environ.get("KEDGE_SPECTRAL_PRIOR_DENSITY_BLEND", "0.55"))

EMPTY_TILES = {0, 7, 56, 63}
BIN_LABELS = ["<28", "28-33", "33-38*", "38-48", "48-62", "62-80", "80-105", ">105"]
BIN_RANGES = [(0, 28), (28, 33), (33, 38), (38, 48), (48, 62), (62, 80), (80, 105), (105, 200)]
MATERIALS = ["Water", "Iodine", "Calcium", "Gadolinium"]

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def as_numpy(x):
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def array_module(x):
    if cp is not None and isinstance(x, cp.ndarray):
        return cp
    return np


def to_device(x):
    if CUDA_ENABLED:
        return cp.asarray(x)
    return np.asarray(x)


def backend_zoom(x, zoom_factors, order=1):
    xp = array_module(x)
    if xp is cp:
        return gpu_zoom(x, zoom_factors, order=order)
    return cpu_zoom(np.asarray(x), zoom_factors, order=order)


def fig_to_b64(fig, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return encoded


def b64_to_png_file(encoded, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(encoded))


MEDIASTINAL_WINDOW_CENTER = 40.0
MEDIASTINAL_WINDOW_WIDTH = 400.0


def mediastinal_window_limits():
    half_width = MEDIASTINAL_WINDOW_WIDTH / 2.0
    return MEDIASTINAL_WINDOW_CENTER - half_width, MEDIASTINAL_WINDOW_CENTER + half_width


def add_threshold_colorbar(fig, ax, image_handle, lo, hi, label):
    cbar = fig.colorbar(image_handle, ax=ax, orientation="vertical", fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(f"{label}\n阈值范围 {lo:.3f} ~ {hi:.3f}", fontsize=8)
    return cbar


def array_to_png_b64(x, lo, hi, cmap="gray"):
    arr = np.asarray(x, dtype=np.float64)
    if hi - lo < 1e-9:
        norm = np.zeros_like(arr, dtype=np.float64)
    else:
        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    buf = io.BytesIO()
    plt.imsave(buf, norm, cmap=cmap, format="png", vmin=0.0, vmax=1.0)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def downsample_for_preview(x, max_size=1024):
    arr = np.asarray(x)
    step = max(1, int(np.ceil(max(arr.shape) / float(max_size))))
    return arr[::step, ::step]


IODINE_CMAP_NAME = "magma"
KEDGE_CMAP_NAME = "viridis"
IODINE_CMAP_RANGE = (0.06, 0.88)
KEDGE_CMAP_RANGE = (0.08, 0.86)


def get_colormap(name, samples=None, value_range=None):
    registry = getattr(matplotlib, "colormaps", None)
    if registry is not None:
        cmap = registry.get_cmap(name)
    elif samples is not None:
        cmap = matplotlib.cm.get_cmap(name, samples)
    else:
        cmap = matplotlib.cm.get_cmap(name)

    if value_range is not None:
        lo, hi = value_range
        lo = float(np.clip(lo, 0.0, 1.0))
        hi = float(np.clip(hi, lo + 1e-6, 1.0))
        sample_count = samples or 256
        vals = np.linspace(lo, hi, sample_count)
        return matplotlib.colors.ListedColormap(cmap(vals))

    if samples is not None and hasattr(cmap, "resampled"):
        return cmap.resampled(samples)
    return cmap


def body_mask_from_pixel(pixel_data):
    pixel_data = np.asarray(pixel_data, dtype=np.float64)
    return pixel_data > np.percentile(pixel_data, 50)


def percentile_range_in_mask(x, mask=None, lo=1, hi=99, fallback=None):
    valid = np.asarray(x, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape == valid.shape:
            valid = valid[mask]
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return fallback if fallback is not None else (-1.0, 1.0)
    lo_v = float(np.percentile(valid, lo))
    hi_v = float(np.percentile(valid, hi))
    if hi_v - lo_v < 1e-6:
        center = float(np.median(valid))
        span = max(float(np.std(valid)), 1e-3)
        lo_v = center - span
        hi_v = center + span
    return lo_v, hi_v


def build_colormap_lut(name, value_range=None):
    cmap = get_colormap(name, 256, value_range=value_range)
    lut = (cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).round().astype(np.uint8)
    return lut.tolist()


def build_colormap_css_gradient(name, steps=16, value_range=None):
    cmap = get_colormap(name, steps, value_range=value_range)
    stops = []
    for idx, rgba in enumerate(cmap(np.linspace(0.0, 1.0, steps))):
        pct = 100.0 * idx / max(steps - 1, 1)
        rgb = tuple(int(round(v * 255.0)) for v in rgba[:3])
        stops.append(f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]}) {pct:.1f}%")
    return "linear-gradient(to top, " + ", ".join(stops) + ")"


def build_iodine_interactive_payload(pixel_data, iodine_full, kedge_full, overlay_alpha=0.45):
    p_lo, p_hi = mediastinal_window_limits()
    body = body_mask_from_pixel(pixel_data)
    i_lo, i_hi = percentile_range_in_mask(iodine_full, body, lo=5, hi=99.5, fallback=(0.0, 1.0))
    k_lo, k_hi = percentile_range_in_mask(kedge_full, body, lo=5, hi=99.5, fallback=(0.0, 1.0))
    pixel_preview = downsample_for_preview(pixel_data, max_size=1024)
    iodine_preview = downsample_for_preview(iodine_full, max_size=1024)
    kedge_preview = downsample_for_preview(kedge_full, max_size=1024)
    return {
        "pixel_png": array_to_png_b64(pixel_preview, p_lo, p_hi, cmap="gray"),
        "iodine_png": array_to_png_b64(iodine_preview, i_lo, i_hi, cmap="gray"),
        "kedge_png": array_to_png_b64(kedge_preview, k_lo, k_hi, cmap="gray"),
        "width": int(pixel_preview.shape[1]),
        "height": int(pixel_preview.shape[0]),
        "threshold_min": float(i_lo),
        "threshold_max": float(i_hi),
        "threshold_default": float(i_lo),
        "kedge_threshold_min": float(k_lo),
        "kedge_threshold_max": float(k_hi),
        "overlay_alpha": float(overlay_alpha),
        "iodine_lut": build_colormap_lut(IODINE_CMAP_NAME, value_range=IODINE_CMAP_RANGE),
        "iodine_gradient": build_colormap_css_gradient(IODINE_CMAP_NAME, steps=24, value_range=IODINE_CMAP_RANGE),
        "kedge_lut": build_colormap_lut(KEDGE_CMAP_NAME, value_range=KEDGE_CMAP_RANGE),
        "kedge_gradient": build_colormap_css_gradient(KEDGE_CMAP_NAME, steps=24, value_range=KEDGE_CMAP_RANGE),
    }


def write_native_report(payload):
    out_json = OUT_DIR / "iodine_kedge_report_native.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_json


def representative_cache_path():
    return OUT_DIR / "representative_products.npz"


def save_representative_cache(source_path, products):
    np.savez_compressed(
        representative_cache_path(),
        source_path=np.asarray(str(source_path)),
        recon_mode=np.asarray(products.get("recon_mode", RECON_MODE)),
        pixel_data=np.asarray(products["pixel_data"], dtype=np.float32),
        structure_base=np.asarray(products["structure_base"], dtype=np.float32),
        g1_g0_full=np.asarray(products["g1_g0_full"], dtype=np.float32),
        band_stack_raw=np.asarray(products.get("band_stack_raw", products["band_stack"]), dtype=np.float32),
        band_stack=np.asarray(products["band_stack"], dtype=np.float32),
        bin_curve_matrix=np.asarray(products["bin_curve_matrix"], dtype=np.float32),
        iodine_weights=np.asarray(products["iodine_weights"], dtype=np.float32),
        kedge_weights=np.asarray(products["kedge_weights"], dtype=np.float32),
        iodine_low_direct=np.asarray(products.get("iodine_low_direct", products["iodine_low"]), dtype=np.float32),
        kedge_low_direct=np.asarray(products.get("kedge_low_direct", products["kedge_low"]), dtype=np.float32),
        iodine_low=np.asarray(products["iodine_low"], dtype=np.float32),
        kedge_low=np.asarray(products["kedge_low"], dtype=np.float32),
        iodine_full=np.asarray(products["iodine_full"], dtype=np.float32),
        kedge_full=np.asarray(products["kedge_full"], dtype=np.float32),
        iodine_coarse=np.asarray(products["iodine_coarse"], dtype=np.float32),
        kedge_coarse=np.asarray(products["kedge_coarse"], dtype=np.float32),
    )


def load_representative_cache():
    cache_path = representative_cache_path()
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=False)
    return {
        "source_path": str(data["source_path"].item()),
        "recon_mode": str(data["recon_mode"].item()) if "recon_mode" in data else "direct_projection",
        "pixel_data": np.asarray(data["pixel_data"], dtype=np.float64),
        "structure_base": np.asarray(data["structure_base"], dtype=np.float64),
        "g1_g0_full": np.asarray(data["g1_g0_full"], dtype=np.float64),
        "band_stack_raw": np.asarray(data["band_stack_raw"], dtype=np.float64) if "band_stack_raw" in data else np.asarray(data["band_stack"], dtype=np.float64),
        "band_stack": np.asarray(data["band_stack"], dtype=np.float64),
        "bin_curve_matrix": np.asarray(data["bin_curve_matrix"], dtype=np.float64),
        "iodine_weights": np.asarray(data["iodine_weights"], dtype=np.float64),
        "kedge_weights": np.asarray(data["kedge_weights"], dtype=np.float64),
        "iodine_low_direct": np.asarray(data["iodine_low_direct"], dtype=np.float64) if "iodine_low_direct" in data else np.asarray(data["iodine_low"], dtype=np.float64),
        "kedge_low_direct": np.asarray(data["kedge_low_direct"], dtype=np.float64) if "kedge_low_direct" in data else np.asarray(data["kedge_low"], dtype=np.float64),
        "iodine_low": np.asarray(data["iodine_low"], dtype=np.float64),
        "kedge_low": np.asarray(data["kedge_low"], dtype=np.float64),
        "iodine_full": np.asarray(data["iodine_full"], dtype=np.float64),
        "kedge_full": np.asarray(data["kedge_full"], dtype=np.float64),
        "iodine_coarse": np.asarray(data["iodine_coarse"], dtype=np.float64),
        "kedge_coarse": np.asarray(data["kedge_coarse"], dtype=np.float64),
    }


def save_volume_file(out_path, **arrays):
    if FAST_VOLUME_SAVE:
        np.savez(out_path, **arrays)
        return
    np.savez_compressed(out_path, **arrays)


def lowres_body_mask_from_pixel(pixel_data):
    body = np.asarray(pixel_data > np.percentile(pixel_data, 50), dtype=bool)
    if body.shape[0] % 8 != 0:
        raise ValueError(f"Unexpected pixel_data shape for low-res body downsampling: {body.shape}")
    lowres = body.reshape(body.shape[0] // 8, 8, body.shape[1]).mean(axis=1)
    return lowres > 0.2


def block_reduce_grid(img, row_step, col_step, mask=None):
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    grid_h = h // row_step
    grid_w = w // col_step
    out = np.zeros((grid_h, grid_w), dtype=np.float64)
    for iy in range(grid_h):
        for ix in range(grid_w):
            y0 = iy * row_step
            y1 = y0 + row_step
            x0 = ix * col_step
            x1 = x0 + col_step
            patch = img[y0:y1, x0:x1]
            if mask is not None:
                patch_mask = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
                vals = patch[patch_mask]
                if vals.size < 8:
                    vals = patch[np.isfinite(patch)]
            else:
                vals = patch[np.isfinite(patch)]
            out[iy, ix] = float(np.median(vals)) if vals.size else 0.0
    return out


def expand_block_grid(grid, row_step, col_step, target_shape):
    expanded = np.repeat(np.repeat(np.asarray(grid, dtype=np.float64), row_step, axis=0), col_step, axis=1)
    return expanded[: target_shape[0], : target_shape[1]]


def estimate_block_bias(img, row_step=32, col_step=256, mask=None):
    cell_grid = block_reduce_grid(img, row_step, col_step, mask=mask)
    row_trend = np.median(cell_grid, axis=1, keepdims=True)
    col_trend = np.median(cell_grid, axis=0, keepdims=True)
    global_level = float(np.median(cell_grid))
    interaction = cell_grid - (row_trend + col_trend - global_level)
    return expand_block_grid(interaction, row_step, col_step, np.asarray(img).shape)


def suppress_lowres_block_artifacts(img, strength, mask=None, row_step=32, col_step=256):
    if strength <= 0:
        return np.asarray(img, dtype=np.float64)
    img = np.asarray(img, dtype=np.float64)
    block_bias = estimate_block_bias(img, row_step=row_step, col_step=col_step, mask=mask)
    return img - strength * block_bias


def suppress_band_stack_block_artifacts(band_stack, lowres_body, strength):
    band_stack = np.asarray(band_stack, dtype=np.float64)
    out = np.empty_like(band_stack)
    for band_idx in range(band_stack.shape[0]):
        out[band_idx] = suppress_lowres_block_artifacts(
            band_stack[band_idx],
            strength=strength,
            mask=lowres_body,
            row_step=32,
            col_step=256,
        )
    return out


def smooth_1d_signal(x, window):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or window <= 1:
        return x.copy()
    window = min(window if window % 2 == 1 else window + 1, x.size if x.size % 2 == 1 else max(x.size - 1, 1))
    if window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, kernel, mode="valid")


def robust_normalize_1d(x, lo=1, hi=99):
    x = np.asarray(x, dtype=np.float64)
    valid = x[np.isfinite(x)]
    if valid.size == 0:
        return np.zeros_like(x, dtype=np.float64)
    lo_v = float(np.percentile(valid, lo))
    hi_v = float(np.percentile(valid, hi))
    if hi_v - lo_v < 1e-9:
        return np.zeros_like(x, dtype=np.float64)
    return np.clip((x - lo_v) / (hi_v - lo_v), 0.0, 1.0)


def masked_profile_median(img, mask, axis):
    img = np.asarray(img, dtype=np.float64)
    if mask is None:
        return np.median(img, axis=axis)
    masked = np.where(np.asarray(mask, dtype=bool), img, np.nan)
    out = np.nanmedian(masked, axis=axis)
    fallback = np.nanmedian(img, axis=axis)
    out = np.where(np.isfinite(out), out, fallback)
    return np.nan_to_num(out, nan=0.0)


def downsample_rows_mean(x, factor=8):
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] % factor != 0:
        raise ValueError(f"Unexpected shape for row downsample: {x.shape}, factor={factor}")
    return x.reshape(x.shape[0] // factor, factor, x.shape[1]).mean(axis=1)


def compute_direct_low_projection(band_stack, weights):
    if CUDA_ENABLED:
        return as_numpy(cp.tensordot(to_device(weights), to_device(band_stack), axes=(0, 0)))
    return np.tensordot(weights, band_stack, axes=(0, 0))


def build_spectral_prior_lowres(direct_low, pixel_data, structure_base, lowres_body):
    direct_low = np.asarray(direct_low, dtype=np.float64)
    if BLOCK_SUPPRESS_ENABLED:
        direct_low = suppress_lowres_block_artifacts(
            direct_low,
            strength=PROJECTION_BLOCK_SUPPRESS_STRENGTH,
            mask=lowres_body,
            row_step=32,
            col_step=256,
        )

    row_profile = masked_profile_median(direct_low, lowres_body, axis=1)
    col_profile = masked_profile_median(direct_low, lowres_body, axis=0)
    row_profile = smooth_1d_signal(row_profile, SPECTRAL_PRIOR_ROW_WINDOW)
    col_profile = smooth_1d_signal(col_profile, SPECTRAL_PRIOR_COL_WINDOW)
    row_norm = robust_normalize_1d(row_profile)
    col_norm = robust_normalize_1d(col_profile)

    pixel_low = downsample_rows_mean(pixel_data, factor=8)
    structure_pos = orient_full_map_positive(structure_base, pixel_data)
    structure_low = downsample_rows_mean(structure_pos, factor=8)
    density_norm = normalize_in_mask(pixel_low, lowres_body)
    structure_norm = normalize_in_mask(structure_low, lowres_body)

    direct_norm = normalize_in_mask(direct_low, lowres_body)
    separable_prior = 0.5 * row_norm[:, None] + 0.5 * col_norm[None, :]
    spectral_prior = 0.55 * separable_prior + 0.45 * direct_norm
    coarse = (1.0 - SPECTRAL_PRIOR_STRUCTURE_BLEND) * spectral_prior + SPECTRAL_PRIOR_STRUCTURE_BLEND * structure_norm
    coarse = (1.0 - SPECTRAL_PRIOR_DENSITY_BLEND) * coarse + SPECTRAL_PRIOR_DENSITY_BLEND * density_norm
    coarse = normalize_in_mask(coarse, lowres_body)
    coarse[~lowres_body] = 0.0
    return coarse, direct_low, row_profile, col_profile


def read_pixel_array(ds):
    arr = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def ordered_dicom_files(data_dir):
    data_dir = Path(data_dir)
    if data_dir.is_file():
        search_paths = [data_dir]
    else:
        search_paths = sorted(path for path in data_dir.rglob("*") if path.is_file())

    series_groups = {}
    fallback_counter = 0
    for path in search_paths:
        try:
            ds = pydicom.dcmread(str(path), force=True)
        except Exception:
            continue
        if (0x7FE0, 0x0010) not in ds or (0xEFE1, 0x1001) not in ds:
            continue

        series_uid = str(getattr(ds, "SeriesInstanceUID", "")) or "__NO_SERIES__"
        if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
            order_key = float(ds.ImagePositionPatient[2])
        elif hasattr(ds, "SliceLocation"):
            order_key = float(ds.SliceLocation)
        elif hasattr(ds, "InstanceNumber"):
            order_key = float(ds.InstanceNumber)
        else:
            order_key = float(fallback_counter)
            fallback_counter += 1

        series_groups.setdefault(series_uid, []).append((order_key, path))

    if not series_groups:
        return []

    # Prefer the largest valid series so roots containing multiple exports can
    # still be used directly.
    best_series_uid, best_items = max(
        series_groups.items(),
        key=lambda item: (
            len(item[1]),
            sum(1 for _, p in item[1] if p.parent == data_dir) if data_dir.is_dir() else 0,
            item[0],
        ),
    )
    best_items.sort(key=lambda item: (item[0], item[1].name))
    return [path for _, path in best_items]


def compute_slice_products(ds, prefix):
    blob = ds[0xEFE1, 0x1001].value
    pixel_data = read_pixel_array(ds)
    lowres_body = lowres_body_mask_from_pixel(pixel_data)
    markers = find_markers(blob)
    curves = parse_curves(blob)

    g0_tiles = decode_group(blob, markers, 0, prefix)
    g1_tiles = decode_group(blob, markers, 1, prefix)
    g2_tiles = decode_group(blob, markers, 2, prefix)
    g3_tiles = decode_group(blob, markers, 3, prefix)
    diff_tiles = g1_tiles - g0_tiles
    g1_g0_full = assemble_interleaved(diff_tiles)
    structure_base = assemble_interleaved(g3_tiles - g2_tiles)
    band_stack_raw = assemble_band_stack(diff_tiles)
    if BLOCK_SUPPRESS_ENABLED:
        band_stack = suppress_band_stack_block_artifacts(
            band_stack_raw,
            lowres_body=lowres_body,
            strength=BLOCK_SUPPRESS_STRENGTH,
        )
    else:
        band_stack = band_stack_raw

    bin_curve_matrix = compute_bin_curve_matrix(curves)
    iodine_weights, kedge_weights = build_weight_vectors(bin_curve_matrix)

    iodine_low_direct = compute_direct_low_projection(band_stack, iodine_weights)
    kedge_low_direct = compute_direct_low_projection(band_stack, kedge_weights)

    if RECON_MODE == "spectral_prior":
        iodine_low, iodine_low_direct, iodine_row_profile, iodine_col_profile = build_spectral_prior_lowres(
            iodine_low_direct, pixel_data, structure_base, lowres_body
        )
        kedge_low, kedge_low_direct, kedge_row_profile, kedge_col_profile = build_spectral_prior_lowres(
            kedge_low_direct, pixel_data, structure_base, lowres_body
        )
    else:
        iodine_low = np.asarray(iodine_low_direct, dtype=np.float64)
        kedge_low = np.asarray(kedge_low_direct, dtype=np.float64)
        if BLOCK_SUPPRESS_ENABLED:
            iodine_low = suppress_lowres_block_artifacts(
                iodine_low,
                strength=PROJECTION_BLOCK_SUPPRESS_STRENGTH,
                mask=lowres_body,
                row_step=32,
                col_step=256,
            )
            kedge_low = suppress_lowres_block_artifacts(
                kedge_low,
                strength=PROJECTION_BLOCK_SUPPRESS_STRENGTH,
                mask=lowres_body,
                row_step=32,
                col_step=256,
            )
        iodine_row_profile = masked_profile_median(iodine_low, lowres_body, axis=1)
        iodine_col_profile = masked_profile_median(iodine_low, lowres_body, axis=0)
        kedge_row_profile = masked_profile_median(kedge_low, lowres_body, axis=1)
        kedge_col_profile = masked_profile_median(kedge_low, lowres_body, axis=0)

    if CUDA_ENABLED:
        structure_base_dev = to_device(structure_base)
        pixel_data_dev = to_device(pixel_data)
        iodine_full_dev, iodine_coarse_dev = fuse_structure_preserving_map(
            structure_base_dev, to_device(iodine_low), pixel_data_dev
        )
        kedge_full_dev, kedge_coarse_dev = fuse_structure_preserving_map(
            structure_base_dev, to_device(kedge_low), pixel_data_dev
        )
        iodine_full = as_numpy(iodine_full_dev)
        kedge_full = as_numpy(kedge_full_dev)
        iodine_coarse = as_numpy(iodine_coarse_dev)
        kedge_coarse = as_numpy(kedge_coarse_dev)
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
    else:
        iodine_full, iodine_coarse = fuse_structure_preserving_map(structure_base, iodine_low, pixel_data)
        kedge_full, kedge_coarse = fuse_structure_preserving_map(structure_base, kedge_low, pixel_data)
    return {
        "recon_mode": RECON_MODE,
        "pixel_data": pixel_data,
        "lowres_body": lowres_body,
        "g1_g0_full": g1_g0_full,
        "structure_base": structure_base,
        "band_stack_raw": band_stack_raw,
        "band_stack": band_stack,
        "bin_curve_matrix": bin_curve_matrix,
        "iodine_weights": iodine_weights,
        "kedge_weights": kedge_weights,
        "iodine_low_direct": iodine_low_direct,
        "kedge_low_direct": kedge_low_direct,
        "iodine_row_profile": iodine_row_profile,
        "iodine_col_profile": iodine_col_profile,
        "kedge_row_profile": kedge_row_profile,
        "kedge_col_profile": kedge_col_profile,
        "iodine_low": iodine_low,
        "kedge_low": kedge_low,
        "iodine_full": iodine_full,
        "kedge_full": kedge_full,
        "iodine_coarse": iodine_coarse,
        "kedge_coarse": kedge_coarse,
    }


def build_volume_mpr_figure(pixel_volume, iodine_volume, kedge_volume):
    z = pixel_volume.shape[0] // 2
    y = pixel_volume.shape[1] // 2
    x = pixel_volume.shape[2] // 2
    views = [
        ("原始体数据 Axial", pixel_volume[z], "gray"),
        ("碘体数据 Axial", iodine_volume[z], "iodine"),
        ("K-edge 体数据 Axial", kedge_volume[z], "kedge"),
        ("原始体数据 Coronal", np.flipud(pixel_volume[:, y, :]), "gray"),
        ("碘体数据 Coronal", np.flipud(iodine_volume[:, y, :]), "iodine"),
        ("K-edge 体数据 Coronal", np.flipud(kedge_volume[:, y, :]), "kedge"),
        ("原始体数据 Sagittal", np.flipud(pixel_volume[:, :, x]), "gray"),
        ("碘体数据 Sagittal", np.flipud(iodine_volume[:, :, x]), "iodine"),
        ("K-edge 体数据 Sagittal", np.flipud(kedge_volume[:, :, x]), "kedge"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for ax, (title, img, cmap) in zip(axes.ravel(), views):
        if cmap == "gray":
            v1, v2 = mediastinal_window_limits()
            cmap_obj = "gray"
        else:
            view_body = body_mask_from_pixel(img) if img.shape == pixel_volume[z].shape else (np.asarray(img) > np.percentile(img, 50))
            v1, v2 = percentile_range_in_mask(img, view_body, 5, 99.5, fallback=(0.0, 1.0))
            cmap_obj = get_colormap(
                IODINE_CMAP_NAME if cmap == "iodine" else KEDGE_CMAP_NAME,
                value_range=IODINE_CMAP_RANGE if cmap == "iodine" else KEDGE_CMAP_RANGE,
            )
        im = ax.imshow(img, cmap=cmap_obj, vmin=v1, vmax=v2)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        if cmap != "gray":
            add_threshold_colorbar(fig, ax, im, v1, v2, "颜色映射")
    fig.suptitle("3D 数据集的三正交面概览", fontsize=15, y=0.98)
    fig.tight_layout()
    return fig_to_b64(fig)


def find_markers(blob):
    markers = []
    pos = -1
    while True:
        pos = blob.find(b"jp2c", pos + 1)
        if pos == -1:
            break
        markers.append(pos)
    return markers


def decode_group(blob, markers, group_id, prefix):
    CACHE_DIR.mkdir(exist_ok=True)
    tiles = np.zeros((60, 256, 256), dtype=np.float64)
    start_slot = group_id * 64
    slots = [slot for slot in range(start_slot, start_slot + 64) if (slot - start_slot) not in EMPTY_TILES]

    def decode_slot(slot):
        start = markers[slot] + 4
        end = markers[slot + 1] if slot + 1 < len(markers) else len(blob)
        jp2_path = CACHE_DIR / f"{prefix}_g{group_id}_{slot}.jp2k"
        expected_size = end - start
        if not jp2_path.exists() or jp2_path.stat().st_size != expected_size:
            with open(jp2_path, "wb") as fh:
                fh.write(blob[start:end])
        return glymur.Jp2k(str(jp2_path))[:].astype(np.float64)

    if DECODE_THREADS > 1 and len(slots) > 1:
        with ThreadPoolExecutor(max_workers=min(DECODE_THREADS, len(slots)), thread_name_prefix=f"jp2-g{group_id}") as executor:
            for out_idx, tile in enumerate(executor.map(decode_slot, slots)):
                tiles[out_idx] = tile
    else:
        for out_idx, slot in enumerate(slots):
            tiles[out_idx] = decode_slot(slot)
    return tiles


def assemble_interleaved(tiles):
    full = np.zeros((2048, 2048), dtype=np.float64)
    idx = 0
    for tile_pos in range(64):
        if tile_pos in EMPTY_TILES:
            continue
        tile_row = tile_pos // 8
        tile_col = tile_pos % 8
        tile = tiles[idx]
        idx += 1
        for band_idx in range(8):
            row0 = (tile_row * 8 + band_idx) * 32
            row1 = row0 + 32
            col0 = tile_col * 256
            col1 = col0 + 256
            full[row0:row1, col0:col1] = tile[band_idx * 32 : (band_idx + 1) * 32, :]
    return full


def assemble_band_stack(diff_tiles):
    bands = np.zeros((8, 256, 2048), dtype=np.float64)
    idx = 0
    for tile_pos in range(64):
        if tile_pos in EMPTY_TILES:
            continue
        tile_row = tile_pos // 8
        tile_col = tile_pos % 8
        tile = diff_tiles[idx]
        idx += 1
        for band_idx in range(8):
            row0 = tile_row * 32
            row1 = row0 + 32
            col0 = tile_col * 256
            col1 = col0 + 256
            bands[band_idx, row0:row1, col0:col1] = tile[band_idx * 32 : (band_idx + 1) * 32, :]
    return bands


def parse_curves(blob):
    return {
        "Water": np.frombuffer(blob[0x1C0 : 0x1C0 + 200 * 4], dtype=np.float32),
        "Iodine": np.frombuffer(blob[0x850 : 0x850 + 200 * 4], dtype=np.float32),
        "Calcium": np.frombuffer(blob[0xB70 : 0xB70 + 200 * 4], dtype=np.float32),
        "Gadolinium": np.frombuffer(blob[0xED4 : 0xED4 + 200 * 4], dtype=np.float32),
    }


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / max(x.std(), 1e-9)


def robust_limits(x, pct=99.5):
    valid = np.asarray(x, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return -1.0, 1.0
    vmax = np.percentile(np.abs(valid), pct)
    vmax = max(float(vmax), 1e-6)
    return -vmax, vmax


def percentile_range(x, lo=1, hi=99):
    valid = np.asarray(x, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return -1.0, 1.0
    return float(np.percentile(valid, lo)), float(np.percentile(valid, hi))


def scale_for_display(x):
    lo, hi = percentile_range(x, 1, 99)
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=np.float64)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def normalize_in_mask(x, mask):
    xp = array_module(x)
    x = xp.asarray(x, dtype=xp.float64)
    valid = x[mask]
    if valid.size == 0:
        return xp.zeros_like(x, dtype=xp.float64)
    lo = float(xp.percentile(valid, 1))
    hi = float(xp.percentile(valid, 99))
    if hi - lo < 1e-9:
        return xp.zeros_like(x, dtype=xp.float64)
    return xp.clip((x - lo) / (hi - lo), 0.0, 1.0)


def project_out(signal, nuisance_columns):
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    nuisance_columns = np.asarray(nuisance_columns, dtype=np.float64)
    if nuisance_columns.ndim == 1:
        nuisance_columns = nuisance_columns[:, None]
    if nuisance_columns.shape[0] != signal.shape[0]:
        raise ValueError(
            f"project_out shape mismatch: signal={signal.shape}, nuisance={nuisance_columns.shape}"
        )

    # Avoid MKL DGELSD issues on some environments by solving the tiny
    # regularized normal equations directly.
    gram = nuisance_columns.T @ nuisance_columns
    rhs = nuisance_columns.T @ signal
    reg = 1e-8 * np.eye(gram.shape[0], dtype=np.float64)
    try:
        coef = np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        coef = np.matmul(np.linalg.pinv(gram + reg), rhs)
    return signal - nuisance_columns @ coef


def compute_bin_curve_matrix(curves):
    matrix = np.zeros((8, 4), dtype=np.float64)
    for bi, (lo, hi) in enumerate(BIN_RANGES):
        for mi, name in enumerate(MATERIALS):
            matrix[bi, mi] = float(np.mean(curves[name][lo:hi]))
    return matrix


def build_weight_vectors(bin_curve_matrix):
    nuisance = np.stack(
        [
            zscore(bin_curve_matrix[:, 0]),
            zscore(bin_curve_matrix[:, 2]),
            zscore(bin_curve_matrix[:, 3]),
            np.ones(8, dtype=np.float64),
        ],
        axis=1,
    )

    iodine_target = zscore(bin_curve_matrix[:, 1])
    iodine_vec = project_out(iodine_target, nuisance)
    iodine_vec = iodine_vec / max(np.sum(np.abs(iodine_vec)), 1e-9)

    kedge_target = np.zeros(8, dtype=np.float64)
    kedge_target[1] = -1.0
    kedge_target[2] = 1.0
    kedge_vec = project_out(kedge_target, nuisance)
    kedge_vec = kedge_vec / max(np.sum(np.abs(kedge_vec)), 1e-9)
    return iodine_vec, kedge_vec


def orient_map_positive(map_low, pixel_data):
    xp = array_module(map_low)
    up = backend_zoom(map_low, (8, 1), order=1)
    body = pixel_data > xp.percentile(pixel_data, 50)
    high = pixel_data > xp.percentile(pixel_data, 99.5)
    if int(xp.count_nonzero(high)) < 100:
        high = pixel_data > xp.percentile(pixel_data, 98.5)
    body_mean = float(xp.mean(up[body])) if int(xp.count_nonzero(body)) else 0.0
    high_mean = float(xp.mean(up[high])) if int(xp.count_nonzero(high)) else 0.0
    if high_mean - body_mean < 0:
        return -map_low
    return map_low


def orient_full_map_positive(map_full, pixel_data):
    xp = array_module(map_full)
    body = pixel_data > xp.percentile(pixel_data, 50)
    high = pixel_data > xp.percentile(pixel_data, 99.5)
    if int(xp.count_nonzero(high)) < 100:
        high = pixel_data > xp.percentile(pixel_data, 98.5)
    body_mean = float(xp.mean(map_full[body])) if int(xp.count_nonzero(body)) else 0.0
    high_mean = float(xp.mean(map_full[high])) if int(xp.count_nonzero(high)) else 0.0
    if high_mean - body_mean < 0:
        return -map_full
    return map_full


def fuse_structure_preserving_map(base_full, coarse_low, pixel_data, coarse_weight=0.75):
    xp = array_module(base_full)
    body = pixel_data > xp.percentile(pixel_data, 50)
    base_full = orient_full_map_positive(base_full, pixel_data)
    coarse_up = orient_map_positive(coarse_low, pixel_data)
    coarse_up = backend_zoom(coarse_up, (8, 1), order=1)
    base_norm = normalize_in_mask(base_full, body)
    coarse_norm = normalize_in_mask(coarse_up, body)
    fused = base_norm * (1.0 - coarse_weight + coarse_weight * coarse_norm)
    fused[~body] = 0.0
    return fused, coarse_up


def map_metrics(map_full, pixel_data):
    body = pixel_data > np.percentile(pixel_data, 50)
    high = pixel_data > np.percentile(pixel_data, 99.5)
    if np.count_nonzero(high) < 100:
        high = pixel_data > np.percentile(pixel_data, 98.5)
    soft = body & (pixel_data > np.percentile(pixel_data, 65)) & (pixel_data < np.percentile(pixel_data, 80))
    mask = body & np.isfinite(map_full)
    corr = np.corrcoef(map_full[mask].ravel(), pixel_data[mask].ravel())[0, 1] if np.count_nonzero(mask) > 100 else np.nan
    top = map_full >= np.percentile(map_full[mask], 99.0) if np.count_nonzero(mask) > 100 else np.zeros_like(mask)
    overlap = float(np.count_nonzero(top & high) / max(np.count_nonzero(top), 1))
    high_mean = float(np.mean(map_full[high])) if np.count_nonzero(high) else np.nan
    soft_mean = float(np.mean(map_full[soft])) if np.count_nonzero(soft) else np.nan
    return {
        "corr_pixel": float(corr),
        "top1_overlap_hi_pixel": overlap,
        "high_mean": high_mean,
        "soft_mean": soft_mean,
        "contrast_hi_soft": float(high_mean - soft_mean),
    }


def convert_to_int16(map_full):
    vmax = np.percentile(np.abs(map_full[np.isfinite(map_full)]), 99.5)
    vmax = max(float(vmax), 1e-6)
    slope = vmax / 30000.0
    arr = np.clip(np.round(map_full / slope), -32768, 32767).astype(np.int16)
    return arr, slope


def export_derived_dicom(
    ds,
    pixel_array,
    out_path,
    series_desc,
    derivation_desc,
    slope,
    series_instance_uid=None,
    instance_number=None,
):
    derived = copy.deepcopy(ds)
    derived.file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    derived.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    derived.SOPClassUID = SecondaryCaptureImageStorage
    derived.SOPInstanceUID = derived.file_meta.MediaStorageSOPInstanceUID
    derived.SeriesInstanceUID = series_instance_uid or generate_uid()
    derived.SeriesDescription = series_desc
    derived.ImageType = ["DERIVED", "SECONDARY", "OTHER"]
    derived.Modality = "OT"
    derived.Rows, derived.Columns = pixel_array.shape
    derived.SamplesPerPixel = 1
    derived.PhotometricInterpretation = "MONOCHROME2"
    derived.BitsAllocated = 16
    derived.BitsStored = 16
    derived.HighBit = 15
    derived.PixelRepresentation = 1
    derived.RescaleSlope = float(slope)
    derived.RescaleIntercept = 0.0
    derived.WindowCenter = int(np.median(pixel_array))
    derived.WindowWidth = int(max(np.percentile(pixel_array, 99) - np.percentile(pixel_array, 1), 1))
    derived.DerivationDescription = derivation_desc
    if instance_number is not None:
        derived.InstanceNumber = int(instance_number)
    derived.PixelData = pixel_array.astype(np.int16).tobytes()
    derived.save_as(str(out_path), write_like_original=False)


def process_slice_job(
    idx,
    file_path,
    rep_idx,
    iodine_series_dir,
    kedge_series_dir,
    iodine_series_uid,
    kedge_series_uid,
):
    ds = pydicom.dcmread(str(file_path), force=True)
    products = compute_slice_products(ds, f"iodine_report_{file_path.stem}")
    iodine_i16, iodine_slope = convert_to_int16(products["iodine_full"])
    kedge_i16, kedge_slope = convert_to_int16(products["kedge_full"])

    iodine_slice_path = iodine_series_dir / file_path.name
    kedge_slice_path = kedge_series_dir / file_path.name
    export_derived_dicom(
        ds,
        iodine_i16,
        iodine_slice_path,
        "Derived Iodine Weight Map Series",
        "Structure-preserving heuristic iodine map series built from all DICOM slices in the selected path.",
        iodine_slope,
        series_instance_uid=iodine_series_uid,
        instance_number=getattr(ds, "InstanceNumber", idx + 1),
    )
    export_derived_dicom(
        ds,
        kedge_i16,
        kedge_slice_path,
        "Derived K-edge Weight Map Series",
        "Structure-preserving heuristic K-edge map series built from all DICOM slices in the selected path.",
        kedge_slope,
        series_instance_uid=kedge_series_uid,
        instance_number=getattr(ds, "InstanceNumber", idx + 1),
    )
    volume_entry = {
        "source_path": str(file_path),
        "iodine_path": str(iodine_slice_path),
        "kedge_path": str(kedge_slice_path),
        "pixel": products["pixel_data"].astype(np.float32),
        "iodine": products["iodine_full"].astype(np.float32),
        "kedge": products["kedge_full"].astype(np.float32),
    }
    return {
        "idx": idx,
        "file_path": file_path,
        "volume_entry": volume_entry,
        "rep_products": products if idx == rep_idx else None,
    }


def build_overview_figure(pixel_data, structure_base, iodine_full, kedge_full):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    p_lo, p_hi = mediastinal_window_limits()
    d_lo, d_hi = percentile_range(structure_base)
    body = body_mask_from_pixel(pixel_data)
    i_lo, i_hi = percentile_range_in_mask(iodine_full, body, 5, 99.5, fallback=(0.0, 1.0))
    k_lo, k_hi = percentile_range_in_mask(kedge_full, body, 5, 99.5, fallback=(0.0, 1.0))
    iodine_cmap = get_colormap(IODINE_CMAP_NAME, value_range=IODINE_CMAP_RANGE)
    kedge_cmap = get_colormap(KEDGE_CMAP_NAME, value_range=KEDGE_CMAP_RANGE)

    axes[0, 0].imshow(pixel_data, cmap="gray", vmin=p_lo, vmax=p_hi)
    axes[0, 0].set_title("PixelData")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(structure_base, cmap="gray", vmin=d_lo, vmax=d_hi)
    axes[0, 1].set_title("结构底图 G3 - G2")
    axes[0, 1].axis("off")

    iodine_im = axes[0, 2].imshow(iodine_full, cmap=iodine_cmap, vmin=i_lo, vmax=i_hi)
    axes[0, 2].set_title("碘权重图")
    axes[0, 2].axis("off")
    add_threshold_colorbar(fig, axes[0, 2], iodine_im, i_lo, i_hi, "碘权重")

    kedge_im = axes[1, 0].imshow(kedge_full, cmap=kedge_cmap, vmin=k_lo, vmax=k_hi)
    axes[1, 0].set_title("K-edge 权重图")
    axes[1, 0].axis("off")
    add_threshold_colorbar(fig, axes[1, 0], kedge_im, k_lo, k_hi, "K-edge 权重")

    axes[1, 1].imshow(pixel_data, cmap="gray", vmin=p_lo, vmax=p_hi)
    iodine_overlay_im = axes[1, 1].imshow(iodine_full, cmap=iodine_cmap, alpha=0.55, vmin=i_lo, vmax=i_hi)
    axes[1, 1].set_title("PixelData + 碘叠加")
    axes[1, 1].axis("off")
    add_threshold_colorbar(fig, axes[1, 1], iodine_overlay_im, i_lo, i_hi, "碘叠加")

    axes[1, 2].imshow(pixel_data, cmap="gray", vmin=p_lo, vmax=p_hi)
    kedge_overlay_im = axes[1, 2].imshow(kedge_full, cmap=kedge_cmap, alpha=0.55, vmin=k_lo, vmax=k_hi)
    axes[1, 2].set_title("PixelData + K-edge 叠加")
    axes[1, 2].axis("off")
    add_threshold_colorbar(fig, axes[1, 2], kedge_overlay_im, k_lo, k_hi, "K-edge 叠加")

    fig.suptitle("PixelData 与启发式材料权重图对比", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_difference_figure(pixel_data, iodine_full, kedge_full):
    diff_map = np.asarray(iodine_full, dtype=np.float64) - np.asarray(kedge_full, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    p_lo, p_hi = mediastinal_window_limits()
    diff_lo, diff_hi = robust_limits(diff_map, pct=99.0)

    diff_im = axes[0].imshow(diff_map, cmap="RdBu_r", vmin=diff_lo, vmax=diff_hi)
    axes[0].set_title("碘权重 - K-edge 权重")
    axes[0].axis("off")
    add_threshold_colorbar(fig, axes[0], diff_im, diff_lo, diff_hi, "差异值")

    axes[1].imshow(pixel_data, cmap="gray", vmin=p_lo, vmax=p_hi)
    diff_overlay_im = axes[1].imshow(diff_map, cmap="RdBu_r", alpha=0.45, vmin=diff_lo, vmax=diff_hi)
    axes[1].set_title("PixelData + 差异叠加")
    axes[1].axis("off")
    add_threshold_colorbar(fig, axes[1], diff_overlay_im, diff_lo, diff_hi, "差异叠加")

    fig.suptitle("碘权重与 K-edge 权重的差异图", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_weights_figure(bin_curve_matrix, iodine_weights, kedge_weights):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    xs = np.arange(8)
    colors = {
        "Water": "#58a6ff",
        "Iodine": "#e53935",
        "Calcium": "#7ee787",
        "Gadolinium": "#f0883e",
    }

    for mi, name in enumerate(MATERIALS):
        axes[0].plot(xs, zscore(bin_curve_matrix[:, mi]), marker="o", linewidth=1.2, color=colors[name], label=name)
    axes[0].axvline(1, color="#8b949e", linestyle="--", linewidth=0.8)
    axes[0].axvline(2, color="#e53935", linestyle="--", linewidth=0.8)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(BIN_LABELS, rotation=20)
    axes[0].set_title("8 个能区中的材料响应形状")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].bar(xs - 0.18, iodine_weights, width=0.36, color="#e53935", label="碘权重")
    axes[1].bar(xs + 0.18, kedge_weights, width=0.36, color="#58a6ff", label="K-edge 权重")
    axes[1].axhline(0, color="#8b949e", linewidth=0.8)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(BIN_LABELS, rotation=20)
    axes[1].set_title("由现有材料曲线投影得到的权重向量")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle("重建使用的 8-bin 权重", fontsize=14, y=1.03)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_band_figure(band_stack, iodine_weights, kedge_weights):
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    for band_idx in range(8):
        row, col = divmod(band_idx, 4)
        img = band_stack[band_idx]
        lo, hi = robust_limits(img)
        axes[row, col].imshow(img, cmap="gray", vmin=lo, vmax=hi)
        axes[row, col].set_title(
            f"Bin {band_idx} | {BIN_LABELS[band_idx]}\nI={iodine_weights[band_idx]:+.3f}, K={kedge_weights[band_idx]:+.3f}",
            fontsize=9,
        )
        axes[row, col].axis("off")

    ax = axes[2, 0]
    for band_idx in range(8):
        ax.plot(band_stack[band_idx].mean(axis=1), label=f"B{band_idx}", linewidth=0.9)
    ax.set_title("各能区逐行均值")
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=7)

    ax = axes[2, 1]
    ax.bar(np.arange(8), [band_stack[i].std() for i in range(8)], color=plt.cm.plasma(np.linspace(0.1, 0.9, 8)))
    ax.set_title("各能区标准差")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2, 2]
    ax.plot(np.arange(8), iodine_weights, marker="o", color="#e53935", label="碘")
    ax.plot(np.arange(8), kedge_weights, marker="o", color="#58a6ff", label="K-edge")
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels([f"B{i}" for i in range(8)])
    ax.set_title("权重分配")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 3]
    ax.axis("off")
    ax.text(
        0.05,
        0.95,
        "说明\n\n"
        "1. 每个 Bin 图像由 G1-G0 的同一 band 拼成 256x2048 低分辨率能区图。\n"
        "2. 这些 Bin 图只提供低分辨率频谱信息，不直接作为最终 DICOM。\n"
        "3. 最终输出用完整结构底图 G3-G2 承载空间解剖结构。\n"
        "4. 碘 / K-edge 低分辨率信号只作为调制因子叠加到底图上。",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(facecolor="#161b22", edgecolor="#30363d"),
    )

    fig.suptitle("8 个能区及其对重建的贡献", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_scatter_figure(pixel_data, iodine_full, kedge_full):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    mask = (pixel_data > np.percentile(pixel_data, 50)) & np.isfinite(iodine_full) & np.isfinite(kedge_full)
    ys = pixel_data[mask][::64]
    iod = iodine_full[mask][::64]
    ked = kedge_full[mask][::64]

    axes[0].scatter(ys, iod, s=4, alpha=0.15, color="#e53935")
    axes[0].set_title("PixelData vs 碘权重")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(ys, ked, s=4, alpha=0.15, color="#58a6ff")
    axes[1].set_title("PixelData vs K-edge 权重")
    axes[1].grid(alpha=0.3)

    axes[2].scatter(iod, ked, s=4, alpha=0.15, color="#d2a8ff")
    axes[2].set_title("碘权重 vs K-edge 权重")
    axes[2].grid(alpha=0.3)

    fig.suptitle("派生权重图与原始 PixelData 的关系", fontsize=14, y=1.03)
    fig.tight_layout()
    return fig_to_b64(fig)


def write_html(
    pixel_path,
    iodine_dcm_path,
    kedge_dcm_path,
    backend_name,
    recon_mode,
    overview_img,
    difference_img,
    weights_img,
    band_img,
    scatter_img,
    iodine_metrics,
    kedge_metrics,
    iodine_weights,
    kedge_weights,
    iodine_interactive=None,
):
    iodine_weight_text = ", ".join(f"{v:+.3f}" for v in iodine_weights)
    kedge_weight_text = ", ".join(f"{v:+.3f}" for v in kedge_weights)
    mode_label = "频谱先验版" if recon_mode == "spectral_prior" else "直接 2D 投影版"
    coarse_desc = (
        "先把 direct_low 压成平滑的行/列/全局谱形先验，再交给 G3-G2 承载全分辨率结构"
        if recon_mode == "spectral_prior"
        else "直接对 8-bin 二维图做线性投影，再与 G3-G2 做结构融合"
    )
    interactive_block = ""
    interactive_script = ""
    if iodine_interactive is not None:
        slider_step = max(
            (iodine_interactive["threshold_max"] - iodine_interactive["threshold_min"]) / 200.0,
            0.001,
        )
        interactive_block = f"""
<div class="figure">
<h3 style="margin-top:0;color:#58a6ff">对比视图联动阈值截断</h3>
<div class="interactive-controls">
<label for="iodine-threshold-slider">最低阈值</label>
<input id="iodine-threshold-slider" type="range" min="{iodine_interactive['threshold_min']:.6f}" max="{iodine_interactive['threshold_max']:.6f}" value="{iodine_interactive['threshold_default']:.6f}" step="{slider_step:.6f}">
<span id="iodine-threshold-value" class="mono"></span>
</div>
<div class="interactive-stage">
<div class="interactive-panel">
<div class="interactive-panel-title">碘权重图</div>
<div class="interactive-canvas-wrap">
<canvas id="iodine-map-canvas" width="{iodine_interactive['width']}" height="{iodine_interactive['height']}"></canvas>
</div>
<div class="interactive-scale-wrap">
<div class="interactive-scale-title">碘配色范围</div>
<div class="interactive-scale-track">
<div class="interactive-scale-gradient" style="background:{iodine_interactive['iodine_gradient']}"></div>
<div id="iodine-map-marker" class="interactive-scale-marker"></div>
</div>
<div class="interactive-scale-labels">
<span>{iodine_interactive['threshold_max']:.3f}</span>
<span>{iodine_interactive['threshold_min']:.3f}</span>
</div>
</div>
</div>
<div class="interactive-panel">
<div class="interactive-panel-title">K-edge 权重图</div>
<div class="interactive-canvas-wrap">
<canvas id="kedge-map-canvas" width="{iodine_interactive['width']}" height="{iodine_interactive['height']}"></canvas>
</div>
<div class="interactive-scale-wrap">
<div class="interactive-scale-title">K-edge 配色范围</div>
<div class="interactive-scale-track">
<div class="interactive-scale-gradient" style="background:{iodine_interactive['kedge_gradient']}"></div>
<div id="kedge-map-marker" class="interactive-scale-marker"></div>
</div>
<div class="interactive-scale-labels">
<span>{iodine_interactive['kedge_threshold_max']:.3f}</span>
<span>{iodine_interactive['kedge_threshold_min']:.3f}</span>
</div>
</div>
</div>
<div class="interactive-panel">
<div class="interactive-panel-title">碘叠加视图</div>
<div class="interactive-canvas-wrap">
<canvas id="iodine-threshold-canvas" width="{iodine_interactive['width']}" height="{iodine_interactive['height']}"></canvas>
</div>
<div class="interactive-scale-wrap">
<div class="interactive-scale-title">碘配色范围</div>
<div class="interactive-scale-track">
<div class="interactive-scale-gradient" style="background:{iodine_interactive['iodine_gradient']}"></div>
<div id="iodine-threshold-marker" class="interactive-scale-marker"></div>
</div>
<div class="interactive-scale-labels">
<span>{iodine_interactive['threshold_max']:.3f}</span>
<span>{iodine_interactive['threshold_min']:.3f}</span>
</div>
</div>
</div>
<div class="interactive-panel">
<div class="interactive-panel-title">K-edge 叠加视图</div>
<div class="interactive-canvas-wrap">
<canvas id="kedge-threshold-canvas" width="{iodine_interactive['width']}" height="{iodine_interactive['height']}"></canvas>
</div>
<div class="interactive-scale-wrap">
<div class="interactive-scale-title">K-edge 配色范围</div>
<div class="interactive-scale-track">
<div class="interactive-scale-gradient" style="background:{iodine_interactive['kedge_gradient']}"></div>
<div id="kedge-threshold-marker" class="interactive-scale-marker"></div>
</div>
<div class="interactive-scale-labels">
<span>{iodine_interactive['kedge_threshold_max']:.3f}</span>
<span>{iodine_interactive['kedge_threshold_min']:.3f}</span>
</div>
</div>
</div>
</div>
<div class="note">
当前交互视图使用下采样预览图以提升滑动流畅度。滑块以碘阈值为基准，同时按相同归一化比例联动 K-edge 视图。权重图和叠加图都会联动更新；低于阈值的像素在权重图中转为灰阶，在叠加图中恢复为原始纵膈窗灰阶。
</div>
<img id="iodine-base-img" class="hidden-asset" src="data:image/png;base64,{iodine_interactive['pixel_png']}" alt="iodine-base">
<img id="iodine-mask-img" class="hidden-asset" src="data:image/png;base64,{iodine_interactive['iodine_png']}" alt="iodine-mask">
<img id="kedge-mask-img" class="hidden-asset" src="data:image/png;base64,{iodine_interactive['kedge_png']}" alt="kedge-mask">
</div>
"""
        interactive_script = f"""
<script>
(() => {{
  const slider = document.getElementById("iodine-threshold-slider");
  const valueEl = document.getElementById("iodine-threshold-value");
  const iodineMapCanvas = document.getElementById("iodine-map-canvas");
  const kedgeMapCanvas = document.getElementById("kedge-map-canvas");
  const iodineCanvas = document.getElementById("iodine-threshold-canvas");
  const kedgeCanvas = document.getElementById("kedge-threshold-canvas");
  const iodineMapMarker = document.getElementById("iodine-map-marker");
  const kedgeMapMarker = document.getElementById("kedge-map-marker");
  const iodineMarker = document.getElementById("iodine-threshold-marker");
  const kedgeMarker = document.getElementById("kedge-threshold-marker");
  const baseImg = document.getElementById("iodine-base-img");
  const iodineMaskImg = document.getElementById("iodine-mask-img");
  const kedgeMaskImg = document.getElementById("kedge-mask-img");
  if (!slider || !valueEl || !iodineMapCanvas || !kedgeMapCanvas || !iodineCanvas || !kedgeCanvas || !iodineMapMarker || !kedgeMapMarker || !iodineMarker || !kedgeMarker || !baseImg || !iodineMaskImg || !kedgeMaskImg) {{
    return;
  }}

  const iodineLut = {json.dumps(iodine_interactive["iodine_lut"])};
  const kedgeLut = {json.dumps(iodine_interactive["kedge_lut"])};
  const iodineThresholdMin = {iodine_interactive['threshold_min']:.12f};
  const iodineThresholdMax = {iodine_interactive['threshold_max']:.12f};
  const kedgeThresholdMin = {iodine_interactive['kedge_threshold_min']:.12f};
  const kedgeThresholdMax = {iodine_interactive['kedge_threshold_max']:.12f};
  const overlayAlpha = {iodine_interactive['overlay_alpha']:.12f};
  const iodineMapCtx = iodineMapCanvas.getContext("2d", {{ willReadFrequently: true }});
  const kedgeMapCtx = kedgeMapCanvas.getContext("2d", {{ willReadFrequently: true }});
  const iodineCtx = iodineCanvas.getContext("2d", {{ willReadFrequently: true }});
  const kedgeCtx = kedgeCanvas.getContext("2d", {{ willReadFrequently: true }});
  const offscreenBase = document.createElement("canvas");
  const offscreenIodineMask = document.createElement("canvas");
  const offscreenKedgeMask = document.createElement("canvas");
  const offscreenBaseCtx = offscreenBase.getContext("2d", {{ willReadFrequently: true }});
  const offscreenIodineMaskCtx = offscreenIodineMask.getContext("2d", {{ willReadFrequently: true }});
  const offscreenKedgeMaskCtx = offscreenKedgeMask.getContext("2d", {{ willReadFrequently: true }});
  let basePixels = null;
  let iodineMaskPixels = null;
  let kedgeMaskPixels = null;

  function clampRatio(value, lo, hi) {{
    const denom = Math.max(hi - lo, 1e-9);
    return Math.min(1, Math.max(0, (value - lo) / denom));
  }}

  function updateMarkers(iodineThreshold, kedgeThreshold, ratio) {{
    iodineMapMarker.style.top = `${{(1 - ratio) * 100}}%`;
    kedgeMapMarker.style.top = `${{(1 - ratio) * 100}}%`;
    iodineMarker.style.top = `${{(1 - ratio) * 100}}%`;
    kedgeMarker.style.top = `${{(1 - ratio) * 100}}%`;
    valueEl.textContent = `碘阈值 ${{iodineThreshold.toFixed(3)}} / K-edge 联动阈值 ${{kedgeThreshold.toFixed(3)}}`;
  }}

  function renderWeightMap(ctx, canvas, maskPixels, lut, thresholdNorm) {{
    const out = ctx.createImageData(canvas.width, canvas.height);
    const dst = out.data;
    for (let i = 0; i < dst.length; i += 4) {{
      const maskGray = maskPixels[i];
      const overlayNorm = maskGray / 255;
      if (overlayNorm >= thresholdNorm) {{
        const lutIdx = Math.min(255, Math.max(0, Math.round(overlayNorm * 255)));
        const rgb = lut[lutIdx];
        dst[i] = rgb[0];
        dst[i + 1] = rgb[1];
        dst[i + 2] = rgb[2];
      }} else {{
        dst[i] = maskGray;
        dst[i + 1] = maskGray;
        dst[i + 2] = maskGray;
      }}
      dst[i + 3] = 255;
    }}
    ctx.putImageData(out, 0, 0);
  }}

  function renderOverlay(ctx, canvas, maskPixels, lut, thresholdNorm) {{
    const out = ctx.createImageData(canvas.width, canvas.height);
    const dst = out.data;
    for (let i = 0; i < dst.length; i += 4) {{
      const gray = basePixels[i];
      const overlayNorm = maskPixels[i] / 255;
      if (overlayNorm >= thresholdNorm) {{
        const lutIdx = Math.min(255, Math.max(0, Math.round(overlayNorm * 255)));
        const rgb = lut[lutIdx];
        dst[i] = Math.round(gray * (1 - overlayAlpha) + rgb[0] * overlayAlpha);
        dst[i + 1] = Math.round(gray * (1 - overlayAlpha) + rgb[1] * overlayAlpha);
        dst[i + 2] = Math.round(gray * (1 - overlayAlpha) + rgb[2] * overlayAlpha);
      }} else {{
        dst[i] = gray;
        dst[i + 1] = gray;
        dst[i + 2] = gray;
      }}
      dst[i + 3] = 255;
    }}
    ctx.putImageData(out, 0, 0);
  }}

  function render() {{
    if (!basePixels || !iodineMaskPixels || !kedgeMaskPixels) {{
      return;
    }}
    const iodineThreshold = parseFloat(slider.value);
    const ratio = clampRatio(iodineThreshold, iodineThresholdMin, iodineThresholdMax);
    const kedgeThreshold = kedgeThresholdMin + ratio * (kedgeThresholdMax - kedgeThresholdMin);
    updateMarkers(iodineThreshold, kedgeThreshold, ratio);
    renderWeightMap(iodineMapCtx, iodineMapCanvas, iodineMaskPixels, iodineLut, ratio);
    renderWeightMap(kedgeMapCtx, kedgeMapCanvas, kedgeMaskPixels, kedgeLut, ratio);
    renderOverlay(iodineCtx, iodineCanvas, iodineMaskPixels, iodineLut, ratio);
    renderOverlay(kedgeCtx, kedgeCanvas, kedgeMaskPixels, kedgeLut, ratio);
  }}

  function primePixels() {{
    const width = baseImg.naturalWidth || baseImg.width;
    const height = baseImg.naturalHeight || baseImg.height;
    iodineMapCanvas.width = width;
    iodineMapCanvas.height = height;
    kedgeMapCanvas.width = width;
    kedgeMapCanvas.height = height;
    iodineCanvas.width = width;
    iodineCanvas.height = height;
    kedgeCanvas.width = width;
    kedgeCanvas.height = height;
    offscreenBase.width = width;
    offscreenBase.height = height;
    offscreenIodineMask.width = width;
    offscreenIodineMask.height = height;
    offscreenKedgeMask.width = width;
    offscreenKedgeMask.height = height;
    offscreenBaseCtx.drawImage(baseImg, 0, 0, width, height);
    offscreenIodineMaskCtx.drawImage(iodineMaskImg, 0, 0, width, height);
    offscreenKedgeMaskCtx.drawImage(kedgeMaskImg, 0, 0, width, height);
    basePixels = offscreenBaseCtx.getImageData(0, 0, width, height).data;
    iodineMaskPixels = offscreenIodineMaskCtx.getImageData(0, 0, width, height).data;
    kedgeMaskPixels = offscreenKedgeMaskCtx.getImageData(0, 0, width, height).data;
    render();
  }}

  function waitForImage(img) {{
    if (img.complete && img.naturalWidth > 0) {{
      return Promise.resolve();
    }}
    return new Promise((resolve) => {{
      img.addEventListener("load", resolve, {{ once: true }});
    }});
  }}

  slider.addEventListener("input", render);
  Promise.all([waitForImage(baseImg), waitForImage(iodineMaskImg), waitForImage(kedgeMaskImg)]).then(primePixels);
}})();
</script>
"""
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>碘权重图 / K-edge 权重图重建报告</title>
<style>
body{{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#0d1117;color:#c9d1d9;max-width:1280px;margin:0 auto;padding:20px;line-height:1.7}}
h1{{color:#58a6ff;text-align:center;border-bottom:2px solid #30363d;padding-bottom:12px}}
h2{{color:#f0883e;border-left:4px solid #f0883e;padding-left:12px;margin-top:28px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:.9em}}
th{{background:#21262d;color:#8b949e;padding:8px 12px;border:1px solid #30363d;text-align:left}}
td{{padding:7px 12px;border:1px solid #30363d;vertical-align:top}}
tr:nth-child(even) td{{background:#161b22}}
.figure{{margin:18px 0;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}}
.figure img{{width:100%;border-radius:6px}}
.note{{background:#1f242b;border-left:3px solid #58a6ff;padding:12px 16px;margin:12px 0}}
.warn{{background:#1f242b;border-left:3px solid #d2991d;padding:12px 16px;margin:12px 0}}
.ok{{background:#1f242b;border-left:3px solid #7ee787;padding:12px 16px;margin:12px 0}}
.mono{{font-family:Consolas,monospace;color:#a5d6ff}}
a{{color:#58a6ff}}
ul{{margin:8px 0 8px 20px}}
ol{{margin:8px 0 8px 20px}}
code{{font-family:Consolas,monospace;background:#161b22;padding:1px 4px;border-radius:4px;color:#a5d6ff}}
.interactive-controls{{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:14px}}
.interactive-controls input[type=range]{{flex:1;min-width:280px}}
.interactive-stage{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px;align-items:start}}
.interactive-panel{{display:grid;grid-template-columns:minmax(0,1fr) 84px;gap:16px;align-items:start}}
.interactive-panel-title{{grid-column:1 / -1;color:#58a6ff;font-size:.95em}}
.interactive-canvas-wrap{{min-width:0}}
.interactive-canvas-wrap canvas{{display:block;width:100%;height:auto;border-radius:6px;background:#000}}
.interactive-scale-wrap{{display:flex;flex-direction:column;align-items:center;gap:8px;min-width:84px}}
.interactive-scale-title{{font-size:.85em;color:#8b949e;text-align:center}}
.interactive-scale-track{{position:relative;width:22px;height:420px;border:1px solid #30363d;border-radius:999px;overflow:hidden}}
.interactive-scale-gradient{{position:absolute;inset:0}}
.interactive-scale-marker{{position:absolute;left:-5px;width:32px;height:2px;background:#ffffff;box-shadow:0 0 0 1px rgba(0,0,0,.45);transform:translateY(-1px)}}
.interactive-scale-labels{{width:100%;display:flex;flex-direction:column;justify-content:space-between;height:420px;font-size:.8em;color:#8b949e}}
.hidden-asset{{display:none}}
</style>
</head>
<body>
<h1>碘能量分解权重图 / K-edge 权重图重建报告</h1>
<p style="text-align:center;color:#8b949e">基于现有 EFE1 结构、8-bin 假设与材料曲线做启发式重建，不宣称为厂商原始定量图</p>
<p style="text-align:center;color:#58a6ff">当前计算后端：<span class="mono">{backend_name}</span></p>
<p style="text-align:center;color:#f0883e">当前重建模式：<span class="mono">{mode_label}</span></p>

<h2>1. 输出文件</h2>
<table>
<tr><th>文件</th><th>路径</th><th>说明</th></tr>
<tr><td>原始 PixelData 参考</td><td class="mono">{pixel_path}</td><td>用于并排和叠加对比</td></tr>
<tr><td>碘权重图 DICOM</td><td class="mono">{iodine_dcm_path}</td><td>结构保真版本：G3-G2 底图 + 碘频谱调制</td></tr>
<tr><td>K-edge 权重图 DICOM</td><td class="mono">{kedge_dcm_path}</td><td>结构保真版本：G3-G2 底图 + K-edge 差分调制</td></tr>
</table>

<h2>2. 结论摘要</h2>
<div class="ok">
<ul>
<li>当前项目中没有现成的逐像素碘图 DICOM，但已有完整的 8-bin、材料曲线和 PixelData 重建线索。</li>
<li>低分辨率频谱信息仍来自 <span class="mono">G1-G0</span> 的 8 个 band，但最终 DICOM 不再直接使用插值后的 band 图。</li>
<li>当前模式会{coarse_desc}，所以 8-bin 更像“频谱先验”而不是整图材料空间分布。</li>
<li>最终输出仍以“<span class="mono">G3-G2</span> 结构底图 × 频谱调制因子”为主，因此空间结构更接近原始 DCM。</li>
<li>碘图的调制因子来自与碘曲线一致的正交投影，K-edge 图的调制因子来自跨越 33.2 keV 的差分目标。</li>
</ul>
</div>

<div class="warn">
<ul>
<li>这里的权重图是“基于现有信息的重建尝试”，不是厂商协议确认后的定量浓度图。</li>
<li>低分辨率 8-bin 图本身仍然不是最终 DCM 结构，所以它们现在只参与调制，不直接作为导出图像。</li>
<li>Gap 区域仍然没有足够证据支持逐像素权重解释，因此本次没有把 Gap 直接用于图像重建。</li>
</ul>
</div>

<h2>3. 对比视图</h2>
<div class="figure"><img src="data:image/png;base64,{overview_img}" alt="overview"></div>
{interactive_block}
<div class="figure"><img src="data:image/png;base64,{difference_img}" alt="difference"></div>
<div class="note">
差异图按 <code>碘权重 - K-edge 权重</code> 计算：偏红表示碘权重更强，偏蓝表示 K-edge 权重更强，接近白色表示两者接近。
</div>

<h2>4. 权重来源</h2>
<div class="figure"><img src="data:image/png;base64,{weights_img}" alt="weights"></div>
<div class="note">
<ul>
<li>能区边界采用现有报告里的 8-bin 假设：<span class="mono">{", ".join(BIN_LABELS)}</span>。</li>
<li>碘权重来自“碘曲线投影到 Water / Calcium / Gadolinium 的正交补空间”。</li>
<li>K-edge 权重来自对 <span class="mono">Bin1(28-33)</span> 与 <span class="mono">Bin2(33-38)</span> 差分目标的同类正交投影。</li>
<li>这些权重现在优先用于提取平滑的频谱先验，再与完整结构底图融合；仅在 direct 模式下才把 2D band 投影直接当 coarse map。</li>
</ul>
</div>

<h2>5. 8-bin 重建基础</h2>
<div class="figure"><img src="data:image/png;base64,{band_img}" alt="bands"></div>

<h2>6. 统计关系</h2>
<table>
<tr><th>指标</th><th>碘权重图</th><th>K-edge 权重图</th></tr>
<tr><td>与 PixelData 的相关系数</td><td>{iodine_metrics['corr_pixel']:.3f}</td><td>{kedge_metrics['corr_pixel']:.3f}</td></tr>
<tr><td>Top 1% 热点与高 PixelData 区域重叠率</td><td>{iodine_metrics['top1_overlap_hi_pixel']:.3f}</td><td>{kedge_metrics['top1_overlap_hi_pixel']:.3f}</td></tr>
<tr><td>高密度区均值</td><td>{iodine_metrics['high_mean']:.1f}</td><td>{kedge_metrics['high_mean']:.1f}</td></tr>
<tr><td>软组织区均值</td><td>{iodine_metrics['soft_mean']:.1f}</td><td>{kedge_metrics['soft_mean']:.1f}</td></tr>
<tr><td>高密度区 - 软组织区对比</td><td>{iodine_metrics['contrast_hi_soft']:.1f}</td><td>{kedge_metrics['contrast_hi_soft']:.1f}</td></tr>
</table>

<div class="figure"><img src="data:image/png;base64,{scatter_img}" alt="scatter"></div>

<h2>7. 方法说明</h2>
<div class="note">
<ul>
<li>输入数据：原始 DICOM 的 <span class="mono">PixelData</span> 与私有标签 <span class="mono">(EFE1,1001)</span>。</li>
<li>中间结果：先解码 4 组 JPEG2000，再取 <span class="mono">G1-G0</span> 形成 8 个 256x2048 的低分辨率能区图。</li>
<li>低分辨率调制图：<span class="mono">direct_low = sum(w[b] * bin[b])</span>，再按当前模式生成 <span class="mono">coarse_map</span>。</li>
<li>最终导出图：<span class="mono">final_map = normalize(G3-G2) * (0.25 + 0.75 * normalize(coarse_map))</span>。</li>
<li>输出 DICOM：把权重图缩放到 int16 存成派生 Secondary Capture，以便后续查看和归档。</li>
</ul>
</div>

<h2>8. 从数据块到逐像素伪彩权重的重建流程</h2>
<div class="note">
<p>这一节回答的核心问题是：<b>如何从私有数据块里重建出“对应到每个像素位置”的碘 / K-edge 伪彩图</b>。当前实现不是厂商协议级还原，而是基于现有数据结构做的启发式反演，因此流程里会区分“已有证据支持的部分”和“为得到可读权重图所做的重建假设”。</p>
</div>

<table>
<tr><th>阶段</th><th>输入</th><th>输出</th><th>空间尺寸</th><th>说明</th></tr>
<tr><td>1. 解析数据块</td><td><code>(EFE1,1001)</code></td><td>4 组 JPEG2000 tile 流</td><td>每组 64 slots</td><td>通过搜索 <code>jp2c</code> marker 切分出嵌入式 JP2 codestream</td></tr>
<tr><td>2. 解码 tile</td><td>G0/G1/G2/G3</td><td>60 张 tile 图</td><td>每 tile 256x256</td><td>每组 64 个槽位中有 4 个空 tile，因此有效 tile 为 60 张</td></tr>
<tr><td>3. 重组 band</td><td>G1-G0</td><td>8 张 Bin 图</td><td>每 Bin 256x2048</td><td>每个 tile 被切成 8 条 32x256 的 band，按 band 索引跨 tile 拼接</td></tr>
<tr><td>4. 频谱投影</td><td>8-bin + 权重向量</td><td>direct_low</td><td>256x2048</td><td>先按 bin 权重做线性组合，得到直接投影图</td></tr>
<tr><td>5. 频谱先验</td><td>direct_low + PixelData + G3-G2</td><td>coarse map</td><td>256x2048</td><td>默认频谱先验版会把 direct_low 压成更平滑的行/列/全局谱形先验，避免把 tile 拼图直接当全图材料分布</td></tr>
<tr><td>6. 结构融合</td><td>G3-G2 + coarse map</td><td>最终权重图</td><td>2048x2048</td><td>用完整结构底图承载空间结构，再用 coarse map 做频谱调制</td></tr>
<tr><td>7. 伪彩显示</td><td>最终权重图</td><td>HTML 叠加图</td><td>2048x2048</td><td>DICOM 中实际存灰度权重，HTML 用 colormap 映射成伪彩叠加</td></tr>
</table>

<h2>9. 详细流程拆解</h2>
<div class="figure">
<ol>
<li><b>从私有数据块中定位 JPEG2000 codestream</b><br>
原始 DICOM 中除 <code>PixelData</code> 外，还存在私有标签 <code>(EFE1,1001)</code>。本脚本先读取整个 blob，然后在字节流中搜索 <code>jp2c</code> marker。每个 marker 后面的内容被当作一段独立的 JPEG2000 codestream 写出并解码。这一步对应代码中的 <code>find_markers()</code> 与 <code>decode_group()</code>。</li>

<li><b>按组解码为 G0 / G1 / G2 / G3 四组 tile</b><br>
现有结构里把 codestream 分成 4 组，每组 64 个 slot。由于索引 <code>{{0, 7, 56, 63}}</code> 被识别为空 tile，实际每组只解出 60 张 256x256 图。代码里把这四组分别命名为 <code>g0_tiles</code>、<code>g1_tiles</code>、<code>g2_tiles</code>、<code>g3_tiles</code>。</li>

<li><b>构造两个基础域：结构域和频谱域</b><br>
当前实现使用两个不同差分：
<ul>
<li><code>G1 - G0</code>：提供低分辨率频谱信息，用于能区分解。</li>
<li><code>G3 - G2</code>：提供完整 2048x2048 结构底图，用于保持 DCM 解剖结构。</li>
</ul>
也就是说，<code>G1-G0</code> 负责“材料趋势”，<code>G3-G2</code> 负责“空间形态”。</li>

<li><b>为什么 G1-G0 可以拆成 8 个能区图</b><br>
每张 256x256 tile 在重组时不是被当作普通 tile 使用，而是被按行切成 8 个 band：每个 band 大小是 <code>32x256</code>。代码中的 <code>assemble_band_stack()</code> 会把所有 tile 的第 0 个 band 拼成 Bin0，把所有 tile 的第 1 个 band 拼成 Bin1，依此类推，最终得到 8 张 <code>256x2048</code> 的低分辨率 Bin 图。这一步是从“数据块结构”走向“能区图像”的关键桥梁。</li>

<li><b>如何从 8 个 Bin 图得到某个像素位置的低分辨率权重</b><br>
对 low-res 坐标 <code>(r, c)</code>，8 个 Bin 图在该位置形成一个 8 维向量：
<br><code>s(r,c) = [B0(r,c), B1(r,c), ..., B7(r,c)]</code>
<br>然后与预先构造的权重向量做点积：
<br><code>coarse_iodine(r,c) = Σ wI[b] * Bb(r,c)</code>
<br><code>coarse_kedge(r,c) = Σ wK[b] * Bb(r,c)</code>
<br>其中当前脚本中的实际权重为：
<ul>
<li>碘权重向量：<span class="mono">{iodine_weight_text}</span></li>
<li>K-edge 权重向量：<span class="mono">{kedge_weight_text}</span></li>
</ul>
这一步的含义是：对每个 low-res 像素，先看它在 8 个能区上的响应，再用“更像碘”或“更像跨 K-edge 差分”的方向去投影。</li>

<li><b>这些权重向量是怎么来的</b><br>
脚本先从 blob 内的材料曲线区解析出 <code>Water / Iodine / Calcium / Gadolinium</code> 四条 200 点响应曲线，再按 8 个能区边界取平均，形成一个 <code>8 x 4</code> 的材料响应矩阵。随后：
<ul>
<li>碘权重：取 Iodine 曲线形状，并对 Water / Calcium / Gadolinium 及常数项做正交投影剔除，尽量保留“碘特异性”。</li>
<li>K-edge 权重：构造一个在 <code>28-33 keV</code> 为负、在 <code>33-38 keV</code> 为正的差分目标，再做同样的正交投影，强调跨 K-edge 的符号变化。</li>
</ul>
所以这些权重不是直接从 DICOM tag 中读取的逐像素系数，而是从“材料曲线 + 8-bin 假设”反推出的频谱投影方向。</li>

<li><b>为什么低分辨率权重图不能直接作为最终伪彩图</b><br>
因为 <code>coarse_iodine</code> 和 <code>coarse_kedge</code> 的空间尺寸只有 <code>256x2048</code>，并且本质上反映的是 band 级别频谱分解结果。如果直接插值放大到 <code>2048x2048</code>，会出现条带感、拉伸感和伪结构，无法匹配原始 DCM 的真实空间组织关系。这也是之前版本“不像 DCM 结构”的根本原因。</li>

<li><b>如何把 low-res 频谱权重对应回 full-res pixel</b><br>
当前实现采用“结构保真融合”：
<ol>
<li>先把 low-res coarse map 沿行方向做 8 倍插值，得到与原图对齐的 full-res 调制图。</li>
<li>对 <code>G3-G2</code> 结构底图在 body mask 内归一化，得到 <code>base_norm(x,y)</code>。</li>
<li>对上采样后的 coarse map 在同一 body mask 内归一化，得到 <code>coarse_norm(x,y)</code>。</li>
<li>按下面公式逐像素融合：
<br><code>final_map(x,y) = base_norm(x,y) * (1 - α + α * coarse_norm(x,y))</code>
<br>当前 <code>α = 0.75</code>。</li>
</ol>
这一步的意义是：每个最终像素的权重，不是凭空生成，而是由该像素所在位置的结构强度乘上对应位置的频谱调制因子得到。</li>

<li><b>伪彩是怎么得到的</b><br>
DICOM 导出时保存的是单通道灰度权重值，不直接存 RGB 伪彩。HTML 报告中做伪彩有两种方式：
<ul>
<li>单独显示：把权重图映射到截断后的 <code>magma</code>（碘）和 <code>viridis</code>（K-edge）色表，避免高端颜色过白。</li>
<li>叠加显示：先显示灰度 <code>PixelData</code>，再把归一化权重作为 alpha/颜色层覆盖上去。</li>
</ul>
所以“伪彩权重图”本质上是 <b>灰度权重 + 浏览器中的色表映射</b>，而不是 DICOM 原生存储的彩色图。</li>

<li><b>最终每个 pixel 的物理含义是什么</b><br>
当前版本里，某个 full-res pixel 的值表示：
<ul>
<li>它在结构底图中处于什么强度位置；</li>
<li>它所在的频谱区域，在 8-bin 投影后更偏向“碘响应”还是“K-edge 差分响应”；</li>
<li>这个频谱偏向经过 body-mask 归一化后，以调制因子的方式叠加回该像素。</li>
</ul>
因此它更接近“结构保真的材料敏感权重”，而不是严格意义上的定量碘浓度值。</li>
</ol>
</div>

<h2>10. 需要明确的假设与限制</h2>
<div class="warn">
<ul>
<li><b>已有证据支持</b>：<code>(EFE1,1001)</code> 内确实嵌入了可解码的 JPEG2000 数据流；<code>G1-G0</code> 可以按 8-band 重组成能区图；<code>G3-G2</code> 可以提供与原始 DCM 更一致的结构底图。</li>
<li><b>重建假设</b>：8 个 band 对应 8 个能区、材料曲线平均可作为能区权重设计依据、以及 <code>G3-G2 × coarse modulation</code> 是合理的结构保真融合方式。</li>
<li><b>尚未证实</b>：这些步骤不等价于厂商私有协议中的原始物质分解公式，也不能保证对应真实碘浓度或真实 K-edge 定量值。</li>
<li><b>结论</b>：本报告中的伪彩权重图适合做结构一致的材料敏感可视化和对比分析，不应直接当作临床定量图使用。</li>
</ul>
</div>
{interactive_script}
</body>
</html>"""
    out_html = OUT_DIR / "iodine_kedge_report.html"
    out_html.write_text(html, encoding="utf-8")
    return out_html


def generate_single_slice_report(file_path, output_root=None):
    source_path = Path(file_path)
    if not source_path.exists():
        raise FileNotFoundError(f"切片不存在: {source_path}")

    target_root = Path(output_root) if output_root is not None else ROOT / "ui_slice_reports"
    case_name = source_path.parent.name or "case"
    target_out_dir = target_root / case_name / source_path.stem
    target_out_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    prev_out_dir = OUT_DIR
    globals()["OUT_DIR"] = target_out_dir
    try:
        ds = pydicom.dcmread(str(source_path), force=True)
        if (0xEFE1, 0x1001) not in ds:
            raise RuntimeError(f"切片缺少私有标签 (EFE1,1001): {source_path.name}")

        products = compute_slice_products(ds, f"iodine_report_{source_path.stem}")
        iodine_metrics = map_metrics(products["iodine_full"], products["pixel_data"])
        kedge_metrics = map_metrics(products["kedge_full"], products["pixel_data"])

        iodine_i16, iodine_slope = convert_to_int16(products["iodine_full"])
        kedge_i16, kedge_slope = convert_to_int16(products["kedge_full"])
        iodine_dcm = target_out_dir / "iodine_weight_map.dcm"
        kedge_dcm = target_out_dir / "kedge_weight_map.dcm"
        export_derived_dicom(
            ds,
            iodine_i16,
            iodine_dcm,
            "Derived Iodine Weight Map Representative Slice",
            "Single-slice iodine report generated from the selected DICOM slice.",
            iodine_slope,
        )
        export_derived_dicom(
            ds,
            kedge_i16,
            kedge_dcm,
            "Derived K-edge Weight Map Representative Slice",
            "Single-slice K-edge report generated from the selected DICOM slice.",
            kedge_slope,
        )

        overview_img = build_overview_figure(
            products["pixel_data"],
            products["structure_base"],
            products["iodine_full"],
            products["kedge_full"],
        )
        difference_img = build_difference_figure(
            products["pixel_data"], products["iodine_full"], products["kedge_full"]
        )
        weights_img = build_weights_figure(
            products["bin_curve_matrix"], products["iodine_weights"], products["kedge_weights"]
        )
        band_img = build_band_figure(
            products["band_stack"], products["iodine_weights"], products["kedge_weights"]
        )
        scatter_img = build_scatter_figure(
            products["pixel_data"], products["iodine_full"], products["kedge_full"]
        )

        assets_dir = target_out_dir / "native_assets" / "iodine_report"
        overview_png = assets_dir / "overview.png"
        difference_png = assets_dir / "difference.png"
        weights_png = assets_dir / "weights.png"
        bands_png = assets_dir / "bands.png"
        scatter_png = assets_dir / "scatter.png"
        b64_to_png_file(overview_img, overview_png)
        b64_to_png_file(difference_img, difference_png)
        b64_to_png_file(weights_img, weights_png)
        b64_to_png_file(band_img, bands_png)
        b64_to_png_file(scatter_img, scatter_png)

        out_html = write_html(
            pixel_path=str(source_path),
            iodine_dcm_path=str(iodine_dcm),
            kedge_dcm_path=str(kedge_dcm),
            backend_name=CUDA_BACKEND,
            recon_mode=products.get("recon_mode", RECON_MODE),
            overview_img=overview_img,
            difference_img=difference_img,
            weights_img=weights_img,
            band_img=band_img,
            scatter_img=scatter_img,
            iodine_metrics=iodine_metrics,
            kedge_metrics=kedge_metrics,
            iodine_weights=products["iodine_weights"],
            kedge_weights=products["kedge_weights"],
            iodine_interactive=build_iodine_interactive_payload(
                products["pixel_data"], products["iodine_full"], products["kedge_full"]
            ),
        )
        native_json = write_native_report(
            {
                "type": "iodine_kedge_report",
                "title": "碘能量分解权重图 / K-edge 权重图重建报告",
                "subtitle": "基于当前选中的单张 DICOM 切片生成。",
                "source_path": str(source_path),
                "files": [
                    {"name": "原始 PixelData 切片", "path": str(source_path), "description": "当前选中 DICOM"},
                    {"name": "碘权重图 DICOM", "path": str(iodine_dcm), "description": "单切片生成结果"},
                    {"name": "K-edge 权重图 DICOM", "path": str(kedge_dcm), "description": "单切片生成结果"},
                ],
                "figures": [
                    {"title": "对比视图", "path": str(overview_png), "description": "PixelData、结构底图及叠加视图"},
                    {"title": "差异图", "path": str(difference_png), "description": "碘权重减去 K-edge 权重的差异"},
                    {"title": "权重来源", "path": str(weights_png), "description": "材料曲线与权重向量"},
                    {"title": "8-bin 重建基础", "path": str(bands_png), "description": "各能区 band 与贡献"},
                    {"title": "统计关系", "path": str(scatter_png), "description": "派生权重与 PixelData 的关系"},
                ],
                "metrics": {
                    "iodine": iodine_metrics,
                    "kedge": kedge_metrics,
                },
                "weight_vectors": {
                    "iodine": [float(v) for v in products["iodine_weights"]],
                    "kedge": [float(v) for v in products["kedge_weights"]],
                },
            }
        )
        return {
            "source_path": source_path,
            "output_dir": target_out_dir,
            "html_path": out_html,
            "native_json": native_json,
            "iodine_dcm_path": iodine_dcm,
            "kedge_dcm_path": kedge_dcm,
        }
    finally:
        globals()["OUT_DIR"] = prev_out_dir


def main():
    OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    files = ordered_dicom_files(DATA_DIR)
    if not files:
        raise RuntimeError(f"{DATA_DIR} 中没有同时包含 PixelData 与 EFE1 的 DICOM 文件")
    print(f"COMPUTE_BACKEND\t{CUDA_BACKEND}", flush=True)
    print(f"SLICE_WORKERS\t{SLICE_WORKERS}", flush=True)
    print(f"DECODE_THREADS\t{DECODE_THREADS}", flush=True)
    print(f"FAST_VOLUME_SAVE\t{int(FAST_VOLUME_SAVE)}", flush=True)
    print(f"BLOCK_SUPPRESS\t{int(BLOCK_SUPPRESS_ENABLED)}", flush=True)
    print(f"BLOCK_SUPPRESS_STRENGTH\t{BLOCK_SUPPRESS_STRENGTH:.2f}", flush=True)
    print(f"PROJECTION_BLOCK_SUPPRESS_STRENGTH\t{PROJECTION_BLOCK_SUPPRESS_STRENGTH:.2f}", flush=True)
    print(f"RECON_MODE\t{RECON_MODE}", flush=True)

    iodine_series_dir = OUT_DIR / "iodine_series"
    kedge_series_dir = OUT_DIR / "kedge_series"
    iodine_series_dir.mkdir(parents=True, exist_ok=True)
    kedge_series_dir.mkdir(parents=True, exist_ok=True)
    iodine_series_uid = generate_uid()
    kedge_series_uid = generate_uid()

    volume_entries = [None] * len(files)
    rep_idx = len(files) // 2
    rep_products = None
    effective_workers = min(SLICE_WORKERS, len(files))
    completed = 0
    if effective_workers > 1:
        with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="slice-recon") as executor:
            future_map = {
                executor.submit(
                    process_slice_job,
                    idx,
                    file_path,
                    rep_idx,
                    iodine_series_dir,
                    kedge_series_dir,
                    iodine_series_uid,
                    kedge_series_uid,
                ): (idx, file_path)
                for idx, file_path in enumerate(files)
            }
            for future in as_completed(future_map):
                result = future.result()
                volume_entries[result["idx"]] = result["volume_entry"]
                if result["rep_products"] is not None:
                    rep_products = result["rep_products"]
                completed += 1
                print(f"SLICE_PROGRESS\t{completed}\t{len(files)}\t{result['file_path'].name}", flush=True)
    else:
        for idx, file_path in enumerate(files):
            result = process_slice_job(
                idx,
                file_path,
                rep_idx,
                iodine_series_dir,
                kedge_series_dir,
                iodine_series_uid,
                kedge_series_uid,
            )
            volume_entries[result["idx"]] = result["volume_entry"]
            if result["rep_products"] is not None:
                rep_products = result["rep_products"]
            completed += 1
            print(f"SLICE_PROGRESS\t{completed}\t{len(files)}\t{result['file_path'].name}", flush=True)

    if rep_products is None:
        raise RuntimeError("未能生成代表切片结果")
    rep_ds = pydicom.dcmread(str(files[rep_idx]), force=True)
    save_representative_cache(files[rep_idx], rep_products)

    pixel_volume = np.stack([x["pixel"] for x in volume_entries], axis=0)
    iodine_volume = np.stack([x["iodine"] for x in volume_entries], axis=0)
    kedge_volume = np.stack([x["kedge"] for x in volume_entries], axis=0)
    volume_npz = OUT_DIR / "reconstructed_volumes.npz"
    save_volume_file(
        volume_npz,
        pixel=pixel_volume,
        iodine=iodine_volume,
        kedge=kedge_volume,
        source_paths=np.asarray([x["source_path"] for x in volume_entries], dtype=object),
        iodine_paths=np.asarray([x["iodine_path"] for x in volume_entries], dtype=object),
        kedge_paths=np.asarray([x["kedge_path"] for x in volume_entries], dtype=object),
    )

    volume_manifest = {
        "shape": list(pixel_volume.shape),
        "slice_count": int(pixel_volume.shape[0]),
        "representative_index": int(rep_idx),
        "source_paths": [x["source_path"] for x in volume_entries],
        "iodine_paths": [x["iodine_path"] for x in volume_entries],
        "kedge_paths": [x["kedge_path"] for x in volume_entries],
    }
    volume_manifest_path = OUT_DIR / "volume_manifest.json"
    volume_manifest_path.write_text(json.dumps(volume_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    iodine_metrics = map_metrics(rep_products["iodine_full"], rep_products["pixel_data"])
    kedge_metrics = map_metrics(rep_products["kedge_full"], rep_products["pixel_data"])
    vol_mask = pixel_volume > np.percentile(pixel_volume, 50)
    volume_metrics = {
        "iodine_mean_body": float(np.mean(iodine_volume[vol_mask])) if np.count_nonzero(vol_mask) else 0.0,
        "kedge_mean_body": float(np.mean(kedge_volume[vol_mask])) if np.count_nonzero(vol_mask) else 0.0,
        "iodine_max": float(np.max(iodine_volume)),
        "kedge_max": float(np.max(kedge_volume)),
    }

    rep_iodine_i16, rep_iodine_slope = convert_to_int16(rep_products["iodine_full"])
    rep_kedge_i16, rep_kedge_slope = convert_to_int16(rep_products["kedge_full"])
    iodine_dcm = OUT_DIR / "iodine_weight_map.dcm"
    kedge_dcm = OUT_DIR / "kedge_weight_map.dcm"
    export_derived_dicom(
        rep_ds,
        rep_iodine_i16,
        iodine_dcm,
        "Derived Iodine Weight Map Representative Slice",
        "Representative slice copied from the full iodine series volume.",
        rep_iodine_slope,
    )
    export_derived_dicom(
        rep_ds,
        rep_kedge_i16,
        kedge_dcm,
        "Derived K-edge Weight Map Representative Slice",
        "Representative slice copied from the full K-edge series volume.",
        rep_kedge_slope,
    )

    overview_img = build_overview_figure(
        rep_products["pixel_data"],
        rep_products["structure_base"],
        rep_products["iodine_full"],
        rep_products["kedge_full"],
    )
    difference_img = build_difference_figure(
        rep_products["pixel_data"], rep_products["iodine_full"], rep_products["kedge_full"]
    )
    weights_img = build_weights_figure(
        rep_products["bin_curve_matrix"], rep_products["iodine_weights"], rep_products["kedge_weights"]
    )
    band_img = build_band_figure(
        rep_products["band_stack"], rep_products["iodine_weights"], rep_products["kedge_weights"]
    )
    scatter_img = build_scatter_figure(rep_products["pixel_data"], rep_products["iodine_full"], rep_products["kedge_full"])
    volume_img = build_volume_mpr_figure(pixel_volume, iodine_volume, kedge_volume)

    assets_dir = OUT_DIR / "native_assets" / "iodine_report"
    overview_png = assets_dir / "overview.png"
    difference_png = assets_dir / "difference.png"
    weights_png = assets_dir / "weights.png"
    bands_png = assets_dir / "bands.png"
    scatter_png = assets_dir / "scatter.png"
    volume_png = assets_dir / "volume_overview.png"
    b64_to_png_file(overview_img, overview_png)
    b64_to_png_file(difference_img, difference_png)
    b64_to_png_file(weights_img, weights_png)
    b64_to_png_file(band_img, bands_png)
    b64_to_png_file(scatter_img, scatter_png)
    b64_to_png_file(volume_img, volume_png)

    out_html = write_html(
        pixel_path=str(files[rep_idx]),
        iodine_dcm_path=str(iodine_dcm),
        kedge_dcm_path=str(kedge_dcm),
        backend_name=CUDA_BACKEND,
        recon_mode=rep_products.get("recon_mode", RECON_MODE),
        overview_img=overview_img,
        difference_img=difference_img,
        weights_img=weights_img,
        band_img=band_img,
        scatter_img=scatter_img,
        iodine_metrics=iodine_metrics,
        kedge_metrics=kedge_metrics,
        iodine_weights=rep_products["iodine_weights"],
        kedge_weights=rep_products["kedge_weights"],
        iodine_interactive=build_iodine_interactive_payload(
            rep_products["pixel_data"], rep_products["iodine_full"], rep_products["kedge_full"]
        ),
    )
    native_json = write_native_report(
        {
            "type": "iodine_kedge_report",
            "title": "碘能量分解权重图 / K-edge 权重图重建报告",
            "subtitle": "基于现有 EFE1 结构、8-bin 假设与材料曲线，对整个路径下的全部 DICOM 切片一次性重建 3D 数据集。",
            "source_path": str(files[rep_idx]),
            "files": [
                {"name": "原始 PixelData 代表切片", "path": str(files[rep_idx]), "description": "用于并排和叠加对比"},
                {"name": "碘权重图代表 DICOM", "path": str(iodine_dcm), "description": "全序列重建后的代表层"},
                {"name": "K-edge 权重图代表 DICOM", "path": str(kedge_dcm), "description": "全序列重建后的代表层"},
                {"name": "碘权重图序列目录", "path": str(iodine_series_dir), "description": f"共 {len(volume_entries)} 张 DICOM"},
                {"name": "K-edge 权重图序列目录", "path": str(kedge_series_dir), "description": f"共 {len(volume_entries)} 张 DICOM"},
                {"name": "3D 体数据", "path": str(volume_npz), "description": "原始/碘/K-edge 三套体数据"},
            ],
            "highlights": [
                f"当前会对路径下全部 {len(volume_entries)} 张 DICOM 一次性逐张重建，并堆成 3D 体数据。",
                f"当前计算后端为 {CUDA_BACKEND}，会在可用时自动走 CUDA，否则回退到 NumPy。",
                f"当前重建模式为 {rep_products.get('recon_mode', RECON_MODE)}，默认把 8-bin 视为频谱先验来源，而不是直接当整图低分辨率材料图。",
                "低分辨率频谱信息来自 G1-G0 的 8 个 band，但最终 DICOM 不再直接使用 band 拼图本身。",
                "最终输出改为 G3-G2 结构底图乘以频谱调制因子，所以空间结构更接近原始 DCM。",
                "UI 现在可直接读取原始 / 碘 / K-edge 三套体数据做 MPR 层切换。",
            ],
            "warnings": [
                "这里的权重图是基于现有信息的重建尝试，不是厂商协议确认后的定量浓度图。",
                "低分辨率 8-bin 图本身不是最终 DCM 结构，所以它们只参与调制，不直接作为导出图像。",
                "Gap 区域仍然没有足够证据支持逐像素权重解释，因此本次没有把 Gap 直接用于图像重建。",
            ],
            "figures": [
                {"title": "对比视图", "path": str(overview_png), "description": "PixelData、结构底图、碘图、K-edge 图及叠加视图"},
                {"title": "差异图", "path": str(difference_png), "description": "碘权重减去 K-edge 权重的有符号差异图"},
                {"title": "3D 数据集概览", "path": str(volume_png), "description": "原始 / 碘 / K-edge 三套体数据的三正交面概览"},
                {"title": "权重来源", "path": str(weights_png), "description": "材料响应曲线与 8-bin 权重向量"},
                {"title": "8-bin 重建基础", "path": str(bands_png), "description": "各个能区 band 及其对重建的贡献"},
                {"title": "统计关系", "path": str(scatter_png), "description": "派生权重图与 PixelData 的关系"},
            ],
            "metrics": {
                "iodine": iodine_metrics,
                "kedge": kedge_metrics,
            },
            "weight_vectors": {
                "iodine": [float(v) for v in rep_products["iodine_weights"]],
                "kedge": [float(v) for v in rep_products["kedge_weights"]],
            },
            "volume_info": {
                "npz_path": str(volume_npz),
                "manifest_path": str(volume_manifest_path),
                "shape": list(pixel_volume.shape),
                "slice_count": int(pixel_volume.shape[0]),
                "representative_index": int(rep_idx),
                "metrics": volume_metrics,
            },
            "runtime": {
                "cuda_enabled": bool(CUDA_ENABLED),
                "cuda_disabled_by_env": bool(CUDA_DISABLED_BY_ENV),
                "backend": CUDA_BACKEND,
                "recon_mode": rep_products.get("recon_mode", RECON_MODE),
                "slice_workers": int(effective_workers),
                "decode_threads": int(DECODE_THREADS),
                "fast_volume_save": bool(FAST_VOLUME_SAVE),
                "block_suppress_enabled": bool(BLOCK_SUPPRESS_ENABLED),
                "block_suppress_strength": float(BLOCK_SUPPRESS_STRENGTH),
                "projection_block_suppress_strength": float(PROJECTION_BLOCK_SUPPRESS_STRENGTH),
            },
            "pipeline_table": [
                {"stage": "解析数据块", "input": "(EFE1,1001)", "output": "4 组 JPEG2000 tile 流", "shape": "每组 64 slots", "description": "通过搜索 jp2c marker 切分嵌入式 JP2 codestream"},
                {"stage": "解码 tile", "input": "G0/G1/G2/G3", "output": "60 张 tile 图", "shape": "每 tile 256x256", "description": "每组 64 个槽位中有 4 个空 tile，因此有效 tile 为 60 张"},
                {"stage": "重组 band", "input": "G1-G0", "output": "8 张 Bin 图", "shape": "每 Bin 256x2048", "description": "每个 tile 被切成 8 条 32x256 的 band，按 band 索引跨 tile 拼接"},
                {"stage": "频谱投影", "input": "8-bin + 权重向量", "output": "direct_low", "shape": "256x2048", "description": "先按 bin 权重做线性组合，得到直接投影图"},
                {"stage": "频谱先验", "input": "direct_low + PixelData + G3-G2", "output": "coarse map", "shape": "256x2048", "description": "默认 spectral_prior 模式会把 direct_low 压成平滑的行/列/全局谱形先验，再交给结构底图承载空间细节"},
                {"stage": "结构融合", "input": "G3-G2 + coarse map", "output": "最终权重图", "shape": "2048x2048", "description": "用完整结构底图承载空间结构，再用 coarse map 做频谱调制"},
                {"stage": "3D 堆叠", "input": "所有切片的最终权重图", "output": "3D 体数据", "shape": f"{pixel_volume.shape[0]}x{pixel_volume.shape[1]}x{pixel_volume.shape[2]}", "description": "把路径下所有可读 DICOM 的派生结果堆成原始 / 碘 / K-edge 三套体数据"},
                {"stage": "伪彩显示", "input": "最终权重图", "output": "Qt 原生叠加图", "shape": "2048x2048", "description": "DICOM 中实际存灰度权重，UI 中再映射成伪彩"},
            ],
            "detail_steps": [
                f"先收集路径下全部 {len(volume_entries)} 张同时包含 PixelData 与 EFE1 的 DICOM，并按空间位置或实例号排序。",
                "从私有数据块 (EFE1,1001) 中搜索 jp2c marker，切出多段 JPEG2000 codestream。",
                "按组解码为 G0 / G1 / G2 / G3 四组 tile，其中空 tile 索引为 0、7、56、63。",
                "构造两个基础域：G1-G0 作为频谱域，G3-G2 作为结构域。",
                "把 G1-G0 的每张 tile 按行切成 8 个 32x256 band，并跨 tile 拼成 8 张 256x2048 的 Bin 图。",
                "对每个 low-res 坐标 (r,c)，取 8 个 Bin 值组成 8 维向量，再与碘 / K-edge 权重向量做点积，得到 coarse map。",
                "碘权重来自 Iodine 曲线对 Water / Calcium / Gadolinium 及常数项做正交投影后的结果。",
                "K-edge 权重来自 Bin1(28-33) 为负、Bin2(33-38) 为正的差分目标做同类正交投影后的结果。",
                "将 coarse map 上采样回 full-res，并与 G3-G2 结构底图按 final_map = base_norm * (1 - α + α * coarse_norm) 融合，当前 α=0.75。",
                "对每张切片都独立输出派生 DICOM，再按切片顺序堆成 3D 体数据并保存为 npz，供 UI 里的 MPR 层切换使用。",
                "最终导出 DICOM 保存的是单通道灰度权重值；伪彩只在报告查看阶段通过 magma / viridis 色表叠加显示。",
            ],
            "assumptions": [
                "已有证据支持：EFE1 内确实嵌入了可解码的 JPEG2000 数据流；G1-G0 可按 8-band 重组成能区图；G3-G2 可提供与原始 DCM 更一致的结构底图。",
                "重建假设：8 个 band 对应 8 个能区、材料曲线平均可作为能区权重设计依据、以及 G3-G2 乘 coarse modulation 是合理的结构保真融合方式。",
                "尚未证实：这些步骤不等价于厂商私有协议中的原始物质分解公式，也不能保证对应真实碘浓度或真实 K-edge 定量值。",
            ],
        }
    )

    print(f"Saved DICOM: {iodine_dcm}")
    print(f"Saved DICOM: {kedge_dcm}")
    print(f"Saved DICOM series: {iodine_series_dir}")
    print(f"Saved DICOM series: {kedge_series_dir}")
    print(f"Saved Volume NPZ: {volume_npz}")
    print(f"Saved HTML: {out_html}")
    print(f"Saved Native JSON: {native_json}")


if __name__ == "__main__":
    main()
