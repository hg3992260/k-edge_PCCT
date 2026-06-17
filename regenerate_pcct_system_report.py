import base64
import io
import json
import os
import struct
from pathlib import Path

import glymur
import matplotlib
import numpy as np
import pydicom
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from skimage.filters import threshold_otsu

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import minimal_iodine_weight_model as minimal_iodine_model
import verify_final_response_candidates as final_response_analysis
import verify_iodine_hotspot_continuity as hotspot_continuity_analysis
import verify_iodine_spatial_correspondence as iodine_spatial_analysis
import verify_tile_basis_analysis as basis_analysis
import verify_anatomy_regions as region_analysis

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(r"d:\xl\00020006")
DATA_DIR = ROOT / "00020006"
OUT_HTML = ROOT / "pcct_systematic_report.html"
CACHE_DIR = ROOT / "_pcct_cache"

EMPTY_TILES = {0, 7, 56, 63}
SLICE_SAMPLES = [0, 600, 1200, 1800, 2400, 2634]


def fig_to_b64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return encoded


def file_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def read_slice(file_name):
    return pydicom.dcmread(str(DATA_DIR / file_name))


def find_markers(blob):
    markers = []
    pos = -1
    token = b"jp2c"
    while True:
        pos = blob.find(token, pos + 1)
        if pos == -1:
            break
        markers.append(pos)
    return markers


def decode_group(blob, markers, group_id, prefix):
    CACHE_DIR.mkdir(exist_ok=True)
    tiles = np.zeros((60, 256, 256), dtype=np.float64)
    out_idx = 0
    start_slot = group_id * 64
    for slot in range(start_slot, start_slot + 64):
        rel = slot - start_slot
        if rel in EMPTY_TILES:
            continue
        start = markers[slot] + 4
        end = markers[slot + 1] if slot + 1 < len(markers) else len(blob)
        jp2_path = CACHE_DIR / f"{prefix}_g{group_id}_{slot}.jp2k"
        with open(jp2_path, "wb") as f:
            f.write(blob[start:end])
        tiles[out_idx] = glymur.Jp2k(str(jp2_path))[:].astype(np.float64)
        out_idx += 1
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
        for strip_idx in range(8):
            row0 = (tile_row * 8 + strip_idx) * 32
            row1 = row0 + 32
            col0 = tile_col * 256
            col1 = col0 + 256
            full[row0:row1, col0:col1] = tile[strip_idx * 32 : (strip_idx + 1) * 32, :]
    return full


def corr_middle(a, b):
    mask = (
        (a != 0)
        & (b != 0)
        & (b > np.percentile(b, 5))
        & (b < np.percentile(b, 95))
    )
    if mask.sum() < 100:
        return np.nan
    return float(np.corrcoef(a[mask].ravel(), b[mask].ravel())[0, 1])


def parse_material_headers(blob):
    offset = 4
    materials = []
    for _ in range(4):
        end = blob.find(b"\x00", offset)
        name = blob[offset:end].decode("ascii")
        offset = end + 1
        if offset % 4:
            offset += 4 - offset % 4
        params_u32 = [struct.unpack_from("<I", blob, offset + i * 4)[0] for i in range(14)]
        params_f32 = [struct.unpack_from("<f", blob, offset + i * 4)[0] for i in range(14)]
        materials.append(
            {"name": name, "u32": params_u32, "f32": params_f32, "offset": offset}
        )
        offset += 56
    return materials


def parse_curves(blob):
    return {
        "Water": np.frombuffer(blob[0x1C0 : 0x1C0 + 200 * 4], dtype=np.float32),
        "Iodine": np.frombuffer(blob[0x850 : 0x850 + 200 * 4], dtype=np.float32),
        "Calcium": np.frombuffer(blob[0xB70 : 0xB70 + 200 * 4], dtype=np.float32),
        "Gadolinium": np.frombuffer(blob[0xED4 : 0xED4 + 200 * 4], dtype=np.float32),
    }


def format_num(x):
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):,.3f}"


def build_overview_figure(pixel_data, g0, g1, g2, g3, diff):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    items = [
        ("PixelData", pixel_data, "gray", np.percentile(pixel_data, 1), np.percentile(pixel_data, 99)),
        ("G0", g0, "gray", np.percentile(g0, 1), np.percentile(g0, 99)),
        ("G1", g1, "gray", np.percentile(g1, 1), np.percentile(g1, 99)),
        ("G2", g2, "gray", np.percentile(g2, 1), np.percentile(g2, 99)),
        ("G3", g3, "gray", np.percentile(g3, 1), np.percentile(g3, 99)),
        ("G1 - G0", diff, "gray", np.percentile(diff, 1), np.percentile(diff, 99)),
    ]
    for ax, (title, img, cmap, vmin, vmax) in zip(axes.ravel(), items):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\n[{img.min():.0f}, {img.max():.0f}]", fontsize=10)
        ax.axis("off")
    fig.suptitle("Slice 0 关键图像总览", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_tile_figure(diff_tiles):
    tile = diff_tiles[34]
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(2, 5, 1)
    vmax = np.percentile(np.abs(tile), 99.5)
    ax.imshow(tile, cmap="gray", vmin=-vmax, vmax=vmax)
    for i in range(1, 8):
        ax.axhline(i * 32, color="#ff7b72", linewidth=0.5, linestyle="--")
    ax.set_title("单个 256x256 tile\n8 个 32x256 strip", fontsize=10)
    ax.axis("off")
    for strip_idx in range(8):
        ax = fig.add_subplot(2, 5, strip_idx + 2)
        strip = tile[strip_idx * 32 : (strip_idx + 1) * 32, :]
        vmax = np.percentile(np.abs(strip), 99.5)
        if vmax == 0:
            vmax = 1
        ax.imshow(strip, cmap="gray", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"Strip {strip_idx}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Tile 内部结构：更像空间交错行带，而非 8 张完整子图", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_curve_figure(curves, curve8, curve8_corr):
    kev = np.arange(200)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {
        "Water": "#58a6ff",
        "Iodine": "#e53935",
        "Calcium": "#7ee787",
        "Gadolinium": "#f0883e",
    }
    for name, curve in curves.items():
        axes[0].plot(kev, curve, linewidth=1.0, label=name, color=colors[name])
        axes[1].semilogy(kev, np.maximum(curve, 1e-6), linewidth=1.0, color=colors[name])
        axes[2].plot(
            kev,
            curve / max(curve.max(), 1e-6),
            linewidth=1.0,
            label=name,
            color=colors[name],
        )
    for ax in axes:
        ax.axvline(33, color="#e53935", linestyle="--", linewidth=1, alpha=0.7)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 200)
        ax.set_xlabel("Index")
    axes[0].set_title("4 条主光谱曲线")
    axes[1].set_title("对数显示")
    axes[2].plot(
        kev,
        curve8 / max(curve8.max(), 1e-6),
        linewidth=1.2,
        color="#d2a8ff",
        linestyle=":",
        label=f"Curve #8 (r={curve8_corr:.3f})",
    )
    axes[2].set_title("归一化形状对比")
    axes[2].legend(fontsize=8)
    fig.suptitle("光谱曲线：碘曲线的 33 keV K-edge 可直接确认", fontsize=14, y=1.03)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_gap_figure(gap_s16):
    bins = [gap_s16[i * 4167 : (i + 1) * 4167] for i in range(4)]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes[0, 0].plot(gap_s16[::5], linewidth=0.3, color="#58a6ff")
    axes[0, 0].set_title(f"Gap 信号采样图\n{len(gap_s16):,} 个 int16")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 1].hist(gap_s16, bins=80, color="#f0883e", alpha=0.8)
    axes[0, 1].set_title("Gap 直方图")
    axes[0, 1].grid(alpha=0.3)
    corr23 = float(np.corrcoef(bins[2].astype(float), bins[3].astype(float))[0, 1])
    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.05,
        0.95,
        (
            "Gap 区域摘要\n\n"
            f"长度: {len(gap_s16):,} int16\n"
            f"字节数: {len(gap_s16) * 2:,}\n"
            "4 等分: 4,167 / bin\n"
            f"B2-B3 相关: {corr23:.3f}\n"
            f"均值: {gap_s16.mean():.1f}\n"
            f"标准差: {gap_s16.std():.1f}"
        ),
        transform=axes[0, 2].transAxes,
        va="top",
        fontsize=10,
        bbox=dict(facecolor="#161b22", edgecolor="#30363d"),
    )
    axes[0, 3].scatter(bins[2][::8], bins[3][::8], s=6, alpha=0.25, color="#7ee787")
    axes[0, 3].set_title("B2 vs B3 散点图")
    axes[0, 3].set_xlabel("B2")
    axes[0, 3].set_ylabel("B3")
    axes[0, 3].grid(alpha=0.3)
    for i, b in enumerate(bins):
        ax = axes[1, i]
        ax.plot(b[:300], linewidth=0.4)
        ax.set_title(f"Bin {i}\nmean={b.mean():.1f}, std={b.std():.1f}", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("未解码 Gap 区域：可统计，但仍无明确物理语义", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_consistency_figure(sample_rows):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    x = np.arange(len(sample_rows))
    labels = [str(row["slice"]) for row in sample_rows]
    gap_corrs = [row["gap_corr"] for row in sample_rows]
    header_same = [1 if row["header_same"] else 0 for row in sample_rows]
    axes[0].bar(x, gap_corrs, color="#7ee787")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Gap 区域跨切片相关性")
    axes[0].set_xlabel("Slice index")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x, header_same, color="#58a6ff")
    axes[1].set_ylim(0, 1.2)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["不同", "相同"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("0x000-0x27F Header 是否完全一致")
    axes[1].set_xlabel("Slice index")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("跨切片稳定性：校准区稳定，图像码流变化", fontsize=14, y=1.04)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_rowband_figure(diff):
    row_abs = np.mean(np.abs(diff), axis=1)
    row_mean = diff.mean(axis=1)
    x = np.arange(len(row_abs))

    def phase_profile(arr, period):
        vals = []
        for off in range(period):
            idx = np.arange(off, len(arr), period)
            vals.append(arr[idx].mean())
        return np.array(vals, dtype=np.float64)

    fft = np.fft.rfft(row_abs - row_abs.mean())
    power = np.abs(fft) ** 2
    top = np.argsort(power[1:])[-5:] + 1
    top = top[np.argsort(power[top])[::-1]]
    peak_lines = [
        f"k={int(k)} => period={len(row_abs)/k:.1f} rows, energy={power[k]/max(power[1:].sum(), 1e-12):.3f}"
        for k in top
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes[0, 0].plot(x, row_abs, linewidth=0.8, color="#58a6ff")
    axes[0, 0].set_title("逐行绝对能量曲线 | mean(abs(G1-G0), axis=1)")
    axes[0, 0].set_xlabel("Row")
    axes[0, 0].set_ylabel("Energy")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(x, row_mean, linewidth=0.8, color="#f0883e")
    axes[0, 1].set_title("逐行均值曲线 | mean(G1-G0, axis=1)")
    axes[0, 1].set_xlabel("Row")
    axes[0, 1].set_ylabel("Mean")
    axes[0, 1].grid(alpha=0.3)

    p8 = phase_profile(row_abs, 8)
    p32 = phase_profile(row_abs, 32)
    p256 = phase_profile(row_abs, 256)
    axes[1, 0].plot(np.arange(8), p8, marker="o", label="period 8")
    axes[1, 0].plot(np.arange(32), p32, marker=".", label="period 32")
    axes[1, 0].plot(np.arange(256), p256, linewidth=0.8, label="period 256")
    axes[1, 0].set_title("相位平均对比：8 / 32 / 256 行周期")
    axes[1, 0].set_xlabel("Phase Offset")
    axes[1, 0].set_ylabel("Mean Energy")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.05,
        0.95,
        (
            "逐行频谱摘要\n\n"
            + "\n".join(peak_lines)
            + "\n\n结论:\n"
            "8/32 行周期只带来很弱的相位起伏，\n"
            "主导变化来自数百到上千行尺度的低频分量。"
        ),
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=10,
        bbox=dict(facecolor="#161b22", edgecolor="#30363d"),
    )

    fig.suptitle("总览带状观感的逐行检验：低频纹理强于固定 strip 周期", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_anatomy_relation_figure(diff, pixel_data):
    thr = threshold_otsu(pixel_data)
    body = pixel_data > thr
    body_vals = pixel_data[body]
    hi_thr = np.percentile(body_vals, 99.0)

    row_tex = gaussian_filter1d(np.mean(np.abs(diff), axis=1), sigma=32)
    row_body_width = gaussian_filter1d(body.sum(axis=1).astype(float), sigma=32)
    row_body_mean = gaussian_filter1d(
        np.where(body, pixel_data, np.nanmean(pixel_data)).sum(axis=1) / np.maximum(body.sum(axis=1), 1),
        sigma=32,
    )
    row_hi = gaussian_filter1d((pixel_data > hi_thr).sum(axis=1).astype(float), sigma=32)
    row_grad = gaussian_filter1d(np.mean(np.abs(np.gradient(pixel_data, axis=1)), axis=1), sigma=32)

    def norm(x):
        x = x.astype(np.float64)
        return (x - np.mean(x)) / max(np.std(x), 1e-9)

    corr_width = float(np.corrcoef(row_tex, row_body_width)[0, 1])
    corr_mean = float(np.corrcoef(row_tex, row_body_mean)[0, 1])
    corr_hi = float(np.corrcoef(row_tex, row_hi)[0, 1])
    corr_grad = float(np.corrcoef(row_tex, row_grad)[0, 1])

    low_diff = gaussian_filter(diff, sigma=48)
    low_pix = gaussian_filter(pixel_data, sigma=48)
    mask = (
        body
        & (pixel_data > np.percentile(pixel_data[body], 5))
        & (pixel_data < np.percentile(pixel_data[body], 95))
    )
    corr_2d = float(np.corrcoef(low_diff[mask].ravel(), low_pix[mask].ravel())[0, 1])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[0, 0].imshow(low_diff, cmap="gray")
    axes[0, 0].set_title("Low-pass(G1-G0), sigma=48")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(low_pix, cmap="gray")
    axes[0, 1].set_title("Low-pass(PixelData), sigma=48")
    axes[0, 1].axis("off")

    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.05,
        0.95,
        (
            "低频纹理 vs 解剖结构\n\n"
            f"2D low-pass corr = {corr_2d:.3f}\n"
            f"row vs body width = {corr_width:.3f}\n"
            f"row vs body mean = {corr_mean:.3f}\n"
            f"row vs high-density load = {corr_hi:.3f}\n"
            f"row vs edge strength = {corr_grad:.3f}\n\n"
            "判读:\n"
            "低频纹理与广义解剖结构显著相关，\n"
            "更像体轮廓/高衰减区驱动的低频调制。"
        ),
        transform=axes[0, 2].transAxes,
        va="top",
        fontsize=10,
        bbox=dict(facecolor="#161b22", edgecolor="#30363d"),
    )

    x = np.arange(len(row_tex))
    axes[1, 0].plot(x, norm(row_tex), label="low-freq texture", linewidth=1.0)
    axes[1, 0].plot(x, norm(row_body_width), label="body width", linewidth=0.9)
    axes[1, 0].set_title("逐行: 低频纹理 vs 体轮廓宽度")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(x, norm(row_tex), label="low-freq texture", linewidth=1.0)
    axes[1, 1].plot(x, norm(row_grad), label="edge strength", linewidth=0.9)
    axes[1, 1].set_title("逐行: 低频纹理 vs 解剖边缘强度")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].plot(x, norm(row_tex), label="low-freq texture", linewidth=1.0)
    axes[1, 2].plot(x, norm(row_hi), label="high-density load", linewidth=0.9)
    axes[1, 2].set_title("逐行: 低频纹理 vs 高衰减结构负荷")
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("低频纹理与解剖结构相关性检验", fontsize=14, y=0.98)
    fig.tight_layout()
    return fig_to_b64(fig), {
        "corr_2d": corr_2d,
        "corr_width": corr_width,
        "corr_mean": corr_mean,
        "corr_hi": corr_hi,
        "corr_grad": corr_grad,
    }


def main():
    files = sorted(os.listdir(DATA_DIR))
    ds0 = read_slice(files[0])
    blob = ds0[0xEFE1, 0x1001].value
    pixel_data = np.frombuffer(ds0.PixelData, dtype=np.uint16).reshape(ds0.Rows, ds0.Columns).astype(np.float64)
    markers = find_markers(blob)
    materials = parse_material_headers(blob)
    curves = parse_curves(blob)
    curve8 = np.frombuffer(blob[0x1878 : 0x1878 + 200 * 4], dtype=np.float32)
    gap_s16 = np.frombuffer(blob[0x1BA8:0x9DE0], dtype=np.uint16).astype(np.int16)

    g0_tiles = decode_group(blob, markers, 0, "slice0")
    g1_tiles = decode_group(blob, markers, 1, "slice0")
    g2_tiles = decode_group(blob, markers, 2, "slice0")
    g3_tiles = decode_group(blob, markers, 3, "slice0")

    g0 = assemble_interleaved(g0_tiles)
    g1 = assemble_interleaved(g1_tiles)
    g2 = assemble_interleaved(g2_tiles)
    g3 = assemble_interleaved(g3_tiles)
    diff = g1 - g0

    diff_corr = corr_middle(diff, pixel_data)
    mask = (
        (diff != 0)
        & (pixel_data != 0)
        & (pixel_data > np.percentile(pixel_data, 5))
        & (pixel_data < np.percentile(pixel_data, 95))
    )
    scale = pixel_data[mask].std() / max(diff[mask].std(), 1e-9)
    offset = pixel_data[mask].mean() - scale * diff[mask].mean()
    diff_fit = diff * scale + offset
    fit_residual = diff_fit[mask] - pixel_data[mask]

    iodine = curves["Iodine"]
    curve8_corr = float(
        np.corrcoef(
            curve8 / max(curve8.max(), 1e-6),
            iodine / max(iodine.max(), 1e-6),
        )[0, 1]
    )
    gap_bins = [gap_s16[i * 4167 : (i + 1) * 4167] for i in range(4)]
    gap_corr23 = float(np.corrcoef(gap_bins[2].astype(float), gap_bins[3].astype(float))[0, 1])

    sample_rows = []
    for idx in SLICE_SAMPLES:
        ds = read_slice(files[idx])
        blob_i = ds[0xEFE1, 0x1001].value
        gap_i = np.frombuffer(blob_i[0x1BA8:0x9DE0], dtype=np.uint16).astype(np.int16)
        row = {
            "slice": idx,
            "header_same": bool(blob_i[:0x280] == blob[:0x280]),
            "curves_same": bool(blob_i[0x1C0:0x1878] == blob[0x1C0:0x1878]),
            "gap_corr": float(np.corrcoef(gap_s16.astype(float), gap_i.astype(float))[0, 1]),
            "blob_len": len(blob_i),
        }
        sample_rows.append(row)

    overview_img = build_overview_figure(pixel_data, g0, g1, g2, g3, diff)
    tile_img = build_tile_figure(g1_tiles - g0_tiles)
    curve_img = build_curve_figure(curves, curve8, curve8_corr)
    gap_img = build_gap_figure(gap_s16)
    consistency_img = build_consistency_figure(sample_rows)
    rowband_img = build_rowband_figure(diff)
    anatomy_img, anatomy_metrics = build_anatomy_relation_figure(diff, pixel_data)
    basis_payload = basis_analysis.run_analysis(save_outputs=True)
    basis_img = file_to_b64(basis_analysis.OUT_PNG)
    region_payload = region_analysis.run_analysis(save_outputs=True)
    region_img = file_to_b64(region_analysis.OUT_PNG)
    final_payload = final_response_analysis.run_analysis(save_outputs=True)
    final_img = file_to_b64(final_response_analysis.OUT_PNG)
    minimal_iodine_payload = minimal_iodine_model.run_analysis(save_outputs=True)
    minimal_iodine_img = file_to_b64(minimal_iodine_model.OUT_PNG)
    iodine_spatial_payload = iodine_spatial_analysis.run_analysis(save_outputs=True)
    iodine_spatial_img = file_to_b64(iodine_spatial_analysis.OUT_PNG)
    hotspot_continuity_payload = hotspot_continuity_analysis.run_analysis(save_outputs=True)
    hotspot_continuity_img = file_to_b64(hotspot_continuity_analysis.OUT_PNG)
    sine_payload = json.loads((ROOT / "verify_tile_sine_hypothesis.json").read_text(encoding="utf-8"))

    plausible_float_rows = []
    for material in materials:
        for idx, value in enumerate(material["f32"]):
            if np.isfinite(value) and 0.01 < abs(value) < 5000 and abs(value - round(value)) > 1e-3:
                plausible_float_rows.append(
                    (material["name"], idx, material["u32"][idx], value)
                )

    slice_rows_html = "\n".join(
        (
            f"<tr><td>{row['slice']}</td>"
            f"<td class='{'good' if row['header_same'] else 'bad'}'>{'相同' if row['header_same'] else '不同'}</td>"
            f"<td class='{'good' if row['curves_same'] else 'bad'}'>{'相同' if row['curves_same'] else '不同'}</td>"
            f"<td class='good'>{row['gap_corr']:.6f}</td>"
            f"<td>{row['blob_len']:,}</td></tr>"
        )
        for row in sample_rows
    )

    float_rows_html = "\n".join(
        (
            f"<tr><td>{name}</td><td>{idx}</td><td>{u32}</td><td>{value:.6f}</td></tr>"
            for name, idx, u32, value in plausible_float_rows
        )
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>P10 光子计数 CT 系统复核报告</title>
<style>
body{{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#0d1117;color:#c9d1d9;max-width:1280px;margin:0 auto;padding:20px;line-height:1.7}}
h1{{color:#58a6ff;text-align:center;border-bottom:2px solid #30363d;padding-bottom:12px}}
h2{{color:#f0883e;border-left:4px solid #f0883e;padding-left:12px;margin-top:28px}}
h3{{color:#d2a8ff;margin-top:18px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:.9em}}
th{{background:#21262d;color:#8b949e;padding:8px 12px;border:1px solid #30363d;text-align:left}}
td{{padding:7px 12px;border:1px solid #30363d;vertical-align:top}}
tr:nth-child(even) td{{background:#161b22}}
.figure{{margin:18px 0;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}}
.figure img{{width:100%;border-radius:6px}}
.good{{color:#7ee787;font-weight:bold}}
.warn{{color:#d2991d;font-weight:bold}}
.bad{{color:#ff7b72;font-weight:bold}}
.mono{{font-family:Consolas,monospace;color:#a5d6ff}}
.panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin:14px 0}}
.note{{background:#1f242b;border-left:3px solid #58a6ff;padding:12px 16px;margin:12px 0}}
.warnbox{{background:#1f242b;border-left:3px solid #d2991d;padding:12px 16px;margin:12px 0}}
.okbox{{background:#1f242b;border-left:3px solid #7ee787;padding:12px 16px;margin:12px 0}}
ul{{margin:8px 0 8px 20px}}
li{{margin:4px 0}}
footer{{text-align:center;color:#6e7681;margin:40px 0 16px}}
</style>
</head>
<body>
<h1>P10 光子计数 CT 系统复核报告</h1>
<p style="text-align:center;color:#8b949e">基于原始 DICOM 与既有报告复核后重生成 | 目录 <span class="mono">{DATA_DIR}</span> | 文件数 {len(files):,}</p>

<h2>1. 执行摘要</h2>
<div class="okbox">
<ul>
<li><span class="good">已确认</span>：私有标签 <span class="mono">(EFE1,1001)</span> 是一个约 15.17 MB 的复合二进制块，前半部分包含材料头、光谱曲线与校准数据，后半部分包含 <span class="mono">256</span> 个 JPEG2000 码流。</li>
<li><span class="good">已确认</span>：码流可分成 4 组，每组 64 个槽位，其中四角为空，所以每组有效 tile 数为 <span class="mono">60</span>；每个 tile 解码后大小为 <span class="mono">256x256 int16</span>。</li>
<li><span class="good">已确认</span>：tile 内部由 8 个连续的 <span class="mono">32x256</span> strip 构成，这些 strip 以空间交错方式拼成 <span class="mono">2048x2048</span> 图像。</li>
<li><span class="good">已检验</span>：`strip` 不支持“8 组正弦编码”解释。1D 单正弦拟合在非平坦 strip 上中位 <span class="mono">R²={sine_payload['summary']['nonflat_strips']['median_fit_r2']:.3f}</span>，2D 频谱方向一致性仅中等，且 DCT/小波压缩性优于 Fourier。</li>
<li><span class="good">已确认</span>：光谱曲线中存在 4 个稳定材料形状，碘曲线在 index 33 处出现明显跃迁，对应约 33 keV K-edge。</li>
<li><span class="warn">保守修正</span>：本次不再把 <span class="mono">G1-G0</span> 直接表述为“等于最终 CT 图像”。它与 PixelData 仅表现出中等结构相关性，线性拟合残差也很大。</li>
<li><span class="warn">保守修正</span>：已明确 DICOM 标签 <span class="mono">ME 60keV/SI</span> 中的 <span class="mono">SI</span> 是能量数据标记，不代表硅元素；因此本次仍不根据该字段外推探测器材料，缺少厂商文档前不再直接断言为 <span class="mono">CdZnTe</span>。</li>
</ul>
</div>

<h2>2. 关键统计</h2>
<table>
<tr><th>项目</th><th>值</th><th>说明</th></tr>
<tr><td>DICOM 文件数</td><td>{len(files):,}</td><td>从 <span class="mono">{files[0]}</span> 到 <span class="mono">{files[-1]}</span></td></tr>
<tr><td>PixelData 尺寸</td><td>{ds0.Rows} x {ds0.Columns}</td><td>标准图像像素矩阵</td></tr>
<tr><td>EFE1 数据长度</td><td>{len(blob):,} bytes</td><td>私有标签 <span class="mono">(EFE1,1001)</span></td></tr>
<tr><td>JPEG2000 码流数</td><td>{len(markers)}</td><td>通过查找 <span class="mono">jp2c</span> 标记得到</td></tr>
<tr><td>每组槽位 / 有效 tile</td><td>64 / 60</td><td>空槽位为 0, 7, 56, 63</td></tr>
<tr><td>Gap 区域</td><td>{len(gap_s16):,} int16</td><td>偏移 <span class="mono">0x1BA8-0x9DE0</span></td></tr>
<tr><td><span class="mono">G1-G0</span> vs PixelData 相关性</td><td>{diff_corr:.3f}</td><td>仅说明存在结构对应，不能证明两者数值等价</td></tr>
<tr><td>线性拟合残差</td><td>mean {fit_residual.mean():.0f}, std {fit_residual.std():.0f}</td><td>拟合公式 <span class="mono">y = {scale:.3f}x + {offset:.1f}</span></td></tr>
</table>

<h2>3. 图像总览</h2>
<div class="figure"><img src="data:image/png;base64,{overview_img}" alt="overview"></div>
<div class="note">
<ul>
<li><span class="mono">G0/G2</span> 更像参考帧或基线帧，整体偏负且动态范围大。</li>
<li><span class="mono">G1/G3</span> 更接近信号帧，数值范围围绕零附近展开。</li>
<li><span class="mono">G1-G0</span> 可以恢复出明显的解剖结构，但当前证据不足以把它与最终 PixelData 等同。</li>
<li>总览中的带状观感对应的是<span class="mono">低频空间纹理</span>，而非固定 <span class="mono">strip</span> 边界。</li>
</ul>
</div>

<h2>4. Tile 与 Strip 结构</h2>
<div class="figure"><img src="data:image/png;base64,{tile_img}" alt="tile"></div>
<div class="warnbox">
<ul>
<li>每个 tile 的 256 行不是 1 张完整子图，而是 8 个 <span class="mono">32x256</span> 的连续 strip。</li>
<li>把 strip 直接按空间位置交错回填后，能拼出完整 <span class="mono">2048x2048</span> 图像。</li>
<li>因此，DICOM 标签虽然显示设备支持 <span class="mono">8</span> 个能区，但当前文件里的 strip 是否一一对应“8 张独立能区图像”，本次不做超证据结论。</li>
</ul>
</div>

<h3>4.1 正弦假设、2D 频谱与压缩性检验</h3>
<div class="figure"><img src="data:image/png;base64,{rowband_img}" alt="rowband"></div>
<div class="warnbox">
<ul>
<li>逐行能量曲线的主频位于数百到上千行尺度，而不是 <span class="mono">8</span> 或 <span class="mono">32</span> 行的固定周期。</li>
<li><span class="mono">8/32</span> 行相位平均曲线只有轻微起伏，说明 strip 边界不是总览带状感的主要来源。</li>
<li><span class="mono">256</span> 行及更大尺度的相位起伏明显增强，符合低频空间纹理的视觉表现。</li>
</ul>
</div>
<div class="figure"><img src="data:image/png;base64,{basis_img}" alt="basis-analysis"></div>
<table>
<tr><th>检验项</th><th>结果</th><th>解读</th></tr>
<tr><td>1D 单正弦拟合（非平坦 strip）</td><td>mean R² = {sine_payload['summary']['nonflat_strips']['mean_fit_r2']:.3f}, median R² = {sine_payload['summary']['nonflat_strips']['median_fit_r2']:.3f}</td><td class="warn">大多数 strip 不能由单一正弦解释</td></tr>
<tr><td>1D 主频分布</td><td>{', '.join(str(x) for x in sine_payload['summary']['nonflat_strips']['unique_freq_bins'])}</td><td class="warn">主频分散，不像固定 8 组规则编码</td></tr>
<tr><td>2D 方向一致性（diff tile）</td><td>mean coherence = {basis_payload['diff']['orientation_summary']['mean_coherence']:.3f}, median = {basis_payload['diff']['orientation_summary']['median_coherence']:.3f}</td><td class="warn">只有中等偏弱的方向性</td></tr>
<tr><td>2D 轴向能量</td><td>horizontal = {basis_payload['diff']['orientation_summary']['mean_horizontal_axis_ratio']:.3f}, vertical = {basis_payload['diff']['orientation_summary']['mean_vertical_axis_ratio']:.3f}</td><td class="warn">水平/垂直轴能量接近，未见强单轴条纹</td></tr>
<tr><td>1% 系数压缩（diff tile）</td><td>Fourier R² = {basis_payload['diff']['compression_summary']['0.010']['fft_r2_mean']:.3f}, DCT R² = {basis_payload['diff']['compression_summary']['0.010']['dct_r2_mean']:.3f}</td><td class="warn">远非“少数基函数即可近乎无损”</td></tr>
<tr><td>5% / 10% 系数压缩（diff tile）</td><td>DCT R² = {basis_payload['diff']['compression_summary']['0.050']['dct_r2_mean']:.3f} / {basis_payload['diff']['compression_summary']['0.100']['dct_r2_mean']:.3f}</td><td class="good">逐步提升，更像普通图像行带</td></tr>
</table>
<div class="note">
<ul>
<li>若 tile/strip 真是“少数正弦基函数编码”，通常会看到更高的方向一致性，以及在 <span class="mono">1%</span> 左右系数预算下就接近完美重建。</li>
<li>本次样本中，Fourier 压缩并不优于 DCT/小波；尤其在 <span class="mono">diff tile</span> 上，DCT 在 <span class="mono">5%</span> 和 <span class="mono">10%</span> 预算下分别达到 <span class="mono">{basis_payload['diff']['compression_summary']['0.050']['dct_r2_mean']:.3f}</span> 与 <span class="mono">{basis_payload['diff']['compression_summary']['0.100']['dct_r2_mean']:.3f}</span>。</li>
<li>因此，当前证据更支持“空间图像行带/低频纹理”解释，而不是“8 组被压缩的正弦数据”。</li>
</ul>
</div>

<h3>4.2 低频纹理与解剖结构的关系</h3>
<div class="figure"><img src="data:image/png;base64,{anatomy_img}" alt="anatomy-relation"></div>
<table>
<tr><th>检验项</th><th>相关系数</th><th>解读</th></tr>
<tr><td>2D low-pass(G1-G0) vs low-pass(PixelData)</td><td>{anatomy_metrics['corr_2d']:.3f}</td><td class="good">强相关，说明低频纹理与广义解剖形态一致</td></tr>
<tr><td>逐行低频纹理 vs 体轮廓宽度</td><td>{anatomy_metrics['corr_width']:.3f}</td><td class="good">中等正相关</td></tr>
<tr><td>逐行低频纹理 vs 行平均衰减</td><td>{anatomy_metrics['corr_mean']:.3f}</td><td class="warn">中等负相关，说明强度分布会调制纹理</td></tr>
<tr><td>逐行低频纹理 vs 高衰减结构负荷</td><td>{anatomy_metrics['corr_hi']:.3f}</td><td class="good">中等相关，提示与致密结构分布有关</td></tr>
<tr><td>逐行低频纹理 vs 横向边缘强度</td><td>{anatomy_metrics['corr_grad']:.3f}</td><td class="good">中等相关，提示与解剖边界复杂度有关</td></tr>
</table>
<div class="note">
<ul>
<li>这里的“相关”是指与<span class="mono">广义解剖结构</span>相关，例如体轮廓、边界复杂度、高衰减结构负荷，而不是已经精确定位到某个器官。</li>
<li><span class="mono">2D low-pass</span> 相关达到 <span class="mono">{anatomy_metrics['corr_2d']:.3f}</span>，说明低频纹理并非随机背景，也不是固定采样条带，更像随解剖形态变化的低频调制。</li>
</ul>
</div>

<h3>4.3 区域分块与环带检验</h3>
<div class="figure"><img src="data:image/png;base64,{region_img}" alt="region-analysis"></div>
<table>
<tr><th>检验项</th><th>结果</th><th>解读</th></tr>
<tr><td>上/中/下 最强区域</td><td>{region_payload['summary']['best_horizontal_by_tex_pix']['name']} (r = {region_payload['summary']['best_horizontal_by_tex_pix']['corr_tex_pix']:.3f})</td><td class="good">中部最强，明显高于上部与下部</td></tr>
<tr><td>左/中/右 最强区域</td><td>{region_payload['summary']['best_vertical_by_tex_pix']['name']} (r = {region_payload['summary']['best_vertical_by_tex_pix']['corr_tex_pix']:.3f})</td><td class="good">中侧最强，明显高于左右两侧</td></tr>
<tr><td>3x3 热图峰值</td><td>下-中 = {region_payload['grid'][2][1]['corr_tex_pix']:.3f}, 中-中 = {region_payload['grid'][1][1]['corr_tex_pix']:.3f}</td><td class="good">中轴附近格点最强</td></tr>
<tr><td>中心环带 vs 外周环带</td><td>{region_payload['rings'][0]['corr_tex_pix']:.3f} / {region_payload['rings'][1]['corr_tex_pix']:.3f}</td><td class="warn">两者都高，说明不是简单“越靠中心越强”</td></tr>
</table>
<div class="note">
<ul>
<li><span class="mono">3x3</span> 分块结果更支持“低频纹理沿身体中轴附近更强”，而不是只由图像边缘或角落决定。</li>
<li>环带检验中，中心环带与外周环带的相关性都接近 <span class="mono">0.94</span>，说明“是否位于中心”不是唯一因素；更可能是<span class="mono">中轴方向的结构组织方式</span>在起作用。</li>
<li>因此，当前最保守的表述是：低频纹理更贴近<span class="mono">中心解剖通道/中轴邻域</span>，而不是简单贴近外周边界或图像中心点。</li>
</ul>
</div>

<h3>4.4 最终成像响应候选比较</h3>
<div class="figure"><img src="data:image/png;base64,{final_img}" alt="final-response-candidates"></div>
<table>
<tr><th>候选</th><th>raw corr</th><th>low corr</th><th>edge corr</th><th>bone-soft sep</th><th>综合分数</th></tr>
{''.join(f"<tr><td>{row['name']}</td><td>{row['raw_corr']:.3f}</td><td>{row['low_corr']:.3f}</td><td>{row['edge_corr']:.3f}</td><td>{row['bone_soft_sep']:.3f}</td><td>{row['image_like_score']:.3f}</td></tr>" for row in final_payload['candidates'])}
</table>
<div class="note">
<ul>
<li>当前在 <span class="mono">G3</span>、<span class="mono">G1-G0</span>、<span class="mono">G3-G2</span> 三者中，综合最像最终成像响应的是 <span class="mono">{final_payload['best_candidate']}</span>。</li>
<li><span class="mono">G3-G2</span> 的原始结构相关最高，达到 <span class="mono">{next(row['raw_corr'] for row in final_payload['candidates'] if row['name']=='G3-G2'):.3f}</span>；说明它最接近最终图像外观。</li>
<li><span class="mono">G1-G0</span> 的低频相关与骨/软组织分离更强，更像“强校正/增强后的中间响应”；<span class="mono">G3</span> 更像单组图像通道，但完整性不如 <span class="mono">G3-G2</span>。</li>
</ul>
</div>

<h2>5. 碘响应近似反演</h2>
<h3>5.1 最小碘权重模型</h3>
<div class="figure"><img src="data:image/png;base64,{minimal_iodine_img}" alt="minimal-iodine-model"></div>
<table>
<tr><th>步骤</th><th>公式/说明</th></tr>
<tr><td>频谱差分源</td><td><span class="mono">diff_tiles = G1 - G0</span></td></tr>
<tr><td>8-bin 低分辨率图</td><td><span class="mono">band_stack[b]</span> 由第 <span class="mono">b</span> 个 strip 按空间位置拼成 <span class="mono">256x2048</span> 图像</td></tr>
<tr><td>材料响应矩阵</td><td><span class="mono">bin_curve_matrix[b,m]</span> = 对材料曲线在第 <span class="mono">b</span> 个能区内求平均</td></tr>
<tr><td>碘权重向量</td><td><span class="mono">wI = project_out(Iodine, [Water, Calcium, Gd, 1])</span></td></tr>
<tr><td>低分辨率碘响应</td><td><span class="mono">iodine_low(r,c) = Σ_b wI[b] * band_stack[b,r,c]</span></td></tr>
<tr><td>逐像素近似图</td><td><span class="mono">iodine_full = norm(G3-G2) * (1 - 0.75 + 0.75 * norm(upsample(iodine_low)))</span></td></tr>
</table>
<table>
<tr><th>指标</th><th>结果</th><th>解读</th></tr>
<tr><td>8-bin 碘权重向量</td><td>{', '.join(f'{v:+.3f}' for v in minimal_iodine_payload['iodine_weights'])}</td><td>跨 8 个能区的启发式碘投影方向</td></tr>
<tr><td>低分辨率图 vs PixelData</td><td>{minimal_iodine_payload['summary']['low_to_pixel_corr']:.3f}</td><td class="warn">仅中等相关，单靠低分辨率频谱图不足以恢复像素结构</td></tr>
<tr><td>近似逐像素图 vs PixelData</td><td>{minimal_iodine_payload['summary']['full_to_pixel_corr']:.3f}</td><td class="good">结构底图 <span class="mono">G3-G2</span> 明显提升了解剖一致性</td></tr>
<tr><td>高衰减区 - 软组织区对比</td><td>{minimal_iodine_payload['metrics']['contrast_hi_soft']:.3f}</td><td class="good">对高响应区域有明显拉开效果</td></tr>
</table>
<div class="note">
<ul>
<li>这一步产出的 <span class="mono">iodine_full</span> 是“近似逐像素碘权重代理图”，不是厂商协议级的定量碘浓度图。</li>
<li>从结果上看，<span class="mono">G1-G0</span> 主要提供频谱调制，<span class="mono">G3-G2</span> 主要提供空间结构，因此两者缺一不可。</li>
</ul>
</div>

<h3>5.2 近似碘权重图的空间对应性</h3>
<div class="figure"><img src="data:image/png;base64,{iodine_spatial_img}" alt="iodine-spatial-correspondence"></div>
<table>
<tr><th>检验项</th><th>结果</th><th>解读</th></tr>
<tr><td>6 张采样层中 iod 热点落在高衰减区</td><td>{iodine_spatial_payload['aggregate']['mean_iod_hot_on_high']:.3f}</td><td class="warn">比例较低，不支持“只是骨性高密度染亮”</td></tr>
<tr><td>6 张采样层中 iod 热点落在非高衰减区</td><td>{iodine_spatial_payload['aggregate']['mean_iod_hot_outside_high']:.3f}</td><td class="good">多数热点位于非骨性高衰减区域</td></tr>
<tr><td>热点相对高衰减区富集倍数</td><td>{iodine_spatial_payload['aggregate']['mean_hot_enrichment_vs_high']:.1f}x</td><td class="good">热点面积远大于高衰减骨性区域面积占比</td></tr>
<tr><td>热点到高衰减区中位距离</td><td>{iodine_spatial_payload['aggregate']['mean_hot_dist_to_high_median']:.1f} px</td><td class="warn">热点通常不贴着最高衰减区边缘</td></tr>
<tr><td>软组织样热点占体内比例</td><td>{iodine_spatial_payload['aggregate']['mean_suspicious_soft_share_body']:.6f}</td><td class="warn">整体占比很小，仅在部分下部切片出现</td></tr>
<tr><td>软组织样连通域数（均值）</td><td>{iodine_spatial_payload['aggregate']['mean_suspicious_component_count']:.2f}</td><td class="warn">仅提示存在候选区域，不能直接视为真实造影灶</td></tr>
</table>
<div class="note">
<ul>
<li>代表层中，<span class="mono">iodine_full</span> 热点落在高衰减区的比例仅 <span class="mono">{iodine_spatial_payload['representative_slice']['metrics']['iod_hot_on_high']:.3f}</span>，落在非高衰减区的比例达 <span class="mono">{iodine_spatial_payload['representative_slice']['metrics']['iod_hot_outside_high']:.3f}</span>。</li>
<li>跨 6 张采样层，这一趋势保持稳定，说明 <span class="mono">iodine_full</span> 并不是简单把骨头或最高密度结构整体抬亮。</li>
<li>在靠近序列下部的切片中，可见少量与高衰减区分离的软组织样热点；例如最大两层分别出现 <span class="mono">{max(row['metrics']['suspicious_soft_count'] for row in iodine_spatial_payload['samples'])}</span> 与 <span class="mono">{sorted((row['metrics']['suspicious_soft_count'] for row in iodine_spatial_payload['samples']), reverse=True)[1]}</span> 个候选像素。</li>
<li>因此，更稳妥的表述是：近似碘权重图反映的是<span class="mono">非骨性结构响应 + 部分高衰减耦合</span>，但尚不能直接当作定量碘浓度图使用。</li>
</ul>
</div>

<h3>5.3 软组织样热点的相邻切片连续性</h3>
<div class="figure"><img src="data:image/png;base64,{hotspot_continuity_img}" alt="iodine-hotspot-continuity"></div>
<table>
<tr><th>窗口</th><th>候选像素数</th><th>最长连续层数</th><th>连续 pair 数</th><th>pair 平均重叠</th></tr>
{''.join(f"<tr><td>{row['anchor_index']} ± {hotspot_continuity_payload['window_radius']}</td><td>{', '.join(str(x) for x in row['counts'])}</td><td>{row['longest_run']}</td><td>{row['continuous_pair_count']}</td><td>{row['mean_pair_overlap']:.3f}</td></tr>" for row in hotspot_continuity_payload['windows'])}
</table>
<div class="note">
<ul>
<li>最佳窗口位于 <span class="mono">{hotspot_continuity_payload['best_window']['anchor_index']}</span> 附近，5 个相邻切片中的候选像素数分别为 <span class="mono">{', '.join(str(x) for x in hotspot_continuity_payload['best_window']['counts'])}</span>。</li>
<li>该窗口的最长连续层数达到 <span class="mono">{hotspot_continuity_payload['best_window']['longest_run']}</span>，相邻切片膨胀重叠均值达到 <span class="mono">{hotspot_continuity_payload['best_window']['mean_pair_overlap']:.3f}</span>，支持这些热点不是单层闪现噪声。</li>
<li>靠近序列末端的 <span class="mono">{hotspot_continuity_payload['windows'][-1]['anchor_index']}</span> 邻域也保持了 <span class="mono">{hotspot_continuity_payload['windows'][-1]['longest_run']}</span> 层连续和 <span class="mono">{hotspot_continuity_payload['windows'][-1]['mean_pair_overlap']:.3f}</span> 的重叠。</li>
<li>质心距离在部分窗口会因主连通域切换而波动较大，因此这里更看重<span class="mono">连续层数</span>与<span class="mono">相邻膨胀重叠</span>，而不是单一质心轨迹。</li>
</ul>
</div>

<h2>6. 光谱曲线与材料头</h2>
<div class="figure"><img src="data:image/png;base64,{curve_img}" alt="curves"></div>
<table>
<tr><th>材料名</th><th>头偏移</th><th>14 个参数存储</th><th>备注</th></tr>
{''.join(f"<tr><td>{m['name']}</td><td class='mono'>0x{m['offset']:03X}</td><td>14 x uint32</td><td>{'含可解释 float32 值' if any(np.isfinite(v) and 0.01 < abs(v) < 5000 and abs(v-round(v)) > 1e-3 for v in m['f32']) else '主要表现为整型或零值'}</td></tr>" for m in materials)}
</table>
<table>
<tr><th>曲线</th><th>偏移</th><th>最大值</th><th>确认方式</th></tr>
<tr><td>Water</td><td class="mono">0x01C0</td><td>{curves['Water'].max():.1f}</td><td>头部顺序 + 曲线形状</td></tr>
<tr><td>Iodine</td><td class="mono">0x0850</td><td>{curves['Iodine'].max():.1f}</td><td class="good">K-edge @ index 33</td></tr>
<tr><td>Calcium</td><td class="mono">0x0B70</td><td>{curves['Calcium'].max():.1f}</td><td>头部顺序 + 曲线形状</td></tr>
<tr><td>Gadolinium</td><td class="mono">0x0ED4</td><td>{curves['Gadolinium'].max():.1f}</td><td>头部顺序 + 曲线形状</td></tr>
<tr><td>Curve #8</td><td class="mono">0x1878</td><td>{curve8.max():.1f}</td><td>与碘曲线形状相关系数 {curve8_corr:.3f}</td></tr>
</table>

<h3>可疑似 float32 的材料头参数</h3>
<table>
<tr><th>材料</th><th>参数序号</th><th>uint32</th><th>同位解释为 float32</th></tr>
{float_rows_html}
</table>

<h2>7. 未解码 Gap 区域</h2>
<div class="figure"><img src="data:image/png;base64,{gap_img}" alt="gap"></div>
<table>
<tr><th>属性</th><th>值</th></tr>
<tr><td>偏移范围</td><td class="mono">0x1BA8-0x9DE0</td></tr>
<tr><td>数据类型</td><td>按 <span class="mono">int16</span> 解释最稳定</td></tr>
<tr><td>总长度</td><td>{len(gap_s16):,} 个值，{len(gap_s16) * 2:,} bytes</td></tr>
<tr><td>4 等分</td><td class="good">4,167 / bin，可整除</td></tr>
<tr><td>8 等分</td><td class="warn">不可整除</td></tr>
<tr><td>B2-B3 相关性</td><td>{gap_corr23:.3f}</td></tr>
<tr><td>状态</td><td class="warn">统计结构清晰，但物理意义仍未知</td></tr>
</table>

<h2>8. 跨切片稳定性</h2>
<div class="figure"><img src="data:image/png;base64,{consistency_img}" alt="consistency"></div>
<table>
<tr><th>Slice</th><th>Header 0x000-0x27F</th><th>Curve 0x1C0-0x1878</th><th>Gap 相关性</th><th>EFE1 长度</th></tr>
{slice_rows_html}
</table>
<div class="okbox">
<ul>
<li>Header、主曲线区、Gap 区在抽样切片间几乎完全稳定，说明它们更像全局校准/模板信息。</li>
<li>真正随切片变化的是后段 JPEG2000 图像码流。</li>
</ul>
</div>

<h2>9. DICOM 标签复核</h2>
<table>
<tr><th>标签</th><th>VR</th><th>值</th><th>解读</th></tr>
<tr><td class="mono">(01E7,1001)</td><td>{ds0[(0x01E7,0x1001)].VR}</td><td>{ds0[(0x01E7,0x1001)].value}</td><td class="warn">保留字面值，不外推为 CdZnTe</td></tr>
<tr><td class="mono">(01E7,1002)</td><td>{ds0[(0x01E7,0x1002)].VR}</td><td>{ds0[(0x01E7,0x1002)].value}</td><td>标称 60</td></tr>
<tr><td class="mono">(01E7,1004)</td><td>{ds0[(0x01E7,0x1004)].VR}</td><td>{ds0[(0x01E7,0x1004)].value}</td><td>值为 2</td></tr>
<tr><td class="mono">(01F3,1031)</td><td>{ds0[(0x01F3,0x1031)].VR}</td><td>{ds0[(0x01F3,0x1031)].value}</td><td>设备配置中存在 8</td></tr>
<tr><td class="mono">(01F3,1032)</td><td>{ds0[(0x01F3,0x1032)].VR}</td><td>{ds0[(0x01F3,0x1032)].value}</td><td>两个模块</td></tr>
<tr><td class="mono">(01F3,1046)</td><td>{ds0[(0x01F3,0x1046)].VR}</td><td>{ds0[(0x01F3,0x1046)].value}</td><td class="warn">语义未确认</td></tr>
<tr><td class="mono">(01F1,104B)</td><td>{ds0[(0x01F1,0x104B)].VR}</td><td>{ds0[(0x01F1,0x104B)].value}</td><td>292 排 * 0.274 mm</td></tr>
<tr><td class="mono">(01F1,1008)</td><td>{ds0[(0x01F1,0x1008)].VR}</td><td>{ds0[(0x01F1,0x1008)].value}</td><td>几何参数</td></tr>
</table>

<h2>10. EFE1 数据块布局</h2>
<table>
<tr><th>偏移</th><th>大小</th><th>内容</th><th>结论级别</th></tr>
<tr><td class="mono">0x000-0x003</td><td>4 B</td><td>材料数量 = 4</td><td class="good">已确认</td></tr>
<tr><td class="mono">0x004-0x0FF</td><td>252 B</td><td>Water / Iodine / Calcium / Gadolinium 材料头</td><td class="good">已确认</td></tr>
<tr><td class="mono">0x100-0x1BF</td><td>192 B</td><td>校准/表项区</td><td class="warn">部分确认</td></tr>
<tr><td class="mono">0x1C0-0x1877</td><td>约 6 KB</td><td>7 条主光谱曲线</td><td class="good">已确认</td></tr>
<tr><td class="mono">0x1878-0x1BA7</td><td>800 B</td><td>Curve #8</td><td class="good">已确认</td></tr>
<tr><td class="mono">0x1BA8-0x9DE0</td><td>约 33 KB</td><td>未解码 int16 Gap 区</td><td class="warn">仅统计确认</td></tr>
<tr><td class="mono">0x9DE0-0x9E0F</td><td>48 B</td><td>JP2 容器头</td><td class="good">已确认</td></tr>
<tr><td class="mono">0x9E10-EOF</td><td>约 15.1 MB</td><td>256 个 JPEG2000 码流</td><td class="good">已确认</td></tr>
</table>

<h2>11. 本次相对旧稿的修正</h2>
<div class="warnbox">
<ul>
<li>把“<span class="mono">G1-G0 = CT 图像</span>”下调为“<span class="mono">G1-G0</span> 与 PixelData 存在结构对应，但不具备数值等价证据”。</li>
<li>把“8 能区图像”下调为“8 个空间交错 strip；与能区的一一对应关系尚未证实”。</li>
<li>新增“正弦/方向性条纹/少数基函数编码”检验：当前证据不支持该解释，更像普通图像行带。</li>
<li>新增“低频纹理与解剖结构相关性”检验：与广义解剖形态强相关，但尚不足以定位到具体器官级结构。</li>
<li>新增“区域分块与环带检验”：更支持沿中轴邻域增强，而非简单中心-外周单调变化。</li>
<li>新增“最终成像响应候选比较”：在 <span class="mono">G3</span>、<span class="mono">G1-G0</span>、<span class="mono">G3-G2</span> 三者中，<span class="mono">G3-G2</span> 当前最像最终成像响应。</li>
<li>新增“最小碘权重模型”：用 <span class="mono">G1-G0</span> 的 8-bin 频谱投影与 <span class="mono">G3-G2</span> 结构底图融合，得到近似逐像素碘权重代理图。</li>
<li>新增“近似碘权重图的空间对应性”检验：热点大多不落在最高衰减骨性区，更像非骨性结构响应，但仍需警惕这只是启发式反演结果。</li>
<li>新增“软组织样热点相邻切片连续性”检验：在部分下部窗口可见 3-5 层连续的候选热点，支持其并非单层随机噪声。</li>
<li>把“Redlen CdZnTe”下调为“与现有 DICOM 标签字面值冲突，暂不下定论”。</li>
</ul>
</div>

<footer>
<p>输出文件：<span class="mono">{OUT_HTML}</span></p>
<p>生成时间：2026-06-10</p>
</footer>
</body>
</html>
"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved: {OUT_HTML}")


if __name__ == "__main__":
    main()
