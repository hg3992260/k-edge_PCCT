import pydicom, os, numpy as np, base64, io
import glymur, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

base = r'D:\xl\00020006\00020006'
f = sorted(os.listdir(base))[0]
ds = pydicom.dcmread(os.path.join(base, f))
data = ds[0xefe1,0x1001].value

jp2c_marker = b'jp2c'
positions = []; p = -1
while True:
    p = data.find(jp2c_marker, p+1)
    if p == -1: break
    positions.append(p)

empty = {1:[64,71,120,127]}
g1_active = [i for i in range(64,128) if i not in empty[1]]

pixels = np.frombuffer(ds.PixelData, dtype=np.uint16).reshape(2048,2048).astype(np.float64)
gt_256 = zoom(pixels, 256/2048, order=1)

print("Decoding 12 G1 images...")
imgs = []
for cs_idx in g1_active[:12]:
    s = positions[cs_idx]+4; e = positions[cs_idx+1]
    fp = rf'D:\xl\00020006\ds_{cs_idx}.jp2k'
    with open(fp,'wb') as fh: fh.write(data[s:e])
    imgs.append(glymur.Jp2k(fp)[:].astype(np.float64))
imgs = np.array(imgs)
print(f"Done. Shape: {imgs.shape}")

# --- MAIN DIAGRAM ---
fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(5, 6, hspace=0.45, wspace=0.35)

# Row 0: Tags table
ax_tags = fig.add_subplot(gs[0, 0:6])
ax_tags.axis('off')
tag_text = (
    'DICOM Private Tags\n'
    '============================================================\n'
    'Group 01E7 (Energy):  1001=ME 60keV/SI (SI=energy marker)  1002=60 keV    1004=2 foci\n'
    'Group 01F3 (Config):  1031=8 bins          1032=2 modules 1046=290\n'
    'Group 01F1 (Geo):     104B=292*0.274mm     1008=S-D 361mm 1093=4 kVp\n'
    'Group EFE1 (DATA):    1001= OB blob ~15.1MB  (JPEG2000 compressed)\n'
    'Standard Tags:        KVP=120, RevTime=0.3s, FOV=433mm, Kernel=H30\n'
)
ax_tags.text(0.02, 0.95, tag_text, transform=ax_tags.transAxes, fontsize=9,
            family='monospace', verticalalignment='top', color='black',
            bbox=dict(boxstyle='round', facecolor='#e8f0fe', alpha=0.9))
ax_tags.set_title('DICOM Metadata Tags', fontsize=13, fontweight='bold', color='#1a237e', pad=10)

# Row 1: 4-group overview
for col_idx, (gp_name, gp_range, color, desc) in enumerate([
    ('G0', (0,63), '#ff7b72', 'Reference A\n(-33000~-9000)'),
    ('G1', (64,127), '#58a6ff', 'PRIMARY\n(-1500~+4400)'),
    ('G2', (128,191), '#f0883e', 'Reference B\n(-32768~-5000)'),
    ('G3', (192,255), '#7ee787', 'Data B\n(some saturated)'),
    ]):
    ax = fig.add_subplot(gs[1, col_idx])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.add_patch(plt.Rectangle((1, 3), 8, 5, facecolor=color, alpha=0.3,
                                edgecolor=color, linewidth=2))
    ax.text(5, 7.2, f'Group {gp_name}', ha='center', fontsize=11, fontweight='bold', color=color)
    ax.text(5, 6.2, f'CS {gp_range[0]}-{gp_range[1]}', ha='center', fontsize=9, family='monospace')
    ax.text(5, 4.5, desc, ha='center', fontsize=7)
    empty_str = {0:'0,7,56,63',1:'64,71,120,127',2:'128,135,184,191',3:'192,199,248,255'}[col_idx]
    ax.text(5, 3.3, f'Empty: {empty_str}', ha='center', fontsize=6, color='gray')

# Row 2: JP2K image structure
for col_idx in range(6):
    ax = fig.add_subplot(gs[2, col_idx])
    if col_idx == 0:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
        colors_eb = plt.cm.plasma(np.linspace(0.1, 0.9, 8))
        for eb in range(8):
            y0 = 1.5 + eb * 1.0
            ax.add_patch(plt.Rectangle((1, y0), 8, 0.9, facecolor=colors_eb[eb],
                                        alpha=0.6, edgecolor=colors_eb[eb], linewidth=0.5))
            ax.text(0.5, y0+0.45, f'B{eb}', fontsize=5, va='center', ha='right')
        ax.text(5, 9.8, '1 JP2K frame', ha='center', fontsize=10, fontweight='bold')
        ax.text(5, 9.0, '256 x 256 int16', ha='center', fontsize=9, family='monospace')
        ax.text(5, 0.8, '= 8 bins x 32 rows', ha='center', fontsize=8, color='gray')
    elif col_idx == 1:
        ax.text(0.5, 0.5, 'Each band =\nEnergy Bin image\n32 x 256 pixels\n\nALREADY\nRECONSTRUCTED\n(not sinogram!)',
                ha='center', va='center', fontsize=10, transform=ax.transAxes)
        ax.axis('off')
    elif col_idx == 2:
        ax.imshow(gt_256, cmap='gray', vmin=np.percentile(gt_256,1), vmax=np.percentile(gt_256,99))
        ax.set_title('DICOM GT\n(2048->256)', fontsize=8); ax.axis('off')
    elif col_idx == 3:
        b4 = imgs[0, 128:160, :]; vabs = np.percentile(np.abs(b4), 99.5)
        ax.imshow(b4, cmap='gray', vmin=-vabs, vmax=vabs, aspect='auto')
        ax.set_title('Bin4 Image 1', fontsize=8); ax.axis('off')
    elif col_idx == 4:
        b4 = imgs[1, 128:160, :]; vabs = np.percentile(np.abs(b4), 99.5)
        ax.imshow(b4, cmap='gray', vmin=-vabs, vmax=vabs, aspect='auto')
        ax.set_title('Bin4 Image 2', fontsize=8); ax.axis('off')
    else:
        b3 = imgs[2, 96:128, :]; vabs = np.percentile(np.abs(b3), 99.5)
        ax.imshow(b3, cmap='gray', vmin=-vabs, vmax=vabs, aspect='auto')
        ax.set_title('Bin3 Image 3', fontsize=8); ax.axis('off')

# Row 3: 8 band strips
colors_eb = plt.cm.plasma(np.linspace(0.1, 0.9, 8))
for eb in range(6):
    ax = fig.add_subplot(gs[3, eb])
    band = imgs[1, eb*32:(eb+1)*32, :]
    vabs = np.percentile(np.abs(band), 99.5) if band.ptp() > 0 else 1
    ax.imshow(band, cmap='gray', vmin=-vabs, vmax=vabs, aspect='auto')
    ax.set_title(f'Bin {eb} ({eb*32}-{eb*32+31})', fontsize=8); ax.axis('off')
ax3 = fig.add_subplot(gs[3, 5])
ax3.axis('off')
ax3.text(0.5, 0.5, '8 energy bin images\nper JPEG2000 frame\nstacked vertically',
         ha='center', va='center', fontsize=13, transform=ax3.transAxes,
         bbox=dict(boxstyle='round', facecolor='#e8f0fe', alpha=0.9))

# Row 4: Summary
ax_sum = fig.add_subplot(gs[4, 0:6])
ax_sum.axis('off')
summary = (
    'DATA FLOW\n'
    '============================================================\n'
    '(EFE1,1001) OB ~15.1MB\n'
    '  |-- Header (0x000-0x27F): 640B material params + calibration\n'
    '  |-- 256 JPEG2000 codestreams: G0(ref) | G1(DATA) | G2(ref) | G3(a)\n'
    '       |-- G1: 60 frames x 256x256 int16\n'
    '            |-- Each: 8 bands x 32 rows x 256 cols\n'
    '                 |-- Band = Energy Bin RECONSTRUCTED IMAGE (32x256)\n'
    '============================================================\n'
    'Physical: CdZnTe detector, 33-200 keV, 292 rows x 0.274mm\n'
    'Values: int16, zero-centered attenuation (-3000 to +4400)\n'
)
ax_sum.text(0.02, 0.98, summary, transform=ax_sum.transAxes, fontsize=8.5,
           family='monospace', verticalalignment='top', color='black',
           bbox=dict(boxstyle='round', facecolor='#e8f0fe', alpha=0.9))

plt.suptitle('P10 (EFE1,1001) Energy Bin Data -- Complete Structure', 
             fontsize=16, fontweight='bold', y=0.99)
plt.tight_layout()
buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120, bbox_inches='tight'); buf.seek(0)
b64_diagram = base64.b64encode(buf.read()).decode(); plt.close(fig)

# --- Band image comparison ---
fig2, axes2 = plt.subplots(2, 4, figsize=(20, 8))
for eb in range(8):
    row, col = divmod(eb, 4)
    band_avg = imgs[:5, eb*32:(eb+1)*32, :].mean(axis=0)
    vabs = np.percentile(np.abs(band_avg), 99.5) if band_avg.ptp() > 0 else 1
    axes2[row,col].imshow(band_avg, cmap='gray', vmin=-vabs, vmax=vabs, aspect='auto')
    axes2[row,col].set_title(f'Bin {eb} [{band_avg.min():.0f}, {band_avg.max():.0f}]'); axes2[row,col].axis('off')
plt.suptitle('8 Energy Bin Reconstructed Images (avg 5 frames)', fontsize=14, y=1.01)
plt.tight_layout()
buf2 = io.BytesIO(); fig2.savefig(buf2, format='png', dpi=100, bbox_inches='tight'); buf2.seek(0)
b64_bands = base64.b64encode(buf2.read()).decode(); plt.close(fig2)

# --- HTML ---
html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>P10 DICOM Tags & Data Structure</title>
<style>
body {{ font-family: 'Segoe UI','Microsoft YaHei',sans-serif; background: #0d1117; color: #c9d1d9; max-width: 1300px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #58a6ff; text-align: center; border-bottom: 2px solid #30363d; padding-bottom: 15px; }}
h2 {{ color: #f0883e; border-left: 4px solid #f0883e; padding-left: 12px; margin-top: 30px; }}
h3 {{ color: #d2a8ff; }}
.figure {{ margin: 20px 0; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }}
.figure img {{ width: 100%; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.85em; }}
th {{ background: #21262d; color: #8b949e; padding: 8px 12px; border: 1px solid #30363d; text-align: left; }}
td {{ padding: 6px 12px; border: 1px solid #30363d; vertical-align: top; }}
tr:nth-child(even) td {{ background: #161b22; }}
.tag-code {{ color: #a5d6ff; font-family: Consolas,monospace; font-size: 0.9em; }}
.tag-v {{ color: #e6edf3; }}
.kv {{ background: #21262d; border-radius: 3px; padding: 1px 5px; font-family: Consolas,monospace; font-size: 0.9em; }}
.highlight {{ background: #1f242b; border-left: 3px solid #7ee787; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 10px 0; }}
footer {{ text-align: center; color: #484f58; margin: 40px 0 20px; }}
</style>
</head>
<body>
<h1>P10 Photon-Counting CT &mdash; DICOM Tags &amp; Data Structure</h1>
<p style="text-align:center;color:#8b949e;">NeuViz P10 | 33&ndash;200 keV | 8 Energy Bins | 292 detector rows</p>

<h2>1. Complete DICOM Tag Map</h2>
<table>
<tr><th>Tag</th><th>VR</th><th>Value</th><th>Meaning</th></tr>
<tr><td colspan="4" style="background:#30363d;color:#58a6ff;font-weight:bold;">Group 01E7 &mdash; Energy / Detector Metadata</td></tr>
<tr><td class="tag-code">(01E7,0010)</td><td>LO</td><td class="tag-v">Group_01E7</td><td>Private Creator</td></tr>
<tr><td class="tag-code">(01E7,1001)</td><td>LO</td><td class="tag-v" style="color:#58a6ff;"><b>ME 60keV/SI</b></td><td>Detector / energy marker: Multi-Energy 60 keV; SI marks energy data, not silicon</td></tr>
<tr><td class="tag-code">(01E7,1002)</td><td>SL</td><td class="tag-v">60</td><td>Nominal keV (NOT view count)</td></tr>
<tr><td class="tag-code">(01E7,1004)</td><td>IS</td><td class="tag-v">2</td><td>Number of X-ray foci (FS0, FS1)</td></tr>
<tr><td colspan="4" style="background:#30363d;color:#7ee787;font-weight:bold;">Group 01F3 &mdash; Energy Bin Configuration</td></tr>
<tr><td class="tag-code">(01F3,0010)</td><td>LO</td><td class="tag-v">Group_01F3</td><td>Private Creator</td></tr>
<tr><td class="tag-code">(01F3,1031)</td><td>IS</td><td class="tag-v" style="color:#7ee787;"><b>8</b></td><td><b>Energy bin count</b></td></tr>
<tr><td class="tag-code">(01F3,1032)</td><td>IS</td><td class="tag-v">2</td><td>Detector module count</td></tr>
<tr><td class="tag-code">(01F3,1046)</td><td>DS</td><td class="tag-v">290</td><td>Meaning TBD (NOT keV range)</td></tr>
<tr><td colspan="4" style="background:#30363d;color:#f0883e;font-weight:bold;">Group 01F1 &mdash; Scan Geometry</td></tr>
<tr><td class="tag-code">(01F1,104B)</td><td>SH</td><td class="tag-v" style="color:#f0883e;"><b>292*0.274</b></td><td><b>292 detector rows x 0.274mm</b></td></tr>
<tr><td class="tag-code">(01F1,1008)</td><td>DS</td><td class="tag-v">361.132</td><td>Source-Detector distance (mm)</td></tr>
<tr><td class="tag-code">(01F1,1093)</td><td>IS</td><td class="tag-v">4</td><td>kVp level count</td></tr>
<tr><td class="tag-code">(01F1,104E)</td><td>LO</td><td class="tag-v">Body_Helical</td><td>Helical scan protocol</td></tr>
<tr><td colspan="4" style="background:#30363d;color:#da3633;font-weight:bold;">Group EFE1 &mdash; CORE DATA (Energy Bin Images) &starf;</td></tr>
<tr><td class="tag-code" style="color:#da3633;font-weight:bold;">(EFE1,1001)</td><td><b>OB</b></td><td class="tag-v" style="color:#da3633;font-weight:bold;">~15.1 MB</td><td><b style="color:#da3633;">Energy bin reconstructed images (JPEG2000)</b></td></tr>
<tr><td colspan="4" style="background:#30363d;color:#8b949e;font-weight:bold;">Standard DICOM Tags</td></tr>
<tr><td class="tag-code">(0008,0070)</td><td></td><td class="tag-v">NMS</td><td>Manufacturer</td></tr>
<tr><td class="tag-code">(0018,0060)</td><td></td><td class="tag-v">120</td><td>KVP</td></tr>
<tr><td class="tag-code">(0018,9305)</td><td></td><td class="tag-v">0.3</td><td>Revolution Time (s)</td></tr>
<tr><td class="tag-code">(0018,9311)</td><td></td><td class="tag-v">0.9</td><td>Spiral Pitch Factor</td></tr>
<tr><td class="tag-code">(0018,1100)</td><td></td><td class="tag-v">433</td><td>Reconstruction Diameter (mm)</td></tr>
<tr><td class="tag-code">(0028,0010)</td><td></td><td class="tag-v">2048</td><td>PixelData Rows</td></tr>
<tr><td class="tag-code">(0028,0011)</td><td></td><td class="tag-v">2048</td><td>PixelData Columns</td></tr>
</table>

<h2>2. Data Structure Diagram</h2>
<div class="figure"><img src="data:image/png;base64,{b64_diagram}"></div>

<h2>3. (EFE1,1001) OB Blob Layout</h2>
<table>
<tr><th>Offset</th><th>Size</th><th>Content</th></tr>
<tr><td><span class="kv">0x000-0x003</span></td><td>4 B</td><td>Material count = 4</td></tr>
<tr><td><span class="kv">0x004-0x0FF</span></td><td>252 B</td><td>Material params: Water, Iodine, Ca, Gd (name + 14 uint32 each)</td></tr>
<tr><td><span class="kv">0x100-0x1BF</span></td><td>192 B</td><td>Energy bin calibration map</td></tr>
<tr><td><span class="kv">0x1C0-0x27F</span></td><td>192 B</td><td>Float32 energy response curve (48 pts)</td></tr>
<tr><td style="color:#7ee787;"><span class="kv">0x280-EOF</span></td><td>~15.1 MB</td><td style="color:#7ee787;"><b>256 JPEG2000 codestreams</b></td></tr>
</table>

<h2>4. 8 Energy Bin Images</h2>
<div class="figure"><img src="data:image/png;base64,{b64_bands}"></div>

<h2>5. Key Summary</h2>
<div class="highlight">
<ul>
<li><b>Data type:</b> RECONSTRUCTED pixel images (not sinogram / raw projection)</li>
<li><b>JPEG2000 codec:</b> OpenJPEG 2.5.2, int16, 256x256 per frame</li>
<li><b>256 rows:</b> 8 energy bins x 32 rows each = stacked energy bin images</li>
<li><b>256 columns:</b> x-direction of reconstructed image</li>
<li><b>60 frames/group:</b> 60 z-slices or projection angles</li>
<li><b>Physical:</b> CdZnTe, 33-200 keV, 292 detector rows, 0.274mm pitch</li>
<li><b>Pixel values:</b> int16, zero-centered attenuation (-3100 to +4400)</li>
</ul>
</div>
<footer><p>P10 DICOM Tags & Data Structure | Redlen CdZnTe | NeuViz P10 | 2026-06-02</p></footer>
</body></html>"""

with open(r'D:\xl\00020006\p10_data_structure.html', 'w', encoding='utf-8') as fh:
    fh.write(html)
print(f'Saved: p10_data_structure.html ({len(html)/1024:.0f} KB)')
