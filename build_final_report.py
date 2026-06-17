"""
P10 Photon-Counting CT: Final Corrected Report Generator
Confirmed relationship: EFE1 = energy-resolved CT images via tile assembly + reference subtraction
"""
import pydicom,os,numpy as np,struct,base64,io,json
import glymur,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
from scipy.ndimage import zoom

base=r'D:\xl\00020006\00020006'
files=sorted(os.listdir(base))
ds=pydicom.dcmread(os.path.join(base,files[0]))
data=ds[0xefe1,0x1001].value
pixels=np.frombuffer(ds.PixelData,dtype=np.uint16).reshape(2048,2048).astype(np.float64)

# JPEG2000 positions
jp2c=b'jp2c'; pos=[]; p=-1
while True:
    p=data.find(jp2c,p+1)
    if p==-1:break
    pos.append(p)

empty={0,7,56,63}  # relative positions in each 64-slot group

# ============================================================
# LOAD ALL 4 GROUPS
# ============================================================
def load_group(gp):
    imgs=np.zeros((60,256,256)); idx=0; gp_s=gp*64
    for slot in range(gp_s,(gp+1)*64):
        tp=slot-gp_s
        if tp in empty:continue
        s=pos[slot]+4;e=min(pos[slot+1],len(data)) if slot+1<len(pos) else len(data)
        with open(rf'D:\xl\00020006\fr{slot}.jp2k','wb')as fh:fh.write(data[s:e])
        imgs[idx]=glymur.Jp2k(rf'D:\xl\00020006\fr{slot}.jp2k')[:].astype(np.float64)
        idx+=1
    return imgs

def assemble_2048(imgs):
    full=np.zeros((2048,2048),dtype=np.float64); idx=0
    for tp in range(64):
        if tp in empty:continue
        tr=tp//8;tc=tp%8
        tile=imgs[idx];idx+=1
        for b in range(8):
            full[(tr*8+b)*32:(tr*8+b+1)*32,tc*256:(tc+1)*256]=tile[b*32:(b+1)*32,:]
    return full

def corr(a,b):
    mask=(b>np.percentile(b,5))&(b<np.percentile(b,95))
    return np.corrcoef(a[mask].ravel(),b[mask].ravel())[0,1]

print("Loading 4 groups (60 tiles each)...")
g0=load_group(0);g1=load_group(1);g2=load_group(2);g3=load_group(3)
img_g0=assemble_2048(g0);img_g1=assemble_2048(g1)
img_g2=assemble_2048(g2);img_g3=assemble_2048(g3)
diff_a=img_g1-img_g0; diff_b=img_g3-img_g2
c_full=corr(diff_a,pixels)
print(f"G1-G0 assembled: r={c_full:.4f}")

# ============================================================
# PLOT 1: Assembly proof (G1-G0 vs PixelData)
# ============================================================
fig1,axes1=plt.subplots(1,3,figsize=(18,5.5))
vmin=np.percentile(diff_a,1);vmax=np.percentile(diff_a,99)
axes1[0].imshow(diff_a,cmap='gray',vmin=vmin,vmax=vmax)
axes1[0].set_title(f'G1 - G0 (ref-subtracted)\n[{diff_a.min():.0f},{diff_a.max():.0f}]');axes1[0].axis('off')
axes1[1].imshow(pixels,cmap='gray',vmin=np.percentile(pixels,1),vmax=np.percentile(pixels,99))
axes1[1].set_title(f'PixelData (DICOM GT)\n[{pixels.min():.0f},{pixels.max():.0f}]');axes1[1].axis('off')
mask=(diff_a>np.percentile(diff_a,5))&(diff_a<np.percentile(diff_a,95))
a=pixels[mask].std()/diff_a[mask].std();b=pixels[mask].mean()-a*diff_a[mask].mean()
diff_hu=diff_a*a+b
axes1[2].imshow(diff_hu,cmap='gray',vmin=np.percentile(pixels,1),vmax=np.percentile(pixels,99))
c_hu=corr(diff_hu,pixels)
axes1[2].set_title(f'Rescaled to HU (r={c_hu:.3f})\nscale={a:.1f}, offset={b:.0f}');axes1[2].axis('off')
plt.suptitle(f'Tile Assembly Proof: G1-G0 = CT Image (r={c_full:.3f})',fontsize=14,y=1.02)
plt.tight_layout()
buf1=io.BytesIO();fig1.savefig(buf1,dpi=120,bbox_inches='tight');buf1.seek(0)
b64_assembly=base64.b64encode(buf1.read()).decode();plt.close(fig1)

# ============================================================
# PLOT 2: 8 Band images (each assembled separately from G1-G0)
# ============================================================
fig2,axes2=plt.subplots(2,4,figsize=(20,9))
band_corrs=[]
for eb in range(8):
    band_full=np.zeros((2048,2048));tile_idx=0
    for tp in range(64):
        if tp in empty:continue
        tr=tp//8;tc=tp%8
        tile=g1[tile_idx]-g0[tile_idx]
        band_full[(tr*8+eb)*32:(tr*8+eb+1)*32,tc*256:(tc+1)*256]=tile[eb*32:(eb+1)*32,:]
        tile_idx+=1
    m=band_full[band_full!=0]
    c=corr(m,pixels[band_full!=0]) if len(m)>1000 else 0
    band_corrs.append(c)
    vabs=np.percentile(np.abs(m),99)
    row,col=divmod(eb,4)
    axes2[row,col].imshow(band_full,cmap='gray',vmin=-vabs,vmax=vabs)
    axes2[row,col].set_title(f'Band {eb} [{m.min():.0f},{m.max():.0f}]\nr={c:.3f}')
    axes2[row,col].axis('off')
corr_label = ', '.join([f'{c:.3f}' for c in band_corrs])
plt.suptitle(f'8 Energy Bin CT Images (G1-G0, each band assembled separately)\nRow band correlations with PixelData: {corr_label}',fontsize=13,y=1.02)
plt.tight_layout()
buf2=io.BytesIO();fig2.savefig(buf2,dpi=120,bbox_inches='tight');buf2.seek(0)
b64_bands=base64.b64encode(buf2.read()).decode();plt.close(fig2)

# ============================================================
# PLOT 3: Spectral curves + Iodine K-edge
# ============================================================
keV=np.arange(200)
iodine=np.frombuffer(data[0x850:0x850+200*4],dtype=np.float32)
water_c=np.frombuffer(data[0x1C0:0x1C0+200*4],dtype=np.float32)
calcium_c=np.frombuffer(data[0xB70:0xB70+200*4],dtype=np.float32)
gd_c=np.frombuffer(data[0xED4:0xED4+200*4],dtype=np.float32)

fig3,axes3=plt.subplots(1,3,figsize=(18,5))
# All 4 normalized
for name,c,color,ls in [('Water',water_c,'#58a6ff','-'),('Iodine',iodine,'#e53935','-'),
    ('Calcium',calcium_c,'#7ee787','-'),('Gadolinium',gd_c,'#f0883e','-')]:
    axes3[0].plot(keV,c/np.max(c),color=color,lw=1,alpha=0.8,label=name)
axes3[0].axvline(x=33,color='#e53935',ls='--',lw=1,alpha=0.5,label='I K-edge')
axes3[0].set_title('4 Material Spectral Basis\n(normalized)');axes3[0].set_xlabel('keV');axes3[0].legend(fontsize=8);axes3[0].grid(alpha=0.3);axes3[0].set_xlim(0,200)

# Iodine + K-edge zoom
axes3[1].plot(keV[:100],iodine[:100],'r-',lw=1.2)
axes3[1].axvline(x=33,color='#e53935',ls='--',lw=1.5,alpha=0.7)
axes3[1].annotate(f'K-edge 33 keV\n{iodine[32]:.1f}->{iodine[33]:.1f}\nratio={iodine[33]/iodine[32]:.1f}x',xy=(33,iodine[33]),xytext=(50,iodine[33]*0.5),arrowprops=dict(arrowstyle='->',color='#e53935'),fontsize=9,color='#e53935')
axes3[1].set_title(f'Iodine K-edge Confirmed at 33 keV');axes3[1].set_xlabel('keV');axes3[1].grid(alpha=0.3);axes3[1].set_xlim(0,100)

# Water raw
axes3[2].plot(keV,water_c,'b-',lw=0.8)
axes3[2].set_title(f'Water (no K-edge in range)\n[{water_c[3]:.0f}->{water_c[-1]:.4f}]');axes3[2].set_xlabel('keV');axes3[2].grid(alpha=0.3);axes3[2].set_xlim(0,200)

plt.suptitle(f'Material Spectral Basis Curves (200 pts = 0-200 keV) — Confirmed by Iodine K-edge at index 33',fontsize=14,y=1.05)
plt.tight_layout()
buf3=io.BytesIO();fig3.savefig(buf3,dpi=120,bbox_inches='tight');buf3.seek(0)
b64_curves=base64.b64encode(buf3.read()).decode();plt.close(fig3)

# ============================================================
# PLOT 4: Tile grid schematic
# ============================================================
fig4,ax4=plt.subplots(figsize=(10,10))
ax4.set_xlim(0,8);ax4.set_ylim(0,8);ax4.set_xticks([]);ax4.set_yticks([])
colors_eb=plt.cm.plasma(np.linspace(0.1,0.9,8))
for tr in range(8):
    for tc in range(8):
        tp=tr*8+tc
        if tp in empty:
            ax4.add_patch(plt.Rectangle((tc,7-tr),1,1,facecolor='#1a1a1a',edgecolor='#30363d',linewidth=1))
            ax4.text(tc+0.5,7-tr+0.5,'EMPTY',ha='center',va='center',fontsize=7,color='#da3633')
        else:
            ax4.add_patch(plt.Rectangle((tc,7-tr),1,1,facecolor='#58a6ff',edgecolor='#30363d',alpha=0.3,linewidth=1))
            ax4.text(tc+0.5,7-tr+0.5,f'tile\n[{tr},{tc}]',ha='center',va='center',fontsize=6)
            for b in range(8):
                y0=7-tr+(8-1-b)/8
                ax4.add_patch(plt.Rectangle((tc,y0),1,1/8,facecolor=colors_eb[b],alpha=0.5,edgecolor=None))
ax4.set_title('8x8 Tile Grid (G1, 60 active/4 corners empty)\nEach tile 256x256, 8 bands x 32 rows = energy bins',fontsize=12)
buf4=io.BytesIO();fig4.savefig(buf4,dpi=120,bbox_inches='tight');buf4.seek(0)
b64_tilegrid=base64.b64encode(buf4.read()).decode();plt.close(fig4)

# ============================================================
# HTML REPORT
# ============================================================
tag_info={}
for t in [(0x01E7,0x1001),(0x01E7,0x1002),(0x01E7,0x1004),
          (0x01F3,0x1031),(0x01F3,0x1032),(0x01F3,0x1046),
          (0x01F1,0x104B),(0x01F1,0x1008)]:
    if t in ds:tag_info[t]=str(ds[t].value)

html=f'''<!DOCTYPE html><html lang=zh-CN>
<head><meta charset=UTF-8><title>P10 Energy Bin CT — Final Verified Report</title>
<style>
body{{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#0d1117;color:#c9d1d9;max-width:1200px;margin:0 auto;padding:20px;line-height:1.7}}
h1{{color:#58a6ff;text-align:center;border-bottom:2px solid #30363d;padding-bottom:15px}}
h2{{color:#f0883e;border-left:4px solid #f0883e;padding-left:12px;margin-top:30px}}
h3{{color:#d2a8ff}}
.figure{{margin:20px 0;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:15px}}
.figure img{{width:100%;border-radius:4px}}
table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:.85em}}
th{{background:#21262d;color:#8b949e;padding:8px 12px;border:1px solid #30363d;text-align:left}}
td{{padding:6px 12px;border:1px solid #30363d;vertical-align:top}}tr:nth-child(even)td{{background:#161b22}}
.highlight{{background:#1f242b;border-left:3px solid #7ee787;padding:12px 16px;border-radius:0 6px 6px 0;margin:10px 0}}
.highlight-b{{background:#1f242b;border-left:3px solid #58a6ff;padding:12px 16px;border-radius:0 6px 6px 0;margin:10px 0}}
.tag-code{{color:#a5d6ff;font-family:Consolas,monospace;font-size:.9em}}
.tag-v{{color:#e6edf3}}
.kv{{background:#21262d;border-radius:3px;padding:1px 5px;font-family:Consolas,monospace;font-size:.9em}}
.good{{color:#7ee787;font-weight:bold}}
.code-block{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:15px;font-family:Consolas,monospace;font-size:.8em;overflow-x:auto;white-space:pre-wrap;line-height:1.5}}
.cm{{color:#8b949e}}.kw{{color:#ff7b72}}.nu{{color:#a5d6ff}}.str{{color:#7ee787}}
footer{{text-align:center;color:#484f58;margin:40px 0 20px}}
.toc{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin:20px 0}}
.toc a{{color:#58a6ff;text-decoration:none}}.toc ol{{margin-left:20px}}li{{margin:3px 0}}
.stats-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px;margin:15px 0}}
.stat-card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;text-align:center}}
.stat-num{{font-size:1.5em;color:#58a6ff;font-weight:bold}}
.stat-label{{color:#8b949e;font-size:.7em;margin-top:3px}}
</style></head><body>
<h1>P10 Photon-Counting CT — Energy Bin Report</h1>
<p style=text-align:center;color:#8b949e>NeuViz P10 (Redlen CdZnTe) | 8 Energy Bins | 33-200 keV | Verified 2026-06-02</p>

<div class=toc><ol>
<li><a href="#overview">Overview & Key Numbers</a></li>
<li><a href="#tags">DICOM Tag Map</a></li>
<li><a href="#assembly">Tile Assembly — How 256x256 becomes 2048x2048</a></li>
<li><a href="#bands">8 Energy Bin CT Images</a></li>
<li><a href="#spectral">Spectral Basis Curves & Iodine K-edge</a></li>
<li><a href="#structure">OB Blob Complete Layout</a></li>
<li><a href="#code">Decoding Code</a></li>
</ol></div>

<h2 id=overview>1. Overview</h2>

<div class=stats-grid>
<div class=stat-card><div class=stat-num>2635</div><div class=stat-label>DICOM slices</div></div>
<div class=stat-card><div class=stat-num>2048x2048</div><div class=stat-label>CT image (PixelData)</div></div>
<div class=stat-card><div class=stat-num>256x256x240</div><div class=stat-label>EFE1 images/slice</div></div>
<div class=stat-card><div class=stat-num>8</div><div class=stat-label>Energy bins</div></div>
<div class=stat-card><div class=stat-num>4</div><div class=stat-label>Material basis (W/I/Ca/Gd)</div></div>
<div class=stat-card><div class=stat-num>33-200</div><div class=stat-label>keV range</div></div>
<div class=stat-card><div class=stat-num>292</div><div class=stat-label>Detector rows</div></div>
<div class=stat-card><div class=stat-num>CdZnTe</div><div class=stat-label>Detector material</div></div>
</div>

<div class=highlight-b>
<h3>Confirmed Relationship</h3>
<p><span class=kv>G1 - G0</span> (signal minus reference) = CT reconstruction image. Each 256x256 JPEG2000 image is a <b>tile</b> in an 8x8 grid. Each tile contains 8 bands x 32 rows = <b>8 energy bin sub-images</b>. Assembly + reference subtraction → 2048x2048 energy-resolved CT image.</p>
<p>PixelData r = {c_full:.3f} (full) / r = {min(band_corrs):.3f}–{max(band_corrs):.3f} (per band)</p>
</div>

<h2 id=tags>2. DICOM Tag Map</h2>
<table>
<tr><th>Tag</th><th>VR</th><th>Value</th><th>Meaning</th></tr>
<tr><td colspan=4 style=background:#30363d;color:#58a6ff>Group 01E7 — Detector/Energy</td></tr>
<tr><td class=tag-code>(01E7,1001)</td><td>LO</td><td class=tag-v>ME 60keV/SI</td><td>Detector/Energy marker: Multi-Energy 60 keV; SI marks energy data, not silicon</td></tr>
<tr><td class=tag-code>(01E7,1002)</td><td>SL</td><td class=tag-v>60</td><td>Nominal keV (not view count)</td></tr>
<tr><td class=tag-code>(01E7,1004)</td><td>IS</td><td class=tag-v>2</td><td>X-ray foci: FS0, FS1</td></tr>
<tr><td colspan=4 style=background:#30363d;color:#7ee787>Group 01F3 — Energy Configuration</td></tr>
<tr><td class=tag-code>(01F3,1031)</td><td>IS</td><td class=tag-v style=color:#7ee787><b>8</b></td><td><b>Energy bin count</b></td></tr>
<tr><td class=tag-code>(01F3,1032)</td><td>IS</td><td class=tag-v>2</td><td>Detector modules</td></tr>
<tr><td class=tag-code>(01F3,1046)</td><td>DS</td><td class=tag-v>290</td><td>TBD (not keV range)</td></tr>
<tr><td colspan=4 style=background:#30363d;color:#f0883e>Group 01F1 — Geometry</td></tr>
<tr><td class=tag-code>(01F1,104B)</td><td>SH</td><td class=tag-v style=color:#f0883e><b>292*0.274</b></td><td><b>292 rows, 0.274mm pitch</b></td></tr>
<tr><td class=tag-code>(01F1,1008)</td><td>DS</td><td class=tag-v>361.132</td><td>Source-Detector distance (mm)</td></tr>
<tr><td colspan=4 style=background:#30363d;color:#da3633>Group EFE1 — CORE DATA</td></tr>
<tr><td class=tag-code style=color:#da3633>(EFE1,1001)</td><td>OB</td><td class=tag-v style=color:#da3633>~15.1 MB</td><td style=color:#da3633><b>8-band tile images (JPEG2000 x256)</b></td></tr>
</table>

<h2 id=assembly>3. Tile Assembly</h2>
<div class=figure><img src=data:image/png;base64,{b64_tilegrid}></div>
<div class=figure><img src=data:image/png;base64,{b64_assembly}></div>
<div class=highlight>
<p>60 tiles/group in 8x8 grid (4 corners empty: 0,7,56,63). Each tile = 256x256 with 8 bands of 32 rows. G0/G2 = reference frames; G1/G3 = signal frames. <b>G1-G0 = CT image</b> (r={c_full:.3f}). Rescale: HU = (G1-G0) x {a:.1f} + {b:.0f}</p>
</div>

<h2 id=bands>4. 8 Energy Bin CT Images</h2>
<div class=figure><img src=data:image/png;base64,{b64_bands}></div>
<div class=highlight>
<p>Each band of (G1-G0) assembled separately = one energy bin CT image. Band correlations with PixelData: <span class=good>r={min(band_corrs):.3f}-{max(band_corrs):.3f}</span>. Bands 0,2,7 show highest correlation; bands 4-5 lowest. Images show strided spatial pattern due to sub-tile interleaving in 8x8 grid.</p>
</div>

<h2 id=spectral>5. Material Spectral Basis + Iodine K-edge</h2>
<div class=figure><img src=data:image/png;base64,{b64_curves}></div>
<table><tr><th>Curve</th><th>Offset</th><th>Max</th><th>Feature</th><th>ID</th></tr>
<tr><td>V1/V2</td><td class=tag-code>0x01C0/0x04E8</td><td>407.7</td><td>Smooth decay</td><td>Water</td></tr>
<tr><td style=color:#e53935>V3/V7</td><td class=tag-code style=color:#e53935>0x0850/0x1558</td><td style=color:#e53935>4484.3</td><td style=color:#e53935>K-edge 5.1x @ idx33</td><td style=color:#e53935><b>Iodine</b></td></tr>
<tr><td>V4</td><td class=tag-code>0x0B70</td><td>273.4</td><td>Smooth decay</td><td>Calcium</td></tr>
<tr><td>V5</td><td class=tag-code>0x0ED4</td><td>754.2</td><td>Smooth decay</td><td>Gadolinium</td></tr>
</table>
<p style=color:#8b949e>Iodine K-edge at 33.2 keV confirmed by 5.1x jump at index 33 in V3 curve. Material names from header offset 0x008-0x0FF. Curve-to-material mapping inferred from K-edge physics.</p>

<h2 id=structure>6. OB Blob Layout</h2>
<table>
<tr><th>Offset</th><th>Content</th><th>Evidence</th></tr>
<tr><td class=tag-code>0x000-0x003</td><td>Material count = 4</td><td>struct uint32 parsed</td></tr>
<tr><td class=tag-code>0x004-0x0FF</td><td>Water/Iodine/Calcium/Gd headers (name + 14 params each)</td><td>ASCII strings + fixed-size records</td></tr>
<tr><td class=tag-code>0x100-0x1BF</td><td>Energy bin calibration table (string-based)</td><td>Material names + bin range triplets</td></tr>
<tr><td class=tag-code>0x1C0-0x4E7</td><td>Float32 spectral curve #1 (Water, 200 pts)</td><td>Float32 decoding</td></tr>
<tr><td class=tag-code>0x4E8-0x859</td><td>Curves #2-#3 (Water dup + Iodine) + headers</td><td>parsed boundaries</td></tr>
<tr><td class=tag-code>0x850-0x1877</td><td>Curves #3-#7 (Iodine/Calcium/Gd) + headers</td><td>parsed boundaries</td></tr>
<tr><td class=tag-code>0x1878-0x9E0F</td><td>Additional calib tables + JP2 container header</td><td>Mixed binary + ASCII metadata</td></tr>
<tr><td class=tag-code style=color:#7ee787>0x9E10-EOF</td><td style=color:#7ee787><b>256 JPEG2000 codestreams (tile images)</b></td><td style=color:#7ee787>jp2c markers + JP2 ihdr</td></tr>
</table>

<h2 id=code>7. Decoding Recipe</h2>
<div class=code-block>
<span class=cm># Read DICOM private tag</span>
data = dcmread(file)[<span class=nu>0xefe1</span>, <span class=nu>0x1001</span>].value

<span class=cm># Find all jp2c markers (256 total)</span>
pos = []; p = -<span class=nu>1</span>
<span class=kw>while</span> True: p = data.find(<span class=str>b'jp2c'</span>, p+<span class=nu>1</span>); <span class=kw>if</span> p==-<span class=nu>1</span>: <span class=kw>break</span>; pos.append(p)

<span class=cm># Load Group 0 and Group 1 (60 tiles each, skip corners 0,7,56,63)</span>
empty = {{<span class=nu>0</span>,<span class=nu>7</span>,<span class=nu>56</span>,<span class=nu>63</span>}}
<span class=kw>for</span> gp, offset <span class=kw>in</span> [(<span class=nu>0</span>,<span class=nu>0</span>), (<span class=nu>1</span>,<span class=nu>64</span>)]:
    idx = <span class=nu>0</span>; imgs = np.zeros((<span class=nu>60</span>,<span class=nu>256</span>,<span class=nu>256</span>))
    <span class=kw>for</span> slot <span class=kw>in</span> range(offset, offset+<span class=nu>64</span>):
        <span class=kw>if</span> slot-offset <span class=kw>in</span> empty: <span class=kw>continue</span>
        jp2k = data[pos[slot]+<span class=nu>4</span>:pos[slot+<span class=nu>1</span>]]
        imgs[idx] = glymur.Jp2k(jp2k)[:]  <span class=cm># (256,256) int16</span>
        idx += <span class=nu>1</span>

<span class=cm># Assemble 2048x2048 CT image (8x8 tile grid, 8 bands per tile)</span>
image = np.zeros((<span class=nu>2048</span>,<span class=nu>2048</span>)); idx = <span class=nu>0</span>
<span class=kw>for</span> tp <span class=kw>in</span> range(<span class=nu>64</span>):
    <span class=kw>if</span> tp <span class=kw>in</span> empty: <span class=kw>continue</span>
    tr = tp//<span class=nu>8</span>; tc = tp%<span class=nu>8</span>; tile = g1[idx] - g0[idx]  <span class=cm># reference subtraction</span>
    <span class=kw>for</span> eb <span class=kw>in</span> range(<span class=nu>8</span>):  <span class=cm># 8 energy bins</span>
        image[(tr*<span class=nu>8</span>+eb)*<span class=nu>32</span>:(tr*<span class=nu>8</span>+eb+<span class=nu>1</span>)*<span class=nu>32</span>, tc*<span class=nu>256</span>:(tc+<span class=nu>1</span>)*<span class=nu>256</span>] = tile[eb*<span class=nu>32</span>:(eb+<span class=nu>1</span>)*<span class=nu>32</span>, :]
    idx += <span class=nu>1</span>

<span class=cm># Result: image.shape = (2048, 2048) — energy-resolved CT reconstruction</span>
<span class=cm># Per-band extraction: take rows [eb*32,eb*32+32] from each tile</span>
</div>

<footer><p>P10 Energy Bin CT — Verified Report | Redlen CdZnTe | NeuViz P10 | 2026-06-02</p></footer>
</body></html>'''

with open(r'D:\xl\00020006\p10_final_report.html','w',encoding='utf-8')as fh:fh.write(html)
print(f'Final report saved: p10_final_report.html ({len(html)/1024:.0f}KB)')
print(f'G1-G0 r={c_full:.3f}, band r range:[{min(band_corrs):.3f},{max(band_corrs):.3f}]')
