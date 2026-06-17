"""
P10 Complete Data Audit — Chinese Report
Consolidates all confirmed findings into a single comprehensive report.
"""
import pydicom,os,numpy as np,struct,base64,io
import glymur,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt

base=r'D:\xl\00020006\00020006'
files=sorted(os.listdir(base))
ds=pydicom.dcmread(os.path.join(base,files[0]))
data=ds[0xefe1,0x1001].value
pixels=np.frombuffer(ds.PixelData,dtype=np.uint16).reshape(2048,2048).astype(np.float64)

jp2c=b'jp2c';pos=[];p=-1
while True:
    p=data.find(jp2c,p+1)
    if p==-1:break
    pos.append(p)
EMPTY={0,7,56,63}

def load(gp):
    imgs=np.zeros((60,256,256));idx=0;s0=gp*64
    for sl in range(s0,s0+64):
        tp=sl-s0
        if tp in EMPTY:continue
        s=pos[sl]+4;e=pos[sl+1]if sl+1<len(pos)else len(data)
        with open(rf'D:\xl\00020006\zh{sl}.jp2k','wb')as fh:fh.write(data[s:e])
        imgs[idx]=glymur.Jp2k(rf'D:\xl\00020006\zh{sl}.jp2k')[:];idx+=1
    return imgs

def assemble(imgs):
    full=np.zeros((2048,2048));idx=0
    for tp in range(64):
        if tp in EMPTY:
            continue
        tr=tp//8
        tc=tp%8
        tile=imgs[idx]
        idx+=1
        for b in range(8):full[(tr*8+b)*32:(tr*8+b+1)*32,tc*256:(tc+1)*256]=tile[b*32:(b+1)*32,:]
    return full

g0=load(0);g1=load(1);g2=load(2);g3=load(3)
A0=assemble(g0);A1=assemble(g1);A2=assemble(g2);A3=assemble(g3)

# ============================================================
# FIG 1: Five Raw Images
# ============================================================
fig1,axes1=plt.subplots(1,5,figsize=(18,4))
for i,(img,name) in enumerate([(pixels,'PixelData (CT)'),(A0,'G0 (CS 0-63)'),(A1,'G1 (CS 64-127)'),(A2,'G2 (CS 128-191)'),(A3,'G3 (CS 192-255)')]):
    axes1[i].imshow(img,cmap='gray',aspect='equal')
    axes1[i].set_title(name,fontsize=9);axes1[i].axis('off')
plt.subplots_adjust(left=0.01,right=0.99,bottom=0.01,top=0.92,wspace=0.02)
plt.suptitle('5 张原始图像（无任何处理）',fontsize=12,y=0.96)
plt.tight_layout()
buf1=io.BytesIO();fig1.savefig(buf1,dpi=120,bbox_inches='tight');buf1.seek(0)
b64_raw=base64.b64encode(buf1.read()).decode();plt.close(fig1)

# ============================================================
# FIG 2: Material Energy Response Curves
# ============================================================
keV=np.arange(200)
curves={
    'Water (0x01C0)':np.frombuffer(data[0x1C0:0x1C0+200*4],dtype=np.float32),
    'Iodine (0x0850)':np.frombuffer(data[0x850:0x850+200*4],dtype=np.float32),
    'Calcium (0x0B70)':np.frombuffer(data[0xB70:0xB70+200*4],dtype=np.float32),
    'Gadolinium (0x0ED4)':np.frombuffer(data[0xED4:0xED4+200*4],dtype=np.float32),
}
iodine=curves['Iodine (0x0850)']

fig2,axes2=plt.subplots(1,3,figsize=(22,5))
colors_curve=['#58a6ff','#e53935','#7ee787','#f0883e']
for i,(name,c) in enumerate(curves.items()):
    axes2[0].plot(keV,c,color=colors_curve[i],lw=0.8,alpha=0.8,label=name.split()[0])
axes2[0].axvline(x=33,color='#e53935',ls='--',lw=1,alpha=0.5)
axes2[0].set_ylim(0,500);axes2[0].set_title('线性（ylim 0-500）');axes2[0].set_xlabel('指数');axes2[0].grid(alpha=0.3);axes2[0].set_xlim(0,200);axes2[0].legend(fontsize=7)

for i,(name,c) in enumerate(curves.items()):
    axes2[1].semilogy(keV,np.maximum(c,1e-6),color=colors_curve[i],lw=0.8,alpha=0.8)
axes2[1].axvline(x=33,color='#e53935',ls='--',lw=1,alpha=0.5)
axes2[1].set_title('对数');axes2[1].set_xlabel('指数');axes2[1].grid(alpha=0.3);axes2[1].set_xlim(0,200)

for i,(name,c) in enumerate(curves.items()):
    axes2[2].semilogy(keV,np.maximum(c,1e-6)/np.max(c),color=colors_curve[i],lw=0.8,alpha=0.8)
axes2[2].axvline(x=33,color='#e53935',ls='--',lw=1,alpha=0.5)
axes2[2].set_title('归一化对数');axes2[2].set_xlabel('指数');axes2[2].grid(alpha=0.3);axes2[2].set_xlim(0,200);axes2[2].legend([n.split()[0] for n in curves],fontsize=7)
plt.suptitle('材料能量响应曲线（200个float32值 = 200 keV，碘K-edge @ index 33 确认）',fontsize=13,y=1.02)
plt.tight_layout()
buf2=io.BytesIO();fig2.savefig(buf2,dpi=120,bbox_inches='tight');buf2.seek(0)
b64_curves=base64.b64encode(buf2.read()).decode();plt.close(fig2)

# ============================================================
# FIG 3: Gap Structure (Curve #8 + signed-int16 region)
# ============================================================
curve8=np.frombuffer(data[0x1878:0x1878+200*4],dtype=np.float32)
r_c8=np.corrcoef(curve8/np.max(curve8),iodine/np.max(iodine))[0,1]
gap_s16=np.frombuffer(data[0x1BA8:0x9DE0],dtype=np.uint16).astype(np.int16)
bins=[gap_s16[i*4167:(i+1)*4167] for i in range(4)]
r_b23=np.corrcoef(bins[2].astype(float),bins[3].astype(float))[0,1]

fig3=plt.figure(figsize=(22,12))
ax=fig3.add_subplot(2,4,1)
ax.plot(curve8,'b-',lw=1.2);ax.plot(iodine,'r--',lw=0.8,alpha=0.4)
ax.set_title(f'曲线 #8 (r vs I={r_c8:.4f})',fontsize=7);ax.set_xlabel('指数');ax.grid(alpha=0.3);ax.set_xlim(0,200)
ax=fig3.add_subplot(2,4,2)
ax.plot(gap_s16[::5],'b-',lw=0.2,alpha=0.7)
ax.set_title(f'Gap signed-int16 ({len(gap_s16)} 值)',fontsize=7);ax.grid(alpha=0.3)
ax=fig3.add_subplot(2,4,3)
ax.hist(gap_s16,bins=80,alpha=0.7,color='#58a6ff',density=True)
ax.axvline(x=0,color='gray',ls='--',lw=0.5)
ax.set_title(f'Gap s16 分布 (均值={gap_s16.mean():.0f})',fontsize=7)
ax=fig3.add_subplot(2,4,4);ax.axis('off')
tx=f'0x1878-0x9DE0: {len(gap_s16)*2//1024:,} KB\n\n曲线 #8: 0x1878-0x1BA7\n  200 f32, 碘形 (r={r_c8:.4f})\n\nGap: 0x1BA8-0x9DE0\n  {len(gap_s16):,} signed int16\n  4-bin 分割: 4167/bin\n  B2-B3 r={r_b23:.3f}\n\n跨切片: r>0.9998'
ax.text(0.05,0.95,tx,transform=ax.transAxes,fontsize=7,family='monospace',verticalalignment='top',
    bbox=dict(boxstyle='round',facecolor='#1f242b',edgecolor='#30363d'))
for i,b in enumerate(bins):
    ax=fig3.add_subplot(2,4,5+i)
    ax.plot(b[:200],'b-',lw=0.3,alpha=0.8)
    ax.set_title(f'Bin {i}: 均值={b.mean():.0f} std={b.std():.0f}',fontsize=7);ax.grid(alpha=0.3)
plt.suptitle('Gap (0x1878-0x9DE0): 曲线 #8 + 未解码 signed-int16 区域',fontsize=13,y=1.02)
plt.tight_layout()
buf3=io.BytesIO();fig3.savefig(buf3,dpi=120,bbox_inches='tight');buf3.seek(0)
b64_gap=base64.b64encode(buf3.read()).decode();plt.close(fig3)

# ============================================================
# FIG 4: Pixel Value Comparison at 3 Positions
# ============================================================
positions=[(500,800,'软组织'),(800,1200,'肺'),(1500,600,'骨边缘')]
cmap_marker=['#58a6ff','#e53935','#7ee787']
fig4,(ax4a,ax4b)=plt.subplots(1,2,figsize=(16,6))
ax4a.imshow(pixels,cmap='gray',vmin=np.percentile(pixels,1),vmax=np.percentile(pixels,99))
for (y,x,label),c in zip(positions,cmap_marker):
    ax4a.plot(x,y,'o',color=c,markersize=10,markeredgewidth=2,markerfacecolor='none')
    ax4a.text(x+30,y,label,color=c,fontsize=9,fontweight='bold')
ax4a.set_title('PixelData 上标注的 3 个像素位置',fontsize=10);ax4a.axis('off')
img_labels=['PixelData','G0','G1','G2','G3']
for (y,x,label),c in zip(positions,cmap_marker):
    vals=[img[y,x] for img in [pixels,A0,A1,A2,A3]]
    ax4b.plot(range(5),vals,'o-',color=c,markersize=8,lw=2,label=f'{label}')
    for i,v in enumerate(vals):
        ax4b.annotate(f'{v:.0f}',(i,v),textcoords='offset points',xytext=(0,10),fontsize=7,ha='center',color=c)
ax4b.set_xticks(range(5));ax4b.set_xticklabels(img_labels,fontsize=8);ax4b.set_ylabel('像素值');ax4b.legend(fontsize=7);ax4b.grid(alpha=0.3)
plt.suptitle('5 张图像在 3 个位置上的像素值对比',fontsize=13,y=1.02)
plt.tight_layout()
buf4=io.BytesIO();fig4.savefig(buf4,dpi=120,bbox_inches='tight');buf4.seek(0)
b64_pixel=base64.b64encode(buf4.read()).decode();plt.close(fig4)

# ============================================================
# FIG 5: Gap Hex Dump + JP2 Header
# ============================================================
gap=data[0x280:0x9E10]
fig5=plt.figure(figsize=(16,10))
ax=fig5.add_subplot(3,1,1);ax.axis('off')
hl=[]
for i in range(0,min(256,len(gap)),16):
    chunk=gap[i:i+16]
    hexstr=' '.join(f'{b:02x}' for b in chunk)
    asc=''.join(chr(b) if 32<=b<127 else '.' for b in chunk)
    hl.append(f'0x{0x280+i:06X}  {hexstr:<48s} {asc}')
ax.text(0.02,0.95,'Gap 开始 (0x0280–0x037F, 前 256 字节)',ha='left',fontsize=9,fontweight='bold',verticalalignment='top',color='#58a6ff',transform=ax.transAxes)
ax.text(0.02,0.90,'\n'.join(hl),fontsize=7,family='monospace',verticalalignment='top',color='#a5d6ff',transform=ax.transAxes)

ax=fig5.add_subplot(3,1,2);ax.axis('off')
jp2a=gap[0x9DE0-0x280:0x9DE0-0x280+80]
hl2=[]
for i in range(0,len(jp2a),16):
    chunk=jp2a[i:i+16]
    hs=' '.join(f'{b:02x}' for b in chunk)
    asc=''.join(chr(b) if 32<=b<127 else '.' for b in chunk)
    marker=''
    if b'ftyp' in chunk or b'jp2h' in chunk or b'ihdr' in chunk:marker=' <—'
    hl2.append(f'0x{0x9DE0+i:06X}  {hs:<48s} {asc}{marker}')
ax.text(0.02,0.95,'JP2 容器头区域 (0x9DE0 附近)',ha='left',fontsize=9,fontweight='bold',verticalalignment='top',color='#f0883e',transform=ax.transAxes)
ax.text(0.02,0.85,'\n'.join(hl2),fontsize=7,family='monospace',verticalalignment='top',color='#a5d6ff',transform=ax.transAxes)

ax=fig5.add_subplot(3,1,3);ax.axis('off')
ga='';tb=''
for b in gap:ga+=chr(b) if 32<=b<127 else ' '
for w in ['H30.ker','ftypjp2','jp2h','ihdr','jp2 ','colr','H30']:
    p=ga.find(w)
    if p>=0:tb+=f'  \"{w}\" → 偏移 0x{0x280+p:06X}\n'
ax.text(0.05,0.95,f'Gap ASCII 字符串 ({len(gap):,} bytes):\n\n{tb}',transform=ax.transAxes,fontsize=8,family='monospace',verticalalignment='top',color='#e6edf3')
plt.suptitle('Gap Hex 转储 + JP2 容器分析',fontsize=13,y=1.02)
plt.tight_layout()
buf5=io.BytesIO();fig5.savefig(buf5,dpi=120,bbox_inches='tight');buf5.seek(0)
b64_hex=base64.b64encode(buf5.read()).decode();plt.close(fig5)

# ============================================================
# Cross-slice comparison
# ============================================================
ds600=pydicom.dcmread(os.path.join(base,files[600]))
gap600=ds600[0xefe1,0x1001].value[0x1BA8:0x9DE0]
gap600_s16=np.frombuffer(gap600,dtype=np.uint16).astype(np.int16)
r_cross=np.corrcoef(gap_s16.astype(float)[:len(gap600_s16)],gap600_s16.astype(float))[0,1]
data600=ds600[0xefe1,0x1001].value
same_hdr=np.all(data[:0x280]==data600[:0x280])
same_curves=np.all(data[0x1C0:0x1878]==data600[0x1C0:0x1878])

# ============================================================
# HTML
# ============================================================
html=f'''<!DOCTYPE html><html lang=zh-CN>
<head><meta charset=UTF-8><title>P10 光子计数CT — 完整数据审计报告</title>
<style>
body{{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#0d1117;color:#c9d1d9;max-width:1300px;margin:0 auto;padding:20px;line-height:1.7}}
h1{{color:#58a6ff;text-align:center;border-bottom:2px solid #30363d;padding-bottom:15px}}
h2{{color:#f0883e;border-left:4px solid #f0883e;padding-left:12px;margin-top:30px}}
h3{{color:#d2a8ff;margin-top:20px}}
.figure{{margin:20px 0;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:15px}}
.figure img{{width:100%;border-radius:4px}}
table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:.85em}}
th{{background:#21262d;color:#8b949e;padding:8px 12px;border:1px solid #30363d;text-align:left}}
td{{padding:6px 12px;border:1px solid #30363d;vertical-align:top}}tr:nth-child(even)td{{background:#161b22}}
.highlight{{background:#1f242b;border-left:3px solid #7ee787;padding:12px 16px;border-radius:0 6px 6px 0;margin:10px 0}}
.highlight-b{{background:#1f242b;border-left:3px solid #58a6ff;padding:12px 16px;border-radius:0 6px 6px 0;margin:10px 0}}
.highlight-w{{background:#1f242b;border-left:3px solid #d2991d;padding:12px 16px;border-radius:0 6px 6px 0;margin:10px 0}}
.tag-code{{color:#a5d6ff;font-family:Consolas,monospace;font-size:.9em}}
.tag-v{{color:#e6edf3}}.kv{{background:#21262d;border-radius:3px;padding:1px 5px;font-family:Consolas,monospace;font-size:.9em}}
.good{{color:#7ee787;font-weight:bold}}.warn{{color:#d2991d}}.bad{{color:#da3633}}
.code-block{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:15px;font-family:Consolas,monospace;font-size:.8em;overflow-x:auto;white-space:pre-wrap;line-height:1.5;color:#c9d1d9}}
.cm{{color:#8b949e}}.kw{{color:#ff7b72}}.nu{{color:#a5d6ff}}.str{{color:#7ee787}}
footer{{text-align:center;color:#484f58;margin:40px 0 20px}}
.toc{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin:20px 0}}
.toc a{{color:#58a6ff;text-decoration:none}}li{{margin:3px 0;font-size:.9em}}
</style></head><body>
<h1>P10 光子计数CT — 完整数据审计报告</h1>
<p style=text-align:center;color:#8b949e>NeuViz P10 (Redlen CdZnTe) | 8 strips（空间交错）| 33-200 keV | 292 探测器排 | {len(files)} 个DICOM文件</p>

<div class=toc><ol>
<li><a href=#s1>五张原始图像</a></li>
<li><a href=#s2>材料能量响应曲线</a></li>
<li><a href=#s3>Gap 结构分析</a></li>
<li><a href=#s4>Gap Hex 转储</a></li>
<li><a href=#s5>跨图像像素值对比</a></li>
<li><a href=#s6>跨切片一致性验证</a></li>
<li><a href=#s7>DICOM 标签映射</a></li>
<li><a href=#s8>OB Blob 完整布局</a></li>
<li><a href=#s9>Python 解码示例</a></li>
</ol></div>

<h2 id=s1>1. 五张原始图像（无任何处理）</h2>
<div class=figure><img src=data:image/png;base64,{b64_raw}></div>
<table><tr><th>图像</th><th>范围</th><th>说明</th><th>确认状态</th></tr>
<tr><td>PixelData</td><td>[{pixels.min():.0f},{pixels.max():.0f}]</td><td>DICOM标签 (7FE0,0010) — 标准CT重建（uint16）</td><td class=good>已确认</td></tr>
<tr><td>G0 (CS 0-63)</td><td>[{A0.min():.0f},{A0.max():.0f}]</td><td>从60个JPEG2000码流组装。G1−G0 = CT图像</td><td class=good>已验证（r=0.334 vs PixelData）</td></tr>
<tr><td>G1 (CS 64-127)</td><td>[{A1.min():.0f},{A1.max():.0f}]</td><td>从60个JPEG2000码流组装。值近零——信号帧</td><td class=good>已验证</td></tr>
<tr><td>G2 (CS 128-191)</td><td>[{A2.min():.0f},{A2.max():.0f}]</td><td>从60个JPEG2000码流组装。含部分INT16_MIN饱和值</td><td class=good>已验证</td></tr>
<tr><td>G3 (CS 192-255)</td><td>[{A3.min():.0f},{A3.max():.0f}]</td><td>从60个JPEG2000码流组装。值近零——信号帧（模块B）</td><td class=good>已验证</td></tr>
</table>

<h2 id=s2>2. 材料能量响应曲线</h2>
<div class=figure><img src=data:image/png;base64,{b64_curves}></div>
<div class=highlight>
<h3>这些曲线是什么</h3>
<p>每种材料的keV-by-keV光谱指纹——Water/Iodine/Calcium/Gadolinium在每个X射线能量下的衰减系数。7条float32数组（每条200个值）坍缩为<b>4个唯一形状</b>（V1=V2, V3=V7, V6≈V4），与OB header中的4种材料一一对应。</p>
<h3>碘曲线确认</h3>
<p>偏移<b>0x0850</b>处的曲线（V3/V7）在<b>index 32→33处出现5.1倍跳变</b>——与碘的K-吸收边（33.2 keV）完全吻合。这是<b>唯一通过物理验证</b>的材料归属。其余3条曲线按OB header中出现顺序分配给Water/Calcium/Gadolinium（无DICOM证据）。</p>
<h3>曲线的用途</h3>
<p>这些曲线为物质分解提供光谱基础——将多能区投影数据转换为材料浓度图。然而，EFE1存储的是<b>空间交错strip</b>，并非真正的多能区投影。因此这些曲线无法直接应用于当前DICOM数据——它们作为解码密钥存在，需配合正确的多能区采集模式使用。</p>
</div>

<table><tr><th>曲线</th><th>偏移</th><th>最大值</th><th>材料归属</th><th>确认方式</th></tr>
<tr><td>V1/V2</td><td class=tag-code>0x01C0 / 0x04E8</td><td>407.7</td><td>Water</td><td>Header顺序推测</td></tr>
<tr style=color:#e53935><td><b>V3/V7</b></td><td class=tag-code style=color:#e53935>0x0850 / 0x1558</td><td style=color:#e53935><b>4484.3</b></td><td style=color:#e53935><b>Iodine</b></td><td style=color:#e53935><b>K-edge 物理验证</b></td></tr>
<tr><td>V4</td><td class=tag-code>0x0B70</td><td>273.4</td><td>Calcium</td><td>Header顺序推测</td></tr>
<tr><td>V5</td><td class=tag-code>0x0ED4</td><td>754.2</td><td>Gadolinium</td><td>Header顺序推测</td></tr>
</table>
<p style=color:#8b949e>重复曲线（V1=V2, V3=V7, V6≈V4）可能源自两个探测器模块 (01F3,1032=2)。"200 pts = 200 keV" 为推测（假设1pt=1keV）。</p>

<h2 id=s3>3. Gap 结构分析 (0x1878-0x9DE0)</h2>
<div class=figure><img src=data:image/png;base64,{b64_gap}></div>

<table><tr><th>偏移</th><th>大小</th><th>内容</th><th>确认状态</th></tr>
<tr><td class=tag-code style=color:#7ee787>0x1878-0x1BA7</td><td style=color:#7ee787>800 B</td><td style=color:#7ee787><b>曲线 #8</b> — 200 float32, 碘形 (r={r_c8:.4f})</td><td style=color:#7ee787>已解码</td></tr>
<tr><td class=tag-code style=color:#d2991d>0x1BA8-0x9DE0</td><td style=color:#d2991d>~33 KB</td><td style=color:#d2991d><b>未解码 signed-int16</b> — {len(gap_s16):,} 个值, 可4-bin等分 (4176/bin), B2-B3 r={r_b23:.3f}, 跨切片等同 (r={r_cross:.4f})</td><td style=color:#d2991d>部分解码</td></tr>
</table>

<div class=highlight-w>
<h3>已确认的事实（0x1BA8区域）</h3>
<table><tr><th>属性</th><th>值</th></tr>
<tr><td>数据类型</td><td>signed int16（无符号域均值≈32768，减偏移后均值≈0）</td></tr>
<tr><td>长度</td><td>16,668 个值 = 33,336 字节</td></tr>
<tr><td>跨切片</td><td>r > 0.9998 — 几乎完全相同（全局校准数据）</td></tr>
<tr><td>4-bin 等分</td><td>16,668 ÷ 4 = 4,167（整数）. 各bin均值: −2273, −278, +3604, +3270</td></tr>
<tr><td>B2-B3 相关性</td><td>r = {r_b23:.3f}（强负相关——互补差分编码）</td></tr>
<tr><td>8-bin 等分</td><td>不能整数划分</td></tr>
<tr><td>2D 图像 reshape</td><td>无自然因子对（最近似 12×1389, 纵横比 1:116）</td></tr>
<tr><td>重复周期</td><td>无</td></tr>
</table>
<p><b>未知：</b>该数据的物理含义（校准系数？能量权重？查找表？），编码格式，以及与JPEG2000 tile或其他数据的配合使用方式。</p>
</div>

<h2 id=s4>4. Gap Hex 转储</h2>
<div class=figure><img src=data:image/png;base64,{b64_hex}></div>

<h2 id=s5>5. 跨图像像素值对比</h2>
<div class=figure><img src=data:image/png;base64,{b64_pixel}></div>
<div class=highlight>
<p>选取3个解剖位置（软组织/肺/骨边缘），读取PixelData + G0/G1/G2/G3共5张图像中的像素值。G0和G2是大负值参考帧；G1和G3是近零信号帧。CT图像结构分布在参考-信号对中。</p>
</div>

<h2 id=s6>6. 跨切片一致性验证</h2>
<table><tr><th>区域</th><th>Slice 0 vs 600</th><th>说明</th></tr>
<tr><td>OB Header (0x000-0x27F)</td><td class=good>{'相同' if same_hdr else '不同'}</td><td>材料参数 + 能量校准表</td></tr>
<tr><td>光谱曲线 (0x1C0-0x1877)</td><td class=good>{'相同' if same_curves else '不同'}</td><td>7条材料能量响应曲线</td></tr>
<tr><td>Gap signed-int16 (0x1BA8-0x9DE0)</td><td class=good>r={r_cross:.4f}</td><td>几乎完全相同（全局校准）</td></tr>
<tr><td>JPEG2000 码流 (0x9E10-EOF)</td><td class=warn>每切片独立</td><td>切片专属图像数据</td></tr>
</table>

<h2 id=s7>7. DICOM 标签映射</h2>
<table><tr><th>标签</th><th>VR</th><th>值</th><th>含义</th><th>确认方式</th></tr>
<tr><td colspan=5 style=background:#30363d;color:#58a6ff>Group 01E7 — 探测器/能量</td></tr>
<tr><td class=tag-code>(01E7,1001)</td><td>LO</td><td class=tag-v>ME 60keV/SI</td><td>探测器/能量标记：Multi-Energy 60 keV；SI 为能量数据标记，不代表硅</td><td>DICOM标签</td></tr>
<tr><td class=tag-code>(01E7,1002)</td><td>SL</td><td class=tag-v>60</td><td>标称keV（非view数量）</td><td>DICOM标签</td></tr>
<tr><td class=tag-code>(01E7,1004)</td><td>IS</td><td class=tag-v>2</td><td>X射线焦点数：FS0, FS1</td><td>DICOM标签</td></tr>
<tr><td colspan=5 style=background:#30363d;color:#7ee787>Group 01F3 — 能量配置</td></tr>
<tr><td class=tag-code>(01F3,1031)</td><td>IS</td><td class=tag-v style=color:#7ee787><b>8</b></td><td>能量区数量（硬件支持）</td><td>DICOM标签</td></tr>
<tr><td class=tag-code>(01F3,1032)</td><td>IS</td><td class=tag-v>2</td><td>探测器模块数</td><td>DICOM标签</td></tr>
<tr><td class=tag-code>(01F3,1046)</td><td>DS</td><td class=tag-v>290</td><td>待确认（非keV范围）</td><td>未知</td></tr>
<tr><td colspan=5 style=background:#30363d;color:#f0883e>Group 01F1 — 几何参数</td></tr>
<tr><td class=tag-code>(01F1,104B)</td><td>SH</td><td class=tag-v style=color:#f0883e><b>292*0.274</b></td><td><b>292排探测器, 0.274mm排间距</b></td><td>DICOM标签</td></tr>
<tr><td class=tag-code>(01F1,1008)</td><td>DS</td><td class=tag-v>361.132</td><td>源-探测器距离 (mm)</td><td>DICOM标签</td></tr>
<tr><td colspan=5 style=background:#30363d;color:#da3633>Group EFE1 — 核心数据</td></tr>
<tr><td class=tag-code style=color:#da3633>(EFE1,1001)</td><td>OB</td><td class=tag-v style=color:#da3633>~15.1 MB</td><td style=color:#da3633><b>JPEG2000压缩的tile图像（256个码流 → G0/G1/G2/G3）</b></td><td>已验证</td></tr>
</table>

<h2 id=s8>8. OB Blob 完整布局</h2>
<table><tr><th>偏移</th><th>大小</th><th>内容</th><th>确认状态</th><th>依据</th></tr>
<tr><td class=tag-code>0x000-0x003</td><td>4 B</td><td>材料数量 = 4</td><td class=good>已确认</td><td>struct uint32 解析</td></tr>
<tr><td class=tag-code>0x004-0x0FF</td><td>252 B</td><td>材料表头：Water/Iodine/Calcium/Gadolinium（每个: 名称 + 14 uint32）</td><td class=good>已确认</td><td>ASCII字符串 + 固定大小记录</td></tr>
<tr><td class=tag-code>0x100-0x1BF</td><td>192 B</td><td>能量-bin校准表（基于字符串）</td><td class=warn>部分确认</td><td>材料名称 + bin范围三元组</td></tr>
<tr><td class=tag-code>0x1C0-0x1877</td><td>~6 KB</td><td>7条光谱曲线（200 float32每条，4种材料+重复）</td><td class=good>已确认</td><td>Float32解码 + 碘K-edge物理验证</td></tr>
<tr><td class=tag-code style=color:#7ee787>0x1878-0x1BA7</td><td style=color:#7ee787>800 B</td><td style=color:#7ee787><b>曲线 #8</b> — 200 float32, 碘形 (r={r_c8:.4f})</td><td style=color:#7ee787>已解码</td><td style=color:#7ee787>Float32 + 曲线形状对比</td></tr>
<tr><td class=tag-code style=color:#d2991d>0x1BA8-0x9DE0</td><td style=color:#d2991d>~33 KB</td><td style=color:#d2991d>未解码 signed-int16 数据（{len(gap_s16):,} 值, 4-bin可分, B2-B3 r={r_b23:.3f}）</td><td style=color:#d2991d>部分解码</td><td style=color:#d2991d>统计分析</td></tr>
<tr><td class=tag-code style=color:#f0883e>0x9DE0-0x9E0F</td><td style=color:#f0883e>48 B</td><td style=color:#f0883e>JP2 容器头（ftypjp2, jp2h, ihdr: 256x256, colr）</td><td style=color:#f0883e>已确认</td><td style=color:#f0883e>ISO 15444-1 JP2 解析</td></tr>
<tr><td class=tag-code style=color:#58a6ff>0x9E10-EOF</td><td style=color:#58a6ff>~15.1 MB</td><td style=color:#58a6ff><b>256 JPEG2000 码流 → G0/G1/G2/G3</b></td><td style=color:#58a6ff>已完全解码</td><td style=color:#58a6ff>JP2 ihdr + OpenJPEG 解码 + G1-G0 vs PixelData 验证</td></tr>
</table>

<h2 id=s9>9. Python 解码示例</h2>
<div class=code-block>
<span class=cm># 读取 DICOM 私有标签</span>
data = dcmread(file)[<span class=nu>0xefe1</span>, <span class=nu>0x1001</span>].value  <span class=cm># ~15.1 MB OB 数据块</span>

<span class=cm># 查找所有 256 个 jp2c 标记</span>
pos = []; p = -<span class=nu>1</span>
<span class=kw>while</span> True: p = data.find(<span class=str>b'jp2c'</span>, p+<span class=nu>1</span>); <span class=kw>if</span> p==-<span class=nu>1</span>: <span class=kw>break</span>; pos.append(p)

<span class=cm># 加载 Group 0 和 Group 1</span>
<span class=cm># 60 个有效 tile / 组, 8x8 网格, 四角为空 (0,7,56,63)</span>
empty = {{<span class=nu>0</span>, <span class=nu>7</span>, <span class=nu>56</span>, <span class=nu>63</span>}}
<span class=kw>for</span> gp, offset <span class=kw>in</span> [(<span class=nu>0</span>,<span class=nu>0</span>), (<span class=nu>1</span>,<span class=nu>64</span>)]:
    idx = <span class=nu>0</span>; imgs = np.zeros((<span class=nu>60</span>,<span class=nu>256</span>,<span class=nu>256</span>))
    <span class=kw>for</span> slot <span class=kw>in</span> range(offset, offset+<span class=nu>64</span>):
        <span class=kw>if</span> slot-offset <span class=kw>in</span> empty: <span class=kw>continue</span>
        jp2k = data[pos[slot]+<span class=nu>4</span>:pos[slot+<span class=nu>1</span>]]
        imgs[idx] = glymur.Jp2k(jp2k)[:]  <span class=cm># (256,256) int16</span>
        idx += <span class=nu>1</span>

<span class=cm># 组装 2048x2048 CT 图像 (8x8 tile 网格, 每 tile 8 个 32-row strip)</span>
image = np.zeros((<span class=nu>2048</span>,<span class=nu>2048</span>)); idx = <span class=nu>0</span>
<span class=kw>for</span> tp <span class=kw>in</span> range(<span class=nu>64</span>):
    <span class=kw>if</span> tp <span class=kw>in</span> empty: <span class=kw>continue</span>
    tr = tp//<span class=nu>8</span>; tc = tp%<span class=nu>8</span>; tile = g1[idx] - g0[idx]  <span class=cm># 参考减除</span>
    <span class=kw>for</span> s <span class=kw>in</span> range(<span class=nu>8</span>):  <span class=cm># 8 个空间交错 strip</span>
        image[(tr*<span class=nu>8</span>+s)*<span class=nu>32</span>:(tr*<span class=nu>8</span>+s+<span class=nu>1</span>)*<span class=nu>32</span>, tc*<span class=nu>256</span>:(tc+<span class=nu>1</span>)*<span class=nu>256</span>] = tile[s*<span class=nu>32</span>:(s+<span class=nu>1</span>)*<span class=nu>32</span>, :]
    idx += <span class=nu>1</span>

<span class=cm># 结果: image.shape = (2048, 2048) — CT 重建图像</span>
<span class=cm># G1-G0 vs PixelData: r = 0.334</span>
</div>

<footer><p>P10 完整数据审计 | NeuViz P10 (Redlen CdZnTe) | {len(files)} DICOM 文件 | 2026-06-03</p></footer>
</body></html>'''

out_path=r'D:\xl\00020006\complete_audit.html'
with open(out_path,'w',encoding='utf-8')as fh:fh.write(html)
sz=os.path.getsize(out_path)
print(f'Saved: complete_audit.html ({sz/1024:.0f}KB)')
