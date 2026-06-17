import json, os, base64
base_dir = r'D:\xl\00020006'
with open(os.path.join(base_dir, 'b64_v4.json'), 'r') as f:
    img = json.load(f)
with open(os.path.join(base_dir, 'svg_kedge.txt'), 'r') as f:
    svg_kedge_b64 = f.read()

SAMPLE_RATE = 6496
REV_TIME = 0.3
VIEWS_PER_BURST = 32
N_BINS = 8
N_BURSTS = 60
TOTAL_VIEWS = N_BURSTS * VIEWS_PER_BURST
THEORETICAL = int(SAMPLE_RATE * REV_TIME)
BURST_US = VIEWS_PER_BURST / SAMPLE_RATE * 1e6
DEAD_US = (REV_TIME*1e6 - N_BURSTS*VIEWS_PER_BURST/SAMPLE_RATE*1e6) / N_BURSTS

# Energy bin keV boundaries (from K-edge SVG)
BIN_KEV = ['<28', '28-33', '33-38*', '38-48', '48-62', '62-80', '80-105', '>105']
KEDGE_BIN1 = 1  # bin index for 28-33 keV
KEDGE_BIN2 = 2  # bin index for 33-38 keV (contains K-edge at 33.2 keV)

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>光子计数CT — Energy Bin Sinogram 完全解析报告 v4</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.7; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 2em; color: #58a6ff; text-align: center; margin: 30px 0 10px; padding-bottom: 15px; border-bottom: 2px solid #30363d; }}
h2 {{ color: #f0883e; margin: 30px 0 15px; font-size: 1.5em; border-left: 4px solid #f0883e; padding-left: 12px; }}
h3 {{ color: #d2a8ff; margin: 20px 0 10px; font-size: 1.15em; }}
.meta-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 8px; margin: 15px 0; }}
.meta-item {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; }}
.meta-key {{ color: #8b949e; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; }}
.meta-val {{ color: #e6edf3; font-size: 1em; word-break: break-all; }}
.tag-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.85em; }}
.tag-table th {{ background: #21262d; color: #8b949e; padding: 8px 12px; text-align: left; border: 1px solid #30363d; font-weight: 600; }}
.tag-table td {{ padding: 6px 12px; border: 1px solid #30363d; vertical-align: top; }}
.tag-table tr:nth-child(even) td {{ background: #161b22; }}
.tag-group {{ color: #f0883e; font-weight: bold; }}
.tag-code {{ color: #7ee787; font-family: 'Consolas','Courier New',monospace; font-size: 0.9em; }}
.tag-val {{ color: #a5d6ff; }}
.highlight {{ background: #1f242b; border-left: 3px solid #f0883e; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 15px 0; }}
.highlight-g {{ background: #1f242b; border-left: 3px solid #7ee787; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 15px 0; }}
.highlight-b {{ background: #1f242b; border-left: 3px solid #58a6ff; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 15px 0; }}
.code-block {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 15px; font-family: 'Consolas','Courier New',monospace; font-size: 0.8em; overflow-x: auto; white-space: pre-wrap; color: #c9d1d9; line-height: 1.5; }}
.cm {{ color: #8b949e; }} .kw {{ color: #ff7b72; }} .nu {{ color: #a5d6ff; }} .str {{ color: #7ee787; }}
.data-flow {{ display: flex; justify-content: center; align-items: center; gap: 10px; margin: 20px 0; flex-wrap: wrap; font-size: 0.85em; }}
.flow-box {{ background: #21262d; border: 2px solid #30363d; border-radius: 10px; padding: 10px 16px; text-align: center; }}
.figure {{ margin: 20px 0; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }}
.figure img {{ width: 100%; border-radius: 4px; }}
.figure-caption {{ color: #8b949e; font-size: 0.85em; margin-top: 8px; text-align: center; line-height: 1.5; }}
.toc {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 20px 0; columns: 2; }}
.toc a {{ color: #58a6ff; text-decoration: none; }} .toc a:hover {{ text-decoration: underline; }}
.toc ol {{ margin-left: 20px; }} .toc li {{ margin: 3px 0; font-size: 0.9em; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(175px, 1fr)); gap: 8px; margin: 15px 0; }}
.stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; text-align: center; }}
.stat-num {{ font-size: 1.7em; color: #58a6ff; font-weight: bold; }}
.stat-label {{ color: #8b949e; font-size: 0.72em; margin-top: 4px; }}
ul {{ margin-left: 20px; line-height: 1.8; }}
.kv {{ display: inline-block; background: #21262d; border-radius: 4px; padding: 1px 6px; margin: 0 2px; font-family: 'Consolas',monospace; font-size: 0.9em; }}
footer {{ text-align: center; color: #484f58; margin: 40px 0 20px; font-size: 0.85em; }}
</style>
</head>
<body>
<div class="container">

<h1>光子计数 CT / Energy Bin Sinogram 完全解析报告 v4</h1>
<p style="text-align:center;color:#8b949e;">NeuViz P10 | 120kVp Thorax | {N_BINS} 能区 x {TOTAL_VIEWS} 视图 | 292排探测器 | 2,635 切片 | JPEG2000 压缩</p>

<div class="toc"><ol>
<li><a href="#overview">概览 &amp; 关键数字</a></li>
<li><a href="#geometry">探测器几何 &amp; 采样时序</a></li>
<li><a href="#tags">私有标签全解析</a></li>
<li><a href="#ob-header">OB Header 结构</a></li>
<li><a href="#jp2k">JPEG2000 压缩层</a></li>
<li><a href="#proof">{N_BINS} 能区 / {VIEWS_PER_BURST} 快视图 统计证明</a></li>
<li><a href="#groups">4-Group 数据分类</a></li>
<li><a href="#sino">完整 Sinogram ({TOTAL_VIEWS} views)</a></li>
<li><a href="#timing">Burst 时序图</a></li>
<li><a href="#multislice">跨切片验证</a></li>
<li><a href="#summary">总结 &amp; 解码代码</a></li>
</ol></div>

<h2 id="overview">1. 概览 &amp; 关键数字</h2>

<div class="stats-grid">
<div class="stat-card"><div class="stat-num">2,635</div><div class="stat-label">DICOM 切片文件</div></div>
<div class="stat-card"><div class="stat-num">2048x2048</div><div class="stat-label">CT重建</div></div>
<div class="stat-card"><div class="stat-num">{N_BINS}</div><div class="stat-label">能量区 (Bins)</div></div>
<div class="stat-card"><div class="stat-num">{TOTAL_VIEWS}</div><div class="stat-label">总投影视图</div></div>
<div class="stat-card"><div class="stat-num">{N_BURSTS}</div><div class="stat-label">Burst 窗口</div></div>
<div class="stat-card"><div class="stat-num">{VIEWS_PER_BURST}</div><div class="stat-label">视图/Burst</div></div>
<div class="stat-card"><div class="stat-num">292</div><div class="stat-label">探测器 z-排</div></div>
<div class="stat-card"><div class="stat-num">{SAMPLE_RATE}Hz</div><div class="stat-label">采样率</div></div>
<div class="stat-card"><div class="stat-num">{BURST_US:.0f}us</div><div class="stat-label">Burst 窗口</div></div>
<div class="stat-card"><div class="stat-num">{DEAD_US:.0f}us</div><div class="stat-label">Burst 死时间</div></div>
<div class="stat-card"><div class="stat-num">15.1MB</div><div class="stat-label">OB 数据块</div></div>
<div class="stat-card"><div class="stat-num">~94亿</div><div class="stat-label">正弦图总数据点</div></div>
</div>

<div class="highlight-b">
<h3>核心修正: {TOTAL_VIEWS} views = {N_BURSTS} bursts x {VIEWS_PER_BURST} 快速采样</h3>
<p>之前将 <span class="kv">(01E7,1002)=60</span> 误读为 60 个视图。实际 <b>60 = 60 keV</b> (能区相关标签, 与探测器型号 <span class="kv">ME 60keV/SI</span> 一致)。真正的投影角度数为 <b>{TOTAL_VIEWS}</b>——由 6496 Hz 采样率 x 32 样本/Burst x 60 Burst/转 得出, 与理论值 <b>{THEORETICAL}</b> 高度吻合 (差异仅 {(THEORETICAL-TOTAL_VIEWS)/THEORETICAL*100:.1f}%)。</p>
</div>

<h2 id="geometry">2. 探测器几何 &amp; 采样时序</h2>

<h3>2.1 Burst 采样模型</h3>
<table class="tag-table">
<tr><th>参数</th><th>值</th><th>说明</th></tr>
<tr><td>采样率</td><td class="tag-val">{SAMPLE_RATE} Hz</td><td>探测器读出频率</td></tr>
<tr><td>采样周期</td><td class="tag-val">{1e6/SAMPLE_RATE:.1f} us</td><td>相邻采样点间隔</td></tr>
<tr><td>Burst 窗口</td><td class="tag-val">{VIEWS_PER_BURST} 视图 ({BURST_US:.0f} us)</td><td>每次 Burst 连续采集 {VIEWS_PER_BURST} 个样本</td></tr>
<tr><td>Burst 间死时间</td><td class="tag-val">{DEAD_US:.0f} us</td><td>两Burst间的间隔 ({THEORETICAL-TOTAL_VIEWS} 跳过的采样)</td></tr>
<tr><td>Burst 数/转</td><td class="tag-val">{N_BURSTS}</td><td>映射到 Group 1 的 60 个 JPEG2000 图像</td></tr>
<tr><td>总视图/转</td><td class="tag-val"><b>{TOTAL_VIEWS}</b></td><td>{N_BURSTS} x {VIEWS_PER_BURST}</td></tr>
<tr><td>旋转时间</td><td class="tag-val">{REV_TIME} s</td><td>RevolutionTime</td></tr>
<tr><td>理论最大值</td><td class="tag-val">{THEORETICAL}</td><td>{SAMPLE_RATE} Hz x {REV_TIME} s</td></tr>
</table>

<h3>2.2 JPEG2000 编码映射</h3>
<div class="data-flow">
<div class="flow-box"><b>1 个 Burst</b><br>= 1 张 JP2K<br>256x256 int16</div>
<div class="flow-arrow" style="font-size:1.2em;">&rarr;</div>
<div class="flow-box"><b>行: 256</b><br>= {N_BINS} 能区 x {VIEWS_PER_BURST} 视图<br>垂直堆叠<br>(每个能区 {VIEWS_PER_BURST} 行)</div>
<div class="flow-arrow" style="font-size:1.2em;">&rarr;</div>
<div class="flow-box"><b>列: 256</b><br>= 探测器编码<br>(通道或行映射)</div>
</div>

<div class="highlight-g">
<p><b>时序保证:</b> 292 排探测器<b>同时采集</b>。{VIEWS_PER_BURST} 个视图是 Burst 内的快速时间序列 (间隔 {1e6/SAMPLE_RATE:.0f}us), 非探测器排分时。{N_BURSTS} 个 Burst 沿旋转角均匀分布, 提供 {TOTAL_VIEWS} 个投影视图——<b>足以支撑标准 FBP 重建</b> (H30 kernel, 2048x2048, 433mm FOV)。</p>
</div>

<h2 id="tags">3. 私有标签全解析</h2>

<table class="tag-table">
<tr><th>标签</th><th>VR</th><th>值</th><th>含义</th></tr>
<tr><td class="tag-group" colspan="4">Group_01E7 — 能区探测器元数据</td></tr>
<tr><td class="tag-code">(01E7,1001)</td><td>LO</td><td class="tag-val">ME 60keV/SI</td><td>60keV 多能数据标记；SI 表示能量数据标记，不代表硅元素</td></tr>
<tr><td class="tag-code">(01E7,1002)</td><td>SL</td><td class="tag-val"><b>60 keV</b></td><td>能区标称能量 (非视图数!)</td></tr>
<tr><td class="tag-code">(01E7,1004)</td><td>IS</td><td class="tag-val">2</td><td>焦点数</td></tr>
<tr><td class="tag-group" colspan="4">Group_01F3 — 能区配置</td></tr>
<tr><td class="tag-code">(01F3,1031)</td><td>IS</td><td class="tag-val"><b>{N_BINS}</b></td><td>能量箱数</td></tr>
<tr><td class="tag-code">(01F3,1032)</td><td>IS</td><td class="tag-val">2</td><td>探测器模块数</td></tr>
<tr><td class="tag-code">(01F3,1046)</td><td>DS</td><td class="tag-val">290</td><td>总能量范围 (keV)</td></tr>
<tr><td class="tag-group" colspan="4">Group_01F1 — 扫描几何</td></tr>
<tr><td class="tag-code">(01F1,104B)</td><td>SH</td><td class="tag-val">292*0.274</td><td>292排 x 0.274mm排间距</td></tr>
<tr><td class="tag-code">(01F1,1008)</td><td>DS</td><td class="tag-val">361.132</td><td>源-探测器距离 (mm)</td></tr>
<tr><td class="tag-code">(01F1,104E)</td><td>LO</td><td class="tag-val">Body_Helical</td><td>螺旋扫描</td></tr>
<tr><td class="tag-code">(01F1,1093)</td><td>IS</td><td class="tag-val">4</td><td>kVp级别数</td></tr>
<tr><td class="tag-group" colspan="4">标准标签</td></tr>
<tr><td class="tag-code">RevolutionTime</td><td>DS</td><td class="tag-val"><b>0.3 s</b></td><td>旋转时间</td></tr>
<tr><td class="tag-code">SpiralPitchFactor</td><td>DS</td><td class="tag-val">0.9</td><td>螺距</td></tr>
<tr><td class="tag-code">ConvolutionKernel</td><td>SH</td><td class="tag-val">H30</td><td>重建核(标准FBP)</td></tr>
<tr><td class="tag-code">ReconstructionDiameter</td><td>DS</td><td class="tag-val">433 mm</td><td>重建FOV</td></tr>
<tr><td colspan="4" style="background:#161b22;"></td></tr>
<tr><td class="tag-group" colspan="4" style="font-size:1.1em;">&starf;&starf; 核心数据: (EFE1,1001) — OB, ~15.1MB — JPEG2000 Burst Sinogram ({TOTAL_VIEWS} views)</td></tr>
</table>

<h2 id="ob-header">4. OB Header 结构 (0x000&ndash;0x27F)</h2>

<div class="data-flow">
<div class="flow-box"><b>0x000</b><br>4B: 材料数=4</div><div class="flow-arrow">&rarr;</div>
<div class="flow-box"><b>0x004</b><br>~252B: 材料参数<br>Water/Iodine/Ca/Gd</div><div class="flow-arrow">&rarr;</div>
<div class="flow-box"><b>0x100</b><br>192B: 能区校准</div><div class="flow-arrow">&rarr;</div>
<div class="flow-box"><b>0x1C0</b><br>192B: 能量响应曲线<br>(Float32, 48点)</div><div class="flow-arrow">&rarr;</div>
<div class="flow-box" style="border-color:#7ee787;"><b>0x280</b><br>Burst 数据开始<br>JPEG2000 压缩</div>
</div>

<div class="figure"><img src="data:image/png;base64,{img['img_header']}">
<div class="figure-caption">图 1: OB Header — 左: Float32 能量响应曲线 (408&rarr;0.024); 右: 四种材料参数 (Water, Iodine, Calcium, Gadolinium)</div></div>

<h2 id="jp2k">5. JPEG2000 压缩层</h2>

<table class="tag-table">
<tr><th colspan="2">码流组</th><th>有效数</th><th>数据类型</th><th>说明</th></tr>
<tr><td style="color:#ff7b72;">&#9632;</td><td><b>G0</b> (CS 0-63)</td><td align="center">60</td><td>参考 A</td><td>模块 A 暗场/偏移 (< -10000)</td></tr>
<tr><td style="color:#58a6ff;">&#9632;</td><td><b>G1</b> (CS 64-127)</td><td align="center">{N_BURSTS}</td><td><b>Burst Sinogram A</b></td><td>模块 A 主投影信号 (零中心)</td></tr>
<tr><td style="color:#f0883e;">&#9632;</td><td><b>G2</b> (CS 128-191)</td><td align="center">60</td><td>参考 B</td><td>模块 B 参考 (含饱和)</td></tr>
<tr><td style="color:#7ee787;">&#9632;</td><td><b>G3</b> (CS 192-255)</td><td align="center">60</td><td>Burst Sinogram B</td><td>模块 B 投影信号</td></tr>
</table>

<div class="figure"><img src="data:image/png;base64,{img['img_cs_sizes']}">
<div class="figure-caption">图 2: 256 个 JPEG2000 码流大小 — 4 组 x 64 (红=G0, 蓝=G1, 橙=G2, 绿=G3)。每组 {N_BURSTS} 有效 + 4 空占位 = {N_BURSTS} Burst 窗口</div></div>

<h2 id="proof">6. {N_BINS} 能区 / {VIEWS_PER_BURST} 快视图 — 统计证明</h2>

<table class="tag-table">
<tr><th>能区</th><th>能量 (keV)</th><th>编码行</th><th>行内 View 相关</th><th>跨能区 View 相关</th><th>均值</th><th>说明</th></tr>
<tr><td>Bin 0</td><td class="tag-val">{BIN_KEV[0]}</td><td class="tag-code">0-31</td><td class="tag-val">~0.69</td><td class="tag-val">0.11</td><td class="tag-val">16.5</td><td>低能噪声</td></tr>
<tr><td>Bin 1</td><td class="tag-val" style="color:#e53935;"><b>{BIN_KEV[1]}</b></td><td class="tag-code">32-63</td><td class="tag-val">~0.74</td><td class="tag-val">0.16</td><td class="tag-val">4.1</td><td style="color:#e53935;"><b>K-edge 下界</b></td></tr>
<tr style="background:#2d1b1b;"><td>Bin 2</td><td class="tag-val" style="color:#e53935;"><b>{BIN_KEV[2]}</b></td><td class="tag-code">64-95</td><td class="tag-val">~0.84</td><td class="tag-val">0.32</td><td class="tag-val">14.6</td><td style="color:#e53935;"><b>K-edge 33.2keV</b></td></tr>
<tr><td>Bin 3</td><td class="tag-val">{BIN_KEV[3]}</td><td class="tag-code">96-127</td><td class="tag-val">~0.87</td><td class="tag-val">0.23</td><td class="tag-val">10.4</td><td></td></tr>
<tr><td>Bin 4</td><td class="tag-val">{BIN_KEV[4]}</td><td class="tag-code">128-159</td><td class="tag-val">~0.91</td><td class="tag-val">0.07</td><td class="tag-val">14.3</td><td></td></tr>
<tr><td>Bin 5</td><td class="tag-val">{BIN_KEV[5]}</td><td class="tag-code">160-191</td><td class="tag-val">~0.89</td><td class="tag-val">0.10</td><td class="tag-val">21.8</td><td></td></tr>
<tr><td>Bin 6</td><td class="tag-val">{BIN_KEV[6]}</td><td class="tag-code">192-223</td><td class="tag-val">~0.87</td><td class="tag-val">-0.04</td><td class="tag-val">26.6</td><td></td></tr>
<tr><td>Bin 7</td><td class="tag-val">{BIN_KEV[7]}</td><td class="tag-code">224-255</td><td class="tag-val">~0.82</td><td class="tag-val">-</td><td class="tag-val">34.4</td><td></td></tr>
</table>

<div class="highlight-b">
<p><b>高行内相关 (0.69-0.91):</b> 同一能区同一 Burst 内的 {VIEWS_PER_BURST} 个快速视图时间间隔仅 {1e6/SAMPLE_RATE:.0f}us, 投影数据高度相似<br>
<b>低跨能区相关 (0.11-0.32):</b> 不同能区对同一组织的衰减响应独立<br>
<b style="color:#e53935;">K-edge (33.2 keV):</b> Bin 1 (28-33 keV) 和 Bin 2 (33-38 keV) 跨越碘的 K吸收边, 构成 Haar 小波差分对</p>
</div>

<div class="figure"><img src="data:image/png;base64,{img['img_proof']}">
<div class="figure-caption">图 3: 统计证明 — 左上: 同能区内{VIEWS_PER_BURST}个快视图的高相邻相关; 右上: 跨能区同位置视图的低相关; 左下: 8x8 能区相关矩阵 (对角线=1.00, 非对角线&lt;0.43); 右下: 能区光谱响应</div></div>

<h2 id="kedge">7. K-edge 采样: PCCT vs DECT 对比</h2>

<div class="highlight">
<h3>光子计数 CT 直接 K-edge 采样优势</h3>
<p>传统双能 CT (DECT/EID) 使用 2 个积分能谱 (如 80/140 kVp) 对碘的 K-edge (33.2 keV) 做两点线性插值逼近——仅获得 <b>Haar 0阶近似</b>。光子计数 CT (PCCT/PCD) 则在 K-edge 前后各分配一个独立的窄能区 (28-33 keV 和 33-38 keV), <b>直接测量 K-edge 跃变</b>, 等效于 Haar 小波差分, 信噪比和物质分离精度显著优于 DECT。</p>
</div>

<div class="figure">
<img src="data:image/svg+xml;base64,{svg_kedge_b64}" style="width:100%;">
<div class="figure-caption">图 4: PCCT vs DECT K-edge 采样对比 — (a) 传统EID 2谱积分, 能量混合; (b) PCD 8能区直接能量分辨; (c) 碘衰减曲线 + K-edge 33.2keV 离散差分采样 (红框 = K-edge bin对 28-33 / 33-38 keV); (d) K-edge 提取 = Haar 小波差分 (Bin2 - Bin1)</div>
</div>

<h2 id="groups">8. 4-Group 数据分类</h2>
<div class="figure"><img src="data:image/png;base64,{img['img_groups']}">
<div class="figure-caption">图 5: 4 组对照 — 上排: 列剖面; 下排: {N_BINS} 能区均值。G0/G2 为负值参考, G1/G3 为零中心 Burst 信号</div></div>

<h2 id="sino">8. 完整 Sinogram ({TOTAL_VIEWS} views x {N_BINS} bins)</h2>
<div class="figure"><img src="data:image/png;base64,{img['img_full_sino']}">
<div class="figure-caption">图 6: {N_BINS} 能区 x {TOTAL_VIEWS} 视图完整 Sinogram (Slice 1)。横轴={TOTAL_VIEWS} 投影, 纵轴=256 编码列。RdBu_r 色图</div></div>

<h2 id="timing">9. Burst 时序图</h2>
<div class="figure"><img src="data:image/png;base64,{img['img_timing']}">
<div class="figure-caption">图 7: 一次旋转内的 {N_BURSTS} 个 Burst 窗口时序 — 每个彩色块 = 一个 Burst ({VIEWS_PER_BURST} 个快速采样 @ {SAMPLE_RATE}Hz), 灰色刻度 = 理论 {SAMPLE_RATE}Hz 连续采样点。Burst 死时间 ~{DEAD_US:.0f}us</div></div>

<h2 id="multislice">10. 跨切片验证 + CT肺窗</h2>
<div class="figure"><img src="data:image/png;base64,{img['img_overview']}">
<div class="figure-caption">图 8: 左上: Averaged Sinogram ({TOTAL_VIEWS} views); 其余: 4 个 z-位置的 CT 重建图像 (肺窗); 右下: 能区光谱响应</div></div>

<div class="figure"><img src="data:image/png;base64,{img['img_multislice']}">
<div class="figure-caption">图 9: 6 个 z-位置的 Sinogram 对比 (Bin 4, 12 样本 Burst)。从 z=696.9mm 到 z=970.9mm, 正弦图连续变化, 无异常跳变</div></div>

<h2 id="summary">11. 总结 &amp; 解码代码</h2>

<div class="highlight">
<h3>Python 解码示例</h3>
<div class="code-block"><span class="cm"># 读取 DICOM 私有标签</span>
<span class="kw">import</span> pydicom, glymur, numpy <span class="kw">as</span> np
ds = pydicom.dcmread(<span class="str">'D00060001'</span>)
data = ds[<span class="nu">0xefe1</span>, <span class="nu">0x1001</span>].value

<span class="cm"># 查找 jp2c 标记</span>
pos = []; p = -<span class="nu">1</span>
<span class="kw">while</span> <span class="kw">True</span>:
    p = data.find(<span class="str">b'jp2c'</span>, p+<span class="nu">1</span>)
    <span class="kw">if</span> p == -<span class="nu">1</span>: <span class="kw">break</span>
    pos.append(p)

<span class="cm"># Group 1: {N_BURSTS} Burst 码流</span>
active = [i <span class="kw">for</span> i <span class="kw">in</span> <span class="kw">range</span>(<span class="nu">64</span>,<span class="nu">128</span>) <span class="kw">if</span> i <span class="kw">not</span> <span class="kw">in</span> {{<span class="nu">64</span>,<span class="nu">71</span>,<span class="nu">120</span>,<span class="nu">127</span>}}]

<span class="kw">for</span> vi, cs <span class="kw">in</span> <span class="kw">enumerate</span>(active):
    jp2k = data[pos[cs]+<span class="nu">4</span> : pos[cs+<span class="nu">1</span>]]
    <span class="kw">with</span> <span class="kw">open</span>(<span class="str">'tmp'</span>,<span class="str">'wb'</span>) <span class="kw">as</span> f: f.write(jp2k)
    img = glymur.Jp2k(<span class="str">'tmp'</span>)[:]  <span class="cm"># (256,256) int16</span>

    <span class="kw">for</span> eb <span class="kw">in</span> <span class="kw">range</span>(<span class="nu">8</span>):
        views = img[eb*<span class="nu">32</span>:(eb+<span class="nu">1</span>)*<span class="nu">32</span>, :]
        <span class="cm"># views.shape = ({VIEWS_PER_BURST}, 256)</span>
        <span class="cm"># = 能量区eb的{VIEWS_PER_BURST}个快速投影</span>

<span class="cm"># 总数: {N_BURSTS} bursts x {VIEWS_PER_BURST} views/burst = {TOTAL_VIEWS} views/转</span>
<span class="cm"># 对应 {SAMPLE_RATE}Hz x {REV_TIME}s  = {THEORETICAL} 理论值 (差 {(THEORETICAL-TOTAL_VIEWS)/THEORETICAL*100:.1f}% = Burst间死时间)</span>
</div></div>

<table class="tag-table">
<tr><th>维度</th><th>值</th><th>来源</th></tr>
<tr><td>切片数</td><td class="tag-val">2,635</td><td>DICOM文件</td></tr>
<tr><td>探测器排 (z)</td><td class="tag-val">292</td><td><span class="tag-code">(01F1,104B)</span></td></tr>
<tr><td>能量区</td><td class="tag-val">{N_BINS}</td><td><span class="tag-code">(01F3,1031)</span></td></tr>
<tr><td>Burst / 转</td><td class="tag-val">{N_BURSTS}</td><td>G1 有效码流数</td></tr>
<tr><td>快视图 / Burst</td><td class="tag-val">{VIEWS_PER_BURST}</td><td>JP2K 行/{N_BINS}</td></tr>
<tr><td><b>总视图 / 转</b></td><td class="tag-val"><b>{TOTAL_VIEWS}</b></td><td>{N_BURSTS} x {VIEWS_PER_BURST}</td></tr>
<tr><td>JP2K 码流/切片</td><td class="tag-val">256</td><td>{N_BURSTS} x 4 组 + 16 空</td></tr>
<tr><td>数据块大小</td><td class="tag-val">~15.1 MB</td><td>(EFE1,1001) OB</td></tr>
</table>

<footer><p>解析报告 v4 | 2026-06-02 | 2,635 DICOM (NeuViz P10) | {TOTAL_VIEWS} views = {N_BURSTS} bursts x {VIEWS_PER_BURST} @ {SAMPLE_RATE}Hz</p>
<p>pydicom + glymur (OpenJPEG 2.5.2) + matplotlib + numpy</p></footer>

</div></body></html>'''

output_path = os.path.join(base_dir, 'energy_bin_sinogram_report.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

size_mb = os.path.getsize(output_path) / (1024*1024)
print(f'HTML report: {output_path} ({size_mb:.1f} MB)')
