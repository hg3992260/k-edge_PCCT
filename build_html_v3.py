import json, os

base_dir = r'D:\xl\00020006'
with open(os.path.join(base_dir, 'b64_v3.json'), 'r') as f:
    img = json.load(f)

html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>光子计数CT — Energy Bin Sinogram 完全解析报告</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.7; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
h1 { font-size: 2em; color: #58a6ff; text-align: center; margin: 30px 0 10px; padding-bottom: 15px; border-bottom: 2px solid #30363d; }
h2 { color: #f0883e; margin: 30px 0 15px; font-size: 1.5em; border-left: 4px solid #f0883e; padding-left: 12px; }
h3 { color: #d2a8ff; margin: 20px 0 10px; font-size: 1.15em; }
.meta-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 8px; margin: 15px 0; }
.meta-item { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; }
.meta-key { color: #8b949e; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; }
.meta-val { color: #e6edf3; font-size: 1em; word-break: break-all; }
.tag-table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.85em; }
.tag-table th { background: #21262d; color: #8b949e; padding: 8px 12px; text-align: left; border: 1px solid #30363d; font-weight: 600; }
.tag-table td { padding: 6px 12px; border: 1px solid #30363d; vertical-align: top; }
.tag-table tr:nth-child(even) td { background: #161b22; }
.tag-group { color: #f0883e; font-weight: bold; }
.tag-code { color: #7ee787; font-family: 'Consolas','Courier New',monospace; font-size: 0.9em; }
.tag-val { color: #a5d6ff; }
.tt-note { color: #8b949e; font-size: 0.8em; }
.highlight { background: #1f242b; border-left: 3px solid #f0883e; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 15px 0; }
.highlight-g { background: #1f242b; border-left: 3px solid #7ee787; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 15px 0; }
.highlight-w { background: #1f242b; border-left: 3px solid #d2991d; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 15px 0; }
.code-block { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 15px; font-family: 'Consolas','Courier New',monospace; font-size: 0.8em; overflow-x: auto; white-space: pre-wrap; color: #c9d1d9; line-height: 1.5; }
.cm { color: #8b949e; }
.kw { color: #ff7b72; }
.nu { color: #a5d6ff; }
.str { color: #7ee787; }
.data-flow { display: flex; justify-content: center; align-items: center; gap: 12px; margin: 20px 0; flex-wrap: wrap; }
.flow-box { background: #21262d; border: 2px solid #30363d; border-radius: 10px; padding: 10px 18px; text-align: center; font-size: 0.85em; }
.figure { margin: 20px 0; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
.figure img { width: 100%; border-radius: 4px; }
.figure-caption { color: #8b949e; font-size: 0.85em; margin-top: 8px; text-align: center; line-height: 1.5; }
.toc { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 20px 0; columns: 2; }
.toc a { color: #58a6ff; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.toc ol { margin-left: 20px; }
.toc li { margin: 3px 0; font-size: 0.9em; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(185px, 1fr)); gap: 8px; margin: 15px 0; }
.stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; text-align: center; }
.stat-num { font-size: 1.8em; color: #58a6ff; font-weight: bold; }
.stat-label { color: #8b949e; font-size: 0.75em; margin-top: 4px; }
.footnote { color: #8b949e; font-size: 0.8em; }
footer { text-align: center; color: #484f58; margin: 40px 0 20px; font-size: 0.85em; }
ul { margin-left: 20px; line-height: 1.8; }
</style>
</head>
<body>
<div class="container">

<h1>光子计数 CT / Energy Bin Sinogram 完全解析报告</h1>
<p style="text-align:center;color:#8b949e;">NeuViz P10 | 120kVp Thorax | 8能区 x 60投影 | 292排探测器 | 2,635层 | JPEG2000压缩</p>

<div class="toc"><ol>
<li><a href="#overview">概览 &amp; 关键数字</a></li>
<li><a href="#geometry">探测器几何与时序</a></li>
<li><a href="#tags">私有标签全解析</a></li>
<li><a href="#ob-header">OB Header 结构</a></li>
<li><a href="#jp2k">JPEG2000 压缩层分析</a></li>
<li><a href="#proof">8能区分割的统计证明</a></li>
<li><a href="#groups">4-Group 数据分类</a></li>
<li><a href="#sino">完整 Sinogram 可视化</a></li>
<li><a href="#multislice">跨切片验证</a></li>
<li><a href="#summary">总结 &amp; 解码代码</a></li>
</ol></div>

<!-- ================================================================ -->
<h2 id="overview">1. 概览 &amp; 关键数字</h2>

<div class="stats-grid">
<div class="stat-card"><div class="stat-num">2,635</div><div class="stat-label">DICOM 切片文件</div></div>
<div class="stat-card"><div class="stat-num">2048 x 2048</div><div class="stat-label">CT重建图像</div></div>
<div class="stat-card"><div class="stat-num">292</div><div class="stat-label">探测器 z-排 (rows)</div></div>
<div class="stat-card"><div class="stat-num">256</div><div class="stat-label">探测器通道 (channels)</div></div>
<div class="stat-card"><div class="stat-num">8</div><div class="stat-label">能量区 (Energy Bins)</div></div>
<div class="stat-card"><div class="stat-num">60</div><div class="stat-label">投影角度 (Views)</div></div>
<div class="stat-card"><div class="stat-num">256</div><div class="stat-label">JPEG2000 码流/切片</div></div>
<div class="stat-card"><div class="stat-num">240</div><div class="stat-label">有效码流 (60 x4组)</div></div>
<div class="stat-card"><div class="stat-num">15.1 MB</div><div class="stat-label">OB 数据块</div></div>
<div class="stat-card"><div class="stat-num">~94亿</div><div class="stat-label">正弦图总数据点</div></div>
</div>

<div class="highlight">
<strong>设备:</strong> Neusoft NeuViz P10 光子计数 CT &nbsp;|&nbsp; <strong>探头:</strong> ME 60keV/SI (SI 为能量数据标记，非硅) &nbsp;|&nbsp; <strong>协议:</strong> 120kVp, Body_Helical Thorax<br>
<strong>数据位置:</strong> DICOM 私有标签 <code>(EFE1,1001)</code> — OB 类型, ~15.1MB, JPEG2000 压缩的能区 Sinogram<br>
<strong>Sinogram 结构:</strong> [4 组, 8 能区, 60 视图, 256 编码行, 256 通道] — 其中 256 编码行编码 292 排物理探测器<br>
<strong>时间特性:</strong> 292 排探测器<b>同时采集</b> — 60 Views 为空间投影角度, 非时间序列
</div>

<!-- ================================================================ -->
<h2 id="geometry">2. 探测器几何与时序</h2>

<h3>2.1 几何参数</h3>
<table class="tag-table">
<tr><th>参数</th><th>来源</th><th>值</th><th>说明</th></tr>
<tr><td>探测器 z-排数</td><td class="tag-code">(01F1,104B)</td><td class="tag-val">292 x0.274mm</td><td>292 rows @ 0.274mm pitch</td></tr>
<tr><td>总 z-覆盖</td><td>计算</td><td class="tag-val">~80.0 mm</td><td>292 x 0.274mm</td></tr>
<tr><td>探测器通道数</td><td>JP2K 图像列</td><td class="tag-val">256</td><td>面内方向 (fan direction)</td></tr>
<tr><td>源-探测器距离</td><td class="tag-code">(01F1,1008)</td><td class="tag-val">361.13 mm</td><td>几何放大参数</td></tr>
<tr><td>焦点-探测器采点</td><td class="tag-code">(01F1,1011)</td><td class="tag-val">[1154.4, 1154.9, 1154.7, 1154.6]</td><td>4个焦斑采样</td></tr>
<tr><td>能区数</td><td class="tag-code">(01F3,1031)</td><td class="tag-val">8</td><td>光子计数能量箱数</td></tr>
<tr><td>能区总范围</td><td class="tag-code">(01F3,1046)</td><td class="tag-val">290 keV</td><td>~36 keV/Bin</td></tr>
<tr><td>探测器模块/源数</td><td class="tag-code">(01F3,1032)</td><td class="tag-val">2</td><td>双源或双模块</td></tr>
<tr><td>焦点数</td><td class="tag-code">(01E7,1004)</td><td class="tag-val">2</td><td>双焦点 (FS0, FS1)</td></tr>
<tr><td>投影角度数</td><td class="tag-code">(01E7,1002)</td><td class="tag-val">60</td><td>每切片投影数</td></tr>
<tr><td>扫描模式</td><td class="tag-code">(01F1,1064)</td><td class="tag-val">STANDARDBODY</td><td>标准体扫</td></tr>
<tr><td>轨迹类型</td><td class="tag-code">(01F1,104E)</td><td class="tag-val">Body_Helical</td><td>螺旋扫描</td></tr>
<tr><td>kVp</td><td class="tag-code">(01F1,1002)</td><td class="tag-val">HIGH (120kV)</td><td>高通量模式</td></tr>
</table>

<h3>2.2 时序关系</h3>
<div class="highlight-g">
<p><b>292 排探测器同步采集:</b> 每个投影角度下, 292 排探测器阵列<b>同时</b>获取 8 个能区的光子计数信号。60 个投影角度对应 60 个不同的 X 射线源-探测器旋转位置, <b>而非</b>时间序列上依次激发的排数据。</p>
<p style="margin-top:8px"><b>不能将 Row 数据按时间拆分:</b> 系统设计保证所有排的信号在同一时间戳下采集。JPEG2000 编码行 (256 行) 只是压缩域组织方式, 不代表探测器排的时间先后。</p>
</div>

<div class="figure">
<img src="data:image/png;base64,''' + img['img_geometry'] + '''">
<div class="figure-caption">图 1: 数据编码几何示意图 — 292 排物理探测器同时采集 → 8 能区行堆叠于 256x256 JPEG2000 图像 → 60 个投影角度 → 共 256 个码流 (4 组)</div>
</div>

<!-- ================================================================ -->
<h2 id="tags">3. 私有标签全解析</h2>

<table class="tag-table">
<tr><th>标签</th><th>VR</th><th>值</th><th>含义</th></tr>
<tr><td class="tag-group" colspan="4">组 01E7 — 能区探测器元数据</td></tr>
<tr><td class="tag-code">(01E7,1001)</td><td>LO</td><td class="tag-val">ME 60keV/SI</td><td>探测器/能量标记（SI 为能量数据标记，不代表硅）</td></tr>
<tr><td class="tag-code">(01E7,1002)</td><td>SL</td><td class="tag-val"><b>60</b></td><td>投影角度/视图数</td></tr>
<tr><td class="tag-code">(01E7,1004)</td><td>IS</td><td class="tag-val"><b>2</b></td><td>焦点数 (双焦点)</td></tr>
<tr><td class="tag-code">(01E7,1011)</td><td>UL</td><td class="tag-val">15,171,801</td><td>OB计数参考值</td></tr>
<tr><td class="tag-group" colspan="4">组 01F3 — 能区配置</td></tr>
<tr><td class="tag-code">(01F3,1031)</td><td>IS</td><td class="tag-val"><b>8</b></td><td>能量箱数 (Energy Bins)</td></tr>
<tr><td class="tag-code">(01F3,1032)</td><td>IS</td><td class="tag-val"><b>2</b></td><td>探测器模块数</td></tr>
<tr><td class="tag-code">(01F3,1046)</td><td>DS</td><td class="tag-val"><b>290</b></td><td>总能量范围 (keV)</td></tr>
<tr><td class="tag-group" colspan="4">组 01F1 — 扫描几何 + 探测器参数</td></tr>
<tr><td class="tag-code">(01F1,104B)</td><td>SH</td><td class="tag-val"><b>292*0.274</b></td><td>292排 x 0.274mm排间距</td></tr>
<tr><td class="tag-code">(01F1,1008)</td><td>DS</td><td class="tag-val">361.132</td><td>S-D 距离 (mm)</td></tr>
<tr><td class="tag-code">(01F1,1093)</td><td>IS</td><td class="tag-val">4</td><td>kVp级别数</td></tr>
<tr><td class="tag-group" colspan="4">组 01F9 — 时间采样</td></tr>
<tr><td class="tag-code">(01F9,1004)</td><td>IS</td><td class="tag-val">1</td><td>每旋转曝光标志</td></tr>
<tr><td class="tag-code">(01F9,1005)</td><td>IS</td><td class="tag-val">6</td><td>每旋转时间采样数</td></tr>
<tr><td class="tag-group" colspan="4">组 01E3 — 扫描状态</td></tr>
<tr><td class="tag-code">(01E3,1021)</td><td>IS</td><td class="tag-val">1</td><td>焦点尺寸标志</td></tr>
<tr><td class="tag-code">(01E3,1025)</td><td>LO</td><td class="tag-val">FS0,FS1</td><td>双焦点名称</td></tr>
<tr><td class="tag-code">(01E3,1026)</td><td>ST</td><td class="tag-val">Thorax</td><td>胸部扫描</td></tr>
<tr><td class="tag-group" colspan="4">组 00E1 — z-位置</td></tr>
<tr><td class="tag-code">(00E1,10C4)</td><td>DS</td><td class="tag-val">696.9</td><td>Slice 1 z-位置 (mm)</td></tr>
<tr><td colspan="4" style="background:#161b22;"></td></tr>
<tr><td class="tag-group" colspan="4" style="font-size:1.1em;">★★ 核心数据标签: (EFE1,1001) — OB, ~15.1MB — JPEG2000 压缩的能区 Sinogram</td></tr>
</table>

<!-- ================================================================ -->
<h2 id="ob-header">4. OB Header 结构分析 (0x000&ndash;0x27F)</h2>

<div class="data-flow">
<div class="flow-box"><b>Offset 0x000</b><br>4B: 材料数=4</div>
<div class="flow-arrow">&#9654;</div>
<div class="flow-box"><b>Offset 0x004</b><br>~252B: 4种材料<br>Water/Iodine/Calcium/Gd<br>(各14 uint32参数)</div>
<div class="flow-arrow">&#9654;</div>
<div class="flow-box"><b>Offset 0x100</b><br>192B: 能区-材料<br>映射校准表</div>
<div class="flow-arrow">&#9654;</div>
<div class="flow-box"><b>Offset 0x1C0</b><br>192B: Float32<br>能量响应曲线<br>(48点, 408→0.024)</div>
<div class="flow-arrow">&#9654;</div>
<div class="flow-box" style="border-color:#7ee787;"><b>Offset 0x280</b><br>Sinogram 开始<br>JPEG2000 压缩</div>
</div>

<div class="figure">
<img src="data:image/png;base64,''' + img['img_header'] + '''">
<div class="figure-caption">图 2: OB Header 分析 — 左: Float32 能量响应曲线 (48个数据点, 从~408到~0.024递减, 代表材料对不同能量光子的响应特征); 右: 四种材料 (Water, Iodine, Calcium, Gadolinium) 的 Param[2] 对比</div>
</div>

<!-- ================================================================ -->
<h2 id="jp2k">5. JPEG2000 压缩层分析</h2>

<div class="highlight">
<h3>码流概况</h3>
<ul>
<li>共 <b>256</b> 个 <code>jp2c</code> (JPEG2000 Codestream) 标记, 分布在 OB 数据区域 (0x280 起)</li>
<li><b>240</b> 有效码流 + <b>16</b> 空占位 (每 64 码流一组的边界位置)</li>
<li>解码参数: 256x256 int16, Profile 0, 1 分量, 可逆 9-7 变换</li>
<li>压缩库: <b>OpenJPEG 2.5.2</b> (码流末尾嵌入版权字符串)</li>
<li>码流大小: 12KB ~ 95KB (有效码流), ~230B (空占位)</li>
</ul>
</div>

<div class="figure">
<img src="data:image/png;base64,''' + img['img_cs_sizes'] + '''">
<div class="figure-caption">图 3: 256 个 JPEG2000 码流大小 — 4 组 x 64 结构 (红=G0 RefA, 蓝=G1 SinoA, 橙=G2 RefB, 绿=G3 SinoB)。每组 60 有效 + 4 空占位, 对应 60 个投影角度</div>
</div>

<table class="tag-table">
<tr><th colspan="2">码流组</th><th>有效数</th><th>数据类型</th><th>值范围</th><th>说明</th></tr>
<tr><td style="color:#ff7b72;">&#9632;</td><td><b>Group 0</b><br>CS 0-63</td><td align="center">60</td><td>参考 A</td><td class="tag-val">-33,000 &sim; -10,000</td><td>模块 A 的暗场/偏移参考</td></tr>
<tr><td style="color:#58a6ff;">&#9632;</td><td><b>Group 1</b><br>CS 64-127</td><td align="center">60</td><td><b>Sinogram A</b></td><td class="tag-val">-1,500 &sim; +4,000</td><td>模块 A 主投影信号</td></tr>
<tr><td style="color:#f0883e;">&#9632;</td><td><b>Group 2</b><br>CS 128-191</td><td align="center">60</td><td>参考 B</td><td class="tag-val">-32,768 &sim; -5,000</td><td>模块 B 的参考数据(含饱和)</td></tr>
<tr><td style="color:#7ee787;">&#9632;</td><td><b>Group 3</b><br>CS 192-255</td><td align="center">60</td><td><b>Sinogram B</b></td><td class="tag-val">-32,768 &sim; +4,000</td><td>模块 B 投影信号(部分饱和)</td></tr>
</table>

<div class="highlight-w">
<b>4 组 x 60 视图 = 240 个有效码流。</b> Group 1 和 Group 3 为主要 Sinogram 数据，分别对应两个探测器模块 (01F3,1032=2)。Group 0 和 Group 2 为对应模块的参考/校准数据。每组 60 个码流精确对应 <code>(01E7,1002)=60</code> 个投影角度。
</div>

<!-- ================================================================ -->
<h2 id="proof">6. 8 能区分割的统计证明</h2>

<div class="highlight">
<h3>数据中的 8-Bin 行堆叠结构</h3>
<p>每个 256x256 JPEG2000 图像在<b>垂直方向 (256 行)</b>按 <b>8 个能区</b> 堆叠, 每能区占用 <b>32 编码行</b>:</p>
</div>

<table class="tag-table">
<tr><th>能区</th><th>编码行范围</th><th>内部行间相关</th><th>跨能区相关</th><th>均值 (Group 1)</th></tr>
<tr><td>Bin 0</td><td class="tag-code">0-31</td><td class="tag-val">~0.69</td><td class="tag-val">0.11 (vs Bin 1)</td><td class="tag-val">16.5</td></tr>
<tr><td>Bin 1</td><td class="tag-code">32-63</td><td class="tag-val">~0.74</td><td class="tag-val">0.16 (vs Bin 2)</td><td class="tag-val">4.1</td></tr>
<tr><td>Bin 2</td><td class="tag-code">64-95</td><td class="tag-val">~0.84</td><td class="tag-val">0.32 (vs Bin 3)</td><td class="tag-val">14.6</td></tr>
<tr><td>Bin 3</td><td class="tag-code">96-127</td><td class="tag-val">~0.87</td><td class="tag-val">0.23 (vs Bin 4)</td><td class="tag-val">10.4</td></tr>
<tr><td>Bin 4</td><td class="tag-code">128-159</td><td class="tag-val">~0.91</td><td class="tag-val">0.07 (vs Bin 5)</td><td class="tag-val">14.3</td></tr>
<tr><td>Bin 5</td><td class="tag-code">160-191</td><td class="tag-val">~0.89</td><td class="tag-val">0.10 (vs Bin 6)</td><td class="tag-val">21.8</td></tr>
<tr><td>Bin 6</td><td class="tag-code">192-223</td><td class="tag-val">~0.87</td><td class="tag-val">-0.04 (vs Bin 7)</td><td class="tag-val">26.6</td></tr>
<tr><td>Bin 7</td><td class="tag-code">224-255</td><td class="tag-val">~0.82</td><td class="tag-val">-</td><td class="tag-val">34.4</td></tr>
</table>

<div class="figure">
<img src="data:image/png;base64,''' + img['img_proof'] + '''">
<div class="figure-caption">图 4: 8能区分割的统计证明 — <b>左上:</b> 同一能区内相邻行的高相关 (~0.7-0.9), 证明行数据来自同一探测器区域同一能区; <b>右上:</b> 不同能区同一行位置的低相关 (~-0.04-0.32), 证明不同能区包含独立的光谱信息; <b>左下:</b> 像素级 8x8 能区相关矩阵, 对角线=1.00, 非对角线均&lt;0.43, 所有能区统计独立; <b>右下:</b> 8个能区在60视图上的均值±SD</div>
</div>

<div class="highlight-g">
<h3>关键结论</h3>
<ul>
<li><b>同一能区内部 (32 编码行):</b> 相邻行高度相关 (r=0.69-0.91) — 这些编码行覆盖相邻的物理探测器排 (292排/32编码行 &asymp; 9物理排/编码行), 查看相邻解剖结构</li>
<li><b>不同能区之间:</b> 对应行低相关 (r=-0.04-0.32) — 不同能区观测独立的能谱特征, <b>确认为 8 个独立能量箱</b></li>
<li><b>频谱响应:</b> 8 个能区具有不同的衰减均值 — Bin 0-4 为低能区 (衰减较大), Bin 5-7 为高能区 (衰减较小)</li>
</ul>
</div>

<!-- ================================================================ -->
<h2 id="groups">7. 4-Group 数据分类</h2>

<div class="figure">
<img src="data:image/png;base64,''' + img['img_groups'] + '''">
<div class="figure-caption">图 5: 4 组数据对照 — 上排: 每组代表性图像的单列剖面 (256通道); 下排: 每组的 8 能区均值柱状图。Group 0 和 Group 2 为参考/暗场 (全负值), Group 1 和 Group 3 为有效 Sinogram (零中心值)。Group 3 存在部分 INT16_MIN (-32768) 饱和值。</div>
</div>

<!-- ================================================================ -->
<h2 id="sino">8. 完整 Sinogram 可视化 (Slice 1, Group 1)</h2>

<div class="figure">
<img src="data:image/png;base64,''' + img['img_full_sino'] + '''">
<div class="figure-caption">图 6: <b>8 能区 x 60 视图完整 Sinogram。</b> 每个子图: 横轴=60个投影角度, 纵轴=256探测器通道 (32编码行均值)。RdBu_r 色图: 白色=零(空气参考), 红/蓝=正/负衰减。</div>
</div>

<div class="figure">
<img src="data:image/png;base64,''' + img['img_overview'] + '''">
<div class="figure-caption">图 7: <b>左:</b> 所有8能区60视图压缩至单图 (256x256 编码行 x 通道); <b>右:</b> 对应切片的 CT 重建图像 (2048x2048, uint16, WL=57000)</div>
</div>

<!-- ================================================================ -->
<h2 id="multislice">9. 跨切片验证</h2>

<div class="figure">
<img src="data:image/png;base64,''' + img['img_multislice'] + '''">
<div class="figure-caption">图 8: 6 个不同 z-位置的 Sinogram 对比 (能区 4, 12 个样本视图)。从 z=696.9mm 到 z=970.9mm, 正弦图呈现连续的空间变化, 验证了解析方案的正确性, 无明显跳变或异常。</div>
</div>

<!-- ================================================================ -->
<h2 id="summary">10. 总结 &amp; 解码代码</h2>

<div class="highlight">
<h3>完整数据流</h3>
</div>

<div class="data-flow" style="font-size:0.8em;">
<div class="flow-box"><b>DICOM (EFE1,1001)</b><br>OB Blob<br>~15.1MB</div>
<div class="flow-arrow" style="font-size:1.2em;">&#9654;</div>
<div class="flow-box"><b>Header 640B</b><br>4材料+校准+响应</div>
<div class="flow-arrow" style="font-size:1.2em;">&#9654;</div>
<div class="flow-box" style="border-color:#7ee787;"><b>256 jp2c 码流</b><br>JPEG2000 解码<br>256x256 int16</div>
<div class="flow-arrow" style="font-size:1.2em;">&#9654;</div>
<div class="flow-box" style="border-color:#58a6ff;"><b>Group 1</b><br>60 码流<br>8 能区 x 60 视图</div>
<div class="flow-arrow" style="font-size:1.2em;">&#9654;</div>
<div class="flow-box" style="border-color:#f0883e;"><b>Sinogram</b><br>[8 bins, 60 views<br>32 enc rows, 256 ch]</div>
</div>

<div class="highlight">
<h3>Python 解码完整示例</h3>
<div class="code-block"><span class="cm"># 1. 读取 DICOM 私有标签</span>
<span class="kw">import</span> pydicom, glymur, numpy <span class="kw">as</span> np
ds = pydicom.dcmread(<span class="str">'D00060001'</span>)
data = ds[<span class="nu">0xefe1</span>, <span class="nu">0x1001</span>].value  <span class="cm"># ~15.1MB OB 数据</span>

<span class="cm"># 2. 查找所有 jp2c 标记</span>
pos = []; p = -<span class="nu">1</span>
<span class="kw">while</span> <span class="kw">True</span>:
    p = data.find(<span class="str">b'jp2c'</span>, p+<span class="nu">1</span>)
    <span class="kw">if</span> p == -<span class="nu">1</span>: <span class="kw">break</span>
    pos.append(p)

<span class="cm"># 3. 从 Group 1 (CS 65-126, 有效60个) 解码 Sinogram</span>
empty = {<span class="nu">1</span>: [<span class="nu">64</span>,<span class="nu">71</span>,<span class="nu">120</span>,<span class="nu">127</span>]}  <span class="cm"># Group 1 中的空占位</span>
active = [i <span class="kw">for</span> i <span class="kw">in</span> <span class="kw">range</span>(<span class="nu">64</span>,<span class="nu">128</span>) <span class="kw">if</span> i <span class="kw">not</span> <span class="kw">in</span> empty[<span class="nu">1</span>]]

full_sino = np.zeros((<span class="nu">8</span>, <span class="kw">len</span>(active), <span class="nu">32</span>, <span class="nu">256</span>), dtype=np.int16)
<span class="kw">for</span> vi, cs <span class="kw">in</span> <span class="kw">enumerate</span>(active):
    start = pos[cs] + <span class="nu">4</span>;  end = pos[cs+<span class="nu">1</span>]
    jp2k = data[start:end]                     <span class="cm"># JPEG2000 压缩码流</span>
    <span class="kw">with</span> <span class="kw">open</span>(<span class="str">'tmp.jp2k'</span>,<span class="str">'wb'</span>) <span class="kw">as</span> f: f.write(jp2k)
    img = glymur.Jp2k(<span class="str">'tmp.jp2k'</span>)[:]        <span class="cm"># 解码: (256, 256) int16</span>

    <span class="kw">for</span> eb <span class="kw">in</span> <span class="kw">range</span>(<span class="nu">8</span>):                       <span class="cm"># 8 个能区按行堆叠</span>
        full_sino[eb, vi] = img[eb*<span class="nu">32</span>:(eb+<span class="nu">1</span>)*<span class="nu">32</span>, :]

<span class="cm"># 结果: full_sino.shape = (8, 60, 32, 256)</span>
<span class="cm">#   = [8能区, 60视图, 32编码行, 256探测器通道]</span>
<span class="cm">#   256编码行编码 292 排物理探测器 (同时采集)</span>
</div>
</div>

<h3>关键数字总汇</h3>
<table class="tag-table">
<tr><th>维度</th><th>大小</th><th>来源</th></tr>
<tr><td>切片数</td><td class="tag-val">2,635</td><td>DICOM文件数</td></tr>
<tr><td>探测器物理排 (z)</td><td class="tag-val">292</td><td><code>(01F1,104B)</code></td></tr>
<tr><td>探测器通道 (面内)</td><td class="tag-val">256</td><td>JP2K 图像列数</td></tr>
<tr><td>能量区</td><td class="tag-val">8</td><td><code>(01F3,1031)</code></td></tr>
<tr><td>投影角度 (Views)</td><td class="tag-val">60</td><td><code>(01E7,1002)</code></td></tr>
<tr><td>探测器模块</td><td class="tag-val">2</td><td><code>(01F3,1032)</code></td></tr>
<tr><td>JPEG2000 码流/切片</td><td class="tag-val">256 (240有效)</td><td>解析码流数</td></tr>
<tr><td>编码行/能区/视图</td><td class="tag-val">~32</td><td>JP2K 图像行 / 8 能区</td></tr>
<tr><td>物理排/编码行</td><td class="tag-val">~9.1</td><td>292 / 32</td></tr>
<tr><td>总数据点 (全切片)</td><td class="tag-val">~94 亿</td><td>2,635 x 60 x 8 x 292 x 256</td></tr>
<tr><td>OB数据大小 (总)</td><td class="tag-val">~39 GB</td><td>2,635 x ~15.1MB</td></tr>
</table>

<footer>
<p>解析报告 | 2026-06-02 | 数据源: 2,635 DICOM 文件 (NeuViz P10 光子计数 CT)</p>
<p>解码工具: pydicom + glymur (OpenJPEG 2.5.2) + matplotlib + numpy</p>
</footer>

</div>
</body>
</html>'''

output_path = os.path.join(base_dir, 'energy_bin_sinogram_report.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

size_mb = os.path.getsize(output_path) / (1024*1024)
print(f'HTML report saved: {output_path} ({size_mb:.1f} MB)')
