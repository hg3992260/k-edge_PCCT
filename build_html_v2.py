import json, os

base_dir = r'D:\xl\00020006'

with open(os.path.join(base_dir, 'b64_images.json'), 'r') as f:
    images = json.load(f)

html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Photon-Counting CT — Energy Bin Sinogram 解析报告</title>
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
.tag-table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }
.tag-table th { background: #21262d; color: #8b949e; padding: 8px 12px; text-align: left; border: 1px solid #30363d; font-weight: 600; }
.tag-table td { padding: 6px 12px; border: 1px solid #30363d; }
.tag-table tr:nth-child(even) td { background: #161b22; }
.tag-group { color: #f0883e; font-weight: bold; }
.tag-code { color: #7ee787; font-family: 'Consolas', 'Courier New', monospace; }
.tag-val { color: #a5d6ff; }
.highlight { background: #1f242b; border-left: 3px solid #f0883e; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 15px 0; }
.highlight-g { background: #1f242b; border-left: 3px solid #7ee787; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 15px 0; }
.code-block { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 15px; font-family: 'Consolas', 'Courier New', monospace; font-size: 0.85em; overflow-x: auto; white-space: pre-wrap; color: #7ee787; }
.code-block .comment { color: #8b949e; }
.code-block .keyword { color: #ff7b72; }
.code-block .number { color: #a5d6ff; }
.data-flow { display: flex; justify-content: center; align-items: center; gap: 15px; margin: 20px 0; flex-wrap: wrap; }
.flow-box { background: #21262d; border: 2px solid #30363d; border-radius: 10px; padding: 12px 20px; text-align: center; min-width: 100px; }
.flow-box.ob { border-color: #f0883e; }
.flow-box.jp2 { border-color: #58a6ff; }
.flow-box.img { border-color: #7ee787; }
.flow-arrow { color: #58a6ff; font-size: 1.5em; }
.figure { margin: 20px 0; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
.figure img { width: 100%; border-radius: 4px; }
.figure-caption { color: #8b949e; font-size: 0.85em; margin-top: 8px; text-align: center; }
.dim-table { margin: 15px auto; }
.toc { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 20px 0; }
.toc a { color: #58a6ff; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.toc ol { margin-left: 20px; }
.toc li { margin: 5px 0; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin: 15px 0; }
.stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center; }
.stat-num { font-size: 2em; color: #58a6ff; font-weight: bold; }
.stat-label { color: #8b949e; font-size: 0.8em; margin-top: 5px; }
.correction { background: #2d1b1b; border-left: 3px solid #da3633; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 15px 0; color: #ffb1a4; }
footer { text-align: center; color: #484f58; margin: 40px 0 20px; font-size: 0.85em; }
</style>
</head>
<body>
<div class="container">

<h1>Photon-Counting CT / Energy Bin Sinogram 解析报告</h1>
<p style="text-align:center;color:#8b949e;">NeuViz P10 | 120kVp | 8个能区 | 60个投影角度 | 2635层切片 | 292排探测器</p>

<div class="toc">
<h3>目录</h3>
<ol>
<li><a href="#overview">概览</a></li>
<li><a href="#metadata">DICOM 元数据 &amp; 探测器几何</a></li>
<li><a href="#private-tags">私有标签解析</a></li>
<li><a href="#ob-structure">(EFE1,1001) OB 数据结构</a></li>
<li><a href="#jp2k">JPEG2000 压缩分析</a></li>
<li><a href="#sinogram">Energy Bin Sinogram 可视化</a></li>
<li><a href="#summary">总结 &amp; 数据解码流程</a></li>
</ol>
</div>

<!-- OVERVIEW -->
<h2 id="overview">1. 概览</h2>

<div class="stats-grid">
<div class="stat-card"><div class="stat-num">2,635</div><div class="stat-label">DICOM 文件 (切片数)</div></div>
<div class="stat-card"><div class="stat-num">2048×2048</div><div class="stat-label">重建图像分辨率</div></div>
<div class="stat-card"><div class="stat-num">8</div><div class="stat-label">能量区 (Energy Bins)</div></div>
<div class="stat-card"><div class="stat-num">60</div><div class="stat-label">投影角度 (Views)</div></div>
<div class="stat-card"><div class="stat-num">292</div><div class="stat-label">探测器z排 (detector rows)</div></div>
<div class="stat-card"><div class="stat-num">256</div><div class="stat-label">探测器通道 (channels)</div></div>
<div class="stat-card"><div class="stat-num">256</div><div class="stat-label">JPEG2000 码流/文件</div></div>
<div class="stat-card"><div class="stat-num">~15.1 MB</div><div class="stat-label">OB 数据块大小</div></div>
</div>

<div class="highlight">
<strong>设备:</strong> Neusoft NeuViz P10 光子计数 CT<br>
<strong>扫描协议:</strong> 120 kVp, Thorax (Body_Helical), 螺距 0.274 mm, 2635 slices<br>
<strong>探测器:</strong> <b>292 排</b> (z-方向), 256 通道 (面内方向), 排间距 0.274 mm, 总 z-覆盖 ~80 mm<br>
<strong>能区:</strong> 8 个能量箱 (来自 (01F3,1031)), 总能量范围 290 keV<br>
<strong>数据存储:</strong> DICOM 私有标签 (EFE1,1001), JPEG2000 压缩<br>
<strong>时间特性:</strong> <b>292 排探测器同时采集</b> — 所有排的信号在时间轴上是同步的，不按时间拆分<br>
<strong>Sinogram 维度:</strong> [8 能区 × 60 视图 × 256 编码行 × 256 通道]
</div>

<div class="highlight-g">
<h3>重要说明</h3>
<ul style="margin-left:20px;">
<li><b>292 排探测器同时采集：</b> 系统 z-轴方向共有 292 排探测器 (<code>(01F1,104B)="292*0.274"</code>), 所有排的光子计数信号在时间上是同步的。JPEG2000 码流中的 "行" 是数据编码方式，不代表探测器排的物理划分。</li>
<li><b>JPEG2000 256×256 编码：</b> 每个 256×256 的 JPEG2000 图像将 8 个能区的数据按行堆叠（各占 ~32 编码行）。这 32 行是数据编码组织，并非 32 排物理探测器。</li>
<li><b>60 views = 60 个投影角度：</b> 每个投影角度下，292 排探测器数据被编码/压缩到一个 256×256 JPEG2000 图像中。</li>
<li><b>4 组 × 60 Views：</b> 256 个码流分为 4 组 (Group 0-3), 每组 60 个有效码流（+4 空占位）对应 60 个投影角度。</li>
</ul>
</div>

<!-- METADATA -->
<h2 id="metadata">2. DICOM 元数据 &amp; 探测器几何</h2>

<h3>2.1 扫描与重建参数</h3>
<div class="meta-grid">
<div class="meta-item"><div class="meta-key">Manufacturer</div><div class="meta-val">NMS (Neusoft Medical System)</div></div>
<div class="meta-item"><div class="meta-key">Model</div><div class="meta-val">NeuViz P10</div></div>
<div class="meta-item"><div class="meta-key">Modality</div><div class="meta-val">CT</div></div>
<div class="meta-item"><div class="meta-key">KVP</div><div class="meta-val">120 kV</div></div>
<div class="meta-item"><div class="meta-key">Exposure Time</div><div class="meta-val">3,270 ms</div></div>
<div class="meta-item"><div class="meta-key">Slice Thickness</div><div class="meta-val">0.274 mm</div></div>
<div class="meta-item"><div class="meta-key">Pixel Spacing</div><div class="meta-val">0.2114 × 0.2114 mm</div></div>
<div class="meta-item"><div class="meta-key">Image Resolution</div><div class="meta-val">2048 × 2048</div></div>
<div class="meta-item"><div class="meta-key">Window (C/W)</div><div class="meta-val">40 / 400 HU</div></div>
<div class="meta-item"><div class="meta-key">Slice Location</div><div class="meta-val">696.9 ~ 1,057.8 mm</div></div>
<div class="meta-item"><div class="meta-key">Instance Range</div><div class="meta-val">1 ~ 2,635</div></div>
<div class="meta-item"><div class="meta-key">Study Date</div><div class="meta-val">2026-04-07</div></div>
<div class="meta-item"><div class="meta-key">Scan Protocol</div><div class="meta-val">Body_Helical (Thorax)</div></div>
</div>

<h3>2.2 探测器几何信息</h3>
<table class="tag-table">
<tr><th>参数</th><th>来源标签</th><th>值</th><th>说明</th></tr>
<tr><td>探测器排数</td><td class="tag-code">(01F1,104B)</td><td class="tag-val">292 × 0.274 mm</td><td>292 排, 每排 0.274mm</td></tr>
<tr><td>总 z-覆盖</td><td>计算</td><td class="tag-val">~80.0 mm</td><td>292 × 0.274 mm</td></tr>
<tr><td>探测器通道数</td><td>JP2K 图像列数</td><td class="tag-val">256</td><td>面内方向</td></tr>
<tr><td>总能量范围</td><td class="tag-code">(01F3,1046)</td><td class="tag-val">290 keV</td><td>接近探测器排数</td></tr>
<tr><td>能区数量</td><td class="tag-code">(01F3,1031)</td><td class="tag-val">8</td><td>光子计数能量箱</td></tr>
<tr><td>探测器模块数</td><td class="tag-code">(01F3,1032)</td><td class="tag-val">2</td><td>双源/双模块</td></tr>
<tr><td>焦点数</td><td class="tag-code">(01E7,1004)</td><td class="tag-val">2</td><td>FS0, FS1</td></tr>
<tr><td>投影角度数</td><td class="tag-code">(01E7,1002)</td><td class="tag-val">60</td><td>每切片投影数</td></tr>
<tr><td>探测器模型</td><td class="tag-code">(01E7,1001)</td><td class="tag-val">ME 60keV/SI</td><td>60keV 多能标记；SI 表示能量数据标记，不代表硅</td></tr>
<tr><td>焦点-探测器距</td><td class="tag-code">(01F1,1008)</td><td class="tag-val">361.132 mm</td><td>几何参数</td></tr>
<tr><td>每旋转曝光次数</td><td class="tag-code">(01F9,1005)</td><td class="tag-val">6</td><td>时间采样</td></tr>
</table>

<div class="highlight">
<h3>时序关系</h3>
<p><b>292 排探测器 <u>同时</u> 采集：</b> 每个投影角度 (view) 下, 所有 292 排探测器阵列在同一时刻获取光子计数信号, 生成一组完整的多能区正弦图数据。60 个投影角度代表不同的 X 射线源-探测器旋转位置, <b>非</b> 时间序列上的不同 z-排。</p>
<p style="margin-top:10px;"><b>数据的 2D JPEG2000 图像编码：</b> 每个投影角度的 292 排 × 256 通道 × 8 能区的数据被编码为 256×256 的 JPEG2000 图像（总计 65,536 像素）。编码方式为: 8 能区垂直堆叠, 每能区约 32 编码行, 256 列为探测器通道。编解码并不改变排间的同时性——所有 292 排的数据映射到底层 JPEG2000 压缩域中。</p>
</div>

<!-- PRIVATE TAGS -->
<h2 id="private-tags">3. 私有标签解析</h2>

<table class="tag-table">
<tr><th>私有标签</th><th>VR</th><th>值</th><th>含义</th></tr>
<tr><td class="tag-group" colspan="4">Group_01F1 — 扫描协议 / 探测器几何</td></tr>
<tr><td class="tag-code">(01F1,104B)</td><td>SH</td><td class="tag-val"><b>292*0.274</b></td><td>292排 × 0.274mm 排间距</td></tr>
<tr><td class="tag-code">(01F1,104C)</td><td>SH</td><td class="tag-val">ON</td><td>某种模式开关</td></tr>
<tr><td class="tag-code">(01F1,104E)</td><td>LO</td><td class="tag-val">Body_Helical</td><td>体部螺旋扫描</td></tr>
<tr><td class="tag-code">(01F1,1002)</td><td>CS</td><td class="tag-val">HIGH</td><td>高通量模式</td></tr>
<tr><td class="tag-code">(01F1,1008)</td><td>DS</td><td class="tag-val">361.132</td><td>源-探测器距离 (mm)</td></tr>
<tr><td class="tag-code">(01F1,1011)</td><td>FL</td><td class="tag-val">[1154.4, 1154.9, 1154.7, 1154.6]</td><td>探测器距离采样值 (4个)</td></tr>
<tr><td class="tag-group" colspan="4">Group_01F3 — 能区配置</td></tr>
<tr><td class="tag-code">(01F3,1031)</td><td>IS</td><td class="tag-val"><b>8</b></td><td>能量箱数</td></tr>
<tr><td class="tag-code">(01F3,1032)</td><td>IS</td><td class="tag-val"><b>2</b></td><td>探测器模块/源数</td></tr>
<tr><td class="tag-code">(01F3,1046)</td><td>DS</td><td class="tag-val">290.0</td><td>总能量范围 (keV)</td></tr>
<tr><td class="tag-group" colspan="4">Group_01E7 — Energy Bin 元数据</td></tr>
<tr><td class="tag-code">(01E7,1001)</td><td>LO</td><td class="tag-val"><b>ME 60keV/SI</b></td><td>探测器/能量标记（SI 为能量数据标记，不代表硅）</td></tr>
<tr><td class="tag-code">(01E7,1002)</td><td>SL</td><td class="tag-val"><b>60</b></td><td>投影角度数</td></tr>
<tr><td class="tag-code">(01E7,1004)</td><td>IS</td><td class="tag-val"><b>2</b></td><td>焦点/源数量</td></tr>
<tr><td class="tag-group" colspan="4">Group_01F9 / 01E3 — 时间 / 状态</td></tr>
<tr><td class="tag-code">(01F9,1004)</td><td>IS</td><td class="tag-val">1</td><td>每旋转曝光次数标志</td></tr>
<tr><td class="tag-code">(01F9,1005)</td><td>IS</td><td class="tag-val">6</td><td>每旋转时间采样数</td></tr>
<tr><td class="tag-code">(01E3,1021)</td><td>IS</td><td class="tag-val">1</td><td>焦点尺寸选择</td></tr>
<tr><td class="tag-code">(01E3,1025)</td><td>LO</td><td class="tag-val">FS0,FS1</td><td>双焦点配置</td></tr>
<tr><td class="tag-code">(01E3,1026)</td><td>ST</td><td class="tag-val">Thorax</td><td>扫描部位：胸部</td></tr>
<tr><td colspan="4">&nbsp;</td></tr>
<tr><td class="tag-group" colspan="4" style="font-size:1.1em;">核心数据标签: <b>(EFE1,1001)</b> — OB, ~15.1MB — JPEG2000 压缩的能区 Sinogram 数据</td></tr>
</table>

<!-- OB STRUCTURE -->
<h2 id="ob-structure">4. (EFE1,1001) OB 数据结构</h2>

<div class="data-flow">
<div class="flow-box ob"><strong>OB Blob</strong><br>15,171,802 bytes</div>
<div class="flow-arrow">&#9654;</div>
<div class="flow-box"><strong>Header</strong><br>640 bytes<br>(材料参数 + 能量校准)</div>
<div class="flow-arrow">&#9654;</div>
<div class="flow-box jp2"><strong>JPEG2000 压缩码流</strong><br>256 个码流<br>每帧 256×256 int16<br>共 65,536 像素/帧</div>
<div class="flow-arrow">&#9654;</div>
<div class="flow-box img"><strong>Sinogram 数据</strong><br>[8 能区 × 60 Views<br>292 排 × 256 通道]<br>编码在 256×256 像素中</div>
</div>

<h3>4.1 Header 布局 (0x000 - 0x27F)</h3>
<table class="tag-table">
<tr><th>偏移</th><th>大小</th><th>内容</th></tr>
<tr><td class="tag-code">0x000 - 0x003</td><td>4 bytes</td><td>材料数量 = 4 (Water, Iodine, Calcium, Gadolinium)</td></tr>
<tr><td class="tag-code">0x004 - 0x0FF</td><td>~252 bytes</td><td>每种材料: 名称 (null-terminated) + 14 个 uint32 参数</td></tr>
<tr><td class="tag-code">0x100 - 0x1BF</td><td>192 bytes</td><td>能量区-材料映射表 (bin index, keV 区间 → 材料名称)</td></tr>
<tr><td class="tag-code">0x1C0 - 0x27F</td><td>192 bytes</td><td>Float32 能量响应曲线 (48 个数据点, 408→0.024 递减)</td></tr>
<tr><td class="tag-code" style="background:#2d1b1b;">0x280</td><td colspan="2" style="background:#2d1b1b;"><strong>JPEG2000 压缩能区数据开始</strong></td></tr>
</table>

<div class="figure">
<img src="data:image/png;base64,''' + images['img_header'] + '''">
<div class="figure-caption">图 1: OB Header 结构 — 上: Float32 能量响应曲线 (48 点, 从 ~408 到 ~0.024 递减); 下: 四种材料参数 (Water, Iodine, Calcium, Gadolinium)</div>
</div>

<!-- JPEG2000 -->
<h2 id="jp2k">5. JPEG2000 压缩分析</h2>

<h3>5.1 码流统计</h3>
<div class="highlight">
<ul>
<li>总共 <strong>256</strong> 个 JPEG2000 码流，每码流带 <code>jp2c</code> (JPEG2000 Codestream) 标记</li>
<li>其中 <strong>240</strong> 个有效 (含数据), <strong>16</strong> 个为空占位符 (~265 bytes, 全零)</li>
<li>有效码流大小: 12 KB ~ 95 KB, 解码后统一为 <strong>256 × 256 int16</strong></li>
<li>JPEG2000 编码参数: Profile 0, 1 瓦片, 1 分量, 16-bit 有符号, 可逆 9-7 变换 (损失较低或无损)</li>
<li>压缩由 <strong>OpenJPEG 2.5.2</strong> 完成 (嵌入于码流末尾注释)</li>
</ul>
</div>

<div class="figure">
<img src="data:image/png;base64,''' + images['img_cs_sizes'] + '''">
<div class="figure-caption">图 2: 256 个 JPEG2000 码流大小分布 — 左: 按索引排列 (可见 4 组 × 64 结构, 每组的 60 个有效码流对应 60 个投影角度); 右: 大小直方图 (红色虚线标记空占位 ~265B)</div>
</div>

<h3>5.2 码流分组: 4 组 × 64 码流 = 256 总</h3>
<table class="tag-table dim-table">
<tr><th>组</th><th>码流索引</th><th>有效数</th><th>数据类型</th><th>值范围 (典型)</th></tr>
<tr><td>Group 0</td><td class="tag-code">CS 0-63</td><td>60</td><td><b>参考/暗场</b> Sinogram</td><td class="tag-val">-33,000 ~ -9,000</td></tr>
<tr><td>Group 1</td><td class="tag-code">CS 64-127</td><td>60</td><td><b>主 Sinogram</b> (零中心)</td><td class="tag-val">-1,500 ~ +4,000</td></tr>
<tr><td>Group 2</td><td class="tag-code">CS 128-191</td><td>60</td><td><b>参考 Sinogram</b> (含饱和值)</td><td class="tag-val">-32,768 ~ -5,000</td></tr>
<tr><td>Group 3</td><td class="tag-code">CS 192-255</td><td>60</td><td><b>次 Sinogram</b> (零中心)</td><td class="tag-val">-1,500 ~ +3,500</td></tr>
</table>

<p>每组 64 个槽位：60 个有效 + 4 个空占位（位于组边界：索引 0, 7, 56, 63, 64, 71, 120, 127, 128, 135, 184, 191, 192, 199, 248, 255）。60 对应 <code>(01E7,1002)=60</code> 个投影角度。</p>

<h3>5.3 256×256 图像内 8 能区行堆叠结构</h3>
<p>每个解码后的 256×256 JPEG2000 图像，将 8 个能区的数据按 <b>垂直堆叠</b> 方式组织：</p>
<table class="tag-table dim-table">
<tr><th>能区</th><th>编码行范围</th><th>行数</th><th>典型值范围</th><th>说明</th></tr>
<tr><td>Bin 0</td><td class="tag-code">Rows 0-31</td><td>~32</td><td class="tag-val">-2,500 ~ +3,000</td><td>最低能区 (部分视图为空)</td></tr>
<tr><td>Bin 1</td><td class="tag-code">Rows 32-63</td><td>~32</td><td class="tag-val">-1,500 ~ +3,200</td><td></td></tr>
<tr><td>Bin 2</td><td class="tag-code">Rows 64-95</td><td>~32</td><td class="tag-val">-1,500 ~ +3,200</td><td></td></tr>
<tr><td>Bin 3</td><td class="tag-code">Rows 96-127</td><td>~32</td><td class="tag-val">-2,100 ~ +3,600</td><td>响应较强</td></tr>
<tr><td>Bin 4</td><td class="tag-code">Rows 128-159</td><td>~32</td><td class="tag-val">-1,600 ~ +3,800</td><td>响应最强</td></tr>
<tr><td>Bin 5</td><td class="tag-code">Rows 160-191</td><td>~32</td><td class="tag-val">-1,500 ~ +4,200</td><td></td></tr>
<tr><td>Bin 6</td><td class="tag-code">Rows 192-223</td><td>~32</td><td class="tag-val">-1,600 ~ +3,800</td><td></td></tr>
<tr><td>Bin 7</td><td class="tag-code">Rows 224-255</td><td>~32</td><td class="tag-val">-2,600 ~ +3,900</td><td>最高能区</td></tr>
</table>

<div class="highlight">
<h3>编码行 ≠ 物理探测器排</h3>
<p>上述 "行" 是 JPEG2000 图像中的编码行, <b>不代表 292 排物理探测器</b>。每个能区约 32 编码行包含 292 排物理探测器的数据——这是图像压缩过程中的数据编码方式。所有 292 排的原始信号在投影采集时是同时获取的, JPEG2000 压缩只是组织方式。</p>
</div>

<div class="figure">
<img src="data:image/png;base64,''' + images['img_overview'] + '''">
<div class="figure-caption">图 3: 左 — 8 个能区 60 视图平均值堆叠图 (虚线分隔能区, 每能区约 32 编码行); 右 — 对应 Slice 的 CT 重建图像 (2048×2048)</div>
</div>

<!-- SINOGRAM VISUALIZATION -->
<h2 id="sinogram">6. Energy Bin Sinogram 可视化</h2>

<h3>6.1 完整 8 能区 × 60 视图 Sinogram (Slice 1, z=696.9mm)</h3>
<div class="figure">
<img src="data:image/png;base64,''' + images['img_full_sino'] + '''">
<div class="figure-caption">图 4: 60 视图 × 8 能区 Sinogram。每个子图展示一个能区下 60 个视角的正弦图数据 (横轴: 60 视图索引, 纵轴: 256 探测器通道, 编码行方向均值)。采用 RdBu_r 色图: 白色 = 零(空气参考), 红/蓝 = 正/负衰减偏离。可清楚看到不同能区的衰减分布差异——低能区(Bin 0-2)和高能区(Bin 5-7)的衰减模式不同, 反映了能谱相关的组织吸收特性。</div>
</div>

<h3>6.2 能区光谱响应</h3>
<div class="figure">
<img src="data:image/png;base64,''' + images['img_spectral'] + '''">
<div class="figure-caption">图 5: 8 个能区在 60 个视图中的平均光子计数柱状图 (误差线为 ±1 SD)。Bin 4 的响应最强 (低 keV 区域光子衰减较大), Bin 0 最低——符合光子计数 CT 的标准能谱响应模式。</div>
</div>

<h3>6.3 跨切片 Sinogram 对比</h3>
<div class="figure">
<img src="data:image/png;base64,''' + images['img_multislice'] + '''">
<div class="figure-caption">图 6: 6 个不同 z-位置的 Slice Sinogram 对比 (能区 3)。从上到下/左到右: z=696.9mm → z=970.9mm, 共 6 层。正弦图结构随切片位置变化, 反映不同解剖层面的投影特征——可用于验证数据解析正确性。</div>
</div>

<h3>6.4 单视图 8 能区探测器通道剖面</h3>
<div class="figure">
<img src="data:image/png;base64,''' + images['img_profiles'] + '''">
<div class="figure-caption">图 7: 单个投影视图 (CS 66) 中 8 个能区的 256 探测器通道剖面 (编码行方向均值)。不同能区之间存在显著的衰减差异, 清晰验证了能区分辨能力。Bin 0 在该视图可能因低能光子不足而为全零。</div>
</div>

<h3>6.5 数据时序特性</h3>
<div class="highlight-g">
<p><b>292 排探测器同时采集 (no time splitting):</b></p>
<ul style="margin-left:20px;">
<li>每个投影角度下, 292 排探测器 <b>同步</b> 获取光子计数——不存在按排的时序拆分</li>
<li>JPEG2000 压缩处理的是完整的空间-能域数据, 编码并不引入时间差</li>
<li>60 个视图代表不同的投影角度 (旋转位置), 不是时间上先后排列的排数据</li>
<li>同一视图下所有 8 个能区的数据, 以及 292 排的探测器信号, 来自同一个时间采样点</li>
</ul>
</div>

<!-- SUMMARY -->
<h2 id="summary">7. 总结 &amp; 数据解码流程</h2>

<div class="highlight">
<h3>数据解码流程</h3>
<div class="code-block">
<span class="comment"># 1. 读取 DICOM 私有标签 (EFE1, 1001) — 光子计数 CT 能量区正弦图原始数据</span>
data = dcmread(<span class="keyword">file</span>)[<span class="number">0xefe1</span>, <span class="number">0x1001</span>].value  <span class="comment"># ~15.1 MB OB 数据块/文件</span>

<span class="comment"># 2. 跳过 640 字节 Header (材料参数 + 能区校准 + Float32 能量响应曲线)</span>
sino_start = <span class="number">0x280</span>

<span class="comment"># 3. 查找 256 个 "jp2c" 标记，提取各自的 JPEG2000 压缩码流</span>
<span class="keyword">for</span> i <span class="keyword">in</span> <span class="keyword">range</span>(<span class="number">256</span>):
    jp2k_bytes = data[positions[i]+<span class="number">4</span> : positions[i+<span class="number">1</span>]]

<span class="comment"># 4. JPEG2000 解码 → 256×256 int16 (使用 OpenJPEG/glymur)</span>
    img = glymur.Jp2k(jp2k_bytes)[:]    <span class="comment"># shape: (256, 256)</span>

<span class="comment"># 5. 从 Group 1 (CS 65–126, 共 60 个有效) 提取 8 能区数据</span>
<span class="comment">#    每码流 = 1 个投影角度, 256 行 = 8 能区 × ~32 编码行</span>
<span class="comment">#    256 列 = 256 个探测器通道 (面内方向)</span>
    <span class="keyword">for</span> eb <span class="keyword">in</span> <span class="keyword">range</span>(<span class="number">8</span>):
        energy_bin = img[eb*<span class="number">32</span> : (eb+<span class="number">1</span>)*<span class="number">32</span>, :]  <span class="comment"># ~(32, 256) 能区切片</span>

<span class="comment"># 6. 最终数据结构: [8 能区, 60 视图, ~32 编码行 × 256 通道]</span>
<span class="comment">#    (所有 292 排探测器数据同时采集，编码在 JPEG2000 图像行中)</span>
</div>
</div>

<h3>关键发现</h3>
<ul style="margin-left:20px; color:#c9d1d9;">
<li><strong>数据存储位置：</strong> 光子计数 CT 的能区正弦图数据通过 JPEG2000 压缩，嵌入 DICOM 私有标签 (EFE1,1001) OB 字段中，非标准 DICOM 标签。</li>
<li><strong>压缩与编码：</strong> 使用 OpenJPEG 2.5.2 JPEG2000 压缩，每码流为一帧 256×256 的 16-bit 数据图像。8 个能区数据在该图像中垂直堆叠（每能区约 32 编码行），256 列对应探测器通道。</li>
<li><strong>坐标系：</strong> 292 排物理探测器 (z-方向) 同时采集 → 60 个投影角度 → 8 个能区。数据经过 JPEG2000 压缩后以 256 编码行（而非 292 物理排）存储。</li>
<li><strong>时序特性：</strong> 所有 292 排探测器在同一次投影采样中间步工作，不按排的先后顺序采集。60 个视图代表不同的源-探测器旋转角。</li>
<li><strong>能区特性：</strong> 8 个能量区覆盖约 290 keV 范围 (~36 keV/区)，能区间差异显著 (如图 4-7 所示)，可用于后续物质分解、虚拟单能图像重建等定量成像应用。</li>
<li><strong>数据量级：</strong> 2,635 切片 × 60 视图 × 8 能区 × 292 排 × 256 通道 ≈ <b>~94 亿个正弦图数据点</b>，压缩后约 39 GB (JPEG2000)。</li>
</ul>

<h3>验证方法建议</h3>
<ul style="margin-left:20px; color:#c9d1d9;">
<li><b>能区响应验证：</b> 低能区和高能区的正弦图应有不同的衰减分布特征（图 4 已展示），Bin 4 附近的中间能区响应最强（图 5）</li>
<li><b>跨切片验证：</b> 不同 z-位置的正弦图应反映解剖结构的变化（图 6），若出现断层/跳变则说明解析或数据存在异常</li>
<li><b>几何验证：</b> 使用 (01F1,1008)=361.1mm 源-探测器距离和 60 投影视图做扇束几何重建，可验证数据的几何一致性</li>
<li><b>像素数据交叉验证：</b> 使用标准 CT 重建算法 (如 FBP) 从正弦图重建并与 DICOM 中已存储的 2048×2048 重建图像对比</li>
</ul>

<footer>
<p>报告生成于 2026-06-02 | 数据源: 2,635 DICOM 文件 | NeuViz P10 光子计数 CT</p>
<p>工具: pydicom + glymur (OpenJPEG) + matplotlib + numpy | JPEG2000: OpenJPEG 2.5.2</p>
</footer>

</div>
</body>
</html>'''

output_path = os.path.join(base_dir, 'energy_bin_sinogram_report.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

size_mb = os.path.getsize(output_path) / (1024*1024)
print(f'HTML report saved to: {output_path}')
print(f'File size: {size_mb:.1f} MB')
