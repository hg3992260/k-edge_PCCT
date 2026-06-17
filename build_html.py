import json, os

base_dir = r'D:\xl\00020006'

with open(os.path.join(base_dir, 'b64_images.json'), 'r') as f:
    images = json.load(f)

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>光子计数CT - Energy Bin Sinogram 解析报告</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.7; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 2em; color: #58a6ff; text-align: center; margin: 30px 0 10px; padding-bottom: 15px; border-bottom: 2px solid #30363d; }}
h2 {{ color: #f0883e; margin: 30px 0 15px; font-size: 1.5em; border-left: 4px solid #f0883e; padding-left: 12px; }}
h3 {{ color: #d2a8ff; margin: 20px 0 10px; font-size: 1.15em; }}
.meta-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 8px; margin: 15px 0; }}
.meta-item {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; }}
.meta-key {{ color: #8b949e; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; }}
.meta-val {{ color: #e6edf3; font-size: 1em; word-break: break-all; }}
.tag-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }}
.tag-table th {{ background: #21262d; color: #8b949e; padding: 8px 12px; text-align: left; border: 1px solid #30363d; font-weight: 600; }}
.tag-table td {{ padding: 6px 12px; border: 1px solid #30363d; }}
.tag-table tr:nth-child(even) td {{ background: #161b22; }}
.tag-group {{ color: #f0883e; font-weight: bold; }}
.tag-code {{ color: #7ee787; font-family: 'Consolas', 'Courier New', monospace; }}
.tag-val {{ color: #a5d6ff; }}
.highlight {{ background: #1f242b; border-left: 3px solid #f0883e; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 15px 0; }}
.code-block {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 15px; font-family: 'Consolas', 'Courier New', monospace; font-size: 0.85em; overflow-x: auto; white-space: pre-wrap; color: #7ee787; }}
.code-block .comment {{ color: #8b949e; }}
.code-block .keyword {{ color: #ff7b72; }}
.code-block .number {{ color: #a5d6ff; }}
.data-flow {{ display: flex; justify-content: center; align-items: center; gap: 15px; margin: 20px 0; flex-wrap: wrap; }}
.flow-box {{ background: #21262d; border: 2px solid #30363d; border-radius: 10px; padding: 12px 20px; text-align: center; min-width: 100px; }}
.flow-box.ob {{ border-color: #f0883e; }}
.flow-box.jp2 {{ border-color: #58a6ff; }}
.flow-box.img {{ border-color: #7ee787; }}
.flow-arrow {{ color: #58a6ff; font-size: 1.5em; }}
.figure {{ margin: 20px 0; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }}
.figure img {{ width: 100%; border-radius: 4px; }}
.figure-caption {{ color: #8b949e; font-size: 0.85em; margin-top: 8px; text-align: center; }}
.dim-table {{ margin: 15px auto; }}
.toc {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 20px 0; }}
.toc a {{ color: #58a6ff; text-decoration: none; }}
.toc a:hover {{ text-decoration: underline; }}
.toc ol {{ margin-left: 20px; }}
.toc li {{ margin: 5px 0; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin: 15px 0; }}
.stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center; }}
.stat-num {{ font-size: 2em; color: #58a6ff; font-weight: bold; }}
.stat-label {{ color: #8b949e; font-size: 0.8em; margin-top: 5px; }}
footer {{ text-align: center; color: #484f58; margin: 40px 0 20px; font-size: 0.85em; }}
</style>
</head>
<body>
<div class="container">

<h1>Photon-Counting CT / Energy Bin Sinogram 解析报告</h1>
<p style="text-align:center;color:#8b949e;">NeuViz P10 | 120kVp | 8个能区 | 60个投影角度 | 2635层切片</p>

<div class="toc">
<h3>目录</h3>
<ol>
<li><a href="#overview">概览</a></li>
<li><a href="#metadata">DICOM 元数据</a></li>
<li><a href="#private-tags">私有标签解析</a></li>
<li><a href="#ob-structure">(EFE1,1001) OB数据结构</a></li>
<li><a href="#jp2k">JPEG2000压缩分析</a></li>
<li><a href="#sinogram">Energy Bin Sinogram可视化</a></li>
<li><a href="#summary">总结</a></li>
</ol>
</div>

<!-- OVERVIEW -->
<h2 id="overview">1. 概览</h2>

<div class="stats-grid">
<div class="stat-card"><div class="stat-num">2,635</div><div class="stat-label">DICOM文件 (切片数)</div></div>
<div class="stat-card"><div class="stat-num">2048×2048</div><div class="stat-label">重建图像分辨率</div></div>
<div class="stat-card"><div class="stat-num">8</div><div class="stat-label">能量区 (Energy Bins)</div></div>
<div class="stat-card"><div class="stat-num">60</div><div class="stat-label">投影角度 (Views)</div></div>
<div class="stat-card"><div class="stat-num">32</div><div class="stat-label">探测器z排/Bin</div></div>
<div class="stat-card"><div class="stat-num">256</div><div class="stat-label">探测器通道</div></div>
<div class="stat-card"><div class="stat-num">256</div><div class="stat-label">JPEG2000码流/文件</div></div>
<div class="stat-card"><div class="stat-num">15.1 MB</div><div class="stat-label">OB数据块大小</div></div>
</div>

<div class="highlight">
<strong>设备:</strong> Neusoft NeuViz P10 光子计数CT<br>
<strong>扫描协议:</strong> 120kVp, Thorax, 螺距0.274mm, 2635 slices<br>
<strong>EDR能区:</strong> 8 bins (290 keV总范围, 约36 keV/Bin)<br>
<strong>数据存储:</strong> DICOM私有标签 (EFE1,1001), JPEG2000有损压缩<br>
<strong>Sinogram维度:</strong> [8能量区 × 60视图 × 32z排 × 256通道]
</div>

<p>本报告对DICOM目录下2,635个光子计数CT DICOM文件进行了完整的私有标签解析，提取并可视化了隐藏的能量区正弦图（Energy Bin Sinogram）数据。</p>

<!-- METADATA -->
<h2 id="metadata">2. DICOM 元数据</h2>
<h3>2.1 标准DICOM标签</h3>
<div class="meta-grid">
<div class="meta-item"><div class="meta-key">Manufacturer</div><div class="meta-val">NMS</div></div>
<div class="meta-item"><div class="meta-key">Model</div><div class="meta-val">NeuViz P10</div></div>
<div class="meta-item"><div class="meta-key">Modality</div><div class="meta-val">CT</div></div>
<div class="meta-item"><div class="meta-key">KVP</div><div class="meta-val">120</div></div>
<div class="meta-item"><div class="meta-key">Exposure Time</div><div class="meta-val">3270 ms</div></div>
<div class="meta-item"><div class="meta-key">Slice Thickness</div><div class="meta-val">0.274 mm</div></div>
<div class="meta-item"><div class="meta-key">Pixel Spacing</div><div class="meta-val">0.2114 × 0.2114 mm</div></div>
<div class="meta-item"><div class="meta-key">Rows/Columns</div><div class="meta-val">2048 × 2048</div></div>
<div class="meta-item"><div class="meta-key">Window Center/Width</div><div class="meta-val">40 / 400 HU</div></div>
<div class="meta-item"><div class="meta-key">SOP Class</div><div class="meta-val">1.2.840.10008.5.1.4.1.1.2</div></div>
<div class="meta-item"><div class="meta-key">Study Date</div><div class="meta-val">2026-04-07</div></div>
<div class="meta-item"><div class="meta-key">Patient ID</div><div class="meta-val">0000552570</div></div>
<div class="meta-item"><div class="meta-key">Slice Location</div><div class="meta-val">696.9 ~ 1057.8 mm</div></div>
<div class="meta-item"><div class="meta-key">Instance Number</div><div class="meta-val">1 ~ 2635</div></div>
</div>

<!-- PRIVATE TAGS -->
<h2 id="private-tags">3. 私有标签解析</h2>
<p>DICOM文件包含多个私有Creator组，关键能区参数集中在以下私有标签中：</p>

<table class="tag-table">
<tr><th>私有标签</th><th>VR</th><th>值</th><th>含义</th></tr>
<tr><td class="tag-group" colspan="4">Group_01E7 — Energy/Sinogram Metadata</td></tr>
<tr><td class="tag-code">(01E7,1001)</td><td>LO</td><td class="tag-val">ME 60keV/SI</td><td>探测器型号/能区模型（SI 为能量数据标记，不代表硅）</td></tr>
<tr><td class="tag-code">(01E7,1002)</td><td>SL</td><td class="tag-val">60</td><td>投影角度数 (Views)</td></tr>
<tr><td class="tag-code">(01E7,1004)</td><td>IS</td><td class="tag-val">2</td><td>探测器模块数</td></tr>
<tr><td class="tag-code">(01E7,1011)</td><td>UL</td><td class="tag-val">15,171,801</td><td>OB数据计数</td></tr>
<tr><td class="tag-group" colspan="4">Group_01F3 — Energy Bin Configuration</td></tr>
<tr><td class="tag-code">(01F3,1031)</td><td>IS</td><td class="tag-val">8</td><td>能量区数量</td></tr>
<tr><td class="tag-code">(01F3,1032)</td><td>IS</td><td class="tag-val">2</td><td>探测器物理排数</td></tr>
<tr><td class="tag-code">(01F3,1046)</td><td>DS</td><td class="tag-val">290.0</td><td>总能量范围 (keV)</td></tr>
<tr><td class="tag-group" colspan="4">Group_01F1 — Scan Parameters</td></tr>
<tr><td class="tag-code">(01F1,1002)</td><td>CS</td><td class="tag-val">HIGH</td><td>扫描模式</td></tr>
<tr><td class="tag-code">(01F1,1008)</td><td>DS</td><td class="tag-val">361.132</td><td>焦点-探测器距离 (mm)</td></tr>
<tr><td class="tag-code">(01F1,1093)</td><td>IS</td><td class="tag-val">4</td><td>kVp等级数</td></tr>
<tr><td class="tag-group" colspan="4">Group_01E3 / 01F9 — 扫描状态</td></tr>
<tr><td class="tag-code">(01E3,1021)</td><td>IS</td><td class="tag-val">1</td><td>焦点尺寸标志</td></tr>
<tr><td class="tag-code">(01E3,1025)</td><td>LO</td><td class="tag-val">FS0,FS1</td><td>焦点尺寸名称</td></tr>
<tr><td class="tag-code">(01E3,1026)</td><td>ST</td><td class="tag-val">Thorax</td><td>扫描部位</td></tr>
<tr><td class="tag-code">(01F9,1005)</td><td>IS</td><td class="tag-val">6</td><td>每旋转曝光次数</td></tr>
<tr><td colspan="4">&nbsp;</td></tr>
<tr><td class="tag-group" colspan="4" style="font-size:1.1em;">核心数据标签: (EFE1,1001) — OB型, ~15.1MB, JPEG2000压缩的能区Sinogram</td></tr>
</table>

<!-- OB STRUCTURE -->
<h2 id="ob-structure">4. (EFE1,1001) OB 数据结构</h2>

<div class="data-flow">
<div class="flow-box ob"><strong>OB Blob</strong><br>15,171,802 bytes</div>
<div class="flow-arrow">&#9654;</div>
<div class="flow-box"><strong>Header</strong><br>640 bytes<br>(材料参数+能量校准)</div>
<div class="flow-arrow">&#9654;</div>
<div class="flow-box jp2"><strong>JPEG2000</strong><br>256 个压缩码流<br>每帧 256×256 int16</div>
<div class="flow-arrow">&#9654;</div>
<div class="flow-box img"><strong>Sinogram</strong><br>[8 Bins × 60 Views<br>× 32 Rows × 256 Ch]</div>
</div>

<h3>4.1 Header 结构 (0x000 - 0x27F)</h3>
<table class="tag-table">
<tr><th>偏移</th><th>大小</th><th>内容</th></tr>
<tr><td class="tag-code">0x000 - 0x003</td><td>4 bytes</td><td>材料数量 = 4 (Water, Iodine, Calcium, Gadolinium)</td></tr>
<tr><td class="tag-code">0x004 - 0x0FF</td><td>~252 bytes</td><td>每种材料: 名称(null-terminated) + 14个uint32参数</td></tr>
<tr><td class="tag-code">0x100 - 0x1BF</td><td>192 bytes</td><td>能量区校准映射表 (bin index → keV范围 → 材料)</td></tr>
<tr><td class="tag-code">0x1C0 - 0x27F</td><td>192 bytes</td><td>Float32 能量响应曲线 (48个数据点)</td></tr>
<tr><td class="tag-code">0x280</td><td colspan="2"><strong>Sinogram数据开始</strong></td></tr>
</table>

<div class="figure">
<img src="data:image/png;base64,{images['img_header']}" alt="Header Structure">
<div class="figure-caption">图1: OB Header结构 — 上: Float32能量响应曲线 (48点, 从~408到~0.024递减); 下: 四种材料参数 (Water, Iodine, Calcium, Gadolinium)</div>
</div>

<!-- JPEG2000 -->
<h2 id="jp2k">5. JPEG2000 压缩分析</h2>

<h3>5.1 码流统计</h3>
<div class="highlight">
<ul>
<li>总共 <strong>256</strong> 个 JPEG2000 码流, 每码流带 <code>jp2c</code> 标记</li>
<li>其中 <strong>240</strong> 个有效 (含sinogram数据)</li>
<li>其中 <strong>16</strong> 个为空占位符 (~265 bytes, 全零数据)</li>
<li>有效码流大小: 12KB ~ 95KB, 解码后统一为 <strong>256×256 int16</strong></li>
<li>JPEG2000参数: Profile 0, 1瓦片, 1分量, 16-bit有符号, 无损可逆变换</li>
</ul>
</div>

<div class="figure">
<img src="data:image/png;base64,{images['img_cs_sizes']}" alt="Codestream Sizes">
<div class="figure-caption">图2: 256个JPEG2000码流大小分布 — 左: 按索引排列(可见4组×64结构); 右: 大小直方图</div>
</div>

<h3>5.2 码流分组</h3>
<p>256个码流分为 <strong>4组 × 64</strong>:</p>
<table class="tag-table dim-table">
<tr><th>组</th><th>码流索引</th><th>有效数</th><th>数据类型</th><th>值范围 (典型)</th></tr>
<tr><td>Group 0</td><td class="tag-code">CS 0-63</td><td>60</td><td>参考/校准 Sinogram</td><td class="tag-val">-33,000 ~ -9,000</td></tr>
<tr><td>Group 1</td><td class="tag-code">CS 64-127</td><td>60</td><td>主Sinogram (零中心)</td><td class="tag-val">-1,500 ~ +4,000</td></tr>
<tr><td>Group 2</td><td class="tag-code">CS 128-191</td><td>60</td><td>参考 Sinogram (含饱和)</td><td class="tag-val">-32,768 ~ -5,000</td></tr>
<tr><td>Group 3</td><td class="tag-code">CS 192-255</td><td>60</td><td>次Sinogram (零中心)</td><td class="tag-val">-1,500 ~ +3,500</td></tr>
</table>

<p>每组64个槽位中, 60个有效 + 4个空占位 (位于每组边界: 如索引56,63,64,71等)。这对应 <strong>(01E7,1002)=60</strong> 个投影角度。</p>

<h3>5.3 256×256图像内的8能区堆叠</h3>
<p>每个256×256的图像按行划分为8个能区, 每能区占用 <strong>32行</strong>:</p>
<table class="tag-table dim-table">
<tr><th>能区</th><th>行范围</th><th>典型值范围</th><th>说明</th></tr>
<tr><td>Bin 0</td><td class="tag-code">Rows 0-31</td><td class="tag-val">-2,500 ~ +3,000</td><td>最低能区</td></tr>
<tr><td>Bin 1</td><td class="tag-code">Rows 32-63</td><td class="tag-val">-1,500 ~ +3,200</td><td></td></tr>
<tr><td>Bin 2</td><td class="tag-code">Rows 64-95</td><td class="tag-val">-1,500 ~ +3,200</td><td></td></tr>
<tr><td>Bin 3</td><td class="tag-code">Rows 96-127</td><td class="tag-val">-2,100 ~ +3,600</td><td>响应较强</td></tr>
<tr><td>Bin 4</td><td class="tag-code">Rows 128-159</td><td class="tag-val">-1,600 ~ +3,800</td><td>响应最强</td></tr>
<tr><td>Bin 5</td><td class="tag-code">Rows 160-191</td><td class="tag-val">-1,500 ~ +4,200</td><td></td></tr>
<tr><td>Bin 6</td><td class="tag-code">Rows 192-223</td><td class="tag-val">-1,600 ~ +3,800</td><td></td></tr>
<tr><td>Bin 7</td><td class="tag-code">Rows 224-255</td><td class="tag-val">-2,600 ~ +3,900</td><td>最高能区</td></tr>
</table>

<p>每张256×256图像代表<strong>一个投影角度</strong>，其中：
<ul>
<li><strong>列 (256)</strong> = 探测器通道 (detector channels)</li>
<li><strong>行 (256)</strong> = 8个能区 × 32个探测器z排 = 256</li>
</ul>
</p>

<!-- SINOGRAM VISUALIZATION -->
<h2 id="sinogram">6. Energy Bin Sinogram 可视化</h2>

<h3>6.1 完整 8 能区 × 60 视图 Sinogram</h3>
<div class="figure">
<img src="data:image/png;base64,{images['img_full_sino']}" alt="Full Sinogram">
<div class="figure-caption">图3: 完整60视图 × 8能区 Sinogram (Slice 1, z=696.9). 每个子图是一个能区的60视图sinogram (横轴: 60视图, 纵轴: 256探测器通道,z方向均值). 采用RdBu_r色图, 零值白色, 正/负值分别对应高/低于空气参考的衰减.</div>
</div>

<h3>6.2 能区光谱响应</h3>
<div class="figure">
<img src="data:image/png;base64,{images['img_spectral']}" alt="Spectral Response">
<div class="figure-caption">图4: 8个能区在60个视图中的平均响应柱状图 (误差线为±1标准差). 显示能区0-7的光子计数分布差异.</div>
</div>

<h3>6.3 跨切片 Sinogram 对比</h3>
<div class="figure">
<img src="data:image/png;base64,{images['img_multislice']}" alt="Multi-slice Sinogram">
<div class="figure-caption">图5: 6个不同z轴位置的Slice Sinogram对比 (能区3, 6个所选视图). 从上到下/左到右: z=696.9mm → z=970.9mm.</div>
</div>

<h3>6.4 单视图 8 能区探测器通道剖面</h3>
<div class="figure">
<img src="data:image/png;base64,{images['img_profiles']}" alt="Channel Profiles">
<div class="figure-caption">图6: 单个投影视图 (CS 66) 中 8个能区的探测器通道剖面 (z方向均值). 可见不同能区的衰减特征存在显著差异, Bin 0为全零 (该视图该能区为空).</div>
</div>

<h3>6.5 总览: Stacked Sinogram + CT 重建图像</h3>
<div class="figure">
<img src="data:image/png;base64,{images['img_overview']}" alt="Overview">
<div class="figure-caption">图7: 左 — 8个能区在60个视图上的平均值堆叠图 (虚线分隔能区); 右 — 对应Slice的CT重建图像 (2048×2048, uint16, 窗位56000-58000).</div>
</div>

<!-- SUMMARY -->
<h2 id="summary">7. 总结</h2>

<div class="highlight">
<h3>数据解析流程</h3>
<div class="code-block">
<span class="comment"># 1. 读取DICOM私有标签</span>
data = dcmread(<span class="keyword">file</span>)[<span class="number">0xefe1</span>, <span class="number">0x1001</span>].value    <span class="comment"># ~15.1MB OB数据块</span>

<span class="comment"># 2. 跳过640字节Header (材料参数+能量校准)</span>
sino_start = <span class="number">0x280</span>

<span class="comment"># 3. 找到256个"jp2c"标记, 提取JPEG2000码流</span>
<span class="keyword">for</span> i <span class="keyword">in</span> <span class="keyword">range</span>(<span class="number">256</span>):
    jp2k_bytes = data[positions[i]+<span class="number">4</span> : positions[i+<span class="number">1</span>]]
    
<span class="comment"># 4. JPEG2000解码 → 256×256 int16 图像</span>
    img = glymur.Jp2k(jp2k_bytes)[:]    <span class="comment"># shape: (256, 256)</span>
    
<span class="comment"># 5. 提取8个能区 (每Bin 32行)</span>
    <span class="keyword">for</span> eb <span class="keyword">in</span> <span class="keyword">range</span>(<span class="number">8</span>):
        energy_bin = img[eb*<span class="number">32</span> : (eb+<span class="number">1</span>)*<span class="number">32</span>, :]    <span class="comment"># (32, 256)</span>

<span class="comment"># 6. 从Group1 (CS 65-126, 60个有效) 获得完整数据</span>
<span class="comment"># 最终维度: [8能区, 60视图, 32z排, 256通道]</span>
</div>
</div>

<h3>关键发现</h3>
<ul style="margin-left:20px; color:#c9d1d9;">
<li><strong>数据存储:</strong> 光子计数CT的能区正弦图数据被压缩为JPEG2000码流，嵌入在DICOM私有标签 (EFE1,1001) OB数据类型中，并非标准DICOM标签。</li>
<li><strong>压缩方式:</strong> 使用OpenJPEG 2.5.2进行JPEG2000压缩，每个码流代表一个投影角度视角下所有8个能区的数据，以行堆叠方式组织。</li>
<li><strong>数据维度:</strong> 每个DICOM文件包含完整的 [8能区 × 60视图 × 32z排 × 256通道] 正弦图数据，对应该z位置的CT切片。</li>
<li><strong>能区分布:</strong> 8个能量区覆盖约290 keV范围，每能区有32个探测器物理排。不同能区的衰减特征呈现明显差异，可用于材料分解和虚拟单能图像重建。</li>
<li><strong>数据量:</strong> 2635个切片 × 60视图 × 8能区 × 32排 × 256通道 = <strong>约103亿个正弦图数据点</strong>，压缩后约39GB (JPEG2000) 或原始约82GB (int16)。</li>
</ul>

<h3>验证方法</h3>
<p>可通过以下方式验证解析结果：</p>
<ul style="margin-left:20px; color:#c9d1d9;">
<li>对比不同能区的正弦图：低能区和高能区应有不同的衰减分布特征 (如上图3所示)</li>
<li>跨切片对比：不同z位置的正弦图应显示解剖结构的变化 (如上图5所示)</li>
<li>能区响应曲线：Bin 4-5应显示较大的方差/响应 (如图4)，符合临床光子计数CT的典型能区分布</li>
<li>OP排空位：Bin 0在某些视图(如CS 66)中为全零是正常现象 (边缘能区可能无数据)</li>
</ul>

<footer>
<p>报告生成于 2026-06-02 | 数据源: D:\\xl\\00020006\\00020006\\ (2635 DICOM files)</p>
<p>工具: pydicom + glymur (OpenJPEG) + matplotlib + numpy</p>
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
