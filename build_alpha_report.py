import json, os
base_dir = r'D:\xl\00020006'
with open(os.path.join(base_dir, 'b64_alpha.json'), 'r') as f:
    img = json.load(f)

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Siemens PCD Coronary CTA — DICOM 解析报告</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI','Microsoft YaHei',sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.7; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 1.8em; color: #58a6ff; text-align: center; margin: 30px 0 10px; padding-bottom: 15px; border-bottom: 2px solid #30363d; }}
h2 {{ color: #f0883e; margin: 30px 0 15px; font-size: 1.4em; border-left: 4px solid #f0883e; padding-left: 12px; }}
h3 {{ color: #d2a8ff; margin: 20px 0 10px; }}
.tag-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.85em; }}
.tag-table th {{ background: #21262d; color: #8b949e; padding: 8px 12px; text-align: left; border: 1px solid #30363d; font-weight: 600; }}
.tag-table td {{ padding: 6px 12px; border: 1px solid #30363d; vertical-align: top; }}
.tag-table tr:nth-child(even) td {{ background: #161b22; }}
.tag-group {{ color: #f0883e; font-weight: bold; }}
.tag-code {{ color: #7ee787; font-family: 'Consolas',monospace; font-size: 0.9em; }}
.tag-val {{ color: #a5d6ff; }}
.kv {{ display: inline-block; background: #21262d; border-radius: 4px; padding: 1px 6px; margin: 0 2px; font-family: 'Consolas',monospace; font-size: 0.9em; }}
.highlight {{ background: #1f242b; border-left: 3px solid #f0883e; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 15px 0; }}
.highlight-g {{ background: #1f242b; border-left: 3px solid #7ee787; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 15px 0; }}
.highlight-r {{ background: #2d1b1b; border-left: 3px solid #da3633; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 15px 0; }}
.highlight-b {{ background: #1f242b; border-left: 3px solid #58a6ff; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 15px 0; }}
.figure {{ margin: 20px 0; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }}
.figure img {{ width: 100%; border-radius: 4px; }}
.figure-caption {{ color: #8b949e; font-size: 0.85em; margin-top: 8px; text-align: center; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; margin: 15px 0; }}
.stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; text-align: center; }}
.stat-num {{ font-size: 1.6em; color: #58a6ff; font-weight: bold; }}
.stat-label {{ color: #8b949e; font-size: 0.72em; margin-top: 4px; }}
.compare-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0; }}
.compare-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }}
.compare-card.p10 {{ border-color: #58a6ff; }}
.compare-card.siemens {{ border-color: #da3633; }}
ul {{ margin-left: 20px; line-height: 1.8; }}
footer {{ text-align: center; color: #484f58; margin: 40px 0 20px; font-size: 0.85em; }}
.toc {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 20px 0; }}
.toc a {{ color: #58a6ff; text-decoration: none; }}
.toc a:hover {{ text-decoration: underline; }}
.toc ol {{ margin-left: 20px; }} .toc li {{ margin: 3px 0; }}
</style>
</head>
<body>
<div class="container">

<h1>Siemens PCD Coronary CTA — DICOM 解析报告</h1>
<p style="text-align:center;color:#8b949e;">Siemens Healthineers | syngo.via VB80D | 光子计数 CT (PCD) | 23 后处理图像 | 无 raw energy bin 数据</p>

<div class="toc"><ol>
<li><a href="#overview">概览</a></li>
<li><a href="#metadata">DICOM 元数据</a></li>
<li><a href="#imtype">ImageType 解析</a></li>
<li><a href="#private">私有标签</a></li>
<li><a href="#vs-p10">与 P10 光子计数 CT 对比</a></li>
<li><a href="#conclusion">结论</a></li>
</ol></div>

<h2 id="overview">1. 概览</h2>

<div class="stats-grid">
<div class="stat-card"><div class="stat-num">23</div><div class="stat-label">DICOM 文件</div></div>
<div class="stat-card"><div class="stat-num">4</div><div class="stat-label">目录组</div></div>
<div class="stat-card"><div class="stat-num">512x512</div><div class="stat-label">图像分辨率</div></div>
<div class="stat-card"><div class="stat-num">0B</div><div class="stat-label">PixelData 大小</div></div>
<div class="stat-card"><div class="stat-num">1</div><div class="stat-label">私有标签组 (0x29)</div></div>
<div class="stat-card"><div class="stat-num">0</div><div class="stat-label">OB 能区数据块</div></div>
<div class="stat-card"><div class="stat-num">80/100/120</div><div class="stat-label">kVp 范围</div></div>
<div class="stat-card"><div class="stat-num">DERIVED</div><div class="stat-label">SECONDARY</div></div>
</div>

<div class="highlight-r">
<h3>核心结论</h3>
<p><b>alpha 目录不包含 energy bin sinogram 数据。</b>该设备为 Siemens 光子计数 CT (PCD), 但所有文件均为 syngo.via 后处理工作站生成的 <b>DERIVED/SECONDARY</b> 图像——无 PixelData、无 OB 标签、无 raw projection data。能量信息仅存在于 ImageType 元数据, 原始采集阶段的 energy bin 数据在处理后被丢弃, 未保留在这些 DICOM 文件中。</p>
</div>

<h2 id="metadata">2. DICOM 元数据</h2>

<table class="tag-table">
<tr><th>参数</th><th>值</th><th>说明</th></tr>
<tr><td>Manufacturer</td><td class="tag-val">Siemens Healthineers</td><td>西门子医疗</td></tr>
<tr><td>Model</td><td class="tag-val">syngo.via VB80D</td><td>后处理工作站 (非 CT 机)</td></tr>
<tr><td>Modality</td><td class="tag-val">CT</td><td></td></tr>
<tr><td>SOP Class</td><td class="tag-val">CT Image Storage</td><td>1.2.840.10008.5.1.4.1.1.2</td></tr>
<tr><td>Image Resolution</td><td class="tag-val">512 x 512</td><td></td></tr>
<tr><td>Pixel Size</td><td class="tag-val">0.058 mm</td><td>高分辨心脏成像</td></tr>
<tr><td>Slice Thickness</td><td class="tag-val">0.75 mm</td><td></td></tr>
<tr><td>Bits Allocated/Stored</td><td class="tag-val">16 / 12 (混合)</td><td>Siemens 典型位深</td></tr>
<tr><td>PixelData</td><td class="tag-val" style="color:#da3633;"><b>0 bytes (所有文件)</b></td><td>像素数据外存/不存</td></tr>
<tr><td>ImageType 标志</td><td class="tag-val">DERIVED / SECONDARY</td><td>所有文件均为衍生图像</td></tr>
<tr><td>StudyDate</td><td class="tag-val">2021-05-11</td><td></td></tr>
<tr><td>PatientID</td><td class="tag-val">CorCTA</td><td>冠脉 CTA</td></tr>
</table>

<h3>2.1 目录结构</h3>

<table class="tag-table">
<tr><th>目录</th><th>文件数</th><th>kVp</th><th>ImageType 特征</th></tr>
<tr><td><b>corcta</b></td><td align="center">8</td><td class="tag-val">80, 120</td><td><span class="kv">CT_SOM7 SPI DUAL</span> + <span class="kv">SNRG</span> + <span class="kv">DET_AB</span></td></tr>
<tr><td><b>ccta2</b></td><td align="center">5</td><td class="tag-val">100, 120</td><td><span class="kv">CT_SOM10 DET3D</span> + <span class="kv">T1</span> + <span class="kv">COUNT_2SRC</span> + <span class="kv">CONVCT</span></td></tr>
<tr><td><b>ccta3</b></td><td align="center">5</td><td class="tag-val">100, 120</td><td><span class="kv">CT_SOM10 DET3D</span> + <span class="kv">T1</span> + <span class="kv">COUNT_2SRC</span></td></tr>
<tr><td><b>ccta4</b></td><td align="center">5</td><td class="tag-val">120</td><td><span class="kv">VMI</span> + <span class="kv">ME60KEV</span> + <span class="kv">DEMEP</span> / <span class="kv">CONVCT</span></td></tr>
</table>

<div class="figure"><img src="data:image/png;base64,{img['img_kvp']}">
<div class="figure-caption">图 1: kVp 分布 — 80/100/120 kVp, 对应双源 CT 的多能采集</div></div>

<h2 id="imtype">3. ImageType 深度解析</h2>

<p>ImageType 是 DICOM 标准标签 <span class="tag-code">(0008,0008)</span>, 在 Siemens 双能 CT 后处理中承载能区分辨信息。</p>

<div class="figure"><img src="data:image/png;base64,{img['img_types']}">
<div class="figure-caption">图 2: ImageType 分类 — 23 个文件按后处理类型分为 4 类</div></div>

<h3>3.1 ImageType 标志编码表</h3>

<table class="tag-table">
<tr><th>标志</th><th>含义</th><th>出现位置</th></tr>
<tr><td class="tag-code">DERIVED</td><td>衍生图像 (非原始采集)</td><td>所有文件</td></tr>
<tr><td class="tag-code">SECONDARY</td><td>二级处理图像</td><td>所有文件</td></tr>
<tr><td class="tag-code">AXIAL</td><td>轴位重建</td><td>所有文件</td></tr>
<tr><td class="tag-code">CT_SOM7 SPI DUAL</td><td>Siemens SOMATOM 双能谱分解</td><td>corcta</td></tr>
<tr><td class="tag-code">CT_SOM10 DET3D</td><td>Siemens SOMATOM 3D 探测器处理</td><td>ccta2/3/4</td></tr>
<tr><td class="tag-code" style="color:#7ee787;"><b>SPI DUAL</b></td><td style="color:#7ee787;"><b>能谱双能</b> (Spectral Dual-Energy)</td><td>corcta</td></tr>
<tr><td class="tag-code" style="color:#7ee787;"><b>SNRG</b></td><td style="color:#7ee787;"><b>能谱能量</b> (Spectral Energy)</td><td>corcta</td></tr>
<tr><td class="tag-code" style="color:#7ee787;"><b>DET_AB</b></td><td style="color:#7ee787;"><b>探测器 A/B 通道</b></td><td>corcta</td></tr>
<tr><td class="tag-code" style="color:#58a6ff;"><b>VMI</b></td><td style="color:#58a6ff;"><b>虚拟单能图像</b> (Virtual Monoenergetic Image)</td><td>ccta4</td></tr>
<tr><td class="tag-code" style="color:#58a6ff;"><b>ME60KEV</b></td><td style="color:#58a6ff;"><b>60 keV 单能</b></td><td>ccta4</td></tr>
<tr><td class="tag-code" style="color:#58a6ff;"><b>DEMEP</b></td><td style="color:#58a6ff;"><b>双能物质分解</b> (Dual-Energy Material Decomposition)</td><td>ccta4</td></tr>
<tr><td class="tag-code">COUNT_2SRC</td><td>双源计数 (Dual-Source CT)</td><td>ccta2/3/4</td></tr>
<tr><td class="tag-code">CONVCT</td><td>常规 CT 等效图像</td><td>ccta2/4</td></tr>
<tr><td class="tag-code">THRESHOLD</td><td>阈值分割处理</td><td>ccta2/3/4</td></tr>
<tr><td class="tag-code">STD / T1</td><td>标准 / T1 阈值水平</td><td>corcta / ccta2/3/4</td></tr>
</table>

<h3>3.2 能区分辨路径对比 (同为 PCD，数据层级不同)</h3>

<div class="compare-grid">
<div class="compare-card p10">
<h3 style="color:#58a6ff;">P10 (Neusoft NViz P10) — 光子计数</h3>
<ul>
<li><b>探测器: </b>光子计数 PCD (ME 60keV/SI，SI 为能量数据标记而非硅元素)</li>
<li><b>能区: </b>8 个硬件能量箱</li>
<li><b>数据存储: </b>DICOM (EFE1,1001) OB 标签</li>
<li><b>数据格式: </b>JPEG2000 压缩 raw sinogram</li>
<li><b>能区提取: </b>直接解码 32 行/能区</li>
<li><b>K-edge: </b>硬件 bin 对 (28-33 / 33-38 keV)</li>
</ul>
</div>
<div class="compare-card siemens">
<h3 style="color:#da3633;">Alpha (Siemens Naeotom Alpha) — PCD 后处理</h3>
<ul>
<li><b>探测器: </b>光子计数 PCD <span style="font-size:0.8em;">(与 P10 同原理)</span></li>
<li><b>能区: </b>同 PCD 硬件, 但后处理中丢弃了原始 bin</li>
<li><b>数据存储: </b>无 raw sinogram</li>
<li><b>数据格式: </b>DERIVED 后处理图像</li>
<li><b>能区分辨: </b>已丢失 (仅 ImageType 元数据残留)</li>
<li><b>K-edge: </b>原始数据采集后已处理, 此批无</li>
</ul>
</div>
</div>

<div class="highlight-b">
<h3>ccta4 的 ME60KEV VMI — PCD 采集后的虚拟单能输出</h3>
<p>ccta4 目录中的 <span class="kv">VMI / ME60KEV / DEMEP</span> 图像来自 PCD 原始采集, 经 syngo.via 物质分解后处理生成 <b>虚拟 60 keV 单能图像</b>——与 P10 的 <span class="kv">ME 60keV/SI</span> 在临床靶输出上等价, 但数据层级不同；其中 <span class="kv">SI</span> 仅表示能量数据标记, 不代表硅元素。P10 保留了 raw 8-bin sinogram 可供解码, Siemens 的这批 DICOM 已丢弃原始能区数据, 仅保留后处理结果。</p>
</div>

<h2 id="private">4. 私有标签解析</h2>

<table class="tag-table">
<tr><th>标签</th><th>VR</th><th>值</th><th>含义</th></tr>
<tr><td class="tag-group" colspan="4">Group 0x29 — SIEMENS MEDCOM HEADER</td></tr>
<tr><td class="tag-code">(0029,0010)</td><td>LO</td><td class="tag-val">SIEMENS MEDCOM HEADER</td><td>私有 Creator</td></tr>
<tr><td class="tag-code">(0029,1040)</td><td>SQ</td><td class="tag-val">Application Header Sequence (1 item)</td><td>应用元数据</td></tr>
<tr><td colspan="4" style="font-size:0.8em; color:#8b949e;">Sequence 内包含:</td></tr>
<tr><td class="tag-code" style="padding-left: 20px;">(0029,1041)</td><td>CS</td><td class="tag-val">VIA_NO_VOLUME</td><td>非容积工作流</td></tr>
<tr><td class="tag-code" style="padding-left: 20px;">(0029,1042)</td><td>LO</td><td class="tag-val">NOT FOR VOLUME WORKFLOW</td><td>仅供平面查看</td></tr>
<tr><td class="tag-code" style="padding-left: 20px;">(0029,1043)</td><td>LO</td><td class="tag-val">V1 20120620</td><td>应用 Header 版本</td></tr>
<tr><td colspan="4">&nbsp;</td></tr>
<tr><td colspan="4" style="color:#8b949e; font-size:0.85em;"><b>注意:</b> 无大型二进制数据标签 (OB/OW/OF), 无能源正弦图数据。与 P10 的 (EFE1,1001) ~15MB OB blob 完全不同。</td></tr>
</table>

<h2 id="vs-p10">5. 与 P10 深度对比</h2>

<table class="tag-table">
<tr><th>维度</th><th style="color:#58a6ff;">P10 (00020006)</th><th style="color:#da3633;">Alpha (Siemens)</th></tr>
<tr><td>厂商</td><td class="tag-val">NMS (Neusoft)</td><td class="tag-val">Siemens Healthineers</td></tr>
<tr><td>设备</td><td class="tag-val">NeuViz P10 光子计数 CT</td><td class="tag-val">Naeotom Alpha 光子计数 CT<br>+ syngo.via 后处理</td></tr>
<tr><td>探测器类型</td><td class="tag-val">PCD 光子计数 (8 bins)</td><td class="tag-val">PCD 光子计数 (同原理)</td></tr>
<tr><td>文件类型</td><td class="tag-val">CT Image (主重建)</td><td class="tag-val" style="color:#da3633;">DERIVED / SECONDARY</td></tr>
<tr><td>文件数</td><td class="tag-val">2,635</td><td class="tag-val">23</td></tr>
<tr><td>图像分辨率</td><td class="tag-val">2048 x 2048</td><td class="tag-val">512 x 512</td></tr>
<tr><td>PixelData</td><td class="tag-val">8,388,608 bytes</td><td class="tag-val" style="color:#da3633;"><b>0 bytes</b></td></tr>
<tr><td>私有标签组</td><td class="tag-val">9 组 (0x00E1-0xEFE1)</td><td class="tag-val">1 组 (0x0029)</td></tr>
<tr><td>OB 能区数据</td><td class="tag-val" style="color:#7ee787;"><b>(EFE1,1001) 15.1MB</b></td><td class="tag-val" style="color:#da3633;"><b>无</b></td></tr>
<tr><td>能区数</td><td class="tag-val">8 (硬件 bins, 可解码)</td><td class="tag-val">PCD 硬件支持<br>但后处理中已丢弃</td></tr>
<tr><td>投影视图</td><td class="tag-val">1,920 / 转</td><td class="tag-val">N/A (后处理图像)</td></tr>
<tr><td>时序采样</td><td class="tag-val">60 Burst x 32 快视图 @ 6496Hz</td><td class="tag-val">无原始时序</td></tr>
<tr><td>压缩格式</td><td class="tag-val">JPEG2000 (OpenJPEG 2.5.2)</td><td class="tag-val">N/A</td></tr>
<tr><td>能量信息承载</td><td class="tag-val">硬编码在 pixel 值中</td><td class="tag-val" style="color:#da3633;">已丢失<br>(仅 ImageType 残留)</td></tr>
<tr><td>60keV 输出方式</td><td class="tag-val">PCD 直接能区分辨</td><td class="tag-val">VMI 后处理生成</td></tr>
</table>

<h2 id="conclusion">6. 结论</h2>

<div class="highlight-r">
<ul>
<li><b>alpha (Siemens Naeotom Alpha) 同样是光子计数 CT (PCD) 设备，</b>与 P10 探测器原理一致。但这批 DICOM 文件是 syngo.via 后处理生成的 <b>DERIVED/SECONDARY</b> 图像, PixelData 均被置空。</li>
<li><b>原始采集阶段的 energy bin 数据在 syngo.via 后处理中被丢弃，</b>未保留在这些 DICOM 中。能量信息仅以 ImageType 元数据标志 <span class="kv">SPI DUAL / SNRG / DET_AB</span> 和 <span class="kv">VMI / ME60KEV / DEMEP</span> 残留。</li>
<li><b>如需 energy bin raw data,</b> 需要从 Naeotom Alpha 的原始采集路径获取（非 syngo.via 处理后的 DICOM）。这与 P10 直接将 raw sinogram 写在 (EFE1,1001) 的做法不同。</li>
<li><b>ccta4 的 ME60KEV VMI</b> 在临床上与 P10 的 60keV 靶输出等效——两者起点都是 PCD, 差异在于 raw 数据是否保留在 DICOM 文件内。</li>
</ul>
</div>

<div class="highlight-b">
<h3>适用场景判断</h3>
<p><b>如需 raw energy bin sinogram:</b> P10 (Neusoft) 的 (EFE1,1001) 可直接解码; Siemens alpha 的这批文件不具备。<br>
<b>如需 PCD 后处理参考:</b> alpha 的 ImageType 标志展示了 Siemens PCD 后处理管线 (SPI_DUAL → DEMEP → VMI@60keV)。<br>
<b>如需 Siemens PCD 的 raw 数据:</b> 需获取 Naeotom Alpha 的原始采集数据 (非 syngo.via 处理过的 DERIVED 文件)。</p>
</div>

<footer>
<p>Siemens Alpha 分析报告 | 2026-06-02 | 数据源: 4 目录 23 文件 | syngo.via VB80D</p>
<p>工具: pydicom + matplotlib</p>
</footer>

</div>
</body>
</html>'''

output_path = os.path.join(base_dir, 'alpha_siemens_report.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)
size_mb = os.path.getsize(output_path) / (1024*1024)
print(f'Report saved: {output_path} ({size_mb:.1f} MB)')
