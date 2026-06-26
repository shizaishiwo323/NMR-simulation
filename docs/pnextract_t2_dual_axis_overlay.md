# pnextract 孔径分布与 NMR T2 双轴对比图

本文档沉淀 Sample 16 图件调整后的复用经验。目标是把 pnextract 提取的孔径分布直方图、实验 NMR T2 反演谱和模拟平均 T2 谱画在同一张图中，同时避免把视觉对齐误写成物理换算结论。

## 推荐图式

优先使用“双独立横轴”：

- 下方横轴只表示 NMR `T2 (ms)`，用于实验 T2 和模拟 T2 曲线。
- 上方横轴只表示 pnextract 孔径 `pore diameter (um)`，用于孔径直方图。
- 两个横轴都使用 log scale。
- 孔径直方图使用 `_node2.dat` 中 pore volume 作为权重，显示 volume-weighted frequency。
- 为了让主峰在视觉上便于比较，可以只微调上方孔径轴的显示范围，使孔径主峰和 T2 主峰落在相近的画布位置。
- 不要缩放、平移或改写孔径值，也不要缩放、平移或改写 T2 值，除非用户明确要求做 display-only alignment。

这个图式的科学含义是“并列比较孔径分布和 T2 谱的位置关系”，不是证明二者已经通过某个唯一物理比例严格换算。

## 复用脚本

通用入口：

```powershell
python -m advanced_tools.pnextract_t2_dual_axis_overlay `
  --input-tiff "C:\path\to\segmented.tiff" `
  --pnextract-exe "C:\path\to\pnextract.exe" `
  --experiment-spectrum "C:\path\to\experiment_spectrum.csv" `
  --simulation-spectrum "C:\path\to\simulation_average_t2.csv" `
  --output-dir "simulation_outputs\sample_x_pnextract_t2_overlay" `
  --pore-value 2 `
  --solid-value 1 `
  --voxel-size-um 1.92 `
  --histogram-axis-mode top_pore_diameter `
  --xlim-min-ms 0.01 `
  --xlim-max-ms 100000
```

默认只绘制实验 T2 和 2D/平均模拟 T2；如果确实需要叠加 3D 模拟结果，再额外传入 `--simulation-3d-spectrum "C:\path\to\pygimli_3d_t2.csv"`。

已有 pnextract 输出时可跳过提取：

```powershell
python -m advanced_tools.pnextract_t2_dual_axis_overlay `
  --output-dir "simulation_outputs\sample_x_pnextract_t2_overlay" `
  --skip-pnextract `
  --histogram-axis-mode top_pore_diameter `
  --xlim-min-ms 0.01 `
  --xlim-max-ms 100000
```

如需手动微调上轴显示范围：

```powershell
python -m advanced_tools.pnextract_t2_dual_axis_overlay `
  --output-dir "simulation_outputs\sample_x_pnextract_t2_overlay" `
  --skip-pnextract `
  --histogram-axis-mode top_pore_diameter `
  --xlim-min-ms 0.01 `
  --xlim-max-ms 100000 `
  --top-axis-min-um 0.8 `
  --top-axis-max-um 3000
```

## 数据和单位约定

- pnextract 的输入二值体中，孔隙应映射为 `0`，固体应映射为 `1`。
- MHD 中记录 `threshold 0 0`，确保 pnextract 识别孔隙相。
- Windows pnextract 流程使用未压缩 `.raw`，避免 `.raw.gz` 兼容性问题。
- `_node2.dat` 中 pore radius 和 pore volume 按 SI 单位输出，脚本读取后转换为 `um` 和 `um3`：
  - `pore_radius_um = pore_radius_m * 1e6`
  - `pore_volume_um3 = pore_volume_m3 * 1e18`
  - `pore_diameter_um = 2 * pore_radius_um`
- 直方图必须用 `pore_volume_um3` 加权，而不是简单 pore count。

## 可视化对齐规则

默认使用 `--histogram-axis-mode top_pore_diameter`。

在这个模式下：

- T2 曲线画在下轴 `ax`。
- pnextract 孔径直方图画在上轴 `ax.twiny()`。
- 下轴范围由 `--xlim-min-ms` 和 `--xlim-max-ms` 控制。
- 上轴范围由 `--top-axis-min-um` 和 `--top-axis-max-um` 控制；如果没有给出 `--top-axis-min-um`，脚本会根据实验 T2 主峰和孔径直方图主峰自动计算一个上轴最小值，使两个主峰在 log 坐标中的视觉位置一致。
- 这种对齐只改变坐标轴显示范围，不改变任何孔径或 T2 数据值。

如果用户要求把孔径换算到等效 T2，才使用 `converted_t2` 模式，并在图注或 manifest 中记录公式和参数：

```text
d_um = 2 * rho_um_per_ms * T2_ms
```

其中 `rho_um_per_ms` 必须记录来源。没有明确物理依据时，这个换算只能作为近似展示，不应作为严格定量解释。

## 输出和复查

脚本会输出：

- `sample16_pnextract_pore_histogram_vs_t2.png`
- `pnextract_pore_radius_table.csv`
- `pnextract_pore_histogram_equivalent_t2.csv`
- `run_manifest.json`
- `pnextract_stdout.log`

`run_manifest.json` 中应保留输入 TIFF、phase label、voxel size、pnextract 路径、T2 谱路径、横轴范围、上轴范围和显示模式。

修改脚本后运行：

```powershell
python -m pytest tests/test_sample16_pore_t2_overlay.py -q
```
