# 理想二维“大孔–狭长喉道–小孔”几何的 NMR + SIP 联合模拟

入口：`advanced_tools/ideal_two_pore_channel_nmr_sip.py`（本仓库）
SIP 二维求解器：`SIP模拟/sip模拟/code/src/pore_scale_electrical_2d/`（独立目录，见其 README）
SIP 运行脚本：`SIP模拟/sip模拟/code/scripts/sip_simulation/run_ideal_two_pore_channel_sip2d.py`

## 1. 几何与维度

- 物理模型：沿 x 的周期链 `… B – 喉道 – S – 喉道 – B …`，二维（每单位纵深，平移不变挤出体）。
  SIP 需要贯通的导电路径，因此“大圆–狭长通道–小圆”单元沿通道方向周期延拓，每个周期含 1 个大孔、
  1 个小孔和 2 段相同的喉道。
- NMR 在镜像对称半单元 `[B/2 – 喉道 – S/2]` 上求解（截线过两孔圆心、无通量），对该周期链是精确等价的；
  SIP 在完整周期单元上求解。两者使用同一套解析几何。
- 结果只能称为 **2D**，不能写成 2.5D 或 3D。

| 量 | 取值 | 说明 |
|---|---|---|
| 大孔半径 R_B | 20 µm | 孤立孔 T2 ≈ 1.21 s（低于 T2B = 3 s）；Stern 峰 ≈ 3.0e-3 Hz |
| 小孔半径 R_S | 5 µm | 孤立孔 T2 ≈ 0.43 s；Stern 峰 ≈ 4.8e-2 Hz |
| 喉道宽度 w | 0.1 µm（h = 0.05 µm） | 孤立狭缝 T2 ≈ 10 ms；Stern 峰 ≈ 4.8e2 Hz |
| 喉道长度 L | 15 µm（长宽比 150） | Titov 膜极化 tau_m = L²/4D ≈ 43 ms |
| 固相边距 | 2 µm | 使相邻链在 y 方向绝缘隔开 |

## 2. 尺寸选取依据（同时考虑 NMR 与 SIP 的响应窗口）

见输出 `figures/fig2_size_selection_windows.png`：

1. **NMR 窗口**：快扩散（kappa = rho2 R / D0 ≤ 0.05）；表面弛豫主导需 R/(2 rho2) 低于体弛豫 T2B = 3 s，
   即 R ≲ 20–30 µm；下限受回波间隔（TE = 0.25 ms，可分辨 T2 ≳ 1 ms）限制。
2. **SIP 窗口**：Stern 弛豫 f = D+/(pi r²)，D+ = 3.8e-12 m²/s；模拟频带 1e-4–1e5 Hz 对应 r ≈ 0.005–35 µm。
   三个尺寸 20 µm、5 µm、0.05 µm 的特征频率彼此相隔 1 个数量级以上；
   Titov 膜极化（L = 15 µm，体相 D = 1.3e-9 m²/s）位于两者之间（≈ 4 Hz）。
3. **NMR 耦合约束**：二维狭缝壁面对相邻孔是弛豫“汇”。汇的强度约为每个喉口 2·sqrt(D0 rho2 h)，
   对小孔 T2 的相对影响约 (2/pi)·l/R_S，其中 l = sqrt(D0 h / rho2) 是喉道内的扩散–弛豫长度。
   因此在 SIP 可解析的前提下取尽量窄的喉道（h = 0.05 µm，l ≈ 4.5 µm），并取较大的 R_S。
   图 2c 中解析汇模型与模拟小孔区域的直接拟合吻合。
4. **数值可解性**：喉道在 NMR 三角网格中跨宽 4 个单元，在 SIP 有限体积网格中跨宽 8 个单元（y 向局部加密）。

## 3. NMR 方法

- pyGIMLi `createPolygon/mergePLC/createMesh` 生成三角网格（孔体最大面积 0.04 µm²，喉道局部加密），
  固壁为 Robin（rho2），对称截线为无通量。
- 时间推进与 `Auto_NMR._pg_solve_step` 相同：隐式 Euler，
  `pg.solve(mesh, a=D dt, b=-(1+dt/T2B), f=u, bc={'Robin': {1: rho dt}})`。
  该系统等价于“左端一致质量矩阵、右端集中质量载荷”；脚本只分解一次 LU 并复用，
  每次运行都会与 `pg.solve` 逐步对比（记录于 manifest）。
- 参数：D0 = 2.0 µm²/ms、rho2 = 5 µm/s、T2B = 3000 ms（Auto_NMR 默认值，Zhou & Zhang 2025）；
  dt = TE = 0.25 ms，记录到 6000 ms。
- 参照：孤立大孔、小孔（圆盘 Bessel 解析解）与孤立狭缝（平板解析解），以及“未耦合”合成衰减。
- 网格收敛：粗/细两套网格，同一噪声实现。

## 4. T2 反演（NNLS + L-curve）

- `nmr_t2.nnls`（MATLAB 兼容 NNLS）与 `nmr_t2.lcurve.invert_single_signal_lcurve`（项目 L-curve 准则：
  alpha² eta / zeta → 0.25）。
- T2 网格 200 个对数点，1 ms – 1e5 ms（1 ms = 4 TE；0.01 ms 下限会让首批回波噪声产生约 0.7 ms 的伪峰）。
- alpha 扫描扩展到 1e-3 – 1e5：默认上限 1e2 时，1% 噪声数据在扫描范围内达不到 0.25 目标，会落在边界。
- 合成噪声：高斯，sigma = 0.1% M0（SNR = 1000，代表叠加采集，属采集设计选择）；
  1%（Auto_NMR 约定）作为敏感性算例同时给出。
- 正则化会把 6% 量级、与主峰相距约 4 倍的小孔峰推向较短 T2；因此用同样方法反演“未耦合”参照，
  并用小孔区域磁化的直接拟合给出不经反演的耦合 T2。

## 5. SIP 方法（二维改写）

- 控制方程、调和面电导、周期修正电势与 `sigma_eff = <J>/E0` 同 AC3D。
- 机制：界面（Maxwell–Wagner）、孔（Stern，Eq.17 + Eq.12）、膜（Titov 孔喉）、all（一次联合求解）。
- **主结果 `regional`**：每个孔区域用自身半径 r 与场加权动态孔径 Lambda_k 计算 Eq.12 增量；
  **对照 `uniform_niu`**：Niu 三维原式（面积加权 C* + 全局 Lambda）。
- 参数：sigma_w = 0.043 S/m、Sigma_S = 1.3e-9 S、eta = 0.01、膜极化 D = 1.3e-9 m²/s（Niu 2020）；
  孔极化 D+ = 3.8e-12 m²/s（用户指定，与半径换算同值；Niu 原文为 1.3e-9）。

## 6. 换算与联合图

- 实际尺寸：局部厚度（最大内切圆半径），对本几何是精确解析值：圆盘内处处为 R，喉道内为 h。
  按孔面积加权分为孔体、孔喉两类直方图。
- NMR：`r = 2 rho2 / (1/T2 − 1/T2B)`（圆截面形状因子；狭缝为 `h = rho2/(1/T2 − 1/T2B)`，
  即同一 T2 下狭缝映射为 2h）。为物理换算，不做显示对齐。
- SIP：Debye 分解（非负、二阶差分平滑、含 `i w eps_inf` 列），`r = sqrt(2 tau D+)`；
  sigma''(f) 按 `r = sqrt(D+/(pi f))` 绘制。
- 图 5（主图）：同一 y 轴上每条曲线除以自身最大值（单一纵轴，避免双 y 轴），直方图柱上标注真实面积分数；
  顶部两条横轴分别是 T2 与 SIP 频率，均为同一半径轴的精确换算。
  图 6：三联小图，给出绝对量。

## 7. 运行

```powershell
# 1) SIP（约 25 分钟，6 进程）
Set-Location 'C:\Users\imgw\Documents\Codex\SIP模拟\sip模拟'
$env:PYTHONNOUSERSITE = '1'
& 'C:\Users\imgw\.conda\envs\t2agent\python.exe' code/scripts/sip_simulation/run_ideal_two_pore_channel_sip2d.py --output-dir results/ideal_two_pore_channel_sip2d_20261012

# 2) NMR + 联合图（约 5 分钟；若 SIP 结果不存在会自动调用上一步）
Set-Location 'C:\Users\imgw\Documents\Codex\NMR模拟'
& 'C:\Users\imgw\.conda\envs\t2agent\python.exe' advanced_tools/ideal_two_pore_channel_nmr_sip.py
```

`t2agent` 环境同时含 pyGIMLi 与 SIP 所需依赖；需设 `PYTHONNOUSERSITE=1`，避免用户目录下与 numpy 2 不兼容的旧 scipy。

## 8. 解释边界

- 本结果是 2D 理想几何；狭缝壁对 NMR 的汇效应在 2D 中比 3D 圆柱喉道强得多。
- SIP 相位幅值（regional 下可达约 150 mrad）来自 0.1 µm 狭缝主导整个电阻且 Dukhin 数较大，
  不代表岩石典型值。
- Debye 分解的 `r = sqrt(2 tau D+)` 假定 Stern 机制；膜极化（体相扩散、长度 L）在该换算下出现在
  `L·sqrt(D+/2D) ≈ 0.57 µm` 的“表观尺寸”，不对应任何真实孔径。
- `tau = r²/(2D+)` 是 Niu/Schwarz 核函数的预设，不是从独立 Stern 层扩散方程得到的。
