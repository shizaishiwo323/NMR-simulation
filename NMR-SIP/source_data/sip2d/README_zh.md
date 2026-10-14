# 理想二维大孔-喉道-小孔周期链 SIP 结果包

- 状态：complete
- 模型维度：二维（每单位纵深的平移不变挤出体），不是三维。
- 求解器：`code/src/pore_scale_electrical_2d`（AC2D-periodic 有限体积，Niu 2020 机制的二维改写）。
- 运行命令：`python.exe code/scripts/sip_simulation/run_ideal_two_pore_channel_sip2d.py --output-dir results/ideal_two_pore_channel_sip2d_20261012 --workers 6`

## 包含内容

- `configs/`：模板配置副本与实际解析参数（JSON）。
- `simulation_sweeps/<run>/sweep_results.csv`：每个机制每个频点的有效复电导率、相位、残差、未知数和 LU 时间。
  直接 LU 求解每个频点只有一个相对残差，没有 Krylov 残差历史。
- `source_data/`：区域标签、网格、直流场、动态孔径表、网格收敛表、Debye 分解结果。
- `figures/`：几何与直流场、机制谱、Debye 半径分布。

## 不适用项

AGENTS.md 要求的三维数字岩心 HTML 与 pnextract 孔隙网络 HTML 不适用：本模型是解析定义的二维理想几何，
没有 CT 体数据，也没有提取网络。实际孔体/孔喉尺寸由解析几何给出（`source_data/dynamic_pore_size_table.csv`
中的 analytic_* 列），孔径分布直方图在 NMR 项目的联合结果中绘制。

## 参数口径

- 孔（Stern）极化 D+ = 3.8e-12 m2/s：用户指定（与 Debye 半径换算同值）；
  Niu 2020 原文为 1.3e-9 m2/s。
- 膜极化 D = 1.3e-09 m2/s、eta = 0.01、
  Sigma_S = 1.3e-09 S、sigma_w = 0.043 S/m：Niu 2020。
- 几何尺寸、网格与频带：本次理想几何的设计选择。
- `*_regional`：各区域独立 Eq.12 增量（二维显式几何主结果）；`*_uniform_niu`：Niu 三维原式的全局均匀增量（对照）。
