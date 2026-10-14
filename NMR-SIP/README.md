# NMR-SIP：理想二维“大孔–喉道–小孔”的 NMR 与 SIP 联合模拟

本目录汇总理想二维孔隙几何上 NMR 与 SIP 联合研究的文档、PPT、图件和汇总数据。
模拟与绘图代码仍放在本仓库 `advanced_tools/`（Python 包内相互 import，不能挪动）；
SIP 二维求解器在 SIP 仓库 [pore-scale-sip-simulation-reproduction](https://github.com/shizaishiwo323/pore-scale-sip-simulation-reproduction)
的 `code/src/pore_scale_electrical_2d/`。

**全部结果都是 2D（每单位纵深）**，不能写成 2.5D 或 3D。

## 目录

| 路径 | 内容 |
|---|---|
| `docs/ideal_two_pore_channel_nmr_sip.md` | 方法说明：几何、尺寸选取依据（同时考虑 NMR 与 SIP 的响应窗口）、NMR 与 SIP 求解设置 |
| `docs/nmr_analytic_solution/` | **孤立孔**：圆盘/狭缝的精确解（Bessel/平板）与快扩散近似、代入实际参数的求解过程、由 T2 反算孔径、四种 T2 提取方式比较。`*.tex` → `*.pdf`；`ppt/` 下为 PPT 及其生成脚本 |
| `docs/coupled_pore_backcalc/` | **耦合孔（推导版）**：20 个喉道宽/长算例的分区判据（Ψ_eff、η）、孔–喉–孔一维（TL）模型、5 个代表算例代入求解、四条反算路线、SNR 影响。`*.tex` → `*.pdf`；`ppt/` 下为 PPT |
| `docs/coupled_pore_backcalc/c3_intuitive/` | **耦合孔（通俗版）**：只讲算例 C3（w = 1 µm，L = 15 µm），每个公式按“形象理解 → 公式含义 → 代入数值 → 结论”展开；含喉道磁化分布与普适曲线的来历 |
| `figures/channel_nmr_sip/` | 周期链 B–喉道–S（w = 0.1 µm）：几何网格、尺寸窗口、NMR 衰减与 T2、SIP 机制与 Debye 分解、实际孔径直方图 + NMR T2 + SIP 叠加图（fig1–fig7） |
| `figures/throat_sweep_lcurve/` | 20 个喉道算例（w = 0.1–4 µm × L = 2–30 µm）的 L-curve NNLS 叠加网格、耦合控制量图（图例已修正）、几何网格 |
| `figures/throat_sweep_brd/` | 同 20 个算例用 MATLAB `brd1.p`（α = 0.1）重新反演的结果 |
| `figures/coupled_backcalc/` | 耦合孔反算分析图（分区、模型验证、反算孔径、SNR、快交换非唯一性） |
| `figures/sip2d/` | SIP 二维求解器结果：几何与直流电场、机制谱、Debye 半径分布 |
| `source_data/` | 上述图件对应的汇总 CSV/JSON 与运行清单（manifest），按同名子目录组织 |

这里的图件和汇总数据是 `simulation_outputs/`（以及 SIP 仓库 `results/`）中运行结果的**快照**；
这两个目录按仓库约定只保留在本地、不进 Git。重新运行脚本会刷新本地输出，若要更新这里需手动复制。

## 生成方式（在 NMR 仓库根目录运行，依赖项目的 Python 环境）

| 步骤 | 命令 | 输出（本地） |
|---|---|---|
| SIP 二维求解（在 SIP 仓库根目录） | `python code/scripts/sip_simulation/run_ideal_two_pore_channel_sip2d.py --output-dir results/ideal_two_pore_channel_sip2d_20261012` | `results/ideal_two_pore_channel_sip2d_20261012/` |
| 周期链 NMR + 读取 SIP + 联合图 | `python advanced_tools/ideal_two_pore_channel_nmr_sip.py` | `simulation_outputs/ideal_two_pore_channel_nmr_sip_20261012/` |
| 只含 NMR 的叠加图（fig7） | `python advanced_tools/plot_ideal_two_pore_channel_nmr_overlay.py` | 同上 `figures/` |
| 喉道宽/长扫参（每个算例单独保存，超过 10 分钟跳过） | `python advanced_tools/ideal_two_pore_throat_sweep_nmr.py` | `simulation_outputs/ideal_two_pore_throat_sweep_nmr_20261012/` |
| 扫参 Nature 风格图 | `python advanced_tools/ideal_two_pore_nature_plots.py sweep` | 同上 `figures_nature/` |
| BRD 重新反演 | `python advanced_tools/ideal_two_pore_sweep_brd.py --alpha 0.1`，再按脚本说明运行 `ideal_two_pore_nature_plots.py sweep --sweep-dir ...` | `simulation_outputs/ideal_two_pore_throat_sweep_nmr_brd_20261013/` |
| 耦合孔反算分析 | `python advanced_tools/coupled_two_pore_backcalc.py`，再 `python advanced_tools/plot_coupled_two_pore_backcalc.py` | `simulation_outputs/coupled_two_pore_backcalc_20261013/` |
| 推导版 PDF/PPT 数值 | `python NMR-SIP/docs/coupled_pore_backcalc/make_numbers.py` | `generated_*.tex`、`numbers.json` |
| 通俗版图件 | `python NMR-SIP/docs/coupled_pore_backcalc/c3_intuitive/make_figures.py` | `c3_intuitive/figures/` |
| PDF | 在对应目录 `xelatex <name>.tex`（两次；需 xeCJK 与 Microsoft YaHei） | `<name>.pdf` |
| PPT | 在 `ppt/` 下 `python build_equations.py`，再 `node build_deck.js`（需 pptxgenjs，环境变量 `PPTX_SKILL_DIR` 指向 pptx skill 以调用 `apply_theme.js`） | `ppt/*.pptx` |

## 主要参数与结论

- **参数**：D = 2 µm²/ms，ρ₂ = 0.005 µm/ms，T2B = 3000 ms（与 `Auto_NMR.py` 默认一致，来源记为 Zhou & Zhang, 2025，需核对）；
  R_B = 20 µm，R_S = 5 µm。NNLS 为 L-curve（α 扫描 1e-3–1e5），200 个 T2 箱 1–1e5 ms，SNR 1000。
  SIP：σ_w = 0.043 S/m，Σ_S = 1.3e-9 S，D⁺ = 3.8e-12 m²/s（Stern），膜极化 D = 1.3e-9 m²/s，η = 0.01。
- **孤立孔**：快扩散近似与精确解相差 < 1%；单孔分别反演可恢复约 5 µm / 20 µm；
  在复合信号中，主峰旁的小峰会因正则化偏差而被低估。
- **耦合孔**：
  - 喉道同时是交换通道和弛豫“汇”。小孔信号保留比 η = A_f/p_S 随有效交换数 Ψ_eff 连续下降，分为三区：
    区 I（η > 0.8）4 例、区 II（0.1–0.8）8 例、区 III（< 0.1）8 例。
  - 孔–喉–孔一维模型与全二维有限元的衰减最大偏差 0.65%。
  - 已知喉道几何时，区 I–II 的反算结果为 r_S 4.6–4.8 µm、r_B 19.6–20.1 µm（真值 5 / 20）。
    区 III 只能得到等效半径 R_eff = (r_B² + r_S²)/(r_B + r_S)。
  - 原图“一个峰”的判断是探测门槛造成的：20 个算例的 NNLS 结果中小孔模态都存在，只是幅值低于门槛。
- **BRD（α = 0.1）**：20 个算例全部只给出一个峰，这是 α = 0.1 的平滑强度造成的，不能解释为扩散耦合。

## 需要核对的事项

- Zhou & Zhang (2025, Water Resources Research)：作者全名、卷期、DOI；Fleury & Soualem (2009)：卷期页码。
- SIP 的 D⁺ = 3.8e-12 m²/s 取自文献（Schwarz, 1962; Nordsiek & Weller, 2008; Zhang et al., 2018），不是本样品实测值。
- 三维样品中，圆柱喉道的“交换”与“汇”都比二维狭缝弱，Ψ 的几何因子需要重新计算。
