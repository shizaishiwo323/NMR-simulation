# HANDOFF — NMR 项目（NMR-simulation）

更新时间：2026-10-14 04:43（本机时间，UTC+01:00）
更新人：Claude Code 会话“NMR考虑耦合的知识分解”，汇总了“NMR/SIP二维几何模拟与反演”和“NMR的知识分解”两个会话的工作

## 1. 最近的工作主线

理想二维“大孔 B – 狭长喉道 – 小孔 S”几何上的 NMR + SIP 联合模拟，以及由 NMR T2 反算孔径（孤立孔 → 耦合孔）。
所有成果文档、PPT、图件快照和汇总数据都放在 **`NMR-SIP/`**，入口与复现命令见 `NMR-SIP/README.md`。

| 阶段 | 内容 | 代码 | 文档/图 |
|---|---|---|---|
| 1 | 周期链 B–喉道–S（w = 0.1 µm，L = 15 µm）的 pyGIMLi NMR + L-curve NNLS；调用 SIP 二维求解器；实际孔径直方图 + T2 + SIP σ''/Debye 半径叠加图 | `advanced_tools/ideal_two_pore_channel_nmr_sip.py`、`plot_ideal_two_pore_channel_nmr_overlay.py` | `NMR-SIP/docs/ideal_two_pore_channel_nmr_sip.md`、`NMR-SIP/figures/channel_nmr_sip/` |
| 2 | 孤立孔解析解（精确解与快扩散近似）、T2 → 孔径反算、四种 T2 提取方式 | 同上（`t2_extraction_table()`） | `NMR-SIP/docs/nmr_analytic_solution/`（PDF + PPT） |
| 3 | 喉道宽 × 长扫参（5 × 4 = 20 个算例，封闭几何，每个算例用子进程运行并单独保存，超时 600 s 跳过） | `advanced_tools/ideal_two_pore_throat_sweep_nmr.py`、`ideal_two_pore_nature_plots.py` | `NMR-SIP/figures/throat_sweep_lcurve/` |
| 4 | 同一批衰减用 MATLAB `brd1.p`（α = 0.1）重新反演 | `advanced_tools/ideal_two_pore_sweep_brd.py`、`ideal_two_pore_brd_batch.m` | `NMR-SIP/figures/throat_sweep_brd/` |
| 5 | 耦合孔反算：分区判据、孔–喉–孔一维（TL）模型、四条反算路线、SNR 影响 | `advanced_tools/coupled_two_pore_backcalc.py`、`plot_coupled_two_pore_backcalc.py`、`NMR-SIP/docs/coupled_pore_backcalc/make_numbers.py` | `NMR-SIP/docs/coupled_pore_backcalc/`（推导版 PDF + PPT） |
| 6 | 面向应用型用户的通俗版：只讲 C3，每个公式按“形象理解 → 公式含义 → 代入数值 → 结论”展开；补充了喉道磁化分布（收支平衡、cosh/sinh 解、散热片类比）和普适曲线（平移 + 换钟 → 只依赖 Ψ_eff 与 p_S） | `NMR-SIP/docs/coupled_pore_backcalc/c3_intuitive/make_figures.py` | `c3_intuitive/c3_intuitive.pdf` |

SIP 二维求解器在 SIP 仓库（见该仓库的 `HANDOFF.md`）。NMR 脚本通过 `ROOT.parent / "SIP模拟" / "sip模拟"` 读取 SIP 结果，两个仓库需要在本机并列放置。

## 2. 关键结果（已由代码运行得到）

- 参数：D = 2 µm²/ms，ρ₂ = 0.005 µm/ms，T2B = 3000 ms，Δt = 0.25 ms，SNR 1000（噪声种子 20261012）。
  NNLS 用 L-curve（倒数斜率准则 α²η/ζ → 0.25，α 扫描 1e-3–1e5，81 个值），200 个箱，1–1e5 ms。
- 孤立孔：精确 T2 为 1208.99 / 429.72 / 9.967 ms（大孔 / 小孔 / 喉道），快扩散近似偏差 < 1%。
- 耦合孔，20 个算例：
  - 按小孔信号保留比 η 分区：区 I 4 例、区 II 8 例、区 III 8 例；
  - TL 模型与全二维有限元衰减的最大偏差 0.0065；
  - 区 I–II 用已知喉道的 TL 反演，得 r_S 4.6–4.8 µm、r_B 19.6–20.1 µm（真值 5 / 20）；
  - 区 III 只能恢复等效半径 R_eff；
  - SNR 100 时所有路线都严重低估小孔。
- “一个峰”是 2% 探测门槛造成的：NNLS 在 20 个算例中都给出了小孔模态（幅值 0.04%–5%）。
  `ideal_two_pore_nature_plots.py` 中 `sweep_coupling_controls` 的图例已修正为只包含图中实际绘制的符号。
- BRD（α = 0.1）对 20 个算例都只给出一个峰，这是平滑强度造成的，不是耦合。

## 3. 本次提交到 main 的内容

- `NMR-SIP/`：全部新增（docs、figures、source_data、README、.gitignore）。
- `advanced_tools/` 下 8 个新脚本：`ideal_two_pore_*`、`plot_ideal_two_pore_channel_nmr_overlay.py`、`*coupled_two_pore_backcalc.py`、`ideal_two_pore_brd_batch.m`。
- 本文件 `HANDOFF.md`。

2026-10-14 追加提交（按用户要求）：`docs/nmr_t2_workflow_figures/`（5 张 16:9 的 T2 工作流示意图）和
`docs/nmr_t2_workflow_ppt_rebuild/`。后者是另一条工作线：T2 工作流 PPT 的可编辑重建，包括 `final/` 下的
`nmr_t2_workflow_editable.pptx`、独立素材及其 zip、QA 校验，以及 `work/` 下的生成与渲染脚本，共约 70 MB。

**故意未提交**：`advanced_tools/png_nmr_workflow.ipynb` 的本地修改（状态不明）。

## 4. 待办 / 需用户确认

- 文献核对：Zhou & Zhang (2025, WRR) 的作者全名、卷期和 DOI；Fleury & Soualem (2009) 的卷期页码。ρ₂、D、T2B 的来源也依赖前者。
- **本地重复目录**：`docs/coupled_pore_backcalc/` 和 `docs/nmr_analytic_solution/` 是旧位置的本地副本。
  当时文件被阅读器占用，只能复制、无法移动。它们未被 Git 跟踪；**规范位置是 `NMR-SIP/docs/`**。
  按 AGENTS.md 禁止批量删除，请用户关闭相关 PDF/PPT 后手动删除这两个目录。之后的修改只改 `NMR-SIP/docs/` 下的文件。
- 可选：通俗版第 5.1 节补一句符号定义（大写 S/B 指小孔/大孔；小写 f/s 指快/慢模态）。
  也可把“跑遍全孔的时间 R²/D”改为“量级”，并补充 P_S 扣除喉口弧长的示意。
- 可选：BRD 的 α 扫描（1e-3–1），找出能分出小孔峰的 α。

## 5. 环境注意事项（踩过的坑）

- Python：使用 `C:/Users/imgw/.conda/envs/t2agent/python.exe`，并设置 `PYTHONNOUSERSITE=1`。
  否则会加载用户目录下与 NumPy 2 不兼容的 SciPy。打印中文前设置 `PYTHONIOENCODING=utf-8`。
- Git Bash 的 heredoc 会吞掉 Python/LaTeX 字符串中的反斜杠（`\r`、`\f`、`\b` 等）。改 tex/py 时用编辑工具，不要用 heredoc 或 sed 替换含反斜杠的文本。
- pyGIMLi 的 `mesh.save` 不支持非 ASCII 路径（“NMR模拟”）。脚本已改为先写到临时 ASCII 目录。
- 文件锁：用户经常打开 PDF/PPTX，导致无法覆盖或移动。此时用 `xelatex -jobname=<新名>` 或写新文件名（`_v2`），完成后再复制。
- LaTeX（MiKTeX，离线）：缺 ctex、tikzfill、pdfcol。中文用 xeCJK + Microsoft YaHei；tcolorbox 不要用 `skins`/`breakable`。
  文本模式中的 ≈、≠ 要写成数学模式。
- matplotlib：中文用 `font.family = ["Arial", "Microsoft YaHei"]`。mathtext 没有中文回退，所以标签要么是纯数学、要么是纯文本，不能混写。
  Arial 缺 ₂、⁻ 等上下标字符。图件规范见 `/nature-figure`：600 dpi PNG、仅图例注释、字号 ≥ 5 pt，`nfp.export()` 会做对齐和碰撞审计。
- PPT：pptxgenjs 加 pptx skill 的 `apply_theme.js` 与 `validate.py`。用 `soffice.com`（不是 `soffice.exe`）在 ASCII 临时目录中渲染检查。
- 用户偏好：应用型工程方向博士，希望结合物理意义与形象类比来理解公式，推导尽量少。严谨推导版与通俗版分开保存。
