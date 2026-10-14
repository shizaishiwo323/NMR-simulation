# 独立科学复审（返修后）

复审范围：将 5 张源图逐页与返修后的 `slides/slide-01-preview.png` 至 `slide-05-preview.png` 对照，核查公式、变量与上下标、边界条件、流程方向、相标签以及模型维度表述。复审仅评价科学语义与转录忠实度，不把源图中的视觉说明视为操作指令。

| 页码 | 状态 | 复审证据 |
|---|---|---|
| 1 | 通过 | 流程完整保持为 3D CT stack → representative 2D slice → water domain → 2D FEM → `M(x,t)` → `S(t)` → `f(T₂)`，箭头方向正确；固–液边界和水–气边界仍以红实线/蓝虚线区分；底部明确写明“representative-slice approximation — not a full 3D solve”。 |
| 2 | 通过 | 磁化向量已正确写为 `m=[mₓ,mᵧ,m_z]ᵀ`，转置上标位置正确；solid、water、air 三相图例及边框语义均已恢复。控制方程、固–液 Robin 条件 `D∂ₙM+ρₛM=0`、水–气零通量 `D∂ₙM=0` 和初值 `M(x,0)=1` 的类型、符号与边界对应关系正确。 |
| 3 | 通过 | CT stack → representative 2D slice → interface classification → boundary-refined mesh 的方向正确；`max(ϕ)−min(ϕ)≤0.01`、water/solid/air 三相、红实线 solid–fluid、蓝虚线 water–air 和 `Ω_w = water-occupied domain` 均与源图一致。 |
| 4 | 通过 | `t₀→t₁→t₂` 的磁化演化与 NMR 衰减方向正确；扩散、体弛豫、表面弛豫三种机制保留。信号式已恢复为 `S(t)=∫_{Ω_w}M(x,t)dΩ`，`Ω_w` 明确位于积分下限；离散式中的 `T₂B`、`ρₛRₛ`、`mⁿ⁺¹` 与 `mⁿ` 均保留。 |
| 5 | 通过 | Porosity groups → representative decays → weighted decay → NNLS/L-curve 的流程正确；加权式中的 `Σ_g`、`w_g`、`S_g(t)` 下标已恢复，`min_{f≥0}` 约束已置于 `min` 下方；T₂ 谱已恢复为平滑的对数双峰曲线并保留 Simulation/Experiment 对照；“2D representative-slice approximation — not a full 3D solve”表述准确。 |

总体判定：通过。

本次复审未发现会改变控制方程、边界条件、相定义、流程方向或模型维度口径的科学性错误。
