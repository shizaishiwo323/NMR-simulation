"""Plot original MATLAB BRD alpha=0.1 for the completed Sample 24 grouped decay.

First run sample16_brd_alpha_sweep with the original brd1.p, alpha=0.1,
normalize_first_echo=true, and the existing grouped decay. Then run this module
with conda base Python. Original L-curve results are read only.
"""

from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


SOURCE = Path("simulation_outputs/sample_24_grouped_nmr_lcurve_20261011")
OUTPUT = Path("simulation_outputs/sample_24_grouped_nmr_brd_alpha0p1_20261011")
BRD_DIR = Path("tmp/T2process_20260921/brd/matlab")


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    original = json.loads((SOURCE / "run_manifest.json").read_text(encoding="utf-8"))
    decay_path = SOURCE / "group_weighted_normalized_decay.csv"
    decay = pd.read_csv(decay_path)
    time_ms = decay.time_ms.to_numpy(float)
    raw_signal = decay.group_weighted_normalized_signal.to_numpy(float)
    first_echo = np.flatnonzero(time_ms > 0)[0]
    normalized_signal = raw_signal / raw_signal[first_echo]
    spectrum = np.loadtxt(OUTPUT / "brd_t2_spectra.csv", delimiter=",")
    summary = np.loadtxt(OUTPUT / "brd_summary.csv", delimiter=",")
    sampled = np.loadtxt(OUTPUT / "brd_sampled_decay.csv", delimiter=",")
    t2_ms, amplitude = spectrum.T
    assert spectrum.shape == (200, 2) and sampled.shape == (200, 2)
    assert summary.shape == (4,) and np.isclose(summary[0], 0.1)
    assert np.all(np.isfinite(amplitude)) and np.all(amplitude >= 0)
    assert np.all(np.diff(t2_ms) > 0) and amplitude.sum() > 0
    assert np.isclose(t2_ms[0], 0.01) and np.isclose(t2_ms[-1], 100000)
    assert np.all(np.diff(sampled[:, 0]) > 0)
    sample_indices = np.searchsorted(time_ms, sampled[:, 0])
    assert np.allclose(time_ms[sample_indices], sampled[:, 0])
    assert np.allclose(normalized_signal[sample_indices], sampled[:, 1])
    fitted = np.exp(-time_ms[:, None] / t2_ms) @ amplitude
    fitted_sampled = np.exp(-sampled[:, 0, None] / t2_ms) @ amplitude
    sampled_error = float(np.linalg.norm(sampled[:, 1] - fitted_sampled) / np.linalg.norm(sampled[:, 1]))
    positive = time_ms > 0
    raw_error = float(np.linalg.norm(normalized_signal[positive] - fitted[positive]) / np.linalg.norm(normalized_signal[positive]))
    assert np.isclose(sampled_error, summary[2], rtol=1e-8)
    assert raw_error < 0.03 and sampled_error < 0.03
    fraction = amplitude / amplitude.sum()
    peak_ms = float(t2_ms[np.argmax(amplitude)])
    assert np.isclose(peak_ms, summary[1]) and np.isclose(fraction.sum(), 1)
    pd.DataFrame({"t2_ms": t2_ms, "amplitude": amplitude, "signal_fraction": fraction}).to_csv(
        OUTPUT / "simulation_t2_brd_alpha0p1.csv", index=False)
    pd.DataFrame({"time_ms": time_ms, "original_weighted_signal": raw_signal,
                  "first_echo_normalized_signal": normalized_signal, "brd_reconstruction": fitted,
                  "residual": normalized_signal - fitted}).to_csv(OUTPUT / "raw_decay_reconstruction.csv", index=False)

    # Reuse the completed full-resolution CT histograms without altering bins or weights.
    old_overlay = pd.read_csv(SOURCE / "overlay_source_data.csv")
    body_data = old_overlay[old_overlay.series == "CT pore body"].copy()
    throat_data = old_overlay[old_overlay.series == "CT internal throat"].copy()
    overlap_data = old_overlay[old_overlay.series == "Display histogram overlap"].copy()
    edges = np.r_[body_data.radius_left_um.to_numpy(), body_data.radius_right_um.iloc[-1]]
    body, throat, overlap = [frame.fraction.to_numpy(float) for frame in [body_data, throat_data, overlap_data]]
    assert np.allclose(throat_data.radius_left_um, edges[:-1])
    assert np.allclose(overlap, np.minimum(body, throat))
    assert np.isclose(body.sum() + throat.sum(), 1)
    display_rho = original["reference_display_mapping"]["rho2_effective_um_per_s"]
    scale = 0.003 * display_rho
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.linewidth": 0.8, "legend.frameon": False})
    for mode in ["reference_radius", "independent_axes"]:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        right = ax.twinx()
        ax.set_xscale("log")
        h_throat = ax.stairs(throat, edges, fill=True, facecolor="white", edgecolor="black",
                             hatch="////", linewidth=0.8, label="CT internal throat")
        h_overlap = ax.stairs(overlap, edges, fill=True, facecolor="0.6", edgecolor="black",
                              linewidth=0.8, label="Histogram overlap")
        h_body = ax.stairs(body, edges, color="black", linewidth=1.1, label="CT pore body")
        right.set_ylim(0, fraction.max() * 1.3)
        ax.set_ylim(0, max(body.max(), throat.max()) * 1.2)
        if mode == "reference_radius":
            label = "Simulated NMR BRD ($\\alpha=0.1$)\n" + rf"(display $\rho_2^*={display_rho:.2f}$ $\mu$m/s)"
            line, = right.plot(t2_ms * scale, fraction, color="black", lw=1.6, label=label)
            ax.set_xlim(0.01, max(edges[-1], (t2_ms * scale)[amplitude > 0].max()) * 1.25)
            top = ax.secondary_xaxis("top", functions=(lambda r: r / scale, lambda t: t * scale))
            top.set_xlabel(r"$T_2$ (ms)", labelpad=7)
            ax.set_xlabel(r"CT pore radius / NMR display radius, $r$ ($\mu$m)")
        else:
            top = right.twiny()
            top.set_xscale("log")
            top.set_xlim(0.01, 100000)
            line, = top.plot(t2_ms, fraction, color="black", lw=1.6, label=r"Simulated NMR BRD ($\alpha=0.1$)")
            top.set_xlabel(r"$T_2$ (ms), independent axis", labelpad=7)
            ax.set_xlim(1, 1000)
            ax.set_xlabel(r"CT pore radius, $r$ ($\mu$m)")
        right.set_ylabel("NMR signal fraction per log-T2 bin")
        ax.set_ylabel("CT/PNM volume fraction per log-radius bin")
        ax.spines["top"].set_visible(False)
        right.spines["top"].set_visible(False)
        ax.grid(axis="y", color="0.87", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(handles=[h_body, h_throat, h_overlap, line], loc="upper left", ncol=2,
                  fontsize=7, handlelength=2, columnspacing=1)
        fig.tight_layout()
        fig.savefig(OUTPUT / f"sample24_ct_simulated_t2_{mode}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.semilogx(t2_ms, fraction, color="black", lw=1.6)
    ax.set(xlabel=r"$T_2$ (ms)", ylabel="NMR signal fraction per log-T2 bin", xlim=(0.01, 100000),
           ylim=(0, fraction.max() * 1.15), title=r"Conditional simulation: BRD ($\alpha=0.1$)")
    ax.grid(axis="y", color="0.87", lw=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT / "simulation_t2_brd_alpha0p1.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].plot(time_ms, normalized_signal, color="0.65", lw=1.2, label="Simulated decay")
    axes[0].plot(time_ms, fitted, color="black", lw=1, label="BRD reconstruction")
    axes[0].set(xlabel="Time (ms)", ylabel="First-echo normalized signal", yscale="log")
    axes[0].legend(fontsize=7)
    axes[1].plot(time_ms[positive], (normalized_signal - fitted)[positive], color="black", lw=0.8)
    axes[1].axhline(0, color="0.7", lw=0.6)
    axes[1].set(xlabel="Time (ms)", ylabel="Signal residual")
    fig.tight_layout()
    fig.savefig(OUTPUT / "brd_decay_fit.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    new_curve = pd.DataFrame({"series": "Simulated NMR BRD alpha=0.1", "t2_ms": t2_ms,
                              "display_radius_um": t2_ms * scale, "fraction": fraction, "amplitude": amplitude})
    pd.concat([body_data, throat_data, overlap_data, new_curve], ignore_index=True).to_csv(
        OUTPUT / "overlay_source_data.csv", index=False)
    log = (OUTPUT / "matlab_brd.log").read_text(encoding="utf-8")
    manifest = {
        "model": original["model"], "conditional_simulation": True,
        "source_simulation_manifest": str((SOURCE / "run_manifest.json").resolve()),
        "input_tiff": original["input_tiff"], "input_sha256": original["input_sha256"],
        "groups": original["groups"], "weighted_slices": original["weighted_slices"],
        "parameters": original["parameters"], "representative_substitutions": original["representative_substitutions"],
        "inversion": "Original T2process MATLAB brd1.p", "alpha": 0.1, "alpha_selection": "User-specified fixed smoothing factor",
        "matlab_version": log.splitlines()[0], "t2_bins": 200, "t2_grid_ms": [0.01, 100000],
        "preprocessing": "Exclude t=0; divide by first valid echo at 5 ms; original AmplitudeSample selects 200 echoes; no added noise, trend weighting or post-inversion smoothing.",
        "first_echo_ms": float(time_ms[first_echo]), "first_echo_signal": float(raw_signal[first_echo]),
        "kernel": "exp(-time_ms/T2_ms), no fitted baseline", "peak_t2_ms": peak_ms,
        "sampled_relative_residual": sampled_error, "full_positive_time_relative_residual": raw_error,
        "minimum_amplitude": float(amplitude.min()),
        "solver_log_termination": "c变化幅度过小！" if "c变化幅度过小！" in log else "See matlab_brd.log",
        "solver_validation_scope": "BRD p-code internal stopping threshold is opaque; external checks verify finite nonnegative output and decay reconstruction, not an independent optimality certificate.",
        "source_brd_commit": "47461a72e0852e5d2a5a08db19a1f8404145e236",
        "ct_histogram": original["ct_histogram"], "fraction_scope": original["fraction_scope"],
        "reference_display_mapping": original["reference_display_mapping"],
        "independent_axes": original["independent_axes"], "figure_contract": original["figure_contract"],
        "python_executable": sys.executable, "outputs": [str(p.resolve()) for p in sorted(OUTPUT.glob("*.png"))],
    }
    manifest["source_hashes"] = {}
    for path in [decay_path, SOURCE / "overlay_source_data.csv", SOURCE / "run_manifest.json",
                 BRD_DIR / "brd1.p", BRD_DIR / "AmplitudeSample.m",
                 Path("advanced_tools/sample16_brd_alpha_sweep.m"), Path(__file__)]:
        with path.open("rb") as handle:
            manifest["source_hashes"][str(path.resolve())] = hashlib.file_digest(handle, "sha256").hexdigest()
    write_json(OUTPUT / "run_manifest.json", manifest)
    qa = {"alpha": 0.1, "algorithm": manifest["inversion"], "spectrum_bins": len(t2_ms),
          "sampled_echoes": len(sampled), "peak_t2_ms": peak_ms, "sampled_relative_residual": sampled_error,
          "full_positive_time_relative_residual": raw_error, "finite_nonnegative_spectrum": True,
          "normalized_spectral_sum": float(fraction.sum()), "checks_passed": True,
          "png_sizes": {p.name: list(Image.open(p).size) for p in OUTPUT.glob("*.png")}}
    write_json(OUTPUT / "qa_validation.json", qa)
    (OUTPUT / "结果说明.md").write_text(f"""# Sample 24：BRD α=0.1 条件性模拟反演

复用原始分辨率代表切片的 27 组、901 张切片权重加权衰减；未重复模拟。物理参数保持用户选定的条件性基准：ρ₂=5 μm/s、D=2 μm²/ms、T₂,bulk=3000 ms。模型为孔隙率分组代表切片近似三维（2.5D），不包含跨切片扩散或真正三维求解。

使用原版 T2process MATLAB brd1.p，固定平滑因子 α=0.1，不采用 L-curve 选参。调用项目 sample16_brd_alpha_sweep，输入原加权衰减，alpha_values=0.1、normalize_first_echo=true。排除 t=0，按首个有效回波（5 ms）归一化，AmplitudeSample 选取 200 个回波；T₂ 网格为 0.01–100000 ms、200 个对数分箱。没有添加随机噪声或谱后平滑，拟合核为 exp(-t/T₂)，不含基线项。

主峰 T₂={peak_ms:.6g} ms。200 个采样回波相对残差 {sampled_error:.6%}；全部 t>0 原始时间点相对残差 {raw_error:.6%}。谱为有限非负值。原版求解器日志以“c变化幅度过小！”结束；内部阈值封装于 p-code，外部验证核对输出与重建误差，未独立证明最优性。5 ms 回波间隔下，0.01 ms 网格端点不是可靠分辨下限。

sample24_ct_simulated_t2_reference_radius.png 沿用附图式样：CT 孔体轮廓、内部孔喉斜线、灰色图形重叠、黑色 BRD 谱。显示坐标采用原有实验终点系数 ρ₂*={display_rho:.8g} μm/s，r_display=0.003ρ₂*T₂；只用于显示，不含体弛豫校正，与求解器 ρ₂=5 μm/s 不同。sample24_ct_simulated_t2_independent_axes.png 保留相互独立的 CT 半径和 T₂ 横轴。

CT 直方图沿用全分辨率 pnextract 几何数据及原有 30 个对数半径分箱，按孔体与内部孔喉总网络体积归一化。BRD 谱按幅值总和归一化，纵轴使用独立尺度；灰色逐箱重叠只表示图形重叠。simulation_t2_brd_alpha0p1.csv、overlay_source_data.csv、原始 MATLAB 输出、衰减重建 CSV、日志及 manifest 均保留；原 L-curve 目录未改写。

绘图复现：在仓库根目录用 conda base 运行 python -m advanced_tools.sample24_plot_brd_alpha0p1。
""", encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False))


if __name__ == "__main__":
    main()
