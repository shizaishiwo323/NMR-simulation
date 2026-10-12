"""Full-resolution porosity-grouped Sample 24 NMR and L-curve NNLS.

Run simulate with the nmr-simulation Python, then finalize with base Python.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advanced_tools.sample16_grouped_porosity_nmr import (
    candidate_indices_for_group, completed_group_run, compute_slice_porosity, group_porosity_series,
    is_physical_normalized_decay, run_one_slice,
)
from nmr_t2.config import LCurveConfig
from nmr_t2.lcurve import invert_single_signal_lcurve
from nmr_t2.nnls import build_exponential_kernel, estimate_noise_vector, solve_t2_nnls_matlab_compatible


CT_DIR = Path(r"E:\python\jupyter\codex\震电模拟\Wang_SE_experiment\CT\Post-24-1075μm")
OUTPUT = Path("simulation_outputs/sample_24_grouped_nmr_lcurve_20261011")
INPUT_SHA = "6d4a5b488d650bf9189a27e33d9f4ed0e4fc6e9a931bf6fe35f200a3498e40e0"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def simulate(output, selected_group):
    args = argparse.Namespace(
        input_tiff=CT_DIR / "4-24.tif", sample_name="sample24", pore_value=0,
        solid_value=255, pixel_size_um=10.7494500402588,
        diffusion_um2_per_ms=2.0, bulk_t2_ms=3000.0,
        rho_solid_um_per_ms=0.005, rho_gas_um_per_ms=0.0,
        dt_ms=5.0, t_max_ms=21000.0, max_grid_size="full",
        mesh_bulk_size_um=12.0, mesh_boundary_size_um=5.0,
        mesh_max_points=500000, t2_min_ms=0.01, t2_max_ms=100000.0,
        t2_bins=200, fixed_alpha=1.0, slice_timeout_s=None,
    )
    with args.input_tiff.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == INPUT_SHA
    profile = compute_slice_porosity(args.input_tiff, 0, 255)
    groups = group_porosity_series(profile.porosity, 0.01)
    assert sum(g.slice_count for g in groups) == len(profile) == 901
    assert all(g.porosity_max - g.porosity_min <= 0.01 + 1e-12 for g in groups)
    profile.to_csv(output / "slice_porosity_profile.csv", index=False)
    pd.DataFrame([asdict(g) for g in groups]).to_csv(output / "porosity_groups_initial.csv", index=False)
    mesh_dir = output / "representative_mesh_figures"
    mesh_dir.mkdir(exist_ok=True)
    parameter_record = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    record_path = output / "simulation_parameters.json"
    if record_path.exists():
        assert json.loads(record_path.read_text(encoding="utf-8")) == parameter_record
    write_json(record_path, parameter_record)
    previous_attempts = {}
    previous_groups_path = output / "porosity_groups_and_representatives.csv"
    if previous_groups_path.exists():
        previous_groups = pd.read_csv(previous_groups_path, dtype={"attempted_representatives": str})
        previous_attempts = {int(r.group_id): [int(v) for v in r.attempted_representatives.split(",")]
                             for r in previous_groups.itertuples()}
    def run_group(group):
        print(f"START group {group.group_id + 1}/{len(groups)} central slice {group.representative_index}", flush=True)
        started = time.monotonic()
        attempts = previous_attempts.get(group.group_id, []).copy()
        candidates = candidate_indices_for_group(group.start_index, group.end_index)
        completed = [v for v in candidates if completed_group_run(
            output / "group_runs" / f"group_{group.group_id:03d}_slice_{v:04d}")]
        candidates = completed + [v for v in candidates if v not in completed]
        for candidate in candidates:
            folder = output / "group_runs" / f"group_{group.group_id:03d}_slice_{candidate:04d}"
            folder.mkdir(parents=True, exist_ok=True)
            if candidate not in attempts:
                attempts.append(candidate)
            manifest_path = folder / "run_manifest.json"
            reused = manifest_path.exists()
            if not reused and not run_one_slice(args, group, candidate, folder):
                print(f"FAILED slice {candidate}; see {folder / 'subprocess_stdout.log'}", flush=True)
                continue
            group_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            par = group_manifest["simulation_params"]
            assert par["max_grid_size"] is None
            assert par["pixel_size_x_um"] == args.pixel_size_um
            assert par["rho_solid_um_per_ms"] == args.rho_solid_um_per_ms
            assert par["t_max_ms"] == args.t_max_ms and par["dt_ms"] == args.dt_ms
            decay = pd.read_csv(folder / "average_normalized_decay.csv")
            signal = decay.average_normalized_signal.to_numpy(float)
            if not is_physical_normalized_decay(signal):
                print(f"REJECTED nonphysical decay on slice {candidate}", flush=True)
                continue
            mesh = next(folder.glob("*triangular_mesh.png"))
            dest = mesh_dir / f"group_{group.group_id:03d}_slice_{candidate:04d}_mesh.png"
            shutil.copy2(mesh, dest)
            mesh_stats = pd.read_csv(folder / "mesh_summary.csv").iloc[0].to_dict()
            # Per-group inversion is an auxiliary fixed-alpha diagnostic only.
            row = {**asdict(group), "used_representative_index": candidate,
                         "weight_slice_count": group.slice_count, "attempted_representatives": ",".join(map(str, attempts)),
                         "decay_csv": str((folder / "average_normalized_decay.csv").resolve()),
                         "mesh_png": str(dest.resolve()), "reused_existing_run": reused}
            print(f"DONE group {group.group_id + 1}/{len(groups)} slice {candidate}; "
                  f"nodes={mesh_stats.get('mesh_nodes', mesh_stats.get('mesh_vertices'))}; "
                  f"tail={signal[-1]:.3g}; seconds={time.monotonic() - started:.1f}", flush=True)
            return row, {"group_id": group.group_id, **mesh_stats}, (group.slice_count, decay.time_ms.to_numpy(float), signal)
        raise RuntimeError(f"No valid representative for group {group.group_id}")
    if selected_group is not None:
        run_group(groups[selected_group])
        return
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run_group, groups))
    rows, stats_rows, signals = map(list, zip(*results))
    assert len(rows) == len(groups) == 27
    time_ms = signals[0][1]
    assert all(np.array_equal(time_ms, v[1]) for v in signals)
    weighted = sum(w * s for w, _, s in signals) / sum(w for w, _, _ in signals)
    assert is_physical_normalized_decay(weighted) and weighted[-1] < 0.001
    pd.DataFrame({"time_ms": time_ms, "group_weighted_normalized_signal": weighted}).to_csv(
        output / "group_weighted_normalized_decay.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "porosity_groups_and_representatives.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(output / "representative_mesh_summary.csv", index=False)
    pd.DataFrame(rows)[["group_id", "used_representative_index", "weight_slice_count", "mesh_png"]].to_csv(
        mesh_dir / "representative_mesh_figures_manifest.csv", index=False)
    write_json(output / "simulation_summary.json", {
        "input_tiff": str(args.input_tiff.resolve()), "input_sha256": INPUT_SHA,
        "model": "porosity-grouped 2D representative-slice approximation (2.5D)",
        "groups": len(groups), "completed_groups": len(rows), "weighted_slices": 901,
        "skipped_groups": [], "porosity_threshold": 0.01, "ct_porosity": float(profile.porosity.mean()),
        "representative_substitutions": [
            {"group_id": r["group_id"], "central_index": r["representative_index"],
             "used_index": r["used_representative_index"], "attempts": r["attempted_representatives"]}
            for r in rows if r["representative_index"] != r["used_representative_index"]
        ],
        "parameters": parameter_record, "python_executable": sys.executable,
        "parameter_source": "User-approved project baseline, conditional simulation; all pores water-filled",
        "numerics": "Original 822x822 slices; 12/5 um mesh targets; node guard raised without coarsening. "
                    "Backward Euler dt=5 ms; 21000 ms = 7 bulk T2, for a sufficiently decayed tail.",
        "boundary": "Reuse existing water-solid Robin boundary and cut-face classification; rho_gas=0. "
                    "No interslice diffusion or 3D connectivity solve.",
        "tail_signal": float(weighted[-1]),
    })


def finalize(output):
    summary = json.loads((output / "simulation_summary.json").read_text(encoding="utf-8"))
    decay = pd.read_csv(output / "group_weighted_normalized_decay.csv")
    t, y = decay.time_ms.to_numpy(float), decay.group_weighted_normalized_signal.to_numpy(float)
    # Retain the full decay including t=0 in the CSV; the legacy solver requires t>0.
    config = LCurveConfig(alpha_min=1e-6, alpha_max=1e6, alpha_count=120)
    result = invert_single_signal_lcurve(t, y, signal_name="sample24_group_weighted", config=config)
    assert 0 < result.best_index < len(result.alpha_values) - 1
    amp, t2 = result.spectrum, result.t2_bins_ms
    assert np.all(np.isfinite(amp)) and np.all(amp >= 0) and amp.sum() > 0
    fraction = amp / amp.sum()
    pd.DataFrame({"t2_ms": t2, "amplitude": amp, "signal_fraction": fraction}).to_csv(output / "simulation_t2_lcurve.csv", index=False)
    pd.DataFrame({"alpha": result.alpha_values, "weighted_residual_norm": result.residual_norms,
                  "roughness_norm": result.roughness_norms, "reciprocal_slope": result.slope_reciprocal_values,
                  "selected": np.arange(len(result.alpha_values)) == result.best_index}).to_csv(output / "lcurve_metrics.csv", index=False)
    pd.DataFrame({"fit_time_ms": result.fit_time_ms, "fit_amplitude": result.fit_amplitude,
                  "residual": result.residual}).to_csv(output / "lcurve_fit.csv", index=False)
    fit_relative_error = float(np.linalg.norm(result.residual) / np.linalg.norm(result.fit_amplitude + result.residual))
    assert fit_relative_error < 0.03
    positive_time = t > 0
    model, _, _ = solve_t2_nnls_matlab_compatible(y[positive_time], t[positive_time],
        estimate_noise_vector(y[positive_time]), t2, result.best_regularization)
    assert np.allclose(model[:-1], amp)
    raw_reconstruction = build_exponential_kernel(t, t2) @ model[:-1] + model[-1]
    raw_relative_error = float(np.linalg.norm(y - raw_reconstruction) / np.linalg.norm(y))
    assert raw_relative_error < 0.03
    pd.DataFrame({"time_ms": t, "simulated_signal": y, "nnls_reconstruction": raw_reconstruction,
                  "residual": y - raw_reconstruction}).to_csv(output / "raw_decay_reconstruction.csv", index=False)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.linewidth": 0.8, "legend.frameon": False})
    # The legacy helper's hard-coded random-slice title does not describe these runs.
    representatives = pd.read_csv(output / "porosity_groups_and_representatives.csv")
    for row in representatives.itertuples():
        frame = pd.read_csv(row.decay_csv)
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.plot(frame.time_ms, frame.average_normalized_signal, color="black", lw=1.5)
        ax.set(xlabel="Time (ms)", ylabel="Normalized NMR signal",
               title=f"Group {row.group_id + 1}: representative slice {row.used_representative_index + 1}")
        fig.tight_layout()
        fig.savefig(Path(row.decay_csv).parent / "average_and_slice_decay_curves.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].loglog(result.residual_norms, result.roughness_norms, color="black", lw=1)
    axes[0].plot(result.residual_norms[result.best_index], result.roughness_norms[result.best_index], "o", color="#b34a3b", ms=4)
    axes[0].set(xlabel="Weighted residual norm", ylabel="Roughness norm")
    axes[0].set_title(rf"L-curve ($\alpha={result.best_regularization:.4g}$)", fontsize=9)
    axes[1].plot(t, y, color="0.65", lw=1, label="Simulated decay")
    axes[1].plot(t, raw_reconstruction, color="black", lw=1, label="NNLS reconstruction")
    axes[1].set(xlabel="Time (ms)", ylabel="Normalized signal", yscale="log")
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output / "lcurve_and_decay_fit.png", dpi=600, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.semilogx(t2, fraction, color="black", lw=1.6)
    ax.set(xlabel=r"$T_2$ (ms)", ylabel="NMR signal fraction per log-T2 bin", xlim=(0.01, 100000), ylim=(0, fraction.max()*1.15))
    ax.grid(axis="y", color="0.87", lw=0.5)
    fig.tight_layout(); fig.savefig(output / "simulation_t2_lcurve.png", dpi=600, bbox_inches="tight"); plt.close(fig)
    network = CT_DIR / "pore_network/network_parsed"
    pores, throats = pd.read_csv(network / "pores.csv"), pd.read_csv(network / "throats.csv")
    throats = throats[(throats.pore1_id > 0) & (throats.pore2_id > 0)]
    arrays = []
    for frame, prefix in [(pores, "pore"), (throats, "throat")]:
        r, v = frame[f"{prefix}_radius_m"].to_numpy(float) * 1e6, frame[f"{prefix}_volume_m3"].to_numpy(float)
        keep = np.isfinite(r) & np.isfinite(v) & (r > 0) & (v > 0)
        arrays.append((r[keep], v[keep]))
    edges = np.geomspace(min(a[0].min() for a in arrays)*(1-1e-12), max(a[0].max() for a in arrays)*(1+1e-12), 31)
    denominator = sum(a[1].sum() for a in arrays)
    body, throat = [np.histogram(r, bins=edges, weights=v)[0]/denominator for r,v in arrays]
    overlap = np.minimum(body, throat)
    assert np.isclose(body.sum()+throat.sum(), 1) and np.isclose(fraction.sum(), 1)
    calibration = json.loads(Path("simulation_outputs/sample_24_ct_nmr_calibrated_radius_overlay_20261011/run_manifest.json").read_text(encoding="utf-8"))
    display_rho = calibration["rho2_effective_um_per_s"]
    scale = 0.003 * display_rho
    plot_rows = []
    for mode in ["reference_radius", "independent_axes"]:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        right = ax.twinx()
        ax.set_xscale("log")
        h_throat = ax.stairs(throat, edges, fill=True, facecolor="white", edgecolor="black", hatch="////", linewidth=0.8, label="CT internal throat")
        h_overlap = ax.stairs(overlap, edges, fill=True, facecolor="0.6", edgecolor="black", linewidth=0.8, label="Histogram overlap")
        h_body = ax.stairs(body, edges, color="black", linewidth=1.1, label="CT pore body")
        right.set_ylim(0, fraction.max()*1.3)
        ax.set_ylim(0, max(body.max(), throat.max())*1.2)
        if mode == "reference_radius":
            label = "Simulated NMR L-curve NNLS\n" + rf"(display $\rho_2^*={display_rho:.2f}$ $\mu$m/s)"
            line, = right.plot(t2*scale, fraction, color="black", lw=1.6, label=label)
            ax.set_xlim(0.01, max(edges[-1], (t2*scale)[amp>0].max())*1.25)
            top = ax.secondary_xaxis("top", functions=(lambda r:r/scale, lambda tt:tt*scale))
            top.set_xlabel(r"$T_2$ (ms)", labelpad=7)
            ax.set_xlabel(r"CT pore radius / NMR display radius, $r$ ($\mu$m)")
            right.set_ylabel("NMR signal fraction per log-T2 bin")
        else:
            top = right.twiny()
            top.set_xscale("log"); top.set_xlim(0.01, 100000)
            line, = top.plot(t2, fraction, color="black", lw=1.6, label=r"Simulated NMR L-curve NNLS ($\rho_2=5$ $\mu$m/s)")
            top.set_xlabel(r"$T_2$ (ms), independent axis", labelpad=7)
            ax.set_xlim(1, 1000); ax.set_xlabel(r"CT pore radius, $r$ ($\mu$m)")
            right.set_ylabel("NMR signal fraction per log-T2 bin")
        ax.set_ylabel("CT/PNM volume fraction per log-radius bin")
        ax.spines["top"].set_visible(False); right.spines["top"].set_visible(False)
        ax.grid(axis="y", color="0.87", linewidth=0.5); ax.set_axisbelow(True)
        ax.legend(handles=[h_body,h_throat,h_overlap,line],loc="upper left",ncol=2,fontsize=7,handlelength=2,columnspacing=1)
        fig.tight_layout(); fig.savefig(output / f"sample24_ct_simulated_t2_{mode}.png", dpi=600, bbox_inches="tight"); plt.close(fig)
    centers = np.sqrt(edges[:-1]*edges[1:])
    for name, values in [("CT pore body",body),("CT internal throat",throat),("Display histogram overlap",overlap)]:
        plot_rows.append(pd.DataFrame({"series":name,"radius_um":centers,"radius_left_um":edges[:-1],"radius_right_um":edges[1:],"fraction":values}))
    plot_rows.append(pd.DataFrame({"series":"Simulated NMR L-curve NNLS","t2_ms":t2,"display_radius_um":t2*scale,"fraction":fraction,"amplitude":amp}))
    pd.concat(plot_rows,ignore_index=True).to_csv(output / "overlay_source_data.csv",index=False)
    manifest = {**summary, "lcurve_config":asdict(config),"alpha":float(result.best_regularization),
                "alpha_selection":"Project L-curve reciprocal slope alpha^2*eta/zeta nearest 0.25 in [0.1,10]; not maximum-curvature selection",
                "selected_index":int(result.best_index),"interior_alpha":True,"relative_fit_residual":fit_relative_error,
                "raw_decay_relative_reconstruction_error":raw_relative_error,"nnls_baseline":float(model[-1]),
                "simulation_t2_peak_ms":float(t2[np.argmax(amp)]),"positive_t2_range_ms":[float(t2[amp>0].min()),float(t2[amp>0].max())],
                "inversion_preprocessing":"No added random noise. Existing MATLAB-compatible trend-based noise weighting, reverse 5-point averaging and logarithmic downsampling; t=0 excluded from inversion.",
                "reference_display_mapping":{"rho2_effective_um_per_s":display_rho,"formula":"display_radius_um=0.003*rho2_effective*T2_ms",
                    "source":"Existing Sample24 experimental Vc=1 display calibration","scope":"Figure display only. Different from solver rho2=5 um/s; no bulk correction and not an inferred physical simulated radius."},
                "independent_axes":"CT radii and simulated T2 plotted on independent log axes; no data transformation or peak alignment",
                "ct_histogram":{"valid_pores":len(arrays[0][0]),"valid_internal_throats":len(arrays[1][0]),"bins":30,
                    "denominator_m3":float(denominator),"body_fraction":float(body.sum()),"throat_fraction":float(throat.sum())},
                "fraction_scope":"CT fractions share pore-body plus internal-throat volume denominator; NMR normalized by total spectral amplitude. Different log-bin widths, separate y axes. Overlap is graphical only.",
                "figure_contract":{"claim":"Compare full-resolution CT network radii with conditional grouped-slice NMR T2 response",
                    "archetype":"quantitative grid","backend":"Python/matplotlib","export":"600 dpi PNG","statistics":"One CT sample; 27 representative solves are not independent sample replicates"},
                "finalize_python":sys.executable}
    manifest["source_hashes"] = {}
    for path in [network / "pores.csv", network / "throats.csv", Path(__file__)]:
        with path.open("rb") as handle:
            manifest["source_hashes"][str(path.resolve())] = hashlib.file_digest(handle,"sha256").hexdigest()
    write_json(output / "run_manifest.json",manifest)
    report = f"""# Sample 24：CT–NMR 条件性模拟结果

对原始 CT 的 901 张 822×822 切片按 z 方向连续分组，组内孔隙率极差不超过 0.01，共完成 27 组代表切片二维求解。每组使用中间切片；若更换代表切片，分组表记录全部尝试。按每组切片数量加权归一化衰减后，统一进行 L-curve NNLS 反演。它是代表切片近似三维（2.5D），不计算跨切片扩散或三维连通。

本次共有 {len(summary['representative_substitutions'])} 组采用相邻代表切片，所有 27 组和 901 张切片权重均保留；失败尝试的日志保留在对应分组目录。

## 输入与物理前提

CT 孔隙为 0、固体为 255；体素边长 10.7494500402588 μm；CT 可分辨孔隙率 {summary['ct_porosity']:.8%}。全部孔隙按充水处理。用户选定项目基准参数：ρ₂=5 μm/s、D=2 μm²/ms、T₂,bulk=3000 ms。它们是条件性模拟参数，并非该样品独立测量值。

代表切片保持原始分辨率，网格目标尺度为内部 12 μm、边界 5 μm；节点上限提高至 500000，未通过放粗网格降低计算量。时间步长 5 ms，模拟至 21000 ms；加权信号末值 {summary['tail_signal']:.6g}。保留项目现有截面边缘处理和 Robin 弛豫边界。

## 反演

T₂ 使用 200 个对数分箱，范围 0.01–100000 ms；在 10⁻⁶–10⁶ 内扫描 120 个 α。复用项目的 L-curve 规则：在有效范围 0.1–10 内选择 α²η/ζ 最接近 0.25 的点。所选 α={result.best_regularization:.8g}，处于扫描范围内部；主峰 T₂={t2[np.argmax(amp)]:.6g} ms。拟合的相对残差为 {fit_relative_error:.6g}。

0.01 ms 是反演网格端点，不代表在 5 ms 采样间隔下可可靠分辨的最短 T₂。

没有添加随机噪声。现有反演器使用移动趋势残差估计权重，进行 5 点反向平滑与对数降采样；该残差描述预处理后的拟合。原始加权衰减包括 t=0，反演排除 t=0。每组目录中的 fixed-alpha 谱只作辅助诊断，最终谱是 simulation_t2_lcurve.csv。

在完整原始时间点上重建衰减，含求解器非负基线项 {model[-1]:.6g}，相对重建误差为 {raw_relative_error:.6g}；数据保存在 raw_decay_reconstruction.csv。

## 图的读取方式

- sample24_ct_simulated_t2_reference_radius.png 沿用参考图样式和原有实验 Vc=1 显示系数 ρ₂*={display_rho:.8g} μm/s，r_display=0.003ρ₂*T₂。此式只用于显示；不含体弛豫校正，与求解用的 ρ₂=5 μm/s 不同，也不是模拟孔径的物理反演。
- sample24_ct_simulated_t2_independent_axes.png 使用相互独立的 CT 半径和 T₂ 对数横轴，不转换数据、不对齐主峰。
- simulation_t2_lcurve.png 是未经孔径显示换算的 T₂ 谱。lcurve_and_decay_fit.png 给出 L-curve 与衰减拟合。

CT 孔体、内部孔喉使用全分辨率 pnextract 数据，排除连接边界的孔喉及非正/无效半径或体积。孔体和孔喉共用总网络体积归一化分母；NMR 谱按正幅值总和归一化。两者分箱宽度不同，使用独立纵轴，不能据图高比较绝对孔隙率。灰色重叠是两个直方图逐箱较小值，仅是图形重叠。

本次未进行网格、时间步长或物理参数的收敛/敏感性扫描；这些结果不构成真实三维模拟或独立物性测量。可复查数据包括分组表、全部代表切片与网格、加权衰减、反演谱、L-curve 指标、拟合残差、绘图源数据和 run_manifest.json。
"""
    (output / "结果说明.md").write_text(report,encoding="utf-8")
    print(json.dumps({"alpha":manifest["alpha"],"peak_ms":manifest["simulation_t2_peak_ms"],"fit_error":fit_relative_error},ensure_ascii=False),flush=True)


def main():
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["simulate","finalize"],required=True)
    parser.add_argument("--group-id",type=int)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.stage=="simulate": simulate(args.output_dir,args.group_id)
    else: finalize(args.output_dir)


if __name__=="__main__":
    main()
