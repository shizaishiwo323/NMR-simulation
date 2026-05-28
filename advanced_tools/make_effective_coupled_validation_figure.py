"""Add an effective exchange-coupled comparison based on Zhou et al. (2026).

This is not the strict water-throat PDE coupling.  It is a two-compartment
exchange model whose exchange scale follows the GRL 2026 pore-coupling
concept: saturation controls water-path connectivity phi_c and pathway
tortuosity lambda_mean, with D_eff = D / lambda_mean.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from make_combined_figures import (
    OUT,
    TRI_TABLES,
    calc_arbitrary_pore_properties,
    cfg_get,
    create_mesh_from_geom,
    create_pore_geom_arb,
    depth_large,
    depth_small,
    get_single_pore_water_area,
    L_large,
    L_small,
    table_path,
)
from triangle_contact_angle_full_suite import D_scaled, L_t, t2_axis

try:
    import pygimli as pg
except Exception:
    pg = None


ALPHA = 1.0


def invert_t2_fixed(times: np.ndarray, signal: np.ndarray, bins: np.ndarray, alpha: float) -> np.ndarray:
    if np.max(signal) <= 1e-12 or signal[0] <= 1e-12:
        return np.zeros(len(bins))
    kernel = np.exp(-np.outer(times, 1.0 / bins))
    n = len(bins)
    reg = np.zeros((n, n))
    for i in range(1, n - 1):
        reg[i, i - 1] = 1.0
        reg[i, i] = -2.0
        reg[i, i + 1] = 1.0
    reg[0, 0] = -1.0
    reg[0, 1] = 1.0
    reg[-1, -2] = 1.0
    reg[-1, -1] = -1.0
    spec, _ = nnls(
        np.vstack((kernel, reg * alpha)),
        np.concatenate((signal / signal[0], np.zeros(n))),
    )
    return spec * signal[0]


def zhou_2026_connectivity(sw_global: float) -> tuple[float, float]:
    """Approximate phi_c and lambda_mean from Zhou et al. (2026) trends.

    The paper states that water pathways are well connected at 100%-85%,
    both tortuosity and connectivity change strongly from 85%-70%, and
    coupling becomes blocked/decoupled at Sw <= 40%.  Exact digitized figure
    values are not in the manuscript text, so these are transparent control
    points for a GRL-style sensitivity model.
    """

    sw_ref = np.array([0.0258, 0.25, 0.40, 0.70, 0.85, 1.00])
    phi_ref = np.array([0.0, 0.0, 0.0, 0.35, 0.85, 1.0])
    lambda_ref = np.array([3.0, 3.0, 3.0, 2.7, 2.2, 1.2])
    sw = float(np.clip(sw_global, sw_ref[0], sw_ref[-1]))
    return float(np.interp(sw, sw_ref, phi_ref)), float(np.interp(sw, sw_ref, lambda_ref))


def zhou_2026_exchange_rate(sw_global: float) -> tuple[float, float, float]:
    """Return k_ex [1/ms], phi_c, and lambda_mean.

    k_ex is the diffusion rate through the exchange length reduced by the
    GRL-style water connectivity:

        D_eff = D / lambda_mean
        k_ex = phi_c^2 * D_eff / L_t^2

    phi_c = 0 below the decoupling threshold (Sw <= 40%), matching the paper's
    stated transition to a decoupled state.
    """

    phi_c, lambda_mean = zhou_2026_connectivity(sw_global)
    d_eff = D_scaled / max(lambda_mean, 1e-12)
    k_ex = (phi_c**2) * d_eff / max(L_t**2, 1e-12)
    return float(k_ex), phi_c, lambda_mean


def apply_exchange(large: np.ndarray, small: np.ndarray, times: np.ndarray, k_ex: float) -> tuple[np.ndarray, np.ndarray]:
    large = np.asarray(large, dtype=float)
    small = np.asarray(small, dtype=float)
    if k_ex <= 0 or large[0] <= 1e-12 or small[0] <= 1e-12:
        return large.copy(), small.copy()

    dt = float(np.median(np.diff(times)))
    out_l = np.zeros_like(large)
    out_s = np.zeros_like(small)
    out_l[0] = large[0]
    out_s[0] = small[0]

    v_l0 = max(large[0], 1e-12)
    v_s0 = max(small[0], 1e-12)
    v_tot = v_l0 + v_s0
    k_l_to_s = k_ex * v_s0 / v_tot
    k_s_to_l = k_ex * v_l0 / v_tot

    for i in range(len(times) - 1):
        r_l = large[i + 1] / max(large[i], 1e-12)
        r_s = small[i + 1] / max(small[i], 1e-12)
        m_l = out_l[i] * r_l
        m_s = out_s[i] * r_s
        flux_l_to_s = dt * k_l_to_s * m_l
        flux_s_to_l = dt * k_s_to_l * m_s
        out_l[i + 1] = max(m_l - flux_l_to_s + flux_s_to_l, 0.0)
        out_s[i + 1] = max(m_s - flux_s_to_l + flux_l_to_s, 0.0)
    return out_l, out_s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_excel(TRI_TABLES / "AllStates_Summary.xlsx")
    decay = pd.read_excel(table_path("AllStates_Decay_Components_Uncoupled_Coupled.xlsx"))
    times = decay["Time_ms"].to_numpy(dtype=float)

    gammas = [float(v) for v in cfg_get("triangle_ca.triangle_angles_deg", [60.0, 60.0, 60.0])]
    theta = float(cfg_get("triangle_ca.contact_angle_deg", 0.0))
    area_l, _, r_i_l, r_d_l, verts_l, inc_l = calc_arbitrary_pore_properties(L_large, gammas)
    area_s, _, r_i_s, r_d_s, verts_s, inc_s = calc_arbitrary_pore_properties(L_small, gammas)
    total_volume = area_l * depth_large + area_s * depth_small
    selected_rm = [r_i_l * 1.5, r_i_l * 0.99, r_d_l * 0.99, r_i_s * 0.99, r_d_s * 0.99]

    export = {"T2_Time_ms": t2_axis}
    audit_rows = []
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    fig.suptitle("Effective exchange-coupled validation after Zhou et al. (2026) scaling", fontsize=16, fontweight="bold")
    row_for = {("Drainage", "morph"): 0, ("Drainage", "t2"): 1, ("Imbibition", "morph"): 2, ("Imbibition", "t2"): 3}

    for process in ["Drainage", "Imbibition"]:
        for idx, rm in enumerate(selected_rm, start=1):
            aw_l = get_single_pore_water_area(L_large, rm, process, theta, gammas)
            aw_s = get_single_pore_water_area(L_small, rm, process, theta, gammas)
            sw = (aw_l * depth_large + aw_s * depth_small) / total_volume
            sw_l = aw_l / area_l
            sw_s = aw_s / area_s
            key = f"P{idx}_{process}_Sw{sw*100:.2f}pct"
            k_ex, phi_c, lambda_mean = zhou_2026_exchange_rate(sw)

            ax_m = axes[row_for[(process, "morph")], idx - 1]
            geom_l, _ = create_pore_geom_arb(area_l, verts_l, inc_l, gammas, area_l - aw_l, theta, False, 0.0)
            geom_s, _ = create_pore_geom_arb(area_s, verts_s, inc_s, gammas, area_s - aw_s, theta, True, 0.0)
            mesh_l = create_mesh_from_geom(geom_l, area_l)
            mesh_s = create_mesh_from_geom(geom_s, area_s)
            v_l = verts_l + [verts_l[0]]
            v_s = [[v[0], -v[1]] for v in verts_s] + [[verts_s[0][0], -verts_s[0][1]]]
            ax_m.plot([v[0] for v in v_l], [v[1] for v in v_l], "k-", lw=1.0)
            ax_m.plot([v[0] for v in v_s], [v[1] for v in v_s], "k-", lw=1.0)
            if pg is not None:
                if mesh_l.nodeCount() > 5:
                    pg.show(mesh_l, ax=ax_m, hold=True)
                if mesh_s.nodeCount() > 5:
                    pg.show(mesh_s, ax=ax_m, hold=True)
            ax_m.set_aspect("equal")
            ax_m.set_xlim(-13, 13)
            ax_m.set_ylim(-9, 20)
            ax_m.set_title(f"{process[:3]} P{idx}: Sw={sw*100:.1f}%", fontsize=10)
            ax_m.set_xticks([])
            ax_m.set_yticks([])
            if idx == 1:
                ax_m.set_ylabel(f"{process}\nwater distribution")

            unc_l = decay[f"{key}_Uncoupled_Large"].to_numpy(dtype=float)
            unc_s = decay[f"{key}_Uncoupled_Small"].to_numpy(dtype=float)
            coup_l_decay, coup_s_decay = apply_exchange(unc_l, unc_s, times, k_ex)
            spec_l = invert_t2_fixed(times, coup_l_decay, t2_axis, ALPHA)
            spec_s = invert_t2_fixed(times, coup_s_decay, t2_axis, ALPHA)
            scale = max(float(np.nanmax(spec_l + spec_s)), 1e-12)

            export[f"{key}_EffectiveCoupled_Large"] = spec_l
            export[f"{key}_EffectiveCoupled_Small"] = spec_s
            export[f"{key}_EffectiveCoupled_Total"] = spec_l + spec_s
            audit_rows.append(
                {
                    "key": key,
                    "sw_global": sw,
                    "sw_large_local": sw_l,
                    "sw_small_local": sw_s,
                    "phi_c": phi_c,
                    "lambda_mean": lambda_mean,
                    "D_eff_um2_per_ms": D_scaled / max(lambda_mean, 1e-12),
                    "k_ex_1_per_ms": k_ex,
                    "mixing_time_ms": 1.0 / k_ex if k_ex > 0 else np.inf,
                }
            )

            ax_t = axes[row_for[(process, "t2")], idx - 1]
            ax_t.fill_between(t2_axis, spec_s / scale, color="#fdae6b", alpha=0.75, label="Small pore")
            ax_t.fill_between(t2_axis, spec_l / scale, color="#66c2a4", alpha=0.75, label="Large pore")
            ax_t.set_xscale("log")
            ax_t.set_xlim(1, 10000)
            ax_t.set_ylim(0, 1.5)
            ax_t.set_title(f"k={k_ex:.4f} 1/ms", fontsize=9)
            if idx == 1:
                ax_t.set_ylabel("Normalized amplitude")
                ax_t.legend(fontsize=7)
            if process == "Imbibition":
                ax_t.set_xlabel("T2 (ms)")

    pd.DataFrame(export).to_excel(OUT / "EffectiveCoupled_T2_Zhou2026GRLScaled.xlsx", index=False)
    pd.DataFrame(audit_rows).to_excel(OUT / "EffectiveCoupled_Zhou2026GRLScaled_Parameters.xlsx", index=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / "Combined_00E_EffectiveCoupled_LargeSmallOnly_Zhou2026GRLScaled.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
