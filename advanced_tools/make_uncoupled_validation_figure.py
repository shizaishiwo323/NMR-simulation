"""Plot uncoupled large/small T2 spectra for validation against Auto_T2-T2."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_combined_figures import (
    OUT,
    TRI_TABLES,
    create_mesh_from_geom,
    create_pore_geom_arb,
    calc_arbitrary_pore_properties,
    cfg_get,
    depth_large,
    depth_small,
    get_single_pore_water_area,
    L_large,
    L_small,
    table_path,
)

try:
    import pygimli as pg
except Exception:
    pg = None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t2_frame = pd.read_excel(table_path("AllStates_T2_Components_Uncoupled_Coupled.xlsx"))
    t2 = t2_frame["T2_Time_ms"].to_numpy(dtype=float)

    gammas = [float(v) for v in cfg_get("triangle_ca.triangle_angles_deg", [60.0, 60.0, 60.0])]
    theta = float(cfg_get("triangle_ca.contact_angle_deg", 0.0))
    area_l, _, r_i_l, r_d_l, verts_l, inc_l = calc_arbitrary_pore_properties(L_large, gammas)
    area_s, _, r_i_s, r_d_s, verts_s, inc_s = calc_arbitrary_pore_properties(L_small, gammas)
    total_volume = area_l * depth_large + area_s * depth_small
    selected_rm = [r_i_l * 1.5, r_i_l * 0.99, r_d_l * 0.99, r_i_s * 0.99, r_d_s * 0.99]

    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    fig.suptitle("Uncoupled validation: water distribution and large/small T2 spectra", fontsize=16, fontweight="bold")
    row_for = {("Drainage", "morph"): 0, ("Drainage", "t2"): 1, ("Imbibition", "morph"): 2, ("Imbibition", "t2"): 3}

    for process in ["Drainage", "Imbibition"]:
        for idx, rm in enumerate(selected_rm, start=1):
            aw_l = get_single_pore_water_area(L_large, rm, process, theta, gammas)
            aw_s = get_single_pore_water_area(L_small, rm, process, theta, gammas)
            sw = (aw_l * depth_large + aw_s * depth_small) / total_volume
            key = f"P{idx}_{process}_Sw{sw*100:.2f}pct"

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

            ax_t = axes[row_for[(process, "t2")], idx - 1]
            large = t2_frame[f"{key}_Uncoupled_Large"].to_numpy(dtype=float)
            small = t2_frame[f"{key}_Uncoupled_Small"].to_numpy(dtype=float)
            total = large + small
            scale = max(float(np.nanmax(total)), 1e-12)
            ax_t.fill_between(t2, small / scale, color="#fdae6b", alpha=0.75, label="Small pore")
            ax_t.fill_between(t2, large / scale, color="#66c2a4", alpha=0.75, label="Large pore")
            # Keep this validation figure strictly component-only.
            ax_t.set_xscale("log")
            ax_t.set_xlim(1, 10000)
            ax_t.set_ylim(0, 1.5)
            ax_t.grid(False)
            if idx == 1:
                ax_t.set_ylabel("Normalized amplitude")
                ax_t.legend(fontsize=7)
            if process == "Imbibition":
                ax_t.set_xlabel("T2 (ms)")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / "Combined_00D_Uncoupled_LargeSmallOnly_Normalized_Validation.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
