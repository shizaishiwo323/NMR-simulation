"""Back-calculating pore sizes from NMR T2 when a big and a small pore are diffusively coupled.

Inputs: the 20 cached cases of ``ideal_two_pore_throat_sweep_nmr.py`` (closed 2-D geometry: one big disk B,
one straight throat of width w and length L, one small disk S; pyGIMLi Bloch-Torrey simulation, nmr_t2
L-curve NNLS at SNR 1000).

Forward models (units: um, ms)
  * two-site exchange (Zimmerman & Brittin, 1957): each pore well mixed, throat = diffusive conductance D w / L;
  * pore-throat-pore with a 1-D relaxing throat ("TL", transmission line): throat profile solves
    D u'' = (R_t - lambda) u, R_t = rho2 / h + 1 / T2B; exact characteristic equation, modes and amplitudes.
    Its quasi-static reduction is a two-site model with exchange conductance G_x = D w q csch(qL) and
    sink conductance G_s = D w q tanh(qL/2), q = sqrt(R_t / D).

Back-calculation routes
  A  naive isolated-pore formula per NNLS peak, r = 2 rho2 / (1/T2 - 1/T2B);
  B  amplitude-ratio route: r_S / r_B = sqrt(f_S / f_B) (2-D disks), r_B from the main peak;
  C  coupled-model inversion: two-site exchange with free conductance (no throat information);
  D  coupled-model inversion: TL model with throat w, L known (e.g. from imaging or SIP).

    python advanced_tools/coupled_two_pore_backcalc.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import advanced_tools.ideal_two_pore_channel_nmr_sip as base  # noqa: E402

D, RHO, T2B = base.DIFFUSION_UM2_MS, base.RHO2_UM_MS, base.BULK_T2_MS
R_BIG, R_SMALL = base.GEOMETRY_UM["big_radius"], base.GEOMETRY_UM["small_radius"]
SWEEP = ROOT / "simulation_outputs" / "ideal_two_pore_throat_sweep_nmr_20261012"
OUT = ROOT / "simulation_outputs" / "coupled_two_pore_backcalc_20261013"
MODE_T2_FLOOR_MS = 5.0  # NNLS peaks below this are noise / throat-wall modes, not pore modes
REPRESENTATIVE = [(0.1, 30.0), (0.5, 15.0), (1.0, 15.0), (0.5, 2.0), (4.0, 2.0)]
LOW_SNR_NOISE = 0.01  # SNR 100 re-inversion of the representative cases


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def disk(radius, h=0.0):
    """Area, relaxing wall length (mouth arc removed) and fast-diffusion relaxation rate of a 2-D disk."""

    mouth = 2.0 * radius * math.asin(min(h / radius, 1.0)) if h > 0 else 0.0
    area = math.pi * radius**2
    wall = 2.0 * math.pi * radius - mouth
    return area, wall, RHO * wall / area + 1.0 / T2B


def throat_rate(width):
    return RHO / (0.5 * width) + 1.0 / T2B


def naive_radius(t2_ms):
    rate = 1.0 / t2_ms - 1.0 / T2B
    return 2.0 * RHO / rate if rate > 0 else math.nan


# ---------------------------------------------------------------------------
# Forward model 1: two-site exchange
# ---------------------------------------------------------------------------
def two_site(rate_s, rate_b, k_s, k_b, p_s):
    """Eigen-rates (fast, slow) and fast-mode amplitude for dm/dt = -K m, m(0) = (p_S, p_B)."""

    a, d = rate_s + k_s, rate_b + k_b
    disc = math.sqrt((a - d) ** 2 + 4.0 * k_s * k_b)
    lam_fast, lam_slow = 0.5 * (a + d + disc), 0.5 * (a + d - disc)
    mean_rate = p_s * rate_s + (1.0 - p_s) * rate_b  # = -dS/dt at t = 0 (exchange conserves magnetization)
    amp_fast = (mean_rate - lam_slow) / (lam_fast - lam_slow)
    return lam_fast, lam_slow, amp_fast


# ---------------------------------------------------------------------------
# Forward model 2: pore - 1-D relaxing throat - pore (transmission line)
# ---------------------------------------------------------------------------
def _s_c(lam, y, rate_t):
    q2 = (rate_t - lam) / D
    if q2 > 0:
        q = math.sqrt(q2)
        return math.sinh(q * y) / q, math.cosh(q * y)
    if q2 < 0:
        k = math.sqrt(-q2)
        return math.sin(k * y) / k, math.cos(k * y)
    return y, 1.0


def tl_modes(area_s, area_b, rate_s, rate_b, width, length, n_scan=6000):
    """All modes (rate, amplitude) of the pore-throat-pore network, sorted by rate.

    Characteristic equation (multiplied by S(L)^2 so it has no poles):
        [(R_S - l) A_S S + g C] [(R_B - l) A_B S + g C] - g^2 = 0,  S = sinh(qL)/q, C = cosh(qL), g = D w.
    Amplitude of mode n for uniform initial magnetization: (int phi)^2 / (int phi^2 * A_tot).
    """

    rate_t, g = throat_rate(width), D * width

    def char(lam):
        s, c = _s_c(lam, length, rate_t)
        return ((rate_s - lam) * area_s * s + g * c) * ((rate_b - lam) * area_b * s + g * c) - g * g

    lam_max = rate_t + 30.0 * D * (math.pi / length) ** 2
    grid = np.geomspace(1e-7, lam_max, n_scan)
    vals = np.array([char(lam) for lam in grid])
    xs, wq = np.polynomial.legendre.leggauss(160)
    x, wx = 0.5 * length * (xs + 1.0), 0.5 * length * wq
    area_tot = area_s + area_b + width * length
    modes = []
    for i in np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]:
        lam = brentq(char, grid[i], grid[i + 1], xtol=1e-15, rtol=1e-13)
        s_l, c_l = _s_c(lam, length, rate_t)
        if abs(s_l) < 1e-9 * length:  # spurious zero introduced by the S^2 factor (eigenfunction is identically 0)
            continue
        a11 = (rate_s - lam) * area_s * s_l + g * c_l
        a22 = (rate_b - lam) * area_b * s_l + g * c_l
        alpha, beta = (a22, g) if abs(a11) < abs(a22) else (g, a11)  # null vector, u = alpha S(L-x) + beta S(x)
        c_s, c_b = alpha * s_l, beta * s_l
        u = np.array([alpha * _s_c(lam, length - xi, rate_t)[0] + beta * _s_c(lam, xi, rate_t)[0] for xi in x])
        i1 = area_s * c_s + area_b * c_b + width * np.sum(wx * u)
        i2 = area_s * c_s**2 + area_b * c_b**2 + width * np.sum(wx * u**2)
        modes.append((lam, i1**2 / (i2 * area_tot)))
    return sorted(modes)


def quasi_static(area_s, area_b, rate_s, rate_b, width, length):
    """Two-site reduction of the TL model: exchange conductance G_x and sink conductance G_s."""

    q = math.sqrt(throat_rate(width) / D)
    ql = q * length
    g_x = D * width * q / math.sinh(ql)
    g_s = D * width * q * math.tanh(0.5 * ql)
    return {"q_per_um": q, "qL": ql, "G_x": g_x, "G_s": g_s,
            "rate_s_eff": rate_s + g_s / area_s, "rate_b_eff": rate_b + g_s / area_b,
            "k_s": g_x / area_s, "k_b": g_x / area_b}


# ---------------------------------------------------------------------------
# Observables from an NNLS spectrum
# ---------------------------------------------------------------------------
def nnls_modes(t2_bins, spectrum):
    """Two largest pore peaks (T2 > floor): rate-mean T2, T2lm, bin peak, fraction (of the two-peak total)."""

    t2, a = np.asarray(t2_bins), np.asarray(spectrum)
    peaks = [p for p in base.spectrum_peaks(t2, a, threshold=0.0) if p["position"] > MODE_T2_FLOOR_MS]
    peaks = sorted(sorted(peaks, key=lambda p: -p["fraction"])[:2], key=lambda p: p["position"])
    # Valley between the two peaks splits the spectrum into modes.
    out = []
    if len(peaks) == 2:
        i0, i1 = (int(np.argmin(np.abs(t2 - p["position"]))) for p in peaks)
        valley = i0 + int(np.argmin(a[i0 : i1 + 1]))
        windows = [(t2 > MODE_T2_FLOOR_MS) & (np.arange(t2.size) < valley), np.arange(t2.size) >= valley]
    else:
        windows = [t2 > MODE_T2_FLOOR_MS]
    total = sum(a[m].sum() for m in windows)
    for peak, m in zip(peaks if len(peaks) == 2 else peaks[-1:], windows):
        rate = float(np.sum(a[m] / t2[m]) / a[m].sum())
        out.append({"t2_peak_ms": peak["position"], "t2lm_ms": peak["t2lm"], "t2_rate_mean_ms": 1.0 / rate,
                    "fraction": float(a[m].sum() / total)})
    return out


# ---------------------------------------------------------------------------
# Back-calculation routes
# ---------------------------------------------------------------------------
def route_a(modes):
    return [naive_radius(m["t2lm_ms"]) for m in modes]


def route_b(modes):
    """r_B from the main peak (isolated formula), r_S / r_B = sqrt(f_S / f_B) for 2-D disks."""

    if len(modes) < 2:
        return [math.nan, naive_radius(modes[-1]["t2lm_ms"])]
    small, big = modes
    r_b = naive_radius(big["t2lm_ms"])
    return [r_b * math.sqrt(small["fraction"] / big["fraction"]), r_b]


def _disk_rates(r_s, r_b):
    a_s, a_b = math.pi * r_s**2, math.pi * r_b**2
    return a_s, a_b, 2 * RHO / r_s + 1 / T2B, 2 * RHO / r_b + 1 / T2B


def route_c(modes):
    """Two-site exchange with unknown r_S, r_B and conductance G: 3 equations (lambda_fast, lambda_slow, A_fast)."""

    if len(modes) < 2:
        return [math.nan, math.nan, math.nan]
    small, big = modes
    obs = np.array([math.log(1 / small["t2lm_ms"]), math.log(1 / big["t2lm_ms"]), small["fraction"]])

    def resid(p):
        r_s, r_b, g = np.exp(p)
        a_s, a_b, rs, rb = _disk_rates(r_s, r_b)
        lf, ls, af = two_site(rs, rb, g / a_s, g / a_b, a_s / (a_s + a_b))
        return [math.log(lf) - obs[0], math.log(ls) - obs[1], (af - obs[2]) / max(obs[2], 1e-3)]

    x0 = np.log([route_b(modes)[0], naive_radius(big["t2lm_ms"]), 0.01])
    fit = least_squares(resid, x0, xtol=1e-12, ftol=1e-12)
    return list(np.exp(fit.x))


def route_d(modes, width, length):
    """TL model with known throat (w, L): unknown r_S, r_B; fit lambda_fast, lambda_slow, A_fast."""

    if len(modes) < 2:
        return [math.nan, math.nan]
    small, big = modes
    obs = np.array([math.log(1 / small["t2lm_ms"]), math.log(1 / big["t2lm_ms"]), small["fraction"]])

    def resid(p):
        r_s, r_b = np.exp(p)
        h = 0.5 * width
        a_s, _, rs = disk(r_s, h)
        a_b, _, rb = disk(r_b, h)
        pore = tl_modes(a_s, a_b, rs, rb, width, length, n_scan=1500)[:2]  # keep weak exchange modes too
        if len(pore) < 2:
            return [10.0, 10.0, 10.0]
        (ls, _), (lf, af) = pore[0], pore[1]
        return [math.log(lf) - obs[0], math.log(ls) - obs[1], 0.5 * (math.log(max(af, 1e-12)) - math.log(obs[2]))]

    x0 = np.log(route_b(modes))
    fit = least_squares(resid, x0, xtol=1e-10, ftol=1e-10, diff_step=1e-4)
    return list(np.exp(fit.x))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def load_cases():
    cases = {}
    for path in (SWEEP / "cases").glob("*.json"):
        c = json.loads(path.read_text(encoding="utf-8"))
        cases[(c["width_um"], c["length_um"])] = c
    return dict(sorted(cases.items()))


def analyse_case(c):
    w, length_nom = c["width_um"], c["length_um"]
    h = 0.5 * w
    a_s, wall_s, rate_s = disk(R_SMALL, h)
    a_b, wall_b, rate_b = disk(R_BIG, h)
    length = c["area_throat_um2"] / w  # wall-to-wall throat length of the mesh
    g = D * w / length
    p_s = a_s / (a_s + a_b)
    psi = (g / a_s) / (rate_s - rate_b)
    ts = two_site(rate_s, rate_b, g / a_s, g / a_b, p_s)
    qs = quasi_static(a_s, a_b, rate_s, rate_b, w, length)
    qs_two = two_site(qs["rate_s_eff"], qs["rate_b_eff"], qs["k_s"], qs["k_b"], p_s)
    psi_eff = qs["k_s"] / (qs["rate_s_eff"] - qs["rate_b_eff"])
    tl = tl_modes(a_s, a_b, rate_s, rate_b, w, length)
    pore_modes = [m for m in tl if m[1] > 1e-6][:2]
    t = np.arange(len(c["decay_every_1ms"]), dtype=float)
    sim = np.asarray(c["decay_every_1ms"])
    tl_decay = sum(a * np.exp(-lam * t) for lam, a in tl)
    ts_decay = ts[2] * np.exp(-ts[0] * t) + (1 - ts[2]) * np.exp(-ts[1] * t)
    nn = nnls_modes(c["t2_bins_ms"], c["spectrum"])
    row = {
        "width_um": w, "length_um": length_nom, "throat_length_mesh_um": length,
        "area_small_um2": a_s, "area_big_um2": a_b, "area_throat_um2": w * length, "p_small": p_s,
        "rate_small_iso": rate_s, "rate_big_iso": rate_b, "rate_throat": throat_rate(w),
        "G_um2_ms": g, "Psi": psi, "qL": qs["qL"], "q_per_um": qs["q_per_um"], "Psi_eff": psi_eff, "G_x": qs["G_x"], "G_s": qs["G_s"],
        "k_s_qs": qs["k_s"], "k_b_qs": qs["k_b"], "rate_small_eff": qs["rate_s_eff"], "rate_big_eff": qs["rate_b_eff"],
        "wall_small_um": wall_s, "wall_big_um": wall_b,
        "throat_to_small_area": w * length / a_s,
        "two_site_t2_fast_ms": 1 / ts[0], "two_site_t2_slow_ms": 1 / ts[1], "two_site_amp_fast": ts[2],
        "qs_t2_fast_ms": 1 / qs_two[0], "qs_t2_slow_ms": 1 / qs_two[1], "qs_amp_fast": qs_two[2],
        "tl_t2_slow_ms": 1 / pore_modes[0][0], "tl_amp_slow": pore_modes[0][1],
        "tl_t2_fast_ms": 1 / pore_modes[1][0], "tl_amp_fast": pore_modes[1][1],
        "tl_amp_sum": sum(a for _, a in tl),
        "eta_retained": pore_modes[1][1] / p_s,
        "regime": "I" if pore_modes[1][1] / p_s > 0.8 else ("II" if pore_modes[1][1] / p_s > 0.1 else "III"),
        "tl_decay_max_abs_err": float(np.max(np.abs(tl_decay - sim))),
        "two_site_decay_max_abs_err": float(np.max(np.abs(ts_decay - sim))),
        "sim_t2_small_region_fit_ms": c["t2_small_region_fit_ms"], "sim_t2_big_region_fit_ms": c["t2_big_region_fit_ms"],
        "nnls_alpha": c["alpha"],
        "nnls_t2_fast_lm_ms": nn[0]["t2lm_ms"] if len(nn) == 2 else math.nan,
        "nnls_amp_fast": nn[0]["fraction"] if len(nn) == 2 else 0.0,
        "nnls_t2_slow_lm_ms": nn[-1]["t2lm_ms"],
    }
    return row, nn


def low_snr_modes(c):
    t = np.arange(len(c["decay_every_1ms"]), dtype=float)
    inv = base.invert(t, np.asarray(c["decay_every_1ms"]), f"w{c['width_um']}_L{c['length_um']}_snr100",
                      np.random.default_rng(base.NOISE_SEED), LOW_SNR_NOISE)
    lc = inv["lcurve"]
    return lc, nnls_modes(lc.t2_bins_ms, lc.spectrum)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    rows, observed = [], {}
    for key, c in cases.items():
        row, nn = analyse_case(c)
        rows.append(row)
        observed[key] = nn
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "forward_models_vs_simulation.csv", index=False)

    back = []
    snr = {}
    for key in REPRESENTATIVE:
        c, nn = cases[key], observed[key]
        r_a = route_a(nn)
        r_b = route_b(nn)
        r_c = route_c(nn)
        r_d = route_d(nn, c["width_um"], c["area_throat_um2"] / c["width_um"])
        row_tl = table[(table.width_um == key[0]) & (table.length_um == key[1])].iloc[0]
        exact_obs = [{"t2lm_ms": row_tl.tl_t2_fast_ms, "fraction": row_tl.tl_amp_fast / (row_tl.tl_amp_fast + row_tl.tl_amp_slow)},
                     {"t2lm_ms": row_tl.tl_t2_slow_ms, "fraction": row_tl.tl_amp_slow / (row_tl.tl_amp_fast + row_tl.tl_amp_slow)}]
        r_d_exact = route_d(exact_obs, c["width_um"], c["area_throat_um2"] / c["width_um"])
        lc, nn100 = low_snr_modes(c)
        snr[key] = {"t2_bins_ms": lc.t2_bins_ms.tolist(), "spectrum": lc.spectrum.tolist(), "alpha": float(lc.best_regularization), "modes": nn100}
        row = table[(table.width_um == key[0]) & (table.length_um == key[1])].iloc[0]
        back.append({
            "width_um": key[0], "length_um": key[1], "Psi": row.Psi, "qL": row.qL, "Psi_eff": row.Psi_eff,
            "n_modes_snr1000": len(nn),
            "fast_t2lm_ms": nn[0]["t2lm_ms"] if len(nn) == 2 else math.nan,
            "fast_fraction": nn[0]["fraction"] if len(nn) == 2 else 0.0,
            "slow_t2lm_ms": nn[-1]["t2lm_ms"],
            "A_r_small": r_a[0] if len(nn) == 2 else math.nan, "A_r_big": r_a[-1],
            "B_r_small": r_b[0], "B_r_big": r_b[1],
            "C_r_small": r_c[0], "C_r_big": r_c[1], "C_G": r_c[2],
            "D_r_small": r_d[0], "D_r_big": r_d[1],
            "D_exactobs_r_small": r_d_exact[0], "D_exactobs_r_big": r_d_exact[1], "regime": row_tl.regime,
            "true_r_small": R_SMALL, "true_r_big": R_BIG, "true_G": D * key[0] / (c["area_throat_um2"] / key[0]),
            "R_eff_true_um": 2 * (row.area_small_um2 + row.area_big_um2) / (2 * math.pi * (R_SMALL + R_BIG)),
            "snr100_alpha": snr[key]["alpha"],
            "snr100_fast_fraction": nn100[0]["fraction"] if len(nn100) == 2 else 0.0,
            "snr100_fast_t2lm_ms": nn100[0]["t2lm_ms"] if len(nn100) == 2 else math.nan,
            "snr100_slow_t2lm_ms": nn100[-1]["t2lm_ms"],
        })
    back = pd.DataFrame(back)
    back.to_csv(OUT / "backcalculation_representative_cases.csv", index=False)
    (OUT / "nnls_modes.json").write_text(json.dumps({f"{k[0]:g}_{k[1]:g}": v for k, v in observed.items()}, indent=1), encoding="utf-8")
    (OUT / "snr100_spectra.json").write_text(json.dumps({f"{k[0]:g}_{k[1]:g}": v for k, v in snr.items()}), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(table[["width_um", "length_um", "Psi", "qL", "Psi_eff", "two_site_t2_fast_ms", "qs_t2_fast_ms", "tl_t2_fast_ms", "nnls_t2_fast_lm_ms",
                 "two_site_amp_fast", "qs_amp_fast", "tl_amp_fast", "nnls_amp_fast", "tl_t2_slow_ms", "nnls_t2_slow_lm_ms",
                 "tl_decay_max_abs_err", "two_site_decay_max_abs_err"]].round(4).to_string(index=False))
    print(back.round(3).T.to_string())


if __name__ == "__main__":
    main()
