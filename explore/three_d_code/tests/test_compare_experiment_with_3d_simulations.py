from pathlib import Path

import numpy as np
import pandas as pd

from advanced_tools.compare_experiment_with_3d_simulations import (
    load_experiment_decay,
    load_simulation_decay,
    parse_simulation_arg,
    save_decay_long,
)


def test_load_experiment_uses_first_two_columns_and_normalizes(tmp_path: Path):
    path = tmp_path / "experiment.csv"
    pd.DataFrame(
        {
            0: [0.0, 1.0, 2.0, 3.0],
            1: [99.0, 10.0, 5.0, 2.5],
            2: [123.0, 123.0, 123.0, 123.0],
        }
    ).to_csv(path, header=False, index=False)

    series = load_experiment_decay(path)

    assert series.name == "Experiment"
    assert series.time_ms.tolist() == [1.0, 2.0, 3.0]
    assert np.allclose(series.normalized_signal, [1.0, 0.5, 0.25])


def test_load_simulation_accepts_normalized_or_signal_columns(tmp_path: Path):
    normalized_path = tmp_path / "normalized.csv"
    signal_path = tmp_path / "signal.csv"
    pd.DataFrame({"time_ms": [0.0, 5.0, 10.0], "normalized_signal": [1.0, 0.8, 0.6]}).to_csv(
        normalized_path, index=False
    )
    pd.DataFrame({"time_ms": [0.0, 5.0, 10.0], "signal": [20.0, 10.0, 5.0]}).to_csv(signal_path, index=False)

    a = load_simulation_decay("A", normalized_path)
    b = load_simulation_decay("B", signal_path)

    assert np.allclose(a.normalized_signal, [1.0, 0.75])
    assert np.allclose(b.normalized_signal, [1.0, 0.5])


def test_parse_simulation_arg_and_save_long_form(tmp_path: Path):
    label, path = parse_simulation_arg("Full CPU 3D|C:/tmp/decay.csv")
    assert label == "Full CPU 3D"
    assert str(path).replace("\\", "/") == "C:/tmp/decay.csv"

    exp_path = tmp_path / "experiment.csv"
    sim_path = tmp_path / "sim.csv"
    pd.DataFrame({0: [1.0, 2.0], 1: [10.0, 8.0]}).to_csv(exp_path, header=False, index=False)
    pd.DataFrame({"time_ms": [5.0, 10.0], "normalized_signal": [1.0, 0.7]}).to_csv(sim_path, index=False)
    exp = load_experiment_decay(exp_path)
    sim = load_simulation_decay("Sim", sim_path)
    out = tmp_path / "long.csv"

    save_decay_long([exp, sim], out)

    saved = pd.read_csv(out)
    assert set(saved["series"]) == {"Experiment", "Sim"}
    assert len(saved) == 4
