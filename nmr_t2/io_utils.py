"""Shared I/O and preprocessing utilities for NMR T2 workflows.

This module intentionally centralizes all parsing and export conventions so
that NNLS, L-curve, plotting, and Gaussian decomposition pipelines produce
consistent file names, sheet names, and data columns.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .models import TrimmedSignal


def safe_token(text: str) -> str:
    """Convert arbitrary text into a filesystem-safe token.

    Parameters
    ----------
    text:
        Any source string such as dataset name, sheet name, or signal name.

    Returns
    -------
    str
        A compact token containing only alphanumeric characters, `_`, and `-`.
    """

    cleaned = re.sub(r"\s+", "_", str(text).strip())
    cleaned = re.sub(r"[^0-9a-zA-Z_\-]+", "", cleaned)
    return cleaned or "item"


def safe_sheet_name(name: str) -> str:
    """Create an Excel-safe worksheet name.

    Excel sheet names:
    - cannot contain `[]:*?/\\`
    - cannot exceed 31 characters
    """

    invalid = set("[]:*?/\\")
    cleaned = "".join("_" if c in invalid else c for c in str(name)).strip()
    return (cleaned or "Sheet")[:31]


def parse_time_cell(value: object) -> Tuple[bool, float]:
    """Parse one Excel time cell.

    Supported values include:
    - numeric (int/float)
    - scientific notation in string form
    - text like `t=0.001s` or values containing extra symbols
    """

    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return True, float(value)

    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return False, np.nan

        text = text.replace("t=", "").replace("s", "").replace("秒", "")
        try:
            parsed = float(text)
            if np.isfinite(parsed):
                return True, parsed
        except ValueError:
            pass

        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if match:
            parsed = float(match.group(0))
            if np.isfinite(parsed):
                return True, parsed

    return False, np.nan


def cell_to_float(value: object) -> float:
    """Convert a generic Excel cell value into float, returning NaN if invalid."""

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if np.isfinite(value) else np.nan

    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return np.nan
        try:
            parsed = float(text)
            return parsed if np.isfinite(parsed) else np.nan
        except ValueError:
            return np.nan

    return np.nan


def load_decay_table_multi_column(
    file_path: Path,
    *,
    time_to_ms_scale: float = 1.0,
    signal_name_prefix: str = "col",
) -> Tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load a decay workbook where column A is time and B..N are signals.

    Parameters
    ----------
    file_path:
        Input workbook path.
    time_to_ms_scale:
        Multiplicative scale for converting source time unit to milliseconds.
        For example, when time is in seconds, use `1000.0`.
    signal_name_prefix:
        Prefix used to build standardized signal names like `col_2`.

    Returns
    -------
    time_ms:
        1D array of valid positive time samples after scaling.
    signal_matrix:
        2D array with one column per valid signal.
    signal_names:
        Standardized names aligned with `signal_matrix` columns.
    valid_excel_column_ids:
        Excel 1-based column ids for retained signal columns.
    """

    table = pd.read_excel(file_path, header=None, dtype=object)
    if table.empty:
        raise ValueError(f"Input workbook is empty: {file_path}")

    matrix = table.values
    n_row, n_col = matrix.shape

    parsed_time: list[float] = []
    parsed_signals: list[list[float]] = []

    for row_idx in range(n_row):
        ok, time_value = parse_time_cell(matrix[row_idx, 0])
        if not ok:
            continue

        row_signal = [cell_to_float(matrix[row_idx, col_idx]) for col_idx in range(1, n_col)]
        parsed_time.append(time_value)
        parsed_signals.append(row_signal)

    if not parsed_time:
        raise ValueError("No valid time rows were detected in the first column.")

    time_values = np.asarray(parsed_time, dtype=float)
    signal_matrix = np.asarray(parsed_signals, dtype=float)

    valid_time_mask = np.isfinite(time_values) & (time_values > 0)
    time_values = time_values[valid_time_mask]
    signal_matrix = signal_matrix[valid_time_mask, :]

    valid_signal_column_mask = np.any(np.isfinite(signal_matrix), axis=0)
    signal_matrix = signal_matrix[:, valid_signal_column_mask]
    valid_excel_column_ids = np.where(valid_signal_column_mask)[0] + 2

    if signal_matrix.size == 0 or signal_matrix.shape[1] == 0:
        raise ValueError("No valid signal columns were detected.")

    signal_names = [f"{signal_name_prefix}_{int(col_id)}" for col_id in valid_excel_column_ids]
    time_ms = time_values * float(time_to_ms_scale)

    return time_ms, signal_matrix, signal_names, valid_excel_column_ids


def sort_and_filter_signal(time_ms: np.ndarray, amplitude: np.ndarray, *, minimum_points: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Filter invalid rows and sort one signal by time in ascending order."""

    t_arr = np.asarray(time_ms, dtype=float).ravel()
    y_arr = np.asarray(amplitude, dtype=float).ravel()
    valid = np.isfinite(t_arr) & np.isfinite(y_arr) & (t_arr > 0)

    t_valid = t_arr[valid]
    y_valid = y_arr[valid]
    if t_valid.size < minimum_points:
        raise ValueError(f"Signal has fewer than {minimum_points} valid points.")

    order = np.argsort(t_valid)
    return t_valid[order], y_valid[order]


def trim_signal_from_global_peak(
    signal_name: str,
    time_ms: np.ndarray,
    amplitude: np.ndarray,
    *,
    min_points_after_trim: int = 10,
) -> TrimmedSignal:
    """Trim one signal so inversion starts at the global maximum.

    This behavior is useful for simulation datasets where a short initial rise
    precedes the physical decay section.
    """

    sorted_time, sorted_amp = sort_and_filter_signal(time_ms, amplitude, minimum_points=min_points_after_trim)
    peak_idx = int(np.argmax(sorted_amp))

    trimmed_time = sorted_time[peak_idx:]
    trimmed_amp = sorted_amp[peak_idx:]
    if trimmed_time.size < min_points_after_trim:
        raise ValueError(
            f"Signal '{signal_name}' has only {trimmed_time.size} points after trimming; "
            f"at least {min_points_after_trim} are required."
        )

    return TrimmedSignal(
        signal_name=signal_name,
        raw_time_ms=sorted_time,
        raw_amplitude=sorted_amp,
        trimmed_time_ms=trimmed_time,
        trimmed_amplitude=trimmed_amp,
        peak_index_in_sorted_raw=peak_idx,
    )


def export_sheet_map_to_excel(sheet_map: Dict[str, pd.DataFrame], output_path: Path) -> None:
    """Write a dictionary of DataFrames into an Excel workbook.

    Keys are converted to valid sheet names automatically.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for key, frame in sheet_map.items():
            frame.to_excel(writer, sheet_name=safe_sheet_name(key), index=False)


def read_sheet_map_from_excel(file_path: Path) -> Dict[str, pd.DataFrame]:
    """Read all sheets from a workbook as `{sheet_name: DataFrame}`."""

    workbook = pd.ExcelFile(file_path)
    return {sheet_name: pd.read_excel(file_path, sheet_name=sheet_name) for sheet_name in workbook.sheet_names}


def find_numeric_pair_columns(dataframe: pd.DataFrame, candidate_x: Iterable[str], candidate_y: Iterable[str]) -> Tuple[str, str]:
    """Infer x/y column names from a table using candidate names and fallbacks."""

    normalized = {re.sub(r"[^a-z0-9]+", "", str(col).lower()): str(col) for col in dataframe.columns}

    def pick(candidates: Iterable[str], fallback_idx: int) -> str:
        for item in candidates:
            key = re.sub(r"[^a-z0-9]+", "", str(item).lower())
            if key in normalized:
                return normalized[key]
        if fallback_idx < dataframe.shape[1]:
            return str(dataframe.columns[fallback_idx])
        raise ValueError(f"Cannot infer required column from {list(dataframe.columns)}")

    x_col = pick(candidate_x, 0)
    y_col = pick(candidate_y, 1)
    return x_col, y_col


def dataframe_columns_to_numeric_xy(dataframe: pd.DataFrame, x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convert two columns from a DataFrame to finite numeric NumPy arrays."""

    x = pd.to_numeric(dataframe[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(dataframe[y_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]
