"""Export helpers for simulation suite artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write a UTF-8 JSON file and return its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_csv_rows(path: Path, rows: list[dict], columns: list[str]) -> Path:
    """Write row dictionaries to CSV without requiring pandas."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def try_write_excel_workbook(path: Path, sheets: Dict[str, tuple[list[dict], list[str]]]) -> Path | None:
    """Write an Excel workbook when pandas/openpyxl are installed."""

    try:
        import pandas as pd
    except ImportError:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        for sheet_name, (rows, columns) in sheets.items():
            frame = pd.DataFrame(rows, columns=columns)
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return path
