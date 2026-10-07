"""
Persona Engine — Dataset File Loader
Supports loading, streaming, and saving JSON and JSONL datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Union


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Loads all records from a JSON Lines (.jsonl) file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                records.append(json.loads(line_str))
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Invalid JSON on line {line_num} in {path.name}: {err}"
                ) from err
    return records


def iter_jsonl(file_path: Union[str, Path]) -> Generator[Dict[str, Any], None, None]:
    """Memory-efficient streaming generator for reading large JSONL files."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                yield json.loads(line_str)
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Invalid JSON on line {line_num} in {path.name}: {err}"
                ) from err


def save_jsonl(
    data: List[Dict[str, Any]],
    file_path: Union[str, Path],
    ensure_ascii: bool = False,
) -> Path:
    """Saves a list of dictionaries to a JSON Lines (.jsonl) file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")
    return path


def load_json(file_path: Union[str, Path]) -> Any:
    """Loads a standard JSON file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    data: Any,
    file_path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False,
) -> Path:
    """Saves data to a standard JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    return path
