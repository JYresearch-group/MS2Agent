import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def read_json_array(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
            tmp_path = Path(f.name)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            tmp_path = Path(f.name)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def sanitize_short_text(value: Any) -> str:
    text = normalize_space(value)
    return text if text else "unknown"


def safe_int(value: Any) -> Optional[int]:
    if value in (None, "", "null", "None", "N/A"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> Optional[float]:
    if value in (None, "", "null", "None", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_present(record: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return default


def ensure_string_list(value: Any, max_items: int = 8) -> List[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    result: List[str] = []
    seen = set()
    for item in items:
        text = sanitize_short_text(item)
        key = text.lower()
        if text != "unknown" and key not in seen:
            result.append(text)
            seen.add(key)
    return result[:max_items]


def ensure_float_list(value: Any, max_items: int = 8, ndigits: int = 6) -> List[float]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    result: List[float] = []
    seen = set()
    for item in items:
        number = safe_float(item)
        if number is None:
            continue
        rounded = round(number, ndigits)
        if rounded not in seen:
            result.append(rounded)
            seen.add(rounded)
    return result[:max_items]


def format_optional_number(value: Optional[float], ndigits: int = 6) -> str:
    if value is None:
        return "unknown"
    return f"{value:.{ndigits}f}"
