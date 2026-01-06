
from typing import Dict, Any, Iterable
import csv, os, json
import numpy as np

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_scan_long_csv(path: str, header: Iterable[str], rows_iter):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows_iter:
            writer.writerow(row)

def write_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_npy(path: str, arr: np.ndarray):
    import numpy as np
    np.save(path, arr)
