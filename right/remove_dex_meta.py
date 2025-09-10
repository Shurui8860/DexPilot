from pathlib import Path
from typing import Union
import shutil

def remove_all_dex_meta(root: Union[str, Path]) -> int:
    """
    Recursively remove every folder named 'dex_meta' under `root`.

    Returns:
        Number of folders removed.
    """
    root = Path(root).expanduser().resolve()
    removed = 0

    for p in root.rglob("dex_meta"):
        if p.is_dir():
            try:
                shutil.rmtree(p)
                removed += 1
                print(f"[OK] removed: {p}")
            except Exception as e:
                print(f"[WARN] failed to remove {p}: {e}")

    print(f"[remove_all_dex_meta] removed {removed} folder(s) under {root}")
    return removed

# e.g., clean everything under the subject root
remove_all_dex_meta("right/20200709-subject-01")
