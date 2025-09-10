from __future__ import annotations
import numpy as np
from typing import Dict, List, Sequence, Tuple, Iterable
import torch
import os
from pathlib import Path

def project_root() -> Path:
    """
    Use env var PROJECT_ROOT if set; otherwise parent→parent→parent of this file.
    """
    env = os.getenv("PROJECT_ROOT")
    return Path(env).expanduser().resolve() if env else Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, exists_OK=False) -> Path:
    """
    Resolve a given path into an absolute Path object.

    - If `path` is absolute, it is returned as-is (after expanding ~).
    - If `path` is relative, it is resolved under the project root.
    - Raises FileNotFoundError if the final resolved path does not exist.
    """

    # Convert input to Path object and expand ~ (user home directory)
    path = Path(path).expanduser()

    # If relative path, prepend project root and resolve to absolute
    if not path.is_absolute():
        path = (project_root() / path).resolve()

    if exists_OK:
        path = path.expanduser()
        path.mkdir(parents=True, exist_ok=True)

    # Ensure the resolved path actually exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return path


class JointConvention:
    """A simple joint indexing convention: finger -> ordered joint indices."""
    def __init__(self, name: str, layout: Dict[str, Sequence[int]]):
        self.name = name
        self.layout = {k: list(v) for k, v in layout.items()}
        # Precompute semantic <-> index maps
        self._idx_to_sem: Dict[int, Tuple[str, int]] = {
            idx: (finger, k) for finger, ids in self.layout.items() for k, idx in enumerate(ids)
        }
        self._sem_to_idx: Dict[Tuple[str, int], int] = {
            (finger, k): idx for finger, ids in self.layout.items() for k, idx in enumerate(ids)
        }
        self.size = max(self._idx_to_sem) + 1

    def idx_to_sem(self) -> Dict[int, Tuple[str, int]]:
        return self._idx_to_sem

    def sem_to_idx(self) -> Dict[Tuple[str, int], int]:
        return self._sem_to_idx

    @property
    def get_name(self):
        return self.name

    @property
    def get_layout(self):
        return self.layout


class JointReindexer:
    """Reindex joints from `src` convention to `dst` via a permutation."""
    def __init__(self, src: JointConvention, dst: JointConvention):
        self.src = src
        self.dst = dst
        N = dst.size
        self.perm = np.fromiter(
            (src.sem_to_idx()[dst.idx_to_sem()[i]] for i in range(N)),
            dtype=int, count=N
        )

    def apply(self, joints: np.ndarray) -> np.ndarray:
        """Reorder joints: joints shape (..., N, D)."""
        if joints.shape[-2] != self.perm.size:
            raise ValueError(f"Expected joints.shape[-2]=={self.perm.size}, got {joints.shape[-2]}")
        return joints[..., self.perm, :]

    def inverse(self) -> "JointReindexer":
        """Return the inverse mapper (dst -> src)."""
        inv = JointReindexer.__new__(JointReindexer)
        inv.src, inv.dst = self.dst, self.src
        inv.perm = np.argsort(self.perm)
        return inv

    def __repr__(self):
        return f"JointReindexer({self.src.name} -> {self.dst.name}, N={self.perm.size})"


# ---- Conventions ----
MANO21 = JointConvention(
    "MANO21/OpenPose",
    {
        "wrist":  [0],
        "thumb":  [1, 2, 3, 4],
        "index":  [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring":   [13, 14, 15, 16],
        "pinky":  [17, 18, 19, 20],
    },
)

HO3D = JointConvention(
    "HO3D",
    {
        "wrist":  [0],
        "index":  [1, 2, 3, 17],
        "middle": [4, 5, 6, 18],
        "ring":   [10, 11, 12, 19],
        "pinky":  [7, 8, 9, 20],
        "thumb":  [13, 14, 15, 16],
    },
)

# ---- Ready-to-use mappers ----
MANO_TO_HO3D = JointReindexer(MANO21, HO3D)
HO3D_TO_MANO = MANO_TO_HO3D.inverse()

# Convenience wrappers
def mano_to_ho3d(x: np.ndarray) -> np.ndarray:
    return MANO_TO_HO3D.apply(x)

def ho3d_to_mano(x: np.ndarray) -> np.ndarray:
    return HO3D_TO_MANO.apply(x)

