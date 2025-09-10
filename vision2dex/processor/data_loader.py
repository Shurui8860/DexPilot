from __future__ import annotations

import re
import os
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from processor_utils import ho3d_to_mano
from vision2dex.processor.processor_utils import resolve_path

_NUM_RX = re.compile(r"(\d+)")


def _numeric_sort_key(p: Path) -> Tuple[int, str]:
    """
    Sort by first integer in the filename stem (e.g., '0007' -> 7), then fallback by name.
    Ensures 0000.pkl, 0001.pkl, ... order.
    """
    m = _NUM_RX.search(p.stem)
    return (int(m.group(1)) if m else 10**9, p.name)


class DataLoader:
    """
    Minimal OOP loader that accepts EITHER:
      - a single per-frame .pkl file, OR
      - a directory containing many per-frame .pkl files (recursively)

    Exposes sequence-shaped arrays:
      - hand_joints3d        : (T, 21, 3) in the file's original order
      - hand_joints3d_mano   : (T, 21, 3) reindexed to MANO order
      - hand_trans           : (T, 3)     MANO root translation (m)
      - hand_pose3           : (T, 3)     wrist global axis-angle (radians)
      - frames               : List[str]  frame names like ['0000','0001',...]

    Note
    * `order` indicates the joint convention stored in the .pkl files: "ho3d" or "mano".
      If "ho3d", we remap to MANO via `ho3d_to_mano`.
    * All arrays are float32.
    * If you pass a single .pkl, outputs still have a leading T=1 dimension
      for a consistent downstream interface.
    """

    def __init__(self, path: Union[str, Path], order: str = "ho3d") -> None:

        self._order = str(order).lower().strip()
        self.root = resolve_path(Path(path))

        if self.root.is_file():
            if self.root.suffix != ".pkl":
                raise ValueError(f"Expected a .pkl file, got: {self.root}")
            self._pkl_paths = [self.root]
        else:
            # Search recursively; typical layout has files under .../meta/0000.pkl etc.
            self._pkl_paths = sorted(self.root.rglob("*.pkl"), key=_numeric_sort_key)
            if not self._pkl_paths:
                raise FileNotFoundError(f"No .pkl files found under: {self.root}")

        # Load and stack
        self._frames: List[str] = []
        joints_list: List[np.ndarray] = []
        trans_list: List[np.ndarray] = []
        pose3_list: List[np.ndarray] = []

        for p in self._pkl_paths:
            fr_name, j3d, trans, pose3 = self._read_one(p)
            self._frames.append(fr_name)
            joints_list.append(j3d.astype(np.float32, copy=False))
            trans_list.append(trans.astype(np.float32, copy=False))
            pose3_list.append(pose3.astype(np.float32, copy=False))

        # (T, 21, 3), (T, 3), (T, 3)
        self._hand_joints3d = np.stack(joints_list, axis=0)
        self._hand_trans = np.stack(trans_list, axis=0)
        self._hand_pose3 = np.stack(pose3_list, axis=0)

        # Lazily computed cache for MANO order
        self._hand_joints3d_mano: np.ndarray | None = None

    # ---------- Internals ----------
    def _read_one(self, pkl_path: Path) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load one per-frame pickle and return:
          - frame_name: str               (e.g., '0000' from '.../0000.pkl')
          - handJoints3D: (21, 3) float32
          - handTrans:    (3,)    float32
          - wrist_pose3:  (3,)    float32  # first 3 of handPose (global wrist axis-angle)
        """
        # Read the pickle
        with pkl_path.open("rb") as f:
            data = pickle.load(f)

        # Ensure required keys exist
        for k in ("handJoints3D", "handTrans", "handPose"):
            if k not in data:
                raise KeyError(f"Missing '{k}' in {pkl_path}")

        # Convert to numpy (accepts list/tuple), enforce float32
        j3d = np.asarray(data["handJoints3D"], dtype=np.float32)
        trans = np.asarray(data["handTrans"], dtype=np.float32).reshape(-1)
        pose = np.asarray(data["handPose"], dtype=np.float32).ravel()

        # Basic shape checks (fail fast with clear messages)
        if j3d.shape != (21, 3):
            raise ValueError(f"'handJoints3D' expected (21,3), got {j3d.shape} in {pkl_path}")
        if trans.size != 3:
            raise ValueError(f"'handTrans' expected 3 values, got shape {trans.shape} in {pkl_path}")
        if pose.size != 48:
            raise ValueError(f"'handPose' must have 48 values, got {pose.shape} in {pkl_path}")

        frame_name = pkl_path.stem  # '0000' from '.../0000.pkl'
        return frame_name, j3d, trans, pose[:3]

    # ---------- Public API ----------
    @property
    def frames(self) -> List[str]:
        """List of frame names like ['0000','0001', ...]."""
        return self._frames

    @property
    def T(self) -> int:
        """Number of frames."""
        return len(self._frames)

    @property
    def order(self) -> str:
        """Joint order convention of the source files (e.g. 'ho3d' or 'mano')."""
        return self._order

    @property
    def hand_joints3d(self) -> np.ndarray:
        """(T, 21, 3) joints in the file's original order."""
        return self._hand_joints3d

    @property
    def hand_joints3d_mano(self) -> np.ndarray:
        """
        (T, 21, 3) joints reindexed to MANO order.
        If files already store MANO, this is identical to `hand_joints3d`.
        """
        if self._hand_joints3d_mano is not None:
            return self._hand_joints3d_mano

        if "mano" in self._order:
            self._hand_joints3d_mano = self._hand_joints3d.deepcopy()
        elif "ho3d" in self._order:
            # Map each frame; ho3d_to_mano expects (21,3)
            self._hand_joints3d_mano = ho3d_to_mano(self._hand_joints3d)
        else:
            raise NotImplementedError(
                f"Unsupported joint order for MANO mapping: '{self._order}'"
            )
        return self._hand_joints3d_mano

    @property
    def hand_trans(self) -> np.ndarray:
        """(T, 3) MANO root translations (meters)."""
        return self._hand_trans

    @property
    def hand_pose3(self) -> np.ndarray:
        """(T, 3) wrist global axis-angle (radians)."""
        return self._hand_pose3

    def __len__(self) -> int:
        return self.T

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Random-access one frame as a small dict (useful for visualization):
          {'frame': str, 'hand_joints3d': (21,3), 'hand_trans': (3,), 'hand_pose3': (3,)}
        """
        return {
            "frame": self._frames[idx],
            "hand_joints3d": self._hand_joints3d[idx],
            "hand_trans": self._hand_trans[idx],
            "hand_pose3": self._hand_pose3[idx],
        }

    def as_dict(self) -> Dict[str, np.ndarray]:
        """
        Return all stacked arrays (sequence-first), handy for downstream consumption.
        Keys: 'frames', 'order', 'hand_joints3d', 'hand_joints3d_mano', 'hand_trans', 'hand_pose3'
        """
        return {
            "frames": np.array(self._frames),
            "order": np.array([self._order], dtype=object),
            "hand_joints3d": self.hand_joints3d,
            "hand_joints3d_mano21": self.hand_joints3d_mano,
            "hand_trans": self.hand_trans,
            "hand_pose3": self.hand_pose3,
        }

    def __str__(self) -> str:
        """
        Pretty, robust summary for `print(loader)`.
        - Avoids AttributeError by using getattr with defaults.
        - Shows basic metadata, frame range, and key array shapes.
        """
        lines = ["[DataLoader]"]

        # Basic metadata (only print if present)
        order = getattr(self, "_order", None)
        if order is not None:
            lines.append(f"  order:              {order}")

        seq_dir = getattr(self, "seq_dir", None)
        if seq_dir is not None:
            lines.append(f"  seq_dir:            {seq_dir}")

        # Frame count and quick range preview
        T = getattr(self, "T", None)
        frames = getattr(self, "frames", [])
        lines.append(f"  frames (T):         {T if T is not None else 'N/A'}")
        if frames:
            lines.append(f"  first/last frame:   {frames[0]} … {frames[-1]}")

        # Helper: append shape line if array-like attribute exists
        def _add_shape(attr: str, label: str) -> None:
            arr = getattr(self, attr, None)
            if arr is not None:
                lines.append(f"  {label}: {getattr(arr, 'shape', None)}")

        # Key tensors
        _add_shape("hand_joints3d", "hand_joints3d")
        _add_shape("hand_joints3d_mano", "hand_joints3d_mano")
        _add_shape("hand_trans", "hand_trans")
        _add_shape("hand_pose3", "hand_pose3")

        return "\n".join(lines)


def main() -> None:
    """Resolve the sequence path, load it, and print the default summary."""
    parser = argparse.ArgumentParser(description="Quick inspect of a DexYCB/HO3D sequence.")
    parser.add_argument(
        "seq_path",
        nargs="?",
        default="right/20200709-subject-01/20200709_141841",
        help="Path to the sequence directory (relative to project root or absolute).",
    )
    parser.add_argument(
        "--order",
        default="ho3d",
        choices=["ho3d", "mano"],
        help="Joint order for the loader.",
    )
    args = parser.parse_args()

    # Resolve sequence path: keep absolute; map relative under project root
    seq_dir = resolve_path(Path(args.seq_path))

    # Load and print (uses DataLoader.__str__)
    loader = DataLoader(seq_dir, order=args.order)
    print(loader)

if __name__ == "__main__":
    main()