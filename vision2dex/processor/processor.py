from __future__ import annotations

import os
import yaml
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from vision2dex.dex_retargeting.robot_wrapper import RobotWrapper
from vision2dex.dex_retargeting.optimizer import DexPilotOptimizer
from vision2dex.dex_retargeting.seq_retarget import SeqRetargeting
from vision2dex.dex_retargeting.optimizer_utils import LPFilter
from vision2dex.processor.data_loader import DataLoader
from vision2dex.processor.processor_utils import project_root, resolve_path


class Processor:
    """
    Simple, readable, and minimal processor for DexPilot-style teleoperation.

    Call order:
      1) build_robot(cfg_path)
      2) prepare_links_and_joints()
      3) setup_optimizer(cfg_path)
    """
    def __init__(self) -> None:
        # Populated by methods
        self.cfg: Dict[str, Any] = {}

        # Project root = parent → parent → parent of this file
        self.project_root = project_root()

        self.robot: Optional[RobotWrapper] = None

        self.finger_tip_link_names: List[str] = []
        self.wrist_link_name: str = ""
        self.target_joint_names: List[str] = []

        self.scale: float = 2.0
        self.dt: float = 1.0 / 15.0
        self.fc_hz: float = 6.0
        self.lp_filter: Optional[LPFilter] = None

        self.dp_opt: Optional[DexPilotOptimizer] = None
        self.seq_retar: Optional[SeqRetargeting] = None

    def from_yaml(self, yml_path: Union[str, Path]):
        """
        Read a YAML config and split it into component dictionaries:
          - robot, filter, and optimizer
        Also sets:
          - self.data_root (absolute Path if provided)

        """
        # Resolve the yaml path (relative -> under project root)
        yml_path = resolve_path(Path(yml_path))
        cfg = yaml.safe_load(yml_path.read_text(encoding="utf-8")) or {}
        self.cfg = cfg

        # warm_start is a boolean in config.yaml
        self.warm_start: bool = cfg.get("warm_start", True)
        assert isinstance(self.warm_start, bool), "warm_start must be a boolean in config.yaml"

        # Extract sections as dicts (empty if missing)
        robot = dict(cfg.get("robot", {}) or {})
        flt = dict(cfg.get("filter", {}) or {})
        opt = dict(cfg.get("optimizer", {}) or {})

        # Build subcomponents in a clear order:
        self.build_robot(robot)
        self.build_filter(flt)
        self.build_optimizer(opt)
        return self

    # ---------------------------- steps -----------------------------
    def build_robot(self, robot: Dict[str, Any]):
        """
        Build a RobotWrapper and set key attributes from a 'robot' dict.

        Expected keys in 'robot':
          - urdf_path (str|Path, optional): URDF path (defaults to right-hand flyingbase)
          - finger_tip_link_names (list[str], required): exactly 5 fingertip link names
          - wrist_link_name (str, required): wrist link name
          - target_joint_names (list[str], optional): subset of DOFs, defaults to all DOFs
        """
        # URDF path (resolve under project root if relative)
        urdf = robot.get("urdf_path", "dexrobot_urdf/urdf/dexhand021_right_flyingbase.urdf")
        self.robot = RobotWrapper(urdf_path=str(resolve_path(Path(urdf))))

        # Required names
        tip_names = robot.get("finger_tip_link_names")
        wrist_name = robot.get("wrist_link_name")

        # Validate
        if not (isinstance(tip_names, (list, tuple)) and len(tip_names) == 5 and all(tip_names)):
            raise ValueError("finger_tip_link_names must be a list of 5 non-empty strings.")
        if not wrist_name:
            raise ValueError("wrist_link_name must be a non-empty string.")

        # Obtain the full set of DOF joint names from the built robot ---
        target_joint_names = list(getattr(self.robot, "dof_joint_names", []))

        # Persist normalized results on 'self' for downstream components ---
        self.finger_tip_link_names: List[str] = list(tip_names)
        self.wrist_link_name: str = str(wrist_name)
        self.target_joint_names: List[str] = list(target_joint_names)

        return self.robot, self.finger_tip_link_names, self.wrist_link_name, self.target_joint_names

    def build_filter(self, flt: Union[str, Path]):
        """
        Initialize a basic low-pass filter from 'filter' config.
        Keys:
          - dt (float): timestep in seconds. Default 1/30.
          - fc_hz (float): cutoff frequency in Hz. Default 6.0.
        """
        # Read params with tiny, sane defaults
        self.dt = float(flt.get("dt", 1.0 / 15.0))
        self.fc_hz = float(flt.get("fc_hz", 6.0))

        # Precompute alpha and build LP filter
        alpha = LPFilter.alpha_from_cutoff(self.dt, fc_hz=self.fc_hz)
        self.lp_filter = LPFilter(alpha)

        return self.lp_filter

    def build_optimizer(self, opt: Dict[str, Any]) -> Tuple[DexPilotOptimizer, SeqRetargeting]:
        """
        Build:
          - self.dp_opt (DexPilotOptimizer)
          - self.seq_retar (SeqRetargeting)

        YAML (optimizer) keys:
          - huber_delta, norm_delta, project_dist, escape_dist, eta1, eta2, scale
          - has_joint_limit is always treated as True.
        """
        if self.robot is None or not (self.finger_tip_link_names and self.wrist_link_name and self.target_joint_names):
            raise RuntimeError("Call build_robot() and prepare_links_and_joints() before build_optimizer().")

        # Read params with simple defaults
        scale = float(opt.get("scale", getattr(self, "scale", 2.0)))
        huber_delta = float(opt.get("huber_delta", 0.03))
        norm_delta = float(opt.get("norm_delta", 0.004))
        project_dist = float(opt.get("project_dist", 0.03))
        escape_dist = float(opt.get("escape_dist", 0.05))
        eta1 = float(opt.get("eta1", 0.0001))
        eta2 = float(opt.get("eta2", 0.03))

        self.scale = scale  # persist chosen scale

        # DexPilot optimizer
        self.dp_opt = DexPilotOptimizer(
            robot=self.robot,
            target_joint_names=self.target_joint_names,
            finger_tip_link_names=self.finger_tip_link_names,
            wrist_link_name=self.wrist_link_name,
            scaling=scale,
            huber_delta=huber_delta,
            norm_delta=norm_delta,
            project_dist=project_dist,
            escape_dist=escape_dist,
            eta1=eta1,
            eta2=eta2,
        )

        # Always enforce joint limits; pass LP filter by default
        self.seq_retar = SeqRetargeting(
            optimizer=self.dp_opt,
            has_joint_limits=True,
            lp_filter=self.lp_filter,
        )

        return self.dp_opt, self.seq_retar

    def print_summary(self) -> None:
        """Print present (non-empty) attributes of this Processor."""
        print("[Processor]")

        # Robot & names
        if getattr(self, "robot", None):
            print(f"  robot:               {type(self.robot).__name__}")
        if self.finger_tip_link_names:
            print(f"  finger_tip_links:    {self.finger_tip_link_names}")
        if self.wrist_link_name:
            print(f"  wrist_link_name:     {self.wrist_link_name}")
        if self.target_joint_names:
            tj = self.target_joint_names
            head = ", ".join(tj[:5]) + (" ..." if len(tj) > 5 else "")
            print(f"  target_joints ({len(tj)}): {head}")

        # Filter/optimizer
        if getattr(self, "lp_filter", None):
            print(f"  filter:              dt={self.dt}, fc_hz={self.fc_hz}")
        if getattr(self, "dp_opt", None):
            print(f"  optimizer:           {type(self.dp_opt).__name__} (scale={self.scale})")
        if getattr(self, "seq_retar", None):
            print(f"  seq_retarget:        {type(self.seq_retar).__name__}")


    def compute_trajectory(self, dataset) -> np.ndarray:
        """
        Compute a joint-space trajectory with SeqRetargeting.
        Returns:
            (T, dof) float32 array.
        """
        # Basic checks
        if self.seq_retar is None or self.robot is None:
            raise RuntimeError("Build robot/optimizer before compute_trajectory().")

        # (T, 21, 3) human keypoints in MANO order
        hand_trajectory = dataset.hand_joints3d_mano.astype(np.float32, copy=False)

        traj = np.empty((dataset.T, len(self.target_joint_names)), dtype=np.float32)

        # Human keypoint indices defining origin and task links for vector computation
        orig_idxs, task_idxs = self.dp_opt.target_link_human_indices

        # Fixed part (if provided)
        idx_fixed = self.dp_opt.idx_pin2fixed
        fixed_qpos = self.robot.q0[idx_fixed].astype(np.float32) if idx_fixed is not None else None

        # Main retargeting loop: process each frame in the hand trajectory
        for t, hand_pos in enumerate(hand_trajectory):
            # Set bounds for this frame (position / velocity windows)
            self.seq_retar.set_bounds(dt=self.dt)

            # Target vectors: displacement from origin to task points
            target_vec = hand_pos[orig_idxs] - hand_pos[task_idxs]

            if self.warm_start:
                # Use wrist position and the per-frame compact axis-angle (3,)
                wrist_pos = hand_pos[0]
                wrist_orientation = dataset.hand_pose3[t]
                self.seq_retar.warm_start(
                    wrist_pos=wrist_pos,
                    wrist_orientation=wrist_orientation,
                    global_rot=np.eye(3, dtype=np.float32),
                )

            # One optimization step to robot joint configuration
            qpos = self.seq_retar.retarget(ref_value=target_vec, fixed_qpos=fixed_qpos)
            traj[t] = qpos.astype(np.float32, copy=False)

        self.seq_retar.verbose()
        self.seq_retar.reset()

        return traj

    def write_pickles(self, traj: np.ndarray, out_dir: Union[str, Path]):
        """
        Save T pickle files, one per frame:
          {<joint_name>: <position_float>, ...}
        """
        # Basic checks
        if traj.ndim != 2:
            raise ValueError("traj must be shape (T, dof)")
        T, dof = traj.shape

        names: List[str] = list(self.target_joint_names)

        # Frame names: prefer dataset frames; fallback to 0000, 0001, ...
        frames = [f"{i:04d}" for i in range(T)]

        out_dir = resolve_path(out_dir, exists_OK=True)

        # Write one pickle per frame
        for t in range(T):
            data = {jn: float(traj[t, i]) for i, jn in enumerate(names)}
            with (out_dir / f"{frames[t]}.pkl").open("wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def process_subject(self, subject_dir: Union[str, Path],
                        order: str = "ho3d", out_subdir: str = "dex_meta"):
        """
        For a subject folder like:
          right/20200709-subject-01/
            ├── 20200709_141754/
            ├── 20200709_141841/
            └── ...
        For each sequence folder:
          1) Load it with DataLoader
          2) Compute trajectory via self.compute_trajectory(...)
          3) Write per-frame pickles into <seq>/dex_meta/
        """
        subject_dir = resolve_path(subject_dir)

        for seq_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            if seq_dir.name == out_subdir:
                continue  # skip any existing output dirs

            # Load sequence
            dataset = DataLoader(seq_dir / "meta", order=order)

            # Retarget to robot joint space
            traj = self.compute_trajectory(dataset)

            # Write one pickle per frame into <seq>/dex_meta/
            out_dir = seq_dir / out_subdir
            self.write_pickles(traj, out_dir)

            print(f"[OK] {seq_dir.name} → {out_dir}")

    def process_all_subjects(self, folder: Union[str, Path], out_subdir: str = "dex_meta", order: str = "ho3d"):
        """
        Process every subject under `folder`.
        Accepts:
          - a single side folder ('right/')
        For each subject, calls `process_subject(subject_dir, out_subdir=...)`.
        """
        root = resolve_path(folder)

        # Collect subject dirs to process
        subjects = sorted(p for p in root.iterdir() if p.is_dir())

        # Run
        count = 0
        for subj in subjects:
            self.process_subject(subj, out_subdir=out_subdir, order=order)
            count += 1
        print(f"[DONE] processed {count} subject(s) into '{out_subdir}'")


if name == "__main__":
    proc = Processor().from_yaml("vision2dex/processor/config.yaml")
    proc.print_summary()
    proc.process_all_subjects("right", order="ho3d", out_subdir="dex_meta")
