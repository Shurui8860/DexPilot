from typing import List
import numpy as np
import numpy.typing as npt
import pinocchio as pin
from vision2dex.hand_detector.keypoint_transformation import *

class RobotWrapper:
    def __init__(self, urdf_path: str, use_collision=False, use_visual=False):
        # Create robot model and data
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.data: pin.Data = self.model.createData()
        self.urdf_path = urdf_path

        if use_visual or use_collision:
            raise NotImplementedError

        self.q0 = pin.neutral(self.model)
        if self.model.nv != self.model.nq:
            raise NotImplementedError(f"Can not handle robot with special joint.")

    # -------------------------------------------------------------------------- #
    # Robot property
    # -------------------------------------------------------------------------- #
    @property
    def joint_names(self) -> List[str]:
        return list(self.model.names)

    @property
    def dof_joint_names(self) -> List[str]:
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def dof(self) -> int:
        return self.model.nq

    @property
    def link_names(self) -> List[str]:
        link_names = []
        for i, frame in enumerate(self.model.frames):
            link_names.append(frame.name)
        return link_names

    @property
    def joint_limits(self):
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    # -------------------------------------------------------------------------- #
    # Query function
    # -------------------------------------------------------------------------- #
    def get_joint_index(self, name: str):
        return self.dof_joint_names.index(name)

    def get_link_index(self, name: str):
        if name not in self.link_names:
            raise ValueError(
                f"{name} is not a link name. Valid link names: \n{self.link_names}"
            )
        return self.model.getFrameId(name)

    # -------------------------------------------------------------------------- #
    # Kinematics function
    # -------------------------------------------------------------------------- #
    def compute_forward_kinematics(self, qpos: npt.NDArray):
        pin.forwardKinematics(self.model, self.data, qpos)

    def get_link_pose(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.homogeneous

    def get_link_pose_inv(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.inverse().homogeneous

    def compute_single_link_local_jacobian(self, qpos, link_id: int) -> npt.NDArray:
        J = pin.computeFrameJacobian(self.model, self.data, qpos, link_id)
        return J

    def reset(self) -> np.ndarray:
        """
        Reset the robot to its neutral pose:
        1. Re-create the Pinocchio Data (clearing any previous computations).
        2. Use the neutral configuration q0.
        3. Run forward kinematics so link poses match q0.
        Returns:
            A copy of the neutral joint configuration (shape: (nq,)).
        """
        # 1) Reinitialize Pinocchio Data
        self.data = self.model.createData()

        # 2) Get the neutral configuration
        neutral_q = pin.neutral(self.model)
        self.q0 = neutral_q

        # 3) Compute forward kinematics at neutral pose
        self.compute_forward_kinematics(self.q0)

        # 4) Return the neutral joint vector for external use
        return self.q0.copy()

    def get_velocity_limits(self) -> np.ndarray:
        """
        Return absolute velocity limits for each DOF (shape: (nq,)).
        If velocity limits are missing in the URDF (Pinocchio may fill with inf),
        we return +inf for those entries.
        """
        vlim = getattr(self.model, "velocityLimit", None)
        if vlim is None:
            return np.full(self.model.nq, np.inf, dtype=float)

        vlim = np.asarray(vlim, dtype=float)
        if vlim.shape[0] != self.model.nq:
            raise RuntimeError(
                f"velocityLimit has shape {vlim.shape} but nq={self.model.nq}. "
                "This robot likely has special joints (nq != nv), which this wrapper does not support."
            )
        # Ensure non-negative and finite where provided
        vlim = np.where(vlim < 0, np.inf, vlim)
        return vlim




