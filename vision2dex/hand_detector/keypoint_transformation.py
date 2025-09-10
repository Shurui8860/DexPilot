import numpy as np
import pinocchio as pin

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """
    Compute the 3D coordinate frame (orientation only) from detected 3d key points
    :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
    :return: the coordinate frame of wrist in MANO convention
    """
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]

    # Compute vector from palm to the first joint of middle finger
    x_vector = points[0] - points[2]

    # Normal fitting with SVD
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)

    normal = v[2, :]

    # Gram–Schmidt Orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # We assume that the vector from pinky to index is similar the z axis in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame

def get_wrist_pose(
    human_xyz: np.ndarray,
    side: str = "right"
) -> np.ndarray:
    """
    Generate the 6 joint values for:
      dummy_x_translation_joint,
      dummy_y_translation_joint,
      dummy_z_translation_joint,
      dummy_x_rotation_joint,
      dummy_y_rotation_joint,
      dummy_z_rotation_joint

    Returns:
      np.ndarray of shape (6,)
        [tx, ty, tz, rx, ry, rz] where
          tx,ty,tz = wrist translation (m),
          [rx,ry,rz] = rotation vector (axis * angle in radians).
    """
    # 1) Estimate local MANO frame at the wrist
    mano_frame = estimate_frame_from_hand_points(human_xyz)  # 3×3

    # 2) Align operator frame → MANO
    if side.lower().startswith("r"):
        R = OPERATOR2MANO_RIGHT @ mano_frame
    else:
        R = OPERATOR2MANO_LEFT  @ mano_frame

    # 3) Convert to quaternion
    quat = pin.Quaternion(R)
    x, y, z, w = quat.coeffs()  # Eigen‐order: [x,y,z,w]

    # 4) Translation = wrist point
    trans = human_xyz[0].astype(float)  # (3,)

    # 5) Rotation = axis-angle vector
    vec = np.array([x, y, z], dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        rot_vec = np.zeros(3, dtype=float)
    else:
        # angle = 2*acos(w), axis = vec/norm
        angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
        axis  = vec / norm
        rot_vec = axis * angle  # (3,)

    # 6) Pack into one 6-vector
    return np.concatenate([trans, rot_vec])  # shape (6,)

