import enum
from pathlib import Path


class HandType(enum.Enum):
    right = enum.auto()
    left = enum.auto()


# Mapping from link/joint names to MediaPipe landmark indices
link_to_mediapipe = [
    ("r_p_link0",      0),  # WRIST

    # Thumb (landmarks 1–4)
    ("r_p_link1",      1),  # THUMB_CMC
    ("r_f_joint1_1",   2),  # THUMB_MCP
    ("r_f_joint1_2",   3),  # THUMB_IP
    ("r_f_link1_tip",  4),  # THUMB_TIP

    # Index finger (landmarks 5–8)
    ("r_f_joint2_1",   5),  # INDEX_MCP
    ("r_f_joint2_2",   6),  # INDEX_PIP
    ("r_f_joint2_3",   7),  # INDEX_DIP
    ("r_f_link2_tip",  8),  # INDEX_TIP

    # Middle finger (landmarks 9–12)
    ("r_f_joint3_1",   9),   # MIDDLE_MCP
    ("r_f_joint3_2",   10),  # MIDDLE_PIP
    ("r_f_joint3_3",   11),  # MIDDLE_DIP
    ("r_f_link3_tip",  12),  # MIDDLE_TIP

    # Ring finger (landmarks 13–16)
    ("r_f_joint4_1",   13),  # RING_MCP
    ("r_f_joint4_2",   14),  # RING_PIP
    ("r_f_joint4_3",   15),  # RING_DIP
    ("r_f_link4_tip",  16),  # RING_TIP

    # Little (pinky) finger (landmarks 17–20)
    ("r_f_joint5_1",   17),  # PINKY_MCP
    ("r_f_joint5_2",   18),  # PINKY_PIP
    ("r_f_joint5_3",   19),  # PINKY_DIP
    ("r_f_link5_tip",  20),  # PINKY_TIP
]

# Define the MediaPipe landmark indices for each finger tip (thumb to pinky)
tip_indices = [4, 8, 12, 16, 20]

# Look up the corresponding link names from link_to_mediapipe mapping
finger_tip_link_names = [link_to_mediapipe[i][0] for i in tip_indices]

# The wrist is at the root of the hand mapping (index 0)
wrist_link_name = link_to_mediapipe[0][0]