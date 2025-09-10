import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional
import cv2
import numpy as np
import tyro
from loguru import logger
import os, sys
#
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Header
# from sensor_msgs.msg import PointCloud2, PointField
# import struct

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# retargeting algorithm
from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    ROBOT_NAME_MAP,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from vision2dex.hand_detector.image_detector import SingleHandDetector

# realsense
import pyrealsense2 as rs

# simulator
import mujoco, mujoco.viewer


def create_point_cloud(points):
    """Wrap a list of 3D points into a PointCloud2 message.

    Example publisher usage:

    self.publisher_ = self.create_publisher(PointCloud2, 'point_cloud', 10)
    def timer_callback(self):
        points = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        msg = create_point_cloud(points)
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing PointCloud2 with %d points' % len(points))

    Example subscriber usage:

    self.subscription = self.create_subscription(
        PointCloud2,
        'point_cloud',
        self.listener_callback,
        10)
    def listener_callback(self, msg):
        self.get_logger().info('Received PointCloud2 data')
        points = []
        for i in range(msg.width):
            offset = msg.point_step * i
            x, y, z = struct.unpack_from('fff', msg.data, offset)
            points.append([x, y, z])
        self.get_logger().info('Received %d points' % len(points))
    """
    point_cloud = PointCloud2()
    point_cloud.header.stamp = rclpy.clock.Clock().now().to_msg()
    point_cloud.header.frame_id = "map"
    point_cloud.height = 1
    point_cloud.width = len(points)
    point_cloud.is_dense = False
    point_cloud.is_bigendian = False
    point_cloud.point_step = 12  # 3 * 4 bytes (float32)
    point_cloud.row_step = point_cloud.point_step * point_cloud.width

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    point_cloud.fields = fields

    points_flat = np.array(points, dtype=np.float32).flatten()
    point_cloud.data = struct.pack("f" * len(points_flat), *points_flat)

    return point_cloud


class RetargetAlgorithm(Node):
    def __init__(self, robot_dir: str, config_path: str, show_skeleton: bool = True):
        super().__init__("retarget_algorithm")
        self.config_path = config_path
        self.show_skeleton = show_skeleton
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        self.get_logger().info(f"Start retargeting with config {config_path}")
        self.config = RetargetingConfig.load_from_file(config_path)
        self.retargeting = self.config.build()

        # Create publisher for joint states
        self.joint_state_publisher = self.create_publisher(
            JointState, "joint_states", 10
        )
        self.point_cloud_publisher = self.create_publisher(
            PointCloud2, "mp_hand_coord_right", 10
        )

        # Initialize detector
        hand_type = "Right" if "right" in self.config_path.lower() else "Left"
        self.detector = SingleHandDetector(hand_type=hand_type, selfie=False)

        # realsense config
        self.realsense_config = rs.config()
        self.realsense_config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30
        )
        self.realsense_pipeline = rs.pipeline()
        self.realsense_pipeline.start(self.realsense_config)

        try:
            while True:
                frames = self.realsense_pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                rgb = np.asanyarray(color_frame.get_data())
                self.update(rgb)
        except Exception as e:
            logger.error(e)
            logger.error("stop realsense")
            self.realsense_pipeline.stop()

    def update(self, rgb):
        if rgb is None:
            logger.warning("rgb is None")
            return

        _, joint_pos, _, _, processed_image = self.detector.detect(
            rgb, return_processed_image=self.show_skeleton
        )

        # Visualization
        if self.show_skeleton:
            if processed_image is not None:
                cv2.imshow("skeleton", processed_image)
            else:
                cv2.imshow("skeleton", rgb)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
        # Joint position processing
        if joint_pos is not None:
            retargeting_type = self.retargeting.optimizer.retargeting_type
            indices = self.retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

            logger.warning(f"ref_value={ref_value}")
            retargeted_joint_positions = self.retargeting.retarget(ref_value)
            self.publish_joint_states(retargeted_joint_positions)

    def publish_joint_states(self, joint_positions):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = self.retargeting.joint_names
        joint_state_msg.position = joint_positions.tolist()
        self.joint_state_publisher.publish(joint_state_msg)
        # logger.warning(f"publish {joint_state_msg.name} {joint_state_msg.position}")

    def publish_point_cloud(self, points: np.ndarray):
        point_cloud_msg = create_point_cloud(points)
        self.point_cloud_publisher.publish(point_cloud_msg)

    def __del__(self):
        if hasattr(self, "realsense_manager"):
            self.realsense_manager.stop()


def get_default_config_path(
    robot_name: RobotName, retargeting_type: RetargetingType, hand_type: HandType
) -> Path:
    config_path = Path(__file__).parent / "configs"
    if retargeting_type is RetargetingType.position:
        config_path = config_path / "offline"
    else:
        config_path = config_path / "teleop"

    robot_name_str = ROBOT_NAME_MAP[robot_name]
    hand_type_str = hand_type.name
    if retargeting_type == RetargetingType.dexpilot:
        config_name = f"{robot_name_str}_{hand_type_str}_dexpilot.yml"
    else:
        config_name = f"{robot_name_str}_{hand_type_str}.yml"
    return config_path / config_name


def main(
    robot_name: RobotName = RobotName.dex,
    retargeting_type: RetargetingType = RetargetingType.dexpilot,
    hand_type: HandType = HandType.right,
    use_mujoco: bool = True,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        use_mujoco: If True, the simulation will be rendered using Mujoco. Otherwise, the simulation will be rendered using Sapien.
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = Path(__file__).absolute().parent / "assets" / "hands"
    logger.warning(f"{config_path}, {robot_dir}")

    # class
    retarget_algo = RetargetAlgorithm(
        str(robot_dir), str(config_path), show_skeleton=True
    )

    time.sleep(5)
    print("done")


if __name__ == "__main__":
    # rclpy.init()
    main()
