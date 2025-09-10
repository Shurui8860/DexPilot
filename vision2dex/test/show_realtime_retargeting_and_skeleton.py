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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# retargeting algorithm
from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    ROBOT_NAME_MAP,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from hand_detector.single_hand_detector_from_image import SingleHandDetector

# realsense
import pyrealsense2 as rs

# simulator
import mujoco, mujoco.viewer


class RealsenseManager:
    def __init__(self, show_raw: bool = True):
        self.show_raw = show_raw
        # config
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    def get_frame(self, queue: multiprocessing.Queue):
        # Start streaming from the default device
        self.pipeline = rs.pipeline()
        self.pipeline.start(self.config)

        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                # Convert images to numpy arrays
                raw_image = np.asanyarray(color_frame.get_data())
                queue.put(raw_image)

                time.sleep(1 / 60.0)
                if self.show_raw:
                    cv2.imshow("demo", raw_image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except:
            print("Error in get_frame")
        finally:
            # Stop streaming
            self.pipeline.stop()
            cv2.destroyAllWindows()


class MujocoSimulator:
    def __init__(self, urdf_filepath: str, retargeting_joint_names: list):
        self.urdf_filepath = urdf_filepath
        self.retargeting_joint_names = retargeting_joint_names

    def setup_model(self):
        model = mujoco.MjModel.from_xml_path(self.urdf_filepath)
        model.opt.timestep = 0.001
        data = mujoco.MjData(model)
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat = np.array([-0.00756184, -0.10470366, 0.54123664])
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (
            0.10568949644609987,
            -90.4502489701014,
            -62.54961832061071,
        )
        self.model, self.data, self.viewer = model, data, viewer

    def render(self, qpos: multiprocessing.Array):
        self.setup_model()
        while True:
            qpos_np = np.frombuffer(qpos.get_obj())

            size = len(qpos_np)
            # self.data.qpos[-size:] = qpos_np
            self.data.ctrl[:] = qpos_np
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            # print(f"look_at={self.viewer.cam.lookat}, distance={self.viewer.cam.distance}, azimuth={self.viewer.cam.azimuth}, elevation={self.viewer.cam.elevation}")


class RetargetAlgorithm:
    def __init__(self, robot_dir: str, config_path: str, show_skeleton: bool = True):
        self.config_path = config_path
        self.show_skeleton = show_skeleton
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        logger.info(f"Start retargeting with config {config_path}")
        self.config = RetargetingConfig.load_from_file(config_path)
        self.retargeting = self.config.build()

    def update(self, queue: multiprocessing.Queue, qpos: np.ndarray):
        # detector
        hand_type = "Right" if "right" in self.config_path.lower() else "Left"
        self.detector = SingleHandDetector(hand_type=hand_type, selfie=False)
        while True:
            try:
                rgb = queue.get(timeout=5)
            except Empty:
                logger.error(
                    f"Fail to fetch image from camera in 5 secs. Please check your web camera device."
                )
                return
            _, joint_pos, _, _, processed_image = self.detector.detect(
                rgb, return_processed_image=self.show_skeleton
            )
            # visualization
            if self.show_skeleton:
                if processed_image is not None:
                    cv2.imshow("skeleton", processed_image)
                else:
                    cv2.imshow("skeleton", rgb)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            # joint pos
            if joint_pos is None:
                pass
                # logger.warning(f"{hand_type} hand is not detected.")
            else:
                retargeting_type = self.retargeting.optimizer.retargeting_type
                indices = self.retargeting.optimizer.target_link_human_indices
                if retargeting_type == "POSITION":
                    indices = indices
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = (
                        joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                    )
                np.copyto(
                    np.frombuffer(qpos.get_obj()), self.retargeting.retarget(ref_value)
                )
            # finally:
            #     cv2.destroyAllWindows()


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
    realsense_manager = RealsenseManager(show_raw=False)
    retarget_algo = RetargetAlgorithm(
        str(robot_dir), str(config_path), show_skeleton=True
    )
    if use_mujoco:
        xml_path = retarget_algo.config.urdf_path.replace(".urdf", ".xml")
        simulator = MujocoSimulator(xml_path, retarget_algo.retargeting.joint_names)
    else:
        simulator = SapienSimulator(
            retarget_algo.config.urdf_path, retarget_algo.retargeting.joint_names
        )

    # shared variables
    queue = multiprocessing.Queue(maxsize=100)
    qpos = multiprocessing.Array("d", len(retarget_algo.retargeting.joint_names))

    # threads
    producer_process = multiprocessing.Process(
        target=realsense_manager.get_frame, args=(queue,)
    )
    consumer_process = multiprocessing.Process(
        target=retarget_algo.update, args=(queue, qpos)
    )
    renderer_process = multiprocessing.Process(target=simulator.render, args=(qpos,))

    producer_process.start()
    consumer_process.start()
    renderer_process.start()

    producer_process.join()
    consumer_process.join()
    renderer_process.join()

    time.sleep(5)
    print("done")


if __name__ == "__main__":
    tyro.cli(main)
