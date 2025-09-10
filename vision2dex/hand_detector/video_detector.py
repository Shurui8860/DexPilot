import pickle

import cv2
import tqdm

from vision2dex.dex_retargeting.constants import HandType
from vision2dex.hand_detector.image_detector import SingleHandDetector
from pathlib import Path
from typing import Union

def detect_video(video_path: str, output_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        data = []
        detector = SingleHandDetector(hand_type="Right", selfie=False)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm.tqdm(total=length) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                rgb = frame[..., ::-1]
                _, joint_pos, _, _ = detector.detect(rgb)
                data.append(joint_pos)
                pbar.update(1)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(data, f)

        cap.release()
        cv2.destroyAllWindows()


def main(
    video_path:  Union[str, Path],
    output_path: Union[str, Path],
    hand_type: HandType = HandType.right,
) -> Path:

    detect_video(str(video_path), str(output_path))

    return output_path

