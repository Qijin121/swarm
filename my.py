import os
import os.path as osp
import time
import cv2
import numpy as np
from loguru import logger
# from trackers.ocsort_tracker.ocsort import OCSort
from ocsort import OCSort
from timer import Timer

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def load_detections(detection_file):
    """
    Load detections from a txt file.
    Format: frame_id,x1,y1,x2,y2,confidence
    """
    detections = {}
    with open(detection_file, "r") as f:
        for line in f.readlines():
            frame_id, x1, y1, x2, y2, conf = map(float, line.strip().split(","))
            frame_id = int(frame_id)
            if frame_id not in detections:
                detections[frame_id] = []
            detections[frame_id].append([x1, y1, x2, y2, conf])
    return detections

def imageflow_demo(detection_file, video_path, output_path, tracker_params):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    logger.info(f"Video properties: {width}x{height} at {fps} FPS")

    vid_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    tracker = OCSort(**tracker_params)
    timer = Timer()

    detections = load_detections(detection_file)
    frame_id = 0
    colours = np.random.rand(64, 3) * 255  # Random colors for tracking IDs

    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break

        if frame_id in detections:
            dets = np.array(detections[frame_id])
            timer.tic()
            online_targets = tracker.update(dets[:, :5], [height, width],(height, width))
            timer.toc()

            for t in online_targets:
                x1, y1, x2, y2, tid = map(int, t[:5])
                color = colours[tid % 64]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, f"ID: {tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        vid_writer.write(frame)
        logger.info(f"Processed frame {frame_id}")
        frame_id += 1

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q"):
            break

    cap.release()
    vid_writer.release()
    logger.info(f"Output saved to {output_path}")

if __name__ == "__main__":
    # Set paths
    video_path = "C:/Users/DELL/Desktop/p1.mp4"
    detection_file = "C:/Users/DELL/Desktop/track/output.txt"
    output_path = "C:/Users/DELL/Desktop/output_video.avi"

    # Tracker parameters
    tracker_params = {
        "det_thresh": 0.5,
        "iou_threshold": 0.3,
        "use_byte": False,
    }

    imageflow_demo(detection_file, video_path, output_path, tracker_params)
