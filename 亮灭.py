import os
import os.path as osp
import time
import cv2
import numpy as np
from loguru import logger
# from trackers.ocsort_tracker.ocsort import OCSort
from ocsort import OCSort
from timer import Timer
import csv

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

    # 用于记录ID是否出现过的字典
    id_status = {}

    # 创建保存帧的文件夹
    frames_folder = f"{output_path}_frames"
    os.makedirs(frames_folder, exist_ok=True)

    # 创建保存跟踪结果的 CSV 文件
    results_csv = f"{output_path}_tracking_results.csv"
    id_status_file = f"{output_path}_id_status.txt"

    with open(results_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["frame_id", "id", "x1", "y1", "x2", "y2"])  # 表头

        while True:
            ret_val, frame = cap.read()
            if not ret_val:
                break

            if frame_id in detections:
                dets = np.array(detections[frame_id])
                timer.tic()
                online_targets = tracker.update(dets[:, :5], [height, width], (height, width))
                timer.toc()

                # 当前帧 ID 集合
                current_ids = set()

                for t in online_targets:
                    x1, y1, x2, y2, tid = map(int, t[:5])
                    current_ids.add(tid)

                    # 绘制跟踪框和ID
                    color = colours[tid % 64]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, f"ID: {tid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

                    # 写入 CSV 文件
                    csv_writer.writerow([frame_id, tid, x1, y1, x2, y2])

                    # 更新 ID 状态
                    if tid not in id_status:
                        id_status[tid] = []
                    id_status[tid].append((frame_id, 1))  # 当前帧该ID出现，记录为 (frame_id, 1)

                # 对于未在当前帧出现的 ID，记录为 (frame_id, 0)
                for tracked_id in id_status:
                    if tracked_id not in current_ids:
                        id_status[tracked_id].append((frame_id, 0))

            # 保存当前帧
            frame_path = os.path.join(frames_folder, f"frame_{frame_id}.png")
            cv2.imwrite(frame_path, frame)

            # 写入视频帧
            vid_writer.write(frame)
            logger.info(f"Processed frame {frame_id}")
            frame_id += 1

            # 按键退出
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q"):
                break

    cap.release()
    vid_writer.release()

    # 输出 ID 的出现记录到文件
    with open(id_status_file, 'w') as f:
        for tid, status in id_status.items():
            f.write(f"ID {tid}: {status}\n")

    logger.info(f"Output saved to {output_path}")
    logger.info(f"Frames saved to {frames_folder}")
    logger.info(f"Tracking results saved to {results_csv}")
    logger.info(f"ID status saved to {id_status_file}")



if __name__ == "__main__":
    # Set paths
    video_path = "C:/Users/DELL/Desktop/p3.mp4"
    detection_file = "C:/Users/DELL/Desktop/track/output.txt"
    output_path = "C:/Users/DELL/Desktop/p3_output.avi"

    # Tracker parameters
    tracker_params = {
        "det_thresh": 0.5,
        "iou_threshold": 0.3,
        "use_byte": False,
    }

    imageflow_demo(detection_file, video_path, output_path, tracker_params)
