#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import rospy
import numpy as np
import threading
import sys
sys.path.append('/home/nvidia/swarm/devel/lib/python3/dist-packages')
from cam.msg import LightInfo, Cam1, Cam2, Cam3, Cam4, TrackStatus, g_r,Tracker
from ocsort import OCSort
from collections import deque

class UnifiedTracker:
    def __init__(self, output_path, tracker_params):
        self.output_path = output_path
        self.tracker_params = tracker_params
        self.tracker = OCSort(**tracker_params)
        self.id_status = {}
        self.frame_id = 0
        self.lock = threading.Lock()

        # 使用队列存储每个相机的数据，设置最大长度为10
        self.camera_queues = {
            1: deque(maxlen=10),
            2: deque(maxlen=10),
            3: deque(maxlen=10),
            4: deque(maxlen=10),
        }

        # 创建保存目录
        os.makedirs(output_path, exist_ok=True)
        self.id_status_file = f"{output_path}/id_status.txt"

        # ROS发布者
        self.tracking_pub = rospy.Publisher('/unified_tracker', TrackStatus, queue_size=10)
        self.tracked_lights_pub = rospy.Publisher('/tracked_lights', Tracker, queue_size=10)

    def process_data(self):
        """使用所有相机当前最新的方向向量和距离数据进行处理"""
        with self.lock:
            # 检查所有相机队列是否都有数据
            if not all(len(queue) > 0 for queue in self.camera_queues.values()):
                return

            current_ids = set()
            tracked_lights = []
            all_dets = []

            # 从每个相机队列中取出最早的数据进行处理
            for cam_id in range(1, 5):
                g_r_data = self.camera_queues[cam_id].popleft()  # 取出并移除最早的数据

                if g_r_data:
                    dets = []
                    for light_data in g_r_data:
                        g = np.array([light_data.x, light_data.y, 0])
                        g_norm = np.linalg.norm(g)
                        if g_norm > 0:
                            g = g / g_norm
                        else:
                            rospy.logwarn(f"Camera {cam_id} received zero vector for light data in frame {self.frame_id}.")
                            continue

                        r = light_data.distance
                        score = 1.0
                        if len(g) == 3 and not np.any(np.isnan(g)) and not np.isnan(r):
                            dets.append((g, r, score))
                        else:
                            rospy.logwarn(f"Camera {cam_id} received invalid data (NaN or wrong dim) in frame {self.frame_id}: g={g}, r={r}")

                    if dets:
                        all_dets.extend(dets)

            if all_dets:
                online_targets = self.tracker.update(all_dets)

                for t in online_targets:
                    if len(t) >= 4:
                        x, y, z, tid = t[:4]
                        current_ids.add(tid)

                        if tid not in self.id_status:
                            self.id_status[tid] = []

                        light = LightInfo()
                        light.x = x
                        light.y = y
                        if len(t) >= 6:
                            light.vx = t[4]
                            light.vy = t[5]
                        else:
                            light.vx = 0.0
                            light.vy = 0.0
                        tracked_lights.append(light)

            if tracked_lights:
                self.tracked_lights_pub.publish(Tracker(lights=tracked_lights))
                rospy.loginfo(f"Published {len(tracked_lights)} tracked lights for frame {self.frame_id}.")
            else:
                rospy.loginfo(f"Processed frame {self.frame_id}, but no lights tracked.")

            self.frame_id += 1

    def save_id_status(self):
        with self.lock:
            with open(self.id_status_file, 'w') as f:
                for tid, status in self.id_status.items():
                    f.write(f"ID {tid}: {status}\n")
            print(f"ID status saved to {self.id_status_file}")


class TrackerNode:
    def __init__(self):
        rospy.init_node('unified_tracker_node', anonymous=True)

        # Initialize unified tracker
        output_path = "/home/nvidia/swarm/tracking_results"
        tracker_params = {
            "det_thresh": 0.5,
            "iou_threshold": 0.3,
            "use_byte": False,
        }
        self.tracker = UnifiedTracker(output_path, tracker_params)

        # 创建订阅者
        self.cam1_sub = rospy.Subscriber('/Cam1', Cam1, lambda msg: self.camera_callback(msg, 1), queue_size=100)
        self.cam2_sub = rospy.Subscriber('/Cam2', Cam2, lambda msg: self.camera_callback(msg, 2), queue_size=100)
        self.cam3_sub = rospy.Subscriber('/Cam3', Cam3, lambda msg: self.camera_callback(msg, 3), queue_size=100)
        self.cam4_sub = rospy.Subscriber('/Cam4', Cam4, lambda msg: self.camera_callback(msg, 4), queue_size=100)

    def camera_callback(self, msg, camera_id):
        """处理单个相机的回调"""
        # 将新数据添加到对应相机的队列中
        if msg and hasattr(msg, 'lights'):
            self.tracker.camera_queues[camera_id].append(msg.lights)
            # 尝试处理数据
            self.tracker.process_data()

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        tracker_node = TrackerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'tracker_node' in locals() and hasattr(tracker_node, 'tracker'):
            tracker_node.tracker.save_id_status() 