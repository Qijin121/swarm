#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import rospy
import numpy as np
import threading
from datetime import datetime
from cam.msg import LightInfo, Cam1, Cam2, Cam3, Cam4, TrackStatus, g_r
from ocsort import OCSort
from message_filters import TimeSynchronizer, Subscriber
from collections import deque

class UnifiedTracker:
    def __init__(self, output_path, tracker_params):
        self.output_path = output_path
        self.tracker_params = tracker_params
        self.tracker = OCSort(**tracker_params)
        self.id_status = {}
        self.frame_id = 0
        self.lock = threading.Lock()
        self.camera_ready = {1: False, 2: False, 3: False, 4: False}
        self.first_frame_time = None
        self.sync_threshold = 0.1  # 100ms 同步阈值
        self.message_buffer = {1: deque(maxlen=10), 2: deque(maxlen=10), 
                             3: deque(maxlen=10), 4: deque(maxlen=10)}

        # 创建保存目录
        os.makedirs(output_path, exist_ok=True)
        self.id_status_file = f"{output_path}/id_status.txt"

        # ROS发布者
        self.tracking_pub = rospy.Publisher('/unified_tracker', TrackStatus, queue_size=10)
        # 添加一个统一的发布者，用于发布所有相机的跟踪结果
        self.tracked_lights_pub = rospy.Publisher('/tracked_lights', g_r, queue_size=10)

    def check_sync(self, camera_id, msg_time):
        """检查消息时间是否在同步阈值内"""
        if self.first_frame_time is None:
            self.first_frame_time = msg_time
            self.camera_ready[camera_id] = True
            return True
        
        time_diff = abs(msg_time - self.first_frame_time)
        if time_diff <= self.sync_threshold:
            self.camera_ready[camera_id] = True
            return True
        return False

    def all_cameras_ready(self):
        """检查是否所有相机都已准备就绪"""
        return all(self.camera_ready.values())

    def process_data(self, g_r_data, camera_id, msg_time):
        """处理方向向量和距离数据"""
        with self.lock:
            # 将消息存入缓冲区
            self.message_buffer[camera_id].append((g_r_data, msg_time))
            
            # 检查时间同步
            if not self.check_sync(camera_id, msg_time):
                return []

            # 如果所有相机都未就绪，等待
            if not self.all_cameras_ready():
                return []

            # 获取所有相机的最新消息
            latest_messages = {}
            for cam_id in range(1, 5):
                if self.message_buffer[cam_id]:
                    latest_messages[cam_id] = self.message_buffer[cam_id][-1]

            # 检查是否所有相机都有消息
            if len(latest_messages) < 4:
                return []

            # 检查消息时间是否同步
            times = [msg[1] for msg in latest_messages.values()]
            max_time_diff = max(times) - min(times)
            if max_time_diff > self.sync_threshold:
                return []

            current_ids = set()
            tracked_lights = []

            # 处理所有相机的数据
            for cam_id, (data, _) in latest_messages.items():
                if len(data) > 0:
                    # 构建检测数据 [(hat_g, hat_r, score)]
                    dets = []
                    for light_data in data:
                        # 构建3维方向向量
                        g = np.array([light_data.x, light_data.y, 0])
                        g = g / np.linalg.norm(g)  # 归一化
                        r = light_data.distance
                        score = 1.0  # 设置检测置信度
                        # 确保g是3维向量
                        if len(g) == 3 and not np.any(np.isnan(g)) and not np.isnan(r):
                            dets.append((g, r, score))
                    
                    if len(dets) > 0:
                        # 更新跟踪器
                        online_targets = self.tracker.update(dets)

                        for t in online_targets:
                            x, y, z, tid = t[:4]
                            current_ids.add(tid)

                            # 更新ID状态
                            if tid not in self.id_status:
                                self.id_status[tid] = []
                            self.id_status[tid].append((self.frame_id, 1))

                            # 发布跟踪消息
                            track_msg = TrackStatus()
                            track_msg.track_id = int(tid)
                            track_msg.status = 1
                            track_msg.camera_id = cam_id
                            self.tracking_pub.publish(track_msg)

            # 获取当前所有跟踪器的状态
            tracked_states = self.tracker.get_state()
            
            # 处理持续跟踪的目标
            if tracked_states:
                for state in tracked_states:
                    if len(state) >= 3:  # 确保有完整的坐标[x,y,z]
                        x, y, z, tid, vx, vy, vz = state
                        if tid not in current_ids:
                            # 更新ID状态
                            if tid not in self.id_status:
                                self.id_status[tid] = []
                            self.id_status[tid].append((self.frame_id, 0))
                            
                            # 发布跟踪消息
                            track_msg = TrackStatus()
                            track_msg.track_id = tid
                            track_msg.status = 0
                            track_msg.camera_id = camera_id
                            self.tracking_pub.publish(track_msg)
                        
                        # 创建新的light数据
                        light = LightInfo()
                        light.x = x
                        light.y = y
                        light.vx = vx
                        light.vy = vy
                        tracked_lights.append(light)

            # 发布所有跟踪后的消息到一个统一的话题
            if tracked_lights:
                self.tracked_lights_pub.publish(g_r(lights=tracked_lights))

            self.frame_id += 1
            
            return tracked_lights

    def save_id_status(self):
        with self.lock:
            with open(self.id_status_file, 'w') as f:
                for tid, status in self.id_status.items():
                    f.write(f"ID {tid}: {status}\n")
            print(f"ID status saved to {self.id_status_file}")

class TrackerNode:
    def __init__(self):
        rospy.init_node('unified_tracker_node', anonymous=True)
        
        # 初始化统一跟踪器
        output_path = "/home/nvidia/swarm/tracking_results"
        tracker_params = {
            "det_thresh": 0.5,
            "iou_threshold": 0.3,
            "use_byte": False,
        }
        self.tracker = UnifiedTracker(output_path, tracker_params)
        
        # 使用 TimeSynchronizer 订阅所有相机的消息
        self.cam1_sub = Subscriber('/Cam1', Cam1)
        self.cam2_sub = Subscriber('/Cam2', Cam2)
        self.cam3_sub = Subscriber('/Cam3', Cam3)
        self.cam4_sub = Subscriber('/Cam4', Cam4)

        # 创建时间同步器
        self.ts = TimeSynchronizer(
            [self.cam1_sub, self.cam2_sub, self.cam3_sub, self.cam4_sub],
            queue_size=100
        )
        self.ts.registerCallback(self.sync_callback)

    def sync_callback(self, cam1_msg, cam2_msg, cam3_msg, cam4_msg):
        """同步处理所有相机的消息"""
        current_time = rospy.Time.now().to_sec()
        
        # 处理每个相机的数据
        if cam1_msg.lights:
            self.tracker.process_data(cam1_msg.lights, 1, current_time)
        if cam2_msg.lights:
            self.tracker.process_data(cam2_msg.lights, 2, current_time)
        if cam3_msg.lights:
            self.tracker.process_data(cam3_msg.lights, 3, current_time)
        if cam4_msg.lights:
            self.tracker.process_data(cam4_msg.lights, 4, current_time)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        tracker_node = TrackerNode()
        tracker_node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        tracker_node.tracker.save_id_status() 