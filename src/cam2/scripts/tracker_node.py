#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import rospy
import numpy as np
import threading
# from datetime import datetime # Not needed for manual time sync
import sys
sys.path.append('/home/nvidia/swarm/devel/lib/python3/dist-packages')
from cam.msg import LightInfo, Cam1, Cam2, Cam3, Cam4, TrackStatus, g_r,Tracker
from ocsort import OCSort
from message_filters import ApproximateTimeSynchronizer, Subscriber
# from collections import deque # Not needed for manual buffering

class UnifiedTracker:
    def __init__(self, output_path, tracker_params):
        self.output_path = output_path
        self.tracker_params = tracker_params
        self.tracker = OCSort(**tracker_params)
        self.id_status = {}
        self.frame_id = 0
        self.lock = threading.Lock()

        # 存储每个相机的最新消息数据
        self.latest_camera_data = {
            1: None,
            2: None,
            3: None,
            4: None,
        }
        # 标记每个相机是否收到了新消息，用于在第一次同步后进行处理触发
        self.new_message_flags = {
            1: False,
            2: False,
            3: False,
            4: False,
        }
        self.first_sync_done = False # 标志位，标记第一次同步是否完成


        # 创建保存目录
        os.makedirs(output_path, exist_ok=True)
        self.id_status_file = f"{output_path}/id_status.txt"

        # ROS发布者
        self.tracking_pub = rospy.Publisher('/unified_tracker', TrackStatus, queue_size=10)
        # 添加一个统一的发布者，用于发布所有相机的跟踪结果
        self.tracked_lights_pub = rospy.Publisher('/tracked_lights', Tracker, queue_size=10)


    def all_new_messages_received(self):
        """检查是否所有相机都收到了新消息"""
        return all(self.new_message_flags.values())

    def reset_new_message_flags(self):
        """重置所有相机的新消息标志"""
        self.new_message_flags = {1: False, 2: False, 3: False, 4: False}


    # 修改 process_data 来处理所有相机当前最新的数据
    def process_data(self):
        """使用所有相机当前最新的方向向量和距离数据进行处理"""
        # Using frame_id for logging to distinguish processing cycles
        # rospy.loginfo(f"Processing data for frame {self.frame_id}. Attempting to acquire lock.")
        with self.lock:
            # Use frame_id in acquired lock log as well
            # rospy.loginfo(f"Lock acquired for frame {self.frame_id} at {rospy.Time.now().to_sec()}.")

            current_ids = set()
            tracked_lights = []
            all_dets = [] # Collect all detections from all cameras

            # Process data from each camera using the latest stored data
            has_any_data_this_frame = False # Check if there's any valid light data to process
            for cam_id in range(1, 5):
                 g_r_data = self.latest_camera_data[cam_id] # Get the latest data for this camera

                 if g_r_data: # Check if the data for this camera is not empty
                     # has_any_data_this_frame = True # This check needs to be done after building dets

                     # Build detection data [(hat_g, hat_r, score)]
                     dets = []
                     for light_data in g_r_data:
                         # Build 3D direction vector
                         g = np.array([light_data.x, light_data.y, 0])
                         g_norm = np.linalg.norm(g)
                         if g_norm > 0: # Avoid division by zero
                            g = g / g_norm  # Normalize
                         else:
                            rospy.logwarn(f"Camera {cam_id} received zero vector for light data in frame {self.frame_id}.")
                            continue # Skip invalid data

                         r = light_data.distance
                         score = 1.0  # Set detection confidence
                         # Ensure g is a 3D vector and data is valid
                         if len(g) == 3 and not np.any(np.isnan(g)) and not np.isnan(r):
                             dets.append((g, r, score))
                         else:
                             rospy.logwarn(f"Camera {cam_id} received invalid data (NaN or wrong dim) in frame {self.frame_id}: g={g}, r={r}")

                     if dets: # If there are valid detections for this camera
                          all_dets.extend(dets) # Add to the total detections list

            # Check if there's any data to process after collecting from all cameras
            if len(all_dets) > 0:
                 has_any_data_this_frame = True

            # Update tracker with all detections from the current data snapshot
            if all_dets:
                 online_targets = self.tracker.update(all_dets)

                 for t in online_targets:
                     # Ensure tracked result has enough dimensions
                     if len(t) >= 4:
                         x, y, z, tid = t[:4]
                         current_ids.add(tid)

                         # Update ID status (consider how to record status per frame)
                         if tid not in self.id_status:
                             self.id_status[tid] = []
                         # Example: self.id_status[tid].append((self.frame_id, 1)) # Record frame ID and status

                         # Publish tracking message (optional, depending on need for per-camera status)
                         # If /unified_tracker is for overall track status, this should be based on final tracks, not per detection.
                         # track_msg = TrackStatus()
                         # track_msg.track_id = int(tid)
                         # track_msg.status = 1 # 1 means tracked
                         # track_msg.camera_id = ? # This might not be meaningful here
                         # self.tracking_pub.publish(track_msg)

                         # Create new light data for /tracked_lights_pub
                         light = LightInfo()
                         light.x = x
                         light.y = y
                         # Get velocity from tracker state if available
                         if len(t) >= 6:
                              light.vx = t[4]
                              light.vy = t[5]
                         else:
                              light.vx = 0.0
                              light.vy = 0.0 # Or other default
                         tracked_lights.append(light)

            # Publish all tracked lights for the frame
            if tracked_lights:
                self.tracked_lights_pub.publish(Tracker(lights=tracked_lights))
                rospy.loginfo(f"Published {len(tracked_lights)} tracked lights for frame {self.frame_id}.")
            elif has_any_data_this_frame: # Log only if there was input data but no tracked lights
                 rospy.loginfo(f"Processed frame {self.frame_id}, had input data, but no lights tracked.")
            else: # Log if callback triggered but no cameras had any data
                 rospy.loginfo(f"Processed frame {self.frame_id}, but no cameras had any data.")


            # Handle targets not detected in this frame but still tracked (if needed)
            # This depends on OCSort providing access to all current tracks.
            # You would iterate through all tracks and publish status=0 for those not in current_ids.

            self.frame_id += 1

            # rospy.loginfo(f"Lock released for frame {self.frame_id-1} at {rospy.Time.now().to_sec()}.")

    def save_id_status(self):
        with self.lock:
            with open(self.id_status_file, 'w') as f:
                for tid, status in self.id_status.items():
                    # Adjust status saving based on how you track ID status
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

        # ApproximateTimeSynchronizer slop can be adjusted based on acceptable time difference
        # queue_size should be large enough
        sync_slop = 0.05 # Example value, may need tuning

        # *** Initial synchronization using ApproximateTimeSynchronizer ***
        self.cam1_sub_sync = Subscriber('/Cam1', Cam1)
        self.cam2_sub_sync = Subscriber('/Cam2', Cam2)
        self.cam3_sub_sync = Subscriber('/Cam3', Cam3)
        self.cam4_sub_sync = Subscriber('/Cam4', Cam4)

        self.ts = ApproximateTimeSynchronizer(
            [self.cam1_sub_sync, self.cam2_sub_sync, self.cam3_sub_sync, self.cam4_sub_sync],
            queue_size=100,
            slop=sync_slop
        )
        self.ts.registerCallback(self.initial_sync_callback)

        # *** Individual subscribers (will be created after initial sync) ***
        # Use large queue size for individual subscribers to not drop messages
        self.individual_queue_size = 100
        self.cam1_sub_individual = None
        self.cam2_sub_individual = None
        self.cam3_sub_individual = None
        self.cam4_sub_individual = None


    def initial_sync_callback(self, cam1_msg, cam2_msg, cam3_msg, cam4_msg):
        """Callback for the initial synchronization."""
        rospy.loginfo("Initial sync callback triggered.")

        if not self.tracker.first_sync_done:
            # Store the data from the first synchronized messages
            self.tracker.latest_camera_data[1] = cam1_msg.lights if (cam1_msg and hasattr(cam1_msg, 'lights')) else None
            self.tracker.latest_camera_data[2] = cam2_msg.lights if (cam2_msg and hasattr(cam2_msg, 'lights')) else None
            self.tracker.latest_camera_data[3] = cam3_msg.lights if (cam3_msg and hasattr(cam3_msg, 'lights')) else None
            self.tracker.latest_camera_data[4] = cam4_msg.lights if (cam4_msg and hasattr(cam4_msg, 'lights')) else None

            # Process the first synchronized data
            # Only process if there is any data from any camera in the first sync
            has_any_initial_data = False
            for data in self.tracker.latest_camera_data.values():
                if data is not None and len(data) > 0:
                    has_any_initial_data = True
                    break

            if has_any_initial_data:
                 self.tracker.process_data()
                 rospy.loginfo("Processed initial synchronized data.")
            else:
                 rospy.logwarn("Initial sync triggered but no cameras had light data. Will wait for individual messages.")


            # Set the flag regardless of whether there was data, to switch modes
            self.tracker.first_sync_done = True
            rospy.loginfo("First sync completed. Switching to individual subscribers.")

            # Unsubscribe the TimeSynchronizer subscribers
            # Give a small delay to ensure any pending messages in the sync queue are processed
            rospy.sleep(0.1) # Adjust delay if needed
            self.cam1_sub_sync.unregister()
            self.cam2_sub_sync.unregister()
            self.cam3_sub_sync.unregister()
            self.cam4_sub_sync.unregister()
            rospy.loginfo("Unregistered initial sync subscribers.")

            # Create individual subscribers
            self.cam1_sub_individual = Subscriber('/Cam1', Cam1, queue_size=self.individual_queue_size, callback=lambda msg: self.individual_camera_callback(msg, 1))
            self.cam2_sub_individual = Subscriber('/Cam2', Cam2, queue_size=self.individual_queue_size, callback=lambda msg: self.individual_camera_callback(msg, 2))
            self.cam3_sub_individual = Subscriber('/Cam3', Cam3, queue_size=self.individual_queue_size, callback=lambda msg: self.individual_camera_callback(msg, 3))
            self.cam4_sub_individual = Subscriber('/Cam4', Cam4, queue_size=self.individual_queue_size, callback=lambda msg: self.individual_camera_callback(msg, 4))
            rospy.loginfo("Created individual subscribers.")

        # Note: After the first sync, this callback might still be triggered
        # for any remaining messages in the TimeSynchronizer's queue, but the
        # 'if not self.tracker.first_sync_done:' check prevents reprocessing.


    def individual_camera_callback(self, msg, camera_id):
        """Callback for individual camera subscribers."""
        # Check if first sync is done before processing
        if not self.tracker.first_sync_done:
             # This should not happen if unregister works correctly, but as a safeguard
             rospy.logwarn(f"Received individual message from Camera {camera_id} before first sync completed.")
             return # Ignore messages before first sync is done

        # Store the latest data for this camera
        self.tracker.latest_camera_data[camera_id] = msg.lights if (msg and hasattr(msg, 'lights')) else None

        # Mark this camera as having received a new message
        self.tracker.new_message_flags[camera_id] = True
        # rospy.loginfo(f"Received new message from Camera {camera_id} at {msg.header.stamp.to_sec()}. Flags: {self.tracker.new_message_flags}")


        # Check if all cameras have received a new message since the last processing
        if self.tracker.all_new_messages_received():
            rospy.loginfo(f"New messages received from all cameras. Triggering processing for frame {self.tracker.frame_id}.")
            # Trigger processing with the latest data from all cameras
            self.tracker.process_data()
            # Reset the new message flags for the next cycle
            self.tracker.reset_new_message_flags()
            # rospy.loginfo(f"Flags reset: {self.tracker.new_message_flags}")
        # else:
            # Optional: Log if waiting for other cameras
            # rospy.loginfo(f"Waiting for messages from other cameras. Current flags: {self.tracker.new_message_flags}")


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        tracker_node = TrackerNode()
        rospy.spin() # Use rospy.spin() here to keep the node alive
    except rospy.ROSInterruptException:
        pass
    finally:
        # Save status before shutting down
        if 'tracker_node' in locals() and hasattr(tracker_node, 'tracker'):
             tracker_node.tracker.save_id_status() 