#!/usr/bin/env python3
import sys
import os
import json
import time
import cv2

import config
from identity_db import IdentityDatabase
from face_reid_engine import FaceReIDEngine

try:
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from cv_bridge import CvBridge
    ROS1_AVAILABLE = True
except ImportError:
    ROS1_AVAILABLE = False


class PersonReIDNodeROS1:
    def __init__(self):
        if not ROS1_AVAILABLE:
            print("[ERROR] rospy or cv_bridge is not installed. Please source your ROS 1 workspace.")
            sys.exit(1)

        rospy.init_node('person_reid_node', anonymous=True)
        rospy.loginfo("Initializing ROS 1 Person Re-Identification Node...")

        self.db = IdentityDatabase()
        self.engine = FaceReIDEngine(db=self.db)
        self.bridge = CvBridge()

        # Publishers
        self.pub_annotated = rospy.Publisher(config.ROS_ANNOTATED_TOPIC, Image, queue_size=10)
        self.pub_detections = rospy.Publisher(config.ROS_DETECTIONS_TOPIC, String, queue_size=10)
        self.pub_events = rospy.Publisher(config.ROS_EVENTS_TOPIC, String, queue_size=10)

        # Subscriber
        self.sub_image = rospy.Subscriber(config.ROS1_CAMERA_TOPIC, Image, self.image_callback)
        rospy.loginfo(f"Subscribed to topic: {config.ROS1_CAMERA_TOPIC}")

        self.cap = None
        self.last_msg_time = time.time()
        self.timer = rospy.Timer(rospy.Duration(1.0 / config.TARGET_FPS), self.fallback_camera_check)

    def image_callback(self, msg):
        self.last_msg_time = time.time()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.process_and_publish(cv_image)
        except Exception as e:
            rospy.logerr(f"Failed to convert ROS Image: {e}")

    def fallback_camera_check(self, event):
        if time.time() - self.last_msg_time > 2.0:
            if self.cap is None or not self.cap.isOpened():
                rospy.logwarn(f"No ROS topic on {config.ROS1_CAMERA_TOPIC}. Opening laptop camera device {config.DEFAULT_CAMERA_INDEX}...")
                self.cap = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.process_and_publish(frame)

    def process_and_publish(self, frame):
        annotated_frame, detections = self.engine.process_frame(frame)

        if annotated_frame is not None:
            try:
                img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
                self.pub_annotated.publish(img_msg)
            except Exception as e:
                rospy.logerr(f"Error publishing image: {e}")

            if detections:
                det_msg = String()
                det_msg.data = json.dumps({
                    "timestamp": time.time(),
                    "detections": detections
                })
                self.pub_detections.publish(det_msg)

                for det in detections:
                    if det.get('is_new', False):
                        evt_msg = String()
                        evt_msg.data = json.dumps({
                            "event": "NEW_PERSON_REGISTERED",
                            "person_id": det['person_id'],
                            "name": det['name'],
                            "timestamp": time.time()
                        })
                        self.pub_events.publish(evt_msg)
                        rospy.loginfo(f"⚡ [EVENT] Registered new person: {det['person_id']}")

    def spin(self):
        rospy.spin()
        if self.cap and self.cap.isOpened():
            self.cap.release()


if __name__ == '__main__':
    if not ROS1_AVAILABLE:
        print("[ERROR] ROS 1 Python libraries (rospy) not found in current environment.")
        sys.exit(1)
    node = PersonReIDNodeROS1()
    node.spin()
