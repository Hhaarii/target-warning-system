#!/usr/bin/env python3
import sys
import os
import json
import time
import cv2

import config
from identity_db import IdentityDatabase
from face_reid_engine import FaceReIDEngine

# Check ROS 2 imports gracefully
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


class PersonReIDNodeROS2(Node if ROS2_AVAILABLE else object):
    def __init__(self):
        if not ROS2_AVAILABLE:
            print("[ERROR] rclpy or cv_bridge is not installed. Please source your ROS 2 workspace.")
            sys.exit(1)

        super().__init__('person_reid_node')
        self.get_logger().info("Initializing ROS 2 Person Re-Identification & Face Tracking Node...")

        self.db = IdentityDatabase()
        self.engine = FaceReIDEngine(db=self.db)
        self.bridge = CvBridge()

        # Publishers
        self.pub_annotated = self.create_publisher(Image, config.ROS2_ANNOTATED_TOPIC, 10)
        self.pub_detections = self.create_publisher(String, config.ROS2_DETECTIONS_TOPIC, 10)
        self.pub_events = self.create_publisher(String, config.ROS_EVENTS_TOPIC, 10)

        # Subscriber
        self.sub_image = self.create_subscription(
            Image,
            config.ROS2_CAMERA_TOPIC,
            self.image_callback,
            10
        )

        self.get_logger().info(f"Subscribed to topic: {config.ROS2_CAMERA_TOPIC}")
        self.get_logger().info(f"Publishing annotated images to: {config.ROS2_ANNOTATED_TOPIC}")
        self.get_logger().info(f"Publishing detection JSON to: {config.ROS2_DETECTIONS_TOPIC}")

        # Fallback local video capture if camera topic not active
        self.cap = None
        self.last_msg_time = time.time()
        self.timer = self.create_timer(1.0 / config.TARGET_FPS, self.fallback_camera_check)

    def image_callback(self, msg):
        """Callback for camera image topic."""
        self.last_msg_time = time.time()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.process_and_publish(cv_image)
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS Image: {e}")

    def fallback_camera_check(self):
        """If ROS camera topic receives no messages for 2 seconds, use laptop camera directly."""
        if time.time() - self.last_msg_time > 2.0:
            if self.cap is None or not self.cap.isOpened():
                self.get_logger().warn(f"No ROS topic on {config.ROS2_CAMERA_TOPIC}. Opening laptop webcam device {config.DEFAULT_CAMERA_INDEX}...")
                self.cap = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.process_and_publish(frame)

    def process_and_publish(self, frame):
        """Run perception pipeline and publish ROS messages."""
        annotated_frame, detections = self.engine.process_frame(frame)

        if annotated_frame is not None:
            # 1. Publish annotated ROS Image
            try:
                img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
                self.pub_annotated.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f"Error publishing image: {e}")

            # 2. Publish JSON detection topic
            if detections:
                det_msg = String()
                det_msg.data = json.dumps({
                    "timestamp": time.time(),
                    "detections": detections
                })
                self.pub_detections.publish(det_msg)

                # 3. Publish alert event if NEW person registered
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
                        self.get_logger().info(f"⚡ [EVENT] Registered new person: {det['person_id']}")

    def destroy_node(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    if not ROS2_AVAILABLE:
        print("[ERROR] ROS 2 Python libraries (rclpy) not found in current environment.")
        return

    rclpy.init(args=args)
    node = PersonReIDNodeROS2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
