#!/usr/bin/env python3
"""
Standalone OpenCV GUI Launcher for Person Detection & Identification
Run this script directly to test laptop camera & face embedding recognition in a GUI window!
"""
import cv2
import sys
import time
import config
from identity_db import IdentityDatabase
from face_reid_engine import FaceReIDEngine

def main():
    print("=" * 60)
    print("  ROS Person Re-ID Standalone Camera Test")
    print("  Press 'q' or 'ESC' in the window to quit.")
    print("  Press 'r' to clear database & re-enroll.")
    print("=" * 60)

    db = IdentityDatabase()
    engine = FaceReIDEngine(db=db)

    cap = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera device index {config.DEFAULT_CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    window_name = "ROS Person Identification & Tracking HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Failed to grab frame from camera.")
            time.sleep(0.1)
            continue

        # Process frame
        annotated_frame, detections = engine.process_frame(frame)

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (15, annotated_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow(window_name, annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            db.identities.clear()
            db.save_database()
            print("[DB] Cleared all identities.")

    cap.release()
    cv2.destroyAllWindows()
    print("Camera test completed.")

if __name__ == '__main__':
    main()
