import os
import json
import time
import cv2
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

import config
from identity_db import IdentityDatabase
from face_reid_engine import FaceReIDEngine

app = Flask(__name__, template_folder='templates', static_folder='static')

db = IdentityDatabase()
engine = FaceReIDEngine(db=db)

# Global camera capture object
camera_cap = None
camera_index = config.DEFAULT_CAMERA_INDEX
recent_events = []


def get_camera():
    global camera_cap, camera_index
    if camera_cap is None or not camera_cap.isOpened():
        camera_cap = cv2.VideoCapture(camera_index)
        camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return camera_cap


def generate_frames():
    """Video streaming generator function."""
    global recent_events
    cap = get_camera()

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            time.sleep(0.03)
            continue

        annotated_frame, detections = engine.process_frame(frame)

        # Log new events
        for det in detections:
            if det.get("is_new", False):
                event_entry = {
                    "type": "new_person",
                    "title": f"New Identity Enrolled",
                    "details": f"{det['name']} ({det['person_id']}) registered",
                    "time": time.strftime("%H:%M:%S")
                }
                recent_events.insert(0, event_entry)
                if len(recent_events) > 50:
                    recent_events.pop()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/profiles', methods=['GET'])
def get_profiles():
    profiles = db.get_all_profiles()
    return jsonify({"success": True, "profiles": profiles, "total": len(profiles)})


@app.route('/api/rename', methods=['POST'])
def rename_person():
    data = request.json or {}
    person_id = data.get('person_id')
    new_name = data.get('name', '').strip()

    if not person_id or not new_name:
        return jsonify({"success": False, "error": "Invalid parameters"}), 400

    if db.rename_person(person_id, new_name):
        return jsonify({"success": True, "message": f"Updated {person_id} to '{new_name}'"})
    return jsonify({"success": False, "error": "Person ID not found"}), 404


@app.route('/api/delete', methods=['POST'])
def delete_person():
    data = request.json or {}
    person_id = data.get('person_id')

    if not person_id:
        return jsonify({"success": False, "error": "Missing person_id"}), 400

    if db.delete_person(person_id):
        return jsonify({"success": True, "message": f"Deleted identity {person_id}"})
    return jsonify({"success": False, "error": "Person ID not found"}), 404


@app.route('/api/thumbnail/<person_id>')
def get_thumbnail(person_id):
    thumb_path = os.path.join(config.PROFILES_DIR, f"{person_id}.jpg")
    if os.path.exists(thumb_path):
        return send_from_directory(config.PROFILES_DIR, f"{person_id}.jpg")
    return jsonify({"error": "Thumbnail not found"}), 404


@app.route('/api/events', methods=['GET'])
def get_events():
    return jsonify({"events": recent_events})


@app.route('/api/clear', methods=['POST'])
def clear_database():
    for pid in list(db.identities.keys()):
        db.delete_person(pid)
    return jsonify({"success": True, "message": "Cleared all registered identities."})


if __name__ == '__main__':
    print("[Web Server] Starting ROS Person Re-Identification Web Control Dashboard on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
