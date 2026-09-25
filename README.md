# Warning System for Target Identification

A ROS / ROS 2 & standalone computer vision system designed for laptop cameras on Linux. It detects targets coming into camera view, extracts 512-dimensional L2-normalized feature vector embeddings (InsightFace), stores them in a persistent vector database, automatically recognizes returning targets, and sends **instant warning notifications directly to your phone via KDEConnect**.

![ROS Re-ID Architecture](https://img.shields.io/badge/ROS-ROS%201%20%7C%20ROS%202-blue?logo=ros)
![Python](https://img.shields.io/badge/Python-3.8%2B-green?logo=python)
![InsightFace](https://img.shields.io/badge/InsightFace-buffalo__l-purple)
![KDEConnect](https://img.shields.io/badge/KDEConnect-Phone%20Alert-orange)

---

## 🌟 Key Features

1. **Automatic Target Enrollment**:
   - Automatically detects and registers new targets entering the camera view with unique IDs (`target_1`, `target_2` / `person_001`).
2. **Feature Embedding & Cosine Distance Matching**:
   - Uses deep feature vector embeddings (`buffalo_l`) and Cosine distance similarity metrics.
   - Smooths feature embeddings across sightings to maintain recognition across varying angles, lighting, and poses.
3. **📱 Instant Phone Warning Alerts (KDEConnect)**:
   - When a target's visit count exceeds the configured threshold (`ALERT_VISIT_COUNT = 3`), the system dispatches an instant warning ping directly to your phone (e.g. Nothing Phone 1 via `kdeconnect-cli`).
4. **ROS 1 & ROS 2 Native Support**:
   - Publishes annotated image streams, structured JSON detection arrays, and real-time event alerts over ROS topics.
5. **Interactive Web Dashboard**:
   - Modern dark glassmorphism dashboard running on `http://localhost:5000` with live camera feed, real-time activity log, target gallery, and inline renaming tool (e.g. rename `person_001` to "Target Alpha").
6. **Standalone GUI Mode**:
   - Run directly using `python3 reid.py` or `python3 target_warning_system.py` in an OpenCV window.

---

## 📁 Project Structure

```
ros_person_face_id/
├── target_warning_system.py # Main target identification script with phone alerts (reid.py)
├── reid.py                 # Core InsightFace target recognition & KDEConnect warning script
├── config.py               # Settings, phone device ID (KDEConnect), thresholds, ROS topics
├── identity_db.py          # Persistent target DB manager & Cosine similarity matcher
├── face_reid_engine.py     # Perception & warning dispatch pipeline
├── ros2_node.py            # ROS 2 Native Node publisher/subscriber
├── ros1_node.py            # ROS 1 Native Node publisher/subscriber
├── web_app.py              # Flask Web Server & API backend
├── test_camera.py          # Standalone OpenCV camera test script
├── templates/
│   └── index.html          # Web Dashboard HTML structure
├── static/
│   ├── css/style.css       # Dark mode glassmorphism UI styling
│   └── js/main.js          # Frontend dashboard interactivity & REST polling
├── data/
│   ├── identities.json     # Saved target profiles & embeddings
│   └── profiles/           # Saved face thumbnail image crops
├── requirements.txt        # Python package dependencies
└── README.md               # Documentation & usage guide
```

---

## ⚡ Quick Start Guide

### 1. Installation

Install required Python dependencies:
```bash
cd ~/.gemini/antigravity/scratch/ros_person_face_id
pip install -r requirements.txt
```

### 2. Standalone Target Warning Launcher
To test target detection, face feature extraction, and phone ping alerts directly:
```bash
python3 target_warning_system.py
```
*Press `q` to quit, `f` to enroll a friend.*

### 3. Interactive Web Dashboard
To launch the interactive control panel and web camera stream:
```bash
python3 web_app.py
```
Then open your web browser and navigate to: **`http://localhost:5000`**

---

## 📱 Phone Alert Setup (KDEConnect)

The warning system uses `kdeconnect-cli` to send instant warning pings.

To pair your phone:
1. Install KDE Connect on Linux and on your Android/iOS phone.
2. Pair your device and find your device ID by running:
   ```bash
   kdeconnect-cli -l
   ```
3. Update `KDECONNECT_DEVICE_ID` in `config.py` or `target_warning_system.py`:
   ```python
   KDECONNECT_DEVICE_ID = "7eb954b051f1476ca23fefd24386c8d4"
   ```

---

## 🤖 Running with ROS / ROS 2

### ROS 2 Setup & Launch
```bash
source /opt/ros/humble/setup.bash
python3 ros2_node.py
```

### ROS 1 Setup & Launch
```bash
source /opt/ros/noetic/setup.bash
python3 ros1_node.py
```

---

## 📡 ROS Topics Reference

| Topic Name | Message Type | Description |
| :--- | :--- | :--- |
| `/camera/image_raw` | `sensor_msgs/Image` | Input camera feed |
| `/person_id/image_annotated` | `sensor_msgs/Image` | Processed frame with HUD bounding boxes & warning tags |
| `/person_id/detections` | `std_msgs/String` (JSON) | Array of detected target IDs, bounding boxes, and match confidence |
| `/person_id/events` | `std_msgs/String` (JSON) | Event alerts published when a target is detected or phone warning fires |
