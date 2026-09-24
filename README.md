# ROS Person Detection, Feature Embedding & Re-Identification System

A ROS / ROS 2 compatible computer vision system designed for laptop cameras on Linux. It detects people/faces coming into camera view, extracts 128-dimensional L2-normalized feature vector embeddings, stores them in a persistent vector database, and automatically recognizes registered individuals upon return.

![ROS Re-ID Architecture](https://img.shields.io/badge/ROS-ROS%201%20%7C%20ROS%202-blue?logo=ros)
![Python](https://img.shields.io/badge/Python-3.8%2B-green?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?logo=opencv)

---

## 🌟 Key Features

1. **Automatic Person Auto-Enrollment**:
   - When an un-enrolled person walks in front of the camera, the system extracts their feature embedding and automatically registers a new persistent profile (`person_001`, `person_002`, etc.).
2. **Feature Embedding & Cosine Distance Matching**:
   - Uses normalized feature vector embeddings and Cosine distance similarity metrics.
   - Smooths feature embeddings over time (exponential moving average) to maintain recognition across varying angles, lighting, and poses.
3. **ROS 1 & ROS 2 Native Support**:
   - Publishes annotated image streams, structured JSON detection arrays, and real-time event alerts over ROS topics.
4. **Interactive Web Dashboard**:
   - Modern dark glassmorphism dashboard running on `http://localhost:5000` with live camera feed, real-time activity log, profile gallery, and inline renaming tool (e.g. rename `person_001` to "Alice").
5. **Standalone GUI Mode**:
   - Option to run directly in an OpenCV GUI window without needing ROS active.

---

## 📁 Project Structure

```
ros_person_face_id/
├── config.py              # Configuration settings, thresholds, and ROS topics
├── identity_db.py         # Persistent identity DB manager & Cosine similarity matcher
├── face_reid_engine.py    # Detection, feature extraction & frame HUD annotation engine
├── ros2_node.py           # ROS 2 Native Node publisher/subscriber
├── ros1_node.py           # ROS 1 Native Node publisher/subscriber
├── web_app.py             # Flask Web Server & API backend
├── test_camera.py         # Standalone OpenCV camera test script
├── templates/
│   └── index.html         # Web Dashboard HTML structure
├── static/
│   ├── css/style.css      # Dark mode glassmorphism UI styling
│   └── js/main.js         # Frontend dashboard interactivity & REST polling
├── data/
│   ├── identities.json    # Saved person feature profiles & embeddings
│   └── profiles/          # Saved face thumbnail image crops
├── requirements.txt       # Python package dependencies
└── README.md              # Documentation & usage guide
```

---

## ⚡ Quick Start Guide

### 1. Installation

Install required Python dependencies:
```bash
cd ~/.gemini/antigravity/scratch/ros_person_face_id
pip install -r requirements.txt
```

### 2. Standalone Webcam Test (No ROS required)
To test camera capture, face detection, and embedding matching in a standard OpenCV GUI window:
```bash
python3 test_camera.py
```
*Press `q` to quit, `r` to reset saved identities.*

### 3. Interactive Web Dashboard
To launch the interactive control panel and web camera stream:
```bash
python3 web_app.py
```
Then open your web browser and navigate to: **`http://localhost:5000`**

---

## 🤖 Running with ROS / ROS 2

### ROS 2 Setup & Launch
1. Source your ROS 2 workspace (e.g., Humble / Foxy / Jazzy):
   ```bash
   source /opt/ros/humble/setup.bash
   ```
2. Run the ROS 2 Re-Identification Node:
   ```bash
   python3 ros2_node.py
   ```

### ROS 1 Setup & Launch
1. Source your ROS 1 workspace:
   ```bash
   source /opt/ros/noetic/setup.bash
   ```
2. Run the ROS 1 Re-Identification Node:
   ```bash
   python3 ros1_node.py
   ```

---

## 📡 ROS Topics Reference

| Topic Name | Message Type | Description |
| :--- | :--- | :--- |
| `/camera/image_raw` | `sensor_msgs/Image` | Input camera feed (Subscribed by Node) |
| `/person_id/image_annotated` | `sensor_msgs/Image` | Processed frame with HUD bounding boxes & identity tags |
| `/person_id/detections` | `std_msgs/String` (JSON) | Array of detected person IDs, bounding boxes `[x,y,w,h]`, and match confidence |
| `/person_id/events` | `std_msgs/String` (JSON) | Event alerts published when a NEW person is registered |

### Example JSON Payload on `/person_id/detections`:
```json
{
  "timestamp": 1727170000.12,
  "detections": [
    {
      "person_id": "person_001",
      "name": "Alice",
      "bbox": [140, 80, 210, 210],
      "confidence": 94.2,
      "distance": 0.18,
      "is_new": false
    }
  ]
}
```

---

## ⚙️ Configuration & Customization (`config.py`)

You can tune system parameters in [config.py](file:///home/ha-r1/.gemini/antigravity/scratch/ros_person_face_id/config.py):

- **`MATCH_THRESHOLD = 0.42`**:
  - Distance threshold for matching embeddings (Cosine distance).
  - Decrease value (e.g. `0.35`) for **stricter** matching (avoids false positives).
  - Increase value (e.g. `0.50`) for **looser** matching.
- **`EMBEDDING_SMOOTHING = 0.85`**:
  - Weight ratio for updating existing person feature embeddings over time.
- **`DEFAULT_CAMERA_INDEX = 0`**:
  - Video device index (default is `/dev/video0`).

---

## 🛠️ How Biometric Re-Identification Works

1. **Detection**: Each frame is scanned for facial bounding regions using multi-scale cascade detectors.
2. **Embedding Extraction**: For each ROI crop, a 128-dimensional L2-normalized feature vector is computed combining spatial HOG descriptors, HSV color distributions, and structural intensity moments.
3. **Cosine Distance Search**: The feature vector is compared against all registered identities in `identities.json` using vector dot products:
   $$\text{Cosine Distance} = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
4. **Auto-Enrollment vs Recognition**:
   - If $\text{Distance} \le 0.42$, identity is matched and sighting updated.
   - If $\text{Distance} > 0.42$, a new person profile is created automatically.
