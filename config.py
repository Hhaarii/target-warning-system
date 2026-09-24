import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROFILES_DIR = os.path.join(DATA_DIR, "profiles")
DB_FILE = os.path.join(DATA_DIR, "identities.json")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROFILES_DIR, exist_ok=True)

# Camera & Processing Settings
DEFAULT_CAMERA_INDEX = 0  # Laptop camera (/dev/video0)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# Feature Extraction & Matching Settings
# Similarity metric: Cosine distance (0.0 = identical, 1.0 = completely different)
MATCH_THRESHOLD = 0.42       # Cosine distance threshold for recognizing a face (lower = stricter)
CONFIDENCE_THRESHOLD = 0.60  # Minimum face/person detection confidence
EMBEDDING_SMOOTHING = 0.85   # Weight for updating known embedding with new observations (Exponential Moving Avg)

# ROS Settings
ROS1_CAMERA_TOPIC = "/camera/image_raw"
ROS2_CAMERA_TOPIC = "/camera/image_raw"
ROS_ANNOTATED_TOPIC = "/person_id/image_annotated"
ROS_DETECTIONS_TOPIC = "/person_id/detections"
ROS_EVENTS_TOPIC = "/person_id/events"
