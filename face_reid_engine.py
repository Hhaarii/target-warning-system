import os
import cv2
import numpy as np
import config
from identity_db import IdentityDatabase

# Try importing InsightFace from user's reid.py setup
INSIGHTFACE_AVAILABLE = False
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class FaceReIDEngine:
    def __init__(self, db: IdentityDatabase = None):
        self.db = db if db else IdentityDatabase()
        self.use_insightface = False
        
        # Try initializing InsightFace model (from reid.py)
        if INSIGHTFACE_AVAILABLE:
            try:
                print("[ReID Engine] Initializing InsightFace (buffalo_l) feature extraction engine...")
                self.app = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CPUExecutionProvider"],
                    allowed_modules=["detection", "recognition"]
                )
                self.app.prepare(ctx_id=0, det_size=(320, 320))
                self.use_insightface = True
                print("[ReID Engine] InsightFace engine successfully active!")
            except Exception as e:
                print(f"[ReID Engine] InsightFace model init notice: {e}. Falling back to OpenCV Engine.")
                self.use_insightface = False
        
        # Load OpenCV Fallback Cascades
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
        self.profile_cascade = cv2.CascadeClassifier(profile_path)
        
        if not self.use_insightface:
            print("[ReID Engine] OpenCV Hybrid Feature Extractor Active.")

    def extract_feature_vector_opencv(self, face_crop):
        """Fallback 128-d feature vector when InsightFace is offline."""
        if face_crop is None or face_crop.size == 0:
            return None
        
        roi = cv2.resize(face_crop, (128, 128))
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        hog_features = []
        cell_size = 32
        for r in range(4):
            for c in range(4):
                cell_mag = mag[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size]
                cell_ang = angle[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size]
                hist, _ = np.histogram(cell_ang, bins=4, range=(0, 360), weights=cell_mag)
                hog_features.extend(hist)
        
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        g_hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        
        spatial_moments = []
        for r in range(4):
            for c in range(4):
                grid = gray[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size]
                spatial_moments.append(np.mean(grid))
                spatial_moments.append(np.std(grid))
        
        raw_embedding = np.concatenate([hog_features, h_hist, s_hist, v_hist, g_hist, spatial_moments]).astype(np.float32)
        norm = np.linalg.norm(raw_embedding)
        return raw_embedding / norm if norm > 0 else raw_embedding

    def process_frame(self, frame):
        """
        Perception Pipeline:
        1. Detect faces & extract normalized feature embeddings
        2. Query database for matching identity via Cosine Distance
        3. Auto-enroll NEW persons or update EXISTING sightings
        4. Render HUD annotations
        """
        if frame is None:
            return None, []
        
        annotated_frame = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        detection_records = []

        if self.use_insightface:
            # --- InsightFace Pipeline (from user's reid.py) ---
            try:
                faces = self.app.get(frame)
                for f in faces:
                    box = f.bbox.astype(int)
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    w, h = max(1, x2 - x1), max(1, y2 - y1)
                    x, y = max(0, x1), max(0, y1)
                    
                    embedding = f.normed_embedding
                    face_crop = frame[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)]
                    
                    self._process_single_face(frame, annotated_frame, x, y, w, h, embedding, face_crop, detection_records)
                
                known_count = len(self.db.identities)
                cv2.putText(annotated_frame, f"ROS ReID (InsightFace) | Profiles: {known_count} | Detections: {len(detection_records)}", 
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                return annotated_frame, detection_records
            except Exception as e:
                print(f"[Engine Warn] InsightFace execution error: {e}. Reverting to OpenCV pipeline.")

        # --- OpenCV Fallback Pipeline ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        face_boxes = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        for (x, y, w, h) in face_boxes:
            pad_w, pad_h = int(w * 0.1), int(h * 0.1)
            x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
            x2, y2 = min(w_frame, x + w + pad_w), min(h_frame, y + h + pad_h)
            face_crop = frame[y1:y2, x1:x2]
            
            embedding = self.extract_feature_vector_opencv(face_crop)
            if embedding is not None:
                self._process_single_face(frame, annotated_frame, x, y, w, h, embedding, face_crop, detection_records)

        known_count = len(self.db.identities)
        cv2.putText(annotated_frame, f"ROS ReID Active | Profiles: {known_count} | Detections: {len(detection_records)}", 
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        
        return annotated_frame, detection_records

    def _process_single_face(self, frame, annotated_frame, x, y, w, h, embedding, face_crop, detection_records):
        matched_id, matched_name, distance, confidence_pct = self.db.find_match(embedding)
        
        is_new = False
        if matched_id is not None:
            person_id = matched_id
            person_name = matched_name
            self.db.update_sighting(person_id, embedding, crop_img=face_crop)
            box_color = (0, 230, 115)  # Green for known match
            label_prefix = f"RECOGNIZED ({confidence_pct:.0f}%)"
        else:
            person_id, person_name = self.db.register_new_person(embedding, crop_img=face_crop)
            is_new = True
            confidence_pct = 100.0
            box_color = (255, 120, 0)  # Blue/Cyan for new registration
            label_prefix = "NEW PERSON REGISTERED"

        detection_records.append({
            "person_id": person_id,
            "name": person_name,
            "bbox": [x, y, w, h],
            "confidence": float(confidence_pct),
            "distance": float(distance),
            "is_new": is_new
        })

        # Draw HUD Box & Text
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 2)
        label_text = f"{person_name} [{person_id}]"
        sub_text = f"{label_prefix} | dist: {distance:.2f}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(label_text, font, 0.55, 2)
        banner_y1 = max(0, y - 40)
        banner_y2 = y
        cv2.rectangle(annotated_frame, (x, banner_y1), (x + max(text_w + 10, 200), banner_y2), (20, 20, 20), -1)
        cv2.rectangle(annotated_frame, (x, banner_y1), (x + max(text_w + 10, 200), banner_y2), box_color, 1)
        
        cv2.putText(annotated_frame, label_text, (x + 5, max(18, y - 22)), font, 0.55, (255, 255, 255), 2)
        cv2.putText(annotated_frame, sub_text, (x + 5, max(34, y - 6)), font, 0.4, box_color, 1)
