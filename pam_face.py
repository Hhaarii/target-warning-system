#!/usr/bin/env python3
"""
PAM Face Recognition Authentication Module
Uses YuNet + SFace with Hybrid Centroid & Top-5 embedding matching for Linux lock screen authentication.
Compatible with pam_exec.so and pam_python.so.
"""

import os
import sys
import time
import cv2
import numpy as np

# Absolute paths so PAM execution (running under GDM/root) loads your user's models & data
DATA_DIR = "/home/ha-r1/face_data"
GALLERY_FILE = os.path.join(DATA_DIR, "gallery.npz")
DETECTOR_MODEL = "/home/ha-r1/face_models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "/home/ha-r1/face_models/face_recognition_sface_2021dec.onnx"

SIMILARITY_THRESHOLD = 0.58
TOP_K_SAMPLES = 5
CONFIRM_FRAMES = 3
MAX_TIMEOUT_SECONDS = 5.0


def authenticate_face():
    """
    Captures camera frames and verifies face identity against admin gallery embeddings.
    Returns True if admin face is verified, False otherwise.
    """
    if not os.path.exists(GALLERY_FILE):
        sys.stderr.write(f"Gallery file missing at {GALLERY_FILE}\n")
        return False

    if not os.path.exists(DETECTOR_MODEL) or not os.path.exists(RECOGNIZER_MODEL):
        sys.stderr.write("SFace or YuNet model files missing.\n")
        return False

    # Load gallery data
    try:
        data = np.load(GALLERY_FILE)
        embeddings = data['embeddings']
        labels = data['labels']
    except Exception as e:
        sys.stderr.write(f"Failed to load gallery: {e}\n")
        return False

    mask = (labels == 'admin')
    admin_embeddings = embeddings[mask]
    if len(admin_embeddings) == 0:
        sys.stderr.write("No admin embeddings found in gallery.\n")
        return False

    # Compute normalized centroid for admin
    centroid = np.mean(admin_embeddings, axis=0, keepdims=True)
    centroid_norm = centroid / np.linalg.norm(centroid)

    detector = cv2.FaceDetectorYN.create(DETECTOR_MODEL, '', (320, 320))
    recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, '')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.stderr.write("Cannot open camera /dev/video0.\n")
        return False

    detector_input_set = False
    consecutive_matches = 0
    start_time = time.time()
    authenticated = False

    try:
        while (time.time() - start_time) < MAX_TIMEOUT_SECONDS:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            if not detector_input_set:
                h, w = frame.shape[:2]
                detector.setInputSize((w, h))
                detector_input_set = True

            _, faces = detector.detect(frame)
            if faces is None or len(faces) == 0:
                consecutive_matches = 0
                continue

            # Pick largest detected face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            aligned = recognizer.alignCrop(frame, largest_face)
            feature = recognizer.feature(aligned)

            # 1. Centroid Cosine Similarity
            centroid_sim = float(recognizer.match(
                feature, centroid_norm, cv2.FaceRecognizerSF_FR_COSINE
            ))

            # 2. Top-K nearest gallery samples
            sims = [
                float(recognizer.match(feature, e.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE))
                for e in admin_embeddings
            ]
            sims_sorted = sorted(sims, reverse=True)
            k_eff = min(TOP_K_SAMPLES, len(sims_sorted))
            top_k_sim = sum(sims_sorted[:k_eff]) / k_eff

            # 3. Hybrid similarity score (60% Centroid + 40% Top-K)
            hybrid_sim = 0.6 * centroid_sim + 0.4 * top_k_sim

            if hybrid_sim >= SIMILARITY_THRESHOLD:
                consecutive_matches += 1
                if consecutive_matches >= CONFIRM_FRAMES:
                    authenticated = True
                    break
            else:
                consecutive_matches = 0

            time.sleep(0.03)
    finally:
        cap.release()

    return authenticated


# PAM Python C-extension interface (if loaded by libpam-python)
def pam_sm_authenticate(pamh, flags, argv):
    PAM_SUCCESS = 0
    PAM_AUTH_ERR = 7
    return PAM_SUCCESS if authenticate_face() else PAM_AUTH_ERR


def pam_sm_setcred(pamh, flags, argv):
    return 0


# Standalone CLI / pam_exec interface
if __name__ == '__main__':
    if authenticate_face():
        sys.exit(0)
    else:
        sys.exit(1)
