import cv2
import numpy as np
import time
import pickle
import os
import math
import subprocess
import threading
from insightface.app import FaceAnalysis

FRIEND_MATCH_THRESHOLD = 0.55   # raised from 0.40 -- debug output showed genuine same-person matches
                                 # landing at 0.41-0.49, so 0.40 was rejecting correct matches
TARGET_MATCH_THRESHOLD = 0.60
REAPPEAR_GAP = 6.0          # seconds a face must be gone before a return counts as "2nd appearance"
PROCESS_EVERY_N = 2         # run face detection every Nth frame (perf)
DEBUG = True                # prints the closest match distance for every face, per frame -- use this
                            # to see real distance numbers and dial in MATCH_THRESHOLD for your setup

GALLERY_PATH = "gallery.pkl"   # friends/targets persist here across runs
ALERT_VISIT_COUNT = 3          # phone alert fires once a target's visit count exceeds this
KDECONNECT_DEVICE_ID = "7eb954b051f1476ca23fefd24386c8d4"   # Nothing Phone 1

# ---- Model setup (trimmed to just what we need, smaller det size = faster) ----
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
    allowed_modules=["detection", "recognition"],
)
app.prepare(ctx_id=0, det_size=(320, 320))

# ---- Identity store: loaded from disk so friends/targets persist across runs ----
if os.path.exists(GALLERY_PATH):
    with open(GALLERY_PATH, "rb") as fh:
        _saved = pickle.load(fh)
    friends = _saved.get("friends", {})
    pending = _saved.get("pending", {})
    targets = _saved.get("targets", {})
    print(f"Loaded {len(friends)} friend(s), {len(targets)} target(s), {len(pending)} pending candidate(s).")
else:
    friends = {}   # name -> {"embs": [...]}
    pending = {}   # cand_N -> {"embs": [...], "last_seen": t}
    targets = {}   # target_N -> {"embs": [...], "last_seen": t, "visits": int, "alerted_this_visit": bool}

next_target_id = max([int(k.split("_")[1]) for k in targets], default=0) + 1
next_pending_id = max([int(k.split("_")[1]) for k in pending], default=0) + 1


def save_gallery():
    with open(GALLERY_PATH, "wb") as fh:
        pickle.dump({"friends": friends, "pending": pending, "targets": targets}, fh)


def cosine_dist(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return 1 - np.dot(a, b)


def best_match(emb, gallery_dict):
    """Returns (closest_id, closest_dist) regardless of threshold -- caller decides what counts as a match."""
    best_id, best_dist = None, 1.0
    for pid, entry in gallery_dict.items():
        for e in entry["embs"]:
            d = cosine_dist(emb, e)
            if d < best_dist:
                best_dist, best_id = d, pid
    return best_id, best_dist


def process_faces(frame, now, on_alert):
    global next_target_id, next_pending_id
    results = []  # list of (box, label)

    for f in app.get(frame):
        box = f.bbox.astype(int)
        emb = f.normed_embedding
        label = None

        # single scan per gallery -- reused below for both matching and debug output
        fid_raw, fd = best_match(emb, friends)
        tid_raw, td = best_match(emb, targets)

        # 1. friends take priority
        if fid_raw is not None and fd < FRIEND_MATCH_THRESHOLD:
            label = fid_raw
            friends[label]["embs"].append(emb)
            friends[label]["embs"] = friends[label]["embs"][-20:]

            # cleanup: if this same face also matches an existing target, that target
            # is a stale duplicate identity for this friend -- remove it.
            if tid_raw is not None and td < TARGET_MATCH_THRESHOLD:
                del targets[tid_raw]
                if DEBUG:
                    print(f"[cleanup] removed duplicate target '{tid_raw}' -- matches friend '{label}'")

        # 2. existing targets
        if label is None and tid_raw is not None and td < TARGET_MATCH_THRESHOLD:
            label = tid_raw
            t = targets[label]
            is_new_visit = (now - t["last_seen"]) >= REAPPEAR_GAP
            if is_new_visit:
                t["visits"] += 1
                t["alerted_this_visit"] = False
            t["embs"].append(emb)
            t["embs"] = t["embs"][-20:]
            t["last_seen"] = now

            if t["visits"] > ALERT_VISIT_COUNT and not t["alerted_this_visit"]:
                t["alerted_this_visit"] = True
                on_alert(label, box)

        # 3. pending candidates (awaiting a genuine 2nd appearance)
        if label is None:
            pid_raw, pd = best_match(emb, pending)
            if pid_raw is not None and pd < TARGET_MATCH_THRESHOLD:
                elapsed = now - pending[pid_raw]["last_seen"]
                if elapsed >= REAPPEAR_GAP:
                    tid = f"target_{next_target_id}"
                    next_target_id += 1
                    targets[tid] = {
                        "embs": pending[pid_raw]["embs"] + [emb],
                        "last_seen": now,
                        "visits": 2,  # first appearance (pending) + this return = 2 visits
                        "alerted_this_visit": False,
                    }
                    del pending[pid_raw]
                    label = tid
                else:
                    pending[pid_raw]["embs"].append(emb)
                    pending[pid_raw]["embs"] = pending[pid_raw]["embs"][-20:]
                    pending[pid_raw]["last_seen"] = now

        # 4. brand new face -> silently enroll, no label yet
        if label is None:
            new_pid = f"cand_{next_pending_id}"
            next_pending_id += 1
            pending[new_pid] = {"embs": [emb], "last_seen": now}

        if DEBUG:
            print(f"[debug] label={label}  closest_friend_dist={fd:.3f}  closest_target_dist={td:.3f}")

        results.append((box, label))

    return results


def enroll_friend(frame):
    faces = app.get(frame)
    if not faces:
        print("No face detected to enroll.")
        return
    biggest = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    name = input("Friend's name: ").strip()
    if name:
        if name in friends:
            friends[name]["embs"].append(biggest.normed_embedding)
            friends[name]["embs"] = friends[name]["embs"][-20:]
        else:
            friends[name] = {"embs": [biggest.normed_embedding]}
        print(f"Enrolled '{name}' as a friend. ({len(friends[name]['embs'])} sample(s) stored)")


def draw(frame, detections):
    for box, label in detections:
        if label in friends:
            color = (0, 255, 0)   # green - friend
        elif label in targets:
            color = (0, 0, 255)   # red - target
        else:
            color = (200, 200, 200)  # gray - unlabeled / still pending

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        if label:
            cv2.putText(frame, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def send_phone_alert(target_id, visits):
    def _worker():
        message = f"{target_id} has appeared {visits} times"
        cmd = ["kdeconnect-cli", "--ping-msg", message]
        if KDECONNECT_DEVICE_ID:
            cmd += ["-d", KDECONNECT_DEVICE_ID]
        try:
            subprocess.run(cmd, check=True, timeout=5)
        except Exception as e:
            print(f"[warn] Failed to send phone alert: {e}")

    threading.Thread(target=_worker, daemon=True).start()


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    last_detections = []

    def trigger_alert(target_id, box):
        visits = targets[target_id]["visits"]
        print(f"\a[ALERT] {target_id} has now appeared {visits} times! Sending phone alert...")
        send_phone_alert(target_id, visits)

    print("Press 'f' to enroll the largest face in frame as a friend.")
    print("Press 'q' to quit (progress is saved automatically).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = time.time()

        if frame_count % PROCESS_EVERY_N == 0:
            last_detections = process_faces(frame, now, trigger_alert)

        draw(frame, last_detections)
        cv2.imshow("Person Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            enroll_friend(frame)

    cap.release()
    cv2.destroyAllWindows()
    save_gallery()
    print(f"Saved {len(friends)} friend(s) and {len(targets)} target(s) to {GALLERY_PATH}.")


if __name__ == "__main__":
    main()
