import cv2
import numpy as np
import time
import pickle
import os
import math
from insightface.app import FaceAnalysis

FRIEND_MATCH_THRESHOLD = 0.55   # raised from 0.40 -- debug output showed genuine same-person matches
                                 # landing at 0.41-0.49, so 0.40 was rejecting correct matches
TARGET_MATCH_THRESHOLD = 0.60
REAPPEAR_GAP = 6.0          # seconds a face must be gone before a return counts as "2nd appearance"
PROCESS_EVERY_N = 2         # run face detection every Nth frame (perf)
DEBUG = True                # prints the closest match distance for every face, per frame -- use this
                            # to see real distance numbers and dial in MATCH_THRESHOLD for your setup

GALLERY_PATH = "gallery.pkl"   # friends/targets persist here across runs
ALERT_VISIT_COUNT = 3          # siren fires once a target's visit count exceeds this
ALERT_DURATION = 2.0            # seconds the siren flash stays on screen
ALERT_FLICKER_HZ = 4            # flashes per second

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


def best_match(emb, gallery_dict, threshold=None, report_all=False):
    best_id, best_dist = None, 1.0
    for pid, entry in gallery_dict.items():
        for e in entry["embs"]:
            d = cosine_dist(emb, e)
            if d < best_dist:
                best_dist, best_id = d, pid
    if report_all:
        return best_id, best_dist  # caller decides against threshold
    return (best_id, best_dist) if best_dist < threshold else (None, None)


def process_faces(frame, now, on_alert):
    global next_target_id, next_pending_id
    results = []  # list of (box, label)

    for f in app.get(frame):
        box = f.bbox.astype(int)
        emb = f.normed_embedding
        label = None

        # 1. friends take priority
        fid, _ = best_match(emb, friends, threshold=FRIEND_MATCH_THRESHOLD)
        if fid:
            label = fid
            friends[fid]["embs"].append(emb)
            friends[fid]["embs"] = friends[fid]["embs"][-20:]

            # cleanup: if this same face also matches an existing target, that target
            # is a stale duplicate identity for this friend (e.g. created before they
            # were enrolled) -- remove it so it stops competing with the friend match.
            dup_tid, _ = best_match(emb, targets, threshold=TARGET_MATCH_THRESHOLD)
            if dup_tid:
                del targets[dup_tid]
                if DEBUG:
                    print(f"[cleanup] removed duplicate target '{dup_tid}' -- matches friend '{fid}'")

        # 2. existing targets
        if label is None:
            tid, _ = best_match(emb, targets, threshold=TARGET_MATCH_THRESHOLD)
            if tid:
                label = tid
                t = targets[tid]
                is_new_visit = (now - t["last_seen"]) >= REAPPEAR_GAP
                if is_new_visit:
                    t["visits"] += 1
                    t["alerted_this_visit"] = False
                t["embs"].append(emb)
                t["embs"] = t["embs"][-20:]
                t["last_seen"] = now

                if t["visits"] > ALERT_VISIT_COUNT and not t["alerted_this_visit"]:
                    t["alerted_this_visit"] = True
                    on_alert(tid, box)

        # 3. pending candidates (awaiting a genuine 2nd appearance)
        if label is None:
            pid, _ = best_match(emb, pending, threshold=TARGET_MATCH_THRESHOLD)
            if pid:
                elapsed = now - pending[pid]["last_seen"]
                if elapsed >= REAPPEAR_GAP:
                    tid = f"target_{next_target_id}"
                    next_target_id += 1
                    targets[tid] = {
                        "embs": pending[pid]["embs"] + [emb],
                        "last_seen": now,
                        "visits": 2,  # first appearance (pending) + this return = 2 visits
                        "alerted_this_visit": False,
                    }
                    del pending[pid]
                    label = tid
                else:
                    pending[pid]["embs"].append(emb)
                    pending[pid]["embs"] = pending[pid]["embs"][-20:]
                    pending[pid]["last_seen"] = now

        # 4. brand new face -> silently enroll, no label yet
        if label is None:
            new_pid = f"cand_{next_pending_id}"
            next_pending_id += 1
            pending[new_pid] = {"embs": [emb], "last_seen": now}

        if DEBUG:
            _, fd = best_match(emb, friends, report_all=True)
            _, td = best_match(emb, targets, report_all=True)
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


def draw_water_balloon_splash(frame, box, elapsed):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    face_w = max(x2 - x1, 20)

    t = min(elapsed / ALERT_DURATION, 1.0)   # 0 -> 1 over the alert's life
    grow = min(t * 3, 1.0)                    # splash grows fast, then holds
    fade = max(0.0, 1.0 - t)                  # fades out over the full duration

    max_radius = int(face_w * 1.3)
    radius = max(4, int(max_radius * grow))
    color = (235, 180, 40)  # BGR - light blue "water" splash

    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius, color, -1)

    # a ring of droplets flying outward from the splash center
    for i in range(8):
        angle = math.radians(i * (360 / 8))
        dx = int((radius * 1.4) * math.cos(angle))
        dy = int((radius * 1.4) * math.sin(angle))
        drop_r = max(3, radius // 6)
        cv2.circle(overlay, (cx + dx, cy + dy), drop_r, color, -1)

    cv2.addWeighted(overlay, 0.55 * fade, frame, 1 - 0.55 * fade, 0, frame)
    cv2.putText(frame, "SPLASH!", (x1, max(30, y1 - 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    last_detections = []
    alert_active_id = None
    alert_active_box = None
    alert_started_at = 0.0

    def trigger_alert(target_id, box):
        nonlocal alert_active_id, alert_active_box, alert_started_at
        alert_active_id = target_id
        alert_active_box = box
        alert_started_at = time.time()
        print(f"\a[ALERT] {target_id} has now appeared {targets[target_id]['visits']} times!")

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

        if alert_active_id is not None:
            elapsed = now - alert_started_at
            if elapsed <= ALERT_DURATION:
                draw_water_balloon_splash(frame, alert_active_box, elapsed)
            else:
                alert_active_id = None
                alert_active_box = None

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
