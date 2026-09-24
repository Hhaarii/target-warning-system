import os
import json
import time
import base64
import numpy as np
import cv2
import config

class IdentityDatabase:
    def __init__(self, db_file=config.DB_FILE, profiles_dir=config.PROFILES_DIR):
        self.db_file = db_file
        self.profiles_dir = profiles_dir
        self.identities = {}
        self.load_database()

    def load_database(self):
        """Load registered identities and feature embeddings from disk."""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r") as f:
                    data = json.load(f)
                    for person_id, item in data.items():
                        item['embedding'] = np.array(item['embedding'], dtype=np.float32)
                        self.identities[person_id] = item
                print(f"[DB] Successfully loaded {len(self.identities)} person profiles.")
            except Exception as e:
                print(f"[DB Error] Failed to load database: {e}")
                self.identities = {}
        else:
            self.identities = {}

    def save_database(self):
        """Persist identities to JSON file."""
        data_to_save = {}
        for person_id, item in self.identities.items():
            item_copy = item.copy()
            if isinstance(item_copy['embedding'], np.ndarray):
                item_copy['embedding'] = item_copy['embedding'].tolist()
            data_to_save[person_id] = item_copy
        
        try:
            with open(self.db_file, "w") as f:
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
            print(f"[DB Error] Failed to save database: {e}")

    def generate_person_id(self):
        """Generate next available sequential person ID (e.g. person_001)."""
        count = 1
        while True:
            pid = f"person_{count:03d}"
            if pid not in self.identities:
                return pid
            count += 1

    def save_thumbnail(self, person_id, crop_img):
        """Save a cropped face/person image thumbnail."""
        if crop_img is None or crop_img.size == 0:
            return ""
        thumb_path = os.path.join(self.profiles_dir, f"{person_id}.jpg")
        try:
            cv2.imwrite(thumb_path, crop_img)
            return thumb_path
        except Exception as e:
            print(f"[DB Error] Failed to save thumbnail for {person_id}: {e}")
            return ""

    def register_new_person(self, embedding, crop_img, default_name=None):
        """Register a brand new person identity with initial embedding vector."""
        person_id = self.generate_person_id()
        name = default_name if default_name else f"Person {person_id.split('_')[-1]}"
        thumb_path = self.save_thumbnail(person_id, crop_img)
        
        # Ensure L2 normalized embedding
        norm_emb = embedding / np.linalg.norm(embedding) if np.linalg.norm(embedding) > 0 else embedding
        
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.identities[person_id] = {
            "id": person_id,
            "name": name,
            "embedding": norm_emb,
            "thumbnail": thumb_path,
            "created_at": now,
            "last_seen": now,
            "sightings": 1
        }
        self.save_database()
        print(f"[DB] Registered NEW Person: {person_id} ('{name}')")
        return person_id, name

    def update_sighting(self, person_id, new_embedding, crop_img=None, alpha=config.EMBEDDING_SMOOTHING):
        """Update last seen timestamp and smooth the feature embedding vector over time."""
        if person_id not in self.identities:
            return
        
        person = self.identities[person_id]
        person['last_seen'] = time.strftime("%Y-%m-%d %H:%M:%S")
        person['sightings'] += 1
        
        # Exponential moving average for feature embedding (handles minor lighting/pose changes)
        if new_embedding is not None and len(new_embedding) > 0:
            norm_new = new_embedding / np.linalg.norm(new_embedding) if np.linalg.norm(new_embedding) > 0 else new_embedding
            old_emb = person['embedding']
            updated_emb = alpha * old_emb + (1.0 - alpha) * norm_new
            person['embedding'] = updated_emb / np.linalg.norm(updated_emb)
        
        # Optionally update thumbnail if image provided
        if crop_img is not None and crop_img.size > 0:
            self.save_thumbnail(person_id, crop_img)
            
        self.save_database()

    def find_match(self, query_embedding, threshold=config.MATCH_THRESHOLD):
        """
        Compare query embedding with database using Cosine Distance.
        Returns: (matched_person_id, matched_name, min_distance, confidence_score)
        """
        if not self.identities or query_embedding is None:
            return None, None, 1.0, 0.0

        # L2 normalize query embedding
        query_norm = query_embedding / np.linalg.norm(query_embedding) if np.linalg.norm(query_embedding) > 0 else query_embedding

        best_id = None
        best_name = None
        min_dist = 1.0

        for person_id, info in self.identities.items():
            db_emb = info['embedding']
            # Cosine similarity = dot product of L2 normalized vectors
            cosine_sim = np.dot(query_norm, db_emb)
            # Cosine distance = 1 - cosine_sim (range 0.0 = identical to 2.0)
            cosine_dist = max(0.0, 1.0 - cosine_sim)
            
            if cosine_dist < min_dist:
                min_dist = cosine_dist
                best_id = person_id
                best_name = info['name']

        if min_dist <= threshold:
            similarity_pct = max(0.0, min(100.0, (1.0 - min_dist) * 100))
            return best_id, best_name, min_dist, similarity_pct

        return None, None, min_dist, 0.0

    def rename_person(self, person_id, new_name):
        """Update display name for a person ID."""
        if person_id in self.identities:
            self.identities[person_id]['name'] = new_name
            self.save_database()
            return True
        return False

    def delete_person(self, person_id):
        """Remove a person identity from database."""
        if person_id in self.identities:
            thumb_path = self.identities[person_id].get('thumbnail')
            if thumb_path and os.path.exists(thumb_path):
                try:
                    os.remove(thumb_path)
                except Exception:
                    pass
            del self.identities[person_id]
            self.save_database()
            return True
        return False

    def get_all_profiles(self):
        """Return clean list of all registered person profiles."""
        profiles = []
        for pid, info in self.identities.items():
            profiles.append({
                "id": info["id"],
                "name": info["name"],
                "created_at": info.get("created_at", ""),
                "last_seen": info.get("last_seen", ""),
                "sightings": info.get("sightings", 0),
                "thumbnail": f"/api/thumbnail/{info['id']}" if os.path.exists(info.get("thumbnail", "")) else ""
            })
        return sorted(profiles, key=lambda x: x['id'])
