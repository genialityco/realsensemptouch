import numpy as np
from scipy.spatial.distance import cdist

class TouchTracker:
    def __init__(self, max_distance=40, max_lost=3):
        self.tracks = {}  # id → {'pos': (y, x), 'radius': r, 'age': n, 'lost': 0}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_lost = max_lost

    def update(self, raw_touches):
        current_centroids = np.array([t["position"] for t in raw_touches])
        current_radii = np.array([t["radius"] for t in raw_touches])
    
        matched_ids = set()
        updated_tracks = {}

        if len(self.tracks) > 0 and len(current_centroids) > 0:
            previous_centroids = np.array([v["pos"] for v in self.tracks.values()])
            prev_ids = list(self.tracks.keys())

            dist_matrix = cdist(previous_centroids, current_centroids)
            for i, prev_id in enumerate(prev_ids):
                j = np.argmin(dist_matrix[i])
                if dist_matrix[i, j] < self.max_distance:
                    new_pos = tuple(current_centroids[j])
                    new_radius = current_radii[j]
                 
                    updated_tracks[prev_id] = {
                        "pos": tuple(0.7*np.array(self.tracks[prev_id]["pos"]) + 0.3*np.array(new_pos)),
                        "radius": 0.7 * self.tracks[prev_id]["radius"] + 0.3 * new_radius,
                        "age": self.tracks[prev_id]["age"] + 1,
                        "lost": 0,
                       
                    }
                    matched_ids.add(j)

        # Add new unmatched touches
        for i, t in enumerate(raw_touches):
            if i not in matched_ids:
                new_id = self.next_id
                self.next_id += 1
                updated_tracks[new_id] = {
                    "pos": tuple(t["position"]),
                    "radius": t["radius"],
                    "age": 1,
                    "lost": 0
                }

        # Handle disappeared tracks
        for tid, track in self.tracks.items():
            if tid not in updated_tracks:
                if track["lost"] < self.max_lost:
                    track["lost"] += 1
                    updated_tracks[tid] = track  # keep it for now

        self.tracks = updated_tracks

        # Only return active, aged tracks
        touches = []
        for tid, t in self.tracks.items():
            if t["age"] >= 2 and t["lost"] == 0:
                touches.append({
                    "id": f"touch-{tid}",
                    "x": int(t["pos"][1]),
                    "y": int(t["pos"][0]),
                    "radius": int(t["radius"]),
                    "label": t.get("label", -1)
                })

        return touches
