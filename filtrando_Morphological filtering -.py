#poetry run pip install pyrealsense2
# Otras plataformasU-Touch, Touché, RealTouch
# RealTouch / UTouch	IR + clusters + smoothing + calibración
# ReactiVision	Fiduciales + cámara + homografía
# Touché (Disney)	Audio + capacitancia profunda
# Leap Motion	AI de mano + profundidad
import pyrealsense2 as rs
import numpy as np
import cv2
from fastapi import FastAPI
from fastapi import Query
from fastapi.responses import StreamingResponse
import threading
from collections import deque
import io
from scipy.spatial import KDTree

app = FastAPI()
latest_frame = None
frame_count = 0
diagnostic_mode = 2  # 0=color, 1=depth colormap, 2=detección
# Global deque to track recent valid detections
recent_detections = deque(maxlen=4)  # Store last 5 frames


def is_similar_cluster(new_points, old_points, max_dist=30):
    if len(new_points) == 0 or len(old_points) == 0:
        return False

    # Compute mean positions
    new_mean = np.mean(new_points, axis=0)
    old_mean = np.mean(old_points, axis=0)
    distance = np.linalg.norm(new_mean - old_mean)
    return distance < max_dist

def detect_close_objects(depth_frame, threshold=300):
    global frame_count
    frame_count += 1    
    # Convertir a arreglo de numpy
    depth_image = np.asanyarray(depth_frame.get_data())
    #pixel with no data might appear to be super close (0 mm) — causing false detection unless handled.
    # Aplica la máscara directamente sobre toda la imagen
    mask = (depth_image > 0) & (depth_image < threshold)

    #Step 1: Morphological filtering - Removes isolated white pixels in the mask - Keeps only clusters
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    points_without_morph_filer = np.argwhere(mask)
    points = np.argwhere(clean_mask)
    valid_pixels = depth_image[mask]

   #1) Morphological filtering 2) Cluster density check   

    points_without_temporal_filter = np.argwhere(clean_mask)
    # # Step 3: Soft temporal smoothing with KDTree  — retain only stable points 
    
    # recent_detections.append(points)
    # required_matches = 2
    # proximity_threshold = 40
    # subsample_step = 30

    # if len(recent_detections) == recent_detections.maxlen and len(points) > 0:
    #     # Build KDTree of recent points
    #     all_recent = np.vstack([p for p in recent_detections if len(p) > 0])
    #     if len(all_recent) > 0:
    #         tree = KDTree(all_recent)

    #         # Filter points: only keep those near previous points
    #         stable_points = []
    #         for pt in points[::subsample_step]:  # Subsample to speed up
    #             neighbors = tree.query_ball_point(pt, r=proximity_threshold)
    #             if len(neighbors) >= required_matches:
    #                 stable_points.append(pt)

    #         points = np.array(stable_points)
    #    #print(f"🧷 {len(points)} puntos estables tras suavizado temporal")


# Print every 30 frames (~once per second if 30 FPS)
    if frame_count % 90 == 0:
        invalid_pixels = np.count_nonzero(depth_image == 0)
        if invalid_pixels < depth_image.size:
            if valid_pixels.size > 0:
             min_distance_mm = valid_pixels.min()
            else:
             min_distance_mm = -1
            print(f"⚠️ Pixels sin profundidad: {invalid_pixels} de {depth_image.size}")
            print(f"📏 Distancia mínima válida: {min_distance_mm:.3f} mm")
            print(f"🔍 {len(points)} puntos detectados debajo de {threshold} mm, sin filtro temporal, {len(points_without_temporal_filter)}, sin filtro mopth {len(points_without_morph_filer)}")
        else:
            print("⚠️ No hay datos válidos de profundidad")

    return points, mask



def start_camera():
    global latest_frame, diagnostic_mode

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        points, mask = detect_close_objects(depth_frame, threshold=250)

        # Modo 0: imagen de color
        if diagnostic_mode == 0:
            pass  # ya es color_image

        # Modo 1: depth en colormap
        elif diagnostic_mode == 1:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            color_image = depth_colormap

        # Modo 2: detección de objetos cercanos
        elif diagnostic_mode == 2:
            for y, x in points[::500]:
                cv2.circle(color_image, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(color_image, f"{x},{y}", (x + 5, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        _, buffer = cv2.imencode('.jpg', color_image)
        latest_frame = buffer.tobytes()


threading.Thread(target=start_camera, daemon=True).start()


from fastapi.responses import StreamingResponse
import time

def gen_frames():
    global latest_frame
    while True:
        if latest_frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        time.sleep(0.03)

@app.get("/video")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/diagnostic_mode/set")
def set_diagnostic_mode(mode: int = Query(..., ge=0, le=2)):
    global diagnostic_mode
    diagnostic_mode = mode
    return {"message": f"🔁 Modo cambiado a {diagnostic_mode}"}