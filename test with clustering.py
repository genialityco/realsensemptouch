import pyrealsense2 as rs
import numpy as np
import cv2
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
import threading
from collections import deque
import time

app = FastAPI()
latest_frame = None
frame_count = 0

diagnostic_mode = 2  # 0=color, 1=depth colormap, 2=touch detection
recent_centroids = deque(maxlen=4)
recent_masks = deque(maxlen=3)

# Homography placeholder (identity by default)
homography_matrix = np.eye(3)


def detect_touch_points(depth_frame, threshold=250):
    global frame_count, recent_centroids, recent_masks
    frame_count += 1

    depth_image = np.asanyarray(depth_frame.get_data())
    mask = (depth_image > 0) & (depth_image < threshold)

    # Morphological filtering
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # Temporal smoothing with past masks
    recent_masks.append(clean_mask)
    if len(recent_masks) == recent_masks.maxlen:
        stable_mask = recent_masks[0].copy()
        for m in list(recent_masks)[1:]:
            stable_mask = cv2.bitwise_and(stable_mask, m)
    else:
        stable_mask = clean_mask

    # Connected components = clusters
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(stable_mask)
    valid_centroids = []

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if 20 < area < 500:  # filter noise and large blobs
            valid_centroids.append(centroids[i])

    # EMA smoothing
    smoothed_centroids = []
    alpha = 0.4
    for c in valid_centroids:
        if recent_centroids:
            prev = recent_centroids[-1]
            distances = [np.linalg.norm(c - p) for p in prev]
            min_idx = np.argmin(distances)
            smoothed = alpha * c + (1 - alpha) * prev[min_idx]
        else:
            smoothed = c
        smoothed_centroids.append(smoothed)

    if valid_centroids:
        recent_centroids.append(valid_centroids)

    # Logging
    if frame_count % 90 == 0:
        print(f"📍 {len(smoothed_centroids)} puntos de toque detectados")

    return np.array(smoothed_centroids), depth_image


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

        points, _ = detect_touch_points(depth_frame)

        if diagnostic_mode == 1:
            color_image = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
        elif diagnostic_mode == 2:
            for x, y in points.astype(int):
                # Warp (homografía, aún identidad)
                warped_point = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), homography_matrix)[0][0]
                cv2.circle(color_image, tuple(warped_point.astype(int)), 6, (0, 255, 0), -1)
                cv2.putText(color_image, f"{int(warped_point[0])},{int(warped_point[1])}",
                            (int(warped_point[0] + 5), int(warped_point[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        _, buffer = cv2.imencode('.jpg', color_image)
        latest_frame = buffer.tobytes()


threading.Thread(target=start_camera, daemon=True).start()


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
