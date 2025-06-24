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

diagnostic_mode = 2  # 0=color, 1=depth colormap, 2=detection
recent_masks = deque(maxlen=3)  # Save binary masks for temporal smoothing

def detect_close_objects(depth_frame, threshold=250):
    global frame_count, recent_masks
    frame_count += 1

    depth_image = np.asanyarray(depth_frame.get_data())
    mask = (depth_image > 0) & (depth_image < threshold)
    points = np.argwhere(mask)
    # Step 1: Morphological filtering
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    #points = np.argwhere(clean_mask)
    #Step 2: Temporal smoothing using bitwise AND of past masks
    recent_masks.append(clean_mask)
    if len(recent_masks) == recent_masks.maxlen:
        stable_mask = recent_masks[0].copy()
        for m in list(recent_masks)[1:]:
            stable_mask = cv2.bitwise_and(stable_mask, m)
    else:
        stable_mask = clean_mask

    points = np.argwhere(stable_mask)
    valid_pixels = depth_image[mask]

    # if frame_count % 90 == 0:
    #     invalid_pixels = np.count_nonzero(depth_image == 0)
    #     if invalid_pixels < depth_image.size:
    #         min_distance_mm = valid_pixels.min() if valid_pixels.size > 0 else -1
    #         print(f"⚠️ Sin profundidad: {invalid_pixels}/{depth_image.size}")
    #         print(f"📏 Distancia mínima válida: {min_distance_mm:.3f} mm")
    #         print(f"🔍 {len(points)} puntos detectados bajo {threshold} mm")
    #     else:
    #         print("⚠️ No hay datos válidos de profundidad")

    return points, stable_mask

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

        points, _ = detect_close_objects(depth_frame, threshold=250)

        if diagnostic_mode == 1:
            color_image = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
        elif diagnostic_mode == 2:
            for y, x in points[::300]:
                cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)

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
