import pyrealsense2 as rs
import numpy as np
import cv2
from fastapi import FastAPI, Query, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import threading
from collections import deque
import time

app = FastAPI()
clients = set()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

latest_frame = None
frame_count = 0
latest_touch_points = []

diagnostic_mode = 2  # default
recent_masks = deque(maxlen=3)
target_y = 0
epsilon = 0.015

from touch_tracker import TouchTracker
touch_tracker = TouchTracker()

def detect_touch_by_y_plane(depth_frame, depth_intrinsics, target_y=0.0, epsilon=0.02):
    global recent_masks
    depth_image = np.asanyarray(depth_frame.get_data())
    height, width = depth_image.shape

    mask_touch = np.zeros_like(depth_image, dtype=np.uint8)
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            depth = depth_image[y, x] * 0.001
            if depth == 0:
                continue
            point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
            point_y = point[1]


            if abs(point_y - target_y) <= epsilon:
                mask_touch[y, x] = 1

    mask_touch = cv2.dilate(mask_touch, np.ones((3, 3), np.uint8), iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(mask_touch, cv2.MORPH_OPEN, kernel)

    recent_masks.append(clean_mask)
    if len(recent_masks) > 3:
        recent_masks.pop(0)
    avg_mask = np.mean(recent_masks, axis=0)
    stable_mask = (avg_mask > 0.7).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(stable_mask, connectivity=4)

    min_area = 5
    raw_touches = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cx, cy = centroids[label]
            radius = int(np.sqrt(area / np.pi))
            raw_touches.append({
                "position": [int(cy), int(cx)],
                "radius": radius,
                "label": label
            })

    tracked_touches = touch_tracker.update(raw_touches)
    touch_points = []
    for tracked in tracked_touches:
        x = tracked["x"]
        y = tracked["y"]
        depth_mm = depth_image[y, x]
        depth_m = depth_mm * 0.001
        point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_m)

        touch_points.append({
            "id": tracked["id"],
            "position": [y, x],
            "radius": tracked["radius"],
            "label": tracked.get("label", -1),
            "depth": round(depth_m, 4),
            "x_virtual": round(point[0], 4),  # <- invertir eje X
            "y_virtual": round(depth_m, 4),
            "z_virtual": round(point[2], 4)
        })

    final_mask = np.zeros_like(stable_mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            final_mask[labels == label] = 1

    return touch_points, final_mask

def update_touch_data(touch_points):
    global latest_touch_points
    latest_touch_points = [
        {
            "id": t["id"],
            "x": t.get("x_virtual", None),
            "y": t.get("y_virtual", None),
            "z": t.get("z_virtual", None),
            "radius": t["radius"]
        }
        for t in touch_points
    ]
    # latest_touch_points = [
    #     {"x": t["position"][1], "y": t["position"][0], "radius": t["radius"], "id": t["id"],"depth": t.get("depth", None)   }
    #     for t in touch_points
    # ]

def start_camera():
    global latest_frame, diagnostic_mode, target_y, epsilon
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
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        touch_points, stable_mask = detect_touch_by_y_plane(depth_frame, depth_intrinsics, target_y=target_y, epsilon=epsilon)
        update_touch_data(touch_points)

        if diagnostic_mode == 1:
            color_image = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

        elif diagnostic_mode == 2:
            for touch in touch_points:
                x, y = touch["position"][1], touch["position"][0]
                radius = touch["radius"]
                cv2.circle(color_image, (x, y), radius, (0, 0, 255), 2)
                cv2.putText(color_image, f"{radius}", (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            # Overlay Y-threshold for touch
            height, width = depth_image.shape
            for y in range(0, height, 2):
                for x in range(0, width, 2):
                    depth = depth_image[y, x] * 0.001
                    if depth == 0:
                        continue
                    point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
                    point_y = point[1]
                    if abs(point_y - target_y) <= epsilon:
                        color_image[y, x] = (0, 255, 255)

        elif diagnostic_mode == 3:
            mask_visual = cv2.cvtColor((stable_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            color_image = mask_visual

        elif diagnostic_mode == 4:
            height, width = depth_image.shape
            y_plane_visual = np.zeros((height, width), dtype=np.uint8)
            for y in range(0, height, 2):
                for x in range(0, width, 2):
                    depth = depth_image[y, x] * 0.001
                    if depth == 0:
                        continue
                    point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
                    point_y = point[1]
                    if abs(point_y - target_y) <= epsilon:
                        y_plane_visual[y, x] = 255
            color_image = cv2.applyColorMap(y_plane_visual, cv2.COLORMAP_HOT)

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

@app.websocket("/ws/touches")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(0.033)
            if latest_touch_points:
                await websocket.send_text(json.dumps(latest_touch_points))
    except:
        clients.remove(websocket)

@app.get("/diagnostic_mode/set")
def set_diagnostic_mode(mode: int = Query(..., ge=0, le=4)):
    global diagnostic_mode
    diagnostic_mode = mode
    return {"message": f"\ud83d\udd01 Modo cambiado a {diagnostic_mode}"}

@app.get("/touch_plane/set")
def set_touch_plane(y: float = Query(...), tol: float = Query(0.015)):
    global target_y, epsilon
    target_y = y
    epsilon = tol
    return {"message": f"Y-plane set to {target_y}m ± {epsilon}m"}
