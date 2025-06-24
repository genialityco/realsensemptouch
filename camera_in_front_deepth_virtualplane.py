import pyrealsense2 as rs
import numpy as np
import cv2
from fastapi import FastAPI, Query,WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import threading
from collections import deque
import time


app = FastAPI()
clients = set()

# Allow your Vite dev server to connect
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

diagnostic_mode = 2  
#diagnostic_mode = 1  # colorized depth
#diagnostic_mode = 2  # touch circles
#diagnostic_mode = 3  # raw mask with contours
recent_masks = deque(maxlen=3)  # Save binary masks for temporal smoothing


from touch_tracker import TouchTracker
touch_tracker = TouchTracker()  # Persistent across calls


def detect_close_objects(depth_frame, threshold=250):
    """
    Detects close touch-like blobs from a depth frame and returns tracked points.

    Args:
        depth_frame: pyrealsense2 depth frame
        threshold: Depth (in mm) to consider "close" to screen

    Returns:
        touch_points: List of dicts with {id, position, radius, label}
        final_mask: Binary image showing all accepted blob regions
    """
    global recent_masks

    # Convert depth to numpy array
    depth_image = np.asanyarray(depth_frame.get_data())

    # STEP 1: Create binary mask of valid "close" points
    mask = (depth_image > 0) & (depth_image < threshold)

    # STEP 2: Morphological cleanup to remove small gaps/noise
    kernel = np.ones((10, 10), np.uint8)
    clean_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # STEP 3: Neighborhood depth consistency (median filter)
    mask_float = depth_image.astype(np.float32)
    local_median = cv2.medianBlur(mask_float, 5)
    clean_mask = clean_mask & (local_median < threshold)

    # STEP 4: Temporal averaging (across last N frames)
    recent_masks.append(clean_mask)
    if len(recent_masks) > 3:
        recent_masks.pop(0)
    avg_mask = np.mean(recent_masks, axis=0)
    stable_mask = (avg_mask > 0.7).astype(np.uint8)

    # STEP 5: Connected components to group points
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(stable_mask, connectivity=8)

    # STEP 6: Extract raw components from blobs
    min_area = 30
    raw_touches = []

    for label in range(1, num_labels):  # skip background (label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cx, cy = centroids[label]
            radius = int(np.sqrt(area / np.pi))  # area ≈ πr²
            raw_touches.append({
                "position": [int(cy), int(cx)],  # [y, x]
                "radius": radius,
                "label": label  # Keep original label for optional visualization
            })

    # STEP 7: Use TouchTracker to assign stable IDs over time
    tracked_touches = touch_tracker.update(raw_touches)

    # STEP 8: Compose final output with both ID and label
    touch_points = []
    for tracked in tracked_touches:
        touch_points.append({
            "id": tracked["id"],                     # Stable across frames
            "position": [tracked["y"], tracked["x"]], # React-compatible [y, x]
            "radius": tracked["radius"],
            "label": tracked.get("label", -1)        # Original label (optional)
        })

    # STEP 9: Reconstruct mask for visualization
    final_mask = np.zeros_like(stable_mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            final_mask[labels == label] = 1

    return touch_points, final_mask




def update_touch_data(touch_points):
    global latest_touch_points
    latest_touch_points = [
        {"x": t["position"][1], "y": t["position"][0], "radius": t["radius"],  "id": t["id"]}
        for t in touch_points
    ]

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

        touch_points, stable_mask = detect_close_objects(depth_frame, threshold=400)
        update_touch_data(touch_points)

        # Diagnostic mode 1: colorized depth
        if diagnostic_mode == 1:
            color_image = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

        # Diagnostic mode 2: draw touch circles with size
        elif diagnostic_mode == 2:
            for touch in touch_points:
                x, y = touch["position"][1], touch["position"][0]
                radius = touch["radius"]
                cv2.circle(color_image, (x, y), radius, (0, 0, 255), 2)
                cv2.putText(color_image, f"{radius}", (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # ✅ Diagnostic mode 3: show raw mask + contours
        elif diagnostic_mode == 3:
            # Convert binary mask to 3-channel for color drawing
            mask_visual = cv2.cvtColor((stable_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Draw contours and centers on top
            for touch in touch_points:
                contour = touch["contour"]
                x, y = touch["position"][1], touch["position"][0]
                radius = touch["radius"]

                cv2.drawContours(mask_visual, [contour], -1, (0, 255, 0), 2)
                cv2.circle(mask_visual, (x, y), 3, (0, 0, 255), -1)
                cv2.putText(mask_visual, f"{radius}", (x+4, y-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            color_image = mask_visual

        # Encode image to JPEG
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
            await asyncio.sleep(0.033)  # 30 FPS
            if latest_touch_points:
                await websocket.send_text(json.dumps(latest_touch_points))
    except:
        clients.remove(websocket)

@app.get("/diagnostic_mode/set")
def set_diagnostic_mode(mode: int = Query(..., ge=0, le=3)):
    global diagnostic_mode
    diagnostic_mode = mode
    return {"message": f"🔁 Modo cambiado a {diagnostic_mode}"}
