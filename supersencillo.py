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

def detect_close_objects(depth_frame, threshold=4000):
    global frame_count, recent_masks
    frame_count += 1

    """
    Detects close objects in a depth frame by applying depth thresholding,
    morphological cleanup, and neighborhood consistency filtering.

    Args:
        depth_frame: pyrealsense2 depth frame
        threshold (int): depth value in mm to consider an object "close"

    Returns:
        points (np.ndarray): Nx2 array of (y, x) coordinates of close objects
        clean_mask (np.ndarray): binary mask of detected regions
    """
    # Convert RealSense frame to numpy array
    depth_image = np.asanyarray(depth_frame.get_data())

    # =====================================
    # Step 1: Binary mask for close objects
    mask = (depth_image > 0) & (depth_image < threshold)
    points = np.argwhere(mask)
    # =====================================
    # Step 2: Morphological opening to remove small noise
    # kernel = np.ones((10, 10), np.uint8)
    # clean_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    # points = np.argwhere(clean_mask)
    # =====================================
    # # Step 3: Neighborhood consistency filter
    # mask_float = depth_image.astype(np.float32)
    # #mean_kernel = np.ones((10, 10), np.float32) / 25
    # #local_mean = cv2.filter2D(mask_float, -1, mean_kernel)
    # local_median = cv2.medianBlur(mask_float, 5)

    # # Keep only regions where both point and neighborhood are below threshold
    # clean_mask = clean_mask & (local_median < threshold)

    # # =====================================
    # # Step 4: Get coordinates of valid points
    # points = np.argwhere(clean_mask)

    # # =====================================
    # # Step 5: Temporal Averaging  keep a rolling average of masks over N frames
    # recent_masks.append(clean_mask)
    # if len(recent_masks) > 3:
    #     recent_masks.pop(0)

    # avg_mask = np.mean(recent_masks, axis=0)
    # stable_mask = (avg_mask > 0.9).astype(np.uint8)
    # points = np.argwhere(stable_mask)

    # # =====================================
    # # Step 6: Connected Components + Area Filtering

    # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(stable_mask.astype(np.uint8), connectivity=8)

    # # Filter by area (remove small blobs)
    # min_area = 35  # adjust as needed
    # final_mask = np.zeros_like(stable_mask, dtype=np.uint8)

    # for i in range(1, num_labels):  # skip label 0 (background)
    #     area = stats[i, cv2.CC_STAT_AREA]
    #     if area >= min_area:
    #         final_mask[labels == i] = 1

    # points = np.argwhere(final_mask)

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

        points, _ = detect_close_objects(depth_frame, threshold=4000)

        if diagnostic_mode == 1:
            color_image = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
        elif diagnostic_mode == 2:
            for y, x in points[::500]:
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
