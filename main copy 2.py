import pyrealsense2 as rs
import numpy as np
import cv2
from fastapi import FastAPI, Query, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import threading
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
latest_touch_points = []
target_y = 0.05  # Distancia a la pantalla (en metros)
tolerance = 0.015

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

profile = pipeline.get_active_profile()
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().intrinsics


def detect_touch_points(depth_frame):
    depth_image = np.asanyarray(depth_frame.get_data())
    height, width = depth_image.shape
    points = []

    for y_px in range(0, height, 4):
        for x_px in range(0, width, 4):
            depth = depth_image[y_px, x_px] * 0.001
            if depth == 0:
                continue

            point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_px, y_px], depth)
            X, Y, Z = point  # RealSense world coords

            if abs(Y - target_y) <= tolerance:
                #print("detectado un punto por debajo del threshold")
                x_virtual = X
                y_virtual = depth  # Altura en mundo virtual
                z_virtual = Z

                points.append({
                    "x": x_virtual,
                    "y": y_virtual,
                    "z": z_virtual,
                    "pixel": [x_px, y_px]
                })

    return points


def start_camera():
    global latest_frame, latest_touch_points
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        touch_points = detect_touch_points(depth_frame)
        latest_touch_points = touch_points

        for pt in touch_points:
            x_px, y_px = pt["pixel"]
            #cv2.circle(color_image, tuple(pt["pixel"]), 5, (0, 0, 255), 2)
            print(f" punto {x_px} {y_px}")
            cv2.circle(color_image, (x_px, y_px), 5, (20, 250, 20), 2)

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
            await websocket.send_text(json.dumps(latest_touch_points))
    except:
        clients.remove(websocket)


@app.get("/touch_plane/set")
def set_touch_plane(y: float = Query(...), tol: float = Query(0.015)):
    global target_y, tolerance
    target_y = y
    tolerance = tol
    return {"message": f"Plano Y de toque actualizado: {target_y}m ± {tolerance}m"}
