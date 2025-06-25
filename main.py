import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import threading
from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio

ws_clients = set()
# Al inicio del script, define el event loop global
event_loop = asyncio.new_event_loop()
asyncio.set_event_loop(event_loop)

# Lanza el loop en segundo plano
threading.Thread(target=event_loop.run_forever, daemon=True).start()

# Constants
touch_threshold_mm = 300  # virtual "touch" distance threshold

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   

# RealSense setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

# Frame buffer
lock = threading.Lock()
frame_bytes = b''

def get_3d_point(depth_frame, x, y):
    # Clamp coordinates to avoid out-of-bounds errors
    height, width = depth_frame.get_height(), depth_frame.get_width()
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))

    depth = depth_frame.get_distance(x, y)
    point = rs.rs2_deproject_pixel_to_point(
        profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics(),
        [x, y],
        depth
    )
    return point, depth


def detect_points_withtouch(depth_frame, hand_landmarks_list):
    touch_points = []
    for hand_index, hand_landmarks in enumerate(hand_landmarks_list):
        for idx in [8]:  # Index fingertip
            lm = hand_landmarks.landmark[idx]
            x_px, y_px = int(lm.x * 640), int(lm.y * 480)
            point_3d, depth = get_3d_point(depth_frame, x_px, y_px)
            is_touch = 0 < depth * 1000 < touch_threshold_mm

            touch_points.append({
                "id": f"hand-{hand_index}:index",
                "name": "Index Tip",
                "bodypart": "hand",
                "x": point_3d[0],
                "y": point_3d[1],
                "z": point_3d[2],
                "is_touching": is_touch,
                "2d_x_px":  x_px,
                "2d_y_px":  y_px,
                "2d_depth": depth,

            })
    return touch_points     

def camera_loop():
    global frame_bytes
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            points_withtouch = detect_points_withtouch(depth_frame, results.multi_hand_landmarks)
            for point in points_withtouch:
                # Draw fingertip
                color = (0, 255, 0) if point["is_touching"] else (0, 0, 255)
                x_px = point["2d_x_px"]
                y_px = point["2d_y_px"]
                cv2.circle(color_image, (x_px, y_px), 12, color, -1)
            #asyncio.run(broadcast_touch_points(points_withtouch))
            asyncio.run_coroutine_threadsafe(broadcast_touch_points(points_withtouch), event_loop)
        # Encode frame
        _, jpeg = cv2.imencode('.jpg', color_image)
        with lock:
            frame_bytes = jpeg.tobytes()

async def broadcast_touch_points(points):
    if ws_clients:
        message = json.dumps(points)
        disconnected = set()
        for ws in list(ws_clients):  # 👈 Esto evita el error
            try:
                await ws.send_text(message)
            except:
                disconnected.add(ws)
        for ws in disconnected:
            ws_clients.remove(ws)

# Start camera thread
threading.Thread(target=camera_loop, daemon=True).start()

@app.get("/")
def index():
    return {"message": "Go to /video for touchless UI stream."}

@app.get("/video")
def video():
    def generate():
        while True:
            with lock:
                if frame_bytes:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/touches")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # Optional: keep alive
    except WebSocketDisconnect:
        ws_clients.remove(websocket)