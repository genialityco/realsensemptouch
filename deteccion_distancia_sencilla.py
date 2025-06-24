# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pyrealsense2 as rs
import numpy as np
import threading
import time

app = FastAPI()

# Permitir peticiones desde el frontend (Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

touch_detected = 0

def realsense_thread():
    global touch_detected


    # Crear un contexto de RealSense
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("❌ No hay cámaras RealSense conectadas.")
    else:
        print(f"✅ {len(devices)} cámara(s) RealSense detectada(s):")
        for i, dev in enumerate(devices):
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            print(f"  📷 {i+1}. Modelo: {name} - Serial: {serial}")


    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            if not depth:
                continue

            # Analizar si hay un dedo muy cerca al plano virtual
            center_x = 320
            center_y = 240
            distance = depth.get_distance(center_x, center_y)

            # Detección simple: si el dedo está a < 10cm
            touch_detected = distance # < 0.1
            print(f"La distancia es {touch_detected} metros")
            time.sleep(0.1)

    finally:
        pipeline.stop()

# Hilo separado para correr la cámara
threading.Thread(target=realsense_thread, daemon=True).start()

@app.get("/touch")
def get_touch_status():
    return {"touch": touch_detected}
