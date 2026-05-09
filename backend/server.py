from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import time
import yaml
import torch
import json
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

from data_collection.mediapipe_runner import MediaPipeRunner
from models.kslr_net import KSLRNet
from realtime.webcam_pipeline import WebcamPipeline
from utils.checkpoint import load_checkpoint

app = FastAPI()

# Add CORS middleware to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = yaml.safe_load(Path('configs/lab_dataset.yaml').read_text(encoding='utf-8'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading model on {device}...")
model = KSLRNet(cfg)
load_checkpoint('runs/kslr_lab_v0/best.pt', model=model, map_location=device)
model.to(device).eval()

print("Initializing MediaPipeRunner...")
runner = MediaPipeRunner(
    cfg['mediapipe']['hand']['model_asset_path'],
    cfg['mediapipe']['face']['model_asset_path'],
)

# Active WebSocket connections
SESSIONS = {}

@app.websocket("/ws/{session_id}")
async def ws_predict(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"Client connected: {session_id}")
    pipeline = WebcamPipeline(model, runner, cfg, device)
    SESSIONS[session_id] = pipeline
    t0 = time.monotonic()
    
    try:
        while True:
            # We expect the frontend to send a JSON indicating close or binary image data
            # First, check what kind of data is received
            # We'll stick to receiving bytes for the frame
            data = await websocket.receive_bytes()
            
            # Decode the image bytes
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
            if bgr is None:
                continue
            
            ts_ms = int((time.monotonic() - t0) * 1000)
            
            # Run inference
            result, prediction = pipeline.step_frame(bgr, ts_ms)
            
            if prediction is not None:
                # Send the result back to the frontend
                await websocket.send_json({
                    'label_id': prediction.label_id,
                    'confidence': prediction.confidence,
                })
                
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"Error in websocket for {session_id}: {e}")
    finally:
        if session_id in SESSIONS:
            del SESSIONS[session_id]
