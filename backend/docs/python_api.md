# Python API — 외부 앱 통합 가이드

`scripts/demo.py` 의 webcam GUI 가 아닌 **자체 애플리케이션에 모델을 호출**하고 싶을 때 사용합니다.

## 활용 시나리오

| 시나리오 | 이 가이드의 어느 섹션 |
|---|---|
| 데스크톱 앱(PyQt, Tkinter, Electron+IPC) 에 인식 기능 추가 | §2 [실시간 webcam 통합](#2-실시간-webcam-통합) |
| Flask/FastAPI 서버로 인식 API 노출 | §3 [HTTP 서버 패턴](#3-http-서버-패턴) |
| 녹화된 비디오 파일 일괄 인식 | §4 [오프라인 비디오 인식](#4-오프라인-비디오-인식) |
| 단일 프레임 다발(numpy) 인식 | §5 [최저 레벨 호출](#5-최저-레벨-호출--직접-tensor-구성) |

---

## 1. 핵심 API 개요

```
realtime/
├── frame_buffer.py      → FrameBuffer       — 16-frame 순환 버퍼
├── webcam_pipeline.py   → WebcamPipeline    — MediaPipe + 모델을 한 번에 묶음
└── demo_app.py          → render_demo_frame — 화면 overlay (선택)

models/
└── kslr_net.py          → KSLRNet           — 모델 자체

data_collection/
└── mediapipe_runner.py  → MediaPipeRunner   — 손/얼굴 landmark 추출

utils/
├── checkpoint.py        → load_checkpoint   — best.pt 로딩
└── text_overlay.py      → KoreanTextRenderer — 한글 표시 (선택)
```

가장 자주 쓰는 클래스는 **`WebcamPipeline`** — 입력은 BGR frame 1장, 출력은 예측 1개.

---

## 2. 실시간 webcam 통합

### 2-1. 최소 예제

```python
import time, yaml, cv2, torch
from pathlib import Path

from data_collection.mediapipe_runner import MediaPipeRunner
from models.kslr_net import KSLRNet
from realtime.webcam_pipeline import WebcamPipeline
from utils.checkpoint import load_checkpoint

cfg = yaml.safe_load(Path('configs/lab_dataset.yaml').read_text(encoding='utf-8'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 + 가중치
model = KSLRNet(cfg)
load_checkpoint('runs/kslr_lab_v0/best.pt', model=model, map_location=device)
model.to(device).eval()

# MediaPipe + 파이프라인
runner = MediaPipeRunner(
    'models_assets/hand_landmarker.task',
    'models_assets/face_landmarker.task',
)
pipeline = WebcamPipeline(model=model, mp_runner=runner, cfg=cfg, device=device)

# vocabulary
vocab = yaml.safe_load(Path('data_collection/vocabulary.yaml').read_text(encoding='utf-8'))
labels = {int(k): v for k, v in vocab['classes'].items()}

# 메인 루프 (cv2.imshow 없이 콘솔 출력)
cap = cv2.VideoCapture(0)
t0 = time.monotonic()
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts_ms = int((time.monotonic() - t0) * 1000)
        result, prediction = pipeline.step_frame(frame, ts_ms)
        if prediction is not None:
            label = labels.get(prediction.label_id, f'class_{prediction.label_id}')
            print(f'{label:>10s}  conf={prediction.confidence*100:5.1f}%')
finally:
    cap.release()
    runner.close()
```

### 2-2. PyQt6 GUI 통합 예제

```python
# kslr_widget.py — PyQt6 Widget 안에 KSLR 인식 띄우기
import sys, time, yaml, torch, cv2
from pathlib import Path
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from data_collection.mediapipe_runner import MediaPipeRunner
from models.kslr_net import KSLRNet
from realtime.webcam_pipeline import WebcamPipeline
from utils.checkpoint import load_checkpoint


class KSLRWidget(QMainWindow):
    def __init__(self, ckpt_path: str, config_path: str = 'configs/lab_dataset.yaml'):
        super().__init__()
        self.setWindowTitle('KSLR Recognition')

        self.cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = KSLRNet(self.cfg)
        load_checkpoint(ckpt_path, model=model, map_location=device)
        model.to(device).eval()
        self.runner = MediaPipeRunner(
            self.cfg['mediapipe']['hand']['model_asset_path'],
            self.cfg['mediapipe']['face']['model_asset_path'],
        )
        self.pipeline = WebcamPipeline(model, self.runner, self.cfg, device)

        vocab = yaml.safe_load(Path('data_collection/vocabulary.yaml').read_text(encoding='utf-8'))
        self.labels = {int(k): v for k, v in vocab['classes'].items()}

        self.cap = cv2.VideoCapture(0)
        self.t0 = time.monotonic()

        # UI
        central = QWidget(); self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.video_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.pred_label = QLabel('warming up...', alignment=Qt.AlignmentFlag.AlignCenter)
        self.pred_label.setStyleSheet('font-size: 36px; font-weight: bold;')
        layout.addWidget(self.video_label)
        layout.addWidget(self.pred_label)

        self.timer = QTimer(self); self.timer.timeout.connect(self.tick)
        self.timer.start(30)   # ~33ms ≈ 30 FPS

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        ts_ms = int((time.monotonic() - self.t0) * 1000)
        result, prediction = self.pipeline.step_frame(frame, ts_ms)

        if prediction is not None:
            label = self.labels.get(prediction.label_id, '?')
            self.pred_label.setText(f'{label}  ({prediction.confidence*100:.0f}%)')

        # BGR → RGB → QImage → QLabel
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w*3, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaledToWidth(720))

    def closeEvent(self, e):
        self.cap.release()
        self.runner.close()
        super().closeEvent(e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = KSLRWidget('runs/kslr_lab_v0/best.pt')
    w.show()
    sys.exit(app.exec())
```

### 2-3. Prediction 객체 구조

```python
@dataclass
class Prediction:
    label_id: int            # 0..9
    confidence: float        # 0..1, smoothed (EMA/majority)
    raw_label_id: int        # 직전 추론 결과만 (smoothing 전)
    raw_confidence: float    # 직전 추론 신뢰도
```

`label_id` 와 `confidence` 가 안정화된 값 — UI 표시용. `raw_*` 는 디버깅용.

---

## 3. HTTP 서버 패턴

웹/모바일 클라이언트에서 인식 결과를 받고 싶을 때.

### 3-1. FastAPI — 클라이언트가 frame을 push

```python
# server.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io, time, yaml, torch, cv2
import numpy as np
from PIL import Image
from pathlib import Path

from data_collection.mediapipe_runner import MediaPipeRunner
from models.kslr_net import KSLRNet
from realtime.webcam_pipeline import WebcamPipeline
from utils.checkpoint import load_checkpoint

app = FastAPI()

cfg = yaml.safe_load(Path('configs/lab_dataset.yaml').read_text(encoding='utf-8'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KSLRNet(cfg)
load_checkpoint('runs/kslr_lab_v0/best.pt', model=model, map_location=device)
model.to(device).eval()
runner = MediaPipeRunner(
    cfg['mediapipe']['hand']['model_asset_path'],
    cfg['mediapipe']['face']['model_asset_path'],
)

# session_id → pipeline (각 클라이언트마다 별도 frame buffer)
SESSIONS: dict[str, tuple[WebcamPipeline, float]] = {}

def _get_pipeline(session_id: str) -> tuple[WebcamPipeline, float]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = (
            WebcamPipeline(model, runner, cfg, device),
            time.monotonic(),
        )
    return SESSIONS[session_id]


@app.post('/predict')
async def predict(session_id: str, frame: UploadFile = File(...)):
    img_bytes = await frame.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    pipeline, t0 = _get_pipeline(session_id)
    ts_ms = int((time.monotonic() - t0) * 1000)
    _, prediction = pipeline.step_frame(bgr, ts_ms)

    if prediction is None:
        return JSONResponse({
            'status': 'warming_up',
            'buffer_fill': len(pipeline.buffer),
            'buffer_capacity': pipeline.buffer.capacity,
        })
    return JSONResponse({
        'status': 'ok',
        'label_id': prediction.label_id,
        'confidence': prediction.confidence,
    })


@app.delete('/session/{session_id}')
async def reset_session(session_id: str):
    SESSIONS.pop(session_id, None)
    return {'status': 'reset'}
```

실행: `uvicorn server:app --host 0.0.0.0 --port 8000`

> **주의**: HTTP overhead 때문에 30 FPS 유지가 어려울 수 있음. WebSocket / gRPC 가 더 적합.

### 3-2. WebSocket — 스트리밍 (권장)

```python
# server_ws.py — 핵심 부분만
from fastapi import FastAPI, WebSocket
import cv2, numpy as np, time

app = FastAPI()

@app.websocket('/ws/{session_id}')
async def ws_predict(websocket: WebSocket, session_id: str):
    await websocket.accept()
    pipeline = WebcamPipeline(model, runner, cfg, device)
    t0 = time.monotonic()
    try:
        while True:
            data = await websocket.receive_bytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            ts_ms = int((time.monotonic() - t0) * 1000)
            _, prediction = pipeline.step_frame(bgr, ts_ms)
            if prediction is not None:
                await websocket.send_json({
                    'label_id': prediction.label_id,
                    'confidence': prediction.confidence,
                })
    except Exception:
        pass
```

---

## 4. 오프라인 비디오 인식

녹화된 .mp4/.avi 를 인식하고 싶을 때.

```python
import cv2, time, torch, yaml
from pathlib import Path
from data_collection.mediapipe_runner import MediaPipeRunner
from models.kslr_net import KSLRNet
from realtime.webcam_pipeline import WebcamPipeline
from utils.checkpoint import load_checkpoint

cfg = yaml.safe_load(Path('configs/lab_dataset.yaml').read_text(encoding='utf-8'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = KSLRNet(cfg)
load_checkpoint('runs/kslr_lab_v0/best.pt', model=model, map_location=device)
model.to(device).eval()
runner = MediaPipeRunner(
    cfg['mediapipe']['hand']['model_asset_path'],
    cfg['mediapipe']['face']['model_asset_path'],
)
pipeline = WebcamPipeline(model, runner, cfg, device)

cap = cv2.VideoCapture('input.mp4')
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

results: list[tuple[float, int, float]] = []   # (timestamp_sec, label_id, confidence)
frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    ts_ms = int(frame_idx * 1000 / fps)
    _, prediction = pipeline.step_frame(frame, ts_ms)
    if prediction is not None:
        results.append((ts_ms / 1000, prediction.label_id, prediction.confidence))
    frame_idx += 1

cap.release()
runner.close()

# 결과: 매 stride 프레임마다 예측 1개
for t_sec, label_id, conf in results:
    print(f't={t_sec:6.2f}s  label={label_id}  conf={conf*100:.1f}%')
```

---

## 5. 최저 레벨 호출 — 직접 tensor 구성

`WebcamPipeline` 을 거치지 않고 KSLRNet 을 직접 호출하고 싶을 때 (예: 다른 landmark 소스, 다른 입력 형태).

```python
import torch
import yaml
from pathlib import Path
from models.kslr_net import KSLRNet
from utils.checkpoint import load_checkpoint

cfg = yaml.safe_load(Path('configs/lab_dataset.yaml').read_text(encoding='utf-8'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = KSLRNet(cfg).to(device).eval()
load_checkpoint('runs/kslr_lab_v0/best.pt', model=model, map_location=device)

# 입력 텐서 직접 구성 (이미 정규화된 landmark + crop을 가지고 있다고 가정)
B, T, H = 1, 16, 64
N_face = cfg['input']['face']['num_landmarks']

inputs = {
    'hand_lm':   torch.randn(B, T, 2, 21, 3).to(device),       # 정규화 후 좌표
    'face_lm':   torch.randn(B, T, N_face, 3).to(device),      # 정규화 후 좌표
    'hand_crop': torch.rand(B, T, 2, 3, H, H).to(device),      # [0, 1] 범위
    'face_crop': torch.rand(B, T, 3, H, H).to(device),         # [0, 1] 범위
    'hand_mask': torch.ones(B, T, 2, dtype=torch.bool).to(device),
    'face_mask': torch.ones(B, T, dtype=torch.bool).to(device),
}

with torch.no_grad():
    logits = model(**inputs)             # (B, 10)
    probs = torch.softmax(logits, dim=-1)
    label_id = probs.argmax(-1).item()
    confidence = probs.max().item()
print(f'label={label_id}  conf={confidence*100:.1f}%')
```

### 입력 형식 명세

| 키 | shape | dtype | 의미 |
|---|---|---|---|
| `hand_lm` | (B, 16, 2, 21, 3) | float32 | wrist-relative 정규화 후 좌표 |
| `face_lm` | (B, 16, 97, 3) | float32 | nose-relative 정규화 후 좌표 |
| `hand_crop` | (B, 16, 2, 3, 64, 64) | float32 | RGB crop, [0, 1] 범위 |
| `face_crop` | (B, 16, 3, 64, 64) | float32 | RGB crop, [0, 1] 범위 |
| `hand_mask` | (B, 16, 2) | bool | True = 해당 hand가 검출됨 |
| `face_mask` | (B, 16) | bool | True = 해당 frame에 face 검출됨 |

정규화 공식은 [`data/normalizer.py`](../data/normalizer.py) 참고:
- 손: 좌표를 wrist(landmark 0) 기준 평행이동 → wrist↔middle MCP(landmark 9) 거리로 scale
- 얼굴: 코끝(subset idx 1) 기준 평행이동 → 양쪽 입꼬리 거리로 scale

---

## 6. 성능 최적화 팁

### GPU 사용
```python
device = torch.device('cuda')   # 5–10× 빠름 (배치 추론 시 더 큰 차이)
model.to(device).eval()
```

### Half precision (FP16) — GPU 환경
```python
model = model.half()   # 거의 정확도 손실 없이 2× 메모리/속도
inputs = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()}
```

### `torch.compile` — PyTorch 2.0+
```python
model = torch.compile(model)   # 첫 호출은 느리지만 이후 빠름
```

### Stride 늘리기 — 지연 vs FPS 트레이드오프
```yaml
# configs/lab_dataset.yaml
realtime:
  stride: 16    # 8 → 16 으로: 추론 빈도 절반, 부하도 절반
```

### Smoothing 끄기 — 디버깅 용
```yaml
realtime:
  smoothing:
    method: none    # ema → none
```

---

## 7. 체크리스트

- [ ] `pip install -r requirements.txt` 완료
- [ ] `python scripts/download_models.py` 로 MediaPipe `.task` 파일 다운로드
- [ ] `runs/kslr_lab_v0/best.pt` 배치
- [ ] `data_collection/vocabulary.yaml` 이 학습 시점과 동일한지 확인
- [ ] `python scripts/demo.py --ckpt runs/kslr_lab_v0/best.pt` 로 정상 동작 확인
- [ ] 그 다음 자체 코드에 통합

문제가 생기면 → [`troubleshooting.md`](troubleshooting.md)
