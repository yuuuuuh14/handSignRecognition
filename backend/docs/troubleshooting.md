# Troubleshooting — 자주 나오는 문제

## 설치 / 환경

### `pip install -r requirements.txt` 실패: `UnicodeDecodeError 'cp949' codec`

Windows 한국어 로케일 + UTF-8 BOM 없는 requirements.txt 파일.

**해결**:
```powershell
$env:PYTHONUTF8 = "1"     # 현재 세션
# 영구 설정 (사용자 환경변수): 시작 → "환경 변수 편집" → 새로 만들기 PYTHONUTF8=1
pip install -r requirements.txt
```

### `pip install` 시 torch 다운로드가 너무 느림 / 멈춤

PyTorch 공식 인덱스 사용:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # CPU 버전
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1 버전
pip install -r requirements.txt    # 나머지
```

### `mediapipe` import 에러: `module 'mediapipe' has no attribute 'solutions'`

mediapipe 0.10+ 에서는 `mp.solutions` 가 default로 노출되지 않음. 본 프로젝트는 Tasks API (`mp.tasks.python.vision`) 만 사용하므로 영향 없음. 만약 외부 코드가 `mp.solutions` 를 쓴다면 mediapipe 0.10.9 이하로 다운그레이드.

---

## MediaPipe 모델 파일

### `FileNotFoundError: hand_landmarker.task`

```powershell
python scripts/download_models.py
```

수동 다운로드:
```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```
→ `models_assets/` 폴더에 배치.

### `download_models.py` 가 SSL/네트워크 에러

방화벽/프록시 환경. 위 URL 을 브라우저로 직접 다운로드 후 `models_assets/` 에 복사.

---

## 카메라 / 웹캠

### `cv2.VideoCapture(0)` 가 열리지 않음

```
[error] Failed to open camera 0
```

**원인 + 해결**:

1. **다른 앱이 카메라를 잡고 있음** (Zoom, Teams, 브라우저 탭) — 닫고 재시도
2. **카메라 인덱스 다름** — `--camera 1` 또는 `--camera 2` 시도
3. **Windows 권한 부족** — 설정 → 개인정보 → 카메라 → 데스크톱 앱 허용
4. **DirectShow backend 문제** (Windows) — record.py / demo.py 가 `cv2.CAP_DSHOW` 사용. 끄려면 코드 수정 필요

카메라 인덱스 확인:
```python
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f'camera {i}: OK ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})')
        cap.release()
```

### 영상이 까만/회색 화면

해상도 불일치. config 의 `clip.webcam_resolution: [640, 480]` 가 카메라가 지원하지 않을 수 있음.
`[1280, 720]` 또는 `[640, 480]` 등 시도.

### FPS 가 30 안 나옴

- USB 2.0 hub 거치면 대역폭 부족. 직결 시도
- `clip.webcam_fps: 30` 이지만 카메라가 실제 24 FPS만 지원하면 자동 적용
- demo.py 우상단 `FPS` 가 실제 측정값. 너무 낮으면 (<15) MediaPipe 가 병목

---

## 학습

### 학습 시작 시 `[error] no clips under data/raw/`

데이터가 없음. `python scripts/record.py --signer 1` 부터.

### 학습 시작 시 `[error] train split is empty`

config 의 `train_signers` 와 실제 녹화한 signer 가 안 맞음.

```powershell
# 현재 데이터 확인
python -c "from data.splits import discover_clips, summarize_clips; print(summarize_clips(discover_clips('data/raw')))"
```

config 의 `train_signers` 리스트에 실제 signer ID가 있어야 함.

### `face_lm shape (16, 96, 3) != (16, 97, 3)`

mediapipe_runner.py 가 face subset에 nose tip(landmark 1) 추가 후 96 → 97 로 늘었음. 이전 96-shape 데이터는 사용 불가 → 폐기 후 재녹화.

```powershell
Remove-Item -Recurse -Force data\raw\1   # 해당 signer 폴더 통째로 삭제
python scripts/record.py --signer 1      # 재녹화
```

### Train loss 가 감소 안 함 / `nan` 값

- `nan` 발생 시: lr 너무 높거나 grad clip 미작동 의심. `lr=5e-4` 로 낮춰서 시도
- 감소 안 함: dataset 라벨 / 입력 텐서 검증
  ```powershell
  python tests/test_dataset.py
  python tests/test_kslr_net.py
  ```

### GPU 인식 안 됨

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

`False` 면:
- CUDA 없는 PyTorch 설치됨 → CUDA 빌드 재설치
- 드라이버 버전 mismatch → `nvidia-smi` 로 driver 버전 확인 후 호환되는 PyTorch 빌드 선택

CUDA 있는데 OOM:
```yaml
train:
  batch_size: 16    # 32 → 16
  amp: true         # 이미 켜져있음
```

### 학습 속도 너무 느림 (CPU)

- AMP는 GPU 전용 — CPU 에선 효과 없음
- `num_workers: 0` 으로 시작해서 점진적으로 늘림 (Windows 에선 `multiprocessing` overhead 큼)
- batch_size 줄여서 메모리 효율 ↑

### 학습 도중 강제 종료 후 재개

```powershell
python scripts/train.py --resume runs/kslr_lab_v0/latest.pt
```

`latest.pt` 는 매 epoch 마다 저장되므로 마지막 epoch 부터 이어서.

---

## 인퍼런스 / 데모

### 데모가 항상 같은 클래스만 예측

**가능성 1 — 학습 부족**: train_loss 가 충분히 안 떨어졌거나 데이터가 한쪽으로 치우침.
```powershell
python scripts/evaluate.py --ckpt runs/kslr_lab_v0/best.pt --split train
```
per-class F1 가 일부만 높으면 데이터 비율 문제.

**가능성 2 — vocabulary 불일치**: 학습 시점과 다른 vocabulary.yaml 사용. 같은 인덱스 매핑인지 확인:
```powershell
cat data_collection/vocabulary.yaml
cat runs/kslr_lab_v0/config_resolved.yaml | grep -A 12 vocabulary
```

**가능성 3 — face/hand 검출 자체가 약함**: 카메라 앞에서 손이 안 보이거나 배경이 너무 복잡할 때 → 검출 실패 → zero-input → 모델은 zero-input 시 특정 클래스로 편향.
- 조명/배경 정리 후 재시도
- 화면에 표시되는 landmark 점이 손/얼굴에 잘 붙는지 확인

### 데모 시 "warming up" 이 사라지지 않음

16 frame 채워야 첫 예측 나옴 (~533ms @30fps). 만약 30초 지나도 그대로면:
- frame buffer 에 push가 안 됨 → MediaPipe 에러 가능성
- 콘솔에 에러 메시지 확인

### `q` 눌러도 종료 안 됨

cv2 창에 포커스가 있어야 함 (창 클릭 후 q).

### 인식이 흔들림 / 너무 빨리 바뀜

```yaml
realtime:
  smoothing:
    method: ema
    alpha: 0.3      # 0.6 → 0.3 (과거 weight 강화)
  stride: 4         # 8 → 4 (추론 빈도 ↑)
```

또는:
```yaml
realtime:
  smoothing:
    method: majority   # ema → majority (window 5 의 다수결)
```

### 추론이 너무 느림 (latency 높음)

```powershell
# 어디가 느린지 확인
python scripts/profile_macs.py
```

해결책:
- GPU 사용 (`--device cuda`)
- `stride` 늘림 (8 → 16) — 추론 횟수 절반
- model 작게 — `per_frame_fusion_dim: 192 → 128`, `irffn.hidden_dim: 576 → 384` (재학습 필요)
- ONNX 변환 → ONNXRuntime — [`deployment.md`](deployment.md) §3

---

## 한글 표시

### 데모 / 녹화 GUI 에서 한글이 `?` 로 보임

`utils/text_overlay.py` 의 폰트 후보에서 사용 가능한 게 없음. 콘솔에 다음 경고가 떴는지 확인:
```
[warn] no Korean-capable font found; falling back to PIL default
```

해결:
- Windows: `C:/Windows/Fonts/malgun.ttf` 가 있어야 함 (대부분 자동 설치됨)
- macOS: Apple SD Gothic Neo 자동 설치
- Linux: `sudo apt install fonts-nanum` 또는 Noto CJK 설치

수동 폰트 추가:
```python
# utils/text_overlay.py 의 FONT_CANDIDATES 리스트에 자기 폰트 경로 추가
FONT_CANDIDATES.insert(0, Path('/path/to/your_korean.ttf'))
```

### vocabulary.yaml 의 한글이 깨짐

UTF-8 인코딩으로 저장. VS Code 우하단 인코딩 표시 확인 → "UTF-8" 이어야 함.

---

## Checkpoint / 파일

### `RuntimeError: Error(s) in loading state_dict`

원인 — 학습한 모델과 로드하려는 모델 구조 불일치.

체크:
1. `configs/lab_dataset.yaml` 의 `model.*` 설정이 학습 시점과 같은가?
2. 클래스 수 (`num_classes`) 가 같은가?
3. `data_collection.mediapipe_runner.NUM_FACE_LANDMARKS` (97) 가 같은가?

학습 시점 설정은 `runs/kslr_lab_v0/config_resolved.yaml` 에 보존됨. 이 파일을 `configs/` 에 복사해서 사용:
```powershell
python scripts/demo.py --config runs/kslr_lab_v0/config_resolved.yaml --ckpt runs/kslr_lab_v0/best.pt
```

### Best.pt 가 너무 큼 (수십 MB)

체크포인트는 model + optimizer + scheduler + AMP scaler 를 모두 포함. 배포할 때는 model 만 추출:

```python
import torch
ck = torch.load('runs/kslr_lab_v0/best.pt', map_location='cpu')
torch.save({'model': ck['model']}, 'best_model_only.pt')   # 더 작음
```

로드 시:
```python
from utils.checkpoint import load_checkpoint
load_checkpoint('best_model_only.pt', model=model)   # 그대로 동작
```

---

## 테스트

### `tests/test_*.py` 실행 시 import 에러

프로젝트 루트에서 실행:
```powershell
cd E:\workspace\handSignRecognition
python tests/test_kslr_net.py
```

또는 pytest:
```powershell
pip install pytest
pytest tests/
```

### `tests/test_dataset.py` 가 `[skip] no clips found` 출력

`data/raw/` 비어있음. 시범 데이터 한두 개라도 녹화 후 재실행.

---

## 그래도 안 되는 경우

수집할 정보:
1. 운영체제 + Python 버전 — `python --version`
2. PyTorch / mediapipe 버전 — `pip show torch mediapipe`
3. 에러 메시지 전문 (traceback 포함)
4. 실행한 명령어
5. config 변경한 부분이 있다면 그 부분

이 정보와 함께 GitHub issues 또는 프로젝트 관리자에게 문의.
