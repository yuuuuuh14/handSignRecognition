# Quick Start — 5분 안에 데모 실행하기

배포받은 사람이 **카메라 앞에서 수어를 인식하는 데모를 실행**하는 가장 빠른 경로입니다.

## 받은 파일 확인

배포자에게 다음 4가지를 받아야 합니다:

| 파일 | 위치 | 용량 | 필수 여부 |
|---|---|---|---|
| 소스 코드 | repo 클론 또는 zip | ~수십 KB | 필수 |
| **체크포인트** `best.pt` | `runs/kslr_lab_v0/best.pt` | ~6 MB | 필수 |
| `config_resolved.yaml` | 위와 같은 폴더 | ~3 KB | 권장 (학습 시 설정 보존) |
| `vocabulary.yaml` | `data_collection/vocabulary.yaml` | ~300 B | 필수 |

> 체크포인트와 데이터는 `.gitignore` 되어 있어 git clone 만으로는 받을 수 없습니다 — 따로 공유받아야 합니다.

## 1단계 — 환경 준비

### Python 3.10+ 필요
```powershell
python --version       # 3.10 이상 확인
```

### 가상환경 + 의존성 설치
```powershell
cd handSignRecognition
python -m venv .venv
.venv\Scripts\activate            # PowerShell
# (macOS/Linux: source .venv/bin/activate)

pip install -r requirements.txt
```

> 설치 시간: torch + mediapipe 가 가장 큼. 인터넷 속도에 따라 5–15분.

### MediaPipe 모델 다운로드 (최초 1회)
```powershell
python scripts/download_models.py
# → models_assets/hand_landmarker.task (약 7 MB)
# → models_assets/face_landmarker.task (약 4 MB)
```

## 2단계 — 체크포인트 배치

받은 `best.pt` 와 `config_resolved.yaml` 을 다음 위치에 두세요:

```
handSignRecognition/
└── runs/
    └── kslr_lab_v0/
        ├── best.pt
        └── config_resolved.yaml
```

폴더가 없으면 생성:
```powershell
mkdir runs\kslr_lab_v0
# 받은 파일을 위 폴더에 복사
```

## 3단계 — vocabulary 확인

[`data_collection/vocabulary.yaml`](../data_collection/vocabulary.yaml) 가 학습 시점과 동일한지 확인:
```yaml
version: 1
classes:
  0: 안녕하세요
  1: 감사합니다
  2: 사랑
  3: 학교
  4: 친구
  5: 물
  6: 밥
  7: 가다
  8: 오다
  9: 끝
```

> 인덱스 0..9 의 매핑은 학습 시 결정된 것이므로 **임의로 바꾸면 안 됩니다**.

## 4단계 — 데모 실행

```powershell
python scripts/demo.py --ckpt runs/kslr_lab_v0/best.pt
```

성공 시 카메라 창이 열리고:

- 좌상단: FPS, 추론 시간 (ms)
- 우상단: "Warming up... N/16" — 첫 16 프레임(약 0.5초) 동안 진행률 바
- 화면 전체: 손/얼굴 landmark 실시간 표시
- 하단 (예측 시작 후): **인식된 한국어 단어** + 신뢰도 bar + top-3 후보

## 키보드

| 키 | 동작 |
|---|---|
| `q` | 종료 |
| `r` | frame buffer 리셋 (warming up 다시 시작) |

## CLI 옵션

```powershell
python scripts/demo.py --help
```

자주 쓰는 옵션:

```powershell
# 다른 카메라 사용 (외장 웹캠 등)
python scripts/demo.py --ckpt runs/kslr_lab_v0/best.pt --camera 1

# CPU 강제 (GPU 있어도)
python scripts/demo.py --ckpt runs/kslr_lab_v0/best.pt --device cpu

# config 경로 변경
python scripts/demo.py --ckpt path/to/best.pt --config configs/lab_dataset.yaml

# 카메라 창 없이 헤드리스 (벤치마크용)
python scripts/demo.py --ckpt runs/kslr_lab_v0/best.pt --no-window
```

## 사용 팁

### 인식 잘 되도록 동작하기

- **카메라와 거리**: 상반신이 보이도록 약 1m 거리
- **조명**: 정면에서 균등하게 (역광 피하기)
- **배경**: 단순한 단색 배경에서 잘 작동 (학습 환경과 비슷할수록 정확도 ↑)
- **속도**: 너무 빠르지 않게, 약 0.5초 동안 동작 한 번
- **양손**: 양손 다 화면 안에 들어오도록

### 신뢰도 색상 의미

| 색 | 신뢰도 범위 | 의미 |
|---|---|---|
| 초록 | ≥ 70% | 강한 확신 — 거의 정답 |
| 주황 | 40–70% | 애매 — 동작 다시 시도 |
| 빨강 | < 40% | 낮음 — 학습된 클래스가 아닐 가능성 |

## 다음 단계

- 실제 사용 결과가 만족스러우면 끝
- 외부 앱에 통합하고 싶으면 → [`python_api.md`](python_api.md)
- 단어 추가/재학습 → [`training_guide.md`](training_guide.md)
- 안 되는 게 있으면 → [`troubleshooting.md`](troubleshooting.md)
