# Deployment Guide — 배포 패키징

이 문서는 **모델 제작자**가 자신이 학습한 모델을 다른 사람에게 전달할 때 사용합니다.

## 1. 무엇을 패키징해야 하나?

### 받는 사람의 목적별 필요 파일

| 목적 | 받는 사람이 필요한 것 |
|---|---|
| **A. 데모 실행만** (가장 흔함) | 코드 + `best.pt` + `vocabulary.yaml` + `config_resolved.yaml` |
| **B. 자체 앱 통합** | A + Python API 가이드 + (옵션) ONNX 변환 |
| **C. 재학습/단어 추가** | A + 원본 데이터 (`data/raw/`) + 학습 가이드 |
| **D. 연구/논문 재현** | C + tensorboard 로그 + eval_report.json |

### 핵심 파일 위치

```
✅ 코드 (git push 가능)
   handSignRecognition/                # 전체 소스 트리
   ├── configs/, data/, data_collection/
   ├── engine/, models/, realtime/
   ├── scripts/, utils/, tests/
   ├── docs/                           # 본 문서들
   ├── requirements.txt
   ├── CLAUDE.md, IMPLEMENTATION_PLAN.md
   └── data_collection/vocabulary.yaml ← 클래스 매핑 (반드시 동봉)

❌ 코드와 같이 갈 수 없는 것 (gitignored)
   runs/kslr_lab_v0/best.pt            ← 별도 전달 (cloud / release / drive)
   runs/kslr_lab_v0/config_resolved.yaml
   data/raw/                           ← 개인정보 포함 (얼굴 영상). 신중히 공유
   models_assets/*.task                ← 받는 사람이 download_models.py 로 직접 받음
```

## 2. 패키징 방법 3가지

### 방법 A — Git + 별도 체크포인트 (권장)

코드는 git, 무거운 파일은 따로.

#### 1) 코드를 git repository에
```powershell
git init
git add .
git commit -m "Initial KSLR release"
git push origin main
```

`.gitignore` 가 이미 `data/raw/`, `runs/`, `*.pt`, `models_assets/` 를 제외하므로 안전합니다.

#### 2) 체크포인트를 GitHub Release / Cloud에
**GitHub Release** (권장 — 코드와 같이 버전 관리):
```powershell
gh release create v1.0 runs/kslr_lab_v0/best.pt runs/kslr_lab_v0/config_resolved.yaml --title "KSLR v1.0" --notes "10-class lab model, top-1 94.85%"
```
받는 사람:
```powershell
git clone https://github.com/<user>/handSignRecognition.git
cd handSignRecognition
gh release download v1.0 -D runs/kslr_lab_v0/
```

**대안 — Google Drive / OneDrive / S3**: 링크만 README에 적고 받는 사람이 다운로드.

### 방법 B — 단일 ZIP 파일 (오프라인 배포)

설치 도움이 어려운 환경(인터넷 제한, 한 번 보내고 끝)에서 유용.

```powershell
# 배포용 zip 생성 (Python으로 안전하게)
python -c "
import shutil, os
shutil.make_archive('kslr_release_v1', 'zip', '.',
    base_dir=None,
    logger=None,
)
" 
```

좀 더 정밀하게, **불필요한 파일 제외**:
```powershell
# PowerShell — 핵심만 묶기
$dist = 'kslr_release_v1'
New-Item -ItemType Directory $dist -Force | Out-Null
Copy-Item -Recurse -Path configs, data_collection, data, models, engine, realtime, scripts, utils, tests, docs $dist
Copy-Item -Path requirements.txt, CLAUDE.md, IMPLEMENTATION_PLAN.md $dist
New-Item -ItemType Directory "$dist\runs\kslr_lab_v0" -Force | Out-Null
Copy-Item runs\kslr_lab_v0\best.pt, runs\kslr_lab_v0\config_resolved.yaml "$dist\runs\kslr_lab_v0\"
# tensorboard 로그 / 다른 epoch 체크포인트는 큼 → 제외 (선택)
Compress-Archive -Path $dist -DestinationPath "$dist.zip" -Force
Remove-Item -Recurse -Force $dist
```

받는 사람: 압축 해제 → [`docs/quick_start.md`](quick_start.md) 따라 진행.

### 방법 C — Docker 이미지 (재현성 최강)

Python/CUDA 버전 차이로 인한 환경 문제를 피하고 싶을 때.

`Dockerfile` 예시:
```dockerfile
FROM python:3.11-slim

# 시스템 의존성 (OpenCV, MediaPipe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python scripts/download_models.py

# 데모는 카메라 접근 필요 → 호스트에서 실행 권장
CMD ["python", "scripts/demo.py", "--ckpt", "runs/kslr_lab_v0/best.pt"]
```

빌드/실행:
```powershell
docker build -t kslr:v1.0 .
docker run --rm -it --device=/dev/video0 -e DISPLAY=$env:DISPLAY kslr:v1.0
```

> Windows 호스트에서 도커 컨테이너의 카메라/디스플레이 접근은 까다로워서, 데모용은 호스트 직접 실행을 권장합니다. Docker는 학습/평가 같은 GUI 없는 작업에 적합.

## 3. ONNX 내보내기 (선택 — cross-platform)

PyTorch 외 환경(C++/모바일/웹/ONNXRuntime)에서 실행하고 싶다면 ONNX로 변환.

`scripts/export_onnx.py` 를 만들어 변환할 수 있습니다 (**참고용 예시 — 현재 repo에는 없음**):

```python
# scripts/export_onnx.py 예시
import sys, torch, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.kslr_net import KSLRNet
from utils.checkpoint import load_checkpoint

cfg = yaml.safe_load(Path('configs/lab_dataset.yaml').read_text(encoding='utf-8'))
model = KSLRNet(cfg).eval()
load_checkpoint('runs/kslr_lab_v0/best.pt', model=model)

T = cfg['clip']['num_frames']
N_face = cfg['input']['face']['num_landmarks']
H = cfg['input']['hand']['crop_size']
dummy = (
    torch.randn(1, T, 2, 21, 3),
    torch.randn(1, T, N_face, 3),
    torch.rand(1, T, 2, 3, H, H),
    torch.rand(1, T, 3, H, H),
    torch.ones(1, T, 2, dtype=torch.bool),
    torch.ones(1, T, dtype=torch.bool),
)
torch.onnx.export(
    model, dummy, 'runs/kslr_lab_v0/best.onnx',
    input_names=['hand_lm','face_lm','hand_crop','face_crop','hand_mask','face_mask'],
    output_names=['logits'],
    opset_version=17,
    dynamic_axes={k: {0: 'B'} for k in ('hand_lm','face_lm','hand_crop','face_crop','hand_mask','face_mask','logits')},
)
print('exported to runs/kslr_lab_v0/best.onnx')
```

> **주의**: KSLRNet은 6개 입력 텐서를 받는 multi-input 모델. ONNXRuntime에서도 6개 입력으로 호출해야 함. ptflops가 dummy로 잘 처리하는 패턴을 그대로 적용.

## 4. 받는 사람을 위한 체크리스트

배포할 때 다음 정보를 함께 전달하세요:

```markdown
# KSLR 모델 v1.0 배포 안내

## 포함 사항
- 코드: <git URL 또는 zip>
- 체크포인트: <다운로드 링크>
- 모델 카드:
  - 인식 클래스: 안녕하세요/감사합니다/사랑/학교/친구/물/밥/가다/오다/끝
  - top-1 정확도: 94.85% (test split, signer-independent)
  - 추론 속도: 67 ms/clip (CPU, 1.61M params)

## 시스템 요구사항
- Python 3.10+
- 2 GB RAM (CPU 추론) / 4 GB GPU 메모리 (GPU 추론)
- 웹캠 (USB 또는 내장)
- Windows / macOS / Linux

## 시작하기
1. docs/quick_start.md 따라 환경 구축
2. python scripts/demo.py --ckpt runs/kslr_lab_v0/best.pt

## 라이선스
- 코드: MIT
- 데이터: 자체 수집 (재배포 금지/허용 여부 명시)
- 사용 시 인용: <원하면 작성>

## 문의
- 이슈: <GitHub Issues 링크>
- 이메일: <연락처>
```

## 5. 보안 / 개인정보

### `data/raw/` 처리 — 매우 중요
녹화 데이터는 **얼굴 영상 + landmark** 를 포함합니다. 다음을 반드시 확인:

1. **녹화 동의서** — signer 10명 모두로부터 데이터 수집/사용 동의 확보
2. **재배포 시** — signer 동의 범위 안에서만 (학술/연구만 허용 등 제한 있을 수 있음)
3. **얼굴 가리기** 옵션 — 데모 시 face landmark만 표시하고 RGB crop은 안 보이게

### 모델 파라미터의 개인정보 노출 위험
KSLRNet은 1.61M params로 작아서 학습 데이터를 그대로 외울 수 없지만, **face crop CNN 가중치가 특정 signer의 얼굴 텍스처에 fit** 될 가능성은 있습니다. 다만:
- Test split을 signer-independent로 분리해 학습한 모델은 일반화 성능이 측정됨
- 일반적인 face recognition 모델 대비 capacity 작음 → 실제 누설 가능성 낮음

학술 발표 시에는 "데이터 수집 동의 + signer-independent eval" 만 명시하면 충분.

## 6. 버전 관리 권장

체크포인트마다 의미 있는 태그를 붙이세요:

```
v1.0  — 초기 학습본 (이번 모델, top-1 94.85%)
v1.1  — 시범 사용 후 confused 클래스(안녕하세요/감사합니다) 추가 데이터 수집 후 fine-tune
v2.0  — vocabulary 10 → 20 확장
v2.1  — augment 강화 (rotation 25°, brightness 0.3) 후 재학습
```

각 버전마다:
- `best.pt` 와 `config_resolved.yaml` 묶어서 보존
- `eval_report.json` 으로 성능 변화 추적
- 어떤 데이터 변경이 반영됐는지 README에 기록

## 다음 단계

- 외부 앱 통합 가이드 → [`python_api.md`](python_api.md)
- 받는 사람의 첫 단계 → [`quick_start.md`](quick_start.md)
