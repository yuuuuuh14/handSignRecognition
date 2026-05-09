# Training Guide — 재학습 / 단어 추가 / 하이퍼파라미터

이 문서는 **모델을 다시 학습**하거나 **vocabulary 를 변경**하고자 하는 사용자를 위한 가이드입니다.

## 시나리오별 가이드

| 하고 싶은 것 | 가이드 섹션 |
|---|---|
| 기존 10 클래스에 데이터를 더 보태서 정확도 ↑ | §1 [데이터 추가 + fine-tune](#1-데이터-추가--fine-tune) |
| 클래스 수 변경 (10 → 20 등) | §2 [vocabulary 확장](#2-vocabulary-확장--클래스-수-변경) |
| 특정 혼동 클래스만 개선 | §3 [부분 클래스 fine-tune](#3-부분-클래스-fine-tune) |
| 학습 속도/정확도 tuning | §4 [하이퍼파라미터 조정](#4-하이퍼파라미터-조정) |
| 처음부터 재학습 | §5 [from scratch 재학습](#5-from-scratch-재학습) |

---

## 1. 데이터 추가 + fine-tune

가장 흔한 시나리오. 기존 모델을 starting point로 두고 새 데이터로 추가 학습.

### Step 1 — 데이터 녹화

```powershell
# signer 11 추가 (기존 1~10 외)
python scripts/record.py --signer 11
```

녹화 GUI에서 (자세한 사용법은 [`quick_start.md`](quick_start.md) 참고):
- 키 `1`~`9, 0` 으로 클래스 선택
- `Space` 로 16 frame 캡처
- `Enter` 저장 / `Backspace` 폐기
- 클래스당 **최소 30개 이상**, 다양한 각도/속도/조명에서 녹화

### Step 2 — config 에 새 signer 반영

[`configs/lab_dataset.yaml`](../configs/lab_dataset.yaml) 의 train/test signer 리스트 업데이트:

```yaml
data:
  train_signers: [1, 2, 3, 4, 5, 6, 7, 8, 11]   # 11 추가
  test_signers: [9, 10]                          # 그대로 유지
```

### Step 3 — fine-tune (기존 best.pt 부터)

`scripts/train.py` 는 `--resume` 옵션을 지원합니다.

```powershell
# 기존 가중치 + 더 작은 lr 로 50 epoch fine-tune
python scripts/train.py --resume runs/kslr_lab_v0/latest.pt --epochs 50 --name kslr_lab_v1
```

> `--name kslr_lab_v1` 은 새 폴더 `runs/kslr_lab_v1/` 에 결과 저장 → 기존 v0 보존.

### fine-tune 시 추천 lr

기본 `lr=1e-3` 은 from-scratch 용. fine-tune 은 더 작게:

```yaml
# configs/lab_dataset.yaml — 임시 fine-tune config
train:
  optimizer:
    lr: 1.0e-4              # 기본의 1/10
  scheduler:
    warmup_epochs: 0        # warmup 불필요 (이미 학습된 weight)
    min_lr: 1.0e-6
  epochs: 50
```

### Step 4 — 평가 + 비교

```powershell
python scripts/evaluate.py --ckpt runs/kslr_lab_v1/best.pt --split test
# → eval_report.json 생성. v0 와 비교
```

---

## 2. Vocabulary 확장 — 클래스 수 변경

10 → 20 같이 클래스 수를 늘리면 **classifier head 의 출력 dim 이 달라지므로 from-scratch 학습 필수** 입니다 (또는 head만 갈아끼우는 transfer learning).

### Step 1 — vocabulary.yaml 갱신

```yaml
# data_collection/vocabulary.yaml
version: 2                  # 변경 시마다 ↑
classes:
  0: 안녕하세요
  1: 감사합니다
  ...
  9: 끝
  10: 가족
  11: 학생
  ...
  19: 미안
```

### Step 2 — config 갱신

```yaml
# configs/lab_dataset.yaml
data:
  num_classes: 20           # 10 → 20
  vocabulary_path: data_collection/vocabulary.yaml
```

자동 반영되는 곳: classifier head, eval metric, recorder UI.

### Step 3 — 데이터 수집

새로 추가한 클래스 10개에 대해 모든 signer가 녹화. 각 클래스당 50개 이상.

> **시간 비용 주의**: 10 signer × 10 new class × 50 clip × ~5초/clip ≈ **40시간** 의 녹화 작업. 스케일업이 가장 큰 비용.

### Step 4 — from-scratch 학습

```powershell
python scripts/train.py --epochs 300 --name kslr_lab_v2_20cls
```

기존 `best.pt` 는 호환 안 됨 (head 차원 불일치). 단, 다음 옵션이 가능:

#### 옵션 A — 전체 from scratch
가장 안전. 위 명령 그대로.

#### 옵션 B — body만 가져오고 head는 새로 (transfer learning)

```python
# scripts/init_from_v1.py — 작성 예시
import torch
import yaml
from pathlib import Path
from models.kslr_net import KSLRNet
from utils.checkpoint import save_checkpoint

cfg_v2 = yaml.safe_load(Path('configs/lab_dataset.yaml').read_text(encoding='utf-8'))
model_v2 = KSLRNet(cfg_v2)

# 기존 v1 (10 cls) 가중치 로드 — classifier head 만 빼고
v1_state = torch.load('runs/kslr_lab_v1/best.pt', map_location='cpu')['model']
v2_state = model_v2.state_dict()

loaded = 0
skipped = []
for k, v in v1_state.items():
    if k in v2_state and v2_state[k].shape == v.shape:
        v2_state[k] = v
        loaded += 1
    else:
        skipped.append(f'{k}  ({v.shape} → {v2_state.get(k, torch.empty(0)).shape})')

model_v2.load_state_dict(v2_state)
print(f'loaded {loaded} tensors; skipped {len(skipped)} (shape mismatch):')
for s in skipped:
    print(f'  {s}')

save_checkpoint('runs/kslr_lab_v2_init.pt', model=model_v2)
print('Saved bootstrap weights for v2 from-scratch run.')
```

그리고:
```powershell
python scripts/train.py --resume runs/kslr_lab_v2_init.pt --epochs 200 --name kslr_lab_v2_20cls
```

---

## 3. 부분 클래스 fine-tune

eval_report 에서 특정 두 클래스가 자주 혼동될 때 (예: 안녕하세요 ↔ 감사합니다).

### Step 1 — 해당 클래스만 추가 데이터 녹화

```powershell
# 해당 두 클래스만 다양화해서 더 녹화
python scripts/record.py --signer 1
# 키 '1' (=class 0 안녕하세요), 키 '2' (=class 1 감사합니다) 만 사용해서 각 20개씩
```

### Step 2 — fine-tune (전체 데이터 + 새 데이터)

전체 데이터로 fine-tune 하면 다른 클래스 정확도가 떨어질 위험 있음. 작은 lr + 짧은 epoch 권장:

```powershell
python scripts/train.py --resume runs/kslr_lab_v0/best.pt --epochs 30 --name kslr_lab_v0_finetune
```

학습 끝나고 confusion matrix 비교:
```powershell
python scripts/evaluate.py --ckpt runs/kslr_lab_v0_finetune/best.pt --split test
```

해당 두 클래스의 F1이 올랐는지, 다른 클래스가 떨어지진 않았는지 확인.

---

## 4. 하이퍼파라미터 조정

### 4-1. Loss 미수렴 / 학습이 너무 느림

증상: 100 epoch 지나도 train_top1 가 50% 이하

```yaml
train:
  optimizer:
    lr: 2.0e-3              # 1e-3 → 2e-3 (높임)
  scheduler:
    warmup_epochs: 10       # warmup 길게
augment:
  enabled: false            # augment 잠시 끄고 학습 속도 확인
```

### 4-2. Train acc 100% / val acc 낮음 (overfit)

증상: train_loss 0.50 (label smoothing floor), train_top1 100%, val_top1 낮음

```yaml
augment:
  rotation_deg: 25.0        # 15 → 25
  scale_range: [0.8, 1.2]   # 0.9-1.1 → 0.8-1.2
  landmark_noise_sigma: 0.02   # 0.01 → 0.02
  color_jitter:
    brightness: 0.3         # 0.2 → 0.3
    contrast: 0.3
model:
  temporal:
    dropout: 0.2            # 0.1 → 0.2
  irffn:
    dropout: 0.2
```

근본 해결책: **데이터 다양성 확보** (signer 추가, 환경 변화).

### 4-3. 학습은 잘 되는데 영상 인퍼런스가 흔들림

증상: `demo.py` 에서 같은 동작에도 예측이 매번 바뀜

```yaml
realtime:
  smoothing:
    method: ema             # 없으면 추가
    alpha: 0.4              # 더 strong smoothing (0.6 → 0.4 — 과거 weight ↑)
  stride: 4                 # 8 → 4 — 더 자주 추론
```

### 4-4. 학습 속도 향상 (GPU 활용)

기본 config는 이미 AMP on. CUDA 환경이면 자동 활성화.

```yaml
train:
  amp: true                 # 이미 켜짐
  batch_size: 64            # 32 → 64 (GPU 메모리 충분하면)
device:
  num_workers: 8            # 4 → 8 (CPU 코어 많으면)
  pin_memory: true
```

### 4-5. 파라미터 budget 조정

[`scripts/profile_macs.py`](../scripts/profile_macs.py) 로 측정 후 조정:

```yaml
model:
  per_frame_fusion_dim: 256     # 192 → 256 — capacity ↑
  irffn:
    hidden_dim: 768             # 576 → 768 — IRFFN capacity ↑
```

---

## 5. From scratch 재학습

처음부터 다시 시작 (vocabulary 그대로, 가중치만 초기화).

```powershell
# 새 실험 이름으로 (기존 v0 보존)
python scripts/train.py --epochs 300 --name kslr_lab_v0_rerun

# 또는 같은 이름으로 덮어쓰기 (이전 결과 잃음)
python scripts/train.py --epochs 300
```

학습 시간 예상:

| 환경 | epoch당 | 300 epoch 총 |
|---|---|---|
| CPU only (i7-12700) | ~26 초 | ~2.2 시간 |
| GPU (RTX 3060) | ~5 초 | ~25 분 |
| GPU (RTX 4090) | ~2 초 | ~10 분 |

> 첫 epoch 은 워밍업 (cudnn benchmark, dataloader spawn) 으로 더 느림.

---

## 6. 학습 중 / 후 점검 포인트

### TensorBoard 모니터링

학습 중 다른 터미널에서:
```powershell
python -m tensorboard.main --logdir runs/kslr_lab_v0/tensorboard
# → http://localhost:6006
```

볼 것:
- `epoch/train_loss` — label smoothing floor (≈0.55) 근처에 도달하면 수렴
- `epoch/train_top1` vs `epoch/val_top1` — 두 곡선이 비슷하게 가야 generalize
- `train/lr` — warmup → cosine 정상 동작 확인

### 학습 끝나고 평가

```powershell
python scripts/evaluate.py --ckpt runs/kslr_lab_v0/best.pt --split test
```

- top-1 ≥ 90% — 만족
- top-3 ≥ 95% — 모델은 정답을 알지만 1위 결정에서 흔들림 → smoothing 강화
- per_class F1 — 특정 클래스가 80% 이하면 §3 (부분 fine-tune) 대상

### Confusion matrix 해석

`runs/kslr_lab_v0/confusion_matrix.png` 에서:
- 대각선이 진하면 OK
- 특정 행/열이 다른 데로 새면 → 그 두 클래스 동작이 비슷함

### 실시간 데모로 sanity check

```powershell
python scripts/demo.py --ckpt runs/kslr_lab_v0/best.pt
```

eval_report 와 demo 결과가 다를 수 있음 — eval은 학습 데이터 분포 안에서만, demo는 실제 환경에서.

---

## 7. 데이터 수집 best practices

### 다양성이 정확도를 결정

| 차원 | 권장 |
|---|---|
| **Signer 수** | 8명 (train) + 2명 (test) — signer-independent 평가 가능 |
| **각 signer × class** | 50회 이상 — 동작 자연 변동 흡수 |
| **각도 변화** | 정면 30, 좌측 10, 우측 10 (대략) |
| **속도 변화** | 평소 속도 + 약간 느림 + 약간 빠름 (각각 1/3) |
| **조명** | 정상 + 약간 어두움 + 측광 |
| **배경** | 단순 단색 + 약간 복잡 |
| **거리** | 90 cm + 120 cm + 60 cm |

### 검출률 80% 미만 clip은 폐기

`record.py` GUI 에서 검출률 화면 확인 → 빨간색이면 `Backspace` 로 다시.

### 라벨 정확성 (가장 중요)

- 키 `1` 누르고 다른 동작 녹화하면 영구 라벨 오염
- 같은 동작을 일관되게 — 시작 시점/끝 시점/궤적 비슷하게
- 단어와 동작이 헷갈리면 사전에 표준 reference 영상 (수어사전) 보고 통일

---

## 8. 새 모델 검증 후 배포

새 버전이 기존보다 나으면 → [`deployment.md`](deployment.md) §6 (버전 관리) 참고하여 v1.1 / v2.0 등 태그 부여 후 배포.
