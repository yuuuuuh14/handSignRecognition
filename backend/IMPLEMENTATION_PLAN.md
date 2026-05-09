# KSLR Implementation Plan

> 이 문서는 구현 에이전트(main agent)가 코드를 작성할 때 따라야 할 **확정된 사양**입니다.
> 원본 설계 문서 `document.md`, `architecture.md`, `transformer.md` 가 정의한 모델 아키텍처(Grain / LPU / LMHSA / CNN / IRFFN)를 보존하되, 다음 사항이 추가/수정되었습니다:
> - 입력: raw RGB → **MediaPipe Tasks 기반 hybrid 입력** (landmark + RGB crop)
> - 목적: KSL Benchmark 재현 → **자체 lab dataset 구축 + 실시간 웹캠 demo**
> - 클래스 수: 77 → **10**
> - 시간 모델링: single frame → **16-frame clip**
>
> 충돌 시 이 문서가 우선합니다.

---

## 1. 프로젝트 목표

연구실 고유의 한국어 수어(Korean Sign Language) dataset을 자체 구축하고, MediaPipe + 경량 hybrid CNN-Transformer 모델로 **실시간 웹캠** 기반 수어 인식 프로그램을 만든다.

- 인식 대상: 10개 KSL 클래스 (vocabulary는 별도 정의)
- 동작 환경: 통제된 실내 (고정 조명/배경/카메라)
- 추론 주체: 단일 사용자 (시연용 데모)
- End-to-end latency 목표: **300ms**
- 모델 파라미터 목표: **~1.5M** (edge/mobile 친화)

---

## 2. 확정 사양

| 항목 | 값 | 비고 |
|---|---|---|
| 클래스 수 | 10 | `data_collection/vocabulary.yaml`에 라벨 정의 |
| Signer 수 | 10명 | train 8 / test 2 (signer-independent) |
| Recordings per (signer, class) | ~50회 | 총 ~5,000 clip |
| Clip 길이 | 16 frame | ≈ 0.53초 @ 30 FPS |
| Webcam FPS | 30 | OpenCV 기본값 |
| 입력 modality | hybrid: landmark + RGB crop | MediaPipe Tasks API |
| Hand landmarks | (T=16, 2 hands, 21 points, 3 dims) | 없는 손 → 0-pad + mask |
| Face landmarks | (T=16, ~50 points, 3 dims) | 눈썹+입 subset (FACE_OVAL/LIPS/EYEBROW) |
| Hand crop | (T=16, 2, 3, 64, 64) RGB | landmark bbox + margin으로 crop |
| Face crop | (T=16, 3, 64, 64) RGB | landmark bbox + margin으로 crop |
| Latency | 300ms end-to-end | sliding window stride=8 frame |
| Optimizer | Adam, lr=1e-3, wd=1e-4 | `architecture.md`와 동일 |
| Dropout | 0.1 | `architecture.md`와 동일 |
| Epochs | 300 (target) | early stopping 적용 |
| Loss | CrossEntropyLoss(label_smoothing=0.1) | |
| Schedule | Cosine annealing + 5-epoch warmup | |
| Batch size | 32 | GPU 메모리 따라 조정 |
| Train/test split | signer 기준 8/2 | random split 금지 |

---

## 3. 프로젝트 디렉토리 구조

```
handSignRecognition/
├── CLAUDE.md
├── IMPLEMENTATION_PLAN.md          # 본 문서
├── document.md                     # 원본 설계 (Korean)
├── architecture.md                 # 원본 설계
├── transformer.md                  # 원본 설계
├── requirements.txt
│
├── configs/
│   └── lab_dataset.yaml            # 학습 hyperparameter, 경로
│
├── data_collection/                # Phase 1: 데이터 수집
│   ├── __init__.py
│   ├── recorder.py                 # 웹캠 녹화 GUI (OpenCV + 키보드)
│   ├── mediapipe_runner.py         # MediaPipe Tasks wrapper
│   ├── quality_check.py            # 검출률 검사 / 통계
│   └── vocabulary.yaml             # 10개 클래스 라벨 정의
│
├── data/                           # Phase 3: 학습용 데이터 처리
│   ├── __init__.py
│   ├── dataset.py                  # ClipDataset (raw 디렉토리 → tensor)
│   ├── normalizer.py               # wrist-relative / nose-relative 정규화
│   ├── augment.py                  # landmark noise, crop jitter, time jitter
│   └── splits.py                   # signer-independent split 생성
│
├── models/                         # Phase 4: 모델 모듈
│   ├── __init__.py
│   ├── grain.py                    # GrainModule (이미지 stream용)
│   ├── landmark_embed.py           # Linear projection (landmark stream용)
│   ├── lpu.py                      # LocalPerceptionUnit
│   ├── lmhsa.py                    # LightweightMultiHeadSelfAttention
│   ├── transformer_block.py        # LPU + LMHSA + MLPConv
│   ├── cnn_branch.py               # 4-block CNN
│   ├── temporal_aggregator.py      # 2-layer Temporal Transformer
│   ├── irffn.py                    # InvertedResidualFFN classifier head
│   └── kslr_net.py                 # 전체 조합 (top-level)
│
├── engine/                         # Phase 5: 학습/평가 엔진
│   ├── __init__.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── losses.py
│
├── realtime/                       # Phase 9: 실시간 데모
│   ├── __init__.py
│   ├── frame_buffer.py             # sliding window (16 frame, stride 8)
│   ├── webcam_pipeline.py          # 웹캠 → MediaPipe → 모델 inference
│   └── demo_app.py                 # 화면 표시 + 예측 결과 오버레이
│
├── utils/
│   ├── __init__.py
│   ├── param_count.py              # per-module param 출력
│   ├── checkpoint.py
│   └── logger.py
│
├── scripts/
│   ├── record.py                   # `python scripts/record.py --signer 1`
│   ├── extract_landmarks.py        # 기존 raw clip의 landmark 재추출
│   ├── train.py                    # `python scripts/train.py --config configs/lab_dataset.yaml`
│   ├── evaluate.py                 # checkpoint 평가
│   ├── profile_macs.py             # MAC/param 검증
│   └── demo.py                     # `python scripts/demo.py --ckpt best.pt`
│
├── tests/
│   ├── test_grain.py
│   ├── test_lmhsa.py
│   ├── test_kslr_net.py            # forward shape, param count 검증
│   └── test_dataset.py
│
└── data/raw/                       # ★ 수집된 clip 저장 위치 (gitignore)
    └── {signer_id}/{class_id}/{timestamp}/...
```

---

## 4. 데이터 수집 사양 (Phase 1: 가장 먼저 구현)

### 4.1 Recorder (`data_collection/recorder.py`) UI 흐름

1. 실행: `python scripts/record.py --signer {1..10}`
2. 윈도우 창에 webcam preview + landmark overlay 실시간 표시
3. **키보드 입력**:
   - `1`–`9`, `0` → 클래스 선택 (10개 → 키 10개)
   - `Space` → 즉시 16 frame 캡처 시작 (캡처 중 화면에 progress bar)
   - `Enter` → 직전 캡처 저장
   - `Backspace` → 직전 캡처 폐기 후 재시도
   - `q` → 종료
4. 캡처 직후 화면에 다음 표시:
   - 16 frame 모두에서 hand 검출 성공 비율
   - face 검출 성공 비율
   - 양손 모두 검출된 frame 수
5. 검출률 < 80%이면 빨간색 경고 (저장은 가능, 사용자 판단)

### 4.2 저장 형식

clip 1개당 디렉토리:
```
data/raw/{signer_id}/{class_id}/{YYYYMMDD_HHMMSS_mmm}/
├── frames.npy            # (16, H, W, 3) uint8, 원본 webcam 해상도
├── hand_landmarks.npy    # (16, 2, 21, 3) float32, MediaPipe normalized coords
├── hand_world.npy        # (16, 2, 21, 3) float32, MediaPipe world coords (선택)
├── face_landmarks.npy    # (16, N_face_subset, 3) float32
├── hand_mask.npy         # (16, 2) bool — 해당 frame/hand 검출 성공 여부
├── face_mask.npy         # (16,) bool
└── meta.json             # {signer_id, class_id, timestamp, fps, vocabulary_version}
```

### 4.3 MediaPipe 설정

`mediapipe.tasks.python.vision.HandLandmarker`:
- `num_hands=2`
- `running_mode=VIDEO`
- `min_hand_detection_confidence=0.5`
- `min_hand_presence_confidence=0.5`
- `min_tracking_confidence=0.5`

`mediapipe.tasks.python.vision.FaceLandmarker`:
- `num_faces=1`
- `running_mode=VIDEO`
- `output_face_blendshapes=False`
- `output_facial_transformation_matrixes=False`

Face landmark subset (468 → ~50): `FACE_OVAL`, `LIPS`, `LEFT_EYEBROW`, `RIGHT_EYEBROW`. MediaPipe의 `FaceLandmarksConnections` 상수에서 인덱스 추출.

### 4.4 Crop 추출 규칙

- **Hand crop**: 21 landmark 의 normalized (x,y) bbox + 25% margin → 원본 frame에서 crop → 64×64 resize. 검출 실패 시 zero image (3,64,64).
- **Face crop**: 동일 방식, face landmark bbox + 15% margin.
- 양손 순서: MediaPipe handedness label로 (Left, Right) 정렬. 1개만 검출 시: handedness에 맞는 슬롯에 할당, 다른 슬롯은 zero + mask=False.

---

## 5. 학습 데이터 처리 (`data/`)

### 5.1 정규화 (`normalizer.py`)

- **Hand**: 손목(landmark 0) 기준 좌표 평행이동 → 중지 MCP(landmark 9)와의 거리로 scale 정규화 → translation/scale 무관해짐.
- **Face**: 코끝(landmark 1) 기준 평행이동 → 양쪽 입꼬리 거리로 scale 정규화.
- 정규화 후 좌표 범위는 대략 [-1, 1] 권장.

### 5.2 Augmentation (`augment.py`, train only)

- Landmark에 가우시안 noise (σ=0.01, 정규화 좌표 기준)
- Random rotation ±15° (모든 landmark에 동일 회전)
- Random scale ×[0.9, 1.1]
- Time jitter: 16 frame 내에서 ±2 frame shift (frames.npy에서 다른 시작점)
- Hand crop / Face crop: ColorJitter(brightness=0.2, contrast=0.2)
- **Horizontal flip 금지** — KSL의 양손 swap은 의미가 달라질 수 있음

### 5.3 Dataset (`dataset.py`)

```python
class ClipDataset(Dataset):
    def __getitem__(idx) -> dict:
        return {
            "hand_lm": Tensor (16, 2, 21, 3),  # 정규화 후
            "face_lm": Tensor (16, N_face, 3),
            "hand_crop": Tensor (16, 2, 3, 64, 64),
            "face_crop": Tensor (16, 3, 64, 64),
            "hand_mask": Tensor (16, 2),
            "face_mask": Tensor (16,),
            "label": int (0-9),
        }
```

---

## 6. 모델 아키텍처

### 6.1 전체 구조

```
입력 (clip 1개)
├─ hand_lm    (B, 16, 2, 21, 3)
├─ face_lm    (B, 16, N_face, 3)
├─ hand_crop  (B, 16, 2, 3, 64, 64)
└─ face_crop  (B, 16, 3, 64, 64)

↓ Per-frame Spatial Encoding (16 frame 각각 독립 처리, weight 공유)

  Hand crops  ──[reshape (B*16*2, 3, 64, 64)]── Grain → CNN 4-block ──→ GAP → 128-d
                                                                          ↓
                                                          [reshape (B*16, 2, 128)] → flatten → 256-d  (per-frame, hand)

  Face crop   ──[reshape (B*16, 3, 64, 64)]── Grain → CNN 4-block ──→ GAP → 128-d  (per-frame, face)

  Hand LM     ──[flatten (B*16, 126)]── Linear(126→64) → LPU → LMHSA(d=64,h=4) → MLPConv → 64-d
  Face LM     ──[flatten (B*16, N_face*3)]── Linear(→64) → LPU → LMHSA(d=64,h=4) → MLPConv → 64-d

  Per-frame concat: 256 + 128 + 64 + 64 = 512-d
  → Linear(512→256) (frame token으로 사용)

↓ Temporal Aggregation: (B, 16, 256)

  2× Temporal Transformer Block
    - PreNorm + MultiHeadAttention(d=256, h=4)
    - PreNorm + FFN(256→512→256, GELU, dropout 0.1)
  → Mean pool over T → (B, 256)

↓ Classifier

  IRFFN: Linear(256→1024) + GELU + Dropout(0.1) + Linear(1024→256) + 잔차
  → Linear(256→10) → logits
```

### 6.2 모듈별 사양

#### GrainModule (`models/grain.py`)
원본 설계 충실히 반영:
```
Conv2d(in, 32, k=3, s=1, p=1) + BN + GELU
Conv2d(32, 32, k=3, s=1, p=1) + BN + GELU
Conv2d(32, 32, k=3, s=2, p=1) + BN + GELU
LayerNorm (channel last)
```
출력: (B, 32, H/2, W/2). 64×64 입력 → 32×32 출력.

#### CNN Branch (`models/cnn_branch.py`)
4 블록, channel schedule **32 → 64 → 64 → 128**:
```
Block 1: Conv(32→64, k=3, s=1) + BN + GELU + MaxPool 2×2
Block 2: Conv(64→64, k=3, s=1) + BN + GELU
Block 3: Conv(64→128, k=3, s=1) + BN + GELU + MaxPool 2×2
Block 4: Conv(128→128, k=3, s=1) + BN + GELU
GlobalAveragePool → 128-d
```
입력 32×32 → 두 번 pool → 8×8 → GAP → 128-d.

#### LPU (`models/lpu.py`)
```python
LPU(x) = DepthwiseConv2d(x, k=3, p=1) + x
```
landmark stream에서는 1D 버전: `DepthwiseConv1d` 또는 sequence가 1개 token이라면 LPU 생략 후 단순 Linear residual로 대체. **본 프로젝트의 landmark stream은 frame당 단일 token이므로 LPU는 1D conv 형태로 구현하되 sequence 길이가 1이면 identity로 동작.**

#### LMHSA (`models/lmhsa.py`)
```
Q = Linear(x)                              # (B, N, d)
K = Conv1d(x, k=2, s=2)  → Linear         # (B, N/2, d)  — landmark stream에서는 N=1이라 stride 적용 어려움
V = Conv1d(x, k=2, s=2)  → Linear         # (B, N/2, d)
B = nn.Parameter (h, N, N/2)               # relative position bias
attn = softmax(QK^T / sqrt(d_k) + B) · V
```
**구현 주의**: landmark stream의 sequence 길이가 1인 경우 stride reduction이 무의미하므로, transformer_block 내부에서 N>1일 때만 stride를 적용. 본 설계에서는 landmark feature를 단일 token으로 취급하므로 LMHSA는 사실상 self-attention 1-token degenerate case가 됨 → **landmark branch는 단순히 Linear+GELU 2-layer MLP로 대체해도 무방**. 첫 구현은 원본 LMHSA를 유지하되 sequence=1일 때 fallback path 분기.

#### Temporal Aggregator (`models/temporal_aggregator.py`)
표준 Transformer encoder block 2개:
```python
class TemporalBlock(nn.Module):
    def __init__(d=256, heads=4):
        self.norm1 = LayerNorm(d)
        self.attn  = MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.norm2 = LayerNorm(d)
        self.ffn   = Sequential(Linear(d, 2*d), GELU(), Dropout(0.1), Linear(2*d, d))
    def forward(x):  # x: (B, 16, 256)
        x = x + self.attn(self.norm1(x), ..., need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x
```
Sinusoidal 또는 learned positional embedding (16 frame, d=256) 추가.

#### IRFFN (`models/irffn.py`)
```
Linear(256→1024) + GELU + Dropout(0.1) + Linear(1024→256)
+ 잔차 add (입력 256-d와)
```
이후 Linear(256→10)으로 logits 출력.

### 6.3 파라미터 예산 (목표 1.5M)

| 모듈 | 파라미터 추정 |
|---|---|
| Grain (이미지용, hand+face 공유 가능) | ~28K |
| Hand CNN Branch (weight 공유) | ~150K |
| Face CNN Branch | ~150K |
| Hand LM Embedding + Transformer block | ~50K |
| Face LM Embedding + Transformer block | ~70K |
| Per-frame fusion Linear(512→256) | ~131K |
| Temporal Aggregator (2 layer, d=256) | ~530K |
| Positional Embedding (16, 256) | ~4K |
| IRFFN (256→1024→256) | ~525K |
| Classifier (256→10) | ~3K |
| **합계** | **~1.64M** |

→ 목표 약간 초과. tuning 여지: temporal d를 256→192 축소하거나 IRFFN 확장비를 4×→3× 축소. **첫 구현 후 `scripts/profile_macs.py`로 측정 → 1.4–1.6M 범위로 조정**.

---

## 7. 학습 파이프라인

### 7.1 Trainer (`engine/trainer.py`)

- AMP (mixed precision) 사용 권장
- Gradient clipping: `clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Checkpoint 저장: best (val acc 기준), latest, every 50 epoch
- Logging: TensorBoard 또는 wandb (configurable)
- Early stopping: val acc plateau 30 epoch

### 7.2 Loss

```python
nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 7.3 LR Schedule

```python
warmup 5 epoch (linear: 0 → 1e-3)
이후 cosine annealing (1e-3 → 1e-5) over 295 epoch
```

---

## 8. 평가 (`engine/evaluator.py`)

추적 metric:
- Top-1 accuracy (primary)
- Top-3 accuracy
- Per-class precision / recall / F1
- Confusion matrix (이미지 저장)
- 평균 inference time per clip (ms)

평가 단위: signer-independent test split (2명) 전체.
출력: `runs/{exp_id}/eval_report.json` + `confusion_matrix.png`.

---

## 9. 실시간 추론 (`realtime/`)

### 9.1 Frame Buffer (`frame_buffer.py`)

- Circular buffer, capacity = 16 frame
- 매 frame: webcam → MediaPipe → buffer push
- Stride = 8 frame 마다 buffer 전체를 모델에 forward → 예측 출력
- 예측 결과는 EMA 또는 majority vote로 안정화 (선택)

### 9.2 Webcam Pipeline (`webcam_pipeline.py`)

```
while True:
    frame = cv2.VideoCapture.read()
    hand_lm, face_lm, hand_crop, face_crop = mediapipe_runner(frame)
    buffer.push(...)
    if buffer.is_full() and step % stride == 0:
        clip = buffer.snapshot()
        with torch.no_grad():
            logits = model(**clip)
        prediction = vocabulary[logits.argmax()]
        confidence = softmax(logits).max()
    overlay(frame, prediction, confidence, mediapipe_landmarks)
    cv2.imshow(...)
```

### 9.3 Latency 예산 (300ms)

| 단계 | 시간 |
|---|---|
| MediaPipe Hand+Face per frame | ~25ms |
| Buffer fill (stride 8) | 8 × 33ms ≈ 264ms |
| Model forward | ~30–50ms |
| **합계** | **~300–340ms** |

CPU 환경에서 모델 forward가 100ms 초과 시: temporal d 축소, ONNX export, 또는 stride 4 → 12로 변경.

---

## 10. 구현 순서 (Build Order)

순서를 **반드시** 지킬 것 — 각 단계가 다음 단계 의존성을 가짐.

| Phase | 작업 | 산출물 | 검증 방법 |
|---|---|---|---|
| **0** | `requirements.txt`, `configs/lab_dataset.yaml` 작성 | 환경 setup | `pip install -r requirements.txt` 성공 |
| **1** | `data_collection/mediapipe_runner.py` | MediaPipe wrapper | 단일 이미지에서 landmark 추출 |
| **2** | `data_collection/recorder.py` + `scripts/record.py` | 녹화 GUI | 본인 1명으로 시범 클립 10개 저장 |
| **3** | 시범 데이터 수집 (1–2명, 클래스당 ~10회) | ~100–200 clip | `data/raw/` 채워짐 |
| **4** | `data/dataset.py`, `normalizer.py`, `splits.py` | DataLoader | iter 1 batch 정상 출력 |
| **5** | `models/grain.py`, `lpu.py`, `lmhsa.py` + `tests/test_*.py` | 단위 모듈 | shape 단언 통과 |
| **6** | `models/cnn_branch.py`, `temporal_aggregator.py`, `irffn.py` | 단위 모듈 | shape 단언 통과 |
| **7** | `models/kslr_net.py` + `scripts/profile_macs.py` | top-level model | param count 1.4–1.6M 검증 |
| **8** | `engine/trainer.py`, `evaluator.py`, `losses.py` | 학습 엔진 | 시범 데이터로 1 epoch overfit 확인 |
| **9** | 본 데이터 수집 (10명, 클래스당 50회) | ~5,000 clip | quality_check 통과 |
| **10** | 본 학습 + hyperparameter 조정 | best.pt | val acc 측정 |
| **11** | `realtime/webcam_pipeline.py`, `demo_app.py`, `scripts/demo.py` | 실시간 데모 | 카메라 앞에서 동작 확인 |

**Phase 2 완료 전에는 Phase 4 이후 작업을 시작하지 말 것.** 데이터 형식은 recorder가 결정하므로 recorder가 먼저 안정화되어야 dataset loader가 일관됨.

---

## 11. 의존성 (`requirements.txt`)

```
torch>=2.0
torchvision>=0.15
mediapipe>=0.10.9
opencv-python>=4.8
numpy>=1.24
pyyaml>=6.0
einops>=0.7
ptflops>=0.7
tensorboard>=2.14
tqdm>=4.66
```

선택 (개발 편의):
```
wandb
matplotlib
seaborn
```

---

## 12. 미해결 항목 (구현 중 결정)

1. **Vocabulary 10개 단어/자모 정의** — `data_collection/vocabulary.yaml` 채워야 함
2. **Sliding window stride** — 기본 8 frame, 데모 시 latency 측정 후 조정
3. **MediaPipe world coords (z) 사용 여부** — 현재 plan: 사용 (3D 정보가 표정 분류에 도움)
4. **Hand weight sharing 방식** — 양손 CNN을 단순 same-module로 처리 (좌우 손이 다르게 학습되지 않도록 mirror augmentation 추가 검토)
5. **Face landmark subset 정확한 인덱스** — MediaPipe `FACEMESH_CONTOURS` 상수에서 추출, Phase 1에서 확정
6. **Temporal positional encoding** — sinusoidal 우선, 성능 부족 시 learned로 교체

---

## 13. Naming Conventions (구현 시 준수)

- 모듈/클래스 명은 본 문서의 영어 이름 그대로: `GrainModule`, `LocalPerceptionUnit`, `LightweightMHSA`, `CNNBranch`, `TemporalAggregator`, `IRFFN`, `KSLRNet`
- 파일명 snake_case
- Korean 식별자 금지 (모듈명, 변수명) — 단, vocabulary.yaml 라벨과 화면 출력 텍스트는 Korean 가능
- 좌표 dimension 순서: `(B, T, ...)` 일관 유지. T를 dimension 1에 두는 것이 PyTorch 관습에 맞음.

---

## 14. 첫 작업 명령

main agent에게:
> `IMPLEMENTATION_PLAN.md`를 따라 Phase 0 → Phase 2까지 작업하라.
> Phase 2 완료 시점에 user에게 시범 녹화를 요청하고 검토받은 후 Phase 3 이후로 진행한다.
> Vocabulary 10개가 정해지지 않았다면 placeholder (`class_0`...`class_9`)로 시작.
