# Korean Sign Language Recognition (KSLR)

Hybrid CNN-Transformer 모델 기반 한국 수어 인식 시스템.
MediaPipe Tasks로 추출한 hand/face landmark와 RGB crop을 multi-modal 입력으로 사용하며, 16-frame clip 단위로 동작 분류 후 실시간 webcam 데모로 추론한다.

> **Status: Phase 0 — design freeze.** 본 repo는 현재 설계 문서와 패키지 스캐폴드만 포함한다. 구현은 [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) Phase 1부터 진행 예정.

## 핵심 설계

```
Input (16 frames) → MediaPipe (Hand + Face) → ┬─ Hand landmark embed (Transformer) ┐
                                              ├─ Face landmark embed (Transformer) ├─ per-frame fusion
                                              ├─ Hand RGB crop  (CNN: Grain+blocks) │      │
                                              └─ Face RGB crop  (CNN: Grain+blocks) ┘      │
                                                                                            ▼
                                                                       Temporal Transformer (16 tokens)
                                                                                            ▼
                                                                        IRFFN classifier → 10-class logits
```

- **Grain Module** — ViT 패치 대신 conv stem 으로 채널 32, 2× downsample.
- **LPU + LMHSA + IRFFN** — `architecture.md`, `transformer.md`, [document.md](document.md) 참조.
- **Temporal Transformer** — clip-level aggregator.
- **Parameter budget**: ≈1.5M (자세한 분해는 IMPLEMENTATION_PLAN.md §6).
- **Latency target**: end-to-end ≤300 ms @ webcam 30 FPS.

## 데이터셋

자체 lab 수집 데이터 — 10 classes × 10 signers × 50 recordings ≈ 5,000 clip.
- Signer-independent split: train [1..8] / test [9, 10] (random split 금지).
- Vocabulary: [data_collection/vocabulary.yaml](data_collection/vocabulary.yaml) (현재 placeholder, 라벨 인덱스 0..9는 학습 후 변경 금지).
- **원본 영상은 repo에 commit 하지 않는다.** (`.gitignore`로 `data/raw/` 차단; IRB 동의 범위 내 별도 저장소에서 관리.)

## 디렉토리 구조

```
.
├─ configs/                # 학습/평가/실시간 YAML (lab_dataset.yaml = single source of truth)
├─ data_collection/        # MediaPipe 기반 webcam 수집 파이프라인 (Phase 1)
├─ data/                   # Dataset / Normalizer / Augment (Phase 2)
├─ models/                 # Grain, LPU, LMHSA, IRFFN, KSLRNet 본체 (Phase 3)
├─ engine/                 # Trainer, Evaluator (Phase 4)
├─ realtime/               # 실시간 추론 데모 (Phase 5)
├─ scripts/                # 모델/도구 다운로드, 진단 스크립트
├─ tests/                  # 단위 테스트
├─ utils/                  # 공용 유틸
├─ models_assets/          # MediaPipe .task 파일 (gitignored, 다운로드 필요)
├─ architecture.md         # 모델 아키텍처 영문 사양
├─ transformer.md          # Transformer 변형 모듈 사양
├─ document.md             # 한국어 원 설계 문서
├─ IMPLEMENTATION_PLAN.md  # 확정 구현 계획 — 모든 결정의 single source of truth
└─ CLAUDE.md               # AI assistant 가이드
```

## 개발 환경 설정

Python 3.10+ 권장. Windows / Linux 모두 동작 목표.

```bash
# 1. 가상환경 생성
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. MediaPipe Tasks 모델 파일 다운로드
python scripts/download_models.py
```

## 사용 (Phase 1+ 구현 후)

> 아래 명령은 IMPLEMENTATION_PLAN.md 각 Phase가 완료된 후 활성화된다.

```bash
# 데이터 수집 (Phase 1)
python -m data_collection.collect --signer 1 --class 0

# 학습 (Phase 4)
python -m engine.trainer --config configs/lab_dataset.yaml

# 평가
python -m engine.evaluator --config configs/lab_dataset.yaml --ckpt runs/<exp_id>/best.pt

# 실시간 webcam 데모 (Phase 5)
python -m realtime.demo --config configs/lab_dataset.yaml --ckpt runs/<exp_id>/best.pt
```

## 참고 문서

- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — Phase 0~5 구현 계획, 모듈 사양, 파라미터 예산, 수집 프로토콜
- [architecture.md](architecture.md) — 영문 아키텍처 사양
- [transformer.md](transformer.md) — Transformer 모듈 (LPU/LMHSA/IRFFN) 상세
- [document.md](document.md) — 한국어 원 설계 문서

## 데이터 / 개인정보 정책

본 프로젝트는 lab 구성원의 webcam 영상을 수집한다. 다음을 준수한다.
- 원본 영상(`data/raw/`)은 repo에 절대 포함하지 않는다.
- IRB 동의서 범위 내에서만 수집/보관/공유한다.
- 외부 공개가 필요한 경우 landmark vector 등 비식별 처리된 형태만 배포한다.

## License

코드는 [MIT License](LICENSE) 하에 배포된다.
데이터셋(lab webcam 수집본)은 본 라이선스에 포함되지 않으며, IRB 동의 범위 내에서만 별도로 관리된다.
