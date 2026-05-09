# KSLR (Korean Sign Language Recognition) — 사용 설명서

이 폴더는 학습이 완료된 KSLR 모델을 **배포**하거나, 받은 사람이 **활용**하기 위한 가이드 모음입니다.

## 모델 개요

| 항목 | 값 |
|---|---|
| 모델 | KSLRNet (CNN + Transformer hybrid) |
| 파라미터 | 1.61M |
| 입력 | 16-frame webcam clip (≈0.53초 @ 30 FPS) |
| 입력 modality | MediaPipe 손/얼굴 landmark + RGB crop |
| 인식 클래스 | **10개 한국어 수어** — 안녕하세요 / 감사합니다 / 사랑 / 학교 / 친구 / 물 / 밥 / 가다 / 오다 / 끝 |
| 정확도 | top-1 94.85% / top-3 100% |
| 추론 지연 | 67 ms/clip (CPU), end-to-end ≈ 300 ms |
| 훈련 데이터 | 자체 lab dataset (signer-independent) |

## 어떤 문서를 읽어야 하나요?

| 당신의 상황 | 읽어야 할 문서 |
|---|---|
| **배포받은 모델을 빨리 실행해보고 싶다** | [`quick_start.md`](quick_start.md) |
| **모델을 다른 사람에게 전달하고 싶다 (제작자)** | [`deployment.md`](deployment.md) |
| **자체 앱/서비스에 모델을 통합하고 싶다** | [`python_api.md`](python_api.md) |
| **단어를 추가하거나 재학습하고 싶다** | [`training_guide.md`](training_guide.md) |
| **에러가 나는데 어떻게 해결하나** | [`troubleshooting.md`](troubleshooting.md) |

## 권장 학습 단계

1. 처음 사용 → **`quick_start.md`** 5분 안에 웹캠 데모 띄우기
2. 자체 프로젝트 통합 → **`python_api.md`** WebcamPipeline 사용 예제
3. vocabulary 변경/단어 추가 → **`training_guide.md`** 데이터 수집부터 재학습까지

## 주요 디렉토리 구조

```
handSignRecognition/
├── configs/lab_dataset.yaml       # 모든 hyperparameter / 경로
├── data_collection/               # 데이터 녹화 + MediaPipe wrapper
│   └── vocabulary.yaml            # 클래스 ID ↔ 한국어 라벨
├── models/                        # 모델 모듈 (Grain / LMHSA / ... / KSLRNet)
├── data/                          # Dataset / 정규화 / augment / split
├── engine/                        # Trainer / Evaluator / Loss
├── realtime/                      # Webcam 파이프라인 + 데모 overlay
├── scripts/                       # CLI 진입점 (record, train, evaluate, demo, ...)
├── utils/                         # checkpoint / logger / 한글 텍스트 overlay
├── tests/                         # 단위/통합 테스트
├── runs/                          # 실험 산출물 (gitignored — 별도 공유 필요)
│   └── kslr_lab_v0/
│       ├── best.pt                # ★ 최고 성능 체크포인트 (배포 핵심)
│       ├── latest.pt
│       ├── epoch_050.pt ... epoch_300.pt
│       ├── eval_report.json
│       ├── confusion_matrix.png
│       ├── config_resolved.yaml   # 학습 시 사용된 정확한 설정 스냅샷
│       └── tensorboard/           # 학습 곡선
├── data/raw/                      # 녹화된 clip (gitignored — 별도 공유)
└── models_assets/                 # MediaPipe .task 파일 (재다운로드 가능)
```

## 라이선스 / 인용

이 모델은 학술/연구 목적으로 자유롭게 사용 가능합니다. 사용 시 다음을 권장합니다:

- 데이터셋: 자체 수집 (10 signer × 10 class). 재배포 시 **개인정보(녹화된 얼굴/신체)** 동의 확인 필수
- 코드: MIT 호환 권장 (별도 LICENSE 파일 첨부 권장)
- 기반 architecture: CMT (CNN Meets Transformer) 변형 — 원 논문 인용 권장

## 빠른 명령 reference

```powershell
# 데모 실행
python scripts/demo.py --ckpt runs/kslr_lab_v0/best.pt

# 평가
python scripts/evaluate.py --ckpt runs/kslr_lab_v0/best.pt --split test

# 새 데이터 녹화
python scripts/record.py --signer 1

# 재학습
python scripts/train.py --epochs 300

# 모델 파라미터 / MAC 측정
python scripts/profile_macs.py
```

각 명령의 자세한 옵션은 `--help` 로 확인하세요.
