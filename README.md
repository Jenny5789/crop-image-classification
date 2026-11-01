# 작물 이미지 분류 (CNN)

> **질문** CNN으로 고추·배추·오이 이미지를 얼마나 정확하게 구분할 수 있을까?
>
> **답** 검증 정확도가 60%대에 머문 원인을 데이터 구성에서 찾아 재분배한 뒤, 파라미터 15,491개의 경량 CNN으로 검증 정확도 86.4% 달성
>
> AI Hub 작물 이미지로 과적합 진단 → 정규화·모델 단순화 → 데이터 재분배 → 모델 비교까지 수행한 이미지 분류 프로젝트

## 기술 스택 및 역량

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-FF6F00?logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

| 기술 | 활용 |
|---|---|
| TensorFlow·Keras | CNN 설계·학습, 데이터 증강, Dropout, EarlyStopping |
| PyTorch | 동일 CNN 구조 재구현, 학습 루프 직접 작성 |
| scikit-learn·seaborn | 혼동행렬, 클래스별 Precision·Recall·F1 |
| matplotlib·PIL | 학습 곡선, 샘플 이미지 비교 |

| 역량 | 내용 |
|---|---|
| 과적합 진단 | Train·Validation 정확도 격차와 Validation Loss 추이로 과적합 판단 |
| 통제된 실험 설계 | 실험마다 한 가지 조건만 바꾸고 동일한 함수로 결과 기록 (실험 11회) |
| 원인 분석 | 모델 조정으로 성능이 정체되자 데이터 구성 차이를 직접 확인하고 재분배 |
| 모델 경량화 | GAP 적용·필터 축소로 파라미터 859,395 → 15,491개 |
| 비교 실험 | DNN vs CNN, Keras vs PyTorch를 같은 데이터·구조로 비교 |
| 추론 점검 | 학습과 예측의 전처리 불일치를 발견해 통일, 혼동행렬로 오분류 패턴 분석 |

## 프로젝트 구성

```
1부 데이터 준비·탐색 ──▶ 실험 1~7 (과적합 대응) ──▶ 데이터 구성 확인
──▶ 2부 데이터 재분배·재검증 (실험 8~11) ──▶ 3부 DNN vs CNN
──▶ 4부 Keras vs PyTorch ──▶ 5부 최종 모델 ──▶ 6부 예측 테스트·오분류 분석
```

| 노트북 | 내용 |
|---|---|
| [crop_classifier_1](notebooks/development/crop_classifier_1.ipynb) | 1부 데이터 준비·탐색, 실험 1~7, Training·Validation 구성 차이 확인 |
| [crop_classifier_2](notebooks/development/crop_classifier_2.ipynb) | 2부 데이터 재분배 후 재검증(실험 8~11), 3부 DNN vs CNN |
| [crop_classifier_3](notebooks/development/crop_classifier_3.ipynb) | 4부 Keras vs PyTorch, 5부 최종 모델, 6부 예측 테스트·혼동행렬, 전체 결론 |

## 주요 발견

### 1. 정규화 기법만으로는 과적합이 해결되지 않음

| 실험 | 변경 내용 | Train Acc | Val Acc |
|---|---|---|---|
| 1 | Baseline CNN (파라미터 859,395개) | 0.9994 | 0.6667 |
| 2 | + Data Augmentation | 0.9622 | 0.6511 |
| 3 | Dropout 0.3 → 0.5 | 0.9817 | 0.6711 |
| 4 | Data Augmentation + Dropout 0.5 | 0.9422 | 0.5756 |

- Baseline은 Train 정확도가 거의 1.0까지 올랐지만 Validation은 60%대에 머물고 Val Loss 증가 → 과적합
- 정규화 기법은 일부 개선(실험 3)에 그쳤고, 두 기법을 함께 쓰면 오히려 하락

### 2. 모델 단순화로 과적합 격차는 줄었지만 정확도는 60%대에 정체

| 실험 | 변경 내용 | 파라미터 | Val Acc | Val Loss |
|---|---|---|---|---|
| 5 | 필터 축소 + GlobalAveragePooling | 15,491 | 0.6511 | 1.0065 |
| 6 | 실험 5 + Data Augmentation | 15,491 | 0.6222 | 1.0469 |
| 7 | 추가 단순화 (필터 16-16-16) | 5,411 | 0.6667 | 0.8253 |

- Train-Val 정확도 격차: 33.27%p(실험 1) → 18.77%p(실험 7)
- 구조를 바꿔도 Validation 정확도가 60%대에 머물러 모델 외 요인을 확인하기로 함

### 3. 원인은 데이터 구성 차이, 재분배 후 성능 크게 개선

- 원본 Training·Validation 이미지를 직접 비교한 결과, 같은 작물이라도 피사체 구성이 달랐음
  - 고추·오이: Training은 열매 중심, Validation은 잎·줄기·꽃 포함
  - 배추: 어린 묘와 성장한 배추 등 생육 단계 차이
- 두 데이터를 합친 뒤 무작위로 다시 나누어 동일 모델로 재학습

| 실험 | 조건 | Val Acc |
|---|---|---|
| 5 | 재분배 전 | 0.6511 |
| 8 | 재분배 후, 실험 5와 동일 구조 | 0.7978 |
| 9 | + Epoch 30 | 0.8267 |
| **10** | **+ Epoch 50 (EarlyStopping) → 최종 모델** | **0.8644** |
| 11 | 재분배 후 추가 단순화 (파라미터 5,411개) | 0.8267 |

### 4. 최종 모델: 검증 정확도 86.4%, 오이↔고추 혼동이 가장 많음

| 클래스 | Precision | Recall | F1 |
|---|---|---|---|
| 배추 | 0.91 | 0.95 | 0.93 |
| 오이 | 0.85 | 0.75 | 0.80 |
| 고추 | 0.84 | 0.89 | 0.86 |

- 전체 정확도 86.44% (389 / 450), Train 정확도 87.22%로 격차가 작음
- 가장 많은 오분류: 오이 → 고추 24건, 반대 방향 고추 → 오이 15건

## 데이터

- 출처: AI Hub "노지 작물 질병 진단 이미지" 중 고추·배추·오이의 정상 이미지
- 원본 규모

| 작물 | Training | Validation |
|---|---|---|
| 고추 | 1,595 | 1,110 |
| 배추 | 825 | 1,183 |
| 오이 | 623 | 1,604 |

- 사용 데이터: 작물별 학습 600장, 검증 150장 무작위 추출 (총 2,250장, `random.seed(42)`)
- 입력: 128×128 RGB, Batch Size 32
- 원본 이미지 해상도: 4032×3024 (또는 3024×4032)

## 분석 기준 상세

- **통제된 실험**
  - 실험마다 한 가지 조건만 변경하고, 동일한 `run_experiment` 함수로 Train·Val 정확도, Loss, 파라미터 수, 학습 시간을 기록
- **과적합 판단**
  - 최종 정확도뿐 아니라 Train-Val 격차, Validation Loss 추이를 함께 확인
- **원인 확인 후 조치**
  - 샘플 이미지 차이만으로 원인을 단정하지 않고, 재분배 전후 동일 모델로 다시 비교
- **비교 실험 조건 통일**
  - DNN vs CNN: 같은 재분배 데이터 사용
  - Keras vs PyTorch: 같은 구조(파라미터 15,491개), 입력 크기, Batch Size, Optimizer, 최대 Epoch
- **추론 일관성**
  - 예측 함수의 이미지 로딩·리사이즈 방식을 학습·평가와 동일하게 맞춤

## 단계별 상세 결과

<details>
<summary><b>DNN vs CNN</b></summary>

| 항목 | DNN (Flatten + Dense) | CNN (최종 모델) |
|---|---|---|
| 파라미터 | 1,572,995 | 15,491 |
| Val Acc | 0.3333 | 0.8644 |

- DNN은 CNN보다 약 101.5배 많은 파라미터를 썼지만 Train·Val 정확도가 모두 약 33%(3개 클래스 무작위 수준)에 머묾
- CNN은 Convolution으로 공간적 특징을 추출해 훨씬 적은 파라미터로 높은 성능

</details>

<details>
<summary><b>Keras vs PyTorch</b></summary>

| 항목 | Keras | PyTorch |
|---|---|---|
| 총 파라미터 | 15,491 | 15,491 |
| Train Accuracy | 0.8722 | 0.8339 |
| Validation Accuracy | 0.8644 | 0.8044 |
| Validation Loss | 0.3261 | 0.4447 |
| 학습 Epoch | 50 | 50 |
| CPU 학습 시간 | 약 21분 | 약 5시간 27분 |

- 같은 CNN 구조를 PyTorch로 재구현하고 학습 루프(`zero_grad` → 순전파 → Loss → `backward` → `step`)를 직접 작성
- 초기 가중치, 데이터 셔플, 내부 최적화 차이가 있어 프레임워크 자체의 성능·속도 우위로 일반화하지 않음

</details>

<details>
<summary><b>예측 파이프라인 점검</b></summary>

- 예측 테스트에서 배추 이미지가 다른 작물로 예측되는 사례가 반복되어 전처리 과정을 점검
- 학습·평가와 예측의 이미지 로딩 방식이 달라 `keras.utils.load_img(..., interpolation='bilinear')`로 통일
- 통일 후에도 일부 오분류가 남아, 전처리 외에 모델 자체의 한계도 있음을 확인하고 혼동행렬로 추가 분석

</details>

## 한계

- 재분배 전후의 Validation 데이터 구성이 달라, 재분배 전후 정확도는 같은 기준의 비교가 아님
- 별도의 테스트 데이터 없이 Validation 데이터로 모델을 선택하고 최종 성능도 평가함
- 작물별 750장만 사용해 원본 데이터의 일부만 반영
- 오이·고추 간 혼동 원인(잎 모양, 촬영 구도 등)은 확인하지 않음
- 정상 이미지만 사용해 질병 이미지 분류는 다루지 않음

## 회고

- **이미지 분류 모델 개발 과정을 처음부터 끝까지 직접 수행**
  - 데이터 준비 → 과적합 대응 → 원인 분석 → 모델 비교 → 예측 테스트까지 진행
- **과정에서 체감한 점**
  - 모델 구조를 조정해도 성능이 정체되면 데이터 구성부터 확인해야 함
  - 학습과 추론의 전처리가 다르면 같은 모델이라도 결과가 달라짐

## 실행 방법

- AI Hub에서 "노지 작물 질병 진단 이미지" 데이터를 내려받아 `data/raw`에 저장
- 노트북의 데이터·모델 경로가 개인 PC 절대 경로로 되어 있어, 실행 전 각 노트북의 경로를 본인 환경에 맞게 변경 필요
- 노트북 순서대로 실행: `crop_classifier_1` → `crop_classifier_2` → `crop_classifier_3`

## 폴더 구조

```
crop-image-classification/
├── notebooks/
│   └── development/
│       ├── crop_classifier_1.ipynb   # 1부, 실험 1~7
│       ├── crop_classifier_2.ipynb   # 2~3부, 실험 8~11, DNN vs CNN
│       └── crop_classifier_3.ipynb   # 4~6부, 최종 모델, 전체 결론
└── README.md
```

## 데이터 출처

- AI Hub, 노지 작물 질병 진단 이미지
