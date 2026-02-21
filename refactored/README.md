# DeepMoney v2

**LSTM/GRU 기반 주가 지수 중장기 예측 시스템**

> 코스피200 선물, S&P 500 등 주요 지수의 일봉 데이터를 입력으로
> 1주일(5일) · 1개월(20일) · 3개월(65일) 후의 지수 변동을 예측합니다.

---

## 목차

1. [버전 변경 이력](#버전-변경-이력)
2. [시스템 요구사항](#시스템-요구사항)
3. [설치](#설치)
4. [프로젝트 구조](#프로젝트-구조)
5. [빠른 시작](#빠른-시작)
6. [설정 파일](#설정-파일)
7. [명령줄 옵션](#명령줄-옵션)
8. [데이터 형식](#데이터-형식)
9. [모델 구조](#모델-구조)
10. [결과 해석](#결과-해석)
11. [자주 묻는 질문](#자주-묻는-질문)

---

## 버전 변경 이력

| 버전 | 주요 변경 사항 |
|------|---------------|
| v1.x | TensorFlow 1.x + `tf.contrib.rnn` + `tf.Estimator` |
| **v2.0** | **TensorFlow 2.x + Keras API + YAML 설정 + 모듈화 구조** |

### v1 → v2 주요 변경점

| 항목 | v1 (구버전) | v2 (현재) |
|------|------------|----------|
| TF 버전 | 1.x | 2.x (2.13+) |
| 모델 API | `tf.Estimator` | `keras.Model` |
| RNN 셀 | `tf.contrib.rnn.BasicLSTMCell` | `keras.layers.LSTM/GRU` |
| 설정 방법 | `path.py` 파일 직접 수정 | `config/config.yaml` |
| 코드 구조 | 기능별 분리 파일 7~9개 | 역할별 모듈 4개 |
| `from __future__` | 필요 (Python 2 호환) | 불필요 |
| `tf.flags` | 사용 | `argparse` 로 교체 |
| 진행 표시 | 없음 | Keras 기본 프로그레스바 |
| 조기 종료 | 없음 | EarlyStopping 콜백 |
| TensorBoard | 제한적 | 완전 지원 |
| 모델 저장 | `tf.estimator` 자동 | `model.save_weights()` |

---

## 시스템 요구사항

- Python 3.10 이상
- TensorFlow 2.13 이상
- RAM 8GB 이상 권장 (943 피처 × 배치)
- GPU 선택 (CUDA 11.8+ 지원 시 자동 감지)

---

## 설치

```bash
# 1. 저장소 클론 또는 압축 해제
cd deepmoney_v2

# 2. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. 패키지 설치
pip install -r requirements.txt
```

---

## 프로젝트 구조

```
deepmoney_v2/
│
├── main.py                  ← 메인 실행 파일
├── requirements.txt
│
├── config/
│   └── config.yaml          ← 모든 설정 (경로·하이퍼파라미터)
│
├── src/
│   ├── __init__.py
│   ├── config_loader.py     ← YAML → Config 데이터클래스 변환
│   ├── data_reader.py       ← CSV 로드 & LSTM 시퀀스 생성
│   ├── model.py             ← Keras LSTM/GRU 모델 정의
│   └── trainer.py           ← 학습·평가·예측 로직
│
├── data/                    ← CSV 데이터 파일 위치
├── models/                  ← 학습된 모델 가중치 저장
├── results/                 ← 예측 결과 CSV 저장
└── logs/                    ← TensorBoard 로그
```

---

## 빠른 시작

### 1. 데이터 준비

`data/` 디렉토리에 CSV 파일을 위치시킵니다.

```
data/kospi200f-943.csv
data/index-ma-std.csv     (선택 - 결과 비교용)
```

### 2. 설정 확인

`config/config.yaml` 에서 데이터 파일명과 날짜 범위를 설정합니다.

```yaml
data:
  file_name: "kospi200f-943.csv"
  train_start: "2000-01-01"
  test_start: "2017-01-02"

prediction:
  predict_term: 20    # 20일(1개월) 후 예측
```

### 3. 학습 및 예측 실행

```bash
# 기본 설정으로 실행
python main.py

# 1주일(5일) 예측, GRU 모델
python main.py --predict_term 5 --rnn_type gru

# 기존 모델 재사용하여 예측만
python main.py --no_train
```

### 4. TensorBoard 확인

```bash
tensorboard --logdir logs/
# 브라우저에서 http://localhost:6006 접속
```

---

## 설정 파일

`config/config.yaml` 의 주요 항목입니다.

```yaml
# 데이터 설정
data:
  file_name: "kospi200f-943.csv"  # CSV 파일명
  input_size: 943                  # 입력 피처 수
  train_start: "2000-01-01"        # 학습 시작일
  test_start: "2017-01-02"         # 테스트 시작일

# 예측 설정
prediction:
  predict_term: 20    # 5=1주, 20=1달, 65=3달
  step_interval: 1    # LSTM 스텝 간격 (일수)

# 모델 하이퍼파라미터
model:
  rnn_type: "lstm"    # "lstm" 또는 "gru"
  num_layers: 2       # RNN 레이어 수
  hidden_size: 200    # RNN 유닛 수
  num_steps: 20       # Look-back 윈도우 (시퀀스 길이)
  dropout_rate: 0.2   # 드롭아웃 비율

# 학습 설정
training:
  batch_size: 20
  epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 10
  bootstrap: false    # 부트스트랩 오버샘플링
  gradual_train: false # 점진적 학습
  model_reset: true   # 기존 모델 삭제 후 재학습
```

---

## 명령줄 옵션

```
python main.py [옵션]

옵션:
  --config CONFIG           YAML 설정 파일 경로 (기본: config/config.yaml)
  --data_path DATA_PATH     데이터 파일 전체 경로
  --predict_term INT        예측 기간: 5, 20, 65
  --hidden_size INT         LSTM 히든 유닛 수
  --num_layers INT          LSTM 레이어 수
  --num_steps INT           시퀀스 길이 (look-back window)
  --batch_size INT          배치 크기
  --epochs INT              최대 에포크 수
  --learning_rate FLOAT     학습률
  --rnn_type {lstm,gru}     RNN 셀 유형
  --model_reset             기존 모델 삭제 후 재학습
  --no_train                학습 없이 예측만 수행
```

---

## 데이터 형식

### 입력 CSV (예: `kospi200f-943.csv`)

| 컬럼 위치 | 내용 |
|----------|------|
| 0 | 일련번호 |
| 1 (date) | 날짜 (YYYY-MM-DD) |
| 2 ~ 944 | 943개 입력 피처 (정규화된 지수 값들) |
| 945 | 타겟: 1일 후 지수 변동 |
| 946 | 타겟: 5일 후 지수 변동 |
| 947 | 타겟: 20일 후 지수 변동 |
| 948 | 타겟: 65일 후 지수 변동 |

### 보조 CSV (`index-ma-std.csv`) - 선택

예측 결과 시각화를 위한 실제 지수값·날짜·표준편차 정보.

---

## 모델 구조

```
입력 (batch, num_steps, 943)
    │
    ▼
LSTM Layer 1 (hidden=200, return_sequences=True)
    │
    ▼
LSTM Layer 2 (hidden=200, return_sequences=True)
    │
    ▼
TimeDistributed Dense (units=1, activation=linear)
    │
    ▼
출력 (batch, num_steps, 1)
```

- **손실 함수**: MSE (Mean Squared Error)
- **최적화**: Adam (lr=0.001, 기본값)
- **정규화**: L2 (0.001) + 드롭아웃 (0.2)
- **조기 종료**: validation loss 기준 patience=10 에포크

---

## 결과 해석

예측 완료 후 `results/` 폴더에 두 파일이 생성됩니다.

### `result_*.csv`

| 컬럼 | 설명 |
|------|------|
| `date_base` | 예측 기준일 |
| `date_pred` | 예측 대상일 |
| `real_diff` | 실제 지수 변동 |
| `pred_diff` | 예측 지수 변동 |
| `index_today` | 예측 기준일의 실제 지수 |
| `index_real` | 예측 대상일의 실제 지수 |
| `index_pred` | 예측 대상일의 예측 지수 |
| `profit` | 방향 적중 시 +, 미적중 시 - |

### `result_*_summary.txt`

```
예측 기간: 20일
방향 정확도: 0.6234    ← 상승/하락 방향 적중률
Precision: 0.6591
Recall: 0.5872
F1 Score: 0.6210
총 누적 손익: 1245.30
```

---

## 자주 묻는 질문

**Q. `FileNotFoundError: 데이터 파일을 찾을 수 없습니다`**  
A. `data/` 폴더에 CSV 파일이 있는지, `config.yaml`의 `file_name`이 정확한지 확인하세요.

**Q. 학습이 너무 느립니다**  
A. `batch_size`를 32~64로 늘리거나, `hidden_size`를 100으로 줄여보세요. GPU가 있다면 TF가 자동으로 사용합니다.

**Q. `predict_term`에 다른 값을 사용할 수 있나요?**  
A. 현재 1, 5, 20, 65일만 지원합니다. CSV 타겟 컬럼이 이 값들에 맞춰져 있어야 합니다.

**Q. 기존 v1 모델을 v2에서 사용할 수 없나요?**  
A. v1은 `tf.Estimator` 형식으로 저장되어 v2와 호환되지 않습니다. v2로 처음부터 재학습해야 합니다.

**Q. S&P 500 등 다른 지수도 예측할 수 있나요?**  
A. 동일한 CSV 형식으로 데이터를 준비하면 됩니다. `config.yaml`의 `file_name`만 변경하면 됩니다.

---

## 라이선스

Copyright 2018-2025 Bumghi Choi. All Rights Reserved.  
Apache License 2.0
