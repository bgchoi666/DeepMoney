# DeepMoney v2 프로그램 설명서

**LSTM 기반 주가 지수 중장기 예측 시스템 기술 문서**

---

## 1. 시스템 개요

DeepMoney는 코스피200 선물, S&P 500 등 주요 주가 지수의 일봉 데이터를 기반으로 향후 5일(1주), 20일(1달), 65일(3달) 후의 지수 변동을 예측하는 딥러닝 시스템입니다.

핵심 알고리즘으로는 LSTM(Long Short-Term Memory) 또는 GRU(Gated Recurrent Unit) 순환 신경망을 사용하며, Many-to-Many 구조를 채택하여 입력 시퀀스의 모든 타임스텝에서 예측을 수행합니다.

### 1.1 원본(v1)과의 비교

v1은 TensorFlow 1.x 시절의 저수준 API인 `tf.contrib.rnn`과 `tf.Estimator`를 사용하여 작성되었습니다. v2는 TensorFlow 2.x의 Keras 고수준 API로 완전히 재작성되어, 코드가 간결해지고 유지보수가 용이해졌습니다.

---

## 2. 아키텍처 설명

### 2.1 모델 구조

모델은 다음 순서로 데이터를 처리합니다.

**입력층**  
shape: `(batch_size, num_steps, input_size)`  
num_steps 길이의 시계열 윈도우, 각 타임스텝마다 943개 피처를 입력받습니다.

**RNN 레이어 스택**  
2개(기본값)의 LSTM 레이어가 순차적으로 쌓입니다. 각 레이어는 `return_sequences=True`로 설정되어 모든 타임스텝의 출력을 다음 레이어로 전달합니다. 선택적으로 GRU 셀을 사용할 수 있습니다.

**출력층**  
`TimeDistributed(Dense(1))`: 각 타임스텝마다 독립적으로 선형 회귀 예측을 수행합니다. 활성화 함수 없이 실수값을 직접 출력합니다(회귀 문제).

**손실 함수**  
MSE(Mean Squared Error)를 사용합니다. 예측값과 실제 지수 변동의 차이를 최소화합니다.

**최적화**  
Adam 옵티마이저를 사용하며, ReduceLROnPlateau 콜백으로 학습률을 자동 조정합니다.

### 2.2 Many-to-Many 구조의 의미

원본 코드는 입력 시퀀스의 모든 타임스텝에서 예측을 생성하는 Many-to-Many 구조입니다. 예측 시에는 마지막 타임스텝의 출력만 최종 예측값으로 사용합니다. 이 구조는 모델이 시퀀스 전체에서 학습 신호를 받으므로 학습 효율이 높습니다.

### 2.3 정규화 및 과적합 방지

L2 정규화(λ=0.001)가 Dense 출력 레이어에 적용됩니다. 드롭아웃(기본 0.2)이 각 RNN 레이어에 적용됩니다. EarlyStopping 콜백이 검증 손실이 개선되지 않으면 학습을 조기 종료합니다.

---

## 3. 모듈 상세 설명

### 3.1 `src/config_loader.py`

YAML 설정 파일을 파이썬 데이터클래스로 변환합니다.

**주요 클래스**

`Config` - 최상위 설정 컨테이너. 아래 5개 하위 설정을 포함합니다.

`PathConfig` - 파일 경로 설정 (데이터, 모델, 결과, 로그 디렉토리).

`DataConfig` - 데이터 관련 설정. 파일명, 인코딩, 컬럼 정보, 날짜 범위.

`PredictionConfig` - 예측 관련 설정. 예측 기간, 스텝 간격, 모드.

`ModelConfig` - 모델 하이퍼파라미터. RNN 유형, 레이어 수, 히든 크기, 드롭아웃.

`TrainingConfig` - 학습 관련 설정. 배치 크기, 에포크, 학습률, 부트스트랩, 점진적 학습.

**주요 함수**

`load_config(config_path, **overrides)` - YAML 파일을 읽고 키워드 인수로 개별 항목을 덮어씁니다.

`config_from_args(args)` - argparse Namespace를 Config로 변환합니다.

### 3.2 `src/data_reader.py`

CSV 데이터를 읽어 LSTM 학습용 3D 시퀀스 배열로 변환합니다.

**`IndexDataset` 클래스**

`load()` 메서드는 CSV를 읽고 날짜 기반 인덱스로 train/test를 분리합니다. 예측 기간만큼 마지막 행을 제거하여 미래 타겟값 누수를 방지합니다.

`get_train_sequences(bootstrap, bootstrap_prob)` 메서드는 학습용 (X, y) 배열을 반환합니다. 부트스트랩 활성화 시 특정 확률로 샘플을 반복 추가합니다.

`get_test_sequences()` 메서드는 테스트용 (X, y) 배열을 반환합니다.

**시퀀스 생성 로직**

원시 데이터 배열에서 슬라이딩 윈도우 방식으로 시퀀스를 생성합니다. i번째 샘플의 인덱스는 `[i, i+step_interval, i+2*step_interval, ..., i+(num_steps-1)*step_interval]`입니다. X는 각 인덱스의 처음 `input_size`개 컬럼, y는 마지막(타겟) 컬럼입니다.

**`conversion(var_index, raw_data, row_size)` 함수**

변수 중요도 분석용. 지정 변수를 제외한 나머지를 첫 행 값으로 고정하여 해당 변수의 영향력을 측정합니다. 원본 v1의 `conversion()` 함수를 현대적 NumPy 코드로 재작성했습니다.

### 3.3 `src/model.py`

Keras `Model` 서브클래싱 방식으로 구현된 LSTM/GRU 모델입니다.

**`DeepMoneyModel` 클래스**

생성자 파라미터: `input_size`, `hidden_size`, `num_layers`, `output_size`, `rnn_type`, `dropout_rate`, `l2_reg`, `use_dense_layer`.

`call(inputs, training)` 메서드에서 입력을 RNN 레이어 스택과 출력 Dense 레이어를 통해 순방향 전파합니다.

**`build_model(config)` 함수**

Config 객체로부터 모델을 생성하고 컴파일합니다. Adam 옵티마이저, MSE 손실, RMSE 평가 지표를 설정합니다.

**모델 저장/로드**

`save_model(model, config)` - 가중치를 `models/{model_name}/weights` 경로에 저장합니다.

`load_model(config)` - 저장된 가중치를 불러옵니다. 파일이 없으면 None을 반환합니다.

`reset_model(config)` - 저장된 모델 디렉토리를 삭제합니다.

### 3.4 `src/trainer.py`

학습, 평가, 예측을 담당합니다.

**`Trainer` 클래스**

`train(X_train, y_train, X_val, y_val)` 메서드는 EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint 콜백을 사용하여 학습합니다.

`gradual_train(X_train, y_train, n_stages)` 메서드는 점진적 학습을 수행합니다. 처음에는 최근 데이터만, 단계별로 더 오래된 데이터를 추가합니다. 이는 원본 v1의 `predict_gradual_train.py` 방식을 일반화한 것입니다.

`evaluate(X, y, label)` 메서드는 모델의 MSE·RMSE를 출력합니다.

**`Predictor` 클래스**

`predict(X)` 메서드는 (samples, num_steps, 1) 형태의 원시 예측에서 마지막 타임스텝 값만 추출하여 반환합니다.

`predict_and_save(X, y, index_today, date_pred, today_list, std)` 메서드는 예측 수행 후 방향 정확도, Precision, Recall, F1을 계산하고 결과를 CSV로 저장합니다.

**`calculate_metrics(labels, predictions)` 함수**

방향성 지표(상승/하락 방향 예측 정확도)를 계산합니다. 금융 예측에서 정확한 수치보다 방향성이 더 실용적인 지표입니다.

---

## 4. 원본 v1 파일 → v2 모듈 매핑

| v1 파일 | v2 모듈 | 설명 |
|---------|---------|------|
| `path.py`, `path_*.py` | `config/config.yaml` + `src/config_loader.py` | 경로·설정 관리 |
| `reader.py`, `reader_*.py` | `src/data_reader.py` | 데이터 로드 및 시퀀스 생성 |
| `predict.py` | `src/model.py` + `src/trainer.py` | 기본 LSTM 학습·예측 |
| `predict_diff.py` | `config.yaml (mode: diff)` | 차분값 타겟 예측 |
| `predict_ma.py` | `config.yaml (data.file_name)` | 이동평균 기반 예측 |
| `predict_std.py` | `config.yaml (data.file_name)` | 표준편차 기반 예측 |
| `predict_4inputs.py` | `config.yaml (input_size: 4)` | 4개 입력 피처 예측 |
| `predict_bootstrap.py` | `config.yaml (training.bootstrap: true)` | 부트스트랩 오버샘플링 학습 |
| `predict_gradual_train.py` | `config.yaml (training.gradual_train: true)` | 점진적 학습 |
| `predict_diff_l2_reg.py` | `config.yaml (model.l2_reg)` | L2 정규화 (기본 포함) |
| `main.py` (없었음) | `main.py` | 통합 진입점 |

---

## 5. 하이퍼파라미터 가이드

### 5.1 `predict_term` (예측 기간)

단기(5일)는 변동성이 높아 정확도가 낮지만 샘플 수가 많습니다. 장기(65일)는 트렌드 예측에 유리하지만 학습 데이터가 적습니다. 일반적으로 20일이 균형 잡힌 선택입니다.

### 5.2 `num_steps` (시퀀스 길이)

LSTM이 한 번에 볼 수 있는 과거 데이터 길이입니다. `step_interval`과 곱해지면 실제 관찰 기간이 됩니다. 예를 들어 `num_steps=20, step_interval=1`이면 20거래일(약 1달)의 데이터를 입력으로 사용합니다.

### 5.3 `hidden_size` (히든 크기)

RNN 셀의 메모리 용량입니다. 더 크면 복잡한 패턴을 학습할 수 있지만 학습이 느려지고 과적합 위험이 높아집니다. 200~500 범위에서 시작하는 것이 좋습니다.

### 5.4 `bootstrap` (부트스트랩)

학습 데이터가 불균형하거나 부족할 때 특정 샘플을 반복 사용하여 데이터를 증강합니다. `bootstrap_prob`를 조정하여 반복 확률을 설정합니다.

### 5.5 `gradual_train` (점진적 학습)

오래된 시장 데이터가 현재 패턴에 맞지 않을 때 효과적입니다. 먼저 최근 데이터로 학습하고, 단계적으로 더 오래된 데이터를 추가합니다.

---

## 6. 출력 결과 해석

### 6.1 방향 정확도

방향 정확도(Accuracy)는 상승/하락 방향 예측의 정확도입니다. 50%는 랜덤과 같으며, 55~60% 이상이면 실용적 가치가 있습니다.

### 6.2 누적 손익

`profit` 컬럼의 합계입니다. 예측 방향이 맞으면 해당 기간 지수 변동폭을 이익으로, 틀리면 손실로 계산합니다. 거래 비용은 포함하지 않습니다.

### 6.3 RMSE

Root Mean Squared Error. 예측값과 실제 지수 차이의 평균 오차입니다. 데이터 스케일에 따라 해석이 달라집니다.

---

## 7. 문제 해결

### 메모리 부족

배치 크기를 줄이거나(`batch_size: 10`), 입력 피처를 선별하거나(`input_size` 감소), `num_steps`를 줄여보세요.

### 학습이 발산하는 경우

학습률을 낮추거나(`learning_rate: 0.0001`), 드롭아웃을 높이거나(`dropout_rate: 0.3`), 배치 크기를 늘려보세요.

### 과적합 (train loss << test loss)

드롭아웃을 높이거나, L2 정규화를 강화하거나(`l2_reg: 0.01`), `early_stopping_patience`를 줄여보세요.

---

## 8. 개발 환경

이 시스템은 다음 환경에서 개발 및 테스트되었습니다.

Python 3.10, TensorFlow 2.13, NumPy 1.24, pandas 2.0, PyYAML 6.0.
