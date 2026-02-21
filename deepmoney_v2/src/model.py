"""
model.py
─────────
Keras (TensorFlow 2.x) 기반 Many-to-Many LSTM/GRU 회귀 모델.

원본의 tf.contrib.rnn + tf.Estimator 구조를 현대적인 
tf.keras.Model 서브클래싱 방식으로 완전히 재작성합니다.

모델 구조:
    입력  →  [LSTM 또는 GRU] × num_layers  →  TimeDistributed(Dense(1))
    손실  :  MSE  +  L2 정규화
    최적화:  Adam
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ──────────────────────────────────────────────────────────────────────────────
# DeepMoney LSTM 모델
# ──────────────────────────────────────────────────────────────────────────────

class DeepMoneyModel(keras.Model):
    """
    Many-to-Many LSTM/GRU 회귀 모델.

    Args:
        input_size:    입력 피처 수
        hidden_size:   RNN 유닛 수
        num_layers:    RNN 레이어 수
        output_size:   출력 크기 (보통 1)
        rnn_type:      "lstm" 또는 "gru"
        dropout_rate:  리커런트 드롭아웃 비율
        l2_reg:        Dense 레이어 L2 정규화 계수
        use_dense_layer: LSTM 이후 추가 Dense hidden layer 사용 여부
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 200,
        num_layers: int = 2,
        output_size: int = 1,
        rnn_type: str = "lstm",
        dropout_rate: float = 0.2,
        l2_reg: float = 0.001,
        use_dense_layer: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.rnn_type = rnn_type.lower()
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_dense_layer = use_dense_layer

        # ── RNN 레이어 스택 구성 ────────────────────────────────────────────
        self.rnn_layers: list = []
        for i in range(num_layers):
            return_seq = True  # Many-to-Many → 모든 스텝에서 출력
            rnn_layer = self._build_rnn_cell(
                hidden_size, return_seq, dropout_rate, name=f"rnn_{i}"
            )
            self.rnn_layers.append(rnn_layer)

        # ── 선택적 중간 Dense 레이어 ────────────────────────────────────────
        self.hidden_dense: Optional[layers.Layer] = None
        if use_dense_layer:
            self.hidden_dense = layers.TimeDistributed(
                layers.Dense(
                    hidden_size // 2,
                    activation="relu",
                    kernel_regularizer=regularizers.l2(l2_reg),
                ),
                name="hidden_dense",
            )

        # ── 출력 레이어 ─────────────────────────────────────────────────────
        self.output_dense = layers.TimeDistributed(
            layers.Dense(
                output_size,
                activation=None,                            # 회귀 → 선형 활성
                kernel_initializer="glorot_uniform",        # Xavier init
                kernel_regularizer=regularizers.l2(l2_reg),
            ),
            name="output_dense",
        )

    def _build_rnn_cell(
        self,
        units: int,
        return_sequences: bool,
        dropout: float,
        name: str,
    ) -> layers.Layer:
        """rnn_type에 따라 LSTM 또는 GRU 레이어를 생성합니다."""
        common = dict(
            units=units,
            return_sequences=return_sequences,
            dropout=dropout,
            recurrent_dropout=0.0,   # recurrent_dropout은 GPU 속도에 영향
            name=name,
        )
        if self.rnn_type == "lstm":
            return layers.LSTM(**common)
        elif self.rnn_type == "gru":
            return layers.GRU(**common)
        else:
            raise ValueError(f"rnn_type='{self.rnn_type}' 은 지원하지 않습니다. 'lstm' 또는 'gru'.")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        순방향 전파.

        Args:
            inputs:   (batch, num_steps, input_size)
            training: 드롭아웃 적용 여부

        Returns:
            logits: (batch, num_steps, output_size)
        """
        x = inputs
        for rnn in self.rnn_layers:
            x = rnn(x, training=training)

        if self.hidden_dense is not None:
            x = self.hidden_dense(x, training=training)

        return self.output_dense(x)

    def get_config(self) -> dict:
        return dict(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            rnn_type=self.rnn_type,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            use_dense_layer=self.use_dense_layer,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 모델 빌더
# ──────────────────────────────────────────────────────────────────────────────

def build_model(config) -> DeepMoneyModel:
    """
    Config 객체로부터 컴파일된 DeepMoneyModel을 생성합니다.

    Args:
        config: Config 데이터클래스

    Returns:
        compile 된 keras.Model
    """
    m_cfg = config.model
    d_cfg = config.data
    t_cfg = config.training

    model = DeepMoneyModel(
        input_size=d_cfg.input_size,
        hidden_size=m_cfg.hidden_size,
        num_layers=m_cfg.num_layers,
        output_size=d_cfg.output_size,
        rnn_type=m_cfg.rnn_type,
        dropout_rate=m_cfg.dropout_rate,
        l2_reg=m_cfg.l2_reg,
        use_dense_layer=m_cfg.use_dense_layer,
        name="DeepMoneyModel",
    )

    optimizer = keras.optimizers.Adam(learning_rate=t_cfg.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    # 모델 구조 출력을 위해 더미 데이터로 build
    dummy = tf.zeros([1, m_cfg.num_steps, d_cfg.input_size])
    model(dummy, training=False)

    return model


# ──────────────────────────────────────────────────────────────────────────────
# 모델 저장 / 로드
# ──────────────────────────────────────────────────────────────────────────────

def get_model_path(config) -> Path:
    return Path(config.paths.model_dir) / config.model_name


def save_model(model: DeepMoneyModel, config) -> Path:
    """모델 가중치를 저장합니다."""
    model_path = get_model_path(config)
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(model_path / "weights"))
    print(f"✔ 모델 저장 완료: {model_path}")
    return model_path


def load_model(config) -> Optional[DeepMoneyModel]:
    """
    저장된 모델 가중치를 로드합니다.

    Returns:
        로드된 DeepMoneyModel, 저장된 모델이 없으면 None
    """
    model_path = get_model_path(config)
    weights_path = model_path / "weights.index"

    if not weights_path.exists():
        return None

    model = build_model(config)
    model.load_weights(str(model_path / "weights"))
    print(f"✔ 기존 모델 로드: {model_path}")
    return model


def reset_model(config) -> None:
    """저장된 모델 디렉토리를 삭제합니다."""
    import shutil
    model_path = get_model_path(config)
    if model_path.exists():
        shutil.rmtree(model_path)
        print(f"🗑  기존 모델 삭제: {model_path}")
