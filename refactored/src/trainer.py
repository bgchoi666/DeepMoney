"""
trainer.py
──────────
모델 학습(train) 및 평가(evaluate)를 담당합니다.

주요 기능:
  - EarlyStopping, ReduceLROnPlateau, TensorBoard 콜백
  - 점진적 학습 (gradual train): 점점 데이터를 늘려가며 학습
  - 학습 이력 CSV 저장
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# ──────────────────────────────────────────────────────────────────────────────
# 기본 학습기
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    DeepMoney 모델 학습기.

    Args:
        model:  컴파일된 keras.Model
        config: Config 데이터클래스
    """

    def __init__(self, model: keras.Model, config):
        self.model = model
        self.config = config
        self._history: Optional[keras.callbacks.History] = None

    # ── 콜백 생성 ────────────────────────────────────────────────────────────

    def _build_callbacks(self) -> list:
        t_cfg = self.config.training
        log_dir = Path(self.config.paths.log_dir) / self.config.model_name
        log_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=t_cfg.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(t_cfg.early_stopping_patience // 2, 3),
                min_lr=1e-6,
                verbose=1,
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=0,
                update_freq="epoch",
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(log_dir / "best_weights"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            ),
        ]
        return callbacks

    # ── 학습 ─────────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> keras.callbacks.History:
        """
        모델을 학습합니다.

        Args:
            X_train: (samples, num_steps, input_size)
            y_train: (samples, num_steps, 1)
            X_val:   검증 데이터 (없으면 자동 분할)
            y_val:   검증 레이블

        Returns:
            Keras History 객체
        """
        t_cfg = self.config.training

        if X_val is not None:
            validation_data = (X_val, y_val)
            val_split = 0.0
        else:
            validation_data = None
            val_split = t_cfg.validation_split

        print(
            f"\n🚀 학습 시작 | "
            f"샘플: {len(X_train)}, "
            f"배치: {t_cfg.batch_size}, "
            f"에포크: {t_cfg.epochs}"
        )

        self._history = self.model.fit(
            X_train,
            y_train,
            batch_size=t_cfg.batch_size,
            epochs=t_cfg.epochs,
            validation_split=val_split,
            validation_data=validation_data,
            shuffle=t_cfg.shuffle,
            callbacks=self._build_callbacks(),
            verbose=1,
        )

        self._save_history()
        return self._history

    # ── 점진적 학습 (Gradual Train) ───────────────────────────────────────────

    def gradual_train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_stages: int = 5,
    ) -> keras.callbacks.History:
        """
        데이터를 점진적으로 늘려가며 학습합니다.
        초반에는 최근 데이터로만 학습하고, 단계별로 오래된 데이터를 추가합니다.

        Args:
            X_train:  전체 학습 데이터 X
            y_train:  전체 학습 데이터 y
            n_stages: 단계 수

        Returns:
            마지막 단계의 History
        """
        total = len(X_train)
        stage_size = total // n_stages
        history = None

        for stage in range(1, n_stages + 1):
            start = max(total - stage * stage_size, 0)
            X_stage = X_train[start:]
            y_stage = y_train[start:]
            print(f"\n📈 점진적 학습 단계 {stage}/{n_stages} | 샘플: {len(X_stage)}")
            history = self.train(X_stage, y_stage)

        return history

    # ── 평가 ─────────────────────────────────────────────────────────────────

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, label: str = "평가"
    ) -> Dict[str, float]:
        """
        모델 성능을 평가합니다.

        Returns:
            {"loss": ..., "rmse": ...}
        """
        results = self.model.evaluate(X, y, batch_size=1, verbose=0)
        metrics = dict(zip(self.model.metrics_names, results))
        print(f"📊 [{label}] Loss: {metrics['loss']:.6f} | RMSE: {metrics['rmse']:.6f}")
        return metrics

    # ── 이력 저장 ─────────────────────────────────────────────────────────────

    def _save_history(self):
        if self._history is None:
            return
        result_dir = Path(self.config.paths.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        hist_path = result_dir / f"train_history_{self.config.model_name}_{ts}.csv"
        pd.DataFrame(self._history.history).to_csv(hist_path, index=False)
        print(f"📄 학습 이력 저장: {hist_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 예측기
# ──────────────────────────────────────────────────────────────────────────────

class Predictor:
    """
    학습된 모델로 예측을 수행하고 결과를 저장합니다.

    Args:
        model:  학습된 keras.Model
        config: Config 데이터클래스
    """

    def __init__(self, model: keras.Model, config):
        self.model = model
        self.config = config

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        LSTM many-to-many 예측을 수행하고
        마지막 스텝의 예측값만 추출합니다.

        Args:
            X: (samples, num_steps, input_size)

        Returns:
            pred_last: (samples,) - 각 샘플의 마지막 스텝 예측값
        """
        raw_pred = self.model.predict(X, batch_size=1, verbose=0)
        # raw_pred: (samples, num_steps, 1)
        pred_last = raw_pred[:, -1, 0]   # 마지막 스텝만 추출
        return pred_last

    def predict_and_save(
        self,
        X: np.ndarray,
        y: np.ndarray,
        index_today: np.ndarray,
        date_pred: np.ndarray,
        today_list: list,
        std: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        예측 후 방향성 정확도·손익 계산 및 CSV 저장.

        Args:
            X:           테스트 입력 시퀀스
            y:           테스트 타겟 (samples, num_steps, 1)
            index_today: 예측 시점의 실제 지수 값
            date_pred:   예측 대상 날짜 배열
            today_list:  예측 기준 날짜 리스트
            std:         표준편차 배열 (선택)

        Returns:
            결과 DataFrame
        """
        pred_last = self.predict(X)
        target_last = y[:, -1, 0]  # (samples,)

        # 방향성 정확도 계산
        accuracy, precision, recall, f1 = calculate_metrics(target_last, pred_last)

        # 실제 지수 vs 예측 지수
        n = min(len(index_today), len(pred_last), len(target_last))
        index_real = index_today[:n] + target_last[:n]
        index_pred_val = index_today[:n] + pred_last[:n]

        # 손익 계산: 방향이 맞으면 +, 틀리면 -
        profits = []
        for i in range(n):
            diff_real = index_today[i] - index_real[i]
            diff_pred = index_today[i] - index_pred_val[i]
            sign = 1 if diff_real * diff_pred > 0 else -1
            profits.append(sign * abs(diff_real))

        result_dict = {
            "date_base": today_list[:n],
            "date_pred": date_pred[:n],
            "real_diff": target_last[:n],
            "pred_diff": pred_last[:n],
            "index_today": index_today[:n],
            "index_real": index_real,
            "index_pred": index_pred_val,
            "profit": profits,
        }
        if std is not None:
            result_dict["std"] = std[:n]

        df_result = pd.DataFrame(result_dict)

        # 결과 저장
        result_dir = Path(self.config.paths.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = result_dir / f"result_{self.config.model_name}_{ts}.csv"

        df_result.to_csv(result_path, index=False)

        summary_path = str(result_path).replace(".csv", "_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"예측 기간: {self.config.prediction.predict_term}일\n")
            f.write(f"모델: {self.config.model_name}\n")
            f.write(f"방향 정확도: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"총 누적 손익: {sum(profits):.4f}\n")

        print(f"\n📊 예측 결과")
        print(f"   방향 정확도 : {accuracy:.4f}")
        print(f"   Precision   : {precision:.4f}")
        print(f"   Recall      : {recall:.4f}")
        print(f"   F1 Score    : {f1:.4f}")
        print(f"   누적 손익   : {sum(profits):.4f}")
        print(f"📄 결과 저장   : {result_path}")

        return df_result


# ──────────────────────────────────────────────────────────────────────────────
# 성능 지표 계산
# ──────────────────────────────────────────────────────────────────────────────

def calculate_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    방향성 기반 정확도, Precision, Recall, F1을 계산합니다.

    Args:
        labels:      실제 값 배열
        predictions: 예측 값 배열

    Returns:
        (accuracy, precision, recall, f1_score)
    """
    tp = fp = tn = fn = 0

    for label, pred in zip(labels, predictions):
        if pred > 0:
            if label > 0:
                tp += 1
            else:
                fp += 1
        else:
            if label < 0:
                tn += 1
            else:
                fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 / (1 / precision + 1 / recall)
        if (precision > 0 and recall > 0)
        else 0.0
    )

    return accuracy, precision, recall, f1
