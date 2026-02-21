"""
data_reader.py
──────────────
CSV 데이터 파일을 읽어 LSTM 학습용 시퀀스 데이터로 변환합니다.

지원 기능:
  - 정규화(norm) / 차분(diff) 두 가지 타겟 모드
  - 부트스트랩(bootstrap) 오버샘플링
  - 다중 입력(4-inputs) 처리
  - 이동평균(MA) / 표준편차(std) 피처 처리
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# 타겟 컬럼 매핑 (predict_term → CSV 컬럼 인덱스)
# ──────────────────────────────────────────────────────────────────────────────
TARGET_COL_MAP = {1: 944, 5: 945, 20: 946, 65: 947}  # 0-based: offset 943


class IndexDataset:
    """
    주가 지수 CSV를 읽어 train/test 데이터셋을 제공하는 클래스.

    Args:
        config: Config 데이터클래스 인스턴스
    """

    def __init__(self, config):
        self.cfg = config
        self._raw_df: Optional[pd.DataFrame] = None
        self._train_raw: Optional[np.ndarray] = None
        self._test_raw: Optional[np.ndarray] = None
        self.test_start_index: int = 0

    # ── 공개 API ────────────────────────────────────────────────────────────

    def load(self) -> "IndexDataset":
        """CSV 파일을 읽고 train/test 원시 데이터를 준비합니다."""
        data_cfg = self.cfg.data
        pred_cfg = self.cfg.prediction

        file_path = self.cfg.data_file_path
        if not Path(file_path).exists():
            raise FileNotFoundError(
                f"데이터 파일을 찾을 수 없습니다: {file_path}\n"
                f"config/config.yaml 의 data.file_name 과 paths.data_dir 을 확인하세요."
            )

        df = pd.read_csv(file_path, encoding=data_cfg.encoding)

        # 예측 기간 만큼 마지막 행 제거 (타겟이 미래 값이므로)
        df = df.iloc[: -pred_cfg.predict_term]

        # 입력 피처 + 타겟 컬럼 선택
        input_cols = list(range(1, 1 + data_cfg.input_size))
        target_col = TARGET_COL_MAP.get(pred_cfg.predict_term)
        if target_col is None:
            raise ValueError(
                f"predict_term={pred_cfg.predict_term} 은 지원하지 않습니다. "
                f"1, 5, 20, 65 중 선택하세요."
            )
        cols = input_cols + [target_col]

        # 날짜 기반 인덱스 계산
        date_col = data_cfg.date_column
        train_start_idx = max(
            len(df[df[date_col] <= data_cfg.train_start]) - 1, 0
        )
        self.test_start_index = len(df[df[date_col] <= data_cfg.test_start]) - 1
        test_end_idx = len(df[df[date_col] < data_cfg.test_end]) - 1

        m = self.cfg.model
        look_back = m.step_interval * (m.num_steps - 1)

        # 학습 데이터: train_start ~ test_start
        actual_train_start = max(train_start_idx - look_back, 0)
        self._train_raw = df.values[actual_train_start : self.test_start_index, cols].astype(np.float32)

        # 테스트 데이터: (test_start - look_back) ~ test_end
        self._test_raw = df.values[
            self.test_start_index - look_back : test_end_idx + 1, cols
        ].astype(np.float32)

        self._raw_df = df
        print(
            f"✔ 데이터 로드 완료 | "
            f"학습 행: {len(self._train_raw)}, "
            f"테스트 행: {len(self._test_raw)}, "
            f"테스트 시작 인덱스: {self.test_start_index}"
        )
        return self

    def get_train_sequences(self, bootstrap: bool = False, bootstrap_prob: float = 0.2
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """학습용 (X, y) 시퀀스를 반환합니다."""
        return self._make_sequences(self._train_raw, mode="train",
                                    bootstrap=bootstrap, bootstrap_prob=bootstrap_prob)

    def get_test_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """테스트용 (X, y) 시퀀스를 반환합니다."""
        return self._make_sequences(self._test_raw, mode="test")

    def get_index_dates(self, index_file: str = "index-ma-std.csv"
                        ) -> Tuple[np.ndarray, np.ndarray, list, Optional[np.ndarray]]:
        """
        예측 결과 비교를 위한 날짜·지수 기준값을 반환합니다.

        Returns:
            (index_today, date_pred, today_list, std)
        """
        pred_cfg = self.cfg.prediction
        model_cfg = self.cfg.model

        data_size = len(self._test_raw) - model_cfg.step_interval * (model_cfg.num_steps - 1)

        index_path = os.path.join(self.cfg.paths.data_dir, index_file)
        if not Path(index_path).exists():
            # 파일 없으면 빈 배열 반환
            dummy = np.zeros(data_size)
            return dummy, dummy, list(range(data_size)), None

        idx_df = pd.read_csv(index_path, encoding="ISO-8859-1")
        si = self.test_start_index
        pt = pred_cfg.predict_term

        index_today = idx_df.values[si : si + data_size, 1].astype(np.float32)
        date_pred = idx_df.values[si + pt : si + pt + data_size, 0]
        today_list = list(idx_df.values[si : si + data_size, 0])

        std = None
        if idx_df.shape[1] > 7:
            std = idx_df.values[si + pt : si + pt + data_size, 7].astype(np.float32)

        return index_today, date_pred, today_list, std

    # ── 내부 구현 ────────────────────────────────────────────────────────────

    def _make_sequences(
        self,
        raw: np.ndarray,
        mode: str = "test",
        bootstrap: bool = False,
        bootstrap_prob: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        원시 배열을 LSTM 입력 시퀀스 (X, y) 로 변환합니다.

        raw shape: (N, input_size + 1)  # 마지막 열이 타겟
        반환 X: (samples, num_steps, input_size)
        반환 y: (samples, num_steps, 1)
        """
        m = self.cfg.model
        num_steps = m.num_steps
        step_interval = m.step_interval
        input_size = self.cfg.data.input_size

        n = len(raw) - (num_steps - 1) * step_interval

        X_list, y_list = [], []

        i = 0
        while i < n:
            indices = list(range(i, i + num_steps * step_interval, step_interval))
            x_seq = raw[indices, :input_size]           # (num_steps, input_size)
            y_seq = raw[indices, input_size]            # (num_steps,)

            if mode == "train" and bootstrap:
                import random
                if random.random() < bootstrap_prob:
                    # 동일 샘플을 최대 5번 반복
                    repeat = min(5, n - i)
                    for _ in range(repeat):
                        X_list.append(x_seq)
                        y_list.append(y_seq)
            else:
                X_list.append(x_seq)
                y_list.append(y_seq)
            i += 1

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)[..., np.newaxis]  # (..., 1)
        return X, y


# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────────────────────────────────────────

def conversion(var_index: int, raw_data: np.ndarray, row_size: int = 260
               ) -> Tuple[np.ndarray, float]:
    """
    변수 중요도 분석을 위해 지정 변수를 제외한 나머지를 첫 행 값으로 고정합니다.

    Args:
        var_index: 분석할 변수 인덱스 (1-based)
        raw_data:  원시 데이터 배열
        row_size:  분석할 행 수

    Returns:
        (변환된 데이터, 해당 변수의 절댓값 변화량 합계)
    """
    var_size = raw_data.shape[1] - 1  # 타겟 제외
    r_data = raw_data.copy()
    sum_var = 0.0

    for j in range(min(row_size, len(r_data))):
        for i in range(1, var_size + 1):
            if i != var_index:
                r_data[j, i] = raw_data[0, i]
            elif j > 0:
                sum_var += abs(r_data[j, i] - r_data[j - 1, i])

    return r_data, sum_var
