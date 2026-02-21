"""
config_loader.py
─────────────────
YAML 설정 파일을 읽어 Config 데이터클래스로 변환합니다.
명령줄 인수로 개별 파라미터를 오버라이드할 수 있습니다.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Config 데이터클래스
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PathConfig:
    data_dir: str = "./data"
    model_dir: str = "./models"
    result_dir: str = "./results"
    log_dir: str = "./logs"


@dataclass
class DataConfig:
    file_name: str = "kospi200f-943.csv"
    encoding: str = "ISO-8859-1"
    date_column: str = "date"
    input_size: int = 943
    output_size: int = 1
    train_start: str = "2000-01-01"
    test_start: str = "2017-01-02"
    test_end: str = "2025-12-31"


@dataclass
class PredictionConfig:
    predict_term: int = 20
    step_interval: int = 1
    mode: str = "diff"


@dataclass
class ModelConfig:
    rnn_type: str = "lstm"
    num_layers: int = 2
    hidden_size: int = 200
    num_steps: int = 20
    dropout_rate: float = 0.2
    l2_reg: float = 0.001
    use_dense_layer: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 20
    epochs: int = 50
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    validation_split: float = 0.1
    shuffle: bool = False
    bootstrap: bool = False
    bootstrap_prob: float = 0.2
    gradual_train: bool = False
    model_reset: bool = True


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    version: str = "v2.0"

    def __post_init__(self):
        """필요한 디렉토리를 자동으로 생성합니다."""
        for dir_path in [
            self.paths.data_dir,
            self.paths.model_dir,
            self.paths.result_dir,
            self.paths.log_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    @property
    def data_file_path(self) -> str:
        return os.path.join(self.paths.data_dir, self.data.file_name)

    @property
    def model_name(self) -> str:
        """모델 저장 시 사용하는 고유 이름을 생성합니다."""
        cfg = self
        return (
            f"{cfg.version}"
            f"_{cfg.prediction.predict_term}d"
            f"_{cfg.model.rnn_type}"
            f"_h{cfg.model.hidden_size}"
            f"_l{cfg.model.num_layers}"
            f"_s{cfg.model.num_steps}"
            f"_b{cfg.training.batch_size}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# YAML 로더
# ──────────────────────────────────────────────────────────────────────────────

def _dict_to_dataclass(cls, data: dict):
    """재귀적으로 딕셔너리를 데이터클래스로 변환합니다."""
    if not isinstance(data, dict):
        return data
    fields_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key in fields_types:
            field_type_str = fields_types[key]
            # 타입 힌트 문자열로부터 실제 클래스 찾기
            field_type = globals().get(field_type_str, None)
            if field_type and hasattr(field_type, "__dataclass_fields__"):
                kwargs[key] = _dict_to_dataclass(field_type, value)
            else:
                kwargs[key] = value
    return cls(**kwargs)


def load_config(config_path: str = "config/config.yaml", **overrides) -> Config:
    """
    YAML 설정 파일을 읽어 Config 객체를 반환합니다.

    Args:
        config_path: YAML 파일 경로
        **overrides: 키워드 인수로 특정 설정 오버라이드
                     예) load_config(predict_term=5, hidden_size=300)

    Returns:
        Config 데이터클래스 인스턴스
    """
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"⚠  설정 파일 없음: {config_path} → 기본값 사용")
        cfg_dict = {}
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f) or {}

    # YAML → 데이터클래스 변환
    config = Config(
        paths=_dict_to_dataclass(PathConfig, cfg_dict.get("paths", {})),
        data=_dict_to_dataclass(DataConfig, cfg_dict.get("data", {})),
        prediction=_dict_to_dataclass(PredictionConfig, cfg_dict.get("prediction", {})),
        model=_dict_to_dataclass(ModelConfig, cfg_dict.get("model", {})),
        training=_dict_to_dataclass(TrainingConfig, cfg_dict.get("training", {})),
        version=cfg_dict.get("version", "v2.0"),
    )

    # 명령줄/키워드 오버라이드 적용
    _apply_overrides(config, overrides)

    return config


def _apply_overrides(config: Config, overrides: dict):
    """평탄화된 key=value 오버라이드를 Config 하위 필드에 적용합니다."""
    sub_configs = {
        "paths": config.paths,
        "data": config.data,
        "prediction": config.prediction,
        "model": config.model,
        "training": config.training,
    }
    for key, value in overrides.items():
        applied = False
        for sub in sub_configs.values():
            if hasattr(sub, key):
                setattr(sub, key, value)
                applied = True
                break
        if not applied and hasattr(config, key):
            setattr(config, key, value)


# ──────────────────────────────────────────────────────────────────────────────
# argparse 통합
# ──────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepMoney v2 - LSTM 기반 주가 지수 중장기 예측 시스템"
    )
    parser.add_argument("--config", default="config/config.yaml", help="YAML 설정 파일 경로")
    parser.add_argument("--data_path", default=None, help="데이터 파일 경로 (config 오버라이드)")
    parser.add_argument("--predict_term", type=int, default=None, help="예측 기간(일): 5, 20, 65")
    parser.add_argument("--hidden_size", type=int, default=None, help="LSTM 히든 유닛 수")
    parser.add_argument("--num_layers", type=int, default=None, help="LSTM 레이어 수")
    parser.add_argument("--num_steps", type=int, default=None, help="시퀀스 길이(look-back)")
    parser.add_argument("--batch_size", type=int, default=None, help="배치 크기")
    parser.add_argument("--epochs", type=int, default=None, help="학습 에포크 수")
    parser.add_argument("--learning_rate", type=float, default=None, help="학습률")
    parser.add_argument("--rnn_type", choices=["lstm", "gru"], default=None, help="RNN 셀 유형")
    parser.add_argument("--mode", choices=["norm", "diff"], default=None, help="예측 모드")
    parser.add_argument("--model_reset", action="store_true", help="기존 모델 삭제 후 재학습")
    parser.add_argument("--no_train", action="store_true", help="학습 없이 예측만 수행")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """argparse Namespace로부터 Config를 생성합니다."""
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    # data_path 처리
    if "data_path" in overrides:
        dp = overrides.pop("data_path")
        overrides["data_dir"] = str(Path(dp).parent)
        overrides["file_name"] = Path(dp).name
    return load_config(args.config, **overrides)
