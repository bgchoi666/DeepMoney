"""
main.py
───────
DeepMoney v2 메인 실행 파일.

사용 예:
    # 기본 설정으로 학습 + 예측
    python main.py

    # 예측 기간 5일, 히든 유닛 300으로 학습
    python main.py --predict_term 5 --hidden_size 300

    # 학습 없이 기존 모델로 예측만
    python main.py --no_train

    # GRU 모델, 설정 파일 지정
    python main.py --rnn_type gru --config config/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── 모듈 경로 추가 ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config_loader import build_arg_parser, config_from_args
from src.data_reader import IndexDataset
from src.model import build_model, save_model, load_model, reset_model
from src.trainer import Trainer, Predictor


def main():
    # ── 1. 설정 로드 ─────────────────────────────────────────────────────────
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    print("=" * 60)
    print("  DeepMoney v2 - LSTM 기반 주가 지수 중장기 예측 시스템")
    print("=" * 60)
    print(f"  모델명     : {config.model_name}")
    print(f"  예측 기간  : {config.prediction.predict_term}일")
    print(f"  RNN 타입   : {config.model.rnn_type.upper()}")
    print(f"  히든 크기  : {config.model.hidden_size}")
    print(f"  레이어 수  : {config.model.num_layers}")
    print(f"  시퀀스 길이: {config.model.num_steps}")
    print("=" * 60)

    # ── 2. 데이터 로드 ───────────────────────────────────────────────────────
    dataset = IndexDataset(config).load()

    X_train, y_train = dataset.get_train_sequences(
        bootstrap=config.training.bootstrap,
        bootstrap_prob=config.training.bootstrap_prob,
    )
    X_test, y_test = dataset.get_test_sequences()

    print(f"\n데이터 형태 | X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"            | X_test : {X_test.shape},  y_test : {y_test.shape}")

    # ── 3. 모델 준비 ─────────────────────────────────────────────────────────
    if config.training.model_reset:
        reset_model(config)

    model = None
    if not config.training.model_reset:
        model = load_model(config)

    if model is None:
        model = build_model(config)
        model.summary()

    # ── 4. 학습 ──────────────────────────────────────────────────────────────
    if not getattr(args, "no_train", False):
        trainer = Trainer(model, config)

        if config.training.gradual_train:
            trainer.gradual_train(X_train, y_train)
        else:
            trainer.train(X_train, y_train)

        # 학습 후 평가
        trainer.evaluate(X_train, y_train, label="학습셋")
        trainer.evaluate(X_test, y_test, label="테스트셋")

        # 모델 저장
        save_model(model, config)
    else:
        if model is None:
            print("❌ 저장된 모델이 없습니다. --no_train 없이 실행하세요.")
            return

    # ── 5. 예측 및 결과 저장 ─────────────────────────────────────────────────
    index_today, date_pred, today_list, std = dataset.get_index_dates()

    predictor = Predictor(model, config)
    predictor.predict_and_save(
        X=X_test,
        y=y_test,
        index_today=index_today,
        date_pred=date_pred,
        today_list=today_list,
        std=std,
    )

    print("\n✅ 완료!")


if __name__ == "__main__":
    main()
