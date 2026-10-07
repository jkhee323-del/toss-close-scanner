from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from . import swing_horizon_models as base


HORIZON = 5
TARGET_RETURN = 0.015  # 5거래일 뒤 +1.5%
DEV_END = pd.Timestamp("2024-12-30")

# 2025~2026 결과를 보고 조정하지 않는다.
# 개발구간에서만 후보를 선택한다.
THRESHOLDS = [0.46, 0.48, 0.50, 0.52, 0.54]

CONFIGS = [
    {
        "name": "d3_regular",
        "learning_rate": 0.035,
        "max_depth": 3,
        "max_iter": 350,
        "l2_regularization": 1.5,
        "min_samples_leaf": 40,
    },
    {
        "name": "d3_reg60",
        "learning_rate": 0.03,
        "max_depth": 3,
        "max_iter": 400,
        "l2_regularization": 3.0,
        "min_samples_leaf": 60,
    },
    {
        "name": "d3_reg80",
        "learning_rate": 0.03,
        "max_depth": 3,
        "max_iter": 400,
        "l2_regularization": 4.0,
        "min_samples_leaf": 80,
    },
]

FOLDS = [
    ("2022-12-29", "2023-01-02", "2023-06-30"),
    ("2023-06-30", "2023-07-03", "2023-12-28"),
    ("2023-12-28", "2024-01-02", "2024-06-28"),
    ("2024-06-28", "2024-07-01", "2024-12-30"),
]


def prepare_data():
    old_dev_end = base.DEV_END

    try:
        base.DEV_END = DEV_END
        data = base.build_horizon_dataset(HORIZON).copy()
    finally:
        base.DEV_END = old_dev_end

    return_col = "return_5d"
    data["target_5d_final"] = (
        data[return_col] >= TARGET_RETURN
    ).astype(int)

    return data


def evaluate(frame, probability, threshold):
    mask = probability >= threshold
    signals = frame.loc[mask].copy()

    if signals.empty:
        return {
            "signals": 0,
            "days": 0,
            "positive_hit": np.nan,
            "target_hit": np.nan,
            "avg_return": np.nan,
        }

    return {
        "signals": int(len(signals)),
        "days": int(signals["date"].nunique()),
        "positive_hit": float(
            (signals["return_5d"] > 0).mean() * 100
        ),
        "target_hit": float(
            signals["target_5d_final"].mean() * 100
        ),
        "avg_return": float(
            signals["return_5d"].mean() * 100
        ),
    }


def run():
    data = prepare_data()

    results = []

    for config in CONFIGS:
        cached = []

        for train_end, val_start, val_end in FOLDS:
            train_end = pd.Timestamp(train_end)
            val_start = pd.Timestamp(val_start)
            val_end = pd.Timestamp(val_end)

            train = data.loc[
                data["date"] <= train_end
            ].copy()

            validation = data.loc[
                data["date"].between(val_start, val_end)
            ].copy()

            # 5일 미래 라벨이 validation으로 넘어가지 않도록
            # 학습 마지막 5거래일 제거
            train_dates = np.sort(train["date"].unique())

            if len(train_dates) <= HORIZON:
                continue

            safe_end = train_dates[-(HORIZON + 1)]

            train = train.loc[
                train["date"] <= safe_end
            ].copy()

            if train.empty or validation.empty:
                continue

            model = make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingClassifier(
                    learning_rate=config["learning_rate"],
                    max_depth=config["max_depth"],
                    max_iter=config["max_iter"],
                    l2_regularization=config[
                        "l2_regularization"
                    ],
                    min_samples_leaf=config[
                        "min_samples_leaf"
                    ],
                    random_state=42,
                ),
            )

            model.fit(
                train[base.FEATURES],
                train["target_5d_final"],
            )

            probability = model.predict_proba(
                validation[base.FEATURES]
            )[:, 1]

            cached.append((validation, probability))

        for threshold in THRESHOLDS:
            metrics = [
                evaluate(frame, probability, threshold)
                for frame, probability in cached
            ]

            if not metrics:
                continue

            total_signals = sum(
                m["signals"] for m in metrics
            )

            if total_signals == 0:
                continue

            weighted_positive_hit = sum(
                m["positive_hit"] * m["signals"]
                for m in metrics
            ) / total_signals

            weighted_target_hit = sum(
                m["target_hit"] * m["signals"]
                for m in metrics
            ) / total_signals

            weighted_return = sum(
                m["avg_return"] * m["signals"]
                for m in metrics
            ) / total_signals

            min_signals = min(
                m["signals"] for m in metrics
            )

            min_days = min(
                m["days"] for m in metrics
            )

            min_positive_hit = min(
                m["positive_hit"] for m in metrics
            )

            # 이번 목표:
            # 적중률만 높고 신호가 거의 없는 모델 제외
            eligible = (
                total_signals >= 3000
                and min_signals >= 500
                and min_days >= 75
                and min_positive_hit >= 48.0
            )

            # 전체 적중률 + 최악 구간 적중률을 우선
            robust_score = (
                weighted_positive_hit
                + 0.30 * min_positive_hit
                + 0.05 * weighted_target_hit
            )

            results.append({
                "model": config["name"],
                "threshold": threshold,
                "eligible": eligible,
                "positive_hit": weighted_positive_hit,
                "target_hit": weighted_target_hit,
                "min_positive_hit": min_positive_hit,
                "total_signals": total_signals,
                "min_fold_signals": min_signals,
                "min_fold_days": min_days,
                "avg_return_pct": weighted_return,
                "robust_score": robust_score,
                "fold_positive_hits": "|".join(
                    f"{m['positive_hit']:.2f}"
                    for m in metrics
                ),
                "fold_signals": "|".join(
                    str(m["signals"])
                    for m in metrics
                ),
                "fold_days": "|".join(
                    str(m["days"])
                    for m in metrics
                ),
            })

    results = pd.DataFrame(results)

    eligible = results.loc[
        results["eligible"]
    ].copy()

    eligible = eligible.sort_values(
        [
            "robust_score",
            "positive_hit",
            "min_positive_hit",
        ],
        ascending=False,
    )

    print()
    print("===== 5D FINAL DEVELOPMENT =====")

    if eligible.empty:
        print("NO ELIGIBLE CANDIDATE")
        print()
        print("Best non-eligible candidates:")
        print(
            results.sort_values(
                "robust_score",
                ascending=False,
            ).head(10).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
    else:
        print(
            eligible.head(10).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print()
        print("===== LOCKED 5D FINAL WINNER =====")
        print(eligible.iloc[0].to_string())

    base.exp.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        base.exp.OUTPUT_DIR
        / "swing_5d_final_development.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    run()