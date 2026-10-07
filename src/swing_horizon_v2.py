from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from . import swing_horizon_models as base


DEV_END = pd.Timestamp("2024-12-30")

# horizon별 의미 있는 상승 목표
TARGET_RETURNS = {
    3: 0.010,   # +1.0%
    5: 0.015,   # +1.5%
    10: 0.020,  # +2.0%
}

THRESHOLDS = [
    0.50,
    0.52,
    0.54,
    0.56,
    0.58,
    0.60,
]

MODEL_CONFIGS = [
    {
        "name": "d3_regular",
        "learning_rate": 0.035,
        "max_depth": 3,
        "max_iter": 350,
        "l2_regularization": 1.5,
        "min_samples_leaf": 40,
    },
    {
        "name": "d3_strong_reg",
        "learning_rate": 0.03,
        "max_depth": 3,
        "max_iter": 400,
        "l2_regularization": 3.0,
        "min_samples_leaf": 60,
    },
    {
        "name": "d4_regular",
        "learning_rate": 0.03,
        "max_depth": 4,
        "max_iter": 400,
        "l2_regularization": 2.5,
        "min_samples_leaf": 50,
    },
]


def prepare_data(horizon: int):
    data = base.build_horizon_dataset(
        horizon
    ).copy()

    return_col = f"return_{horizon}d"
    target_col = f"v2_target_{horizon}d"

    target_return = TARGET_RETURNS[horizon]

    data[target_col] = (
        data[return_col] >= target_return
    ).astype(int)

    data = data.loc[
        data["date"] <= DEV_END
    ].copy()

    return data


def evaluate(
    probability,
    frame,
    threshold,
    horizon,
):
    mask = probability >= threshold
    signals = frame.loc[mask]

    target_col = f"v2_target_{horizon}d"
    return_col = f"return_{horizon}d"

    if signals.empty:
        return {
            "signals": 0,
            "signal_days": 0,
            "target_hit": np.nan,
            "positive_hit": np.nan,
            "avg_return": np.nan,
            "median_return": np.nan,
        }

    return {
        "signals": int(len(signals)),

        "signal_days": int(
            signals["date"].nunique()
        ),

        # 목표 수익률 달성률
        "target_hit": float(
            signals[target_col].mean() * 100
        ),

        # 단순 플러스 수익률 비율도 별도로 확인
        "positive_hit": float(
            (
                signals[return_col] > 0
            ).mean() * 100
        ),

        "avg_return": float(
            signals[return_col].mean() * 100
        ),

        "median_return": float(
            signals[return_col].median() * 100
        ),
    }


def run():
    all_results = []

    for horizon in [3, 5, 10]:

        print()
        print(
            f"===== BUILDING V2 {horizon}D ====="
        )

        data = prepare_data(horizon)

        target_col = (
            f"v2_target_{horizon}d"
        )

        for config in MODEL_CONFIGS:

            cached = []

            for (
                train_end,
                val_start,
                val_end,
            ) in base.FOLDS:

                train_end = pd.Timestamp(
                    train_end
                )
                val_start = pd.Timestamp(
                    val_start
                )
                val_end = pd.Timestamp(
                    val_end
                )

                train = data.loc[
                    data["date"] <= train_end
                ].copy()

                validation = data.loc[
                    data["date"].between(
                        val_start,
                        val_end,
                    )
                ].copy()

                # 미래 horizon일의 라벨이
                # validation 기간으로 넘어가는 것 방지
                train_dates = np.sort(
                    train["date"].unique()
                )

                if len(train_dates) <= horizon:
                    continue

                safe_end = train_dates[
                    -(horizon + 1)
                ]

                train = train.loc[
                    train["date"] <= safe_end
                ].copy()

                if (
                    train.empty
                    or validation.empty
                ):
                    continue

                model = make_pipeline(
                    SimpleImputer(
                        strategy="median"
                    ),
                    HistGradientBoostingClassifier(
                        learning_rate=
                            config["learning_rate"],
                        max_depth=
                            config["max_depth"],
                        max_iter=
                            config["max_iter"],
                        l2_regularization=
                            config["l2_regularization"],
                        min_samples_leaf=
                            config["min_samples_leaf"],
                        random_state=42,
                    ),
                )

                model.fit(
                    train[base.FEATURES],
                    train[target_col],
                )

                probability = (
                    model.predict_proba(
                        validation[
                            base.FEATURES
                        ]
                    )[:, 1]
                )

                cached.append(
                    (
                        validation,
                        probability,
                    )
                )

            for threshold in THRESHOLDS:

                metrics = [
                    evaluate(
                        probability,
                        frame,
                        threshold,
                        horizon,
                    )
                    for (
                        frame,
                        probability,
                    ) in cached
                ]

                if not metrics:
                    continue

                total_signals = sum(
                    m["signals"]
                    for m in metrics
                )

                if total_signals == 0:
                    continue

                min_signals = min(
                    m["signals"]
                    for m in metrics
                )

                min_signal_days = min(
                    m["signal_days"]
                    for m in metrics
                )

                weighted_target_hit = (
                    sum(
                        m["target_hit"]
                        * m["signals"]
                        for m in metrics
                    )
                    / total_signals
                )

                weighted_positive_hit = (
                    sum(
                        m["positive_hit"]
                        * m["signals"]
                        for m in metrics
                    )
                    / total_signals
                )

                weighted_return = (
                    sum(
                        m["avg_return"]
                        * m["signals"]
                        for m in metrics
                    )
                    / total_signals
                )

                min_positive_hit = min(
                    m["positive_hit"]
                    for m in metrics
                )

                min_target_hit = min(
                    m["target_hit"]
                    for m in metrics
                )

                # 지나치게 희소한 후보 제외
                eligible = (
                    total_signals >= 1500
                    and min_signals >= 250
                    and min_signal_days >= 50
                )

                # 목표 달성률뿐 아니라
                # 실제 플러스 수익 비율과
                # 가장 약한 fold도 같이 반영
                robust_score = (
                    weighted_positive_hit
                    + 0.15 * min_positive_hit
                    + 0.05 * weighted_target_hit
                )

                all_results.append({
                    "horizon":
                        horizon,

                    "target_return_pct":
                        TARGET_RETURNS[horizon]
                        * 100,

                    "model":
                        config["name"],

                    "threshold":
                        threshold,

                    "eligible":
                        eligible,

                    "positive_hit":
                        weighted_positive_hit,

                    "target_hit":
                        weighted_target_hit,

                    "min_positive_hit":
                        min_positive_hit,

                    "min_target_hit":
                        min_target_hit,

                    "total_signals":
                        total_signals,

                    "min_fold_signals":
                        min_signals,

                    "min_signal_days":
                        min_signal_days,

                    "avg_return_pct":
                        weighted_return,

                    "robust_score":
                        robust_score,

                    "fold_positive_hits":
                        "|".join(
                            f"{m['positive_hit']:.2f}"
                            for m in metrics
                        ),

                    "fold_target_hits":
                        "|".join(
                            f"{m['target_hit']:.2f}"
                            for m in metrics
                        ),

                    "fold_signals":
                        "|".join(
                            str(m["signals"])
                            for m in metrics
                        ),
                })

    results = pd.DataFrame(
        all_results
    )

    print()
    print(
        "===== SWING V2 DEVELOPMENT RESULTS ====="
    )

    for horizon in [3, 5, 10]:

        section = results.loc[
            (
                results["horizon"]
                == horizon
            )
            & results["eligible"]
        ].copy()

        section = section.sort_values(
            [
                "robust_score",
                "positive_hit",
                "min_positive_hit",
            ],
            ascending=False,
        )

        print()
        print(
            f"===== {horizon}D V2 ====="
        )

        if section.empty:
            print(
                "NO ELIGIBLE CANDIDATE"
            )
            continue

        print(
            section.head(10).to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

        winner = section.iloc[0]

        print()
        print(
            f"LOCKED {horizon}D V2 WINNER"
        )
        print(
            winner.to_string()
        )

    base.exp.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        base.exp.OUTPUT_DIR
        / "swing_horizon_v2_development.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    run()