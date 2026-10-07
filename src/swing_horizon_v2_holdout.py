from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from . import swing_horizon_models as base
from . import swing_horizon_v2 as v2


TRAIN_END = pd.Timestamp("2024-12-30")
HOLDOUT_START = pd.Timestamp("2025-01-02")

# 개발구간에서 고정한 승자 설정
# 방금 v2 실험의 LOCKED WINNER 기준
WINNERS = {
    3: {
        "threshold": 0.54,
        "target_return": 0.010,
    },
    5: {
        "threshold": 0.54,
        "target_return": 0.015,
    },
    10: {
        "threshold": 0.52,
        "target_return": 0.020,
    },
}

# v2에서 사용한 d3_regular 모델
MODEL_PARAMS = {
    "learning_rate": 0.035,
    "max_depth": 3,
    "max_iter": 350,
    "l2_regularization": 1.5,
    "min_samples_leaf": 40,
    "random_state": 42,
}


def build_full_dataset(horizon: int):
    # 최종 holdout 검증에서만 2025~2026 데이터까지 사용한다.
    old_base_dev_end = base.DEV_END
    old_exp_label_end = base.exp.LABEL_DATA_END

    try:
        base.DEV_END = pd.Timestamp("2026-12-31")
        base.exp.LABEL_DATA_END = pd.Timestamp("2026-12-31")

        # build_horizon_dataset 내부에서도 LABEL_DATA_END를
        # 2024-12-31로 다시 설정하므로, builder를 직접 쓰지 않고
        # 임시로 모듈의 개발 종료 범위를 확장한다.
        original_build_dataset = base.exp.build_dataset

        def build_dataset_full(data_dir):
            old_label = base.exp.LABEL_DATA_END
            try:
                base.exp.LABEL_DATA_END = pd.Timestamp("2026-12-31")
                return original_build_dataset(data_dir)
            finally:
                base.exp.LABEL_DATA_END = old_label

        base.exp.build_dataset = build_dataset_full

        data = base.build_horizon_dataset(horizon).copy()

    finally:
        base.exp.build_dataset = original_build_dataset
        base.DEV_END = old_base_dev_end
        base.exp.LABEL_DATA_END = old_exp_label_end

    return_col = f"return_{horizon}d"
    target_col = f"v2_target_{horizon}d"

    data[target_col] = (
        data[return_col]
        >= WINNERS[horizon]["target_return"]
    ).astype(int)

    return data


def evaluate(
    frame,
    probabilities,
    horizon,
    threshold,
):
    mask = probabilities >= threshold
    signals = frame.loc[mask].copy()

    return_col = f"return_{horizon}d"
    target_col = f"v2_target_{horizon}d"

    if signals.empty:
        return {
            "positive_hit": np.nan,
            "target_hit": np.nan,
            "signals": 0,
            "signal_days": 0,
            "avg_return": np.nan,
            "median_return": np.nan,
        }

    return {
        # 실제 N일 뒤 종가가 상승했는가
        "positive_hit": float(
            (signals[return_col] > 0).mean() * 100
        ),

        # 목표 상승폭까지 달성했는가
        "target_hit": float(
            signals[target_col].mean() * 100
        ),

        "signals": int(len(signals)),

        "signal_days": int(
            signals["date"].nunique()
        ),

        "avg_return": float(
            signals[return_col].mean() * 100
        ),

        "median_return": float(
            signals[return_col].median() * 100
        ),
    }


def run():
    overall_rows = []
    stability_rows = []

    periods = [
        ("2025 H1", "2025-01-02", "2025-06-30"),
        ("2025 H2", "2025-07-01", "2025-12-30"),
        ("2026 H1", "2026-01-02", "2026-06-30"),
        ("2026 H2", "2026-07-01", "2026-09-30"),
    ]

    for horizon in [3, 5, 10]:
        print()
        print(
            f"===== TESTING LOCKED {horizon}D ====="
        )

        data = build_full_dataset(horizon)

        target_col = f"v2_target_{horizon}d"

        train = data.loc[
            data["date"] <= TRAIN_END
        ].copy()

        holdout = data.loc[
            data["date"] >= HOLDOUT_START
        ].copy()

        # 라벨 누수 방지:
        # 학습 마지막 horizon 거래일 제거
        train_dates = np.sort(
            train["date"].unique()
        )

        if len(train_dates) <= horizon:
            raise RuntimeError(
                f"Not enough train dates for {horizon}D"
            )

        safe_train_end = train_dates[
            -(horizon + 1)
        ]

        train = train.loc[
            train["date"] <= safe_train_end
        ].copy()

        if train.empty or holdout.empty:
            raise RuntimeError(
                f"Empty train/holdout for {horizon}D"
            )

        model = make_pipeline(
            SimpleImputer(
                strategy="median"
            ),
            HistGradientBoostingClassifier(
                **MODEL_PARAMS
            ),
        )

        model.fit(
            train[base.FEATURES],
            train[target_col].astype(int),
        )

        probability = model.predict_proba(
            holdout[base.FEATURES]
        )[:, 1]

        threshold = WINNERS[horizon]["threshold"]

        overall = evaluate(
            holdout,
            probability,
            horizon,
            threshold,
        )

        overall_rows.append({
            "horizon": horizon,
            "threshold": threshold,
            "positive_hit_pct":
                overall["positive_hit"],
            "target_hit_pct":
                overall["target_hit"],
            "signals":
                overall["signals"],
            "signal_days":
                overall["signal_days"],
            "avg_return_pct":
                overall["avg_return"],
            "median_return_pct":
                overall["median_return"],
        })

        holdout = holdout.copy()
        holdout["probability"] = probability

        for (
            period_name,
            start,
            end,
        ) in periods:

            frame = holdout.loc[
                holdout["date"].between(
                    pd.Timestamp(start),
                    pd.Timestamp(end),
                )
            ].copy()

            if frame.empty:
                continue

            period_result = evaluate(
                frame,
                frame["probability"].to_numpy(),
                horizon,
                threshold,
            )

            stability_rows.append({
                "horizon":
                    horizon,

                "period":
                    period_name,

                "positive_hit_pct":
                    period_result[
                        "positive_hit"
                    ],

                "target_hit_pct":
                    period_result[
                        "target_hit"
                    ],

                "signals":
                    period_result[
                        "signals"
                    ],

                "signal_days":
                    period_result[
                        "signal_days"
                    ],

                "avg_return_pct":
                    period_result[
                        "avg_return"
                    ],

                "median_return_pct":
                    period_result[
                        "median_return"
                    ],
            })

    overall_df = pd.DataFrame(
        overall_rows
    )

    stability_df = pd.DataFrame(
        stability_rows
    )

    print()
    print(
        "===== SWING V2 LOCKED HOLDOUT RESULTS ====="
    )

    print(
        overall_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "===== SWING V2 PERIOD STABILITY ====="
    )

    print(
        stability_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "===== FINAL HORIZON DECISIONS ====="
    )

    for horizon in [3, 5, 10]:

        overall = overall_df.loc[
            overall_df["horizon"] == horizon
        ].iloc[0]

        periods_df = stability_df.loc[
            stability_df["horizon"] == horizon
        ]

        positive_periods = int(
            (
                periods_df["positive_hit_pct"]
                > 50
            ).sum()
        )

        enough_signals = bool(
            (
                periods_df["signals"]
                >= 250
            ).all()
        )

        enough_days = bool(
            (
                periods_df["signal_days"]
                >= 50
            ).all()
        )

        positive_return_periods = int(
            (
                periods_df["avg_return_pct"]
                > 0
            ).sum()
        )

        passed = (
            overall["positive_hit_pct"] > 50
            and positive_periods == len(periods_df)
            and positive_return_periods
                == len(periods_df)
            and enough_signals
            and enough_days
        )

        print()
        print(f"{horizon}D")
        print(
            f"Overall positive hit: "
            f"{overall['positive_hit_pct']:.4f}%"
        )
        print(
            f"Overall target hit:   "
            f"{overall['target_hit_pct']:.4f}%"
        )
        print(
            f"Average return:       "
            f"{overall['avg_return_pct']:.4f}%"
        )
        print(
            f"Positive-hit periods: "
            f"{positive_periods}/"
            f"{len(periods_df)}"
        )
        print(
            f"Positive-return periods: "
            f"{positive_return_periods}/"
            f"{len(periods_df)}"
        )
        print(
            f"Enough signals:       "
            f"{enough_signals}"
        )
        print(
            f"Enough signal days:   "
            f"{enough_days}"
        )

        if passed:
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL")

    base.exp.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_df.to_csv(
        base.exp.OUTPUT_DIR
        / "swing_horizon_v2_holdout.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stability_df.to_csv(
        base.exp.OUTPUT_DIR
        / "swing_horizon_v2_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    run()