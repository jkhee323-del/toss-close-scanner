from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from . import next_day_validation_feature_models as exp


DEV_END = pd.Timestamp("2024-12-30")
HORIZONS = [3, 5, 10]
HISTORY_DIR = Path("data/history")

FEATURES = exp.BASE_FEATURES + [
    "ret_3d",
    "ret_10d",
    "ma5_distance",
    "ma20_distance",
    "ma60_distance",
    "ma20_slope_5d",
    "distance_from_20d_high",
    "distance_from_20d_low",
    "volume_ratio_5d",
    "trading_value_ratio_20d",
    "atr14_pct",
    "volatility_10d",
    "relative_strength_5d",
    "relative_strength_20d",
]

FOLDS = [
    ("2022-12-29", "2023-01-02", "2023-06-30"),
    ("2023-06-30", "2023-07-03", "2023-12-28"),
    ("2023-12-28", "2024-01-02", "2024-06-28"),
    ("2024-06-28", "2024-07-01", "2024-12-30"),
]

THRESHOLDS = [0.52, 0.54, 0.56, 0.58, 0.60]


def build_horizon_dataset(horizon: int) -> pd.DataFrame:
    """
    종목별 원본 CSV에서 horizon 거래일 뒤 수익률과 target을 만든다.
    이후 기존 feature dataset과 날짜 + OHLCV 기준으로 결합한다.
    """
    target_frames = []

    for path in sorted(HISTORY_DIR.glob("*.csv")):
        raw = pd.read_csv(path)

        required = {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

        if not required.issubset(raw.columns):
            continue

        raw["date"] = pd.to_datetime(
            raw["date"],
            errors="coerce",
        )

        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]:
            raw[col] = pd.to_numeric(
                raw[col],
                errors="coerce",
            )

        raw = (
            raw
            .dropna(subset=["date", "close"])
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True)
        )

        future_close = raw["close"].shift(-horizon)

        raw[f"return_{horizon}d"] = (
            future_close / raw["close"] - 1
        )

        raw[f"target_{horizon}d"] = np.where(
            future_close.notna(),
            (future_close > raw["close"]).astype(float),
            np.nan,
        )

        raw["symbol"] = path.stem

        target_frames.append(
            raw[
                [
                    "symbol",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    f"return_{horizon}d",
                    f"target_{horizon}d",
                ]
            ]
        )

    if not target_frames:
        raise RuntimeError(
            "No history CSV files found."
        )

    targets = pd.concat(
        target_frames,
        ignore_index=True,
    )

    exp.LABEL_DATA_END = pd.Timestamp("2024-12-31")

    features = exp.build_dataset(
        exp.DATA_DIR
    ).copy()

    features["date"] = pd.to_datetime(
        features["date"],
        errors="coerce",
    )

    key_cols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        features[col] = pd.to_numeric(
            features[col],
            errors="coerce",
        )

        targets[col] = pd.to_numeric(
            targets[col],
            errors="coerce",
        )

    merged = features.merge(
        targets[
            key_cols
            + [
                "symbol",
                f"return_{horizon}d",
                f"target_{horizon}d",
            ]
        ],
        on=key_cols,
        how="inner",
    )

    merged = merged.loc[
        merged["date"] <= DEV_END
    ].copy()

    merged = merged.dropna(
        subset=[
            f"return_{horizon}d",
            f"target_{horizon}d",
        ]
    )

    print(
        f"{horizon}D dataset: "
        f"{len(merged):,} rows, "
        f"{targets['symbol'].nunique()} symbols"
    )

    return merged


def evaluate(
    probability,
    frame,
    threshold,
    horizon,
):
    mask = probability >= threshold
    signals = frame.loc[mask]

    if signals.empty:
        return {
            "signals": 0,
            "signal_days": 0,
            "hit": np.nan,
            "avg_return": np.nan,
        }

    return {
        "signals": int(len(signals)),
        "signal_days": int(
            signals["date"].nunique()
        ),
        "hit": float(
            signals[
                f"target_{horizon}d"
            ].mean() * 100
        ),
        "avg_return": float(
            signals[
                f"return_{horizon}d"
            ].mean() * 100
        ),
    }


def run():
    all_results = []

    for horizon in HORIZONS:
        print()
        print(
            f"BUILDING {horizon}D DATASET..."
        )

        data = build_horizon_dataset(
            horizon
        )

        target_col = (
            f"target_{horizon}d"
        )

        cached = []

        for (
            train_end,
            val_start,
            val_end,
        ) in FOLDS:

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

            # 라벨 누수 방지:
            # 학습 마지막 horizon 거래일 제거
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
                    learning_rate=0.035,
                    max_depth=3,
                    max_iter=350,
                    l2_regularization=1.5,
                    min_samples_leaf=40,
                    random_state=42,
                ),
            )

            model.fit(
                train[FEATURES],
                train[
                    target_col
                ].astype(int),
            )

            probability = (
                model.predict_proba(
                    validation[FEATURES]
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

            weighted_hit = (
                sum(
                    m["hit"]
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

            min_signals = min(
                m["signals"]
                for m in metrics
            )

            min_hit = min(
                m["hit"]
                for m in metrics
            )

            eligible = (
                total_signals >= 1500
                and min_signals >= 250
            )

            all_results.append(
                {
                    "horizon_days":
                        horizon,
                    "threshold":
                        threshold,
                    "eligible":
                        eligible,
                    "weighted_hit":
                        weighted_hit,
                    "min_fold_hit":
                        min_hit,
                    "total_signals":
                        total_signals,
                    "min_fold_signals":
                        min_signals,
                    "weighted_return_pct":
                        weighted_return,
                    "fold_hits":
                        "|".join(
                            f"{m['hit']:.2f}"
                            for m in metrics
                        ),
                    "fold_signals":
                        "|".join(
                            str(m["signals"])
                            for m in metrics
                        ),
                }
            )

    results = pd.DataFrame(
        all_results
    )

    print()
    print(
        "===== 3 / 5 / 10 DAY "
        "DEVELOPMENT RESULTS ====="
    )

    for horizon in HORIZONS:
        section = results.loc[
            (
                results["horizon_days"]
                == horizon
            )
            & results["eligible"]
        ].copy()

        section = section.sort_values(
            [
                "weighted_hit",
                "min_fold_hit",
                "weighted_return_pct",
            ],
            ascending=False,
        )

        print()
        print(
            f"===== {horizon} "
            "TRADING DAYS ====="
        )

        if section.empty:
            print(
                "NO ELIGIBLE CANDIDATE"
            )
            continue

        print(
            section.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

        winner = section.iloc[0]

        print()
        print(
            f"LOCKED {horizon}D WINNER"
        )
        print(
            winner.to_string()
        )

    exp.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        exp.OUTPUT_DIR
        / "swing_horizon_development.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    run()