from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .indicators import add_indicators


FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "volume_ratio",
    "close_position",
    "near_20d_high",
    "rsi14",
      
]


def make_training_data(data_dir: str | Path) -> pd.DataFrame:
    """저장된 과거 주가들로 머신러닝 학습 데이터를 만듭니다."""

    data_dir = Path(data_dir)
    datasets = []

    for path in data_dir.glob("*.csv"):
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        df = add_indicators(df)

        # 다음 거래일 종가 수익률
        df["next_return"] = df["close"].shift(-1) / df["close"] - 1

        # 다음 거래일 상승 여부
        df["target"] = (df["next_return"] > 0).astype(int)

        df["symbol"] = path.stem

        datasets.append(df)

    if not datasets:
        raise ValueError("학습할 과거 데이터가 없습니다.")

    data = pd.concat(datasets, ignore_index=True)

    # 지표 계산이 불가능한 초반 구간과 마지막 날 제거
    data = data.dropna(
        subset=FEATURES + ["next_return"]
    ).reset_index(drop=True)

    return data


def train_model(data: pd.DataFrame):
    """다음 거래일 상승 여부를 학습합니다."""

    X = data[FEATURES]
    y = data["target"]

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(X, y)

    return model

def time_split_evaluate(data: pd.DataFrame, train_ratio: float = 0.8) -> dict:
    """
    과거 80%로 학습하고 미래 20%로 테스트합니다.
    랜덤 분할을 하지 않아 미래 정보가 과거 학습에 섞이는 것을 줄입니다.
    """

    data = data.sort_values("date").reset_index(drop=True)

    unique_dates = sorted(data["date"].unique())
    split_index = int(len(unique_dates) * train_ratio)

    split_date = unique_dates[split_index]

    train = data[data["date"] < split_date].copy()
    test = data[data["date"] >= split_date].copy()

    model = train_model(train)

    X_test = test[FEATURES]
    y_test = test["target"]

    probabilities = model.predict_proba(X_test)[:, 1]

    test["pred_probability"] = probabilities

    # 전체 방향 정확도
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = (predictions == y_test.to_numpy()).mean()

    # 모델이 강하게 상승을 예상한 경우
    high_confidence = test[test["pred_probability"] >= 0.60].copy()

    if len(high_confidence) > 0:
        high_confidence_win_rate = (
            high_confidence["target"].mean() * 100
        )

        high_confidence_avg_return = (
            high_confidence["next_return"].mean() * 100
        )
    else:
        high_confidence_win_rate = None
        high_confidence_avg_return = None

    return {
        "model": model,
        "train_rows": len(train),
        "test_rows": len(test),
        "split_date": pd.Timestamp(split_date),
        "accuracy": accuracy * 100,
        "high_confidence_signals": len(high_confidence),
        "high_confidence_win_rate": high_confidence_win_rate,
        "high_confidence_avg_return": high_confidence_avg_return,
        "test_data": test,
    }
def evaluate_daily_top5(data: pd.DataFrame, train_ratio: float = 0.8) -> dict:
    """
    과거 데이터로 학습한 뒤 미래 테스트 기간의 매 거래일마다
    상승확률이 가장 높은 5종목을 선택해 성과를 확인합니다.
    """

    data = data.sort_values("date").reset_index(drop=True)

    unique_dates = sorted(data["date"].unique())
    split_index = int(len(unique_dates) * train_ratio)
    split_date = unique_dates[split_index]

    train = data[data["date"] < split_date].copy()
    test = data[data["date"] >= split_date].copy()

    model = train_model(train)

    test["pred_probability"] = model.predict_proba(
        test[FEATURES]
    )[:, 1]

    # 매 거래일마다 모델 확률 TOP 5 선택
    top5 = (
        test.sort_values(
            ["date", "pred_probability"],
            ascending=[True, False],
        )
        .groupby("date", group_keys=False)
        .head(5)
        .copy()
    )

    daily = (
        top5.groupby("date")
        .agg(
            picks=("symbol", "size"),
            winners=("target", "sum"),
            win_rate=("target", "mean"),
            avg_return=("next_return", "mean"),
            avg_probability=("pred_probability", "mean"),
        )
        .reset_index()
    )

    return {
        "split_date": pd.Timestamp(split_date),
        "test_days": daily["date"].nunique(),
        "signals": len(top5),
        "win_rate": top5["target"].mean() * 100,
        "avg_next_return": top5["next_return"].mean() * 100,
        "median_next_return": top5["next_return"].median() * 100,
        "avg_model_probability": top5["pred_probability"].mean() * 100,
        "profitable_days": (daily["avg_return"] > 0).mean() * 100,
        "top5_data": top5,
        "daily_data": daily,
    }

def evaluate_entry_thresholds(data: pd.DataFrame) -> pd.DataFrame:
    """
    모델 예측값별 진입 기준을 비교합니다.
    어떤 기준 이상에서 종목을 포착할지 결정하기 위한 테스트입니다.
    """

    result = time_split_evaluate(data)
    test = result["test_data"].copy()

    rows = []

    for threshold in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
        selected = test[
            test["pred_probability"] >= threshold
        ].copy()

        if len(selected) == 0:
            continue

        rows.append(
            {
                "threshold": threshold,
                "signals": len(selected),
                "win_rate": selected["target"].mean() * 100,
                "avg_next_return": selected["next_return"].mean() * 100,
                "median_next_return": selected["next_return"].median() * 100,
                "avg_signals_per_day": (
                    len(selected) / selected["date"].nunique()
                ),
            }
        )

    return pd.DataFrame(rows)

def compare_extra_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    기존 7개 FEATURES를 기준으로 새 지표를 하나씩 추가해
    미래 테스트 구간의 TOP5 성능을 비교합니다.
    """

    extra_features = [
        "trading_value",
        "range_pct",
        "ma5_distance",
        "ma20_distance",
        "gap",
        "volatility_20d",
    ]

    data = data.sort_values("date").reset_index(drop=True)

    unique_dates = sorted(data["date"].unique())
    split_index = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_index]

    train = data[data["date"] < split_date].copy()
    test = data[data["date"] >= split_date].copy()

    results = []

    # 기존 7개 지표도 기준점으로 테스트
    feature_sets = [("기존_7개", FEATURES)]

    for feature in extra_features:
        feature_sets.append(
            (f"+{feature}", FEATURES + [feature])
        )

    for name, features in feature_sets:
        X_train = train[features]
        y_train = train["target"]

        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
        )

        model.fit(X_train, y_train)

        current_test = test.copy()

        current_test["pred_probability"] = (
            model.predict_proba(current_test[features])[:, 1]
        )

        top5 = (
            current_test.sort_values(
                ["date", "pred_probability"],
                ascending=[True, False],
            )
            .groupby("date", group_keys=False)
            .head(5)
            .copy()
        )

        daily = top5.groupby("date")["next_return"].mean()

        results.append(
            {
                "feature_test": name,
                "signals": len(top5),
                "win_rate": top5["target"].mean() * 100,
                "avg_return": top5["next_return"].mean() * 100,
                "median_return": top5["next_return"].median() * 100,
                "profitable_days": (daily > 0).mean() * 100,
            }
        )

    return pd.DataFrame(results)

def walk_forward_compare(data: pd.DataFrame) -> pd.DataFrame:
    """
    여러 시간 구간에서 기존 7개 모델과
    range_pct 추가 모델의 TOP5 성능을 비교합니다.
    """

    data = data.sort_values("date").reset_index(drop=True)
    unique_dates = sorted(data["date"].unique())

    feature_sets = [
        ("기존_7개", FEATURES),
        ("+range_pct", FEATURES + ["range_pct"]),
    ]

    # 전체 기간의 60%, 70%, 80% 지점에서 학습을 끝내고
    # 그 다음 10% 기간을 각각 테스트
    splits = [0.60, 0.70, 0.80]

    results = []

    for split_ratio in splits:
        train_end_index = int(len(unique_dates) * split_ratio)
        test_end_index = int(len(unique_dates) * (split_ratio + 0.10))

        train_end_date = unique_dates[train_end_index]
        test_end_date = unique_dates[min(test_end_index, len(unique_dates) - 1)]

        train = data[
            data["date"] < train_end_date
        ].copy()

        test = data[
            (data["date"] >= train_end_date)
            & (data["date"] < test_end_date)
        ].copy()

        for name, features in feature_sets:
            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=1.0,
                random_state=42,
            )

            model.fit(
                train[features],
                train["target"],
            )

            current_test = test.copy()

            current_test["pred_probability"] = (
                model.predict_proba(
                    current_test[features]
                )[:, 1]
            )

            top5 = (
                current_test.sort_values(
                    ["date", "pred_probability"],
                    ascending=[True, False],
                )
                .groupby("date", group_keys=False)
                .head(5)
                .copy()
            )

            daily_return = (
                top5.groupby("date")["next_return"].mean()
            )

            results.append(
                {
                    "train_end": pd.Timestamp(train_end_date),
                    "test_end": pd.Timestamp(test_end_date),
                    "model": name,
                    "signals": len(top5),
                    "win_rate": top5["target"].mean() * 100,
                    "avg_return": top5["next_return"].mean() * 100,
                    "median_return": top5["next_return"].median() * 100,
                    "profitable_days": (
                        (daily_return > 0).mean() * 100
                    ),
                }
            )

    return pd.DataFrame(results)

def evaluate_strong_rise_model(
    data: pd.DataFrame,
    rise_threshold: float = 0.05,
) -> dict:
    """
    다음 거래일 종가가 현재 종가보다 5% 이상 상승하는
    종목을 찾는 별도의 모델을 테스트합니다.
    """

    data = data.sort_values("date").reset_index(drop=True).copy()

    # +5% 이상 상승을 1로 표시
    data["strong_target"] = (
        data["next_return"] >= rise_threshold
    ).astype(int)

    unique_dates = sorted(data["date"].unique())
    split_index = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_index]

    train = data[data["date"] < split_date].copy()
    test = data[data["date"] >= split_date].copy()

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(
        train[FEATURES],
        train["strong_target"],
    )

    test["strong_score"] = model.predict_proba(
        test[FEATURES]
    )[:, 1]

    # 매일 강한 상승 점수가 가장 높은 5종목
    top5 = (
        test.sort_values(
            ["date", "strong_score"],
            ascending=[True, False],
        )
        .groupby("date", group_keys=False)
        .head(5)
        .copy()
    )

    daily_return = (
        top5.groupby("date")["next_return"].mean()
    )

    return {
        "split_date": pd.Timestamp(split_date),
        "test_days": top5["date"].nunique(),
        "signals": len(top5),
        "strong_hits": int(top5["strong_target"].sum()),
        "strong_hit_rate": top5["strong_target"].mean() * 100,
        "up_rate": top5["target"].mean() * 100,
        "avg_next_return": top5["next_return"].mean() * 100,
        "median_next_return": top5["next_return"].median() * 100,
        "profitable_days": (daily_return > 0).mean() * 100,
        "top5_data": top5,
    }

def evaluate_combined_model(data: pd.DataFrame) -> pd.DataFrame:
    """
    일반 상승 모델과 +3% 강한 상승 모델을 결합하고,
    결합 비율별 TOP5 성능을 비교합니다.
    """

    data = data.sort_values("date").reset_index(drop=True).copy()

    data["strong_target"] = (
        data["next_return"] >= 0.03
    ).astype(int)

    unique_dates = sorted(data["date"].unique())
    split_index = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_index]

    train = data[data["date"] < split_date].copy()
    test = data[data["date"] >= split_date].copy()

    # 1. 일반 상승 모델
    up_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )

    up_model.fit(
        train[FEATURES],
        train["target"],
    )

    # 2. +3% 강한 상승 모델
    strong_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )

    strong_model.fit(
        train[FEATURES],
        train["strong_target"],
    )

    test["up_score"] = up_model.predict_proba(
        test[FEATURES]
    )[:, 1]

    test["strong_score"] = strong_model.predict_proba(
        test[FEATURES]
    )[:, 1]

    results = []

    # 일반상승 : 강한상승 가중치
    weights = [
        (1.00, 0.00),
        (0.75, 0.25),
        (0.50, 0.50),
        (0.25, 0.75),
        (0.00, 1.00),
    ]

    for up_weight, strong_weight in weights:

        current = test.copy()

        current["combined_score"] = (
            current["up_score"] * up_weight
            + current["strong_score"] * strong_weight
        )

        top5 = (
            current.sort_values(
                ["date", "combined_score"],
                ascending=[True, False],
            )
            .groupby("date", group_keys=False)
            .head(5)
            .copy()
        )

        daily_return = (
            top5.groupby("date")["next_return"].mean()
        )

        results.append(
            {
                "up_weight": up_weight,
                "strong_weight": strong_weight,
                "signals": len(top5),
                "up_rate": top5["target"].mean() * 100,
                "strong_3pct_rate": (
                    (top5["next_return"] >= 0.03).mean() * 100
                ),
                "avg_return": top5["next_return"].mean() * 100,
                "median_return": top5["next_return"].median() * 100,
                "profitable_days": (
                    (daily_return > 0).mean() * 100
                ),
            }
        )

    return pd.DataFrame(results)

def walk_forward_combined_model(data: pd.DataFrame) -> pd.DataFrame:
    """
    50:50 결합 모델을 여러 미래 구간에서 검증합니다.

    각 구간마다 그 시점 이전 데이터만 학습하고,
    이후 10% 기간을 테스트합니다.
    """

    data = data.sort_values("date").reset_index(drop=True).copy()

    data["strong_target"] = (
        data["next_return"] >= 0.03
    ).astype(int)

    unique_dates = sorted(data["date"].unique())

    # 서로 다른 세 미래 구간
    splits = [0.60, 0.70, 0.80]

    results = []

    for split_ratio in splits:

        train_end_index = int(
            len(unique_dates) * split_ratio
        )

        test_end_index = int(
            len(unique_dates) * (split_ratio + 0.10)
        )

        train_end_date = unique_dates[train_end_index]

        test_end_date = unique_dates[
            min(test_end_index, len(unique_dates) - 1)
        ]

        train = data[
            data["date"] < train_end_date
        ].copy()

        test = data[
            (data["date"] >= train_end_date)
            & (data["date"] < test_end_date)
        ].copy()

        # 일반 상승 모델
        up_model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
        )

        up_model.fit(
            train[FEATURES],
            train["target"],
        )

        # +3% 강한 상승 모델
        strong_model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
        )

        strong_model.fit(
            train[FEATURES],
            train["strong_target"],
        )

        test["up_score"] = up_model.predict_proba(
            test[FEATURES]
        )[:, 1]

        test["strong_score"] = strong_model.predict_proba(
            test[FEATURES]
        )[:, 1]

        # 두 모델을 50:50으로 결합
        test["combined_score"] = (
            test["up_score"] * 0.50
            + test["strong_score"] * 0.50
        )

        # 각 거래일 TOP5
        top5 = (
            test.sort_values(
                ["date", "combined_score"],
                ascending=[True, False],
            )
            .groupby("date", group_keys=False)
            .head(5)
            .copy()
        )

        daily_return = (
            top5.groupby("date")["next_return"].mean()
        )

        results.append(
            {
                "train_end": pd.Timestamp(train_end_date),
                "test_end": pd.Timestamp(test_end_date),
                "test_days": top5["date"].nunique(),
                "signals": len(top5),

                "up_rate":
                    top5["target"].mean() * 100,

                "strong_3pct_rate":
                    (top5["next_return"] >= 0.03).mean() * 100,

                "avg_return":
                    top5["next_return"].mean() * 100,

                "median_return":
                    top5["next_return"].median() * 100,

                "profitable_days":
                    (daily_return > 0).mean() * 100,
            }
        )

    return pd.DataFrame(results)

def compare_liquidity_filters(data: pd.DataFrame) -> pd.DataFrame:
    """
    50:50 결합 모델에서 거래대금 필터별 TOP5 성능을 비교합니다.
    테스트 시점의 그날 거래대금만 사용합니다.
    """

    data = data.sort_values("date").reset_index(drop=True).copy()

    data["strong_target"] = (
        data["next_return"] >= 0.03
    ).astype(int)

    unique_dates = sorted(data["date"].unique())

    split_index = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_index]

    train = data[data["date"] < split_date].copy()
    test = data[data["date"] >= split_date].copy()

    # 일반 상승 모델
    up_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )

    up_model.fit(
        train[FEATURES],
        train["target"],
    )

    # +3% 강한 상승 모델
    strong_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )

    strong_model.fit(
        train[FEATURES],
        train["strong_target"],
    )

    test["up_score"] = up_model.predict_proba(
        test[FEATURES]
    )[:, 1]

    test["strong_score"] = strong_model.predict_proba(
        test[FEATURES]
    )[:, 1]

    test["combined_score"] = (
        test["up_score"] * 0.50
        + test["strong_score"] * 0.50
    )

    # 최소 일 거래대금 기준
    liquidity_levels = [
        ("필터없음", 0),
        ("10억원", 1_000_000_000),
        ("30억원", 3_000_000_000),
        ("50억원", 5_000_000_000),
        ("100억원", 10_000_000_000),
    ]

    results = []

    for label, minimum_value in liquidity_levels:

        filtered = test[
            test["trading_value"] >= minimum_value
        ].copy()

        top5 = (
            filtered.sort_values(
                ["date", "combined_score"],
                ascending=[True, False],
            )
            .groupby("date", group_keys=False)
            .head(5)
            .copy()
        )

        daily_return = (
            top5.groupby("date")["next_return"].mean()
        )

        results.append(
            {
                "filter": label,
                "test_days": top5["date"].nunique(),
                "signals": len(top5),
                "up_rate": top5["target"].mean() * 100,
                "strong_3pct_rate": (
                    (top5["next_return"] >= 0.03).mean() * 100
                ),
                "avg_return": top5["next_return"].mean() * 100,
                "median_return": top5["next_return"].median() * 100,
                "profitable_days": (
                    (daily_return > 0).mean() * 100
                ),
            }
        )

    return pd.DataFrame(results)

def walk_forward_liquidity_filter(data: pd.DataFrame) -> pd.DataFrame:
    """
    50:50 결합 모델에서
    필터 없음 vs 거래대금 100억원 이상을
    여러 미래 기간에서 비교합니다.
    """

    data = data.sort_values("date").reset_index(drop=True).copy()

    data["strong_target"] = (
        data["next_return"] >= 0.03
    ).astype(int)

    unique_dates = sorted(data["date"].unique())
    splits = [0.60, 0.70, 0.80]

    results = []

    for split_ratio in splits:

        train_end_index = int(
            len(unique_dates) * split_ratio
        )

        test_end_index = int(
            len(unique_dates) * (split_ratio + 0.10)
        )

        train_end_date = unique_dates[train_end_index]

        test_end_date = unique_dates[
            min(test_end_index, len(unique_dates) - 1)
        ]

        train = data[
            data["date"] < train_end_date
        ].copy()

        test = data[
            (data["date"] >= train_end_date)
            & (data["date"] < test_end_date)
        ].copy()

        # 일반 상승 모델
        up_model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
        )

        up_model.fit(
            train[FEATURES],
            train["target"],
        )

        # +3% 강한 상승 모델
        strong_model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
        )

        strong_model.fit(
            train[FEATURES],
            train["strong_target"],
        )

        test["up_score"] = up_model.predict_proba(
            test[FEATURES]
        )[:, 1]

        test["strong_score"] = strong_model.predict_proba(
            test[FEATURES]
        )[:, 1]

        test["combined_score"] = (
            test["up_score"] * 0.50
            + test["strong_score"] * 0.50
        )

        filters = [
            ("필터없음", 0),
            ("100억원", 10_000_000_000),
        ]

        for label, minimum_value in filters:

            filtered = test[
                test["trading_value"] >= minimum_value
            ].copy()

            top5 = (
                filtered.sort_values(
                    ["date", "combined_score"],
                    ascending=[True, False],
                )
                .groupby("date", group_keys=False)
                .head(5)
                .copy()
            )

            daily_return = (
                top5.groupby("date")["next_return"].mean()
            )

            results.append(
                {
                    "train_end":
                        pd.Timestamp(train_end_date),

                    "test_end":
                        pd.Timestamp(test_end_date),

                    "filter":
                        label,

                    "test_days":
                        top5["date"].nunique(),

                    "signals":
                        len(top5),

                    "up_rate":
                        top5["target"].mean() * 100,

                    "strong_3pct_rate":
                        (
                            top5["next_return"] >= 0.03
                        ).mean() * 100,

                    "avg_return":
                        top5["next_return"].mean() * 100,

                    "median_return":
                        top5["next_return"].median() * 100,

                    "profitable_days":
                        (daily_return > 0).mean() * 100,
                }
            )

    return pd.DataFrame(results)