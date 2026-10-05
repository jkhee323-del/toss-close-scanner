from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from . import next_day_validation_feature_models as exp

# Development-only search. 2025+ is deliberately excluded from selection.
FOLDS = [
    (pd.Timestamp("2022-12-29"), pd.Timestamp("2023-01-02"), pd.Timestamp("2023-06-30")),
    (pd.Timestamp("2023-06-30"), pd.Timestamp("2023-07-03"), pd.Timestamp("2023-12-28")),
    (pd.Timestamp("2023-12-28"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-06-28")),
    (pd.Timestamp("2024-06-28"), pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-30")),
]
FEATURE_SETS = {
    "base7": exp.BASE_FEATURES,
    "trend_liquidity": exp.BASE_FEATURES + [
        "ma20_distance","ma60_distance","ma20_slope_5d","ret_3d","ret_10d",
        "log_trading_value","trading_value_ratio_20d","atr14_pct","volatility_10d",
        "relative_strength_5d","relative_strength_20d",
    ],
    "price_regime": exp.BASE_FEATURES + [
        "ma5_distance","ma20_distance","ma20_slope_5d","ret_3d","ret_10d",
        "distance_from_20d_high","distance_from_20d_low","atr14_pct",
        "market_return_5d","market_return_20d","relative_strength_20d",
    ],
}
PARAMS = [
    {"learning_rate":0.05,"max_depth":4,"max_iter":250,"l2_regularization":0.0},
    {"learning_rate":0.04,"max_depth":5,"max_iter":300,"l2_regularization":0.5},
    {"learning_rate":0.03,"max_depth":6,"max_iter":350,"l2_regularization":1.0},
]
THRESHOLDS = [0.50,0.52,0.54,0.56]
MIN_SIGNALS_PER_FOLD = 350

def score(prob, frame, threshold):
    mask=prob>=threshold
    s=frame.loc[mask]
    return {
        "signals":len(s),
        "hit":float(s["target"].mean()*100) if len(s) else np.nan,
        "ret":float(s["next_return"].mean()*100) if len(s) else np.nan,
        "days":int(s["date"].nunique()),
    }

def run():
    exp.LABEL_DATA_END=pd.Timestamp("2024-12-31")
    data=exp.build_dataset(exp.DATA_DIR)
    rows=[]
    for fs_name,features in FEATURE_SETS.items():
        for pi,p in enumerate(PARAMS):
            fold_cache=[]
            for train_end,val_start,val_end in FOLDS:
                tr=data.loc[data["date"]<=train_end]
                va=data.loc[data["date"].between(val_start,val_end)]
                model=make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingClassifier(random_state=42,**p))
                model.fit(tr[features],tr["target"].astype(int))
                fold_cache.append((va,model.predict_proba(va[features])[:,1]))
            for th in THRESHOLDS:
                ms=[score(prob,va,th) for va,prob in fold_cache]
                eligible=all(m["signals"]>=MIN_SIGNALS_PER_FOLD and m["days"]>=60 for m in ms)
                rows.append({
                    "features":fs_name,"params":pi,"threshold":th,"eligible":eligible,
                    "min_fold_hit":min(m["hit"] for m in ms),
                    "mean_fold_hit":np.mean([m["hit"] for m in ms]),
                    "total_signals":sum(m["signals"] for m in ms),
                    "mean_return_pct":np.average([m["ret"] for m in ms],weights=[m["signals"] for m in ms]),
                    "fold_hits":"|".join(f'{m["hit"]:.3f}' for m in ms),
                    "fold_signals":"|".join(str(m["signals"]) for m in ms),
                })
    out=pd.DataFrame(rows)
    eligible=out.loc[out["eligible"]].sort_values(["mean_fold_hit","min_fold_hit","mean_return_pct"],ascending=False)
    print("TOP DEVELOPMENT CANDIDATES")
    print(eligible.head(12).to_string(index=False))
    if eligible.empty:
        raise RuntimeError("No candidate met minimum fold coverage")
    best=eligible.iloc[0]
    print("LOCKED_BEST",best.to_dict())
    out.to_csv(exp.OUTPUT_DIR/"next_day_walkforward_search.csv",index=False)
    return best

if __name__=="__main__":
    run()
