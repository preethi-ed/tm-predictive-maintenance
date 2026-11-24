import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from tmu.models.classification.vanilla_classifier import TMClassifier
from src.data import load_ai4i
from run_tm import (DATA_PATH, CLAUSES, T, S, EPOCHS, fit_thresholds, fit_thresholds_supervised,
                    booleanize, oversample, class_scores, best_threshold)

MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def tm_fold(Xtr_df, ytr, Xte_df, seed):
    th = fit_thresholds_supervised(Xtr_df, ytr)
    Xtr, _ = booleanize(Xtr_df, th)
    Xte, _ = booleanize(Xte_df, th)
    Xos, yos = oversample(Xtr, ytr.astype(np.uint32), seed=seed)
    tm = TMClassifier(CLAUSES, T, S, platform="CPU", weighted_clauses=True, max_included_literals=10)
    for _ in range(EPOCHS):
        tm.fit(Xos, yos)
    thr = best_threshold(ytr, class_scores(tm, Xtr))
    return (class_scores(tm, Xte) >= thr).astype(int)


def main():
    X, y = load_ai4i(DATA_PATH, physics=True)
    X = X.astype(float).reset_index(drop=True)
    y = np.asarray(y).astype(int)
    raw = pd.read_csv(DATA_PATH).reset_index(drop=True)
    assert len(raw) == len(X), "Row mismatch between raw CSV and loaded data"

    preds = {"Tsetlin Machine": np.zeros(len(y), int),
             "Decision tree": np.zeros(len(y), int),
             "Random forest": np.zeros(len(y), int)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        preds["Tsetlin Machine"][te] = tm_fold(X.iloc[tr], y[tr], X.iloc[te], seed=fold)
        dt = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
        preds["Decision tree"][te] = dt.fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        preds["Random forest"][te] = rf.fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        print(f"fold {fold} done")

    rows = []
    for mode in MODES:
        mask = (raw[mode].values == 1) & (y == 1)
        row = {"Failure mode": mode, "Count": int(mask.sum())}
        for name, p in preds.items():
            row[name] = round(p[mask].mean(), 3) if mask.sum() else np.nan
        rows.append(row)
    allf = {"Failure mode": "ALL", "Count": int((y == 1).sum())}
    for name, p in preds.items():
        allf[name] = round(p[y == 1].mean(), 3)
    rows.append(allf)
    table = pd.DataFrame(rows)
    print("\nRecall by failure mode (out-of-fold):\n")
    print(table.to_string(index=False))
    table.to_csv("tables/table_recall_by_mode.csv", index=False)

    dt = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42).fit(X, y)
    print("\nDecision tree rules (depth 5, full data):\n")
    print(export_text(dt, feature_names=list(X.columns), decimals=1))
    print(f"Decision tree leaves: {dt.get_n_leaves()}")


if __name__ == "__main__":
    main()