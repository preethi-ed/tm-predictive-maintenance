import os, time, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score
from tmu.models.classification.vanilla_classifier import TMClassifier
from src.data import load_ai4i
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = os.path.join("data", "ai4i2020.csv")
TABLES_DIR = "tables"
QUANTILES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
CLAUSES, T, S, EPOCHS = 600, 300, 8.0, 30
POS_RATIO = 0.3


def fit_thresholds(X_train):
    th = {}
    for col in X_train.columns:
        vals = X_train[col].values
        if set(np.unique(vals)) <= {0, 1}:
            th[col] = None
        else:
            th[col] = np.unique(np.quantile(vals, QUANTILES))
    return th



def fit_thresholds_supervised(X_train, y_train, max_cuts=8):
    th = {}
    for col in X_train.columns:
        vals = X_train[col].values
        if set(np.unique(vals)) <= {0, 1}:
            th[col] = None
            continue
        tree = DecisionTreeClassifier(max_leaf_nodes=max_cuts + 1, class_weight="balanced",
                                      random_state=0).fit(vals.reshape(-1, 1), y_train)
        cuts = tree.tree_.threshold[tree.tree_.feature >= 0]
        quant = np.quantile(vals, [0.05, 0.5, 0.95])
        th[col] = np.unique(np.concatenate([cuts, quant]))
    return th


def booleanize(X, th):
    cols, names = [], []
    for col, cuts in th.items():
        v = X[col].values
        if cuts is None:
            cols.append(v.astype(np.uint32)); names.append(f"{col}")
        else:
            for c in cuts:
                cols.append((v > c).astype(np.uint32)); names.append(f"{col} > {c:.4g}")
    return np.stack(cols, axis=1), names


def oversample(Xb, y, seed):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    target = int(POS_RATIO * len(neg))
    extra = rng.choice(pos, size=max(target - len(pos), 0), replace=True)
    idx = rng.permutation(np.concatenate([neg, pos, extra]))
    return Xb[idx], y[idx]


def class_scores(tm, Xb):
    _, sums = tm.predict(Xb, return_class_sums=True)
    sums = np.asarray(sums)
    return (sums[:, 1] - sums[:, 0]).astype(float)


def best_threshold(y, score):
    best_t, best_f1 = 0.0, -1.0
    for t in np.unique(score):
        f1 = f1_score(y, (score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def run(physics, repeats):
    X, y = load_ai4i(DATA_PATH, physics=physics)
    X = X.astype(float)
    y = np.asarray(y).astype(np.uint32)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        th = fit_thresholds_supervised(X.iloc[tr], y[tr])
        Xtr, names = booleanize(X.iloc[tr], th)
        Xte, _ = booleanize(X.iloc[te], th)
        Xtr_os, ytr_os = oversample(Xtr, y[tr], seed=fold)
        tm = TMClassifier(CLAUSES, T, S, platform="CPU", weighted_clauses=True,
                          max_included_literals=10)
        t0 = time.time()
        for _ in range(EPOCHS):
            tm.fit(Xtr_os, ytr_os)
        train_s = time.time() - t0
        thr = best_threshold(y[tr], class_scores(tm, Xtr))
        score = class_scores(tm, Xte)
        pred = (score >= thr).astype(int)
        size_kb = CLAUSES * 2 * 2 * Xtr.shape[1] / 8 / 1024
        rows.append({
            "fold": fold,
            "f1": f1_score(y[te], pred),
            "precision": precision_score(y[te], pred, zero_division=0),
            "recall": recall_score(y[te], pred),
            "pr_auc": average_precision_score(y[te], score),
            "train_s": train_s,
            "size_kb": size_kb,
            "n_bool_features": Xtr.shape[1],
        })
        print(f"fold {fold}: F1={rows[-1]['f1']:.3f} P={rows[-1]['precision']:.3f} "
              f"R={rows[-1]['recall']:.3f} PR-AUC={rows[-1]['pr_auc']:.3f} ({train_s:.0f}s)")
    df = pd.DataFrame(rows)
    tag = "physics" if physics else "raw"
    os.makedirs(TABLES_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLES_DIR, f"tm_folds_{tag}.csv"), index=False)
    summary = df.drop(columns="fold").agg(["mean", "std"]).T.round(3)
    summary.to_csv(os.path.join(TABLES_DIR, f"table_tm_{tag}.csv"))
    print(f"\n=== Tsetlin Machine ({tag}) ===\n{summary}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--physics", action="store_true")
    p.add_argument("--quick", action="store_true", help="1 repeat instead of 3")
    a = p.parse_args()
    run(a.physics, repeats=1 if a.quick else 3)