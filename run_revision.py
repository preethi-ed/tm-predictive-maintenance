# run_revision.py — nested-threshold evaluation for all models + common size accounting
import os, sys, time, pickle, argparse, platform
from importlib.metadata import version
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tmu.models.classification.vanilla_classifier import TMClassifier
from src.data import load_ai4i
from run_tm import (DATA_PATH, TABLES_DIR, CLAUSES, T, S, EPOCHS,
                    fit_thresholds_supervised, booleanize, oversample,
                    class_scores, best_threshold)

CV_SEED = 42        # outer folds (same as run_tm.py)
INNER_VAL = 0.2     # share of each training fold used only to pick the threshold


def save_versions():
    os.makedirs(TABLES_DIR, exist_ok=True)
    lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}"]
    for pkg in ["tmu", "scikit-learn", "numpy", "pandas"]:
        try:
            lines.append(f"{pkg} {version(pkg)}")
        except Exception:
            lines.append(f"{pkg} unknown")
    with open(os.path.join(TABLES_DIR, "versions.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def pickle_kb(model):
    try:
        return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 1024
    except Exception:
        return np.nan


def baselines(seed):
    return {
        "RF": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     n_jobs=-1, random_state=seed),
        "DT": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=seed),
        "LR": make_pipeline(StandardScaler(),
                            LogisticRegression(class_weight="balanced", max_iter=2000)),
        "MLP": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                           early_stopping=True, random_state=seed)),
    }


def metrics(y_true, score, thr):
    pred = (score >= thr).astype(int)
    return {"f1": f1_score(y_true, pred, zero_division=0),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "pr_auc": average_precision_score(y_true, score)}


def run(physics, repeats):
    X, y = load_ai4i(DATA_PATH, physics=physics)
    X = X.astype(float)
    y = np.asarray(y).astype(np.uint32)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=CV_SEED)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        # inner split: fit on `fit_idx`, choose threshold on `val_idx`, test on `te`
        fit_idx, val_idx = train_test_split(tr, test_size=INNER_VAL, stratify=y[tr],
                                            random_state=fold)
        Xf, Xv, Xt = X.iloc[fit_idx], X.iloc[val_idx], X.iloc[te]
        yf, yv, yt = y[fit_idx], y[val_idx], y[te]

        # ---- Tsetlin Machine ----
        th = fit_thresholds_supervised(Xf, yf)
        Bf, _ = booleanize(Xf, th)
        Bv, _ = booleanize(Xv, th)
        Bt, _ = booleanize(Xt, th)
        Bf_os, yf_os = oversample(Bf, yf, seed=fold)
        tm = TMClassifier(CLAUSES, T, S, platform="CPU", weighted_clauses=True,
                          max_included_literals=10)
        t0 = time.time()
        for _ in range(EPOCHS):
            tm.fit(Bf_os, yf_os)
        thr = best_threshold(yv, class_scores(tm, Bv))
        r = metrics(yt, class_scores(tm, Bt), thr)
        r.update(model="TM", fold=fold, train_s=time.time() - t0,
                 size_pickle_kb=pickle_kb(tm),
                 size_bitlevel_kb=CLAUSES * 2 * 2 * Bf.shape[1] / 8 / 1024,
                 n_features=Bf.shape[1])
        rows.append(r)

        # ---- Baselines (same inner split, same threshold rule) ----
        for name, model in baselines(fold).items():
            t0 = time.time()
            model.fit(Xf.values, yf)
            thr = best_threshold(yv, model.predict_proba(Xv.values)[:, 1])
            r = metrics(yt, model.predict_proba(Xt.values)[:, 1], thr)
            r.update(model=name, fold=fold, train_s=time.time() - t0,
                     size_pickle_kb=pickle_kb(model), size_bitlevel_kb=np.nan,
                     n_features=Xf.shape[1])
            rows.append(r)

        msg = "  ".join(f"{x['model']} F1={x['f1']:.3f}" for x in rows[-5:])
        print(f"fold {fold}: {msg}")

    df = pd.DataFrame(rows)
    tag = "physics" if physics else "raw"
    df.to_csv(os.path.join(TABLES_DIR, f"revision_folds_{tag}.csv"), index=False)
    cols = ["f1", "precision", "recall", "pr_auc", "size_pickle_kb", "size_bitlevel_kb", "train_s"]
    summary = df.groupby("model")[cols].agg(["mean", "std"]).round(3)
    summary.to_csv(os.path.join(TABLES_DIR, f"table_revision_{tag}.csv"))
    pretty = pd.DataFrame({c: summary[c]["mean"].map("{:.3f}".format) + " ± " +
                              summary[c]["std"].map("{:.3f}".format) for c in cols[:4]})
    pretty["pickle_KB"] = summary["size_pickle_kb"]["mean"].round(1)
    pretty["bit_KB"] = summary["size_bitlevel_kb"]["mean"].round(1)
    print(f"\n=== Nested-threshold results ({tag}) ===\n{pretty}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--physics", action="store_true")
    p.add_argument("--quick", action="store_true", help="1 repeat instead of 3")
    a = p.parse_args()
    save_versions()
    run(a.physics, repeats=1 if a.quick else 3)