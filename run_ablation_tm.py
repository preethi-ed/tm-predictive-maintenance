# run_ablation_tm.py — TM ablation under the nested protocol (physics features)
import os, time, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import f1_score, average_precision_score
from tmu.models.classification.vanilla_classifier import TMClassifier
from src.data import load_ai4i
from run_tm import (DATA_PATH, TABLES_DIR, CLAUSES, T, S, EPOCHS, fit_thresholds,
                    fit_thresholds_supervised, booleanize, oversample, class_scores, best_threshold)

VARIANTS = {
    "quantile, no cap": (False, False),
    "supervised, no cap": (True, False),
    "supervised + cap 10": (True, True),
}


def run(repeats):
    X, y = load_ai4i(DATA_PATH, physics=True)
    X = X.astype(float)
    y = np.asarray(y).astype(np.uint32)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        fit_idx, val_idx = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=fold)
        for name, (supervised, cap) in VARIANTS.items():
            th = fit_thresholds_supervised(X.iloc[fit_idx], y[fit_idx]) if supervised else fit_thresholds(X.iloc[fit_idx])
            Bf, _ = booleanize(X.iloc[fit_idx], th)
            Bv, _ = booleanize(X.iloc[val_idx], th)
            Bt, _ = booleanize(X.iloc[te], th)
            Bo, yo = oversample(Bf, y[fit_idx], seed=fold)
            kw = {"max_included_literals": 10} if cap else {}
            tm = TMClassifier(CLAUSES, T, S, platform="CPU", weighted_clauses=True, **kw)
            for _ in range(EPOCHS):
                tm.fit(Bo, yo)
            thr = best_threshold(y[val_idx], class_scores(tm, Bv))
            st = class_scores(tm, Bt)
            rows.append({"variant": name, "fold": fold,
                         "f1": f1_score(y[te], (st >= thr).astype(int), zero_division=0),
                         "pr_auc": average_precision_score(y[te], st),
                         "bit_kb": CLAUSES * 2 * 2 * Bf.shape[1] / 8 / 1024})
        print(f"fold {fold}: " + "  ".join(f"{r['variant']} F1={r['f1']:.3f}" for r in rows[-3:]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, "ablation_nested_folds.csv"), index=False)
    s = df.groupby("variant")[["f1", "pr_auc", "bit_kb"]].agg(["mean", "std"]).round(3)
    s.to_csv(os.path.join(TABLES_DIR, "table_ablation_nested.csv"))
    print("\n=== TM ablation, nested protocol (physics) ===\n", s)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    run(1 if p.parse_args().quick else 3)