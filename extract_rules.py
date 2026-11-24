import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tmu.models.classification.vanilla_classifier import TMClassifier
from src.data import load_ai4i
from run_tm import (DATA_PATH, CLAUSES, T, S, EPOCHS, fit_thresholds_supervised, booleanize,
                    oversample, class_scores, best_threshold)

TOP_K = 15
TRUE_RULES = {
    "HDF": "temp_diff < 8.6 K AND rpm < 1380",
    "PWF": "power < 3500 W OR power > 9000 W",
    "OSF": "wear x torque > 11000-13000 minNm (L/M/H)",
    "TWF": "tool wear in [200, 240] min",
}


def ta_action(tm, the_class, clause, ta):
    try:
        return tm.get_ta_action(clause, ta, the_class=the_class)
    except (AttributeError, TypeError):
        return tm.clause_banks[the_class].get_ta_action(clause, ta)


def clause_weights(tm, the_class):
    try:
        return np.array([tm.get_weight(c, the_class=the_class) for c in range(CLAUSES)])
    except (AttributeError, TypeError):
        return np.asarray(tm.weight_banks[the_class].get_weights())


def literals_to_rule(included, names):
    bounds = {}
    flags = []
    for idx, negated in included:
        name = names[idx]
        m = re.match(r"(.+) > (.+)", name)
        if m is None:
            flags.append(("NOT " if negated else "") + name)
            continue
        feat, cut = m.group(1), float(m.group(2))
        lo, hi = bounds.get(feat, (-np.inf, np.inf))
        if negated:
            hi = min(hi, cut)
        else:
            lo = max(lo, cut)
        bounds[feat] = (lo, hi)
    parts = []
    for feat, (lo, hi) in bounds.items():
        if lo > -np.inf and hi < np.inf:
            parts.append(f"{lo:.4g} < {feat} <= {hi:.4g}")
        elif lo > -np.inf:
            parts.append(f"{feat} > {lo:.4g}")
        else:
            parts.append(f"{feat} <= {hi:.4g}")
    return " AND ".join(parts + flags) if parts or flags else "(empty clause)"


def main():
    X, y = load_ai4i(DATA_PATH, physics=True)
    X = X.astype(float)
    y = np.asarray(y).astype(np.uint32)
    Xtr_df, Xte_df, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    th = fit_thresholds_supervised(Xtr_df, ytr)
    Xtr, names = booleanize(Xtr_df, th)
    Xte, _ = booleanize(Xte_df, th)
    Xtr_os, ytr_os = oversample(Xtr, ytr, seed=0)
    tm = TMClassifier(CLAUSES, T, S, platform="CPU", weighted_clauses=True, max_included_literals=10)
    for _ in range(EPOCHS):
        tm.fit(Xtr_os, ytr_os)
    thr = best_threshold(ytr, class_scores(tm, Xtr))
    pred = (class_scores(tm, Xte) >= thr).astype(int)
    print(f"Held-out F1: {f1_score(yte, pred):.3f}\n")

    n_feat = Xtr.shape[1]
    weights = clause_weights(tm, 1)
    order = np.argsort(-weights)
    rows, total_literals = [], 0
    for c in range(CLAUSES):
        inc = [(ta % n_feat, ta >= n_feat) for ta in range(2 * n_feat) if ta_action(tm, 1, c, ta) == 1]
        if weights[c] > 0:
            total_literals += len(inc)
        if c in order[:TOP_K]:
            rows.append({"clause": c, "weight": int(weights[c]),
                         "n_literals": len(inc), "rule": literals_to_rule(inc, names)})
    rules = pd.DataFrame(rows).sort_values("weight", ascending=False)
    pd.set_option("display.max_colwidth", 200)
    print("Top failure-voting clauses:\n")
    print(rules.to_string(index=False))
    print(f"\nPositive clauses: {(weights > 0).sum()}, total literals in them: {total_literals}")
    print(f"Rule-set size estimate: {total_literals * 2} bytes (2 bytes per literal)\n")
    print("Ground-truth AI4I rules for comparison:")
    for k, v in TRUE_RULES.items():
        print(f"  {k}: {v}")
    rules.to_csv("tables/tm_rules_physics.csv", index=False)


if __name__ == "__main__":
    main()