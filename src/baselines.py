"""Baseline models and the evaluation protocol for Paper A.

Protocol: repeated stratified 5-fold cross-validation (3 repeats = 15 folds).
Scaling is fitted inside each training fold only (Pipeline), so no test
information leaks into training. Failures are rare (~3%), so we report
F1 and PR-AUC for the failure class, not just accuracy.
"""
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42


def make_models():
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
        "Decision tree (depth 5)": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=SEED),
        "Random forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "MLP (32-16)": make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000,
                                            early_stopping=True, random_state=SEED)),
    }


def model_size_kb(model):
    return len(pickle.dumps(model)) / 1024


def evaluate(model_factory, X, y, n_splits=5, n_repeats=3):
    """Return one row of metrics per fold for a model built by `model_factory()`."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        model = model_factory()
        model.fit(X.iloc[tr], y.iloc[tr])

        t0 = time.perf_counter()
        pred = model.predict(X.iloc[te])
        infer_ms = 1000 * (time.perf_counter() - t0) / len(te)

        score = model.predict_proba(X.iloc[te])[:, 1]
        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(y.iloc[te], pred),
            "balanced_acc": balanced_accuracy_score(y.iloc[te], pred),
            "f1_failure": f1_score(y.iloc[te], pred),
            "pr_auc": average_precision_score(y.iloc[te], score),
            "size_kb": model_size_kb(model),
            "infer_ms_per_sample": infer_ms,
        })
    return pd.DataFrame(rows)


def summarise(per_fold):
    """mean ± std per model."""
    metrics = ["accuracy", "balanced_acc", "f1_failure", "pr_auc", "size_kb", "infer_ms_per_sample"]
    g = per_fold.groupby("model")[metrics]
    mean, std = g.mean(), g.std()
    out = pd.DataFrame(index=mean.index)
    for m in metrics:
        out[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(mean[m], std[m])]
    return out