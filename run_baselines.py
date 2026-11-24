"""Paper A, step 1: baseline models on AI4I 2020.
Put ai4i2020.csv in data/, then run from this folder:  py -m run_baselines
"""
import os

import pandas as pd

from src.baselines import evaluate, make_models, summarise
from src.data import load_ai4i

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    PHYSICS = False
    X, y = load_ai4i(physics=PHYSICS)
    tag = "physics" if PHYSICS else "raw"
    print(f"Samples: {len(y)}, features: {X.shape[1]}, failure rate: {y.mean():.2%}")

    names = list(make_models())
    all_folds = []
    for name in names:
        per_fold = evaluate(lambda n=name: make_models()[n], X, y)
        per_fold["model"] = name
        all_folds.append(per_fold)
        print(f"  done: {name}")

    per_fold = pd.concat(all_folds, ignore_index=True)
    per_fold.to_csv("results/baselines_per_fold.csv", index=False)

    table = summarise(per_fold).loc[names]
    table.to_csv(f"results/table_baselines_{tag}.csv")
    print("\n" + table.to_string())