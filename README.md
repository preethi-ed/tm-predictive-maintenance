# Tsetlin Machines vs Classical Classifiers for Predictive Maintenance

Code and results for the preprint *"Tsetlin Machines versus Classical Classifiers for Predictive Maintenance: A Controlled Comparative Study on AI4I 2020"* (Preethi S R, 2026).

## What it does
Compares a Tsetlin Machine (TMU) with logistic regression, a decision tree, a random forest and an MLP on the AI4I 2020 dataset. Every model is evaluated under the same nested protocol: 5×3 repeated stratified cross-validation, with booleanisation thresholds, oversampling and decision thresholds fitted on inner splits only.

## Data
Download `ai4i2020.csv` from the UCI Machine Learning Repository (https://doi.org/10.24432/C5HS5C) and place it in `data/`.

## Setup
```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Reproduce the paper
| Step | Command | Output |
|---|---|---|
| Main comparison (Table 1, Table 2) | `python run_revision.py` | `tables/revision_folds_*.csv`, `tables/table_revision_*.csv` |
| TM ablation (Table 4) | `python run_ablation_tm.py` | `tables/ablation_nested_folds.csv`, `tables/table_ablation_nested.csv` |
| Figures 2 and 3 | `python make_paper_figures.py` | `figures/*.png` |

Package versions used are recorded in `tables/versions.txt`. Experiments were run on CPU.

## Citation
Preethi S R (2026). Tsetlin Machines versus Classical Classifiers for Predictive Maintenance: A Controlled Comparative Study on AI4I 2020. Preprint. DOI to be added.

## License
MIT