"""Generate Paper A figures directly from the study's saved results."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TABLES, FIGS = Path("tables"), Path("figures")
FIGS.mkdir(exist_ok=True)
TM_DEPLOYABLE_KB = 5.0   # upper bound from the size analysis
INK, ACCENT = "#333333", "#1f5fa8"
plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

def find_col(df, *keys):
    for c in df.columns:
        if all(k in c.lower() for k in keys):
            return c
    sys.exit(f"Could not find a column with {keys} in {list(df.columns)}")

def load(name):
    for p in (TABLES / name, Path(name), Path("results") / name):
        if p.exists():
            return pd.read_csv(p)
    sys.exit(f"{name} not found in tables/, results/ or the current folder")

# ---------- Accuracy vs model size ----------
rev = load("revision_folds_physics.csv")
mcol, fcol = find_col(rev, "model"), find_col(rev, "f1")
scol = next((c for c in rev.columns if any(k in c.lower() for k in ("pickle", "size", "kb"))), None)
if scol is None:
    sys.exit(f"No size column found in {list(rev.columns)}")
summ = rev.groupby(mcol).agg(f1=(fcol, "mean"), sd=(fcol, "std"), kb=(scol, "mean")).reset_index()
print(summ.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.2, 3.8))
tm = None
for _, r in summ.iterrows():
    is_tm = "tsetlin" in str(r[mcol]).lower() or str(r[mcol]).upper() == "TM"
    ax.errorbar(r.kb, r.f1, yerr=r.sd, fmt="o", ms=8, color=ACCENT if is_tm else INK,
                ecolor="#999999", elinewidth=1.2, capsize=3, zorder=3)
    ax.annotate(f"{r[mcol]} (serialised)" if is_tm else str(r[mcol]), (r.kb, r.f1),
                xytext=(10, -12 if is_tm else 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)
    if is_tm:
        tm = r
if tm is not None:
    ax.errorbar(TM_DEPLOYABLE_KB, tm.f1, yerr=tm.sd, fmt="o", ms=8, mfc="white",
                mec=ACCENT, mew=1.6, ecolor="#bbbbbb", capsize=3, zorder=3)
    ax.plot([TM_DEPLOYABLE_KB, tm.kb], [tm.f1] * 2, ls=":", color=ACCENT, lw=1)
    ax.annotate("Tsetlin Machine\n(deployable estimate, <5 KB)", (TM_DEPLOYABLE_KB, tm.f1),
                xytext=(10, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
ax.set_xscale("log")
ax.set_xlabel("Model size (KB, log scale)")
ax.set_ylabel("F1 (failure class)")
ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
fig.savefig(FIGS / "fig_accuracy_size.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Ablation per fold ----------
abl = load("ablation_nested_folds.csv")
vcol, fcol, kcol = find_col(abl, "variant"), find_col(abl, "f1"), find_col(abl, "fold")
order = ["quantile, no cap", "supervised, no cap", "supervised + cap 10"]
wide = abl.pivot(index=kcol, columns=vcol, values=fcol)[order]
print(wide.mean().round(3).to_string())

fig, ax = plt.subplots(figsize=(5.2, 3.6))
x = range(len(order))
for _, row in wide.iterrows():
    ax.plot(x, row.values, color="#cccccc", lw=1, zorder=1)
for xi, v in zip(x, order):
    ax.scatter([xi] * len(wide), wide[v], s=22, color="#888888", zorder=2)
    m = wide[v].mean()
    ax.plot([xi - 0.18, xi + 0.18], [m, m], color=ACCENT, lw=2.5, zorder=3)
    ax.annotate(f"mean {m:.3f}", (xi + 0.22, m), va="center", fontsize=9, color=INK,
                zorder=4, bbox=dict(fc="white", ec="none", pad=1))
ax.set_xticks(list(x), ["Quantile\nno cap", "Supervised\nno cap", "Supervised\ncap 10"])
ax.set_xlim(-0.4, 2.8)
ax.set_ylabel("F1 (failure class)")
ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
fig.savefig(FIGS / "fig_ablation_folds.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved figures/fig_accuracy_size.png and figures/fig_ablation_folds.png")