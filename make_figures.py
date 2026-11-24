import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR, TAB_DIR = "figures", "tables"
os.makedirs(FIG_DIR, exist_ok=True)

RESULTS = [
    # model, features, F1, F1 std, PR-AUC, size in KB
    ("Logistic regression", "raw", 0.234, None, None, None),
    ("MLP", "raw", 0.389, None, None, None),
    ("Decision tree", "raw", 0.452, None, 0.495, 5.7),
    ("Random forest", "raw", 0.596, None, 0.775, 7300),
    ("Tsetlin Machine", "raw", 0.567, 0.036, 0.609, 22.9),
    ("Logistic regression", "physics", 0.276, None, None, None),
    ("MLP", "physics", 0.347, 0.222, None, None),
    ("Decision tree", "physics", 0.533, None, 0.819, 5.7),
    ("Random forest", "physics", 0.864, None, 0.888, 4400),
    ("Tsetlin Machine", "physics", 0.772, 0.057, 0.808, 26.7),
]
COLORS = {"Tsetlin Machine": "#534AB7", "Random forest": "#0F6E56", "Decision tree": "#BA7517",
          "MLP": "#888780", "Logistic regression": "#B4B2A9"}


def fig_f1_vs_size(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, r in df.dropna(subset=["size_kb"]).iterrows():
        filled = r.features == "physics"
        ax.scatter(r.size_kb, r.f1, s=90, color=COLORS[r.model] if filled else "white",
                   edgecolor=COLORS[r.model], linewidth=2, zorder=3)
        ax.annotate(f"{r.model}\n({r.features})", (r.size_kb, r.f1), textcoords="offset points",
                    xytext=(8, -4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Model size (KB, log scale)")
    ax.set_ylabel("F1 score (failure class)")
    ax.set_title("Accuracy vs model size (filled = physics features, open = raw)")
    ax.grid(alpha=0.3)
    ax.set_xlim(2, 30000)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_f1_vs_size.png"), dpi=300)


def fig_recall_by_mode():
    path = os.path.join(TAB_DIR, "table_recall_by_mode.csv")
    t = pd.read_csv(path)
    t = t[t["Failure mode"] != "RNF"]
    models = ["Tsetlin Machine", "Decision tree", "Random forest"]
    x = range(len(t))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k, m in enumerate(models):
        ax.bar([i + (k - 1) * w for i in x], t[m], width=w, label=m, color=COLORS[m])
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{fm}\n(n={c})" for fm, c in zip(t["Failure mode"], t["Count"])])
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recall by failure mode (5-fold out-of-fold, physics features)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_recall_by_mode.png"), dpi=300)


def fig_ablation():
    labels = ["F1", "PR-AUC"]
    quant, sup = [0.655, 0.717], [0.772, 0.808]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
    x = range(len(labels))
    axes[0].bar([i - 0.18 for i in x], quant, width=0.36, label="Quantile thresholds", color="#B4B2A9")
    axes[0].bar([i + 0.18 for i in x], sup, width=0.36, label="Supervised thresholds + literal cap", color="#534AB7")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=7)
    axes[0].set_title("Accuracy")
    axes[1].bar(["Quantile", "Supervised"], [36.0, 26.7], color=["#B4B2A9", "#534AB7"])
    axes[1].set_ylabel("Model size (KB)")
    axes[1].set_title("Size")
    fig.suptitle("Tsetlin Machine ablation (physics features, 5×3 CV)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_ablation.png"), dpi=300)


def latex_results(df):
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Model & Features & F1 & PR-AUC & Size (KB) \\", r"\midrule"]
    for _, r in df.iterrows():
        f1 = f"{r.f1:.3f}" + (f" $\\pm$ {r.f1_std:.3f}" if pd.notna(r.f1_std) else "")
        pr = f"{r.pr_auc:.3f}" if pd.notna(r.pr_auc) else "--"
        sz = f"{r.size_kb:,.1f}" if pd.notna(r.size_kb) else "--"
        name = r"\textbf{" + r.model + "}" if r.model == "Tsetlin Machine" else r.model
        lines.append(f"{name} & {r.features} & {f1} & {pr} & {sz} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB_DIR, "tab1_results.tex"), "w") as f:
        f.write("\n".join(lines))


def latex_rules():
    rows = [
        ("HDF", r"$\Delta T < 8.6$ K $\wedge$ rpm $< 1380$", r"$\Delta T \le 8.6$ $\wedge$ rpm $\le 1380$",
         r"$\Delta T \le 9.15$ $\wedge$ rpm $\le 1422$"),
        ("PWF (low)", r"$P < 3500$ W", r"$P \le 3496$", "-- (lower-ranked clauses)"),
        ("PWF (high)", r"$P > 9000$ W", r"$P > 9005$", r"$P > 8529$ $\wedge$ $\tau > 60.65$"),
        ("OSF", r"$w\tau > 11000$--$13000$", r"$w\tau > 10566$ / $11014$", r"$w\tau > 10470$ $\wedge$ $\neg$Type H$^\dagger$"),
        ("TWF", r"$200 \le w \le 240$ min", r"$w > 202.5$", r"$w > 202.5$"),
    ]
    lines = [r"\begin{tabular}{llll}", r"\toprule",
             r"Mode & Ground truth & Decision tree & Tsetlin Machine \\", r"\midrule"]
    lines += [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB_DIR, "tab2_rules.tex"), "w") as f:
        f.write("\n".join(lines))


def main():
    df = pd.DataFrame(RESULTS, columns=["model", "features", "f1", "f1_std", "pr_auc", "size_kb"])
    fig_f1_vs_size(df)
    fig_recall_by_mode()
    fig_ablation()
    latex_results(df)
    latex_rules()
    print("Saved figures/fig1_f1_vs_size.png, fig2_recall_by_mode.png, fig3_ablation.png")
    print("Saved tables/tab1_results.tex, tab2_rules.tex")


if __name__ == "__main__":
    main()