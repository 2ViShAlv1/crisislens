"""Collect every results JSON into one table + charts (PNG for README/demo).

  python -I eval/make_report.py <system_name>=<label> ...
e.g. tfidf=TF-IDF+LR qwen9b_zeroshot="Qwen3.5-9B zero-shot" ...
Systems whose JSON is missing are skipped (and listed). Numbers come only from eval/results/*.json.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "eval/results"; A = ROOT / "demo/assets"; A.mkdir(parents=True, exist_ok=True)

# reference palette (dataviz skill, light mode); text in ink tokens, never series colour
S1, S2, MUTED = "#2a78d6", "#eb6834", "#b9b8b2"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})

def load(name):
    if name == "tfidf":
        j = json.load(open(R / "baseline_tfidf.json"))["test_eval"]; return j
    f = R / f"{name}.json"
    if not f.exists():  # partial runs (e.g. base zero-shot killed at 1,035/1,219) are only scored on the common subset
        f = R / f"{name}__common.json"
    return json.load(open(f)) if f.exists() else None

def main(pairs):
    rows, skipped = [], []
    for p in pairs:
        name, label = p.split("=", 1)
        j = load(name)
        if j is None: skipped.append(name); continue
        rows.append({"system": label, "key": name, "macro_f1": j["macro_f1"], "ci_lo": j["macro_f1_ci95"][0],
                     "ci_hi": j["macro_f1_ci95"][1], "weighted_f1": j["weighted_f1"],
                     "recall_urgent_needs": j["recall_requests_or_urgent_needs"],
                     "recall_injured_dead": j["recall_injured_or_dead_people"],
                     "json_valid": j.get("json_valid_rate"),
                     "latency_s": j.get("latency_s_mean", (j["latency_ms_per_post"] / 1000) if j.get("latency_ms_per_post") else None),
                     "usd_per_1k": j.get("usd_per_1k_posts"), "n": j["n"]})
    df = pd.DataFrame(rows); df.to_csv(R / "summary.csv", index=False)
    fmt = df.copy()
    for c in ["macro_f1", "weighted_f1", "recall_urgent_needs", "recall_injured_dead", "json_valid"]:
        fmt[c] = fmt[c].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
    fmt["macro_f1"] = [f"{m} [{lo:.3f}, {hi:.3f}]" for m, lo, hi in zip(fmt.macro_f1, df.ci_lo, df.ci_hi)]
    fmt["latency_s"] = df.latency_s.map(lambda v: "—" if pd.isna(v) else (f"{v * 1000:.2f} ms" if v < 0.01 else f"{v:.2f} s"))
    fmt = fmt.rename(columns={"latency_s": "latency/post"})
    fmt["usd_per_1k"] = df.usd_per_1k.map(lambda v: "—" if pd.isna(v) else f"${v:.3f}")
    md = fmt.drop(columns=["key", "ci_lo", "ci_hi"]).to_markdown(index=False)
    (R / "summary.md").write_text(md, encoding="utf-8"); print(md); print("skipped:", skipped)

    # 1) macro-F1 with 95% CI — highlight the fine-tuned system(s)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(df) + 1.2))
    col = [S1 if "ft" in k or "autosci" in k else MUTED for k in df.key]
    y = range(len(df))
    ax.barh(y, df.macro_f1, color=col, height=0.55)
    ax.errorbar(df.macro_f1, y, xerr=[df.macro_f1 - df.ci_lo, df.ci_hi - df.macro_f1], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    for i, v in enumerate(df.macro_f1): ax.text(v + 0.01, i, f"{v:.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(list(y), df.system); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Macro-F1 (gold labels), 4 held-out disasters, n=%d" % df.n.iloc[0])
    ax.set_title("Category macro-F1 with 95% bootstrap CI", loc="left", color=INK); ax.grid(axis="x", color="#ecebe7")
    fig.tight_layout(); fig.savefig(A / "1_macro_f1.png", dpi=150); plt.close(fig)

    # 2) life-critical recall (2 series)
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(df) + 1.4)); h = 0.36
    yy = list(range(len(df)))
    ax.barh([i - h / 2 for i in yy], df.recall_urgent_needs, height=h, color=S1, label="requests_or_urgent_needs")
    ax.barh([i + h / 2 for i in yy], df.recall_injured_dead, height=h, color=S2, label="injured_or_dead_people")
    for i, (a, b) in enumerate(zip(df.recall_urgent_needs, df.recall_injured_dead)):
        ax.text(a + 0.01, i - h / 2, f"{a:.2f}", va="center", fontsize=8, color=INK); ax.text(b + 0.01, i + h / 2, f"{b:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(yy, df.system); ax.invert_yaxis(); ax.set_xlim(0, 1.08); ax.set_xlabel("Recall")
    ax.set_title("Recall on life-critical classes", loc="left", color=INK); ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.grid(axis="x", color="#ecebe7"); fig.tight_layout(); fig.savefig(A / "2_life_critical_recall.png", dpi=150); plt.close(fig)

def per_class_chart(a, b, la, lb):
    """per-class F1 of two systems (2 series)"""
    fa, fb = R / f"{a}_per_class.csv", R / f"{b}_per_class.csv"
    if not (fa.exists() and fb.exists()): return
    A_, B_ = pd.read_csv(fa, index_col=0), pd.read_csv(fb, index_col=0)
    cls = list(A_.index); y = range(len(cls)); h = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh([i - h / 2 for i in y], A_.f1, height=h, color=S1, label=la)
    ax.barh([i + h / 2 for i in y], B_.f1, height=h, color=S2, label=lb)
    for i, (u, v) in enumerate(zip(A_.f1, B_.f1)):
        ax.text(u + 0.01, i - h / 2, f"{u:.2f}", va="center", fontsize=7.5, color=INK); ax.text(v + 0.01, i + h / 2, f"{v:.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(list(y), [f"{c}  (n={int(n)})" for c, n in zip(cls, A_.support)], fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1 (gold labels)"); ax.set_title("Per-class F1 on 4 held-out disasters", loc="left", color=INK)
    ax.legend(frameon=False, loc="lower right", fontsize=8); ax.grid(axis="x", color="#ecebe7")
    fig.tight_layout(); fig.savefig(A / "5_per_class_f1.png", dpi=150); plt.close(fig)

if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--perclass=")]
    main(args)
    pc = [x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--perclass=")]
    if pc:
        a, b = pc[0].split(","); per_class_chart(a, b, "CrisisLens (fine-tuned)", "TF-IDF + LR")
