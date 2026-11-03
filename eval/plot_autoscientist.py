"""Training-run evidence: AutoScientist win-rate per completed iteration (from polled run state) and
training-loss curves (from the downloaded best checkpoint's trainer_state.json). Writes
eval/results/autoscientist_iterations.csv and demo/assets/3_autoscientist_*.png."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; L = ROOT / "eval/autoscientist"; A = ROOT / "demo/assets"; A.mkdir(parents=True, exist_ok=True)
RUNS = {  # run id -> label (fixed order = fixed colour)
    "99464db5-372f-42ad-ae49-c0163fc1a588": "#1 2B · 8k · v1 prompt · default",
    "2c060ec8-cc23-40e5-9e8e-57bbf0d37a6a": "#2 2B · 17.8k · v1 · default",
    "165c0bf4-8f62-4354-9e41-b6023431e211": "#3 9B · 17.8k · v1 · default",
    "25d04b86-af94-45cf-be37-5acc5822614c": "#4 2B · 17.8k · v1 · target 0.95",
    "7ebe8824-9857-4552-8574-8b58b3d55be5": "#5 2B · 17.8k · v2 prompt · lr/epoch override",
    "fafc81b1-8876-43c1-80e9-75a1791d676c": "#6 4B · 17.8k · v2 · run #5 recipe",
    "2b634ded-e0b7-4a21-9467-bf4025b0bcc1": "#7 2B · 37k (cap 5k) · v2 · run #5 recipe",
    "fab5f195-3c03-4f96-ac63-fb31f6693054": "#8 2B · 49k (all) · v2 · 2 epochs",
    "e70d8d4e-af9a-4885-9681-95352a211b2e": "#9 2B · 17.8k + augmentation (2.2k domain, 8k general) · v2",
}
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # dataviz slots 1-8, fixed order
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#d6d5d0",
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
rows = []
for rid, lab in RUNS.items():
    f = L / f"{rid}.jsonl"
    if not f.exists(): continue
    seen = set()
    for l in open(f):
        r = json.loads(l); it = int(r["iterations_completed"])
        if it and it not in seen and r.get("best_win_rate") is not None:
            seen.add(it); hp = r.get("best_hyperparams") or {}
            rows.append({"run": lab, "run_id": rid, "iteration": it, "best_win_rate": r["best_win_rate"], "status": r["status"],
                         "learning_rate": hp.get("learning_rate"), "n_epochs": hp.get("n_epochs"), "lora_r": hp.get("lora_r"),
                         "lora_alpha": hp.get("lora_alpha"), "warmup_ratio": hp.get("warmup_ratio"), "target": r["target_win_rate"]})
df = pd.DataFrame(rows); df.to_csv(ROOT / "eval/results/autoscientist_iterations.csv", index=False); print(df.to_string())

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for i, (rid, lab) in enumerate(RUNS.items()):
    g = df[df.run_id == rid]
    if g.empty: continue
    ax.plot(g.iteration, g.best_win_rate, "-o", color=PAL[i], lw=2, ms=6,
            label=f"{lab}  ->  {g.best_win_rate.iloc[-1]:.2f}" + ("" if g.status.iloc[-1] == "succeeded" else " (running)"))
ax.axhline(0.5, color="#b9b8b2", lw=1, ls="--"); ax.text(1, 0.505, "parity with base model", fontsize=7, color=INK2)
ax.set_xlabel("AutoScientist iteration"); ax.set_ylabel("best win-rate vs base (platform judge)")
ax.set_xticks(sorted(df.iteration.unique())); ax.set_ylim(0.45, 1.0)
ax.set_title("AutoScientist: win-rate across iterations", loc="left", color=INK, fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=1); ax.grid(axis="y", color="#ecebe7")
fig.tight_layout(); fig.savefig(A / "3_autoscientist_iterations.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.4))
for i, (rid, lab) in enumerate(RUNS.items()):
    f = L / f"{rid}_trainer_state.json"
    if not f.exists(): continue
    lh = json.load(open(f))["log_history"]
    tr = [(h["epoch"], h["loss"]) for h in lh if "loss" in h]
    ax.plot([e for e, _ in tr], [v for _, v in tr], color=PAL[i], lw=2, label=lab + " (best iteration)")
ax.set_xlabel("epoch"); ax.set_ylabel("training loss"); ax.set_title("Training loss of each run's best checkpoint", loc="left", color=INK, fontsize=10)
ax.legend(frameon=False, fontsize=7); ax.grid(axis="y", color="#ecebe7")
fig.tight_layout(); fig.savefig(A / "4_autoscientist_loss.png", dpi=150); plt.close(fig)
