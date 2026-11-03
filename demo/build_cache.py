"""Build the demo's static data from eval outputs (run after eval/score.py + eval/make_report.py):

  demo/cache/feed.jsonl      held-out test posts + the chosen model's cached outputs (+ human gold label)
  demo/cache/dashboard.json  everything the Evaluation / Model tabs show (numbers come only from eval/results)
  demo/cache/examples.json   hand-written example posts + model outputs (cached fallback for the Analyze tab)

  python -I demo/build_cache.py <model_key> "<display label>" [--examples-url http://127.0.0.1:8082/v1]

Post text is HumAID (CC BY-NC-SA 4.0, Alam et al. 2021); attribution is shown in the app.
"""
import json, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.prompt import parse_output
from src.sft_format import ft_prompt
R = ROOT / "eval/results"; C = ROOT / "demo/cache"; C.mkdir(parents=True, exist_ok=True)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
NAME = args[0] if args else "ft2b_run5"
LABEL = args[1] if len(args) > 1 else "CrisisLens 2B"
opt = lambda k: next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == k), None)
EX_URL, RUN = opt("--examples-url"), opt("--run")  # --run <AutoScientist run id> of the served model

EXAMPLES = [  # hand-written, hypothetical posts (not from HumAID) to show the schema on varied cases
    "Water entered the ground floor, 3 families with small kids stuck on the terrace of Shanti Apartments, Aluva. Need boats urgently, phones dying. #KeralaFloods",
    "Bridge on NH-66 near Chalakudy has collapsed, all traffic diverted. Please avoid the route.",
    "Hospital in Chimanimani needs insulin and clean water, generator is running out of diesel.",
    "Missing: my father Joseph, 68, last seen near Beira port on Thursday before the cyclone hit. Please share.",
    "Death toll from the earthquake in Mexico City rises to 217, rescuers still searching collapsed buildings.",
    "Volunteers needed at the Ernakulam collection centre tomorrow 9am to pack relief kits, bring tape and markers.",
    "Evacuation shelters open at Marsh Harbour primary school and the community centre, bring ID and medicines.",
    "Thoughts and prayers to everyone in the Bahamas tonight.",
]

# ---------- live feed ----------
df = pd.read_parquet(ROOT / "data/processed/test_eval.parquet").set_index("id")
P = {}
for l in open(ROOT / f"eval/preds/{NAME}__test_eval.jsonl", encoding="utf-8"):
    j = json.loads(l)
    if j.get("raw") is not None: P[j["id"]] = j  # last real output wins
feed = []
for i, j in P.items():
    o, ok = parse_output(j["raw"])
    if not ok: continue
    r = df.loc[i]
    feed.append({"id": i, "event": r.event, "text": r.text, "gold": r.label, "pred": o, "latency_s": round(j.get("latency_s") or 0, 2)})
feed = pd.DataFrame(feed).sample(frac=1, random_state=1).head(600).to_dict("records")  # interleave events
with open(C / "feed.jsonl", "w", encoding="utf-8") as f:
    for r in feed: f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- dashboard ----------
def kind(k): return "ours" if k.startswith("ft") else ("api" if "@" in k else "baseline")
summ = pd.read_csv(R / "summary.csv")
systems = [{"key": r.key, "label": r.system, "kind": kind(r.key), "macro_f1": r.macro_f1, "ci": [r.ci_lo, r.ci_hi],
            "weighted_f1": r.weighted_f1, "recall_urgent": r.recall_urgent_needs, "recall_injured": r.recall_injured_dead,
            "json_valid": None if pd.isna(r.json_valid) else r.json_valid,
            "latency_s": None if pd.isna(r.latency_s) else r.latency_s,
            "usd_per_1k": None if pd.isna(r.usd_per_1k) else r.usd_per_1k, "n": int(r.n)} for r in summ.itertuples()]
best = json.load(open(R / f"{NAME}.json"))
pc_o = pd.read_csv(R / f"{NAME}_per_class.csv", index_col=0); pc_b = pd.read_csv(R / "tfidf_per_class.csv", index_col=0)
per_event = {}
for s in systems:
    f = R / f"{s['key']}.json"
    if f.exists() and "per_event_macro_f1" in json.load(open(f)):
        per_event[s["label"]] = json.load(open(f))["per_event_macro_f1"]
iters = pd.read_csv(R / "autoscientist_iterations.csv") if (R / "autoscientist_iterations.csv").exists() else pd.DataFrame()
stats = json.load(open(R / "data_stats.json")) if (R / "data_stats.json").exists() else {}
model = {"key": NAME, "label": LABEL, "run_id": RUN}
if RUN:
    A = ROOT / "eval/autoscientist"
    launch = json.load(open(A / f"{RUN}_launch.json"))
    final = [json.loads(l) for l in open(A / f"{RUN}.jsonl")][-1]
    model.update(base=launch["request"]["model"], train_rows=launch.get("rows"), best_win_rate=final.get("best_win_rate"),
                 hyperparams={k: (final.get("best_hyperparams") or {}).get(k) for k in
                              ["learning_rate", "n_epochs", "lora_r", "lora_alpha", "lora_dropout", "warmup_ratio", "weight_decay"]})
dash = {
    "generated": time.strftime("%Y-%m-%d %H:%M"),
    "model": model,
    "systems": systems,
    "per_class": {"classes": list(pc_o.index), "ours": [round(x, 4) for x in pc_o.f1], "baseline": [round(x, 4) for x in pc_b.reindex(pc_o.index).f1],
                  "support": [int(x) for x in pc_o.support], "baseline_label": "TF-IDF + LR"},
    "per_event": per_event,
    "silver": {k: best.get(k) for k in ["actionable_acc", "actionable_f1", "urgency_mae", "urgency_within1", "urgent_ge4_f1",
                                         "location_exact", "location_overlap", "needs_jaccard", "silver_n"]},
    "latency": {k: best.get(k) for k in ["latency_s_mean", "latency_s_p50", "latency_s_p95", "in_tok_mean", "out_tok_mean"]},
    "autoscientist": iters.to_dict("records"),
    "data": stats,
}
json.dump(dash, open(C / "dashboard.json", "w", encoding="utf-8"), indent=1, default=float)

# ---------- example outputs (cached fallback for Analyze) ----------
if EX_URL:
    from openai import OpenAI
    cli = OpenAI(base_url=EX_URL, api_key="none", timeout=120); ex = []
    for t in EXAMPLES:
        t0 = time.time()
        r = cli.chat.completions.create(model="x", temperature=0, max_tokens=200, messages=[{"role": "user", "content": ft_prompt(t)}])
        o, ok = parse_output(r.choices[0].message.content)
        ex.append({"text": t, "pred": o if ok else None, "latency_s": round(time.time() - t0, 2)})
    json.dump(ex, open(C / "examples.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("examples", sum(e["pred"] is not None for e in ex), "/", len(ex))
print(len(feed), "feed rows;", len(systems), "systems;", NAME)
