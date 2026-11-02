"""Build demo/cache/feed.jsonl (held-out test posts + the fine-tuned model's cached outputs) and
demo/cache/metrics.json (results table markdown). Run after eval/score.py + eval/make_report.py.
Post text is HumAID (CC BY-NC-SA 4.0, Alam et al. 2021) — attribution shown in the app."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.prompt import parse_output
NAME = sys.argv[1] if len(sys.argv) > 1 else "ft2b_run5"
df = pd.read_parquet(ROOT / "data/processed/test_eval.parquet").set_index("id")
out = []
for l in open(ROOT / f"eval/preds/{NAME}__test_eval.jsonl", encoding="utf-8"):
    j = json.loads(l); o, ok = parse_output(j.get("raw"))
    if not ok: continue
    r = df.loc[j["id"]]
    out.append({"id": j["id"], "event": r.event, "text": r.text, "gold": r.label, "pred": o, "latency_s": j.get("latency_s")})
# interleave events so the feed looks like a live multi-disaster stream; keep 600 for size
out = pd.DataFrame(out).sample(frac=1, random_state=1).head(600).to_dict("records")
C = ROOT / "demo/cache"; C.mkdir(parents=True, exist_ok=True)
with open(C / "feed.jsonl", "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
md = (ROOT / "eval/results/summary.md").read_text(encoding="utf-8") if (ROOT / "eval/results/summary.md").exists() else ""
json.dump({"table_md": "### Results on 4 held-out disasters (n=1,219)\n\n" + md}, open(C / "metrics.json", "w"), indent=2)
print(len(out), "feed rows")
