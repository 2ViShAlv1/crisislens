"""Pick 15 failure cases of a system for manual explanation (deterministic):
 - up to 6 misses on life-critical gold classes (requests_or_urgent_needs, injured_or_dead_people)
 - then the most frequent confusion pairs, 1-2 examples each, until 15.
Writes eval/results/<name>_errors.csv (id, event, gold, pred, text, model output). Explanations are added in
docs/ERROR_ANALYSIS.md by reading these cases."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.prompt import category_of, parse_output
from src.eval.metrics import LIFE_CRITICAL
name = sys.argv[1] if len(sys.argv) > 1 else "ft2b_run5"
df = pd.read_parquet(ROOT / "data/processed/test_eval.parquet")
P = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / f"eval/preds/{name}__test_eval.jsonl", encoding="utf-8")}
df = df[df.id.isin(P)].copy()
df["raw"] = df.id.map(lambda i: P[i].get("raw")); df["pred"] = df.raw.map(category_of)
err = df[df.pred != df.label].sample(frac=1, random_state=3)
pick = [err[err.label == c].head(3) for c in LIFE_CRITICAL]
rest = err.drop(pd.concat(pick).index)
pairs = rest.groupby(["label", "pred"]).size().sort_values(ascending=False)
for (g, p), _ in pairs.items():
    if sum(len(x) for x in pick) >= 15: break
    pick.append(rest[(rest.label == g) & (rest.pred == p)].head(1 if sum(len(x) for x in pick) > 11 else 2))
out = pd.concat(pick).head(15)[["id", "event", "label", "pred", "text", "raw"]]
out.to_csv(ROOT / f"eval/results/{name}_errors.csv", index=False, encoding="utf-8")
print("top confusion pairs (gold -> pred):"); print(err.groupby(["label", "pred"]).size().sort_values(ascending=False).head(8).to_string())
sys.stdout.reconfigure(encoding="utf-8")
for r in out.itertuples():
    print(f"\n[{r.event}] gold={r.label} pred={r.pred}\n  {r.text[:260]}\n  -> {str(r.raw)[:200]}")
