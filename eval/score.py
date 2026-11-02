"""Score LLM prediction files against gold categories and silver extra fields.

  python -I eval/score.py <name> [<name> ...]   (reads eval/preds/<name>__test_eval.jsonl)

Category metrics use HUMAN gold labels. actionable/urgency/location/needs metrics use SILVER
labels from the teacher LLM (gemma-4-31b-it) and are reported separately.
Cost per 1k posts uses token counts actually logged and the price table in eval/prices.json.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.prompt import parse_output, category_of
from src.eval.metrics import classification_report, INVALID
from sklearn.metrics import f1_score

R = ROOT / "eval/results"; P = ROOT / "eval/preds"

def norm_loc(x):
    return (x or "").lower().replace("#", "").strip() or None

def silver_metrics(pred_objs, silver):
    rows = [(p, s) for p, s in zip(pred_objs, silver) if s is not None]
    ok = [(p, s) for p, s in rows if isinstance(p, dict)]
    if not ok: return {}
    out = {"silver_n": len(rows), "silver_n_parsed": len(ok)}
    ya = [bool(s["actionable"]) for p, s in ok]; pa = [bool(p.get("actionable")) for p, s in ok]
    out["actionable_acc"] = float(np.mean(np.array(ya) == np.array(pa)))
    out["actionable_f1"] = float(f1_score(ya, pa, zero_division=0))
    def u(p):
        try: return int(p.get("urgency"))
        except Exception: return 3
    du = np.array([abs(u(p) - int(s["urgency"])) for p, s in ok])
    out["urgency_mae"] = float(du.mean()); out["urgency_within1"] = float((du <= 1).mean())
    # urgent (>=4) detection
    yu = [int(s["urgency"]) >= 4 for p, s in ok]; pu = [u(p) >= 4 for p, s in ok]
    out["urgent_ge4_f1"] = float(f1_score(yu, pu, zero_division=0))
    L = [(norm_loc(p.get("location") if isinstance(p.get("location"), str) else None), norm_loc(s["location"])) for p, s in ok]
    out["location_exact"] = float(np.mean([a == b for a, b in L]))
    out["location_overlap"] = float(np.mean([bool((a == b) or (a and b and (a in b or b in a))) for a, b in L]))
    def jac(a, b):
        a, b = set(map(str.lower, a or [])), set(map(str.lower, b or []))
        return 1.0 if not a and not b else len(a & b) / len(a | b)
    out["needs_jaccard"] = float(np.mean([jac(p.get("needs") if isinstance(p.get("needs"), list) else [], s["needs"]) for p, s in ok]))
    return out

def load_silver(split="test_eval"):
    S = {}
    f = ROOT / f"data/processed/teacher/{split}.jsonl"
    if f.exists():
        for l in open(f, encoding="utf-8"):
            j = json.loads(l)
            if "label" in j: S[j["id"]] = j["label"]
    return S

def score(name, split="test_eval", restrict=None, tag=""):
    df = pd.read_parquet(ROOT / f"data/processed/{split}.parquet")
    if restrict is not None:
        df = df[df.id.isin(restrict)].reset_index(drop=True)
    preds = {}
    for l in open(P / f"{name}__{split}.jsonl", encoding="utf-8"):
        j = json.loads(l); preds[j["id"]] = j  # last write wins
    missing = int((~df.id.isin(preds)).sum())
    raws = [preds.get(i, {}).get("raw") for i in df.id]
    parsed = [parse_output(r) for r in raws]
    objs = [o for o, v in parsed]
    cats = [category_of(r) if r else INVALID for r in raws]
    rep, per, cm = classification_report(df.label.values, np.array(cats))
    rep["json_valid_rate"] = float(np.mean([v for o, v in parsed]))
    rep["json_parse_rate"] = float(np.mean([isinstance(o, dict) for o in objs]))
    rep["missing_preds"] = missing
    lat = [preds[i]["latency_s"] for i in df.id if i in preds and preds[i].get("latency_s")]
    rep["latency_s_mean"] = float(np.mean(lat)) if lat else None
    rep["latency_s_p50"] = float(np.median(lat)) if lat else None
    rep["latency_s_p95"] = float(np.percentile(lat, 95)) if lat else None
    it = [preds[i].get("in_tok") or 0 for i in df.id if i in preds]; ot = [preds[i].get("out_tok") or 0 for i in df.id if i in preds]
    rep["in_tok_mean"] = float(np.mean(it)) if it else None; rep["out_tok_mean"] = float(np.mean(ot)) if ot else None
    prices = json.load(open(ROOT / "eval/prices.json"))
    base = name.split("@")[0]
    if base in prices and it:
        pr = prices[base]
        rep["usd_per_1k_posts"] = float(1000 * (np.mean(it) * pr["in"] + np.mean(ot) * pr["out"]) / 1e6)
        rep["price_source"] = pr.get("source")
    S = load_silver(split)
    rep.update(silver_metrics(objs, [S.get(i) for i in df.id]))
    # per-event macro-F1
    rep["per_event_macro_f1"] = {ev: classification_report(g.label.values, np.array(cats)[g.index.values], n_boot=1)[0]["macro_f1"]
                                 for ev, g in df.reset_index(drop=True).groupby("event")}
    per.to_csv(R / f"{name}{tag}_per_class.csv"); cm.to_csv(R / f"{name}{tag}_confusion.csv")
    json.dump(rep, open(R / f"{name}{tag}.json", "w"), indent=2)
    return rep

def ids_of(name, split="test_eval"):
    return {json.loads(l)["id"] for l in open(P / f"{name}__{split}.jsonl", encoding="utf-8") if json.loads(l).get("raw")}

if __name__ == "__main__":
    # usage: score.py name ...            -> full eval subset
    #        score.py --common a,b,c      -> each of a,b,c scored on the ids ALL of them answered (tag "__common")
    if sys.argv[1] == "--common":
        names = sys.argv[2].split(","); common = set.intersection(*[ids_of(n) for n in names]); print("common ids:", len(common))
        for n in names:
            r = score(n, restrict=common, tag="__common"); print(n, round(r["macro_f1"], 4), round(r["weighted_f1"], 4))
        sys.exit()
    for n in sys.argv[1:]:
        r = score(n)
        print(n, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "per_event_macro_f1"}))
