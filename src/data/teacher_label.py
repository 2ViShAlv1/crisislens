"""Silver labels for the extra fields (actionable, urgency, location, needs, rationale).

The teacher LLM (hosted on the Adaption Responses API) sees the post AND its human gold category,
and fills in the remaining fields consistently with that category. `category` is always the human
gold label; only the extra fields are silver.

Usage: python -I src/data/teacher_label.py <in.parquet> <out.jsonl> [--model qwen3.5-397b] [--limit N] [--workers 16]
Resumable: ids already in out.jsonl are skipped.
"""
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.prompt import CLASS_DEFS
from src.env import load_env

TEACHER_SYS = (
    "You annotate disaster-related social-media posts for humanitarian triage. You are given the post and its "
    "human-assigned category (trust it). Return ONLY a JSON object with keys:\n"
    '{"actionable": true|false, "urgency": 1-5, "location": string|null, "needs": [string], "rationale": string}\n'
    "Rules:\n"
    "- actionable: true only if a responder could act on it (specific need, specific location of damage/people, "
    "trapped/injured/missing people, a concrete hazard warning, an evacuation order). Prayers, opinions, generic news "
    "and donation appeals for big organisations are NOT actionable.\n"
    "- urgency: 1 no response value; 2 general info/support; 3 useful situational info; 4 time-sensitive need or hazard; "
    "5 life-threatening (trapped, injured, missing, immediate rescue/medical needed).\n"
    "- location: the most specific place name written in the post (city/district/region/country), copied as written; "
    "null if none. Do not infer places that are not written.\n"
    "- needs: short lowercase nouns from this list when stated or clearly implied: "
    "rescue, medical, food, water, shelter, evacuation, power, communication, transport, sanitation, clothing, money, "
    "volunteers, information, missing-person-search. [] if none.\n"
    "- rationale: <= 20 words explaining the category and urgency.\n"
    "Category definitions: " + "; ".join(f"{k} = {v}" for k, v in CLASS_DEFS.items())
)

def label_one(client, model, row):
    msg = f"Category: {row['label']}\nPost: {row['text']}"
    for attempt in range(4):
        try:
            r = client.responses.create(model=model, instructions=TEACHER_SYS, input=msg,
                                        temperature=0, max_output_tokens=400)
            txt = r.output_text
            s = txt[txt.find("{"): txt.rfind("}") + 1]
            o = json.loads(s)
            rec = {"category": row["label"],
                   "actionable": bool(o["actionable"]),
                   "urgency": int(min(5, max(1, int(o["urgency"])))),
                   "location": (o.get("location") or None),
                   "needs": [str(x).lower() for x in (o.get("needs") or [])][:6],
                   "rationale": " ".join(str(o.get("rationale", "")).split()[:20])}
            u = getattr(r, "usage", None)
            return {"id": row["id"], "label": rec, "teacher": model,
                    "in_tok": getattr(u, "input_tokens", None), "out_tok": getattr(u, "output_tokens", None)}
        except Exception as e:  # noqa
            err = repr(e)[:200]; time.sleep(2 * (attempt + 1))
    return {"id": row["id"], "error": err}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--model", default="qwen3.5-397b"); ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    load_env()
    from adaption import Adaption
    client = Adaption()
    df = pd.read_parquet(a.inp)
    if a.limit: df = df.head(a.limit)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            j = json.loads(l)
            if "label" in j: done.add(j["id"])
    todo = df[~df.id.isin(done)].to_dict("records")
    print(f"{len(done)} done, {len(todo)} to label with {a.model}", flush=True)
    lock = threading.Lock(); n_err = 0; t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(label_one, client, a.model, r) for r in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            res = fu.result()
            with lock:
                if "error" in res: n_err += 1
                else: f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
            if i % 200 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} errors={n_err} {i / (time.time() - t0):.1f}/s", flush=True)

if __name__ == "__main__":
    main()
