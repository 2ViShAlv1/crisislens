"""Join teacher silver fields with gold categories -> SFT file for AutoScientist (raw mode).
Columns: prompt (compact CrisisLens instruction + post), completion (canonical JSON record).
Rows without a teacher label are dropped and counted."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.prompt import to_completion, validate
from src.sft_format import ft_prompt

def load_teacher(path):
    T = {}
    for l in open(path, encoding="utf-8"):
        j = json.loads(l)
        if "label" in j: T[j["id"]] = j["label"]
    return T

def main(split="sft_select", out_name="sft_train", version="v2"):
    D = ROOT / "data/processed"
    df = pd.read_parquet(D / f"{split}.parquet")
    T = load_teacher(D / f"teacher/{split}.jsonl")
    rows, bad = [], 0
    for r in df.itertuples():
        if r.id not in T: continue
        rec = dict(T[r.id]); rec["category"] = r.label  # gold category always wins
        if not validate(rec): bad += 1; continue
        rows.append({"id": r.id, "event": r.event, "label": r.label, "prompt": ft_prompt(r.text, version), "completion": to_completion(rec)})
    out = pd.DataFrame(rows)
    out[["prompt", "completion"]].to_csv(D / f"{out_name}.csv", index=False)
    out.to_parquet(D / f"{out_name}_full.parquet", index=False)
    print(f"{split}: {len(df)} rows, {len(T)} teacher-labelled, {bad} failed schema, wrote {len(out)}")
    print(out.label.value_counts().to_string())

if __name__ == "__main__":
    main(*sys.argv[1:])
