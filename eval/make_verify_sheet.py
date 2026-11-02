"""Sample 100 teacher-labelled posts (50 train-subset, 50 test-subset; seed 7) into a review sheet.
Reviewer marks each extra field agree (1) / disagree (0); agreement rate computed by eval/verify_agreement.py."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
rows = []
for split, n in [("sft_select", 50), ("test_eval", 50)]:
    df = pd.read_parquet(ROOT / f"data/processed/{split}.parquet").set_index("id")
    T = [json.loads(l) for l in open(ROOT / f"data/processed/teacher/{split}.jsonl", encoding="utf-8")]
    T = pd.DataFrame([t for t in T if "label" in t]).drop_duplicates("id").sample(n, random_state=7)
    for t in T.itertuples():
        L = t.label
        rows.append({"split": split, "id": t.id, "category_gold": L["category"], "text": df.loc[t.id, "text"],
                     "actionable": L["actionable"], "urgency": L["urgency"], "location": L["location"],
                     "needs": "|".join(L["needs"]), "rationale": L["rationale"],
                     "ok_actionable": "", "ok_urgency": "", "ok_location": "", "ok_needs": "", "note": ""})
pd.DataFrame(rows).to_csv(ROOT / "eval/human_check/verify_sheet.csv", index=False, encoding="utf-8")
print(len(rows))
