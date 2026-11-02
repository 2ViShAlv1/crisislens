"""Agreement between teacher silver labels and an independent reviewer on the 100-row sheet.
Reviewer for eval/human_check/verify_sheet_claude.csv = Claude (LLM) reading each post — NOT a human.
Rules: actionable/location/needs strict agree/disagree vs. rubric; urgency agree if within +-1 of reviewer's own score."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
f = sys.argv[1] if len(sys.argv) > 1 else "eval/human_check/verify_sheet_claude.csv"
d = pd.read_csv(ROOT / f)
res = {"reviewer_file": f, "n": len(d)}
for k in ["actionable", "urgency", "location", "needs"]:
    res[f"agree_{k}"] = float(d[f"ok_{k}"].mean())
res["all_fields_agree"] = float((d[[f"ok_{k}" for k in ["actionable", "urgency", "location", "needs"]]].min(axis=1)).mean())
for s, g in d.groupby("split"):
    res[f"all_fields_agree_{s}"] = float(g[[f"ok_{k}" for k in ["actionable", "urgency", "location", "needs"]]].min(axis=1).mean())
json.dump(res, open(ROOT / "eval/results/teacher_verification.json", "w"), indent=2); print(json.dumps(res, indent=2))
