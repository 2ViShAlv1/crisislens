"""Fixed LLM-evaluation subset of the held-out test events: N random posts per event (seed 42).
All 19 missing_or_found_people test posts are added so every class has support.
LLM inference is the expensive part, so all LLM systems are compared on this identical subset;
the TF-IDF baseline is reported on both the subset and the full test set."""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
t = pd.read_parquet(ROOT / "data/processed/test.parquet")
s = pd.concat([g.sample(min(N, len(g)), random_state=42) for _, g in t.groupby("event")])
# rarest class (missing_or_found_people, 19 test posts) would otherwise be ~absent: include all of them
s = pd.concat([s, t[(t.label == "missing_or_found_people") & ~t.id.isin(s.id)]])
s = s.sample(frac=1, random_state=42).reset_index(drop=True)
s.to_parquet(ROOT / "data/processed/test_eval.parquet", index=False)
print(len(s)); print(s.event.value_counts()); print(s.label.value_counts())
