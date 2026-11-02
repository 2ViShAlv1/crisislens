"""Choose the SFT training subset from training events (class-rebalanced).

Imbalance handling: cap each class at CAP unique posts (random, seed 42) and keep every post of
the rarer classes. This flattens the prior toward rare, life-critical classes
(requests_or_urgent_needs, missing_or_found_people) without duplicating text (duplicates would be
collapsed by platform dedup and invite memorisation). The rare-class share in the result is logged.
"""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
D = ROOT / "data/processed"
tr = pd.read_parquet(D / "train.parquet")
sel = pd.concat([g.sample(min(CAP, len(g)), random_state=42) for _, g in tr.groupby("label")]).sample(frac=1, random_state=42)
sel.to_parquet(D / "sft_select.parquet", index=False)
va = pd.read_parquet(D / "val.parquet")
vs = pd.concat([g.sample(min(60, len(g)), random_state=42) for _, g in va.groupby("label")]).sample(frac=1, random_state=42)
vs.to_parquet(D / "val_eval.parquet", index=False)
c = pd.DataFrame({"pool": tr.label.value_counts(), "selected": sel.label.value_counts()})
c["pool_share"] = (c.pool / c.pool.sum()).round(3); c["selected_share"] = (c.selected / c.selected.sum()).round(3)
print(c.to_string()); print("selected", len(sel), "val_eval", len(vs))
c.to_csv(ROOT / "eval/results/sft_selection_support.csv")
