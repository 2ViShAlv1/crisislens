"""Grow the SFT subset to a higher per-class cap (data-scaling experiment, run #7).

Keeps every row of the existing cap-2000 selection (already teacher-labelled) and adds new rows from
the same training pool until each class has min(CAP, pool) rows. Only TRAIN events are used; the
event-level test split and the val split are untouched. Writes
  data/processed/sft_select_c{CAP//1000}k.parquet        (superset)
  data/processed/sft_select_c{CAP//1000}k_extra.parquet  (new rows only -> to teacher-label)
Usage: python -I src/data/select_train_more.py 5000
"""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; D = ROOT / "data/processed"
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
tag = f"sft_select_c{CAP // 1000}k"
tr = pd.read_parquet(D / "train.parquet"); old = pd.read_parquet(D / "sft_select.parquet")
rest = tr[~tr.id.isin(old.id)]
extra = []
for lab, g in rest.groupby("label"):
    need = max(0, min(CAP, (tr.label == lab).sum()) - (old.label == lab).sum())
    extra.append(g.sample(min(need, len(g)), random_state=7))
extra = pd.concat(extra)
sup = pd.concat([old, extra]).sample(frac=1, random_state=42)
assert sup.id.is_unique
sup.to_parquet(D / f"{tag}.parquet", index=False); extra.to_parquet(D / f"{tag}_extra.parquet", index=False)
c = pd.DataFrame({"pool": tr.label.value_counts(), "cap2k": old.label.value_counts(), tag: sup.label.value_counts()})
print(c.to_string()); print("superset", len(sup), "new rows to label", len(extra))
c.to_csv(ROOT / f"eval/results/{tag}_support.csv")
