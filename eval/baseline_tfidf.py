"""Baseline 3: TF-IDF (word 1-2g + char 2-5g) + Logistic Regression, class_weight=balanced.
C tuned on the validation set (training events only). Reported on full test and the LLM-eval subset."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.eval.metrics import classification_report, CLASSES

D = ROOT / "data/processed"; R = ROOT / "eval/results"; R.mkdir(parents=True, exist_ok=True)
tr, va, te, sub = (pd.read_parquet(D / f) for f in ["train.parquet", "val.parquet", "test.parquet", "test_eval.parquet"])
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000)
X = lambda t: hstack([w.transform(t), c.transform(t)]).tocsr()
w.fit(tr.text); c.fit(tr.text)
Xtr, Xva = X(tr.text), X(va.text)
best = None
for C in [0.5, 1, 2, 4, 8]:
    m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, tr.label)
    f = f1_score(va.label, m.predict(Xva), average="macro"); print(f"C={C} val macroF1={f:.4f}")
    if best is None or f > best[0]: best = (f, C, m)
f, C, m = best
res = {"model": "tfidf_logreg", "C": C, "val_macro_f1": f}
for name, df in [("test_full", te), ("test_eval", sub)]:
    t0 = time.perf_counter(); pred = m.predict(X(df.text)); dt = time.perf_counter() - t0
    rep, per, cm = classification_report(df.label.values, pred)
    rep["latency_ms_per_post"] = 1000 * dt / len(df)
    res[name] = rep
    per.to_csv(R / f"tfidf_{name}_per_class.csv"); cm.to_csv(R / f"tfidf_{name}_confusion.csv")
    pd.DataFrame({"id": df.id, "pred": pred}).to_csv(R / f"tfidf_{name}_preds.csv", index=False)
json.dump(res, open(R / "baseline_tfidf.json", "w"), indent=2)
print(json.dumps(res, indent=2))
