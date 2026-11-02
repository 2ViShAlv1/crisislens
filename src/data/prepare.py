"""CrisisLens data preparation.

Steps (all deterministic, seed=42):
 1. Load HF `QCRI/HumAID-all` splits and assign event ids from per-event counts in the
    HumAID paper (Alam et al., ICWSM 2021, Table 6). The HF mirror is event-contiguous;
    assign_events.py's DP segmentation independently agrees within +-2 rows (logged).
    Note: paper lists Hurricane Matthew dev=329/test=168; the data shows these are swapped
    (dev=168/test=329 is the only assignment consistent with split totals and keywords).
 2. Clean: strip RT prefix, URLs, @handles; unescape HTML; normalise whitespace.
 3. Dedupe: exact + near-duplicate (normalised key: lowercase, no punctuation/digits/hashtag
    symbols, collapsed whitespace). Conflicting labels for the same key -> majority, ties dropped.
 4. EVENT-LEVEL SPLIT: hold out 4 entire events as test. Remove from train any row whose
    normalised key also appears in test (cross-event leakage guard).
 5. Stratified validation set from training events.
 6. Hashed id = sha1(original text)[:16] for traceability (HF mirror has no tweet ids).
"""
import hashlib, html, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"

PAPER_COUNTS = [  # (event, train, dev, test) -- HumAID Table 6, Matthew dev/test swapped (see docstring)
 ("ecuador_earthquake_2016", 1094, 159, 310), ("canada_wildfires_2016", 1569, 228, 445),
 ("italy_earthquake_aug_2016", 840, 122, 239), ("kaikoura_earthquake_2016", 1536, 224, 435),
 ("hurricane_matthew_2016", 1157, 168, 329), ("sri_lanka_floods_2017", 392, 57, 111),
 ("hurricane_harvey_2017", 6378, 929, 1805), ("hurricane_irma_2017", 6579, 958, 1862),
 ("hurricane_maria_2017", 5094, 742, 1442), ("mexico_earthquake_2017", 1410, 205, 400),
 ("maryland_floods_2018", 519, 75, 148), ("greece_wildfires_2018", 1060, 154, 301),
 ("kerala_floods_2018", 5588, 814, 1582), ("hurricane_florence_2018", 4384, 639, 1241),
 ("california_wildfires_2018", 5163, 752, 1461), ("cyclone_idai_2019", 2753, 401, 779),
 ("midwestern_us_floods_2019", 1316, 191, 373), ("hurricane_dorian_2019", 5329, 776, 1508),
 ("pakistan_earthquake_2019", 1370, 199, 389),
]
TEST_EVENTS = ["kerala_floods_2018", "cyclone_idai_2019", "mexico_earthquake_2017", "hurricane_dorian_2019"]
CLASSES = ["caution_and_advice", "displaced_people_and_evacuations", "infrastructure_and_utility_damage",
           "injured_or_dead_people", "missing_or_found_people", "not_humanitarian",
           "other_relevant_information", "requests_or_urgent_needs",
           "rescue_volunteering_or_donation_effort", "sympathy_and_support"]

URL = re.compile(r"https?://\S+|www\.\S+|\bt\.co/\S+")
HANDLE = re.compile(r"@\w+")
RT = re.compile(r"^\s*RT\s*:?\s*")

def clean(t: str) -> str:
    t = html.unescape(t).replace("�", " ")
    t = RT.sub("", t)
    t = URL.sub("", t)
    t = HANDLE.sub("@user", t)
    t = re.sub(r"(@user[\s:]*){2,}", "@user ", t)
    t = re.sub(r"^(@user[\s:]*)+", "", t)  # leading reply/RT handles carry no content
    t = re.sub(r"\s+", " ", t).strip(" |:-")
    return t.strip()

def norm_key(t: str) -> str:
    t = t.lower().replace("@user", " ")
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

NEAR = 0.9

def near_dedupe(a: pd.DataFrame, ref: pd.DataFrame, cross: bool = False):
    """Drop rows of `a` that are near-duplicates (cosine>=NEAR) of an earlier row of `a`
    (cross=False) or of any row of `ref` (cross=True). Returns (filtered a, n_removed)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    vec.fit(pd.concat([a.key, ref.key]))
    A = vec.transform(a.key); R = A if not cross else vec.transform(ref.key)
    drop = np.zeros(A.shape[0], bool)
    for s in range(0, A.shape[0], 2000):
        S = (A[s:s + 2000] @ R.T).tocoo()
        m = S.data >= NEAR
        r, c = S.row[m] + s, S.col[m]
        if cross:
            drop[r] = True
        else:
            drop[r[c < r]] = True  # keep first occurrence
    return a[~drop], int(drop.sum())

def assign_events(d: pd.DataFrame, split_idx: int) -> pd.Series:
    sizes = [c[1 + split_idx] for c in PAPER_COUNTS]
    assert sum(sizes) == len(d), (sum(sizes), len(d))
    return pd.Series(np.repeat([c[0] for c in PAPER_COUNTS], sizes), index=d.index)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    parts = []
    for i, s in enumerate(["train", "validation", "test"]):
        d = pd.read_parquet(RAW / f"humaid_{s}.parquet")
        d["event"] = assign_events(d, i).values
        d["orig_split"] = s
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    # cross-check vs DP segmentation
    seg = ROOT / "data/interim/event_segmentation.csv"
    if seg.exists():
        dp = pd.read_csv(seg)
        dp_ev = []
        for s in ["train", "validation", "test"]:
            for _, r in dp[dp.split == s].iterrows():
                dp_ev += [r.event] * int(r.n)
        stats["dp_vs_paper_event_agreement"] = float(np.mean(np.array(dp_ev) == d.event.values))
    stats["raw_rows"] = len(d)
    d["id"] = [hashlib.sha1(t.encode("utf-8")).hexdigest()[:16] for t in d.tweet_text]
    d["text"] = d.tweet_text.map(clean)
    d["key"] = d.text.map(norm_key)
    d = d.rename(columns={"class_label": "label"})
    stats["rows_with_retweet_prefix"] = int(d.tweet_text.str.match(r"^\s*RT\b").sum())
    # drop empty / ultra-short after cleaning
    short = d.key.str.split().str.len().fillna(0) < 3
    stats["dropped_too_short"] = int(short.sum()); d = d[~short]
    # dedupe within (test-events) and (train-events) separately, then cross-split guard
    d["is_test"] = d.event.isin(TEST_EVENTS)
    out = []
    for is_test, g in d.groupby("is_test"):
        lab = g.groupby("key").label.agg(lambda x: x.value_counts().index[0] if (x.value_counts().iloc[0] > x.value_counts().iloc[1:2].sum() or len(x.value_counts()) == 1) else None)
        conflicts = lab.isna().sum()
        g = g.drop_duplicates("key").copy()
        g["label"] = g.key.map(lab)
        stats[f"{'test' if is_test else 'train'}_label_conflict_keys_dropped"] = int(conflicts)
        out.append(g[g.label.notna()])
    stats["dupes_removed"] = int(len(d) - sum(len(g) for g in out) - sum(stats[k] for k in stats if k.endswith("conflict_keys_dropped")))
    train_pool, test = (out[0], out[1]) if not out[0].is_test.iloc[0] else (out[1], out[0])
    leak = train_pool.key.isin(set(test.key))
    stats["train_rows_removed_exact_overlap_test"] = int(leak.sum())
    train_pool = train_pool[~leak]
    # near-duplicates: cosine >= NEAR on word 1-2gram tf-idf
    train_pool, n_tr = near_dedupe(train_pool, train_pool)
    test, n_te = near_dedupe(test, test)
    stats["near_dupes_removed_train"], stats["near_dupes_removed_test"] = n_tr, n_te
    train_pool, n_x = near_dedupe(train_pool, test, cross=True)
    stats["train_rows_removed_near_dup_of_test"] = n_x
    tr, va = train_test_split(train_pool, test_size=0.10, stratify=train_pool.label, random_state=SEED)
    cols = ["id", "event", "orig_split", "text", "label"]
    tr, va, test = tr[cols].sort_values("id"), va[cols].sort_values("id"), test[cols].sort_values("id")
    assert not set(tr.event) & set(test.event) and not set(va.event) & set(test.event)
    assert not set(tr.id) & set(test.id)
    for name, df in [("train", tr), ("val", va), ("test", test)]:
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        stats[f"{name}_rows"] = len(df)
    # stats tables
    sup = pd.DataFrame({n: df.label.value_counts() for n, df in [("train", tr), ("val", va), ("test", test)]}).reindex(CLASSES).fillna(0).astype(int)
    ev = pd.concat([tr.assign(split="train"), va.assign(split="val"), test.assign(split="test")]).groupby(["event", "split"]).size().unstack(fill_value=0)
    (ROOT / "eval/results").mkdir(parents=True, exist_ok=True)
    sup.to_csv(ROOT / "eval/results/class_support.csv"); ev.to_csv(ROOT / "eval/results/event_support.csv")
    json.dump(stats, open(ROOT / "eval/results/data_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2)); print(sup.to_string()); print(ev.to_string())

if __name__ == "__main__":
    main()
