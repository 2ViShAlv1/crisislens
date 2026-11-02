"""Recover HumAID event ids for the HF `QCRI/HumAID-all` parquet splits.

The HF mirror has only (tweet_text, class_label), but rows are concatenated event-by-event
in the canonical HumAID order. We recover boundaries with an *ordered* segmentation:
each row gets a keyword-vote for an event; dynamic programming finds the 19 contiguous,
ordered segments maximising agreement with the votes. Boundaries are then checked
against the per-event counts reported in the HumAID paper (see eval/event_counts_check.csv).
"""
import re, sys, json
import numpy as np, pandas as pd

EVENTS = [  # canonical HumAID order (Alam et al., ICWSM 2021)
 ("ecuador_earthquake_2016", r"ecuador|quito|guayaquil|manabi|portoviejo"),
 ("canada_wildfires_2016", r"fort ?mc|fortmcmurray|ymm|alberta|canada|wildfire"),
 ("italy_earthquake_aug_2016", r"italy|amatrice|italian|norcia|terremoto"),
 ("kaikoura_earthquake_2016", r"kaikoura|new zealand|\bnz\b|wellington|christchurch"),
 ("hurricane_matthew_2016", r"matthew|haiti"),
 ("sri_lanka_floods_2017", r"sri ?lanka|lanka|colombo"),
 ("hurricane_harvey_2017", r"harvey|houston"),
 ("hurricane_irma_2017", r"\birma\b|florida"),
 ("hurricane_maria_2017", r"maria|puerto|\bpr\b"),
 ("mexico_earthquake_2017", r"mexico|mexican|sismo|cdmx"),
 ("maryland_floods_2018", r"maryland|ellicott"),
 ("greece_wildfires_2018", r"greece|athens|\bmati\b|greek"),
 ("kerala_floods_2018", r"kerala|keralaflood|kochi|india"),
 ("hurricane_florence_2018", r"florence|carolina"),
 ("california_wildfires_2018", r"california|camp ?fire|paradise|woolsey|campfire"),
 ("cyclone_idai_2019", r"idai|mozambique|zimbabwe|malawi|beira"),
 ("midwestern_us_floods_2019", r"nebraska|midwest|iowa|missouri"),
 ("hurricane_dorian_2019", r"dorian|bahamas|abaco"),
 ("pakistan_earthquake_2019", r"pakistan|mirpur|azad|kashmir|\bajk\b"),
]

def votes(texts):
    t = texts.str.lower()
    V = np.zeros((len(t), len(EVENTS)), dtype=np.int32)
    for j, (_, pat) in enumerate(EVENTS):
        V[:, j] = t.str.contains(pat, regex=True).astype(int).values
    return V

def segment(V, present):
    """Ordered segmentation over the subset of events `present` (indices, in order).
    score = sum of votes of each row for its assigned segment's event. O(n*k)."""
    n, k = V.shape[0], len(present)
    cs = np.vstack([np.zeros(k), np.cumsum(V[:, present], axis=0)])  # (n+1, k)
    NEG = -1e18
    # dp[j][i]: best score covering rows [0,i) with first j+1 segments, segment j ending at i
    dp = np.full((k, n + 1), NEG); bp = np.zeros((k, n + 1), dtype=np.int64)
    dp[0, 1:] = cs[1:, 0]
    for j in range(1, k):
        # dp[j,i] = max_{s<i} dp[j-1,s] + cs[i,j]-cs[s,j]  (s>=j so each segment non-empty)
        best = NEG; arg = 0
        for i in range(j + 1, n + 1):
            s = i - 1
            cand = dp[j - 1, s] - cs[s, j]
            if cand > best: best, arg = cand, s
            dp[j, i] = best + cs[i, j]; bp[j, i] = arg
    bounds = [n]; i = n
    for j in range(k - 1, 0, -1):
        i = bp[j, i]; bounds.append(i)
    bounds.append(0); bounds = bounds[::-1]
    return bounds, dp[k - 1, n]

def main(raw_dir, out_dir):
    stats = []
    for split in ["train", "validation", "test"]:
        d = pd.read_parquet(f"{raw_dir}/humaid_{split}.parquet")
        V = votes(d.tweet_text)
        # events present: those with a clear keyword block (val lacks some small events)
        present = list(range(len(EVENTS)))
        b, sc = segment(V, present)
        # drop events that got a tiny/no-vote segment and re-segment (val/test miss a few events)
        while True:
            bad = [p for idx, p in enumerate(present)
                   if V[b[idx]:b[idx+1], p].sum() < 0.2 * max(1, b[idx+1]-b[idx]) ]
            if not bad: break
            present = [p for p in present if p not in bad[:1]]
            b, sc = segment(V, present)
        ev = np.empty(len(d), dtype=object)
        for idx, p in enumerate(present):
            ev[b[idx]:b[idx+1]] = EVENTS[p][0]
            seg = V[b[idx]:b[idx+1]]
            stats.append(dict(split=split, event=EVENTS[p][0], start=b[idx], end=b[idx+1], n=b[idx+1]-b[idx],
                              own_vote_rate=round(seg[:, p].mean(), 3),
                              other_vote_rate=round((seg.sum(1) - seg[:, p] > 0).mean(), 3)))
        d["event"] = ev; d["split_orig"] = split
        d.to_parquet(f"{out_dir}/humaid_{split}_events.parquet", index=False)
    s = pd.DataFrame(stats); print(s.to_string()); s.to_csv(f"{out_dir}/event_segmentation.csv", index=False)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
