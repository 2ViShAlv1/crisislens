# PROGRESS — CrisisLens (8h budget, T0 = 2026-11-03 00:15 IST)

| Time (IST) | T+ | Status |
|---|---|---|
| 00:15 | 0:00 | Started. Repo init, Windows alarm script tested (scripts/alarm.ps1). |
| 00:18 | 0:03 | AutoScientist docs read (docs.adaptionlabs.ai). Raw CrisisNLP zip = HTTP 403; Dataverse file restricted. Using HF mirror `QCRI/HumAID-all` (76,484 tweets, CC-BY-NC-SA-4.0). |
| 00:25 | 0:10 | HF mirror lacks event column; rows are event-contiguous. Ordered DP segmentation matches paper Table 6 per-event counts within ±2 rows → snapping to paper counts. |
| 00:35 | 0:20 | Clean data: 49,417 train / 5,491 val / 21,527 test (held-out events: Kerala floods, Cyclone Idai, Mexico EQ, Hurricane Dorian). Exact + near-dup (cos>=0.9) dedupe; 15 train rows overlapping test removed. 10 classes present (no dont_know in this release). |
| 00:45 | 0:30 | SFT selection 17,786 rows (cap 2k/class). LLM eval subset 1,219 (300/event + all 19 missing_or_found). Prompt/schema, teacher labeler, inference harness written. TF-IDF baseline training. |
| 00:50 | 0:35 | **BLOCKED**: waiting for ADAPTION_API_KEY (+ HF_TOKEN, optional TOGETHER_API_KEY) in .env. Alarm rung 00:27. |
