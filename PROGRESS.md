# PROGRESS — CrisisLens (8h budget, T0 = 2026-11-03 00:15 IST)

| Time (IST) | T+ | Status |
|---|---|---|
| 00:15 | 0:00 | Started. Repo init, Windows alarm script tested (scripts/alarm.ps1). |
| 00:18 | 0:03 | AutoScientist docs read (docs.adaptionlabs.ai). Raw CrisisNLP zip = HTTP 403; Dataverse file restricted. Using HF mirror `QCRI/HumAID-all` (76,484 tweets, CC-BY-NC-SA-4.0). |
| 00:25 | 0:10 | HF mirror lacks event column; rows are event-contiguous. Ordered DP segmentation matches paper Table 6 per-event counts within ±2 rows → snapping to paper counts. |
| 00:35 | 0:20 | Clean data: 49,417 train / 5,491 val / 21,527 test (held-out events: Kerala floods, Cyclone Idai, Mexico EQ, Hurricane Dorian). Exact + near-dup (cos>=0.9) dedupe; 15 train rows overlapping test removed. 10 classes present (no dont_know in this release). |
| 00:45 | 0:30 | SFT selection 17,786 rows (cap 2k/class). LLM eval subset 1,219 (300/event + all 19 missing_or_found). Prompt/schema, teacher labeler, inference harness written. TF-IDF baseline training. |
| 00:50 | 0:35 | **BLOCKED**: waiting for ADAPTION_API_KEY (+ HF_TOKEN, optional TOGETHER_API_KEY) in .env. Alarm rung 00:27. |
| 00:55 | 0:40 | TF-IDF+LR baseline (C=2): full test macro-F1 0.652 / weighted-F1 0.700; eval subset macro-F1 0.662; urgent-needs recall 0.53, injured/dead recall 0.83. Val (in-event) macro-F1 0.741 → event shift costs ~9 pts. |
| 01:10 | 0:55 | Keys in. Live AutoScientist model list differs from docs: sub-10B = Qwen3.5-2B/4B/9B, Qwen3-4B-Base, Gemma-3-4B-pt. Together key → 401; HF Inference Providers → 402 (no credits). Alarm rung. |
| 01:20 | 1:05 | Teacher probe: deepseek-v4-flash / gemma-4-31b-it / glm-5.2 agree closely; qwen3.5-397b burns budget on reasoning. Chose gemma-4-31b-it (fastest, ~$1.5 total). |
| 01:40 | 1:25 | Hit Adaption per-user inference rate limit (429). Switched teacher to 20 posts/request + global back-off: ~6 rows/s. llama.cpp download blocked by permission classifier (needs user OK). |
