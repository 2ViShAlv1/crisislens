# PROGRESS — CrisisLens (8h budget, T0 = 2026-11-03 00:15 IST, hard stop 08:15)

Times are IST, corrected against git commit timestamps (an earlier version of this log had estimated times that ran fast).

| Time | T+ | Status |
|---|---|---|
| 00:15 | 0:00 | Started. Repo init, Windows alarm script tested (scripts/alarm.ps1). |
| 00:17 | 0:02 | AutoScientist docs read (docs.adaptionlabs.ai). CrisisNLP zip = HTTP 403; Dataverse file restricted. Using HF mirror `QCRI/HumAID-all` (76,484 tweets, CC-BY-NC-SA-4.0). |
| 00:19 | 0:04 | HF mirror lacks event column; rows are event-contiguous. Ordered DP segmentation agrees with paper Table 6 counts on 99.99% of rows → events assigned from paper counts. |
| 00:26 | 0:11 | Clean data: 49,417 train / 5,491 val / 21,527 test (held-out events: Kerala floods 2018, Cyclone Idai 2019, Mexico EQ 2017, Hurricane Dorian 2019). Exact + near-dup (cos≥0.9) dedupe; 15 train rows overlapping test removed. 10 classes present. |
| 00:28 | 0:13 | SFT selection 17,786 rows (cap 2k/class). LLM eval subset 1,219 (300/event + all 19 missing_or_found). Prompt/schema, teacher labeler, inference harness. |
| 00:34 | 0:19 | TF-IDF+LR baseline (C=2): full test macro-F1 0.652 / weighted 0.700; eval subset macro-F1 0.662; urgent-needs recall 0.53, injured/dead 0.83. In-event val macro-F1 0.741 → event shift costs ~9 pts. |
| 00:35–01:10 | | **BLOCKED** on API keys (alarm rung 00:27). |
| ~01:12 | 0:57 | Keys in. Live AutoScientist model list differs from docs (sub-10B: Qwen3.5-2B/4B/9B, Qwen3-4B-Base, Gemma-3-4B-pt). Together key → 401; HF Inference Providers → 402. Alarm rung. |
| ~01:15 | 1:00 | Teacher probe (deepseek-v4-flash / gemma-4-31b-it / glm-5.2 agree closely; qwen3.5-397b burns budget on reasoning) → gemma-4-31b-it. |
| ~01:25 | 1:10 | Adaption per-user inference rate limit (429). Teacher switched to 20 posts/request + global back-off (~7–8 rows/s). |
| 01:34 | 1:19 | Test/val silver labels done. Teacher verification on 100 rows (independent Claude review, NOT human): actionable 97%, urgency ±1 100%, location 94%, needs 96%, all-fields 89%. |
| 01:41 | 1:26 | User approved llama.cpp (official b12052 CPU build). Base Qwen3.5-2B Q4_K_M on CPU: ~5–6 s/post, valid JSON. **Plan change:** primary fine-tune = Qwen3.5-2B (fully evaluable + servable on free CPU); Qwen3.5-9B as larger comparison. |
| 01:48 | 1:33 | Base-2B zero-shot → few-shot baselines running on CPU (1,219 posts each). Installing torch/transformers/peft for LoRA→GGUF conversion. |
| 01:51 | 1:36 | **AutoScientist run #1 launched**: id 99464db5, Qwen/Qwen3.5-2B, raw SFT, 8,058 rows (pilot), max_iterations 3, target win-rate 0.70. Recommended recipe: LoRA r16/α32 all-linear, lr 1e-5 cosine, 1 epoch, train_on_inputs=false. Note: passing column_mapping {prompt,completion} → 400 (raw ingest renames columns); omitted → inferred. |
| 02:13 | 1:58 | Teacher labels complete (17,784 train / 1,219 test / 574 val). **Run #2** 2c060ec8 (Qwen3.5-2B, full 17,784 rows) and **Run #3** 165c0bf4 (Qwen3.5-9B, full) launched; same AutoScientist-recommended recipe as pilot. One run per dataset at a time (409) → separate dataset copy for 9B. |
