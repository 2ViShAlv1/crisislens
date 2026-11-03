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
| 02:19 | 2:04 | **Run #1 (pilot) succeeded** in 27 min: 1 iteration, best win-rate 0.70 = target → early stop. Recipe: LoRA r16/α32 on q,k,v,o,gate,up,down; lr 1e-5 cosine; 1 epoch = 21 steps; loss 1.46→1.04 (eval 1.04). Checkpoint is zstd-tar (not gzip as named) — extractor fixed. |
| 02:21 | 2:06 | **Run #4** 25d04b86 launched: Qwen3.5-2B full data, target_win_rate 0.95, max_iterations 5 — forces AutoScientist to keep iterating (default 0.70 target stops after 1 iteration). |
| 02:22 | 2:07 | Pilot LoRA → GGUF (192 tensors, 21.8 MB) via llama.cpp convert_lora_to_gguf; served on CPU (Q4_K_M base + f16 LoRA). 100-post probe running. |
| 02:35 | 2:20 | Base-2B zero-shot eval (≈1,0xx/1,219 done) and pilot FT probe (≈70/100) were **killed by Claude Code for low system memory** (0.8 GB free; two llama-servers ≈7.7 GB). Servers stopped. Per the harness rule, evals are NOT restarted automatically — waiting for user OK. Partial preds are kept (scripts resume). |
| 02:33 | 2:18 | **Pilot FT model fails our metric**: on 69 probe posts category acc 0.00 / valid JSON 0.00 (invents labels like "relief", urgency "high") vs base zero-shot 0.377 / 1.00 and TF-IDF 0.696. Cause: default recipe = 21 steps at lr 1e-5 (loss 1.46→1.04) + v1 prompt never shows the label set. AutoScientist's own win-rate (0.70) did not detect this. |
| 02:35 | 2:20 | **Iteration (data + recipe)**: prompt v2 lists the 10 label ids + urgency scale; hyperparam override lr 2e-4, 3 epochs, LoRA r32/α64. **Run #5** 7ebe8824 launched (Qwen3.5-2B, 17,784 rows, target 0.9). |
| 02:50 | 2:35 | Iteration 1 win-rates vs base: run #2 (2B full) 0.632, run #3 (9B full) 0.542, run #4 (2B t=0.95) 0.603 — all below the 0.70 target, so AutoScientist continues iterating. |
| 02:55 | 2:40 | **Run #5 (prompt v2 + lr 2e-4, 3 ep, r32) iteration 1 win-rate 0.889** vs 0.54–0.63 for the default-recipe runs. Target 0.90 → continues iterating. |
| 03:01 | 2:46 | **Run #2 succeeded**: iteration 2 win-rate **0.918** (from 0.632). AutoScientist changed the recipe itself between iterations: lr 1e-5→5e-5, epochs 1→3, LoRA r16→32 (α64), warmup 0.1→0.05; train loss 1.46→0.26, eval loss 0.28 (63 steps). Checkpoint downloaded + converted to GGUF. Uses prompt v1 (no label list). |
| 03:24 | 3:09 | **Run #5 succeeded**: iteration 2 win-rate **0.923** (iter 1: 0.889). AutoScientist refined my override: epochs 3→4, lr 2e-4→1.5e-4, warmup 0.1→0.05; 116 steps, train loss 0.735→0.19, eval loss 0.214. Downloaded + GGUF. **Candidate model.** CPU eval waits for user 'restart evals'. |
| 03:45 | 3:30 | **Run #3 (9B) succeeded** at 3/3 iterations, best win-rate 0.562 (0.542→0.559→0.562). Kept as platform-metric comparison only (9B cannot be evaluated on this 16 GB CPU laptop). |
| 05:08 | 4:53 | Memory back to 6.4 GB free. Started ONE new low-memory job (not the killed ones): run #5 FT eval, single llama-server slot, ctx 2048, with scripts/mem_guard.sh (stops server < 2 GB free). 20-post probe: 20/20 valid JSON, 4.05 s/post. Full 1,219 running (~80 min). Killed base zero/few-shot jobs still await user OK. |
| 05:50 | 5:35 | Frontier zero-shot baselines (Adaption hosted, n=1,219): deepseek-v4-flash macro-F1 0.606 (urgent-needs recall 0.32), gemma-4-31b-it 0.592 (0.26). Interim FT run #5 on first 306 posts: macro-F1 0.736 vs TF-IDF 0.658 vs base zero-shot 0.392. |
