# Model card — CrisisLens (Qwen3.5-2B + LoRA, trained with AutoScientist)

## What it does
Reads one social-media post written during a natural disaster and returns a strict JSON triage record:
`{"category", "actionable", "urgency" (1–5), "location", "needs", "rationale" (≤20 words)}`.
`category` is one of the 10 HumAID humanitarian classes.

## Training
- **Base model:** `Qwen/Qwen3.5-2B` (Apache-2.0), chosen from AutoScientist's supported list because it can be served on a free CPU (llama.cpp, Q4_K_M base + f16 LoRA). Larger runs (9B, 4B) and data-scaling runs (37k, 49k rows) were also trained; see the README for what was and was not evaluated.
- **Released checkpoint:** AutoScientist run #5 (`7ebe8824-9857-4552-8574-8b58b3d55be5`), iteration 2: LoRA r32/α64 on all linear layers, dropout 0.05, lr 1.5e-4 cosine, 4 epochs, warmup 0.05, weight decay 0.01, 17,784 rows. AutoScientist refined this recipe itself from our iteration-1 override (lr 2e-4, 3 epochs). Judge win-rate 0.923.
- **Trainer:** Adaption Labs **AutoScientist**, raw-mode SFT (`training_method=instruction`), LoRA. AutoScientist chose the recipe (logged per run in `eval/autoscientist/`) and iterated toward a target win-rate vs. the base model.
- **Data:** class-rebalanced HumAID subset from 15 training disasters; gold category + teacher-LLM silver fields (see `docs/DATA_CARD.md`). Prompt v2 = instruction listing the 10 category ids and the urgency scale, then the post (`src/sft_format.py`).

## Results (4 unseen disasters, n = 1,219, human gold labels)
- Macro-F1 **0.715** [0.682, 0.743], weighted-F1 0.759. Best frontier API zero-shot: 0.606. TF-IDF + LR: 0.662.
- Recall: requests_or_urgent_needs **0.76**, injured_or_dead_people **0.90**.
- 99.9% schema-valid JSON. ~5 s per post on a laptop CPU, $0 per 1k posts.
- Full tables, per-disaster and per-class results: README. Failure cases: `docs/ERROR_ANALYSIS.md`.

## Intended use
Decision *support* for humanitarian information managers: pre-sorting a flood of posts into a queue that humans review, surfacing likely requests for help and casualty reports first. Not a replacement for human triage or emergency dispatch.

## Out-of-scope / misuse
- Automated dispatch or denial of aid without a human in the loop.
- Surveillance or profiling of individuals (posts are classified, people are not; do not link outputs to identities).
- Non-English or non-disaster content; non-Twitter registers (SMS, long reports) — untested.
- Using `urgency`/`actionable` as ground truth: these heads are trained on **silver** labels (≈89% all-field agreement with an independent reviewer).

## Known limitations & bias
- **Event shift:** trained on 15 disasters (≈58% US hurricanes); evaluated on 4 unseen ones (Kerala, Idai, Mexico, Dorian). Per-event scores are reported; performance on very different contexts (e.g. conflict, pandemics) is unknown.
- **Gold-label noise** in HumAID caps achievable accuracy; confusions concentrate on `other_relevant_information` ↔ damage/caution and `requests_or_urgent_needs` ↔ donation appeals.
- **Location** inherits the teacher's tendency to leave it null; it is extracted text, not geocoded.
- **Rare class:** `missing_or_found_people` has only 305 training and 19 test posts — its per-class F1 has very wide uncertainty.
- Cultural/linguistic bias: hashtags and place names from non-US events (Malayalam/Hindi transliterations, Shona/Portuguese names) are under-represented.

## Risks
False negatives on life-critical classes are the costliest error; we report recall on `requests_or_urgent_needs` and `injured_or_dead_people` explicitly. A wrong high-urgency label can waste responder time; the queue UI keeps the original post visible so humans verify.

## License
Model weights derived from HumAID (CC BY-NC-SA 4.0) — **non-commercial, share-alike**. Base model Apache-2.0.
