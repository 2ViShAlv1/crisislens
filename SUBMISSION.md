# CrisisLens: disaster triage with a 2B model trained by AutoScientist

**Macro-F1 0.715 on four disasters the model never saw in training, against 0.606 for the best frontier API zero-shot (deepseek-v4-flash), at $0 per 1,000 posts on a laptop CPU.** Recall on urgent requests for help is 0.76 (API: 0.32), and 99.9% of outputs are schema-valid JSON.

**Problem.** In a disaster, response desks are flooded with posts. CrisisLens turns each one into a triage record: HumAID category, actionable, urgency 1–5, location, needs, and a rationale. The demo ranks a live queue by urgency.

**Data.** HumAID (76,484 human-labelled tweets, 19 disasters). We recovered event ids, removed exact and near duplicates, and split **by disaster**: Kerala floods, Cyclone Idai, the Mexico earthquake and Hurricane Dorian are held out entirely. A teacher LLM added the extra fields (silver labels), while the human category label was always kept. Silver and gold metrics are reported separately.

**Training.** Qwen3.5-2B + LoRA, trained with Adaption AutoScientist (raw-mode instruction SFT). Eight runs, all logged. The default recipe failed: the model invented labels. We fixed the prompt and the learning rate, then let AutoScientist refine the recipe itself (lr 1.5e-4, 4 epochs, LoRA r32). Larger (4B, 9B) and data-scaling (37k, 49k rows) runs were also trained. The model was selected on a validation set from the training disasters; the test set was used once.

**Evaluation.** 1,219 test posts with human labels, compared against TF-IDF + LR (0.662), two frontier APIs, and the untuned base model (0.427). We report bootstrap CIs, per-class and per-disaster F1, life-critical recall, latency, cost and 15 analysed failures. Most frequent confusion: donation posts escalated to urgent requests.

**Serving.** PEFT → GGUF, served with llama.cpp on CPU. The web console (triage queue, analyzer, evaluation) runs live locally, and a static version with recorded outputs is hosted on Hugging Face.

Demo: https://huggingface.co/spaces/Vishal-2109/crisislens · Code: https://github.com/2ViShAlv1/crisislens
