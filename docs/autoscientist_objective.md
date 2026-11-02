# AutoScientist objective — CrisisLens

**Task.** Given ONE social-media post written during a natural disaster, output ONLY a JSON triage record:
`{"category", "actionable", "urgency", "location", "needs", "rationale"}`.

**Schema.**
- `category` ∈ {caution_and_advice, displaced_people_and_evacuations, infrastructure_and_utility_damage,
  injured_or_dead_people, missing_or_found_people, not_humanitarian, other_relevant_information,
  requests_or_urgent_needs, rescue_volunteering_or_donation_effort, sympathy_and_support} (HumAID definitions, see src/prompt.py)
- `actionable`: bool — a responder could act on it.
- `urgency`: int 1–5 (5 = life-threatening: trapped/injured/missing/immediate rescue).
- `location`: most specific place written in the post, else null (never invented).
- `needs`: subset of {rescue, medical, food, water, shelter, evacuation, power, communication, transport,
  sanitation, clothing, money, volunteers, information, missing-person-search}.
- `rationale`: ≤ 20 words.

**Success metric (ours).** Macro-F1 of `category` on 4 *unseen* disasters (event-level holdout), recall on
`requests_or_urgent_needs` and `injured_or_dead_people`, valid-JSON rate (target ≥ 99%).
**Platform metric.** AutoScientist optimises win-rate of the adapted model vs the base model on held-out rows of
the training dataset; we report it alongside our metrics.

**Edge cases.** Donation appeals by big organisations → rescue_volunteering (not requests). Death tolls in news
headlines → injured_or_dead. Prayers that also mention damage → category by dominant intent. Ads/politics that
mention a disaster → not_humanitarian. Non-English or code-mixed posts (e.g. Malayalam/Hindi in Kerala) → still classify.

**How this reaches the platform.** The public API has no free-text objective field for raw-mode runs
(`autoscientist.create` takes dataset/model/method/hyperparams/target_win_rate). The objective is therefore
encoded in the data itself: a fixed instruction prefix on every prompt and a canonical JSON completion.
