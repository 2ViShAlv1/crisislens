# Error analysis: CrisisLens 2B (run #5) on 4 held-out disasters

15 errors picked deterministically by `eval/error_analysis.py ft2b_run5` (seed 3): up to 3 misses per life-critical
class, then the most frequent confusion pairs. Full rows (id, event, gold, prediction, raw output) are in
`eval/results/ft2b_run5_errors.csv`. Post text: HumAID (Alam et al., ICWSM 2021), CC BY-NC-SA 4.0.

The verdicts below are **our reading** of each post, not a second human annotation. "Ambiguous" means a careful
annotator could defend either label; HumAID assigns exactly one label per post.

## Most frequent confusions (gold → predicted), n = 1,219

| gold | predicted | count |
|---|---|---|
| rescue_volunteering_or_donation_effort | requests_or_urgent_needs | 45 |
| rescue_volunteering_or_donation_effort | other_relevant_information | 35 |
| other_relevant_information | caution_and_advice | 28 |
| other_relevant_information | not_humanitarian | 22 |
| rescue_volunteering_or_donation_effort | not_humanitarian | 14 |
| other_relevant_information | rescue_volunteering_or_donation_effort | 10 |

Every frequent confusion involves the two broad classes, `rescue_volunteering_or_donation_effort` and
`other_relevant_information`. The life-critical classes are rarely confused with each other.

## The 15 cases

| # | Disaster | Gold | Predicted | Post (abridged) | Verdict |
|---|---|---|---|---|---|
| 1 | Dorian | urgent needs | rescue / donation | "CEOC hours today 11-6pm. We are canvassing the community to help those in need. Please join us." | Ambiguous, leans model: a volunteer call, not a request for help |
| 2 | Idai | urgent needs | rescue / donation | "what are some legitimate charities I can donate to?" | Ambiguous, leans model: the poster offers help rather than needing it |
| 3 | Dorian | urgent needs | other | "a city runs out of food and water prior to hurricane landfall" | **Model error.** It extracted needs `food, water` but did not escalate the category |
| 4 | Kerala | injured / dead | urgent needs | "Death toll rises to 164… We are in urgent need of donations, supplies and volunteers." | Ambiguous (two topics in one post); the model still set urgency 4 and actionable |
| 5 | Idai | injured / dead | infrastructure | "We have lost 2 pupils as the landslide destroyed the dining hall…" | **Model error, life-critical.** Deaths should outrank damage. Urgency 4, so it still ranks high in the queue |
| 6 | Dorian | injured / dead | infrastructure | "could be similar to Hurricane Matthew… killed 12 people. Take this storm seriously!" | Ambiguous: a historical comparison and warning; arguably caution_and_advice |
| 7 | Kerala | rescue / donation | urgent needs | "how you are confirmed that he has donated only ₹10,000?… Kerala need financial help" | **Model error.** An argument about donations is not a request |
| 8 | Kerala | rescue / donation | urgent needs | "1. Immediate Rescue, 2. Emergency Relief… local collectives played key role" | **Model error.** Inconsistent too: urgency 2 and not actionable, yet labelled urgent needs |
| 9 | Kerala | rescue / donation | other | "India unlikely to accept UAE's $100m donation" | **Model error.** News about a donation is a donation-effort post |
| 10 | Kerala | rescue / donation | other | "Department of Food… made available 50,000 tonnes of food grains" | **Model error.** Government relief supply is a relief effort |
| 11 | Dorian | other | caution / advice | "Broward County is preparing storm shelters… persons with special needs must pre-register, call 3-1-1" | Ambiguous, leans model: contains explicit advice |
| 12 | Dorian | other | caution / advice | "City buildings, facilities and parks will remain closed…" | Ambiguous: an official closure notice |
| 13 | Idai | other | not humanitarian | "Let us hear the stories of these gallant children and not grandstanding politicians" | Ambiguous: political commentary |
| 14 | Mexico | rescue / donation | not humanitarian | "CAN YOU IMAGINE PEOPLE SPREADING INFO… DONATING AND STOP CARING ABOUT TRIVIAL THINGS" | Ambiguous: sarcasm |
| 15 | Mexico | other | rescue / donation | "Eating for cause… #Donate" | Ambiguous, leans model |

**Tally:** 6 clear model errors (#3, #5, #7, #8, #9, #10) and 9 ambiguous cases, 4 of which lean toward the model's
label. This sample is chosen from errors, not at random, so the ratio is not an estimate of label noise in HumAID.

## Patterns, and what would fix them

1. **"Need" vocabulary makes the model escalate donation chatter (#7, #8).** A post about money or relief that
   mentions "need" or "help" is pushed to `requests_or_urgent_needs`. This is the main reason precision on that class
   is low. For triage, over-escalating is the safer direction to err, but it adds noise to the queue.
   *Fix being tested:* run #7 trains on 2.5× more `rescue_volunteering_or_donation_effort` examples
   (5,000 instead of 2,000).
2. **Donation news is read as "other" (#9, #10).** Aid announcements by governments or other countries are donation
   efforts in HumAID's scheme. *Fix:* the same data scaling, plus one-line class definitions in the prompt.
3. **The secondary topic wins (#5).** When a damage report also mentions deaths, the model labels the damage. The
   urgency score (4) still puts the post near the top of the queue, which limits the harm.
   *Fix:* a multi-label head, or a rule that lets injured/dead override other categories.
4. **Official updates: `other` vs `caution_and_advice` (#11, #12).** This boundary is also unclear in the
   annotation guidelines. It costs macro-F1 but has little effect on triage.
5. **One label per post.** Posts that mix topics (#4, #6) cannot be fully right under single-label scoring. A
   production version should predict multiple labels.

## What the errors do *not* show

- **Invented labels or broken JSON.** 1 of 1,219 outputs failed the strict schema check, and every output parsed.
- **Life-critical posts buried.** Recall is 0.76 on `requests_or_urgent_needs` and 0.90 on `injured_or_dead_people`,
  against 0.32 and 0.73 for the best frontier API zero-shot. Where the category is wrong, the urgency field usually
  still ranks the post high in the queue (#4, #5).
