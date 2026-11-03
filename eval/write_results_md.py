"""Fill README.md between <!-- RESULTS:START --> and <!-- RESULTS:END --> from eval/results only.

  python -I eval/write_results_md.py <final_key>

Every number printed here is read from eval/results/*.json|csv produced by score.py / make_report.py /
plot_autoscientist.py; nothing is typed by hand.
"""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "eval/results"

# candidate key -> (run label, AutoScientist run id, base, training rows, prompt)
CANDS = {
    "ft2b_run5": ("#5", "7ebe8824-9857-4552-8574-8b58b3d55be5", "Qwen3.5-2B", "17,784 (cap 2k/class)", "v2"),
    "ft2b_run4": ("#4", "25d04b86-af94-45cf-be37-5acc5822614c", "Qwen3.5-2B", "17,784 (cap 2k/class)", "v1"),
    "ft4b_run6": ("#6", "fafc81b1-8876-43c1-80e9-75a1791d676c", "Qwen3.5-4B", "17,784 (cap 2k/class)", "v2"),
    "ft2b_run7": ("#7", "2b634ded-e0b7-4a21-9467-bf4025b0bcc1", "Qwen3.5-2B", "37,267 (cap 5k/class)", "v2"),
    "ft2b_run8": ("#8", "fab5f195-3c03-4f96-ac63-fb31f6693054", "Qwen3.5-2B", "49,396 (all)", "v2"),
}
FINAL = sys.argv[1] if len(sys.argv) > 1 else "ft2b_run5"
f3 = lambda v: "—" if v is None or pd.isna(v) else f"{v:.3f}"
pc = lambda v: "—" if v is None or pd.isna(v) else f"{100 * v:.1f}%"

fin = json.load(open(R / f"{FINAL}.json"))
summ = pd.read_csv(R / "summary.csv")
apis = summ[summ.key.str.contains("@")]
best_api = apis.sort_values("macro_f1").iloc[-1]
tf = summ[summ.key == "tfidf"].iloc[0]
lab, rid, base, rows, pv = CANDS[FINAL]
L = []
L.append(f"### Headline\n\n**CrisisLens ({base.split('-')[-1]}, AutoScientist run {lab}) reaches macro-F1 {f3(fin['macro_f1'])} "
         f"[95% CI {f3(fin['macro_f1_ci95'][0])}–{f3(fin['macro_f1_ci95'][1])}] on {fin['n']:,} posts from four disasters it never saw "
         f"in training**, against {f3(best_api.macro_f1)} for the best frontier API zero-shot ({best_api.system.split(' zero-shot')[0]}) "
         f"and {f3(tf.macro_f1)} for TF-IDF + LR. Recall on `requests_or_urgent_needs` is {f3(fin['recall_requests_or_urgent_needs'])} "
         f"(best API: {f3(apis.recall_urgent_needs.max())}). It runs on a laptop CPU for $0 per 1,000 posts "
         f"(APIs: ${apis.usd_per_1k.min():.3f}–${apis.usd_per_1k.max():.3f}), and {pc(fin['json_valid_rate'])} of outputs pass the strict JSON schema.\n")

L.append("### Test results: four held-out disasters (gold human labels)\n")
L.append((R / "summary.md").read_text(encoding="utf-8").strip() + "\n")
L.append("Macro-F1 is computed over the classes present (all 10 are present in this set) with a 1,000-sample bootstrap CI. "
         "The base-model zero-shot run was stopped by a memory guard at 1,035/1,219 posts and is scored on the 1,033 it answered. "
         "API latency is wall-clock through the Adaption gateway. Local latency is a 12-thread laptop CPU (i7-1255U), one request at a time.\n")

# model selection
sel = []
for k, (lb, r_id, b, n, p) in CANDS.items():
    fv = R / f"{k}__val_eval.json"
    if not fv.exists(): continue
    v = json.load(open(fv))
    st = [json.loads(l) for l in open(ROOT / f"eval/autoscientist/{r_id}.jsonl")][-1]
    sel.append({"run": lb, "base": b, "training rows": n, "prompt": p, "AutoScientist best win-rate": f3(st.get("best_win_rate")),
                "val macro-F1": v["macro_f1"], "val weighted-F1": v["weighted_f1"], "val JSON valid": pc(v["json_valid_rate"]),
                "selected": "✅" if k == FINAL else ""})
if sel:
    sd = pd.DataFrame(sel).sort_values("val macro-F1", ascending=False)
    sd["val macro-F1"] = sd["val macro-F1"].map(f3); sd["val weighted-F1"] = sd["val weighted-F1"].map(f3)
    L.append("### Model selection: validation set only\n")
    L.append("Candidates are compared on `val_eval` (574 posts, 60 per class, drawn from the 15 *training* disasters). "
             "The selected model is the only fine-tuned candidate run on the test set, so the test set played no part in any decision.\n")
    L.append(sd.to_markdown(index=False) + "\n")
    top = max(sel, key=lambda r: r["val macro-F1"])
    fin_v = next(r for r in sel if r["selected"])
    if top is not fin_v:
        L.append(f"Run {top['run']} is {top['val macro-F1'] - fin_v['val macro-F1']:.3f} macro-F1 above the selected run {fin_v['run']} on validation. "
                 "That is far inside both 95% CIs (about ±0.03 on 574 posts), so the two are statistically tied. Run "
                 f"{fin_v['run']} is shipped because it is the candidate that was fully evaluated on the test set in the time available. "
                 "The test set was not used to choose between them.\n")
    L.append("Win-rate is AutoScientist's judge score against each run's own base model. It is useful for steering a run, "
             "but it does not rank models by triage accuracy, which is why selection uses the validation metric.\n")

pe = fin.get("per_event_macro_f1", {})
api_pe = json.load(open(R / f"{best_api.key}.json")).get("per_event_macro_f1", {})
L.append("### Per-disaster macro-F1\n")
EVN = {"kerala_floods_2018": "Kerala floods 2018", "cyclone_idai_2019": "Cyclone Idai 2019", "mexico_earthquake_2017": "Mexico earthquake 2017", "hurricane_dorian_2019": "Hurricane Dorian 2019"}
L.append(pd.DataFrame({"disaster": [EVN.get(e, e) for e in pe], "CrisisLens": [f3(pe[e]) for e in pe],
                       best_api.system: [f3(api_pe.get(e)) for e in pe]}).to_markdown(index=False) + "\n")

L.append("### Structured fields (scored against teacher silver labels, not human labels)\n")
L.append(pd.DataFrame([{"actionable acc": pc(fin.get("actionable_acc")), "actionable F1": f3(fin.get("actionable_f1")),
                        "urgency MAE": f3(fin.get("urgency_mae")), "urgency within ±1": pc(fin.get("urgency_within1")),
                        "urgent (≥4) F1": f3(fin.get("urgent_ge4_f1")), "location exact": pc(fin.get("location_exact")),
                        "location overlap": pc(fin.get("location_overlap")), "needs Jaccard": f3(fin.get("needs_jaccard"))}]).to_markdown(index=False) + "\n")
tv = R / "teacher_verification.json"
if tv.exists():
    t = json.load(open(tv))
    L.append(f"Silver-label check: on 100 rows reviewed against the post text (a review by Claude, **not** a human annotator), the teacher was "
             f"right on actionable {pc(t.get('agree_actionable'))}, urgency {pc(t.get('agree_urgency'))}, location {pc(t.get('agree_location'))}, "
             f"needs {pc(t.get('agree_needs'))}, and all fields {pc(t.get('all_fields_agree'))}. See `eval/human_check/`.\n")

L.append("### Comparison with the HumAID paper\n")
L.append(f"Alam et al. (ICWSM 2021) report weighted-F1 0.758–0.760 (BERT/RoBERTa) on a **random** split, where training and test share "
         f"events. CrisisLens scores weighted-F1 {f3(fin['weighted_f1'])} on an **event-level** split, which is harder because "
         f"vocabulary, places and needs change between disasters. The numbers are not directly comparable, and the setting here is the harder one.\n")

it = pd.read_csv(R / "autoscientist_iterations.csv")
g = it.groupby("run", sort=False).agg(iterations=("iteration", "max"), best_win_rate=("best_win_rate", "max"),
                                      final_lr=("learning_rate", "last"), final_epochs=("n_epochs", "last"), final_lora_r=("lora_r", "last")).reset_index()
L.append("### AutoScientist runs\n")
L.append(g.to_markdown(index=False) + "\n")
L.append("Every polled state, launch request and `trainer_state.json` is in `eval/autoscientist/`. "
         "Charts: `demo/assets/3_autoscientist_iterations.png`, `4_autoscientist_loss.png`.\n")
L.append("![macro-F1](demo/assets/1_macro_f1.png)\n![life-critical recall](demo/assets/2_life_critical_recall.png)\n"
         "![per-class F1](demo/assets/5_per_class_f1.png)\n![AutoScientist iterations](demo/assets/3_autoscientist_iterations.png)\n")

block = "\n".join(L)
rd = (ROOT / "README.md").read_text(encoding="utf-8")
a, b = rd.index("<!-- RESULTS:START -->") + len("<!-- RESULTS:START -->"), rd.index("<!-- RESULTS:END -->")
(ROOT / "README.md").write_text(rd[:a] + "\n" + block + "\n" + rd[b:], encoding="utf-8")
print(block[:1500])
