"""CrisisLens — Gradio demo (HF Spaces).

Tabs: (1) single-post analyzer, (2) live crisis feed replaying held-out test posts into an
urgency-ranked triage queue, (3) metrics. Live inference goes to an OpenAI-compatible endpoint
(CRISISLENS_BASE_URL / CRISISLENS_MODEL / CRISISLENS_API_KEY). If it is unset or down, the app
falls back to cached model outputs (demo/cache/*.jsonl), so it survives an API outage.
"""
import json, os, re, time
from pathlib import Path
import gradio as gr
import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "cache"
FT_INSTRUCTION = ("CrisisLens triage. Classify this disaster-time social-media post and return only the JSON record "
                  "{category, actionable, urgency, location, needs, rationale}.")
COLORS = {
    "requests_or_urgent_needs": "#d7263d", "injured_or_dead_people": "#7a0019", "missing_or_found_people": "#f46036",
    "displaced_people_and_evacuations": "#f9a03f", "infrastructure_and_utility_damage": "#8d6a9f",
    "caution_and_advice": "#e2c044", "rescue_volunteering_or_donation_effort": "#2e86ab",
    "other_relevant_information": "#5c946e", "sympathy_and_support": "#9aa5b1", "not_humanitarian": "#c5c9cf",
}

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []

FEED = load_jsonl(CACHE / "feed.jsonl")          # [{id,event,text,gold,pred:{...}}]
METRICS = json.load(open(CACHE / "metrics.json")) if (CACHE / "metrics.json").exists() else {}

def parse(s):
    s = re.sub(r"<think>.*?</think>", "", s or "", flags=re.S)
    m = re.search(r"\{.*\}", s, flags=re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None

def live_infer(text):
    url, model = os.environ.get("CRISISLENS_BASE_URL"), os.environ.get("CRISISLENS_MODEL")
    if not url or not model:
        return None, "live model not configured"
    try:
        from openai import OpenAI
        cli = OpenAI(base_url=url, api_key=os.environ.get("CRISISLENS_API_KEY", "none"), timeout=60)
        t0 = time.time()
        r = cli.chat.completions.create(model=model, temperature=0, max_tokens=200,
                                        messages=[{"role": "user", "content": f"{FT_INSTRUCTION}\nPost: {text}"}])
        return parse(r.choices[0].message.content), f"live · {time.time() - t0:.2f}s"
    except Exception as e:
        return None, f"live model error: {type(e).__name__}"

def badge(cat):
    c = COLORS.get(cat, "#888")
    return f'<span style="background:{c};color:white;padding:2px 8px;border-radius:10px;font-size:12px">{cat}</span>'

def render(o, status):
    if not o:
        return f"<p><b>No result</b> ({status}). Try one of the cached examples below.</p>"
    u = int(o.get("urgency", 0) or 0)
    bar = "🟥" * u + "⬜" * (5 - u)
    return (f"<div style='font-family:system-ui'>{badge(o.get('category'))}"
            f"<p><b>Urgency</b> {bar} {u}/5 &nbsp; <b>Actionable</b> {'✅' if o.get('actionable') else '—'}</p>"
            f"<p><b>Location</b> {o.get('location') or '—'} &nbsp; <b>Needs</b> {', '.join(o.get('needs') or []) or '—'}</p>"
            f"<p><i>{o.get('rationale', '')}</i></p><p style='color:#888;font-size:12px'>{status}</p></div>")

def analyze(text):
    text = (text or "").strip()
    if not text:
        return "<p>Paste a post.</p>", ""
    o, status = live_infer(text)
    if o is None:
        hit = next((f for f in FEED if f["text"].strip() == text), None)
        if hit:
            o, status = hit["pred"], "cached model output (live endpoint unavailable)"
    return render(o, status), json.dumps(o, indent=2, ensure_ascii=False) if o else ""

def queue_html(items):
    items = sorted(items, key=lambda f: (-int(f["pred"].get("urgency", 0)), -bool(f["pred"].get("actionable"))))
    rows = "".join(
        f"<tr><td style='text-align:center;font-weight:700'>{f['pred'].get('urgency')}</td><td>{badge(f['pred'].get('category'))}</td>"
        f"<td>{'✅' if f['pred'].get('actionable') else ''}</td><td>{f['pred'].get('location') or ''}</td>"
        f"<td style='font-size:13px'>{f['text'][:180]}</td><td style='font-size:11px;color:#888'>{f['event']}</td></tr>"
        for f in items[:40])
    return ("<table style='width:100%;border-collapse:collapse'><tr><th>Urg</th><th>Category</th><th>Act</th><th>Location</th>"
            f"<th>Post</th><th>Event</th></tr>{rows}</table><p style='color:#888'>{len(items)} posts triaged</p>")

def feed_step(state, event):
    pool = [f for f in FEED if event == "all" or f["event"] == event]
    i = len(state)
    if i < len(pool):
        state = state + [pool[i]]
    return state, queue_html(state)

with gr.Blocks(title="CrisisLens", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛰️ CrisisLens — disaster-response triage LLM\nFine-tuned with **AutoScientist** (Adaption Labs) on HumAID. "
                "Evaluated on 4 disasters never seen in training (Kerala floods 2018, Cyclone Idai 2019, Mexico earthquake 2017, Hurricane Dorian 2019).\n\n"
                "<small>Posts: HumAID (Alam et al., ICWSM 2021), CC BY-NC-SA 4.0. Decision support only — a human reviews every item.</small>")
    with gr.Tab("Analyze a post"):
        inp = gr.Textbox(lines=4, label="Social-media post", placeholder="e.g. 3 families trapped on a rooftop near Aluva, water rising, need boats")
        btn = gr.Button("Triage", variant="primary")
        out_html = gr.HTML(); out_json = gr.Code(language="json", label="Raw JSON")
        btn.click(analyze, inp, [out_html, out_json])
        gr.Examples([f["text"] for f in FEED[:8]], inp)
    with gr.Tab("Live crisis feed"):
        gr.Markdown("Replays held-out test posts (never seen in training) through the model; the queue stays ranked by urgency.")
        ev = gr.Dropdown(["all"] + sorted({f["event"] for f in FEED}), value="all", label="Disaster")
        state = gr.State([]); q = gr.HTML()
        timer = gr.Timer(1.0, active=False)
        with gr.Row():
            start = gr.Button("▶ Start", variant="primary"); stop = gr.Button("⏸ Pause"); reset = gr.Button("↺ Reset")
        start.click(lambda: gr.Timer(active=True), None, timer)
        stop.click(lambda: gr.Timer(active=False), None, timer)
        reset.click(lambda: ([], ""), None, [state, q])
        ev.change(lambda: ([], ""), None, [state, q])
        timer.tick(feed_step, [state, ev], [state, q])
    with gr.Tab("Metrics"):
        if METRICS.get("table_md"):
            gr.Markdown(METRICS["table_md"])
        for img in sorted((HERE / "assets").glob("*.png")):
            gr.Image(str(img), label=img.stem, show_label=True)

if __name__ == "__main__":
    demo.launch()
