"""Build the CrisisLens demo video: narrated walkthrough with burned-in captions and an .srt file.

  python -I demo/video/capture.py   WORK/shots      # 1. screenshots of the real console
  python -I demo/video/make_video.py WORK OUT.mp4   # 2. narration + cards + captions -> MP4 (+ OUT.srt)

Voice: edge-tts (Microsoft neural voice); falls back to the offline Windows voice if it is unavailable.
Every number in the script comes from eval/results (README); keep them in sync if results change.
"""
import asyncio, re, subprocess, sys, time, wave
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H = 1920, 1080
BG, PANEL, LINE = (17, 17, 17), (30, 30, 30), (52, 52, 52)
INK, MUTED, RED, BLUE, GREEN = (240, 240, 240), (160, 160, 160), (229, 72, 77), (59, 130, 246), (52, 199, 89)
FONTS = Path("C:/Windows/Fonts")
def font(name, size):
    for n in ([name] if isinstance(name, str) else name):
        try: return ImageFont.truetype(str(FONTS / n), size)
        except OSError: pass
    return ImageFont.load_default(size)
REG, SEMI, BOLD, MONO = ["segoeui.ttf"], ["seguisb.ttf", "segoeuib.ttf"], ["segoeuib.ttf"], ["consola.ttf"]

# ---------------------------------------------------------------- script
SCRIPT = [
    ("card:title", "CrisisLens: disaster triage with a small AI model, trained with Adaption Labs AutoScientist."),
    ("card:problem", "When a disaster strikes, response teams are flooded with thousands of social-media posts. "
                     "Hidden among them are people asking for rescue, water, or medicine. Reading every post by hand is far too slow."),
    ("card:record", "CrisisLens reads each post and returns a strict JSON triage record: the humanitarian category, whether it is actionable, "
                    "an urgency score from one to five, the location, the needs, and a short rationale."),
    ("console_empty", "This is the CrisisLens triage console. It replays real posts from four disasters that the model never saw during training."),
    ("feed", "As posts arrive, each triage record joins a priority queue ranked by urgency, so the most critical posts rise to the top. "
             "The dashboard tracks volume, actionable posts, life-safety posts, and the category mix."),
    ("filter_critical", "Responders can filter the queue to critical posts only, with urgency four or higher."),
    ("detail", "Selecting a post opens its full record, with the location, needs, rationale, and the model's latency on a laptop CPU."),
    ("analyze_ex0", "The Analyze tab triages a single post. Here, families trapped by flood water are flagged as an urgent, "
                    "life-threatening request, with the location and needs extracted."),
    ("analyze_ex2", "A hospital running out of insulin and power is marked actionable, with medical, water, and power needs."),
    ("analyze_ex7", "And a message of sympathy is correctly given the lowest urgency, so it does not distract responders."),
    ("card:data", "The model was trained on HumAID: 76,484 human-labelled tweets from 19 disasters. We removed duplicates and split the data by disaster. "
                  "Four disasters were held out completely: the Kerala floods, Cyclone Idai, the Mexico earthquake, and Hurricane Dorian."),
    ("card:training", "We fine-tuned Qwen 3.5 2B with LoRA, using Adaption Labs AutoScientist. The default recipe failed, "
                      "so we fixed the prompt and the learning rate, and AutoScientist then tuned the recipe itself."),
    ("eval_chIter", "In total we ran eight AutoScientist training runs, testing 2, 4, and 9 billion parameter models and up to 49 thousand training rows."),
    ("card:select", "Run five is our best model. It shares the top validation score, and it is the model we evaluated on the test set. "
                    "The test set was used only once, so the results are honest."),
    ("eval_top", "On 1,219 posts from the unseen disasters, CrisisLens reaches a macro-F1 of 0.715. The best frontier API model reaches 0.606, "
                 "a classic TF-IDF model 0.662, and the untuned base model only 0.427."),
    ("eval_chRecall", "For urgent requests for help, CrisisLens finds 76 percent of them. The frontier API finds only 32 percent."),
    ("eval_sysTable", "It is also practical: 99.9 percent of its outputs are valid JSON, and it runs on a laptop CPU in about five seconds per post, at zero cost."),
    ("card:limits", "We are open about its limits. Its most common mistake is treating donation offers as urgent requests. "
                    "The hosted demo replays recorded outputs, because live hosting needs a paid plan. Live triage runs locally."),
    ("card:end", "CrisisLens: a small, free model that beats frontier APIs at disaster triage. The demo, the model, and all the code are open. Thank you."),
]
SAY = {"Qwen 3.5 2B": "Qwen three point five, 2 B", "TF-IDF": "T F I D F", "macro-F1": "macro F1", "76,484": "76 thousand 484",
       "1,219": "twelve hundred and nineteen", "HumAID": "Hum-aid"}
def spoken(t):
    for a, b in SAY.items(): t = t.replace(a, b)
    return t

# ---------------------------------------------------------------- cards
def base(kicker=None, title=None):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.ellipse((120, 96, 176, 152), fill=RED); d.ellipse((134, 110, 162, 138), outline=(255, 255, 255), width=5); d.ellipse((143, 119, 153, 129), fill=(255, 255, 255))
    d.text((196, 98), "CrisisLens", font=font(BOLD, 34), fill=INK); d.text((196, 134), "Disaster-response triage", font=font(REG, 22), fill=MUTED)
    if kicker: d.text((120, 205), kicker.upper(), font=font(SEMI, 28), fill=BLUE)
    if title: d.text((120, 245), title, font=font(BOLD, 72), fill=INK)
    return im, d

def bullets(d, items, y, size=40, gap=30, x=120, wrapw=1560):
    f = font(REG, size)
    for it in items:
        d.ellipse((x, y + size * 0.45, x + 14, y + size * 0.45 + 14), fill=RED)
        for ln in wrap(d, it, f, wrapw): d.text((x + 40, y), ln, font=f, fill=INK); y += int(size * 1.35)
        y += gap
    return y

def wrap(d, text, f, maxw):
    out, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= maxw: cur = t
        else: out.append(cur); cur = w
    return out + ([cur] if cur else [])

def card(kind):
    if kind == "title":
        im, d = base()
        d.text((120, 270), "CrisisLens", font=font(BOLD, 140), fill=INK)
        d.text((126, 450), "Disaster triage with a 2B model trained by Adaption Labs AutoScientist", font=font(SEMI, 46), fill=INK)
        y = 550
        x = 126
        for k, v in [("0.715", "macro-F1 on unseen disasters"), ("0.76", "urgent-needs recall"), ("$0", "per 1,000 posts")]:
            d.rounded_rectangle((x, y, x + 500, y + 170), 18, fill=PANEL, outline=LINE, width=2)
            d.text((x + 30, y + 20), k, font=font(BOLD, 72), fill=INK); d.text((x + 32, y + 115), v, font=font(REG, 28), fill=MUTED)
            x += 530
        d.text((126, 770), "Vishal Verma", font=font(SEMI, 36), fill=INK)
        return im
    if kind == "problem":
        im, d = base("The problem", "Thousands of posts. A few are life or death.")
        bullets(d, ["In a disaster, response desks are flooded with social-media posts.",
                    "Requests for rescue, water and medicine are buried among news, sympathy and noise.",
                    "Reading every post by hand is too slow, and frontier AI APIs cost money for every post."], 400)
        return im
    if kind == "record":
        im, d = base("The solution", "One post in, one triage record out")
        d.rounded_rectangle((120, 420, 900, 840), 18, fill=PANEL, outline=LINE, width=2)
        d.text((150, 445), "SOCIAL-MEDIA POST", font=font(SEMI, 24), fill=MUTED)
        post = "Water entered the ground floor, 3 families with small kids stuck on the terrace of Shanti Apartments, Aluva. Need boats urgently, phones dying. #KeralaFloods"
        y = 495
        for ln in wrap(d, post, font(REG, 36), 700): d.text((150, y), ln, font=font(REG, 36), fill=INK); y += 50
        d.text((935, 580), "→", font=font(BOLD, 80), fill=RED)
        d.rounded_rectangle((1030, 420, 1800, 840), 18, fill=(12, 18, 28), outline=LINE, width=2)
        js = ['{', '  "category": "requests_or_urgent_needs",', '  "actionable": true,', '  "urgency": 5,', '  "location": "Aluva",',
              '  "needs": ["rescue", "communication"],', '  "rationale": "Urgent need for boats', '     and communication for trapped families."', '}']
        y = 440
        for ln in js: d.text((1060, y), ln, font=font(MONO, 27), fill=(150, 200, 255) if ":" in ln else INK); y += 46
        d.text((120, 355), "Real recorded output of the fine-tuned model (7.8 s on a laptop CPU)", font=font(REG, 28), fill=MUTED)
        return im
    if kind == "data":
        im, d = base("Data", "HumAID, split by disaster")
        x = 120
        for k, v in [("76,484", "human-labelled tweets"), ("19", "real disasters"), ("4", "disasters held out entirely"), ("1,219", "test posts, human labels")]:
            d.rounded_rectangle((x, 380, x + 400, 560), 18, fill=PANEL, outline=LINE, width=2)
            d.text((x + 30, 400), k, font=font(BOLD, 72), fill=INK); d.text((x + 32, 500), v, font=font(REG, 28), fill=MUTED); x += 420
        bullets(d, ["Held-out test disasters: Kerala floods 2018 · Cyclone Idai 2019 · Mexico earthquake 2017 · Hurricane Dorian 2019",
                    "Exact and near duplicates removed; human category labels kept; extra fields (urgency, location, needs) from a teacher LLM, scored separately"], 610, size=32, gap=22)
        return im
    if kind == "training":
        im, d = base("Training", "Fine-tuned with Adaption Labs AutoScientist")
        bullets(d, ["Base model: Qwen3.5-2B + LoRA (rank 32), instruction fine-tuning on 17,784 examples",
                    "The default recipe failed (invented labels), so we listed the label set in the prompt and raised the learning rate",
                    "AutoScientist then refined the recipe itself: learning rate 1.5e-4, 4 epochs",
                    "Served as GGUF with llama.cpp on a laptop CPU, no GPU needed"], 390, size=38, gap=22)
        return im
    if kind == "select":
        im, d = base("Model selection", "Run #5 is our best model")
        rows = [("Run", "Model · training rows", "Validation macro-F1", ""), ("#7", "2B · 37,267 rows", "0.793", "tie"),
                ("#5", "2B · 17,784 rows", "0.791", "shipped")]
        y = 390
        for i, r in enumerate(rows):
            f = font(SEMI if i == 0 else REG, 36); col = MUTED if i == 0 else INK
            if i == 2: d.rounded_rectangle((110, y - 14, 1810, y + 66), 14, fill=(20, 40, 70))
            for x, t in zip((140, 330, 1000, 1500), r): d.text((x, y), t, font=font(BOLD, 36) if i == 2 else f, fill=GREEN if t == "shipped" else col)
            y += 90
        bullets(d, ["Selected on a validation set drawn from the 15 training disasters (574 posts); the 0.002 gap is a statistical tie",
                    "The test set was used once, only for the shipped model, so no test leakage"], y + 30, size=32, gap=22)
        return im
    if kind == "limits":
        im, d = base("Honest limits", "What it still gets wrong")
        bullets(d, ["Most common error: offers of donations or volunteering escalated to urgent requests",
                    "Harder disasters score lower: Cyclone Idai and Hurricane Dorian macro-F1 0.62 vs 0.72 for Kerala",
                    "The hosted demo replays recorded model outputs (live Spaces need a paid plan); live triage of new text runs locally with two commands"], 400, size=38, gap=26)
        return im
    if kind == "end":
        im, d = base("CrisisLens", "Small, free, and better than frontier APIs at triage")
        y = 400
        for k, v in [("Demo", "huggingface.co/spaces/Vishal-2109/crisislens"), ("Model", "huggingface.co/Vishal-2109/crisislens-qwen3.5-2b-lora"),
                     ("Code", "github.com/2ViShAlv1/crisislens")]:
            d.text((120, y), k, font=font(SEMI, 40), fill=MUTED); d.text((340, y), v, font=font(REG, 40), fill=INK); y += 90
        d.text((120, 690), "Built by Vishal Verma with Adaption Labs AutoScientist", font=font(SEMI, 36), fill=INK)
        return im
    raise ValueError(kind)

# ---------------------------------------------------------------- captions
CAP = font(SEMI, 44)
def chunks(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip()); out = []
    for p in parts:
        while len(p) > 100:
            cut = max((m.end() for m in re.finditer(r"[,:]\s", p[:100])), default=0) or p.rfind(" ", 0, 95)
            out.append(p[:cut].strip()); p = p[cut:].strip()
        out.append(p)
    return [c for c in out if c]

def captioned(img, text):
    im = img.convert("RGBA"); ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    lines = wrap(d, text, CAP, 1640); lh = 60; bh = lh * len(lines) + 36; y0 = H - bh - 40
    widest = max(d.textlength(l, font=CAP) for l in lines)
    d.rounded_rectangle(((W - widest) / 2 - 32, y0, (W + widest) / 2 + 32, y0 + bh), 16, fill=(0, 0, 0, 205))
    for i, l in enumerate(lines):
        d.text(((W - d.textlength(l, font=CAP)) / 2, y0 + 16 + i * lh), l, font=CAP, fill=(255, 255, 255, 255))
    return Image.alpha_composite(im, ov).convert("RGB")

# ---------------------------------------------------------------- audio
async def _edge(text, path):
    import edge_tts
    await edge_tts.Communicate(text, "en-US-AndrewNeural", rate="+4%").save(str(path))
def tts(text, wav):
    mp3 = wav.with_suffix(".mp3"); src = None
    if mp3.exists() and mp3.stat().st_size > 1000 and mp3.with_suffix(".txt").exists() and mp3.with_suffix(".txt").read_text(encoding="utf-8") == text:
        src = mp3  # cached clip for identical text
    for attempt in range(6):
        if src: break
        try:
            asyncio.run(_edge(text, mp3)); src = mp3; mp3.with_suffix(".txt").write_text(text, encoding="utf-8")
        except Exception as e:
            print(f"edge-tts attempt {attempt + 1} failed: {e}"); time.sleep(3 + 3 * attempt)
    if src is None:
        print("edge-tts unavailable, using Windows voice")
        src = wav.with_suffix(".sapi.wav"); t = text.replace("'", "''")
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        f"$s.SelectVoice('Microsoft Zira Desktop'); $s.SetOutputToWaveFile('{src}'); $s.Speak('{t}'); $s.Dispose()"], check=True)
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "24000", str(wav)], check=True)
    with wave.open(str(wav)) as w: return w.getnframes() / w.getframerate()

def srt_t(s):
    ms = int(round(s * 1000)); return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

# ---------------------------------------------------------------- build
def main(work, out):
    work, out = Path(work), Path(out); shots = work / "shots"; fr = work / "frames"; au = work / "audio"
    fr.mkdir(parents=True, exist_ok=True); au.mkdir(exist_ok=True); out.parent.mkdir(parents=True, exist_ok=True)
    LEAD, GAP, TAIL = 0.5, 0.55, 1.2
    entries, subs, wavs, t, n = [], [], [], 0.0, 0
    for si, (vis, text) in enumerate(SCRIPT):
        dur = tts(spoken(text), au / f"{si:02d}.wav"); wavs.append((au / f"{si:02d}.wav", dur))
        seg = LEAD + dur + (TAIL if si == len(SCRIPT) - 1 else GAP)
        if vis == "feed": imgs = [(shots / f"feed_{i:02d}.png", 0.45) for i in range(28)] + [(shots / "feed_full.png", 99)]
        elif vis.startswith("card:"): imgs = [(card(vis[5:]), 99)]
        else: imgs = [(shots / f"{vis}.png", 99)]
        cs = chunks(text); total = sum(len(c) for c in cs); bounds = [0.0]
        for c in cs: bounds.append(bounds[-1] + dur * len(c) / total)
        bounds = [b + LEAD for b in bounds]; bounds[0] = 0.0; bounds[-1] = seg
        for ci, c in enumerate(cs):
            subs.append((t + max(bounds[ci], LEAD - 0.1), t + min(bounds[ci + 1], LEAD + dur + 0.3) if ci == len(cs) - 1 else t + bounds[ci + 1], c))
        # cut the segment timeline at every image change and caption change
        cuts, acc = {0.0, seg}, 0.0
        for _, d_ in imgs[:-1]: acc += d_; cuts.add(round(min(acc, seg), 3))
        cuts |= {round(b, 3) for b in bounds if 0 < b < seg}; cuts = sorted(c for c in cuts if c <= seg)
        cache = {}
        for a, b in zip(cuts, cuts[1:]):
            if b - a < 0.02: continue
            acc, idx = 0.0, len(imgs) - 1
            for k, (_, d_) in enumerate(imgs[:-1]):
                if a < acc + d_ - 1e-6: idx = k; break
                acc += d_
            ci = next((i for i in range(len(cs)) if a < bounds[i + 1] - 1e-6), len(cs) - 1)
            key = (idx, ci)
            if key not in cache:
                src = imgs[idx][0]; img = src if isinstance(src, Image.Image) else Image.open(src).convert("RGB")
                if img.size != (W, H): img = img.resize((W, H), Image.LANCZOS)
                p = fr / f"f{n:04d}.png"; n += 1; captioned(img, cs[ci]).save(p); cache[key] = p
            entries.append((cache[key], b - a))
        t += seg
        print(f"segment {si + 1}/{len(SCRIPT)}  {seg:5.1f}s  total {t:6.1f}s", flush=True)

    # audio track: lead silence + speech + gap per segment, exactly matching the video timeline
    full = work / "narration.wav"
    with wave.open(str(full), "wb") as o:
        o.setnchannels(1); o.setsampwidth(2); o.setframerate(24000)
        for si, (wp, dur) in enumerate(wavs):
            o.writeframes(b"\0\0" * int(24000 * LEAD))
            with wave.open(str(wp)) as w: o.writeframes(w.readframes(w.getnframes()))
            o.writeframes(b"\0\0" * int(24000 * (TAIL if si == len(wavs) - 1 else GAP)))
    lst = work / "frames.txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p, d_ in entries: f.write(f"file '{p.as_posix()}'\nduration {d_:.3f}\n")
        f.write(f"file '{entries[-1][0].as_posix()}'\n")
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(full),
                    "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out)], check=True)
    with open(out.with_suffix(".srt"), "w", encoding="utf-8") as f:
        for i, (a, b, c) in enumerate(subs, 1): f.write(f"{i}\n{srt_t(a)} --> {srt_t(b)}\n{c}\n\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {t:.0f} s) and {out.with_suffix('.srt').name}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
