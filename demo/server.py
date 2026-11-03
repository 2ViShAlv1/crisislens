"""CrisisLens demo backend (FastAPI): serves the web console and proxies triage requests to the
fine-tuned model behind an OpenAI-compatible llama.cpp server.

  CRISISLENS_BASE_URL  e.g. http://127.0.0.1:8080/v1   (unset -> cached outputs only)
  CRISISLENS_MODEL     model alias passed to the server (default "crisislens")
  uvicorn server:app --host 0.0.0.0 --port 7860

If the live model is unreachable, /api/analyze falls back to cached outputs for known posts
(held-out feed + examples), and says so in `source`. Nothing is fabricated for unknown text.
"""
import asyncio, json, os, re, time
from pathlib import Path
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
BASE = (os.environ.get("CRISISLENS_BASE_URL") or "").rstrip("/")
MODEL = os.environ.get("CRISISLENS_MODEL", "crisislens")
CLASSES = ["caution_and_advice", "displaced_people_and_evacuations", "infrastructure_and_utility_damage",
           "injured_or_dead_people", "missing_or_found_people", "not_humanitarian", "other_relevant_information",
           "requests_or_urgent_needs", "rescue_volunteering_or_donation_effort", "sympathy_and_support"]
# identical to the training prompt (src/sft_format.py FT_INSTRUCTION_V2)
INSTRUCTION = ("CrisisLens triage. Return only a JSON object with keys category, actionable (true/false), "
               "urgency (integer 1-5, 5=life-threatening), location (place written in the post or null), "
               "needs (list), rationale (<=20 words). category must be one of: " + ", ".join(CLASSES) + ".")

def _load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []
CACHED = {r["text"].strip(): r["pred"] for r in _load_jsonl(HERE / "cache/feed.jsonl")}
if (HERE / "cache/examples.json").exists():
    CACHED.update({e["text"].strip(): e["pred"] for e in json.load(open(HERE / "cache/examples.json", encoding="utf-8")) if e.get("pred")})

LIVE_SLOTS = asyncio.Semaphore(2)   # CPU inference: keep concurrent generations low
PENDING, MAX_PENDING = 0, 8          # requests waiting or running; beyond this answer "busy" instead of queueing forever
app = FastAPI(title="CrisisLens", docs_url=None, redoc_url=None)

class Req(BaseModel):
    text: str = Field(min_length=1, max_length=1200)

def parse(s):
    s = re.sub(r"<think>.*?</think>", "", s or "", flags=re.S)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m: return None
    try:
        o = json.loads(m.group(0))
    except Exception:
        return None
    return o if isinstance(o, dict) else None

def valid(o):
    try:
        return (o.get("category") in CLASSES and isinstance(o.get("actionable"), bool) and 1 <= int(o.get("urgency")) <= 5
                and (o.get("location") is None or isinstance(o.get("location"), str)) and isinstance(o.get("needs"), list))
    except Exception:
        return False

@app.get("/api/health")
async def health():
    if not BASE:
        return {"live": False, "reason": "no model endpoint configured"}
    try:
        async with httpx.AsyncClient(timeout=2.5) as c:
            r = await c.get(BASE.rsplit("/v1", 1)[0] + "/health")
        ok = r.status_code == 200 and "ok" in r.text
        return {"live": ok, "reason": None if ok else f"model server status {r.status_code} (loading?)"}
    except Exception as e:
        return {"live": False, "reason": type(e).__name__}

@app.post("/api/analyze")
async def analyze(req: Req):
    global PENDING
    text = req.text.strip()
    err = "no model endpoint configured"
    if BASE and PENDING >= MAX_PENDING:
        err = "model busy (queue full)"
    elif BASE:
        PENDING += 1
        try:
            async with LIVE_SLOTS:
                t0 = time.perf_counter()
                async with httpx.AsyncClient(timeout=120) as c:
                    r = await c.post(f"{BASE}/chat/completions", json={
                        "model": MODEL, "temperature": 0, "max_tokens": 200,
                        "messages": [{"role": "user", "content": f"{INSTRUCTION}\nPost: {text}"}]})
                r.raise_for_status()
                j = r.json(); raw = j["choices"][0]["message"]["content"]; o = parse(raw)
                return {"result": o, "valid": bool(o and valid(o)), "raw": raw, "source": "live",
                        "latency_s": round(time.perf_counter() - t0, 2), "usage": j.get("usage")}
        except Exception as e:
            err = f"live model error: {type(e).__name__}"
        finally:
            PENDING -= 1
    if text in CACHED:
        return {"result": CACHED[text], "valid": True, "raw": None, "source": "cached", "latency_s": None, "note": err}
    raise HTTPException(503, detail=f"{err}. Try one of the example posts (cached outputs are available for them).")

app.mount("/cache", StaticFiles(directory=HERE / "cache"), name="cache")
app.mount("/static", StaticFiles(directory=HERE / "web"), name="static")

@app.get("/")
async def index():
    return FileResponse(HERE / "web/index.html", headers={"Cache-Control": "no-cache"})
