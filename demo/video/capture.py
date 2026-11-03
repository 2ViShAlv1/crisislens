"""Capture 1920x1080 screenshots of the CrisisLens console for the demo video.

  python -I demo/video/capture.py OUT_DIR

Stages the static site (scripts/publish_hf.py --stage), serves it on localhost and drives the installed
Chrome with Playwright. Every screen shows real recorded model outputs; nothing is mocked.
"""
import functools, http.server, sys, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scripts.publish_hf import stage_static
from playwright.sync_api import sync_playwright

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
site = stage_static(out.parent / "site")
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
handler.log_message = lambda *a, **k: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8765), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()

def shot(page, name):
    page.screenshot(path=str(out / f"{name}.png")); print("shot", name, flush=True)

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    page = b.new_context(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5,
                         color_scheme="dark").new_page()
    page.goto("http://127.0.0.1:8765/index.html"); page.wait_for_timeout(2500)
    shot(page, "console_empty")

    page.select_option("#speed", "450"); page.click("#playBtn")
    for i in range(28):  # live feed: one frame every 0.5 s
        page.wait_for_timeout(500); shot(page, f"feed_{i:02d}")
    page.wait_for_timeout(4000); page.click("#playBtn")  # pause with a full queue
    page.wait_for_timeout(400); shot(page, "feed_full")
    page.click("#queueFilter button[data-f='critical']"); page.wait_for_timeout(500); shot(page, "filter_critical")
    page.locator("#queue .qrow").first.click(); page.wait_for_timeout(600); shot(page, "detail")
    page.locator("#detail").scroll_into_view_if_needed(); page.wait_for_timeout(300); shot(page, "detail_scrolled")

    page.click(".tabs button[data-view='analyze']"); page.wait_for_timeout(600); shot(page, "analyze_empty")
    for i in (0, 2, 7):
        page.locator("#examples .ex").nth(i).click(); page.wait_for_timeout(300)
        if page.locator("#postIn").input_value().strip() and page.locator("#runBtn").is_enabled():
            page.click("#runBtn")
        page.wait_for_timeout(1500); shot(page, f"analyze_ex{i}")

    page.click(".tabs button[data-view='eval']"); page.wait_for_timeout(1500); shot(page, "eval_top")
    for el in ["chMacro", "chRecall", "chIter", "chClass", "chEvent", "sysTable", "silver"]:
        loc = page.locator(f"#{el}")
        if loc.count():
            loc.evaluate("e => e.scrollIntoView({block: 'center'})"); page.wait_for_timeout(700); shot(page, f"eval_{el}")
    page.click(".tabs button[data-view='model']"); page.wait_for_timeout(800); shot(page, "model")
    b.close()
srv.shutdown()
