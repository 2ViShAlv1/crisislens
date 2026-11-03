"""Upload an SFT CSV to Adaption (raw mode), launch an AutoScientist run, and log every poll.

  python -I src/train/autoscientist_run.py launch --csv data/processed/sft_train.csv --name crisislens-v1 --model Qwen/Qwen3.5-9B
  python -I src/train/autoscientist_run.py poll --run <run_id>      # appends to eval/autoscientist/<run>.jsonl
  python -I src/train/autoscientist_run.py download --run <run_id>  # -> models/<run>/

All API responses are stored verbatim in eval/autoscientist/ (the training-run evidence).
"""
import argparse, json, sys, time, tarfile
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.env import load_env
LOG = ROOT / "eval/autoscientist"; LOG.mkdir(parents=True, exist_ok=True)

def dump(o):
    return o.model_dump(mode="json") if hasattr(o, "model_dump") else o

def now():
    return datetime.now(timezone.utc).isoformat()

def launch(c, a):
    if a.dataset:
        ds = a.dataset
    else:
        # The SDK PUTs the file to S3 with bare httpx.put() (5 s default timeout), which times out on a
        # 32 MB CSV over a home connection. Give that one call a long timeout for the upload only.
        import functools, httpx, adaption.lib.upload as U
        orig_put = U.httpx.put; U.httpx.put = functools.partial(orig_put, timeout=httpx.Timeout(900.0, connect=30.0))
        try:
            up = c.datasets.upload_file(a.csv, name=a.name, processing_mode="raw",
                                        column_mapping={"prompt": "prompt", "completion": "completion"})
        finally:
            U.httpx.put = orig_put
        ds = up.dataset_id
    print("dataset", ds, flush=True)
    while True:
        st = c.datasets.get_status(ds)
        if st.status == "failed": raise SystemExit(f"dataset failed: {dump(st)}")
        if st.row_count is not None: break
        time.sleep(5)
    print("rows", st.row_count, flush=True)
    rec = {}
    try:
        rec = dump(c.autoscientist.recommend_hyperparams(dataset_id=ds, model=a.model)); print("recommended", rec, flush=True)
    except Exception as e:
        print("recommend failed", repr(e)[:300])
    # column_mapping omitted: raw-mode ingestion canonicalises the columns, AutoScientist infers the mapping
    kw = dict(dataset_id=ds, model=a.model, training_method="instruction", data_format="chat",
              max_iterations=a.max_iterations, idempotency_key=a.name)
    if a.target: kw["target_win_rate"] = a.target
    if a.hp: kw["hyperparams"] = json.loads(a.hp)
    if a.aug_domain: kw["augmentation_domain_rows"] = a.aug_domain    # docs: data augmentation (optional)
    if a.aug_general: kw["augmentation_general_rows"] = a.aug_general
    run = c.autoscientist.create(**kw)
    meta = {"t": now(), "name": a.name, "csv": a.csv, "dataset_id": ds, "rows": st.row_count, "request": kw,
            "recommended_hyperparams": rec, "run": dump(run)}
    (LOG / f"{run.id}_launch.json").write_text(json.dumps(meta, indent=2, default=str))
    print("run", run.id, dump(run), flush=True)

def poll(c, a):
    f = LOG / f"{a.run}.jsonl"
    while True:
        try:
            r = dump(c.autoscientist.get(a.run))
        except Exception as e:  # transient network/DNS errors must not end the log
            print(now(), "poll error (retrying):", type(e).__name__, flush=True); time.sleep(a.every); continue
        with open(f, "a") as fh: fh.write(json.dumps({"t": now(), **r}, default=str) + "\n")
        print(now(), r["status"], f"iter {r.get('iterations_completed')}/{r.get('max_iterations')}",
              "best_win_rate", r.get("best_win_rate"), flush=True)
        if r["status"] in ("succeeded", "failed", "cancelled") or a.once: break
        time.sleep(a.every)

def download(c, a):
    out = ROOT / "models" / a.run; out.mkdir(parents=True, exist_ok=True)
    tgz = out / "checkpoint.tar.zst"  # platform serves a zstd-compressed tar
    # Resumable: on a slow link the stream can stall (ReadTimeout). Resume with an HTTP Range request;
    # if the server ignores Range (200 instead of 206) start the file over. --fresh discards a partial file.
    if getattr(a, "fresh", False) and tgz.exists(): tgz.unlink()
    import httpx
    for attempt in range(30):
        have = tgz.stat().st_size if tgz.exists() else 0
        hdr = {"Range": f"bytes={have}-"} if have else {}
        try:
            with c.autoscientist.with_streaming_response.download(a.run, extra_headers=hdr,
                                                                    timeout=httpx.Timeout(120.0, connect=30.0)) as r:
                code = r.http_response.status_code
                if code == 416:  # range starts at end of file -> already complete
                    break
                mode = "ab" if (have and code == 206) else "wb"
                print(now(), f"attempt {attempt}: HTTP {code}, resume from {have if mode == 'ab' else 0}", flush=True)
                with open(tgz, mode) as fh:
                    for chunk in r.iter_bytes(): fh.write(chunk)
            break
        except Exception as e:
            print(now(), f"download interrupted at {tgz.stat().st_size if tgz.exists() else 0} bytes: {type(e).__name__}", flush=True)
            time.sleep(5)
    else:
        raise SystemExit("download failed after 30 attempts")
    import zstandard
    with open(tgz, "rb") as fh:
        tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(fh), mode="r|").extractall(out, filter="data")
    import shutil
    if (out / "trainer_state.json").exists():
        shutil.copy(out / "trainer_state.json", LOG / f"{a.run}_trainer_state.json")
    print("extracted", [p.name for p in out.rglob("*")][:30])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    ap.add_argument("--csv"); ap.add_argument("--name"); ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--max-iterations", type=int, default=3); ap.add_argument("--target", type=float)
    ap.add_argument("--run"); ap.add_argument("--dataset"); ap.add_argument("--hp"); ap.add_argument("--every", type=int, default=120); ap.add_argument("--once", action="store_true"); ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--aug-domain", type=int); ap.add_argument("--aug-general", type=int)
    a = ap.parse_args(); load_env()
    import httpx; from adaption import Adaption
    c = Adaption(timeout=httpx.Timeout(900.0, connect=30.0))  # 32 MB CSV uploads time out at the default write timeout
    {"launch": launch, "poll": poll, "download": download}[a.cmd](c, a)
