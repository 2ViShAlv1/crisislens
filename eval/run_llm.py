"""Run an LLM system over an eval split and save raw outputs + latency + token usage.

Systems (--mode): zeroshot (full schema prompt), fewshot (schema prompt + 5 train examples),
finetuned (compact prompt identical to the SFT training prompt).
Backends (--backend): adaption (Responses API, hosted models) | openai (any OpenAI-compatible
endpoint: Together, HF router, llama.cpp server; set --base-url and --key-env).
Resumable: ids already in the output file are skipped.
"""
import argparse, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.prompt import SYSTEM, user_msg, to_completion
from src.sft_format import FT_INSTRUCTION, ft_prompt
from src.env import load_env

def build_messages(mode, text, shots):
    if mode == "finetuned":
        return [{"role": "user", "content": ft_prompt(text)}]
    msgs = [{"role": "system", "content": SYSTEM}]
    if mode == "fewshot":
        for s in shots:
            msgs += [{"role": "user", "content": user_msg(s["text"])}, {"role": "assistant", "content": to_completion(s["out"])}]
    return msgs + [{"role": "user", "content": user_msg(text)}]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_eval"); ap.add_argument("--mode", required=True)
    ap.add_argument("--backend", default="openai"); ap.add_argument("--model", required=True)
    ap.add_argument("--base-url"); ap.add_argument("--key-env", default="TOGETHER_API_KEY")
    ap.add_argument("--name", required=True); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int); ap.add_argument("--max-tokens", type=int, default=200)
    a = ap.parse_args(); load_env()
    df = pd.read_parquet(ROOT / f"data/processed/{a.split}.parquet")
    if a.limit: df = df.head(a.limit)
    shots = []
    if a.mode == "fewshot":
        tr = pd.read_parquet(ROOT / "data/processed/train.parquet").set_index("id")
        shots = [dict(s, text=tr.loc[s["id"], "text"]) for s in json.load(open(ROOT / "eval/prompts/fewshot.json"))]
    out = ROOT / f"eval/preds/{a.name}__{a.split}.jsonl"; out.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    todo = df[~df.id.isin(done)].to_dict("records")
    print(f"{a.name}: {len(done)} done, {len(todo)} todo", flush=True)
    if a.backend == "adaption":
        from adaption import Adaption; cli = Adaption()
        def call(msgs):
            sys_ = "\n".join(m["content"] for m in msgs if m["role"] == "system")
            inp = [{"role": m["role"], "content": m["content"]} for m in msgs if m["role"] != "system"]
            r = cli.responses.create(model=a.model, instructions=sys_ or None, input=inp, temperature=0, max_output_tokens=a.max_tokens)
            u = r.usage; return r.output_text, getattr(u, "input_tokens", None), getattr(u, "output_tokens", None)
    else:
        from openai import OpenAI; cli = OpenAI(base_url=a.base_url, api_key=os.environ.get(a.key_env, "none"))
        def call(msgs):
            r = cli.chat.completions.create(model=a.model, messages=msgs, temperature=0, max_tokens=a.max_tokens)
            u = r.usage; return r.choices[0].message.content, getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None)
    def one(row):
        msgs = build_messages(a.mode, row["text"], shots)
        for k in range(4):
            try:
                t0 = time.perf_counter(); txt, it, ot = call(msgs); dt = time.perf_counter() - t0
                return {"id": row["id"], "raw": txt, "latency_s": dt, "in_tok": it, "out_tok": ot}
            except Exception as e:
                err = repr(e)[:300]; time.sleep(3 * (k + 1))
        return {"id": row["id"], "raw": None, "error": err}
    lock = threading.Lock(); t0 = time.time()
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for i, fu in enumerate(as_completed([ex.submit(one, r) for r in todo]), 1):
            r = fu.result()
            with lock: f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if i % 100 == 0 or i == len(todo): print(f"{i}/{len(todo)} {i/(time.time()-t0):.2f}/s", flush=True)

if __name__ == "__main__":
    main()
