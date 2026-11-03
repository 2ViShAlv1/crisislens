"""Publish CrisisLens to Hugging Face. Reads HF_TOKEN from .env. Writes eval/results/links.json.

  python -I scripts/publish_hf.py                # model repo + static Space (free tier)
  python -I scripts/publish_hf.py --space-only   # only (re)publish the Space
  python -I scripts/publish_hf.py --docker       # Docker Space with the live llama.cpp model (needs HF PRO)
  python -I scripts/publish_hf.py --stage DIR    # only build the static site into DIR (local test, no upload)

Hugging Face now hosts Docker/Gradio Spaces on free CPU only for PRO accounts (HTTP 402); static Spaces are free.
The static Space serves the web console with outputs the fine-tuned model produced during evaluation; the
Docker Space (or a local run, see README) adds live triage of new text.
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
DEMO = ROOT / "demo"
RUN = "7ebe8824-9857-4552-8574-8b58b3d55be5"
TEXT = {".sh", ".py", ".txt", ".html", ".css", ".js", ".json", ".jsonl", ".md"}

def lf(folder):  # CRLF from a Windows checkout breaks Linux containers and is noise on the web
    for p in Path(folder).rglob("*"):
        if p.is_file() and (p.suffix in TEXT or p.name == "Dockerfile"):
            b = p.read_bytes()
            if b"\r\n" in b: p.write_bytes(b.replace(b"\r\n", b"\n"))

def stage_static(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    shutil.copy(DEMO / "web/index.html", out / "index.html")
    (out / "static").mkdir(exist_ok=True); (out / "cache").mkdir(exist_ok=True)
    for f in ["app.js", "app.css"]: shutil.copy(DEMO / "web" / f, out / "static" / f)
    for f in ["feed.jsonl", "dashboard.json", "examples.json"]: shutil.copy(DEMO / "cache" / f, out / "cache" / f)
    (out / "README.md").write_text(
        "---\ntitle: CrisisLens\nemoji: 🛰️\ncolorFrom: blue\ncolorTo: red\nsdk: static\napp_file: index.html\nlicense: cc-by-nc-sa-4.0\n"
        "short_description: Disaster-triage LLM fine-tuned with AutoScientist\n---\n"
        "CrisisLens is a disaster-response triage model (Qwen3.5-2B + LoRA, trained with Adaption AutoScientist on HumAID). "
        "This static console replays outputs the model produced on four held-out disasters. For live triage of new text, "
        "run it locally (see https://github.com/2ViShAlv1/crisislens).\n", encoding="utf-8")
    lf(out); return out

def stage_docker():
    out = Path(tempfile.mkdtemp()) / "demo"
    shutil.copytree(DEMO, out, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    lf(out); return out

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--stage" in args:
        print("staged", stage_static(args[args.index("--stage") + 1])); sys.exit()
    from src.env import load_env
    load_env()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    model_repo, space_repo = f"{user}/crisislens-qwen3.5-2b-lora", f"{user}/crisislens"

    if "--space-only" not in args:
        api.create_repo(model_repo, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=str(ROOT / f"models/{RUN}/adapter-f16.gguf"), path_in_repo="crisislens-lora-f16.gguf", repo_id=model_repo)
        api.upload_file(path_or_fileobj=str(ROOT / f"models/{RUN}/adapter_model.safetensors"), path_in_repo="peft/adapter_model.safetensors", repo_id=model_repo)
        api.upload_file(path_or_fileobj=str(ROOT / f"models/{RUN}/adapter_config.json"), path_in_repo="peft/adapter_config.json", repo_id=model_repo)
        card = (ROOT / "docs/MODEL_CARD.md").read_text(encoding="utf-8")
        hdr = "---\nlicense: cc-by-nc-sa-4.0\nbase_model: Qwen/Qwen3.5-2B\ntags: [crisis-informatics, humaid, lora, gguf, autoscientist]\n---\n"
        api.upload_file(path_or_fileobj=(hdr + card).encode(), path_in_repo="README.md", repo_id=model_repo)

    if "--docker" in args:
        api.create_repo(space_repo, repo_type="space", space_sdk="docker", exist_ok=True)
        api.add_space_variable(space_repo, "ADAPTER_GGUF_URL", f"https://huggingface.co/{model_repo}/resolve/main/crisislens-lora-f16.gguf")
        stage = stage_docker()
        (stage / "README.md").write_text("---\ntitle: CrisisLens\nemoji: 🛰️\ncolorFrom: blue\ncolorTo: red\nsdk: docker\napp_port: 7860\n"
                                         "license: cc-by-nc-sa-4.0\n---\nCrisisLens live console (llama.cpp on CPU).\n", encoding="utf-8")
    else:
        api.create_repo(space_repo, repo_type="space", space_sdk="static", exist_ok=True)
        stage = stage_static(Path(tempfile.mkdtemp()) / "site")
    api.upload_folder(folder_path=str(stage), repo_id=space_repo, repo_type="space")

    links = {"model": f"https://huggingface.co/{model_repo}", "space": f"https://huggingface.co/spaces/{space_repo}"}
    try:
        links["github"] = subprocess.check_output(["gh", "repo", "view", "--json", "url", "-q", ".url"], cwd=ROOT, text=True).strip()
    except Exception:
        pass
    json.dump(links, open(ROOT / "eval/results/links.json", "w"), indent=2); print(json.dumps(links, indent=2))
