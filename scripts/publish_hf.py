"""Upload the run-#5 GGUF LoRA adapter to an HF model repo and the demo/ folder to an HF Docker Space.
Reads HF_TOKEN from .env. Writes eval/results/links.json."""
import json, os, sys, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.env import load_env
load_env()
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
user = api.whoami()["name"]
RUN = "7ebe8824-9857-4552-8574-8b58b3d55be5"
model_repo, space_repo = f"{user}/crisislens-qwen3.5-2b-lora", f"{user}/crisislens"

api.create_repo(model_repo, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj=str(ROOT / f"models/{RUN}/adapter-f16.gguf"), path_in_repo="crisislens-lora-f16.gguf", repo_id=model_repo)
api.upload_file(path_or_fileobj=str(ROOT / f"models/{RUN}/adapter_model.safetensors"), path_in_repo="peft/adapter_model.safetensors", repo_id=model_repo)
api.upload_file(path_or_fileobj=str(ROOT / f"models/{RUN}/adapter_config.json"), path_in_repo="peft/adapter_config.json", repo_id=model_repo)
card = (ROOT / "docs/MODEL_CARD.md").read_text(encoding="utf-8")
hdr = "---\nlicense: cc-by-nc-sa-4.0\nbase_model: Qwen/Qwen3.5-2B\ntags: [crisis-informatics, humaid, lora, gguf, autoscientist]\n---\n"
api.upload_file(path_or_fileobj=(hdr + card).encode(), path_in_repo="README.md", repo_id=model_repo)

api.create_repo(space_repo, repo_type="space", space_sdk="docker", exist_ok=True)
adapter_url = f"https://huggingface.co/{model_repo}/resolve/main/crisislens-lora-f16.gguf"
api.add_space_variable(space_repo, "ADAPTER_GGUF_URL", adapter_url)
space_readme = ("---\ntitle: CrisisLens\nemoji: 🛰️\ncolorFrom: blue\ncolorTo: red\nsdk: docker\napp_port: 7860\nlicense: cc-by-nc-sa-4.0\n---\n"
                "CrisisLens — disaster-response triage LLM (Qwen3.5-2B + LoRA trained with Adaption AutoScientist on HumAID). "
                "Runs on free CPU via llama.cpp; falls back to cached outputs if the model is still loading.\n")
api.upload_file(path_or_fileobj=space_readme.encode(), path_in_repo="README.md", repo_id=space_repo, repo_type="space")
api.upload_folder(folder_path=str(ROOT / "demo"), repo_id=space_repo, repo_type="space", ignore_patterns=["__pycache__/*"])
links = {"model": f"https://huggingface.co/{model_repo}", "space": f"https://huggingface.co/spaces/{space_repo}"}
try:
    links["github"] = subprocess.check_output(["gh", "repo", "view", "--json", "url", "-q", ".url"], cwd=ROOT, text=True).strip()
except Exception:
    pass
json.dump(links, open(ROOT / "eval/results/links.json", "w"), indent=2); print(json.dumps(links, indent=2))
