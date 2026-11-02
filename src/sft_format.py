"""Prompts used for SFT rows and for querying the fine-tuned model (must match exactly).

v1 (runs #1-#4): one-line instruction. The pilot showed the model then invents label names
   (it never sees the label set) -> category accuracy 0 on a 69-post probe.
v2 (runs #5+): instruction also lists the 10 category ids and the urgency scale (~110 tokens).
"""
from src.prompt import CLASSES

FT_INSTRUCTION = ("CrisisLens triage. Classify this disaster-time social-media post and return only the JSON record "
                  "{category, actionable, urgency, location, needs, rationale}.")

FT_INSTRUCTION_V2 = ("CrisisLens triage. Return only a JSON object with keys category, actionable (true/false), "
                     "urgency (integer 1-5, 5=life-threatening), location (place written in the post or null), "
                     "needs (list), rationale (<=20 words). category must be one of: " + ", ".join(CLASSES) + ".")

PROMPT_VERSION = {"v1": FT_INSTRUCTION, "v2": FT_INSTRUCTION_V2}

def ft_prompt(text: str, version: str = "v2") -> str:
    return f"{PROMPT_VERSION[version]}\nPost: {text}"
