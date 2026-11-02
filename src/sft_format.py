"""Compact prompt used for SFT rows and for querying the fine-tuned model (must match exactly)."""
FT_INSTRUCTION = ("CrisisLens triage. Classify this disaster-time social-media post and return only the JSON record "
                  "{category, actionable, urgency, location, needs, rationale}.")

def ft_prompt(text: str) -> str:
    return f"{FT_INSTRUCTION}\nPost: {text}"
