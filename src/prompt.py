"""Single source of truth for the CrisisLens task prompt, schema, and output parsing."""
import json, re

CLASS_DEFS = {
    "caution_and_advice": "warnings, alerts, safety tips, forecasts or instructions about the hazard",
    "displaced_people_and_evacuations": "people evacuated, relocated, displaced, in shelters/camps, or evacuation orders",
    "infrastructure_and_utility_damage": "damage to buildings, roads, bridges, power, water, telecom or other utilities/services",
    "injured_or_dead_people": "reports of people injured, killed or dead, casualty counts",
    "missing_or_found_people": "people missing, trapped and unaccounted for, or found/reunited",
    "not_humanitarian": "not useful for humanitarian response (jokes, politics, ads, unrelated chatter)",
    "other_relevant_information": "relevant disaster information not covered by any other category (situation updates, stats, news)",
    "requests_or_urgent_needs": "requests for help or urgent needs of affected people: food, water, shelter, medical aid, rescue, supplies",
    "rescue_volunteering_or_donation_effort": "rescue operations, volunteering, donations, fundraising, relief distribution, offers of help",
    "sympathy_and_support": "prayers, condolences, thoughts, emotional support for victims",
}
CLASSES = list(CLASS_DEFS)

SYSTEM = (
    "You are CrisisLens, a disaster-response triage assistant. Read ONE social-media post written during a "
    "disaster and output ONLY a JSON object, no prose, with exactly these keys:\n"
    '{"category": <one label>, "actionable": true|false, "urgency": <integer 1-5>, '
    '"location": <most specific place named in the post, or null>, "needs": [<short needs, e.g. "water", "medical", "rescue">], '
    '"rationale": "<at most 20 words>"}\n'
    "category labels:\n" + "\n".join(f"- {k}: {v}" for k, v in CLASS_DEFS.items()) + "\n"
    "actionable = a responder could act on this post (specific need, location, trapped/injured people, hazard to warn about).\n"
    "urgency: 1 = no response value, 2 = general info, 3 = useful situational info, 4 = time-sensitive need or hazard, "
    "5 = life-threatening (trapped, injured, missing, immediate rescue).\n"
    "needs = [] unless the post states or clearly implies a need. location = null if no place is named."
)

def user_msg(text: str) -> str:
    return f"Post: {text}"

KEY_ORDER = ["category", "actionable", "urgency", "location", "needs", "rationale"]

def to_completion(rec: dict) -> str:
    """Canonical compact JSON string used as the training target."""
    return json.dumps({k: rec[k] for k in KEY_ORDER}, ensure_ascii=False, separators=(", ", ": "))

def parse_output(s: str):
    """Return (dict|None, valid: bool). valid = strict schema conformance on the first JSON object."""
    if s is None:
        return None, False
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip())
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None, False
    try:
        o = json.loads(m.group(0))
    except Exception:
        return None, False
    return o, validate(o)

def validate(o) -> bool:
    try:
        return (isinstance(o, dict) and set(o) == set(KEY_ORDER)
                and o["category"] in CLASS_DEFS
                and isinstance(o["actionable"], bool)
                and isinstance(o["urgency"], int) and 1 <= o["urgency"] <= 5
                and (o["location"] is None or isinstance(o["location"], str))
                and isinstance(o["needs"], list) and all(isinstance(x, str) for x in o["needs"])
                and isinstance(o["rationale"], str) and len(o["rationale"].split()) <= 20)
    except Exception:
        return False

def category_of(s: str):
    """Lenient category extraction (for classification metrics even when JSON is malformed)."""
    o, _ = parse_output(s)
    if isinstance(o, dict) and o.get("category") in CLASS_DEFS:
        return o["category"]
    for c in sorted(CLASSES, key=len, reverse=True):
        if s and c in s:
            return c
    return "__invalid__"
