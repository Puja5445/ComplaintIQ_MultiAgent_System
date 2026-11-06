# core.py
# Purpose: shared engine of ComplaintIQ, used by agents.py and app.py.
# Part A: tests Bedrock models and saves the working IDs to models.json (no hard-coding).
# Part B: Bedrock wrapper - retries, cache, cost ledger, JSON validation.
# Run: python core.py   (runs the model probe)

import boto3, json, hashlib, sqlite3, time
from botocore.config import Config
from botocore.exceptions import ClientError
from pydantic import BaseModel, ValidationError
from data import REGION, ROOT, STATE

# ---------- Part A: model probe ----------
MODELS_FILE = ROOT / "models.json"
WANT = {
    "worker": ["sonnet-4-6", "sonnet"],
    "judge":  ["opus-4-8", "opus"],
    "embed":  ["titan-embed-text-v2"],
}

def works(rt, model_id, role):
    try:
        if role == "embed":
            rt.invoke_model(modelId=model_id, body=json.dumps({"inputText": "hello"}))
        else:
            rt.converse(modelId=model_id,
                        messages=[{"role": "user", "content": [{"text": "Say OK"}]}],
                        inferenceConfig={"maxTokens": 5})
        return True
    except Exception as e:
        print("   x", model_id, type(e).__name__)
        return False

def probe_models():
    br = boto3.client("bedrock", region_name=REGION)
    rt = boto3.client("bedrock-runtime", region_name=REGION)
    ids = []
    try:
        ids += [p["inferenceProfileId"]
                for p in br.list_inference_profiles()["inferenceProfileSummaries"]]
    except Exception as e:
        print("cannot list inference profiles:", type(e).__name__)
    base = [m["modelId"] for m in br.list_foundation_models()["modelSummaries"]]
    ids += ["us." + m for m in base if m.startswith("anthropic.")]
    ids += base
    chosen = {}
    for role, keys in WANT.items():
        for key in keys:
            for mid in ids:
                if key in mid and works(rt, mid, role):
                    chosen[role] = mid
                    break
            if role in chosen:
                break
    MODELS_FILE.write_text(json.dumps(chosen, indent=2))
    print("saved models.json:", chosen)

# ---------- Part B: Bedrock wrapper (PDF Phase 4) ----------
brt = boto3.client("bedrock-runtime", region_name=REGION,
                   config=Config(retries={"max_attempts": 12, "mode": "adaptive"},
                                 read_timeout=180))

# USD per 1K tokens (input, output); fill from the Bedrock pricing page later
PRICE_PER_1K = {"sonnet-4-6": (None, None), "opus": (None, None)}

class Ledger:
    # records every real LLM call: model, input tokens, output tokens, milliseconds
    def __init__(self):
        self.rows = []
    def add(self, model, tin, tout, ms):
        self.rows.append((model, tin, tout, ms))
    def usd(self):
        total = 0.0
        for model, tin, tout, _ in self.rows:
            for fam, (pi, po) in PRICE_PER_1K.items():
                if fam in model and pi is not None:
                    total += tin / 1000 * pi + tout / 1000 * po
        return round(total, 4)
    def summary(self):
        return {"calls": len(self.rows),
                "in_tokens": sum(r[1] for r in self.rows),
                "out_tokens": sum(r[2] for r in self.rows)}

LEDGER = Ledger()
_cache = sqlite3.connect(str(STATE / "llm_cache.sqlite"), check_same_thread=False)
_cache.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
_no_temperature = set()

def chat(model_id, system, user, max_tokens=1500, temperature=0.0, use_cache=True):
    # same model + prompt asked before -> answer from the cache, costs nothing
    key = hashlib.sha256(json.dumps([model_id, system, user, max_tokens]).encode()).hexdigest()
    if use_cache:
        hit = _cache.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
        if hit:
            return hit[0]
    cfg = {"maxTokens": max_tokens}
    if model_id not in _no_temperature:
        cfg["temperature"] = temperature
    t0 = time.time()
    try:
        r = brt.converse(modelId=model_id, system=[{"text": system}],
                         messages=[{"role": "user", "content": [{"text": user}]}],
                         inferenceConfig=cfg)
    except ClientError as e:
        if "temperature" in str(e) and model_id not in _no_temperature:
            _no_temperature.add(model_id)
            return chat(model_id, system, user, max_tokens, temperature, use_cache)
        raise
    text = r["output"]["message"]["content"][0]["text"]
    u = r["usage"]
    LEDGER.add(model_id, u["inputTokens"], u["outputTokens"], int((time.time() - t0) * 1000))
    _cache.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (key, text))
    _cache.commit()
    return text

def _json_part(raw):
    return raw[raw.find("{"): raw.rfind("}") + 1]

def chat_json(model_id, system, user, schema: type[BaseModel], **kw):
    # ask for JSON, validate with Pydantic, ONE repair attempt, then fail loudly
    system = system + "\nReturn ONLY one JSON object. No prose, no markdown."
    raw = chat(model_id, system, user, **kw)
    try:
        return schema.model_validate_json(_json_part(raw))
    except (ValidationError, ValueError) as e:
        fix = chat(model_id, system,
                   f"{user}\n\nYour previous answer was invalid: {e}\n"
                   f"Previous answer: {raw}\nReturn the corrected JSON only.", **kw)
        return schema.model_validate_json(_json_part(fix))

if __name__ == "__main__":
    probe_models()