# core.py
# Purpose: shared engine of ComplaintIQ, used by agents.py and app.py.
# Part A: tests Bedrock models and saves the working IDs to models.json (no hard-coding).
# Part B: Bedrock wrapper - retries, cache, cost ledger, JSON validation.
# Run: python core.py   (runs the model probe)

import duckdb
import boto3, json, hashlib, sqlite3, time
from botocore.config import Config
from botocore.exceptions import ClientError
from pydantic import BaseModel, ValidationError
from data import REGION, ROOT, STATE, DATA, PROC

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

# ---------- Part A2: writer model (non-Claude, writes the fake stories) ----------
WRITER_KEYS = ["nova-lite", "nova-pro", "nova-micro", "llama3-3-70b", "llama3-1-70b", "mistral-large"]

def probe_writer():
    # finds one working non-Claude model and ADDS it to models.json (other roles stay as they are)
    br = boto3.client("bedrock", region_name=REGION)
    rt = boto3.client("bedrock-runtime", region_name=REGION)
    base = [m["modelId"] for m in br.list_foundation_models()["modelSummaries"]]
    ids = ["us." + m for m in base if m.startswith(("amazon.nova", "meta.llama"))] + base
    models = json.loads(MODELS_FILE.read_text())
    for key in WRITER_KEYS:
        for mid in ids:
            if key in mid and works(rt, mid, "writer"):
                models["writer"] = mid
                MODELS_FILE.write_text(json.dumps(models, indent=2))
                print("saved writer:", mid)
                return
    print("no writer model works - paste the x lines above")

# ---------- Part C: fake customer stories (trainer approved) ----------
# Writer model (Nova, not Claude) turns real labels into messy customer stories.
# Saved to data/processed/narratives.jsonl and joined to the labels as view v_narratives.

STORY_SYSTEM = (
    "You write realistic consumer complaints for a bank complaint system. "
    "Write 3-5 sentences in first person, informal and a bit messy, like a real customer typing in a hurry. "
    "Do NOT use the exact words of the product, issue or sub-issue labels you are given; "
    "describe what happened in everyday words. "
    "Include a rough date or time frame and one concrete detail (an amount, a place, or what the bank said). "
    "Use only fake details. Output only the complaint text."
    "Write exactly ONE complaint from ONE customer. Never write several, and never use '---'.")

def _story_prompt(prod, sub, iss, subi, comp, st, tags, d, add_pii):
    p = (f"Product: {prod} / {sub}\nIssue: {iss} / {subi}\nCompany: {comp}\n"
         f"State: {st}\nComplaint date: {d}\n")
    if tags and "Older American" in tags:
        p += "The customer is over 62; show it naturally (age or retirement) without saying 'older American'.\n"
    if tags and "Servicemember" in tags:
        p += "The customer is military or a veteran; mention it naturally.\n"
    if add_pii:
        p += "Also include a fake phone number like 555-0142 and a card ending like 'ending 4821'.\n"
    return p

def build_narratives(n_train=300, n_test=60, limit=None):
    out = PROC / "narratives.jsonl"
    done = set()
    if out.exists():                      # safe to re-run: skips finished stories
        done = {json.loads(l)["complaint_id"] for l in open(out)}
    writer = json.loads(MODELS_FILE.read_text())["writer"]
    con = duckdb.connect(str(DATA / "complaintiq.duckdb"))
    for split, view, n in [("train", "v_train", n_train), ("test", "v_test", n_test)]:
        rows = con.execute(f"""SELECT complaint_id, product, subproduct, issue, subissue,
                                      company, state, tags, CAST(date_received AS VARCHAR)
                               FROM {view} USING SAMPLE reservoir({n} ROWS) REPEATABLE (42)""").fetchall()
        if limit:
            rows = rows[:limit]
        for i, (cid, prod, sub, iss, subi, comp, st, tags, d) in enumerate(rows):
            if cid in done:
                continue
            text = chat(writer, STORY_SYSTEM,
                        _story_prompt(prod, sub, iss, subi, comp, st, tags, d, add_pii=(i % 10 == 0)),
                        max_tokens=300, temperature=0.8)
            with open(out, "a") as f:
                f.write(json.dumps({"complaint_id": cid, "split": split,
                                    "narrative": text.strip()}) + "\n")
            if (i + 1) % 20 == 0:
                print(split, i + 1, "/", len(rows))
    # join stories to labels: 12 columns of v_complaints + narrative + split
    con.execute(f"""CREATE OR REPLACE VIEW v_narratives AS
                    SELECT c.*, n.narrative, n.split FROM v_complaints c
                    JOIN read_json_auto('{out}') n ON c.complaint_id = n.complaint_id""")
    print(con.execute("SELECT split, COUNT(*) AS stories FROM v_narratives GROUP BY 1").df())
    con.close()
if __name__ == "__main__":
    probe_models()