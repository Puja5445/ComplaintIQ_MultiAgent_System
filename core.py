# core.py
# Purpose: shared engine of ComplaintIQ, used by agents.py and app.py.
# Part A: tests Bedrock models and saves the working IDs to models.json (no hard-coding).
# Later parts: LLM wrapper (ledger, cache), Titan embeddings, FAISS + BM25, guards.
# Run: python core.py

import boto3, json
from data import REGION, ROOT

MODELS_FILE = ROOT / "models.json"

# role -> name pieces to look for, best choice first
WANT = {
    "worker": ["sonnet-4-6", "sonnet"],
    "judge":  ["opus-4-8", "opus"],
    "embed":  ["titan-embed-text-v2"],
}

def works(rt, model_id, role):
    # send a tiny test call; True if the model answers
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
    br = boto3.client("bedrock", region_name=REGION)          # lists models
    rt = boto3.client("bedrock-runtime", region_name=REGION)  # calls models

    ids = []
    try:   # listing profiles may be blocked in the lab
        ids += [p["inferenceProfileId"]
                for p in br.list_inference_profiles()["inferenceProfileSummaries"]]
    except Exception as e:
        print("cannot list inference profiles:", type(e).__name__)

    base = [m["modelId"] for m in br.list_foundation_models()["modelSummaries"]]
    ids += ["us." + m for m in base if m.startswith("anthropic.")]   # build profile IDs ourselves
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

if __name__ == "__main__":
    probe_models()