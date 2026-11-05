# data.py
# Purpose: Phase 1 of ComplaintIQ - project config + data download.
# Downloads CFPB complaints (CSV), eCFR regulations (XML) and CFPB PDFs into data/raw/.
# Writes data/manifest.json (url, size, sha256, time) so the data is reproducible (Gate G1).
# Run: python data.py   then back up with: aws s3 sync

import boto3, pathlib, hashlib, json, datetime as dt, requests

# ---------- config ----------
REGION = "us-east-1"
ACCOUNT = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
BUCKET = f"sagemaker-complaintiq-{ACCOUNT}"
ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
INDEX = ROOT / "index"
STATE = ROOT / "state"
REPORTS = ROOT / "reports"
for p in (RAW, PROC, INDEX, STATE, REPORTS):
    p.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "ComplaintIQ-capstone/1.0 (puja54454@gmail.com)"}

# ---------- download ----------
API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
ECFR = "https://www.ecfr.gov/api/versioner/v1"
manifest = []

def record(path, url):
    # save proof of each file: where it came from, size, hash, time
    blob = path.read_bytes()
    manifest.append({
        "file": str(path.relative_to(DATA)),
        "url": url,
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    })

def fetch(url, out, params=None):
    # download only if the file is not already there (safe to re-run)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        r = requests.get(url, params=params, headers=UA, stream=True, timeout=300)
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    full_url = requests.Request("GET", url, params=params).prepare().url
    record(out, full_url)

def download_all():
    # D1 - CFPB complaints, 2 products, Jan-Jun 2026, only rows with narrative
    for prod in ["Credit card", "Checking or savings account"]:
        name = prod.lower().replace(" ", "_") + ".csv"
        fetch(API, RAW / "cfpb" / name, params={
            "product": prod, "has_narrative": "true", "format": "csv",
            "no_aggs": "true", "date_received_min": "2026-01-01",
            "date_received_max": "2026-06-30"})

    # D2 - eCFR regulations: Reg E, Z, V, F, DD
    titles = requests.get(f"{ECFR}/titles.json", headers=UA, timeout=60).json()["titles"]
    as_of = next(t for t in titles if t["number"] == 12)["up_to_date_as_of"]
    for part in ["1005", "1026", "1022", "1006", "1030"]:
        fetch(f"{ECFR}/full/{as_of}/title-12.xml",
              RAW / "ecfr" / f"12cfr{part}.xml", params={"part": part})

    # D3 - CFPB PDFs (supervisory highlights + exam manual)
    base = "https://files.consumerfinance.gov/f/"
    pdfs = {
        "sh_issue17.pdf": base + "documents/bcfp_supervisory-highlights_issue-17_2018-09.pdf",
        "sh_issue18.pdf": base + "documents/cfpb_supervisory-highlights_issue-18_032019.pdf",
        "sh_issue22.pdf": base + "documents/cfpb_supervisory-highlights_issue-22_2020-09.pdf",
        "sh_issue23.pdf": base + "documents/cfpb_supervisory-highlights_issue-23_2021-01.pdf",
        "exam_manual.pdf": base + "201210_cfpb_supervision-and-examination-manual-v2.pdf",
    }
    for name, url in pdfs.items():
        out = RAW / "pdf" / name
        fetch(url, out)
        assert out.read_bytes()[:5] == b"%PDF-", f"{name} is not a real PDF"

    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(len(manifest), "files saved in data/manifest.json")

#---------- prepare: load complaints into DuckDB (PDF 2.1) ----------
import duckdb

def build_duckdb():
    con = duckdb.connect(str(DATA / "complaintiq.duckdb"))
    # 1) raw table: both CSVs, all columns as text
    con.execute(f"""
        CREATE OR REPLACE TABLE complaints_raw AS
        SELECT * FROM read_csv_auto('{RAW}/cfpb/*.csv', header=true,
                                    normalize_names=true, all_varchar=true)""")
    print(con.execute("DESCRIBE complaints_raw").df()[["column_name"]])

    # 2) clean view: the ONLY thing the SQL agent may query
    con.execute("""
        CREATE OR REPLACE VIEW v_complaints AS
        SELECT CAST(complaint_id AS BIGINT) AS complaint_id,
               CAST(date_received AS DATE)  AS date_received,
               product, subproduct, issue, subissue, company, state,
               tags, submitted_via,
               company_response_to_consumer AS company_response,
               timely_response              AS timely
        FROM complaints_raw
        QUALIFY ROW_NUMBER() OVER (PARTITION BY complaint_id ORDER BY date_received) = 1""")
    print(con.execute("""SELECT product, COUNT(*) n, MIN(date_received), MAX(date_received)
                         FROM v_complaints GROUP BY 1""").df())
    # 3) train/test split: Jan-May = train, June = test (held out)
    con.execute("""CREATE OR REPLACE VIEW v_train AS
                   SELECT * FROM v_complaints WHERE date_received <  DATE '2026-06-01'""")
    con.execute("""CREATE OR REPLACE VIEW v_test AS
                   SELECT * FROM v_complaints WHERE date_received >= DATE '2026-06-01'""")
    print(con.execute("""SELECT 'train' AS split, COUNT(*) FROM v_train
                         UNION ALL SELECT 'test', COUNT(*) FROM v_test""").df())
    con.close()

if __name__ == "__main__":
    download_all()
    build_duckdb()
        