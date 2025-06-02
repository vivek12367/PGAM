import os
import requests
import csv
import time
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

def fetch_unruly_all(date=None, exclude_domains=None):
    """
    Fetch net revenue, impressions, requests, fill-rate, and eCPM
    from Unruly supply summary for a given date. Returns
    (df, total_revenue) with columns:
      [Date, Domain, SSP_Dash, SSP_Impressions, Requests, Fill Rate, SSP_eCPM].
    """
    # 1) target date
    target = date or (datetime.now() - timedelta(days=1)).date().isoformat()
    excludes = {d.lower() for d in (exclude_domains or [])}

    # 2) creds + auth
    username = os.getenv("UNRULY_USERNAME")
    password = os.getenv("UNRULY_PASSWORD")
    api_key  = os.getenv("UNRULY_API_KEY")
    if not (username and password and api_key):
        raise EnvironmentError("UNRULY_USERNAME, UNRULY_PASSWORD, UNRULY_API_KEY must be set")

    auth_url      = "https://api.unruly.co/ctrl/auth"
    report_url    = "https://api.unruly.co/ctrl/api/insights/supplySummary"
    download_url  = "https://api.unruly.co/ctrl/api/download/fileURL"

    # authenticate
    r = requests.post(auth_url, data={"username": username, "password": password})
    r.raise_for_status()
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError("Failed to obtain Unruly access token")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # request CSV report
    r = requests.post(
        report_url,
        headers=headers,
        params={"dateRange":"YESTERDAY","dateInterval":"DAILY","apiKey":api_key},
        json={}
    )
    r.raise_for_status()
    fn = None
    try:
        fn = r.json().get("fileName")
    except ValueError:
        fn = r.text.strip()
    if not (fn and fn.lower().endswith(".csv")):
        raise RuntimeError(f"Unexpected filename: {fn}")

    # poll for download URL
    csv_url = None
    last_resp = None
    for _ in range(10):
        dl = requests.get(download_url, headers=headers, params={"fileName":fn,"apiKey":api_key})
        last_resp = dl
        if dl.status_code == 200:
            body = dl.json()
            csv_url = body.get("S3_FILE_URL") or body.get("fileUrl")
            if csv_url:
                break
        time.sleep(5)
    if not csv_url:
        raise RuntimeError(f"Could not retrieve CSV URL; last status {last_resp.status_code}")

    # download + parse
    dl = requests.get(csv_url)
    dl.raise_for_status()
    reader = csv.DictReader(StringIO(dl.text))
    cols   = reader.fieldnames or []

    # auto-detect columns
    domain_col = next((c for c in cols if "domain" in c.lower()), None)
    rev_col    = next((c for c in cols if any(k in c.lower() for k in ["earning","revenue"])), None)
    imp_col    = next((c for c in cols if c.lower()=="impressions"), None)
    req_col    = next((c for c in cols if "pub request" in c.lower()), None)
    fill_col   = next((c for c in cols if "fill rate" in c.lower()), None)
    ecpm_col   = next((c for c in cols if "eCPM" in c), None)

    records = []
    total_rev = 0.0

    # if no domain column, treat as single aggregate row
    if not domain_col:
        # read first row
        first = next(reader, {})
        raw_rev = first.get(rev_col,"0").replace("$","").replace(",","").strip()
        rev = float(raw_rev) if raw_rev else 0.0
        total_rev = rev
        imps = int(first.get(imp_col,0) or 0)
        reqs = int(first.get(req_col,0) or 0) if req_col else 0
        fill = float(first.get(fill_col,0) or 0) if fill_col else 0.0
        ecpm = float(first.get(ecpm_col,0) or 0) if ecpm_col else (rev / imps * 1000 if imps else 0.0)

        records.append({
            "Date":            target,
            "Domain":          "<ALL>",
            "SSP_Dash":        round(rev,2),
            "SSP_Impressions": imps,
            "Requests":        reqs,
            "Fill Rate":       round(fill/100 if "%" in (fill_col or "") else fill,4),
            "SSP_eCPM":        round(ecpm,2),
        })

    else:
        # per-domain rows
        for row in reader:
            dom = row.get(domain_col,"").strip().lower()
            if not dom or dom in excludes:
                continue
            raw_rev = row.get(rev_col,"0").replace("$","").replace(",","").strip()
            try: rev = float(raw_rev)
            except: rev = 0.0
            imps = int(row.get(imp_col,0) or 0)
            reqs = int(row.get(req_col,0) or 0) if req_col else 0
            fill = float(row.get(fill_col,0) or 0) if fill_col else 0.0
            ecpm = float(row.get(ecpm_col,0) or 0) if ecpm_col else (rev / imps * 1000 if imps else 0.0)

            total_rev += rev
            records.append({
                "Date":            target,
                "Domain":          dom,
                "SSP_Dash":        round(rev,2),
                "SSP_Impressions": imps,
                "Requests":        reqs,
                "Fill Rate":       round(fill/100 if "%" in (fill_col or "") else fill,4),
                "SSP_eCPM":        round(ecpm,2),
            })

    df = pd.DataFrame(records, columns=[
        "Date","Domain","SSP_Dash","SSP_Impressions","Requests","Fill Rate","SSP_eCPM"
    ])
    return df, round(total_rev,2)


if __name__ == '__main__':
    df, total = fetch_unruly_all()
    print(df.to_string(index=False))
    print(f"\nTotals: ${total:,.2f}")
