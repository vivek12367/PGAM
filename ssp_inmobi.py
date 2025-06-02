import os
from dotenv import load_dotenv
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from json.decoder import JSONDecodeError

# ── 0) load .env and clear any proxy env-vars ───────────────────────────────────
load_dotenv()
for proxy_var in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
    os.environ.pop(proxy_var, None)

def fetch_inmobi_all(date=None):
    """
    Fetch InMobi placement-level earnings via the oRTB endpoint.
    Returns (df, total_revenue).
    """
    # 1) target date (UTC-aware yesterday)
    target = date or (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    start  = target + "T00:00:00Z"
    end    = target + "T23:59:59Z"

    # 2) pull credentials from env
    username   = os.getenv("INMOBI_USERNAME", "")
    secret_key = os.getenv("INMOBI_SECRET_KEY", "")
    if not (username and secret_key):
        raise EnvironmentError("INMOBI_USERNAME and INMOBI_SECRET_KEY must be set in .env")

    # 3) build session
    sess = requests.Session()
    sess.trust_env = False

    # 4) oRTB URL + JSON payload
    url = "https://api.w.inmobi.com/ortb"
    payload = {
        "auth": {
            "email":     username,
            "secretKey": secret_key
        },
        "reportingRequest": {
            "startTime": start,
            "endTime":   end,
            "level":     "placement",
            "metrics":   ["adRequests","adImpressions","clicks","earnings"]
        }
    }
    headers = {
        "Content-Type": "application/json",
        "Accept":       "application/json",
    }

    # 5) POST and handle errors
    try:
        resp = sess.post(url, json=payload, headers=headers, timeout=15)
    except requests.RequestException as e:
        print(f"❗️ Request error: {e}")
        return pd.DataFrame(), 0.0

    print(f"→ Status: {resp.status_code}, Body length: {len(resp.text)}")
    if resp.status_code != 200 or not resp.text:
        print("❗️ No data returned; check credentials, account permissions, or date range.")
        return pd.DataFrame(), 0.0

    # 6) parse JSON
    try:
        doc = resp.json()
    except JSONDecodeError:
        print("❗️ Response not valid JSON.")
        return pd.DataFrame(), 0.0

    data = doc.get("data", [])
    if not data:
        print("❗️ ‘data’ array empty.")
        return pd.DataFrame(), 0.0

    # 7) build DataFrame
    records = []
    for row in data:
        records.append({
            "Date":            target,
            "Domain":          row.get("placement", "<unknown>"),
            "SSP_Dash":        float(row.get("earnings", 0)),
            "SSP_Impressions": int(row.get("adImpressions", 0)),
            "Requests":        int(row.get("adRequests", 0)),
        })
    df = pd.DataFrame(records)

    # 8) aggregate + metrics
    grp = (
        df.groupby("Domain", as_index=False)
          .sum()
          .assign(
              **{
                  "Fill Rate": lambda d: d["SSP_Impressions"] / d["Requests"].replace(0,1),
                  "SSP_eCPM":   lambda d: d["SSP_Dash"] / d["SSP_Impressions"].replace(0,1) * 1000,
                  "Date":       target
              }
          )
    )

    total = grp["SSP_Dash"].sum()
    return grp[[
        "Date","Domain","SSP_Dash","SSP_Impressions",
        "Requests","Fill Rate","SSP_eCPM"
    ]], round(total,2)

if __name__ == "__main__":
    df, total = fetch_inmobi_all()
    if df.empty:
        print("No data returned; check credentials / permissions.")
    else:
        print("InMobi placement breakdown:")
        print(df.to_string(index=False))
        print(f"\nTotals: ${total:,.2f}")
