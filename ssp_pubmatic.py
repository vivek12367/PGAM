import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Required environment variables
try:
    import streamlit as st
    PUBMATIC_TOKEN = st.secrets["PUBMATIC_TOKEN"]
    PUBLISHER_ID   = st.secrets["PUBMATIC_PUBLISHER_IDS"].split(",")[0].strip()
except:
    from dotenv import load_dotenv
    load_dotenv()
    PUBMATIC_TOKEN = os.getenv("PUBMATIC_TOKEN", "")
    PUBLISHER_ID   = os.getenv("PUBMATIC_PUBLISHER_IDS", "").split(",")[0].strip()

# API configuration
BASE_URL  = "https://api.pubmatic.com/v1/analytics/data/publisher"
HEADERS   = {"Authorization": f"Bearer {PUBMATIC_TOKEN}", "Accept": "application/json"}
PAGE_SIZE = 10000  # keep this at a reasonable value (e.g., 10k per page)

# Dimensions & Metrics: max 10 metrics allowed
DEFAULT_DIMENSIONS = "date,appDomain"
# Core metrics: netRevenue, paidImpressions, ecpm
DEFAULT_METRICS    = "netRevenue,paidImpressions,ecpm"


def _yesterday_iso():
    """Return yesterday's date in ISO YYYY-MM-DD format."""
    return (datetime.now().date() - timedelta(days=1)).isoformat()


def fetch_pubmatic_all(date=None, exclude_domains=None):
    """
    Fetch PubMatic domain-level metrics: net revenue, impressions, and eCPM.

    Returns:
      - df: pandas.DataFrame with columns [Date, AppDomain, SSP_Dash, SSP_Impressions, SSP_eCPM]
      - total_revenue: float sum of net revenue
    """
    if not PUBMATIC_TOKEN:
        raise EnvironmentError("PUBMATIC_TOKEN not set")
    if not PUBLISHER_ID:
        raise EnvironmentError("PUBMATIC_PUBLISHER_IDS must include at least one ID")

    target = date or _yesterday_iso()
    excludes = {d.lower().strip() for d in (exclude_domains or [])}

    records = []
    total_rev = 0.0
    page = 1

    while True:
        params = {
            "fromDate":   f"{target}T00:00",
            "toDate":     f"{target}T23:59",
            "dimensions": DEFAULT_DIMENSIONS,
            "metrics":    DEFAULT_METRICS,
            "dateUnit":   "date",
            "pageSize":   PAGE_SIZE,
            "page":       page,
            "sort":       "netRevenue:desc"   # ensure stable ordering by revenue
        }
        url = f"{BASE_URL}/{PUBLISHER_ID}"
        resp = requests.get(url, headers=HEADERS, params=params)
        if resp.status_code != 200:
            print(f"[WARN] PubMatic {PUBLISHER_ID} page {page} -> {resp.status_code}\n{resp.text}")
            break

        data = resp.json() or {}
        columns = data.get("columns", [])
        rows = data.get("rows", [])
        app_map = data.get("displayValue", {}).get("appDomain", {})
        if not rows:
            break

        for rec in rows:
            # rec: [date, appDomain, netRevenue, paidImpressions, ecpm]
            if len(rec) < 5:
                continue
            rec_map = dict(zip(columns, rec))
            domain_raw = rec_map.get("appDomain")
            domain = app_map.get(str(domain_raw), str(domain_raw)).lower().strip()
            if domain in excludes:
                continue

            net_rev = float(rec_map.get("netRevenue", 0) or 0)
            imps = int(rec_map.get("paidImpressions", 0) or 0)
            ecpm_val = float(rec_map.get("ecpm", 0) or 0)
            total_rev += net_rev

            records.append({
                "Date":            target,
                "AppDomain":       domain,
                "SSP_Dash":        round(net_rev, 2),
                "SSP_Impressions": imps,
                "SSP_eCPM":        round(ecpm_val, 2)
            })

        if len(rows) < PAGE_SIZE:
            break
        page += 1

    # ── Debug prints ───────────────────────────────────────────────────────────
    print(f"[DEBUG] Collected {len(records)} total records across all pages")
    print(f"[DEBUG] Sample domains (first 10): {[r['AppDomain'] for r in records[:10]]}")

    df = pd.DataFrame(records, columns=[
        "Date", "AppDomain", "SSP_Dash", "SSP_Impressions", "SSP_eCPM"
    ])
    return df, round(total_rev, 2)


if __name__ == "__main__":
    df, total = fetch_pubmatic_all()
    print(f"✅ PubMatic total net revenue for {_yesterday_iso()}: ${total:,.2f}")
    if df.empty:
        print("No data returned; verify your publisher ID and token permissions.")
    else:
        print(df.head().to_string(index=False))