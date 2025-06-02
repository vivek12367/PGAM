import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
# Load environment variables from .env
load_dotenv()

# API configuration
API_URL = "https://platform.33across.com/api/v2/dashboard_reporting"
try:
    import streamlit as st
    TOKEN = st.secrets["ACROSS_TOKEN"]
except:
    TOKEN = os.getenv("ACROSS_TOKEN", "")
PAGE    = 1

# Dimensions & Metrics
# The 33Across API returns a list of content items with estimated_revenue and impressions/impressions_served


def _yesterday_iso():
    """Return yesterday's date in ISO YYYY-MM-DD format."""
    return (datetime.now().date() - timedelta(days=1)).isoformat()


def fetch_33across_all(date=None, exclude_domains=None):
    """
    Fetch domain-level revenue and impressions for 33Across for a given date,
    and compute eCPM (revenue per thousand impressions).

    Returns:
      - df: pandas.DataFrame with columns [Date, Domain, SSP_Dash, SSP_Impressions, SSP_eCPM]
      - total_revenue: float sum of revenue
    """
    if not TOKEN:
        raise EnvironmentError("ACROSS_TOKEN not set")

    target = date or _yesterday_iso()
    excludes = {d.lower().strip() for d in (exclude_domains or [])}

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "start_date": target,
        "end_date":   target,
        "page":       PAGE
    }

    resp = requests.post(API_URL, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json() or {}

    records = []
    total_rev = 0.0

    for row in data.get("content", []):
        # Extract domain/bundle if available
        domain = row.get("domain") or row.get("site") or ""
        domain = domain.lower().strip()
        if exclude_domains and domain in excludes:
            continue

        # Parse revenue
        rev_str = str(row.get("estimated_revenue", "0")).replace("$", "").replace(",", "")
        try:
            rev = float(rev_str)
        except ValueError:
            rev = 0.0

        # Parse impressions
        imps = row.get("impressions") or row.get("impressions_served") or 0
        try:
            imps = int(imps)
        except ValueError:
            imps = 0

        total_rev += rev

        records.append({
            "Date":            target,
            "Domain":          domain,
            "SSP_Dash":        round(rev, 2),
            "SSP_Impressions": imps
        })

    # Build DataFrame
    if records:
        df = pd.DataFrame(records, columns=["Date", "Domain", "SSP_Dash", "SSP_Impressions"])
    else:
        df = pd.DataFrame(columns=["Date", "Domain", "SSP_Dash", "SSP_Impressions"])

    # Compute eCPM: revenue per 1000 impressions
    df["SSP_eCPM"] = (df["SSP_Dash"] / df["SSP_Impressions"].replace(0, 1)) * 1000
    df["SSP_eCPM"] = df["SSP_eCPM"].round(2)

    return df, round(total_rev, 2)


if __name__ == "__main__":
    df, total = fetch_33across_all()
    print(df.to_string(index=False))
    print(f"Totals: dsp_spend=${total:,.2f}")
