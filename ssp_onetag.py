# ssp_onetag.py

import os
import sys
import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def _yesterday_iso():
    return (datetime.now().date() - timedelta(days=1)).isoformat()

def fetch_onetag_all(date=None, exclude_domains=None):
    try:
        import streamlit as st
        api_url = st.secrets["ONETAG_URL"]
    except:
        from dotenv import load_dotenv
        load_dotenv()
        api_url = os.getenv("ONETAG_URL")
    if not api_url:
        raise EnvironmentError(
            "ONETAG_API_URL is not set. "
            "Please export something like:\n"
            'export ONETAG_API_URL="https://…/onetag.csv?date=2025-05-20"'
        )

    target = date or _yesterday_iso()
    resp = requests.get(api_url)
    if resp.status_code != 200:
        raise RuntimeError(f"OneTag API HTTP {resp.status_code}\n{resp.text[:500]!r}")

    text = resp.text.strip()
    # If it begins with a “Day,” header, treat as CSV
    if text.startswith("Day,"):
        df_csv = pd.read_csv(StringIO(text))
        # Drop the summary “Total” row if present
        df_csv = df_csv[df_csv["Day"].str.lower() != "total"].copy()
        # Rename columns to our standard
        df_csv.rename(columns={
            "Day": "Date",
            "Earnings": "SSP_Dash",
            "Paid Impression": "SSP_Impressions",
            "Incoming BidRequest": "Requests"
        }, inplace=True)
        # Parse numeric fields
        # Earnings is like "$117,22" => 117.22
        df_csv["SSP_Dash"] = (
            df_csv["SSP_Dash"]
              .str.replace(r"[\$ ]", "", regex=True)   # drop $ and any non-breaking spaces
              .str.replace(",", ".", regex=False)      # comma→dot
              .astype(float)
        )
        df_csv["SSP_Impressions"] = df_csv["SSP_Impressions"].astype(int)
        df_csv["Requests"]         = df_csv["Requests"].astype(int)
        # Compute fill rate and eCPM
        df_csv["Fill Rate"] = (df_csv["SSP_Impressions"] / df_csv["Requests"]).fillna(0)
        df_csv["SSP_eCPM"]   = (df_csv["SSP_Dash"] / df_csv["SSP_Impressions"].replace(0, 1)) * 1000

        # Force our Date format to ISO
        df_csv["Date"] = pd.to_datetime(df_csv["Date"], dayfirst=True).dt.date.astype(str)

        total_rev = df_csv["SSP_Dash"].sum()
        # If there was exactly one row (no per-domain breakdown), mark domain as "<ALL>"
        df_csv["Domain"] = df_csv.get("Domain", "<ALL>")
        df = df_csv[[
            "Date", "Domain", "SSP_Dash", "SSP_Impressions", "Requests", "Fill Rate", "SSP_eCPM"
        ]]
        return df, round(float(total_rev), 2)

    # Otherwise try JSON (legacy)
    payload = resp.json()
    rows = payload.get("rows") or payload.get("data") or payload
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected OneTag payload: {payload!r}")

    records = []
    total_rev = 0.0
    for entry in rows:
        domain = entry.get("domain") or entry.get("bundle") or "<ALL>"
        rev  = float(entry.get("netRevenue", entry.get("revenue", 0)) or 0)
        imps = int(entry.get("paidImpressions", entry.get("impressions", 0)) or 0)
        reqs = int(entry.get("requests", 0) or 0)
        fill = float(entry.get("fillRate", 0) or 0)
        total_rev += rev
        ecpm = round((rev / imps * 1000) if imps else 0, 2)
        records.append({
            "Date":            target,
            "Domain":          domain,
            "SSP_Dash":        round(rev, 2),
            "SSP_Impressions": imps,
            "Requests":        reqs,
            "Fill Rate":       fill,
            "SSP_eCPM":        ecpm,
        })

    if not records:
        records.append({
            "Date":            target,
            "Domain":          "<ALL>",
            "SSP_Dash":        0.00,
            "SSP_Impressions": 0,
            "Requests":        0,
            "Fill Rate":       0.0,
            "SSP_eCPM":        0.00,
        })

    df = pd.DataFrame(records, columns=[
        "Date", "Domain", "SSP_Dash", "SSP_Impressions", "Requests", "Fill Rate", "SSP_eCPM"
    ])
    return df, round(total_rev, 2)


if __name__ == "__main__":
    try:
        df, total = fetch_onetag_all()
        print("OneTag summary:")
        print(df.to_string(index=False))
        print(f"\nTotals: ${total:,.2f}")
    except Exception as e:
        print(f"❌ Error fetching OneTag data: {e}", file=sys.stderr)
        sys.exit(1)
