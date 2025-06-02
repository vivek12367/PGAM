import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def fetch_sharethrough_all(date=None, exclude_domains=None):
    """
    Fetch net revenue and rendered impressions for Sharethrough for a given date.
    Returns:
      - df: DataFrame with columns [Date, Domain, SSP_Dash, SSP_Impressions, SSP_eCPM]
      - total_revenue: float
    """
    try:
        import streamlit as st
        token         = st.secrets["SHARETHROUGH_TOKEN"]
        publisher_key = st.secrets["SHARETHROUGH_ACCOUNT_ID"]
    except:
        from dotenv import load_dotenv
        load_dotenv()
        token         = os.getenv("SHARETHROUGH_TOKEN")
        publisher_key = os.getenv("SHARETHROUGH_ACCOUNT_ID")
    if not token or not publisher_key:
        raise EnvironmentError(
            "SHARETHROUGH_TOKEN and SHARETHROUGH_ACCOUNT_ID must be set"
        )

    # determine date
    target = date or (datetime.now() - timedelta(days=1)).date().isoformat()

    url = "https://reporting-api.sharethrough.com/v2/programmatic/supply"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/json",
    }
    payload = {
        "startDate":  target,
        "endDate":    target,
        "publishers": [publisher_key],
        "fields":     ["pub_earnings", "rendered_impressions"],
        "groupBy":    ["date"],
    }

    # call the API
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json().get("results", [])

    if not data:
        # no data for that date
        df = pd.DataFrame([{
            "Date":             target,
            "Domain":           "<ALL>",
            "SSP_Dash":         0.0,
            "SSP_Impressions":  0,
            "SSP_eCPM":         0.0
        }])
        return df, 0.0

    # we only get one row when grouping by date
    row = data[0]
    # parse revenue & impressions
    earnings = row.get("pub_earnings") or row.get("PUB_EARNINGS") or 0
    imps     = row.get("rendered_impressions") or row.get("RENDERED_IMPRESSIONS") or 0
    try:
        rev = float(earnings)
    except (ValueError, TypeError):
        rev = 0.0
    try:
        imps = int(imps)
    except (ValueError, TypeError):
        imps = 0

    # compute eCPM
    ecpm = (rev / imps * 1000) if imps else 0.0

    df = pd.DataFrame([{
        "Date":             target,
        "Domain":           "<ALL>",
        "SSP_Dash":         round(rev, 2),
        "SSP_Impressions":  imps,
        "SSP_eCPM":         round(ecpm, 2),
    }])

    return df, round(rev, 2)


if __name__ == "__main__":
    try:
        df, total = fetch_sharethrough_all()
        print(df.to_string(index=False))
        print(f"Totals: ${total:,.2f}")
    except Exception as e:
        print(f"❌ Error fetching Sharethrough data: {e}")
