import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def _extract_rows(payload):
    """
    Sovrn sometimes returns a list or a dict with 'results' or 'data'.
    Normalize to a list of records.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("results") or payload.get("data") or []
    return []

def fetch_sovrn_all(date=None, exclude_domains=None):
    """
    Fetch net revenue, impressions, and compute eCPM for Sovrn
    (web domain-level + CTV) for a given date.

    Returns:
      - df: pandas.DataFrame with columns
        [Date, Domain, SSP_Dash, SSP_Impressions, SSP_eCPM]
      - total_revenue: float sum of all revenue (web + CTV)
    """
    # 1) Determine date bounds
    target = date or (datetime.now() - timedelta(days=1)).date().isoformat()
    start = f"{target}T00:00:00Z"
    end   = f"{target}T23:59:59Z"

    # 2) Load creds
    try:
        import streamlit as st
        web_key      = st.secrets["SOVRN_WEB_API_KEY"]
        ctv_key      = st.secrets["SOVRN_CTV_API_KEY"]
        publisher_id = st.secrets["SOVRN_PUBLISHER_ID"]
        company_id   = st.secrets.get("SOVRN_COMPANY_ID", publisher_id)
    except:
        from dotenv import load_dotenv
        load_dotenv()
        web_key      = os.getenv("SOVRN_WEB_API_KEY")
        ctv_key      = os.getenv("SOVRN_CTV_API_KEY")
        publisher_id = os.getenv("SOVRN_PUBLISHER_ID")
        company_id   = os.getenv("SOVRN_COMPANY_ID", publisher_id)

    if not (web_key and ctv_key and publisher_id):
        raise EnvironmentError(
            "SOVRN_WEB_API_KEY, SOVRN_CTV_API_KEY, and SOVRN_PUBLISHER_ID must be set"
        )

    excludes = {d.lower() for d in (exclude_domains or [])}
    records = []
    total_web_rev = 0.0

    web_endpoint = (
        f"https://api.sovrn.com/reporting/advertising/"
        f"publishers/{publisher_id}/account"
    )

    # --- a) Fetch revenue by domain ---
    rev_rows = []
    try:
        r = requests.get(web_endpoint,
                         headers={"x-api-key": web_key},
                         params={
                             "start": start,
                             "end": end,
                             "metrics": "publisherRevenue",
                             "dimensions": "domain",
                             "limit": 1000
                         })
        r.raise_for_status()
        rev_rows = _extract_rows(r.json())
    except Exception as e:
        print(f"⚠️ Sovrn web revenue failed: {e}")

    # --- b) Fetch impressions by domain ---
    imp_rows = []
    try:
        r = requests.get(web_endpoint,
                         headers={"x-api-key": web_key},
                         params={
                             "start": start,
                             "end": end,
                             "metrics": "publisherImpressions",
                             "dimensions": "domain",
                             "limit": 1000
                         })
        r.raise_for_status()
        imp_rows = _extract_rows(r.json())
    except Exception as e:
        print(f"⚠️ Sovrn web impressions failed: {e}")

    # 3) Merge into domain_map
    domain_map = {}
    for row in rev_rows:
        dom = str(row.get("domain","")).lower().strip()
        if not dom or dom in excludes:
            continue
        try:
            rev = float(str(row.get("publisherRevenue",0)).replace("$",""))
        except:
            rev = 0.0
        domain_map.setdefault(dom, {})["rev"] = rev
        total_web_rev += rev

    for row in imp_rows:
        dom = str(row.get("domain","")).lower().strip()
        if not dom or dom in excludes:
            continue
        try:
            imp = int(row.get("publisherImpressions",0))
        except:
            imp = 0
        domain_map.setdefault(dom, {})["imp"] = imp

    # 4) Build DataFrame rows with eCPM
    for dom, m in domain_map.items():
        rev = m.get("rev",0.0)
        imp = m.get("imp",0)
        ecpm = (rev / imp * 1000) if imp else 0.0
        records.append({
            "Date":            target,
            "Domain":          dom,
            "SSP_Dash":        round(rev,2),
            "SSP_Impressions": imp,
            "SSP_eCPM":        round(ecpm,2)
        })

    # 5) Fetch CTV spend
    total_ctv_rev = 0.0
    try:
        r = requests.get("https://api.xsp.sovrn.com/api/stats/get", params={
            "companyid": company_id,
            "type":      "exchange",
            "key":       ctv_key,
            "stats":     "cost",
            "interval":  "day",
            "start":     target,
            "end":       target
        })
        r.raise_for_status()
        for row in r.json().get("results",[]):
            try:
                cost = float(row.get("cost",0) or 0)
            except:
                cost = 0.0
            total_ctv_rev += cost
    except Exception:
        pass

    if total_ctv_rev > 0:
        records.append({
            "Date":            target,
            "Domain":          "CTV",
            "SSP_Dash":        round(total_ctv_rev,2),
            "SSP_Impressions": 0,
            "SSP_eCPM":        0.0
        })

    # 6) Build final DataFrame
    df = pd.DataFrame(records, columns=[
        "Date","Domain","SSP_Dash","SSP_Impressions","SSP_eCPM"
    ])
    total = round(total_web_rev + total_ctv_rev,2)
    return df, total

if __name__ == "__main__":
    df, total = fetch_sovrn_all()
    print(df.to_string(index=False))
    print(f"Totals: ${total:,.2f}")
