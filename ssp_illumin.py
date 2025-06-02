import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def fetch_illumin_all(date=None, exclude_domains=None):
    """
    Fetch net revenue, impressions, requests, and fill-rate for Illumin for a given date,
    grouped by domain. Returns (df, total_revenue) where df has columns:
      [Date, Domain, SSP_Dash, SSP_Impressions, Requests, Fill Rate, SSP_eCPM]
    """
    base_url = os.getenv("ILLUMIN_API_BASE")
    if not base_url:
        raise EnvironmentError("ILLUMIN_API_BASE must be set in environment")

    target = date or (datetime.now() - timedelta(days=1)).date().isoformat()
    excludes = {d.lower().strip() for d in (exclude_domains or [])}

    records = []
    total_rev = 0.0
    page = 1

    while True:
        params = {
            "from":        target,
            "to":          target,
            "attribute[]": ["domain"],
            "page":        page,
        }
        resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break

        for entry in data:
            domain = entry.get("domain", "").lower().strip()
            if domain in excludes:
                continue

            # core metrics
            spend = float(entry.get("spend", 0) or 0)
            imps  = int(entry.get("imps", 0) or 0)
            reqs  = int(entry.get("requests", 0) or 0)
            fill  = float(entry.get("fill_rate", 0) or 0)

            total_rev += spend

            # compute eCPM (net spend per 1,000 paid impressions)
            ecpm = round((spend / imps * 1000) if imps else 0, 2)

            records.append({
                "Date":            target,
                "Domain":          domain or "<ALL>",
                "SSP_Dash":        round(spend, 2),
                "SSP_Impressions": imps,
                "Requests":        reqs,
                "Fill Rate":       round(fill, 4),
                "SSP_eCPM":        ecpm,
            })

        page += 1

    # fallback if no rows at all
    if not records:
        records = [{
            "Date":            target,
            "Domain":          "<ALL>",
            "SSP_Dash":        0.00,
            "SSP_Impressions": 0,
            "Requests":        0,
            "Fill Rate":       0.0,
            "SSP_eCPM":        0.00,
        }]

    df = pd.DataFrame(records, columns=[
        "Date",
        "Domain",
        "SSP_Dash",
        "SSP_Impressions",
        "Requests",
        "Fill Rate",
        "SSP_eCPM"
    ])

    return df, round(total_rev, 2)


if __name__ == "__main__":
    df, total = fetch_illumin_all()
    print("Illumin summary:")
    print(df.to_string(index=False))
    print(f"\nTotals: ${total:,.2f}")
