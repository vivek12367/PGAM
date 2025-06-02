import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def fetch_freewheel_all(date=None, exclude_domains=None):
    """
    Fetch Freewheel net revenue, impressions, requests, fill rate and eCPM
    broken down by domain (zone), across PGAM / ADV / BFPREBIID / CTV.
    Returns (df, total_revenue).
    """
    # 1) target date
    target = date or (datetime.now() - timedelta(days=1)).date().isoformat()
    api_date = target.replace('-', '')

    # 2) creds
    token       = os.getenv("FREEWHEEL_TOKEN", "")
    pgam_id     = os.getenv("FREEWHEEL_PUBLISHER_ID", "")
    adv_id      = os.getenv("FREEWHEEL_ADV_ID", "")
    bfprebid_id = os.getenv("FREEWHEEL_BFPREBID_ID", "")
    ctv_id      = os.getenv("FREEWHEEL_CTV_ID", "")

    if not (token and pgam_id and adv_id and bfprebid_id and ctv_id):
        raise EnvironmentError(
            "FREEWHEEL_TOKEN, FREEWHEEL_PUBLISHER_ID, FREEWHEEL_ADV_ID, "
            "FREEWHEEL_BFPREBID_ID and FREEWHEEL_CTV_ID must be set"
        )

    stats_url = "https://sfx.freewheel.tv/api/stats/publisher"
    ids = [pgam_id, adv_id, bfprebid_id, ctv_id]

    # 3) collect zone-level for each ID
    all_records = []
    for pid in ids:
        params = {
            "token":   token,
            "id":      pid,
            "start":   api_date,
            "end":     api_date,
            "group":   "zone",   # one row per zone/domain
            "metrics": "revenue_us_dollar,impressions,requests",
        }
        resp = requests.get(stats_url, params=params)
        if resp.status_code != 200:
            # if zone-level fails for this pid, skip it
            print(f"[WARN] zone-breakdown failed for publisher {pid} (HTTP {resp.status_code})")
            continue

        for r in resp.json().get("results", []):
            domain = r.get("zone_name") or r.get("site_name") or "<unknown>"
            rev    = float(r.get("revenue_us_dollar", 0.0))
            imp    = int(r.get("impressions", 0))
            req    = int(r.get("requests", 0))
            all_records.append({
                "Domain":          domain,
                "SSP_Dash":        rev,
                "SSP_Impressions": imp,
                "Requests":        req,
            })

    # 4) if we got nothing, fall back to day-level aggregate across all IDs
    if not all_records:
        total_rev = total_imp = total_req = 0
        for pid in ids:
            params = {
                "token":   token,
                "id":      pid,
                "start":   api_date,
                "end":     api_date,
                "group":   "day",
                "metrics": "revenue_us_dollar,impressions,requests",
            }
            r = requests.get(stats_url, params=params)
            if r.status_code != 200:
                continue
            row = r.json().get("results", [{}])[0]
            total_rev += float(row.get("revenue_us_dollar", 0.0))
            total_imp += int(row.get("impressions", 0))
            total_req += int(row.get("requests", 0))

        fill = (total_imp / total_req) if total_req else 0
        ecpm = (total_rev / total_imp * 1000) if total_imp else 0
        df = pd.DataFrame([{
            "Date":            target,
            "Domain":          "<ALL>",
            "SSP_Dash":        round(total_rev, 2),
            "SSP_Impressions": total_imp,
            "Requests":        total_req,
            "Fill Rate":       round(fill, 4),
            "SSP_eCPM":        round(ecpm, 2)
        }])
        return df, round(total_rev, 2)

    # 5) roll up across all zones
    df = pd.DataFrame(all_records)
    grouped = (
        df
        .groupby("Domain", as_index=False)
        .agg({
            "SSP_Dash":        "sum",
            "SSP_Impressions": "sum",
            "Requests":        "sum"
        })
    )
    # compute fill & eCPM
    grouped["Fill Rate"] = grouped["SSP_Impressions"] / grouped["Requests"].replace(0, 1)
    grouped["SSP_eCPM"]  = grouped["SSP_Dash"] / grouped["SSP_Impressions"].replace(0, 1) * 1000
    grouped["Date"]      = target

    # 6) reorder columns
    df_final = grouped[[
        "Date", "Domain", "SSP_Dash", "SSP_Impressions",
        "Requests", "Fill Rate", "SSP_eCPM"
    ]]
    total = df_final["SSP_Dash"].sum()
    return df_final, round(total, 2)


if __name__ == "__main__":
    df, total = fetch_freewheel_all()
    if df.empty:
        print("No data returned; check credentials / permissions.")
    else:
        print("Freewheel domain breakdown:")
        print(df.to_string(index=False))
        print(f"\nTotals: ${total:,.2f}")

        # split with partner
        partner_pct = 0.25
        partner_amt = round(total * partner_pct, 2)
        our_amt     = round(total * (1 - partner_pct), 2)

        print(f"Partner ({int(partner_pct*100)}%): ${partner_amt:,.2f}")
        print(f"Ours   ({int((1-partner_pct)*100)}%): ${our_amt:,.2f}")