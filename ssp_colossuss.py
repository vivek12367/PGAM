import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ─── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()

BASE_URL = "https://supply.colossusssp.com/api"

def _get_token():
    email    = os.getenv("COLOSSUS_EMAIL")
    password = os.getenv("COLOSSUS_PASSWORD")
    if not (email and password):
        raise EnvironmentError("COLOSSUS_EMAIL and COLOSSUS_PASSWORD must be set")
    resp = requests.post(f"{BASE_URL}/create_token",
                         data={"email": email, "password": password})
    resp.raise_for_status()
    token = resp.json().get("token")
    if not token:
        raise RuntimeError("Failed to obtain Colossus token")
    return token

def list_colossus_inventory():
    """
    GET /{token}/list_inventory
    Returns a DataFrame of all inventory slots.
    Columns include: inventory_id, title, platform, address, language, status, ...
    """
    token = _get_token()
    url   = f"{BASE_URL}/{token}/list_inventory"
    resp  = requests.get(url)
    # 204 No Content → empty list
    if resp.status_code == 204:
        inventory = []
    else:
        resp.raise_for_status()
        inventory = resp.json() or []

    # Normalize to DataFrame
    df = pd.json_normalize(inventory)
    # Flatten nested arrays into comma‐separated strings
    for col in ("categories","adcategory_support","blocked_categories","rtb_version","blocked_domains"):
        if col in df:
            df[col] = df[col].apply(lambda arr: ",".join(arr) if isinstance(arr,list) else "")
    return df

def get_colossus_inventory(inventory_id):
    """
    GET /{token}/inventory?inventory_id=<id>
    Returns a dict of the inventory object, or None if 204.
    """
    token = _get_token()
    url   = f"{BASE_URL}/{token}/inventory"
    resp  = requests.get(url, params={"inventory_id": inventory_id})
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    return resp.json()

def fetch_colossus_all(date=None, exclude_domains=None):
    """
    Your existing aggregate fetch:
      GET /{token}/report?from=<date>&to=<date>
    Returns (df_agg, total_rev).
    """
    target = date or (datetime.now() - timedelta(days=1)).date().isoformat()
    excludes = {d.lower().strip() for d in (exclude_domains or [])}

    token = _get_token()
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(f"{BASE_URL}/{token}/report",
                        headers=headers,
                        params={"from": target, "to": target})
    resp.raise_for_status()
    data = resp.json().get("data", [])

    # Aggregate row
    total_rev, total_imps, total_req, total_fill = 0.0, 0, 0, 0.0
    for rec in data:
        total_rev  += float(rec.get("money") or 0)
        total_imps += int(rec.get("impressions") or 0)
        total_req  += int(rec.get("requests") or 0)
        total_fill += float(rec.get("fill_rate") or 0)

    row = {
        "Date":            target,
        "Domain":          "",  # no per-domain in this API
        "SSP_Dash":        round(total_rev, 2),
        "SSP_Impressions": total_imps,
        "Requests":        total_req,
        "Fill Rate":       round(total_fill, 2),
    }
    row["SSP_eCPM"] = round((row["SSP_Dash"] / max(1, row["SSP_Impressions"])) * 1000, 2)
    df = pd.DataFrame([row], columns=[
        "Date","Domain","SSP_Dash","SSP_Impressions","Requests","Fill Rate","SSP_eCPM"
    ])
    return df, round(total_rev, 2)

if __name__ == "__main__":
    # Example: list all inventory slots
    inv_df = list_colossus_inventory()
    print("=== Inventory List ===")
    print(inv_df[["inventory_id","title","platform","address"]].to_string(index=False))

    # Example: look up a single inventory
    if not inv_df.empty:
        inv_id = inv_df.loc[0,"inventory_id"]
        print(f"\n=== Info for inventory {inv_id} ===")
        print(get_colossus_inventory(inv_id))

    # Example: run your aggregate report
    print("\n=== Aggregate Report ===")
    df, total = fetch_colossus_all()
    print(df.to_string(index=False))
    print(f"Totals: dsp_spend=${total:,.2f}")
