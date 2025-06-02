#!/usr/bin/env python3
import os
import time
import json
import argparse
import hashlib
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

# ─── CONFIG ────────────────────────────────────────────────────────────────────────
try:
    import streamlit as st
    API_KEY    = st.secrets["MAGNITE_ACCESS_KEY"]
    API_SECRET = st.secrets["MAGNITE_SECRET_KEY"]
    PUBLISHER  = st.secrets["MAGNITE_PUBLISHER_ID"]
except:
    load_dotenv()
    API_KEY    = os.getenv("MAGNITE_ACCESS_KEY")
    API_SECRET = os.getenv("MAGNITE_SECRET_KEY")
    PUBLISHER  = os.getenv("MAGNITE_PUBLISHER_ID")
BASE_URL   = "https://api.rubiconproject.com/analytics/v2/default"
CACHE_FILE = Path(__file__).with_suffix(".cache.json")

if not (API_KEY and API_SECRET and PUBLISHER):
    raise EnvironmentError("Set MAGNITE_ACCESS_KEY, MAGNITE_SECRET_KEY & MAGNITE_PUBLISHER_ID")

# ─── CACHE HELPERS ─────────────────────────────────────────────────────────────────
def _load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}

def _save_cache(c):
    CACHE_FILE.write_text(json.dumps(c, indent=2))

# ─── DATE HELPERS ──────────────────────────────────────────────────────────────────
def _yesterday_iso():
    return (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()

# ─── OFFLINE REPORT POLLING ────────────────────────────────────────────────────────
def _kickoff_and_poll(criteria: dict, cache_key: str, max_attempts=60, wait_secs=5) -> str:
    cache = _load_cache()
    rid   = cache.get(cache_key)
    account_qs = f"publisher/{PUBLISHER}"

    # reuse existing report if it's already successful
    if rid:
        st = requests.get(f"{BASE_URL}/{rid}?account={account_qs}", auth=(API_KEY, API_SECRET))
        if st.ok and st.json().get("status") == "success":
            return rid

    # 1) submit offline report request
    resp = requests.post(
        f"{BASE_URL}?account={account_qs}",
        auth=(API_KEY, API_SECRET),
        json={"criteria": criteria}
    )
    resp.raise_for_status()
    rid = resp.json().get("offline_report_id")
    if not rid:
        raise RuntimeError("No offline_report_id returned from Magnite")

    # 2) poll until completion
    for i in range(1, max_attempts + 1):
        status_resp = requests.get(f"{BASE_URL}/{rid}?account={account_qs}", auth=(API_KEY, API_SECRET))
        status_resp.raise_for_status()
        status = status_resp.json().get("status")
        print(f"[{i}/{max_attempts}] report status: {status}")
        if status == "success":
            cache[cache_key] = rid
            _save_cache(cache)
            return rid
        if status == "failed":
            raise RuntimeError("Offline report failed on Magnite side")
        time.sleep(wait_secs)

    raise RuntimeError(f"Report never succeeded (last status={status})")

# ─── FULL DOMAIN-LEVEL REPORT ───────────────────────────────────────────────────────
def fetch_magnite_all(
    date: str = None,
    exclude_domains: list[str] = None,
    timezone_str: str = "America/Los_Angeles",
    max_attempts: int = 60,
    wait_secs: int = 5
) -> tuple[pd.DataFrame, float]:
    target     = date or _yesterday_iso()
    excludes   = {d.lower().strip() for d in (exclude_domains or [])}
    # include dimension, metric, timezone in cache key to avoid stale RIDs
    criteria = {
        "dimension":  "site,date,referring_domain",
        "metric":     "seller_net_revenue,paid_impression",
        "limit":      500_000,
        "currency":   "USD",
        "start":      f"{target}T00:00:00Z",
        "end":        f"{target}T23:59:59Z",
        "timezone":   timezone_str
    }
    key_string = json.dumps(criteria, sort_keys=True)
    cache_key  = "full:" + hashlib.sha256(key_string.encode()).hexdigest()

    rid = _kickoff_and_poll(criteria, cache_key, max_attempts, wait_secs)

    account_qs = f"publisher/{PUBLISHER}"
    # get pagination metadata
    meta = requests.get(
        f"{BASE_URL}/{rid}/data?account={account_qs}&format=json&page=1&size=1",
        auth=(API_KEY, API_SECRET)
    ).json().get("page", {})
    total_pages = meta.get("total_pages", 1)

    rows = []
    for page in range(1, total_pages + 1):
        url = (
            f"{BASE_URL}/{rid}/data"
            f"?account={account_qs}"
            f"&format=json"
            f"&page={page}"
            f"&size=50000"
        )
        r = requests.get(url, auth=(API_KEY, API_SECRET))
        if r.status_code == 400:
            continue
        r.raise_for_status()
        items = r.json().get("content") or r.json().get("data", {}).get("items", [])
        for rec in items:
            dom = (rec.get("referring_domain") or rec.get("site") or "").lower().strip()
            if dom in excludes:
                continue
            raw_rev = float(rec.get("seller_net_revenue") or 0)
            rows.append({
                "Date":             rec.get("date", target),
                "Domain":           dom or "<ALL>",
                "raw_revenue":      raw_rev,
                "SSP_Dash":         round(raw_rev, 2),
                "SSP_Impressions":  int(rec.get("paid_impression") or 0),
            })

    df = pd.DataFrame(rows)
    df["SSP_eCPM"] = (df["SSP_Dash"] / df["SSP_Impressions"].replace(0,1) * 1000).round(2)
    total = df["raw_revenue"].sum().round(2)
    return df, total

# ─── HOURLY REPORT ──────────────────────────────────────────────────────────────────
def fetch_magnite_hourly(
    date: str = None,
    timezone_str: str = "America/Los_Angeles",
    max_attempts: int = 60,
    wait_secs: int = 5
) -> pd.DataFrame:
    target   = date or _yesterday_iso()
    criteria = {
        "dimension":  "date,hour",
        "metric":     "seller_net_revenue,paid_impression",
        "limit":      500_000,
        "currency":   "USD",
        "start":      f"{target}T00:00:00Z",
        "end":        f"{target}T23:59:59Z",
        "timezone":   timezone_str
    }
    # cache key for hourly
    key_string = json.dumps(criteria, sort_keys=True)
    cache_key  = "hourly:" + hashlib.sha256(key_string.encode()).hexdigest()

    rid = _kickoff_and_poll(criteria, cache_key, max_attempts, wait_secs)

    account_qs = f"publisher/{PUBLISHER}"
    meta = requests.get(
        f"{BASE_URL}/{rid}/data?account={account_qs}&format=json&page=1&size=1",
        auth=(API_KEY, API_SECRET)
    ).json().get("page", {})
    total_pages = meta.get("total_pages", 1)

    rows = []
    for page in range(1, total_pages + 1):
        url = (
            f"{BASE_URL}/{rid}/data"
            f"?account={account_qs}"
            f"&format=json"
            f"&page={page}"
            f"&size=100000"
        )
        r = requests.get(url, auth=(API_KEY, API_SECRET))
        if r.status_code == 400:
            continue
        r.raise_for_status()
        items = r.json().get("content") or r.json().get("data", {}).get("items", [])
        for rec in items:
            rows.append({
                "Date":              rec.get("date", target),
                "Hour":              int(rec.get("hour", 0)),
                "Publisher_Net_USD": round(float(rec.get("seller_net_revenue") or 0), 2),
                "Paid_Impressions":  int(rec.get("paid_impression") or 0),
            })

    df = pd.DataFrame(rows).sort_values(["Date", "Hour"]).reset_index(drop=True)
    return df

# ─── CLI ENTRYPOINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Magnite offline reports")
    parser.add_argument("--date", help="YYYY-MM-DD (default=yesterday)")
    parser.add_argument("--exclude", nargs="*", default=[], help="Domains to exclude (full report only)")
    parser.add_argument(
        "--hourly",
        action="store_true",
        help="Fetch hourly Publisher Net & Impressions"
    )
    args = parser.parse_args()

    if args.hourly:
        df_hourly = fetch_magnite_hourly(date=args.date)
        if df_hourly.empty:
            print("No hourly data returned. Check date or permissions.")
        else:
            print(df_hourly.to_string(index=False))
    else:
        df_full, total = fetch_magnite_all(
            date=args.date,
            exclude_domains=args.exclude
        )
        if df_full.empty:
            print("No data returned. Check date, excludes, or existence.")
        else:
            print(df_full.to_string(index=False))
            print(f"\nTotal Publisher Net: ${total:,.2f}")
