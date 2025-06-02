import os
import requests
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from requests.exceptions import HTTPError, JSONDecodeError

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Load .env ─────────────────────────────────────────────────────────────────

API_BASE    = "https://ssp.pgammedia.com/api"
ADX_EX_BASE = "https://ssp.pgammedia.com/ad-exchange"

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
try:
    import streamlit as st
    TOKEN    = st.secrets["PGAM_TOKEN"]
    EMAIL    = st.secrets["PGAM_EMAIL"]
    PASSWORD = st.secrets["PGAM_PASSWORD"]
except:
    TOKEN       = os.getenv("PGAM_TOKEN")
    EMAIL       = os.getenv("PGAM_EMAIL")
    PASSWORD    = os.getenv("PGAM_PASSWORD")

DSP_NAME_MAP = {
    'HCode':           'AMX',
    'OneTag':          'One Tag',
    '33Across':        '33 Across',
    'Illumin Demand':  'Illumin',
    'Sharethrough':    'Sharethrough',
    'Sovrn':           'Sovrn',
    'InMobi':          'Inmobi',
    'Magnite':         'Magnite',
    'Unruly':          'Unruly',
    'Verve':           'Verve',
}


class PGAMReportAPI:
    def __init__(self, token=TOKEN, email=EMAIL, password=PASSWORD):
        if not token:
            raise ValueError("PGAM_TOKEN must be set")
        self.token    = token
        self.email    = email
        self.password = password
        self.headers  = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def _refresh_token(self):
        logger.debug("Refreshing PGAM token…")
        resp = requests.post(
            f"{API_BASE}/create_token",
            data={"email": self.email, "password": self.password}
        )
        resp.raise_for_status()
        new = resp.json().get("token")
        if not new:
            raise RuntimeError("Failed to refresh PGAM token")
        self.token   = new
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        logger.debug("Token refreshed successfully.")

    def _try_adx_json(self, params):
        url = f"{API_BASE}/{self.token}/adx-report"
        logger.debug("→ GET %s params=%s", url, params)
        r = requests.get(url, headers=self.headers, params=params)
        if r.status_code == 401 and self.email:
            logger.warning("401 from ADX JSON → refreshing token")
            self._refresh_token()
            r = requests.get(f"{API_BASE}/{self.token}/adx-report", headers=self.headers, params=params)
        if r.status_code == 200:
            try:
                return r.json()
            except JSONDecodeError:
                logger.error("ADX JSON returned non-JSON")
        else:
            logger.warning("ADX JSON HTTP %s: %s", r.status_code, r.text)
        return None

    def _request_publisher(self, params):
        url = f"{API_BASE}/{self.token}/report"
        logger.debug("→ GET %s params=%s", url, params)
        r = requests.get(url, headers=self.headers, params=params)
        if r.status_code == 401 and self.email:
            logger.warning("401 from Publisher → refreshing token")
            self._refresh_token()
            r = requests.get(f"{API_BASE}/{self.token}/report", headers=self.headers, params=params)
        try:
            r.raise_for_status()
            return r.json() or {"data": [], "totalPages": 1}
        except HTTPError:
            logger.error("Publisher HTTPError %s: %s", r.status_code, r.text)
            return {"data": [], "totalPages": 1}

    def fetch_summary(self, date=None):
        """
        Returns per‐DSP totals:
          [Date, DSP Company, DSP Spend, Impressions, SSP Revenue, Profit, Margin]
        """
        target = date or (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()

        # 1) ADX JSON with SSP Revenue
        adx_params = {
            "from":        target,
            "to":          target,
            "day_group":   "day",
            "attribute[]": ["company_dsp"],
            "metric[]":    ["dsp_spend", "ssp_revenue", "impressions"],
            "limit":       200,
            "page":        1,
        }
        resp = self._try_adx_json(adx_params)
        if resp and "data" in resp:
            rows = resp["data"]
            for p in range(2, resp.get("totalPages", 1) + 1):
                adx_params["page"] = p
                nxt = self._try_adx_json(adx_params)
                if nxt and "data" in nxt:
                    rows.extend(nxt["data"])
            return self._aggregate(rows, target)

        # 2) Fallback to Publisher
        logger.warning("ADX JSON w/ ssp_revenue failed → falling back to Publisher API")
        pub_params = {
            "from":        target,
            "to":          target,
            "day_group":   "day",
            "attribute[]": ["company_dsp"],
            "metric[]":    ["dsp_spend", "impressions", "publisher_revenue"],
            "limit":       200,
            "page":        1,
        }
        resp = self._request_publisher(pub_params)
        rows = resp.get("data", [])
        for p in range(2, resp.get("totalPages", 1) + 1):
            pub_params["page"] = p
            more = self._request_publisher(pub_params)
            rows.extend(more.get("data", []))
        # convert publisher_revenue → ssp_revenue for aggregation
        for r in rows:
            r["ssp_revenue"] = r.get("publisher_revenue", 0)
        return self._aggregate(rows, target)

    def fetch_dsp_details(self, date=None):
        """
        Returns per‐DSP, per‐domain detail:
          [Date, DSP Company, Domain / Bundle, DSP Spend, Impressions,
           Publisher Revenue, Profit, Margin, DSP eCPM]
        """
        target = date or (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()

        params = {
            "from":        target,
            "to":          target,
            "day_group":   "day",
            "attribute[]": ["company_dsp", "domain"],
            "metric[]":    ["dsp_spend", "impressions", "publisher_revenue"],
            "limit":       5000,
            "page":        1,
        }
        resp = self._request_publisher(params)
        rows = resp.get("data", [])
        records = []
        for r in rows:
            raw   = str(r.get("company_dsp") or "").split('#', 1)[0].strip()
            dsp   = DSP_NAME_MAP.get(raw, raw)
            dom   = r.get("domain") or "<ALL>"
            spend = float(r.get("dsp_spend") or 0)
            imps  = int(r.get("impressions") or 0)
            profit= float(r.get("publisher_revenue") or 0)  # platform profit
            ecpm  = (spend / imps * 1000) if imps else 0
            margin= (profit / spend * 100) if spend else 0

            records.append({
                "Date":              target,
                "DSP Company":       dsp,
                "Domain / Bundle":   dom,
                "DSP Spend":         round(spend, 2),
                "Impressions":       imps,
                "Publisher Revenue": round(profit, 2),
                "Profit":            round(profit, 2),
                "Margin":            round(margin, 2),
                "DSP eCPM":          round(ecpm, 2),
            })

        return pd.DataFrame(records)

    def _aggregate(self, recs, target):
        stats = {}
        totals = {"dsp_spend": 0.0, "ssp_revenue": 0.0, "impressions": 0}

        for rec in recs:
            raw   = str(rec.get("company_dsp") or "").split('#',1)[0].strip()
            dsp   = DSP_NAME_MAP.get(raw, raw)
            spend = float(rec.get("dsp_spend")    or 0)
            ssp   = float(rec.get("ssp_revenue")  or 0)
            imps  = int(rec.get("impressions")    or 0)

            ent = stats.setdefault(dsp, {
                "DSP Spend":    0.0,
                "SSP Revenue":  0.0,
                "Impressions":  0,
            })
            ent["DSP Spend"]   += spend
            ent["SSP Revenue"] += ssp
            ent["Impressions"] += imps

            totals["dsp_spend"]   += spend
            totals["ssp_revenue"] += ssp
            totals["impressions"] += imps

        rows = []
        for dsp, v in stats.items():
            dsp_spend = round(v["DSP Spend"], 2)
            ssp_rev   = round(v["SSP Revenue"], 2)
            profit    = round(dsp_spend - ssp_rev, 2)
            margin    = round((profit / dsp_spend * 100) if dsp_spend else 0, 2)
            rows.append({
                "Date":          target,
                "DSP Company":   dsp,
                "DSP Spend":     dsp_spend,
                "Impressions":   v["Impressions"],
                "SSP Revenue":   ssp_rev,
                "Profit":        profit,
                "Margin":        margin,
            })

        df = pd.DataFrame(rows)

        t_spend  = round(totals["dsp_spend"], 2)
        t_ssp    = round(totals["ssp_revenue"], 2)
        t_profit = round(t_spend - t_ssp, 2)
        t_margin = round((t_profit / t_spend * 100) if t_spend else 0, 2)

        return df, {
            "dsp_spend":   t_spend,
            "ssp_revenue": t_ssp,
            "profit":      t_profit,
            "margin":      t_margin,
            "impressions": totals["impressions"],
        }


if __name__ == "__main__":
    api = PGAMReportAPI()

    # summary
    df, totals = api.fetch_summary()
    print("=== Summary ===")
    print(df.to_string(index=False))
    print(
        f"Totals: dsp_spend=${totals['dsp_spend']:.2f}, "
        f"ssp_revenue=${totals['ssp_revenue']:.2f}, "
        f"profit=${totals['profit']:.2f}, "
        f"margin={totals['margin']:.2f}%, "
        f"impressions={totals['impressions']}"
    )

    # domain‐level details
    detail_df = api.fetch_dsp_details()
    print("\n=== Domain‐level details ===")
    print(detail_df.to_string(index=False))