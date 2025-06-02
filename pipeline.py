# pipeline.py
"""
Orchestrates daily data fetch from PubMatic, Magnite, and PGAM,
combines into domain-level metrics, and outputs:
  - data/combined_metrics_{date}.csv
  - data/summary_metrics_{date}.csv

Usage:
  python pipeline.py [YYYY-MM-DD]
"""
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
# load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ssp_pubmatic import fetch_pubmatic_all
from ssp_magnite import fetch_magnite_all
from pgam_data import PGAMReportAPI


def get_target_date(arg_date=None):
    """
    Return ISO date string for target_date or yesterday local.
    """
    if arg_date:
        try:
            datetime.fromisoformat(arg_date)
        except ValueError:
            raise ValueError(f"Invalid date: {arg_date}. Use YYYY-MM-DD.")
        return arg_date
    return (datetime.now().date() - timedelta(days=1)).isoformat()


def run_pipeline(target_date=None):
    date_str = get_target_date(target_date)
    os.makedirs('data', exist_ok=True)

    # PubMatic
    pub_df, pub_total = fetch_pubmatic_all(date=date_str)
    pub_df['SSP'] = 'PubMatic'

    # Magnite
    mag_df, mag_total = fetch_magnite_all(date=date_str)
    mag_df['SSP'] = 'Magnite'

    # PGAM
    api = PGAMReportAPI()
    pgam_df, pgam_totals = api.fetch_domain_data(date=date_str)
    pgam_df = pgam_df.rename(columns={
        'DSP Spend':'SSP_Dash',
        'Impressions':'SSP_Impressions',
        'DSP Company':'SSP'
    })

    # Combine
    combined = pd.concat([pub_df, mag_df, pgam_df], ignore_index=True)
    combined.to_csv(f'data/combined_metrics_{date_str}.csv', index=False)

    # Summary
    summary = (
        combined.groupby('SSP')
        .agg(total_revenue=('SSP_Dash','sum'),
             total_impressions=('SSP_Impressions','sum'))
        .reset_index()
    )
    summary.to_csv(f'data/summary_metrics_{date_str}.csv', index=False)

    print(f"Pipeline for {date_str} completed:")
    print(f" PubMatic total: ${pub_total:,.2f}")
    print(f" Magnite total:  ${mag_total:,.2f}")
    print(f" PGAM total spend:${pgam_totals.get('dsp_spend',0):,.2f}")


if __name__ == '__main__':
    cli = sys.argv[1] if len(sys.argv)>1 else None
    run_pipeline(cli)
