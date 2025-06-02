#!/usr/bin/env python3
import os
import importlib
import pandas as pd
from datetime import datetime, timedelta, timezone

# ensure your PYTHONPATH includes the directory with these modules
from pgam_data import PGAMReportAPI

# Map partner names to their module and fetch function
PARTNERS = {
    'Pubmatic': ('ssp_pubmatic', 'fetch_pubmatic_all'),
    'Magnite':  ('ssp_magnite',  'fetch_magnite_all'),
    '33 Across':('ssp_33across','fetch_33across_all'),
    # add more partners here as needed
}


def fetch_partner_metrics(date=None, exclude_domains=None):
    """
    Fetch per-domain metrics from both the SSP modules and PGAMReportAPI,
    merge on Date+Partner+Domain, and return a combined DataFrame.
    """
    # determine target date
    target_date = date or (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()

    # 1) Fetch PGAM details (domain-level PGAM revenue & eCPM)
    api = PGAMReportAPI()
    pgam_df = api.fetch_dsp_details(date=target_date)
    pgam_df = pgam_df.rename(columns={
        'DSP Company': 'Partner',
        'Domain / Bundle': 'Domain',
        'DSP Spend': 'PGAM_Revenue',
        'DSP eCPM': 'PGAM_eCPM'
    })
    # fill missing PGAM values
    pgam_df['PGAM_Revenue'] = pgam_df['PGAM_Revenue'].fillna(0)
    pgam_df['PGAM_eCPM']   = pgam_df['PGAM_eCPM'].fillna(0)

    combined = []
    for partner, (mod_name, fn_name) in PARTNERS.items():
        # import the SSP module dynamically
        module = importlib.import_module(mod_name)
        # call its fetch function
        ssp_df, _ = getattr(module, fn_name)(date=target_date, exclude_domains=exclude_domains)
        # rename for consistency
        ssp_df = ssp_df.rename(columns={
            'Domain': 'Domain',
            'SSP_Dash': 'SSP_Revenue',
            'SSP_eCPM': 'SSP_eCPM'
        })
        ssp_df['Partner'] = partner

        # filter PGAM rows for this partner
        pg = pgam_df[pgam_df['Partner'] == partner]
        # merge
        merged = pd.merge(
            ssp_df,
            pg[['Date','Partner','Domain','PGAM_Revenue','PGAM_eCPM']],
            on=['Date','Partner','Domain'], how='outer'
        )
        # fill zeros for any missing metric
        for col in ['SSP_Revenue','SSP_eCPM','PGAM_Revenue','PGAM_eCPM']:
            merged[col] = merged[col].fillna(0)

        combined.append(merged)

    # concatenate all partners
    result_df = pd.concat(combined, ignore_index=True)
    return result_df


if __name__ == '__main__':
    df = fetch_partner_metrics()
    # print to console
    print(df.to_string(index=False))
    # also save to CSV for easy review
    out_name = f"partner_domain_metrics_{datetime.now(timezone.utc).date().isoformat()}.csv"
    df.to_csv(out_name, index=False)
    print(f"\nWritten combined metrics to {out_name}\n")
