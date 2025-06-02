import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def fetch_adaptmx_all(date=None, exclude_domains=None):
    """
    Fetch net revenue and impressions for AdaptMX (AppMonet) for a given date.
    Returns (df, total_revenue) where df has columns [Date, Domain, SSP_Dash, SSP_Impressions].
    """
    api_key = os.getenv("ADAPTMX_KEY")
    if not api_key:
        raise EnvironmentError("ADAPTMX_KEY is missing from environment")

    # Determine target and start dates
    target = date or (datetime.now() - timedelta(days=1)).date().isoformat()
    start  = (datetime.fromisoformat(target) - timedelta(days=1)).isoformat()[:10]

    # Setup retries for HTTP 202 responses
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[202])
    adapter = HTTPAdapter(max_retries=retries)
    session = requests.Session()
    session.headers.update({
        'x-api-key': api_key,
        'accept': 'application/json, text/plain, */*'
    })
    session.mount('https://', adapter)

    # Build query parameters
    params = {
        'dimensions[]': 'hourly',
        'metrics[]':    ['impressions', 'revenue'],
        'tz':           'UTC',
        'sort_by':      'hourly',
        'sort_dir':     'desc',
        'start_date':   start,
        'end_date':     target
    }

    # Request asynchronous report URL
    base_url = 'https://dash-api.appmonet.com/api/v2/reporting/async'
    resp = session.get(base_url, params=params)
    resp.raise_for_status()
    report_url = resp.url

    # Extract filename from URL path without query parameters
    parsed   = urlparse(report_url)
    filename = os.path.basename(parsed.path)
    os.makedirs('reports', exist_ok=True)
    path     = os.path.join('reports', filename)

    # Download report
    dl = session.get(report_url, stream=True)
    dl.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in dl.iter_content(8192):
            f.write(chunk)

    # Parse CSV and compute totals
    df_raw = pd.read_csv(path, skiprows=2)
    # Derive Date column
    if 'Hour' in df_raw.columns:
        df_raw['Date'] = pd.to_datetime(df_raw['Hour']).dt.date.astype(str)
    else:
        df_raw['Date'] = df_raw.iloc[:,0].astype(str)

    # Filter for target date
    day_df = df_raw[df_raw['Date'] == target]
    revenue_total     = day_df['Revenue'].sum() if 'Revenue' in day_df else 0.0
    impressions_total = day_df['Impressions'].sum() if 'Impressions' in day_df else 0

    # Build output DataFrame
    df = pd.DataFrame([{
        'Date':            target,
        'Domain':          '',
        'SSP_Dash':        round(revenue_total, 2),
        'SSP_Impressions': int(impressions_total)
    }])

    return df, round(revenue_total, 2)


if __name__ == '__main__':
    df, total = fetch_adaptmx_all()
    print(df.to_string(index=False))
    print(f"Totals: ${total:,.2f}")
