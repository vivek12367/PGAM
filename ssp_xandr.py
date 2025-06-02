import os
import imaplib
import email
from email.header import decode_header
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
from dotenv import load_dotenv
import sys

# Load environment variables from .env
load_dotenv()

# Configuration via environment variables
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
EMAIL_USER  = os.getenv("EMAIL_USER")
EMAIL_PASS  = os.getenv("EMAIL_PASS")

# Validate credentials
if not EMAIL_USER or not EMAIL_PASS:
    print("ERROR: EMAIL_USER or EMAIL_PASS not set. Please configure your .env correctly.")
    sys.exit(1)


def fetch_xandr_all(date: str):
    """
    Fetch the Xandr (AppNexus) daily PGGAM report email for the given date
    (YYYY-MM-DD), parse its CSV attachment(s), and return a DataFrame plus total spend.
    """
    print(f"Connecting to IMAP server {IMAP_SERVER} as {EMAIL_USER}")

    # Connect to IMAP
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    except Exception as e:
        print(f"ERROR: Could not connect to IMAP server {IMAP_SERVER}: {e}")
        sys.exit(1)

    # Login
    try:
        mail.login(EMAIL_USER, EMAIL_PASS)
    except imaplib.IMAP4.error as e:
        print(f"ERROR: Authentication failed for {EMAIL_USER}: {e}")
        sys.exit(1)

    mail.select('INBOX')

    # Build date filter
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        print(f"ERROR: Date '{date}' is not in YYYY-MM-DD format.")
        sys.exit(1)
    since = (dt - timedelta(days=1)).strftime("%d-%b-%Y")
    before = (dt + timedelta(days=1)).strftime("%d-%b-%Y")

    # Primary search: subject + date window
    typ, data = mail.search(
        None,
        f'(SUBJECT "Your AppNexus Report PGGAM" SINCE "{since}" BEFORE "{before}")'
    )
    if typ != 'OK':
        print(f"ERROR: IMAP search failed: {typ}")
        mail.logout()
        sys.exit(1)

    ids = data[0].split()
    print(f"Found {len(ids)} emails matching date filter")
    if not ids:
        print("No date-filtered emails, falling back to subject-only search...")
        typ2, data2 = mail.search(None, 'SUBJECT', '"Your AppNexus Report PGGAM"')
        if typ2 == 'OK':
            ids = data2[0].split()
            print(f"Found {len(ids)} emails matching subject-only")
        if not ids:
            print(f"ERROR: No report email found for date {date}")
            mail.logout()
            sys.exit(1)

    latest = ids[-1]
    typ, msg_data = mail.fetch(latest, '(RFC822)')
    mail.logout()
    if typ != 'OK':
        print(f"ERROR: Failed to fetch email ID {latest}")
        sys.exit(1)

    # Parse attachments
    msg = email.message_from_bytes(msg_data[0][1])
    df_list = []
    for part in msg.walk():
        if part.get_content_maintype() == 'multipart':
            continue
        if part.get('Content-Disposition') is None:
            continue
        fname = part.get_filename()
        if fname and fname.lower().endswith('.csv'):
            print(f"Parsing attachment: {fname}")
            payload = part.get_payload(decode=True)
            try:
                df = pd.read_csv(BytesIO(payload))
                df_list.append(df)
            except Exception as e:
                print(f"WARNING: Failed to parse {fname}: {e}")

    if not df_list:
        print("ERROR: No CSV attachments found in the report email")
        sys.exit(1)

    # Combine
    df_all = pd.concat(df_list, ignore_index=True)

    # Normalize columns
    df_all = df_all.rename(columns={
        'placement':             'Placement',
        'mobile_application':    'MobileApplication',
        'total_ad_requests':     'TotalAdRequests',
        'filtered_requests':     'FilteredRequests',
        'filtered_request_rate': 'FilteredRequestRate',
        'imps_resold':           'DSP Impressions',
        'seller_revenue':        'DSP Revenue'
    })

    # Mirror DSP Impressions to SSP_Impressions for summary loader
    if 'DSP Impressions' in df_all.columns:
        df_all['SSP_Impressions'] = df_all['DSP Impressions']

    # Compute total
    total = float(df_all['DSP Revenue'].sum())
    print(f"Total DSP Revenue on {date}: {total}")
    return df_all, total


if __name__ == '__main__':
    test_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    df, total = fetch_xandr_all(test_date)
    print(df.head())
    print(f"Total Revenue: {total}")
