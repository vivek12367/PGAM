import os
import importlib
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import UnidentifiedImageError
from dotenv import load_dotenv
from streamlit_chat import message
from ssp_xandr import fetch_xandr_all
from langchain_experimental.agents import create_pandas_dataframe_agent
from openai import OpenAI
import json

load_dotenv()
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

from pgam_data import PGAMReportAPI

# ────────────────────────────────────────────────────────────────────────────────
# 1) Page config ─────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="PGAM Revenue Dashboard",
    page_icon="📊"
)

LOGO_PATH = "logo.png"

# ────────────────────────────────────────────────────────────────────────────────
# 2) CSS loader ─────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# ────────────────────────────────────────────────────────────────────────────────
# 3) Sidebar (now allowing date range) ────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
yesterday = (datetime.now() - timedelta(days=1)).date()
selected_dates = st.sidebar.date_input(
    "Select Date or Date Range",
    [yesterday, yesterday]  # default to a single day as a list of two identical dates
)

if isinstance(selected_dates, (list, tuple)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    # If user selects only one date, treat it as both start and end
    start_date = end_date = selected_dates

start_str = start_date.isoformat()
end_str = end_date.isoformat()

page = st.sidebar.radio("Go to", ["Overview", "Details", "PGAM Data", "Trends", "PGAMbot"])


# ────────────────────────────────────────────────────────────────────────────────
# 4) Cached helpers & data loaders (for single-day and range) ────────────────────
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data
def get_xandr_all_cached(target_date: str):
    """
    Cached wrapper around fetch_xandr_all to avoid re-fetching Xandr data.
    """
    return fetch_xandr_all(date=target_date)


@st.cache_data
def load_ssp_summary(target_date: str) -> pd.DataFrame:
    """
    Fetch per-SSP summary (Dash Net & Impressions) across all SSP modules for a single date.
    """
    modules = {
        'Pubmatic':     ('ssp_pubmatic',    'fetch_pubmatic_all'),
        'Magnite':      ('ssp_magnite',     'fetch_magnite_all'),
        '33 Across':    ('ssp_33across',    'fetch_33across_all'),
        'AdaptMX':      ('ssp_AdaptMX',     'fetch_adaptmx_all'),
        'Colossus':     ('ssp_colossuss',   'fetch_colossus_all'),
        'Freewheel':    ('ssp_freewheel',   'fetch_freewheel_all'),
        'Illumin':      ('ssp_illumin',     'fetch_illumin_all'),
        'One Tag':      ('ssp_onetag',      'fetch_onetag_all'),
        'Sharethrough': ('ssp_sharethrough','fetch_sharethrough_all'),
        'Sovrn':        ('ssp_sovrn',       'fetch_sovrn_all'),
        'Unruly':       ('ssp_unruly',      'fetch_unruly_all'),
        'Xandr':        ('ssp_xandr',       'fetch_xandr_all')
    }
    rows = []

    def fetch(ssp: str, module: str, fn_name: str):
        m = importlib.import_module(module)
        df, total = getattr(m, fn_name)(date=target_date)
        if 'impressions' in df.columns and 'SSP_Impressions' not in df.columns:
            df = df.rename(columns={'impressions': 'SSP_Impressions'})
        ssp_imps = int(df.get('SSP_Impressions', 0).sum())
        return ssp, float(total), ssp_imps

    with ThreadPoolExecutor(max_workers=5) as exe:
        futures = {
            exe.submit(fetch, ssp, mod, fn): ssp
            for ssp, (mod, fn) in modules.items()
        }
        for fut in as_completed(futures):
            ssp = futures[fut]
            try:
                _, rev, imps = fut.result()
                rows.append({
                    'SSP': ssp,
                    'SSP Dash Net': rev,
                    'SSP Impressions': imps
                })
            except Exception as e:
                st.error(f"Error loading {ssp} summary: {e}")

    return pd.DataFrame(rows)


@st.cache_data
def load_ssp_summary_range(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Combine SSP summaries over a date range by summing metrics.
    """
    all_dfs = []
    for d in pd.date_range(start_date, end_date):
        df_single = load_ssp_summary(d.date().isoformat())
        all_dfs.append(df_single)
    concatenated = pd.concat(all_dfs, ignore_index=True)
    aggregated = (
        concatenated
        .groupby('SSP')
        .agg({'SSP Dash Net': 'sum', 'SSP Impressions': 'sum'})
        .reset_index()
    )
    return aggregated


@st.cache_data
def load_pgam_data(target_date: str):
    """
    Fetch PGAM API summary (DSP Revenue, DSP Impressions, etc.) for a single date,
    merge with SSP summary, and compute derived columns.
    Returns (merged_df, totals_dict).
    """
    api = PGAMReportAPI()
    dsp_df, totals = api.fetch_summary(date=target_date)

    dsp_df = dsp_df.rename(columns={
        'DSP Company': 'SSP',
        'DSP Spend':   'DSP Revenue',
        'Impressions': 'DSP Impressions'
    })
    dsp_df['SSP'] = dsp_df['SSP'].astype(str).str.strip()

    s = load_ssp_summary(target_date)[['SSP', 'SSP Dash Net', 'SSP Impressions']]
    merged = dsp_df.merge(s, on='SSP', how='left').fillna({
        'SSP Dash Net':    0,
        'SSP Impressions': 0
    })

    # Overrides for certain SSPs
    mask_override = merged['SSP'].isin(['Stirista', 'OTTA Unruly', 'Xandr'])
    merged.loc[mask_override, 'SSP Dash Net'] = merged.loc[mask_override, 'DSP Revenue']
    merged.loc[mask_override, 'SSP Impressions'] = merged.loc[mask_override, 'DSP Impressions']

    # Freewheel override
    fw = merged['SSP'] == 'Freewheel'
    merged.loc[fw, 'DSP Revenue']     = merged.loc[fw, 'SSP Dash Net']
    merged.loc[fw, 'DSP Impressions'] = merged.loc[fw, 'SSP Impressions']

    # Derived columns
    merged['Revenue Difference'] = merged['SSP Dash Net'] - merged['DSP Revenue']
    merged['Total Spend']        = merged['DSP Revenue'] + merged['SSP Dash Net']
    merged['Profit']             = merged['DSP Revenue'] - merged['SSP Dash Net']
    merged['Margin']             = (
        merged['Profit'] /
        merged['DSP Revenue'].replace(0, 1)
    ).round(4)

    return merged, totals


@st.cache_data
def load_pgam_data_range(start_date: str, end_date: str):
    """
    Combine PGAM data summary over a date range by summing metrics.
    Returns (aggregated_df, aggregated_totals).
    """
    merged_list = []
    totals_accum = {
        'dsp_spend':   0.0,
        'ssp_revenue': 0.0,
        'profit':      0.0,
        'margin':      0.0,
        'impressions': 0
    }

    for d in pd.date_range(start_date, end_date):
        merged_single, totals_single = load_pgam_data(d.date().isoformat())
        merged_list.append(merged_single)
        for key in totals_accum:
            totals_accum[key] += totals_single[key]

    concatenated = pd.concat(merged_list, ignore_index=True)
    aggregated = (
        concatenated
        .groupby('SSP')
        .agg({
            'DSP Revenue':    'sum',
            'DSP Impressions': 'sum',
            'SSP Dash Net':   'sum',
            'SSP Impressions':'sum',
            'Revenue Difference': 'sum',
            'Total Spend':    'sum',
            'Profit':         'sum'
        })
        .reset_index()
    )
    aggregated['Margin'] = (
        aggregated['Profit'] /
        aggregated['DSP Revenue'].replace(0, 1)
    ).round(4)

    return aggregated, totals_accum


@st.cache_data
def load_pgam_api(target_date: str):
    """
    Simple wrapper to fetch PGAM API raw summary (no merging) for a single date.
    """
    api = PGAMReportAPI()
    return api.fetch_summary(date=target_date)


@st.cache_data
def load_pgam_api_range(start_date: str, end_date: str):
    """
    Combine raw PGAM API data over a date range.
    Returns (concatenated_df, aggregated_totals).
    """
    dfs = []
    totals_accum = {
        'dsp_spend':   0.0,
        'ssp_revenue': 0.0,
        'profit':      0.0,
        'margin':      0.0,
        'impressions': 0
    }

    for d in pd.date_range(start_date, end_date):
        df_single, totals_single = load_pgam_api(d.date().isoformat())
        dfs.append(df_single)
        for key in totals_accum:
            totals_accum[key] += totals_single[key]

    concatenated = pd.concat(dfs, ignore_index=True)
    return concatenated, totals_accum


@st.cache_data
def load_ssp_details(target_date: str) -> pd.DataFrame:
    """
    Fetch detailed per-domain metrics for a single date, then attach PGAM details.
    """
    ssps = load_ssp_summary(target_date)['SSP'].tolist()
    mapping = {
        'Pubmatic':     ('ssp_pubmatic',    'fetch_pubmatic_all'),
        'Magnite':      ('ssp_magnite',     'fetch_magnite_all'),
        '33 Across':    ('ssp_33across',    'fetch_33across_all'),
        'AdaptMX':      ('ssp_AdaptMX',     'fetch_adaptmx_all'),
        'Colossus':     ('ssp_colossuss',   'fetch_colossus_all'),
        'Freewheel':    ('ssp_freewheel',   'fetch_freewheel_all'),
        'Illumin':      ('ssp_illumin',     'fetch_illumin_all'),
        'One Tag':      ('ssp_onetag',      'fetch_onetag_all'),
        'Sharethrough': ('ssp_sharethrough','fetch_sharethrough_all'),
        'Sovrn':        ('ssp_sovrn',       'fetch_sovrn_all'),
        'Unruly':       ('ssp_unruly',      'fetch_unruly_all'),
        'Xandr':        ('ssp_xandr',       'fetch_xandr_all')
    }
    parts = []

    for ssp in ssps:
        if ssp not in mapping:
            continue
        mod, fn = mapping[ssp]
        try:
            m = importlib.import_module(mod)
            df_ssp, _ = getattr(m, fn)(date=target_date)
            df_ssp = df_ssp.rename(columns={
                'Domain / Bundle': 'Domain',
                'SSP_Dash':        'SSP Revenue',
                'SSP_eCPM':        'SSP eCPM'
            })
            for c in ['Date', 'Domain', 'SSP Revenue', 'SSP eCPM']:
                if c not in df_ssp.columns:
                    df_ssp[c] = 0
            df_ssp['DSP Company'] = ssp
            parts.append(df_ssp[['Date', 'DSP Company', 'Domain', 'SSP Revenue', 'SSP eCPM']])
        except Exception:
            continue

    pg = PGAMReportAPI().fetch_dsp_details(date=target_date)
    pg = pg.rename(columns={
        'Domain / Bundle': 'Domain',
        'DSP Spend':       'PGAM Revenue',
        'DSP eCPM':        'PGAM eCPM'
    })
    for c in ['Date', 'Domain', 'PGAM Revenue', 'PGAM eCPM']:
        if c not in pg.columns:
            pg[c] = 0
    parts.append(pg[['Date', 'DSP Company', 'Domain', 'PGAM Revenue', 'PGAM eCPM']])

    raw = pd.concat(parts, ignore_index=True)
    desired = ['SSP Revenue', 'SSP eCPM', 'PGAM Revenue', 'PGAM eCPM']
    values = [c for c in desired if c in raw.columns]

    combined = (
        raw
        .pivot_table(
            index=['Date', 'DSP Company', 'Domain'],
            values=values,
            aggfunc='first'
        )
        .reset_index()
        .fillna(0)
    )
    return combined


@st.cache_data
def load_ssp_details_range(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Combine SSP details over a date range into a single DataFrame.
    """
    dfs = [
        load_ssp_details(d.date().isoformat())
        for d in pd.date_range(start_date, end_date)
    ]
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=['Date', 'DSP Company', 'Domain', 'SSP Revenue', 'SSP eCPM', 'PGAM Revenue', 'PGAM eCPM'])


@st.cache_data
def load_partner_gaps(target_date: str) -> pd.DataFrame:
    """
    Computes per-partner, per-domain gap for a single date.
    """
    df = load_ssp_details(target_date)

    def compute_gaps(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        avg = sub['SSP eCPM'].mean()
        sub['gap'] = sub['SSP eCPM'] - avg
        return sub[['DSP Company', 'Domain', 'SSP eCPM', 'gap']]

    return pd.concat(
        [compute_gaps(group) for _, group in df.groupby('DSP Company')],
        ignore_index=True
    )


# ────────────────────────────────────────────────────────────────────────────────
# 5) Header helper ─────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
def render_header(title: str):
    c1, c2 = st.columns([1, 9])
    if os.path.exists(LOGO_PATH):
        try:
            c1.image(LOGO_PATH, width=60)
        except UnidentifiedImageError:
            pass
    c2.markdown(
        f"<h1 style='color:var(--pgam-primary)'>{title}</h1>",
        unsafe_allow_html=True
    )


# ────────────────────────────────────────────────────────────────────────────────
# 6) Load global data once per session for range ───────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
data_ssp, _      = load_ssp_summary_range(start_str, end_str), None
data_pgam, _     = load_pgam_data_range(start_str, end_str)
data_pgam_api, _ = load_pgam_api_range(start_str, end_str)
data_ssp_details = load_ssp_details_range(start_str, end_str)


# ────────────────────────────────────────────────────────────────────────────────
# 7) Overview Page ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
def overview_page():
    st.markdown("<div class='overview'>", unsafe_allow_html=True)
    render_header(f"PGAM Revenue Dashboard ({start_str} to {end_str})")

    # KPI cards (summed over the range)
    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        ("SSP Dash Net",    data_ssp['SSP Dash Net'].sum(),    "$"),
        ("DSP Revenue",     data_pgam['DSP Revenue'].sum(),    "$"),
        ("SSP Impressions", data_ssp['SSP Impressions'].sum(), ""),
        ("DSP Impressions", data_pgam['DSP Impressions'].sum(),"")
    ]
    for c, (lbl, val, s) in zip((col1, col2, col3, col4), kpis):
        c.markdown(
            f"<div class='kpi-card'><h4>{lbl}</h4><p>{s}{val:,.0f}</p></div>",
            unsafe_allow_html=True
        )

    # Summary table (aggregated over the range)
    st.subheader("Summary")
    df_sum = data_pgam[[
        'SSP', 'DSP Revenue', 'DSP Impressions',
        'SSP Dash Net', 'SSP Impressions', 'Revenue Difference'
    ]].copy()

    try:
        # Show Xandr as aggregated over the date range, summing each day's data
        x_all = []
        for d in pd.date_range(start_str, end_str):
            x_df, x_total = get_xandr_all_cached(d.date().isoformat())
            x_df['Date'] = d.date().isoformat()
            x_all.append((x_df, x_total))
        x_concat = pd.concat([pair[0] for pair in x_all], ignore_index=True)
        x_imps = int(x_concat['DSP Impressions'].sum())
        x_total_sum = sum(pair[1] for pair in x_all)
        x_row = pd.DataFrame([{
            'SSP':                'Xandr',
            'DSP Revenue':        x_total_sum,
            'DSP Impressions':    x_imps,
            'SSP Dash Net':       x_total_sum,
            'SSP Impressions':    x_imps,
            'Revenue Difference': 0
        }])
        df_sum = pd.concat([df_sum, x_row], ignore_index=True)
    except Exception as e:
        st.warning(f"Could not fetch Xandr report: {e}")

    df_sum.columns = [
        'DSP Company', 'DSP Spend', 'Impressions',
        'SSP Revenue', 'SSP Impressions', 'Revenue Difference'
    ]
    df_sum['Date Range'] = f"{start_str} to {end_str}"

    st.dataframe(
        df_sum[['Date Range', 'DSP Company', 'DSP Spend', 'Impressions', 'SSP Revenue', 'Revenue Difference']],
        use_container_width=True
    )

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    # 1) Paired Bars (Spend / Impressions) aggregated over the range
    spend_df = data_pgam.melt(['SSP'], ['SSP Dash Net', 'DSP Revenue'], 'Type', 'Spend')
    imp_df   = data_pgam.melt(['SSP'], ['SSP Impressions', 'DSP Impressions'], 'Type', 'Impressions')
    bar1 = (
        alt.Chart(spend_df)
           .mark_bar()
           .encode(x='SSP:N', y='Spend:Q', color='Type:N')
           .properties(title='SSP vs DSP Spend (Aggregated)', width=350, height=300)
    )
    bar2 = (
        alt.Chart(imp_df)
           .mark_bar()
           .encode(x='SSP:N', y='Impressions:Q', color='Type:N')
           .properties(title='SSP vs DSP Impressions (Aggregated)', width=350, height=300)
    )
    st.altair_chart((bar1 | bar2).resolve_scale(y='independent'), use_container_width=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    # 2) Stacked Spend Composition (aggregated)
    comp_df = data_pgam.melt(['SSP'], ['SSP Dash Net', 'DSP Revenue'], 'Type', 'Amount')
    stack = (
        alt.Chart(comp_df)
           .mark_bar()
           .encode(x='SSP:N', y='Amount:Q', color='Type:N')
           .properties(title='Spend Composition (Aggregated)', width=700, height=300)
    )
    st.altair_chart(stack, use_container_width=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    # 3) Butterfly Chart: DSP Spend ←→ SSP Revenue (aggregated)
    st.subheader("Butterfly: DSP Spend ←→ SSP Revenue (Aggregated)")

    bf = data_pgam[[
        'SSP', 'DSP Revenue', 'SSP Dash Net'
    ]].rename(columns={
        'SSP':           'DSP Company',
        'DSP Revenue':   'DSP Spend',
        'SSP Dash Net':  'SSP Revenue'
    }).copy()
    bf['NegSpend'] = -bf['DSP Spend']

    bf_long = bf.melt(
        id_vars=['DSP Company'],
        value_vars=['NegSpend', 'SSP Revenue'],
        var_name='Type',
        value_name='Amount'
    )
    bf_long['Type'] = bf_long['Type'].map({
        'NegSpend':   'DSP Spend',
        'SSP Revenue': 'SSP Revenue'
    })

    butterfly = (
        alt.Chart(bf_long)
           .mark_bar()
           .encode(
               y=alt.Y('DSP Company:N', title=None, sort=bf['DSP Company'].tolist()),
               x=alt.X('Amount:Q', title='Amount', axis=alt.Axis(format='$', tickMinStep=1000)),
               color=alt.Color('Type:N', title='Measure', scale=alt.Scale(range=['#d62728', '#2ca02c'])),
               tooltip=['DSP Company', 'Type', 'Amount']
           )
           .properties(height=350, width=700)
    )
    st.altair_chart(butterfly, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────────
# 8) Details Page ─────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
def details_page():
    render_header(f"Detailed Domain / Bundle Metrics ({start_str} to {end_str})")
    df = data_ssp_details  # combined across the date range

    if df.empty:
        st.info("No detail data available")
        return

    sel = st.multiselect(
        "Choose SSP",
        df['DSP Company'].unique(),
        default=df['DSP Company'].unique()
    )
    df_filtered = df[df['DSP Company'].isin(sel)]
    st.dataframe(df_filtered, use_container_width=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    df_filtered['Total Revenue'] = df_filtered['PGAM Revenue'] + df_filtered['SSP Revenue']

    # Top 10 Domains by Total Revenue (across range)
    top10 = df_filtered.nlargest(10, 'Total Revenue')
    c1 = (
        alt.Chart(top10)
           .mark_bar()
           .encode(
               x='Total Revenue:Q',
               y=alt.Y('Domain:N', sort='-x'),
               color='DSP Company:N'
           )
           .properties(title='Top 10 Domains by Total Revenue (Aggregated)', height=300)
    )
    st.altair_chart(c1, use_container_width=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    # Scatter: eCPM (PGAM vs SSP) (aggregated)
    c2 = (
        alt.Chart(df_filtered)
           .mark_circle()
           .encode(
               x='PGAM eCPM:Q',
               y='SSP eCPM:Q',
               size='Total Revenue:Q',
               color='DSP Company:N'
           )
           .properties(title='eCPM: PGAM vs SSP (Aggregated)', height=300)
           .interactive()
    )
    st.altair_chart(c2, use_container_width=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    # Revenue Composition by Domain (aggregated)
    fold_cols = [c for c in ['PGAM Revenue', 'SSP Revenue'] if c in df_filtered.columns]
    c3 = (
        alt.Chart(df_filtered)
           .transform_fold(fold_cols, as_=['Metric', 'Value'])
           .mark_bar()
           .encode(
               x='Value:Q',
               y=alt.Y('Domain:N', sort='-x'),
               color='Metric:N'
           )
           .properties(title='Revenue Composition by Domain (Aggregated)', height=400)
    )
    st.altair_chart(c3, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────────
# 9) PGAM Data Page ───────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
def pgam_data_page():
    render_header(f"PGAM DSP Summary ({start_str} to {end_str})")

    dsp_df, _ = data_pgam_api, None
    dsp_df['Date Range'] = f"{start_str} to {end_str}"

    ## A) KPI Header: At-a-Glance Margin Health (aggregated)
    total_spend  = dsp_df['DSP Spend'].sum()
    total_profit = dsp_df['Profit'].sum()
    avg_margin   = dsp_df['Margin'].mean()
    best         = dsp_df.loc[dsp_df['Margin'].idxmax()]
    worst        = dsp_df.loc[dsp_df['Margin'].idxmin()]

    cols = st.columns(4)
    kpis = [
        ("Total DSP Spend",  f"${total_spend:,.0f}"),
        ("Total Profit",     f"${total_profit:,.0f}"),
        ("Average Margin",   f"{avg_margin*100:.1f}%"),
        (
            "Worst Margin DSP",
            f"{worst['DSP Company']} ({worst['Margin']*100:.1f}%)",
            f"Δ {(best['Margin'] - worst['Margin']) * 100:.1f}%"
        )
    ]
    for col, item in zip(cols, kpis):
        title = item[0]
        value = item[1]
        delta_html = f"<div class='kpi-delta'>{item[2]}</div>" if len(item) > 2 else ""
        card_html = (
            f"<div class='kpi-card'><h4>{title}</h4>"
            f"<div class='kpi-value'>{value}</div>{delta_html}</div>"
        )
        col.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # B) Summary Table
    st.subheader("Summary Table")
    cols_show = ['Date Range', 'DSP Company', 'DSP Spend', 'Impressions', 'SSP Revenue', 'Profit', 'Margin']
    st.dataframe(dsp_df[cols_show], use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # C) Opportunity Matrix (Quadrant)
    st.subheader("DSP Opportunity Matrix")
    matrix = (
        alt.Chart(dsp_df)
           .mark_point(filled=True)
           .encode(
               x=alt.X('DSP Spend:Q', scale=alt.Scale(zero=False), title='DSP Spend'),
               y=alt.Y('Margin:Q', scale=alt.Scale(zero=False), title='Margin'),
               size=alt.Size('Profit:Q', title='Profit'),
               color=alt.Color('Profit:Q', legend=None),
               tooltip=['DSP Company', 'DSP Spend', 'Profit', 'Margin']
           )
           .properties(height=350)
           .interactive()
    )
    st.altair_chart(matrix, use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # D) Margin Opportunity Score Table
    dsp_df['OppScore'] = (1 - dsp_df['Margin']) * dsp_df['DSP Spend']
    st.subheader("Margin Opportunity Score")
    score_df = dsp_df.sort_values('OppScore', ascending=False)[
        ['DSP Company', 'DSP Spend', 'Margin', 'OppScore']
    ].head(10)
    st.dataframe(score_df, use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # E) Grouped Bar: Profit & Margin by DSP
    st.subheader("Profit & Margin by DSP (Grouped)")
    grouped_df = dsp_df.melt(
        id_vars=['DSP Company'],
        value_vars=['Profit', 'Margin'],
        var_name='Metric',
        value_name='Value'
    )
    grouped_chart = (
        alt.Chart(grouped_df)
           .mark_bar()
           .encode(
               x=alt.X('DSP Company:N', title='DSP Company'),
               xOffset='Metric:N',
               y=alt.Y('Value:Q', title='Metric Value'),
               color='Metric:N',
               tooltip=['DSP Company', 'Metric', 'Value']
           )
           .properties(width=700, height=350)
    )
    st.altair_chart(grouped_chart, use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # F) Domain-Level Drill-Down (aggregated)
    st.subheader("Domain-Level Profit & Margin (Top 10, Aggregated)")
    details = data_ssp_details.copy()
    details['DomainProfit'] = details['PGAM Revenue'] - details['SSP Revenue']
    details['DomainMargin'] = (
        (details['PGAM Revenue'] - details['SSP Revenue'])
        / details['PGAM Revenue'].replace(0, 1)
    ).round(4)

    st.write("**Top 10 Domains by Profit**")
    top_profit = details.nlargest(10, 'DomainProfit')
    st.dataframe(
        top_profit[['Domain', 'DSP Company', 'DomainProfit', 'DomainMargin']],
        use_container_width=True
    )

    st.write("**Top 10 Domains by Margin %**")
    top_margin = details.nlargest(10, 'DomainMargin')
    st.dataframe(
        top_margin[['Domain', 'DSP Company', 'DomainProfit', 'DomainMargin']],
        use_container_width=True
    )


# ────────────────────────────────────────────────────────────────────────────────
# 10) Trends Page ─────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
def trends_page():
    render_header(f"7-Day Trends (From {start_str} to {end_str})")

    # Let the user adjust a more granular date range if desired
    t_start = st.sidebar.date_input("Trends From", start_date - timedelta(days=7))
    t_end   = st.sidebar.date_input("Trends To", yesterday)

    all_ = []
    for d in pd.date_range(t_start, t_end):
        sub, _ = load_pgam_data(d.date().isoformat())
        tmp = sub[['SSP', 'DSP Revenue']].groupby('SSP').sum().reset_index()
        tmp['Date'] = d.date().isoformat()
        all_.append(tmp)

    trend = pd.concat(all_, ignore_index=True)
    chart = (
        alt.Chart(trend)
           .mark_line(point=True)
           .encode(
               x='Date:T',
               y='DSP Revenue:Q',
               color='SSP:N'
           )
           .properties(height=350)
           .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────────
# 11) Chatbot “PGAMbot” Page ───────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────

def get_top_n(table: str, column: str, n: int, asc: bool = False, partner: str = None):
    """
    Returns the top n rows sorted by column (desc unless asc=True),
    optionally filtered by partner.
    """
    df_map = {
        "ssp_summary": load_ssp_summary_range(start_str, end_str),
        "ssp_details": data_ssp_details,
        "pgam_data":   data_pgam
    }
    df = df_map[table]
    if partner:
        df = df[df.get("SSP", df.get("DSP Company", "")) == partner]
    sorted_df = df.sort_values(column, ascending=asc)
    return sorted_df.head(n).to_dict(orient="records")


def get_aggregate(table: str, column: str, agg: str, partner: str = None) -> float:
    """
    Compute sum or mean on a column, optionally for one partner.
    """
    df_map = {
        "ssp_summary": load_ssp_summary_range(start_str, end_str),
        "ssp_details": data_ssp_details,
        "pgam_data":   data_pgam[ ["SSP", "DSP Revenue", "DSP Impressions", "SSP Dash Net", "SSP Impressions", "Revenue Difference", "Total Spend", "Profit", "Margin"] ]
    }
    df = df_map.get(table)
    if df is None:
        raise ValueError(f"Table '{table}' not recognized. Valid tables: {list(df_map)}")

    if column not in df.columns:
        available = df.columns.tolist()
        raise ValueError(f"Column '{column}' not found in '{table}'. Available columns: {available}")

    if partner:
        key = "DSP Company" if table == "ssp_details" else "SSP"
        if key not in df.columns:
            raise ValueError(f"Cannot filter by partner on table '{table}'")
        df = df[df[key] == partner]

    series = df[column]
    if agg == "sum":
        return float(series.sum())
    elif agg == "mean":
        return float(series.mean())
    else:
        raise ValueError("`agg` must be 'sum' or 'mean'")


functions = [
    {
        "name": "get_aggregate",
        "description": "Compute sum or mean on a column, optionally for one partner",
        "parameters": {
            "type": "object",
            "properties": {
                "table":   {"type": "string", "enum": ["ssp_summary", "ssp_details", "pgam_data"]},
                "column":  {"type": "string"},
                "agg":     {"type": "string", "enum": ["sum", "mean"]},
                "partner": {"type": "string"}
            },
            "required": ["table", "column", "agg"]
        }
    },
    {
        "name": "get_top_n",
        "description": "Return top N rows sorted by a column, optionally for one partner",
        "parameters": {
            "type": "object",
            "properties": {
                "table":   {"type": "string", "enum": ["ssp_summary", "ssp_details", "pgam_data"]},
                "column":  {"type": "string"},
                "n":       {"type": "integer"},
                "asc":     {"type": "boolean"},
                "partner": {"type": "string"}
            },
            "required": ["table", "column", "n"]
        }
    }
]


def chatbot_page():
    render_header(f"Data Assistant 🤖 ({start_str} to {end_str})")

    summary_df = data_pgam.rename(columns={'SSP': 'DSP Company', 'DSP Revenue': 'DSP Spend'})
    details_df = data_ssp_details.copy()

    if "summary_agent" not in st.session_state:
        st.session_state.summary_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=summary_df,
            functions=functions,
            verbose=False,
            return_intermediate_steps=False,
            allow_dangerous_code=True,
            max_iterations=10,
        )

    if "details_agent" not in st.session_state:
        st.session_state.details_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=details_df,
            functions=functions,
            verbose=False,
            return_intermediate_steps=False,
            allow_dangerous_code=True,
            max_iterations=10,
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask me anything about data in the selected range…")
    if not prompt:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "domain" in prompt.lower() or "top 10 domains" in prompt.lower():
        agent = st.session_state.details_agent
    else:
        agent = st.session_state.summary_agent

    with st.spinner("Let me get that…"):
        try:
            answer = agent.run(prompt)
        except Exception as e:
            answer = f"⚠️ Error: {e}"

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.markdown("</div>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────────
# 12) Routing ─────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    overview_page()
elif page == "Details":
    details_page()
elif page == "PGAM Data":
    pgam_data_page()
elif page == "Trends":
    trends_page()
else:
    chatbot_page()

st.sidebar.markdown("---")
st.sidebar.write("Built by PGAM Data Team")
