import os
import importlib
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import UnidentifiedImageError
from dotenv import load_dotenv
from streamlit_chat import message
from ssp_xandr import fetch_xandr_all
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
from openai import OpenAI  
from calendar import monthrange
import json
load_dotenv()               
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
from pgam_data import PGAMReportAPI
def partner_peer_benchmark(partner: str, summary: pd.DataFrame):
    """
    Show partner’s eCPM vs. the peer median for the same date.
    """
    # 1) Partner eCPM
    row      = summary.query("SSP == @partner").iloc[0]
    ssp_rev  = row['SSP Dash Net']
    ssp_imp  = row['SSP Impressions']
    ssp_ecpm = (ssp_rev / max(ssp_imp,1) * 1000)

    # 2) Peer median eCPM
    df2 = summary.assign(
        _ecpm = summary['SSP Dash Net'] / summary['SSP Impressions'].replace(0,1) * 1000
    )
    med_ecpm = df2['_ecpm'].median()

    # 3) Show as a single metric with delta
    st.metric(
        label="eCPM vs Peer Median",
        value=f"${ssp_ecpm:,.2f}",
        delta=f"{ssp_ecpm - med_ecpm:+.2f}"
    )


def domain_gap_chart(partner: str, details: pd.DataFrame):
    """
    Show bottom-5 domains by eCPM gap vs. this partner’s average eCPM.
    """
    det = details.query("`DSP Company` == @partner")
    if det.empty:
        st.info("No domain data for this partner today.")
        return

    det = det.copy()
    det['ecpm'] = det['SSP Revenue'] / det['SSP Impressions'].replace(0,1) * 1000
    avg_ecpm    = det['ecpm'].mean()
    det['gap']  = det['ecpm'] - avg_ecpm

    bottom5 = det.nsmallest(5, 'gap')

    chart = (
        alt.Chart(bottom5)
           .mark_bar()
           .encode(
             y=alt.Y("Domain:N", sort="-x", title=None),
             x=alt.X("gap:Q", title="eCPM vs Partner Avg"),
             color=alt.condition(
                 "datum.gap < 0",
                 alt.value("#d62728"),
                 alt.value("#2ca02c")
             ),
             tooltip=["Domain","ecpm","gap"]
           )
           .properties(
             title="Bottom 5 Domains vs Partner Avg eCPM",
             height=300
           )
    )
    st.altair_chart(chart, use_container_width=True)


LOGO_PATH = "logo.png"

# ── 1) Page config ───────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="PGAM Revenue Dashboard", page_icon="📊")

# ── 2) CSS loader ─────────────────────────────────────────────────────────────────
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")

# ── 3) Sidebar ───────────────────────────────────────────────────────────────────
yesterday     = (datetime.now() - timedelta(days=1)).date()
selected_date = st.sidebar.date_input("Select Date", yesterday)
date_str      = selected_date.isoformat()
page          = st.sidebar.radio("Go to", ["Overview", "Details", "PGAM Data", "Trends", "PGAMbot"])

# ── Data loaders ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_ssp_summary(date_str):
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
    def fetch(ssp, mod, fn):
        m = importlib.import_module(mod)
        df, total = getattr(m, fn)(date=date_str)
        if 'impressions' in df.columns and 'SSP_Impressions' not in df.columns:
            df = df.rename(columns={'impressions':'SSP_Impressions'})
        return ssp, float(total), int(df.get('SSP_Impressions',0).sum())

    with ThreadPoolExecutor(max_workers=5) as exe:
        futures = {exe.submit(fetch, ssp, mod, fn): ssp for ssp,(mod,fn) in modules.items()}
        for fut in as_completed(futures):
            ssp = futures[fut]
            try:
                _, rev, imps = fut.result()
                rows.append({'SSP':ssp,'SSP Dash Net':rev,'SSP Impressions':imps})
            except Exception as e:
                st.error(f"Error loading {ssp} summary: {e}")
    return pd.DataFrame(rows)
    mask = df['SSP'] == 'Freewheel'
    df.loc[mask, 'SSP Dash Net'] = (df.loc[mask, 'SSP Dash Net'] * 0.75).round(2)

@st.cache_data
def load_pgam_data(date_str):
    api = PGAMReportAPI()
    dsp_df, totals = api.fetch_summary(date=date_str)
    dsp_df = dsp_df.rename(columns={
        'DSP Company':'SSP',
        'DSP Spend':'DSP Revenue',
        'Impressions':'DSP Impressions'
    })
    
    dsp_df['SSP'] = dsp_df['SSP'].astype(str).str.strip()


    s = load_ssp_summary(date_str)[['SSP','SSP Dash Net','SSP Impressions']]
    merged = dsp_df.merge(s, on='SSP', how='left').fillna({
        'SSP Dash Net':0,
        'SSP Impressions':0
    })

    # override for Stirista & OTTA Unruly
    mask = merged['SSP'].isin(['Stirista','OTTA Unruly','Xandr'])
    merged.loc[mask,'SSP Dash Net']    = merged.loc[mask,'DSP Revenue']
    merged.loc[mask,'SSP Impressions'] = merged.loc[mask,'DSP Impressions']

    # ── override Freewheel DSP to match SSP ─────────────────────────────────
    fw = merged['SSP'] == 'Freewheel'
    merged.loc[fw, 'DSP Revenue']     = merged.loc[fw, 'SSP Dash Net']
    merged.loc[fw, 'DSP Impressions'] = merged.loc[fw, 'SSP Impressions']

    merged['Revenue Difference'] = merged['SSP Dash Net'] - merged['DSP Revenue']
    merged['Total Spend']        = merged['DSP Revenue'] + merged['SSP Dash Net']
    merged['Profit']             = merged['DSP Revenue'] - merged['SSP Dash Net']
    merged['Margin']             = (merged['Profit']/merged['DSP Revenue'].replace(0,1)).round(4)
    return merged, totals

@st.cache_data
def load_pgam_api(date_str):
    api = PGAMReportAPI()
    dsp_df, totals = api.fetch_summary(date=date_str)
    return dsp_df, totals

@st.cache_data
def load_ssp_details(date_str):
    ssps = load_ssp_summary(date_str)['SSP'].tolist()
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
        'Xandr':        ('ssp_xandr',  'fetch_xandr_all')
    }
    parts = []
    for ssp in ssps:
        if ssp not in mapping:
            continue
        mod, fn = mapping[ssp]
        try:
            m = importlib.import_module(mod)
            df_ssp, _ = getattr(m, fn)(date=date_str)
            df_ssp = df_ssp.rename(columns={
                'Domain / Bundle':'Domain',
                'SSP_Dash':'SSP Revenue',
                'SSP_eCPM':'SSP eCPM'
            })
            for c in ['Date','Domain','SSP Revenue','SSP eCPM']:
                if c not in df_ssp.columns:
                    df_ssp[c] = 0
            df_ssp['DSP Company'] = ssp
            parts.append(df_ssp[['Date','DSP Company','Domain','SSP Revenue','SSP eCPM']])
        except Exception:
            continue

    pg = PGAMReportAPI().fetch_dsp_details(date=date_str)
    pg = pg.rename(columns={
        'Domain / Bundle':'Domain',
        'DSP Spend':'PGAM Revenue',
        'DSP eCPM':'PGAM eCPM'
    })
    for c in ['Date','Domain','PGAM Revenue','PGAM eCPM']:
        if c not in pg.columns:
            pg[c] = 0
    parts.append(pg[['Date','DSP Company','Domain','PGAM Revenue','PGAM eCPM']])

    raw = pd.concat(parts, ignore_index=True)
    desired = ['SSP Revenue','SSP eCPM','PGAM Revenue','PGAM eCPM']
    values = [c for c in desired if c in raw.columns]
    combined = raw.pivot_table(
        index=['Date','DSP Company','Domain'],
        values=values,
        aggfunc='first'
    ).reset_index().fillna(0)
    return combined

@st.cache_data
def load_partner_gaps(date_str: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      ['DSP Company','Domain','SSP eCPM','gap']
    where gap = domain eCPM - partner_avg eCPM.
    """
    # reuse your existing loader
    df = load_ssp_details(date_str)
    # calculate per-partner avg and gap
    def compute_gaps(sub):
        sub = sub.copy()
        avg = sub['SSP eCPM'].mean()
        sub['gap'] = sub['SSP eCPM'] - avg
        return sub[['DSP Company','Domain','SSP eCPM','gap']]
    # groupby partner
    return pd.concat([compute_gaps(g) for _, g in df.groupby('DSP Company')], ignore_index=True)


# ── header helper ────────────────────────────────────────────────────────────────
def render_header(title):
    c1, c2 = st.columns([1,9])
    if os.path.exists(LOGO_PATH):
        try:
            c1.image(LOGO_PATH, width=60)
        except UnidentifiedImageError:
            pass
    c2.markdown(f"<h1 style='color:var(--pgam-primary)'>{title}</h1>", unsafe_allow_html=True)

# ── load global data ──────────────────────────────────────────────────────────────
data_ssp, _      = load_ssp_summary(date_str), None
data_pgam, _     = load_pgam_data(date_str)
data_pgam_api, _ = load_pgam_api(date_str)

# ── Overview Page ────────────────────────────────────────────────────────────────
def overview_page():
    st.markdown("<div class='overview'>", unsafe_allow_html=True)

    render_header("PGAM Revenue Dashboard")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        ("SSP Dash Net",    data_ssp['SSP Dash Net'].sum(),    "$"),
        ("DSP Revenue",     data_pgam['DSP Revenue'].sum(),    "$"),
        ("SSP Impressions", data_ssp['SSP Impressions'].sum(), ""),
        ("DSP Impressions", data_pgam['DSP Impressions'].sum(),"")
    ]
    for c,(lbl,val,s) in zip((col1,col2,col3,col4), kpis):
        c.markdown(f"<div class='kpi-card'><h4>{lbl}</h4><p>{s}{val:,.0f}</p></div>",
                   unsafe_allow_html=True)

    # Summary table
    st.subheader("Summary")
    df_sum = data_pgam[[
        'SSP','DSP Revenue','DSP Impressions',
        'SSP Dash Net','SSP Impressions','Revenue Difference'
    ]].copy()
    try:
        x_df, x_total = fetch_xandr_all(date=date_str)
        x_imps = int(x_df['DSP Impressions'].sum())
        x_row = pd.DataFrame([{
        'SSP':                'Xandr',
        'DSP Revenue':        x_total,
        'DSP Impressions':    x_imps,
        'SSP Dash Net':       x_total,    # DSP spend == SSP net for Xandr
        'SSP Impressions':    x_imps,
        'Revenue Difference': 0
        }])
        df_sum = pd.concat([df_sum, x_row], ignore_index=True)
    except Exception as e:
        st.warning(f"Could not fetch Xandr report: {e}")
    df_sum.columns = [
        'DSP Company','DSP Spend','Impressions',
        'SSP Revenue','SSP Impressions','Revenue Difference'
    ]
    df_sum['Date'] = date_str
    st.dataframe(df_sum[['Date','DSP Company','DSP Spend','Impressions','SSP Revenue','Revenue Difference']],
                 use_container_width=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    # 1) Paired Bars (Spend / Impressions)
    spend_df = data_pgam.melt(['SSP'], ['SSP Dash Net','DSP Revenue'], 'Type','Spend')
    imp_df   = data_pgam.melt(['SSP'], ['SSP Impressions','DSP Impressions'], 'Type','Impressions')
    bar1 = (alt.Chart(spend_df).mark_bar()
            .encode(x='SSP:N', y='Spend:Q', color='Type:N')
            .properties(title='SSP vs DSP Spend', width=350,height=300))
    bar2 = (alt.Chart(imp_df).mark_bar()
            .encode(x='SSP:N', y='Impressions:Q', color='Type:N')
            .properties(title='SSP vs DSP Impressions', width=350,height=300))
    st.altair_chart((bar1|bar2).resolve_scale(y='independent'), use_container_width=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    # 2) Stacked Spend Composition
    comp_df = data_pgam.melt(['SSP'], ['SSP Dash Net','DSP Revenue'], 'Type','Amount')
    stack = (alt.Chart(comp_df).mark_bar()
             .encode(x='SSP:N', y='Amount:Q', color='Type:N')
             .properties(title='Spend Composition', width=700, height=300))
    st.altair_chart(stack, use_container_width=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    # 3) Butterfly Chart: DSP Spend vs SSP Revenue
    st.subheader("Butterfly: DSP Spend ←→ SSP Revenue")

    # prepare a small DataFrame
    bf = data_pgam[['SSP','DSP Revenue','SSP Dash Net']].rename(
    columns={'SSP':'DSP Company','DSP Revenue':'DSP Spend','SSP Dash Net':'SSP Revenue'}
    ).copy()
    # flip DSP Spend negative
    bf['NegSpend'] = -bf['DSP Spend']

    # melt into long form
    bf_long = bf.melt(
      id_vars=['DSP Company'],
    value_vars=['NegSpend','SSP Revenue'],
    var_name='Type',
    value_name='Amount'
    )

    # map nicer labels
    bf_long['Type'] = bf_long['Type'].map({'NegSpend':'DSP Spend','SSP Revenue':'SSP Revenue'})

    butterfly = (
      alt.Chart(bf_long)
       .mark_bar()
       .encode(
           y=alt.Y('DSP Company:N', title=None, sort=bf['DSP Company'].tolist()),
           x=alt.X('Amount:Q', title='Amount', 
                   axis=alt.Axis(format='$', tickMinStep=1000)),
           color=alt.Color('Type:N', title='Measure',
                           scale=alt.Scale(range=['#d62728','#2ca02c'])),
           tooltip=['DSP Company','Type','Amount']
       )
       .properties(height=350, width=700)
     )

    st.altair_chart(butterfly, use_container_width=True)

# ── Details Page ─────────────────────────────────────────────────────────────────
def details_page():
    render_header("Detailed Domain / Bundle Metrics")
    df = load_ssp_details(date_str)
    if df.empty:
        st.info("No detail data available")
        return

    sel = st.multiselect("Choose SSP", df['DSP Company'].unique(),
                         default=df['DSP Company'].unique())
    df = df[df['DSP Company'].isin(sel)]
    st.dataframe(df, use_container_width=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    df['Total Revenue'] = df.get('PGAM Revenue',0) + df.get('SSP Revenue',0)
    top10 = df.nlargest(10,'Total Revenue')
    c1 = (alt.Chart(top10).mark_bar()
          .encode(x='Total Revenue:Q', y=alt.Y('Domain:N',sort='-x'),
                  color='DSP Company:N')
          .properties(title='Top 10 Domains by Total Revenue', height=300))
    st.altair_chart(c1, use_container_width=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    c2 = (alt.Chart(df).mark_circle()
          .encode(x='PGAM eCPM:Q', y='SSP eCPM:Q', size='Total Revenue:Q',
                  color='DSP Company:N')
          .properties(title='eCPM: PGAM vs SSP', height=300).interactive())
    st.altair_chart(c2, use_container_width=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    fold_cols = [c for c in ['PGAM Revenue','SSP Revenue'] if c in df.columns]
    c3 = (alt.Chart(df).transform_fold(fold_cols, as_=['Metric','Value'])
          .mark_bar()
          .encode(x='Value:Q', y=alt.Y('Domain:N',sort='-x'),
                  color='Metric:N')
          .properties(title='Revenue Composition by Domain', height=400))
    st.altair_chart(c3, use_container_width=True)

# ── PGAM Data Page ────────────────────────────────────────────────────────────────
def pgam_data_page():
    render_header("PGAM DSP Summary")

    # Load summary from PGAM API (with Profit & Margin)
    dsp_df, _ = data_pgam_api, None
    dsp_df['Date'] = date_str

    ## A) KPI Header: At-a-Glance Margin Health with Styled Cards
    total_spend  = dsp_df['DSP Spend'].sum()
    total_profit = dsp_df['Profit'].sum()
    avg_margin   = dsp_df['Margin'].mean()
    best         = dsp_df.loc[dsp_df['Margin'].idxmax()]
    worst        = dsp_df.loc[dsp_df['Margin'].idxmin()]
    # Use CSS class .kpi-card defined in style.css
    cols = st.columns(4)
    kpis = [
        ("Total DSP Spend", f"${total_spend:,.0f}"),
        ("Total Profit", f"${total_profit:,.0f}"),
        ("Average Margin", f"{avg_margin*100:.1f}%"),
        ("Worst Margin DSP", f"{worst['DSP Company']} ({worst['Margin']*100:.1f}%)", f"Δ {(best['Margin']-worst['Margin'])*100:.1f}%")
    ]
    for col, item in zip(cols, kpis):
        # build card HTML
        title = item[0]
        value = item[1]
        delta_html = f"<div class='kpi-delta'>{item[2]}</div>" if len(item) > 2 else ""
        card_html = f"<div class='kpi-card'><h4>{title}</h4><div class='kpi-value'>{value}</div>{delta_html}</div>"
        col.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # B) Summary Table
    st.subheader("Summary Table")
    cols = ['Date', 'DSP Company', 'DSP Spend', 'Impressions', 'SSP Revenue', 'Profit', 'Margin']
    st.dataframe(dsp_df[cols], use_container_width=True)

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
               tooltip=['DSP Company','DSP Spend','Profit','Margin']
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
        ['DSP Company','DSP Spend','Margin','OppScore']
    ].head(10)
    st.dataframe(score_df, use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # E) Grouped Bar: Profit & Margin by DSP
    st.subheader("Profit & Margin by DSP (Grouped)")
    grouped_df = dsp_df.melt(
        id_vars=['DSP Company'],
        value_vars=['Profit','Margin'],
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
               tooltip=['DSP Company','Metric','Value']
           )
           .properties(width=700, height=350)
    )
    st.altair_chart(grouped_chart, use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)


    # F) Domain-Level Drill-Down
    st.subheader("Domain-Level Profit & Margin (Top 10)")
    details = load_ssp_details(date_str)
    details = details.assign(
        DomainProfit = details['PGAM Revenue'] - details['SSP Revenue'],
        DomainMargin = (
            (details['PGAM Revenue'] - details['SSP Revenue']) /
            details['PGAM Revenue'].replace(0,1)
        ).round(4)
    )

    st.write("**Top 10 Domains by Profit**")
    top_profit = details.nlargest(10, 'DomainProfit')
    st.dataframe(
        top_profit[['Domain','DSP Company','DomainProfit','DomainMargin']],
        use_container_width=True
    )

    st.write("**Top 10 Domains by Margin %**")
    top_margin = details.nlargest(10, 'DomainMargin')
    st.dataframe(
        top_margin[['Domain','DSP Company','DomainProfit','DomainMargin']],
        use_container_width=True
    )



# ── Trends Page ─────────────────────────────────────────────────────────────────
def trends_page():
    render_header("7-Day Trends")
    start = st.sidebar.date_input("From", yesterday-timedelta(days=7))
    end   = st.sidebar.date_input("To", yesterday)
    all_  = []
    for d in pd.date_range(start, end):
        sub, _ = load_pgam_data(d.date().isoformat())
        tmp = sub[['SSP','DSP Revenue']].groupby('SSP').sum().reset_index()
        tmp['Date'] = d.date().isoformat()
        all_.append(tmp)
    trend = pd.concat(all_, ignore_index=True)
    chart = (alt.Chart(trend).mark_line(point=True)
             .encode(x='Date:T', y='DSP Revenue:Q', color='SSP:N')
             .properties(height=350).interactive())
    st.altair_chart(chart, use_container_width=True)


# ─── Define the “tool” functions ─────────────────────────────────────────────────
def get_top_n(table: str, column: str, n: int, asc: bool = False, partner: str = None):
    """
    Returns the top n rows sorted by column (desc unless asc=True),
    optionally filtered by partner.
    """
    df_map = {
        "ssp_summary": load_ssp_summary(date_str),
        "ssp_details": load_ssp_details(date_str),
        "pgam_data":   load_pgam_data(date_str)[0]
    }
    df = df_map[table]
    if partner:
        df = df[df.get("SSP", df.get("DSP Company","")) == partner]
    sorted_df = df.sort_values(column, ascending=asc)
    return sorted_df.head(n).to_dict(orient="records")

# ─── Register functions for OpenAI ────────────────────────────────────────────────
# ─── get_aggregate with column validation ────────────────────────────────────────
# ─── get_aggregate with correct partner key ────────────────────────────────────
def get_aggregate(table: str, column: str, agg: str, partner: str = None) -> float:
    df_map = {
        "ssp_summary": load_ssp_summary(date_str),
        "ssp_details": load_ssp_details(date_str),
        "pgam_data":   load_pgam_data(date_str)[0]
    }
    df = df_map.get(table)
    if df is None:
        raise ValueError(f"Table '{table}' not recognized. Valid tables: {list(df_map)}")

    # validate column
    if column not in df.columns:
        available = df.columns.tolist()
        raise ValueError(f"Column '{column}' not found in '{table}'. Available columns: {available}")

    # choose the correct partner‐column for filtering
    if partner:
        if table == "ssp_details":
            key = "DSP Company"
        else:
            key = "SSP"
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
                "table":   {"type":"string","enum":["ssp_summary","ssp_details","pgam_data"]},
                "column":  {"type":"string"},
                "agg":     {"type":"string","enum":["sum","mean"]},
                "partner": {"type":"string"}
            },
            "required": ["table","column","agg"]
        }
    },
    {
        "name": "get_top_n",
        "description": "Return top N rows sorted by a column, optionally for one partner",
        "parameters": {
            "type": "object",
            "properties": {
                "table":   {"type":"string","enum":["ssp_summary","ssp_details","pgam_data"]},
                "column":  {"type":"string"},
                "n":       {"type":"integer"},
                "asc":     {"type":"boolean"},
                "partner": {"type":"string"}
            },
            "required": ["table","column","n"]
        }
    }
]

# ── Chatbot Page ─────────────────────────────────────────────────────────────────
def chatbot_page():
    render_header("Data Assistant 🤖")

    # 1) Prepare your tables & tools
    summary_df, _ = load_pgam_data(date_str)   # merged_df w/ Profit, Margin
    summary_df = summary_df.rename(columns={'SSP':'DSP Company','DSP Revenue':'DSP Spend'})
    details_df = load_ssp_details(date_str)

    # 2) Bootstrap agents (only once)
    if "summary_agent" not in st.session_state:
        st.session_state.summary_agent = create_pandas_dataframe_agent(
            llm=llm, df=summary_df,
            # make sure get_aggregate & get_top_n are registered
            functions=functions,
            verbose=False, return_intermediate_steps=False,
            allow_dangerous_code=True,
            max_iterations=10,
        )
    if "details_agent" not in st.session_state:
        st.session_state.details_agent = create_pandas_dataframe_agent(
            llm=llm, df=details_df,
            functions=functions,
            verbose=False, return_intermediate_steps=False,
            allow_dangerous_code=True,
            max_iterations=10,
        )

    # 3) Render chat history & get input
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask me anything about today's data…")
    if not prompt:
        return
    st.session_state.chat_history.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4) Route to the right agent
    if "domain" in prompt.lower() or "top 10 domains" in prompt.lower():
        agent = st.session_state.details_agent
    else:
        agent = st.session_state.summary_agent

    with st.spinner("Let me get that…"):
        try:
            answer = agent.run(prompt)
        except Exception as e:
            answer = f"⚠️ Error: {e}"

    # 5) Display
    st.session_state.chat_history.append({"role":"assistant","content":answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.markdown("</div>", unsafe_allow_html=True)
# ── routing ───────────────────────────────────────────────────────────────────────
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