import streamlit as st
from pgam_data import PGAMReportAPI

@st.cache_resource
def get_pgam_api():
    return PGAMReportAPI()

@st.cache_data(ttl=3600)
def get_summary(date):
    api = get_pgam_api()
    return api.fetch_summary(date)

@st.cache_data(ttl=3600)
def get_details(date):
    api = get_pgam_api()
    return api.fetch_dsp_details(date)
