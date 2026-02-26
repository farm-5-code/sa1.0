import streamlit as st

st.set_page_config(page_title="Sport Analyzer", page_icon="🏆", layout="wide")

from sport_analyzer.dashboard.app import main  # noqa: E402

main()
