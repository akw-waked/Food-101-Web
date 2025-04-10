import streamlit as st
st.set_page_config(layout="wide")

from mainPages import dataset_info, model_training, model_comparison, model_testing
from utils.download_models import download_all_models
download_all_models()

# Optional: CSS hide menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# App layout
col1, col2 = st.columns([2, 6])

with col1:
    st.markdown("### Main Menu")
    page = st.radio(
        "Go to",
        ["Dataset Information", "Model Training", "Model Comparison", "Test Model"],
        label_visibility="collapsed"
    )

with col2:
    if page == "Dataset Information":
        dataset_info.show()
    elif page == "Model Training":
        model_training.show()
    elif page == "Model Comparison":
        model_comparison.show()
    elif page == "Test Model":
        model_testing.show()
