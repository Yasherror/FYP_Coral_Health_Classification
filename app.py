"""
app.py
Entry point / orchestrator for the Coral Health Classification dashboard.
Each tab lives in its own module — this file just wires them together:
page config, shared session state, global CSS, header/footer, and the
sidebar navigation that dispatches to the right tab.

Expected folder layout (all files side by side):
    app.py
    data_utils.py - shared helpers, imported by every tab module
    Overview.py
    Prediction.py  - upload + EfficientNet-B3 inference + results
    HotspotMapping.py
    ExplainableAI.py
    Chatbot.py
    Report.py      - CSV + PDF export (deliverables)
"""

import os
import sys

# Make sure Python can find the sibling modules
# even though the files sit right next to app.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from datetime import datetime

from Overview import render_overview
from Prediction import render_prediction
from HotspotMapping import render_hotspots
from ExplainableAI import render_xai
from Chatbot import render_chatbot
from Report import render_report


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Coral Health Classification",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE INITIALIZATION (DO THIS RIGHT AFTER PAGE CONFIG)
# ============================================================
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "overview"

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

if "selected_image_data" not in st.session_state:
    st.session_state.selected_image_data = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "active_analysis" not in st.session_state:
    st.session_state.active_analysis = None

if "error_text" not in st.session_state:
    st.session_state.error_text = None

if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "anthropic_api_key" not in st.session_state:
    st.session_state.anthropic_api_key = ""

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #073742 0%, #04242c 100%);
    }
    
    /* Sidebar - Darker teal */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a4a5a 0%, #063542 100%);
        border-right: 1px solid rgba(149, 198, 206, 0.2);
    }
    
    /* Sidebar buttons - White text */
    [data-testid="stSidebar"] button {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] button:hover {
        background: rgba(180, 226, 233, 0.3) !important;
    }
    
    /* Main button hover effect */
    .stButton > button:hover {
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    /* Status header */
    .status-header {
        background: rgba(7, 55, 66, 0.4);
        padding: 14px 20px;
        border-radius: 18px;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
def render_header():
    now = datetime(2026, 5, 20, 11, 21, 31)
    st.markdown(f"""
    <div class="status-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <div style="font-size: 11px; color: #95c6ce; letter-spacing: 2px;">🪸 Coral Health Classification Dashboard</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
def render_footer():
    st.markdown("---")
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; font-size: 11px; color: #95c6ce;">
        <span>🪸 Koh Tao Reef Watch Consortium © 2026</span>
        <div style="display: flex; gap: 1rem;">
            <span>🤖 Model: EfficientNet-B3 (tuned)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN APP
# ============================================================
def main():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #0EA5E9; margin-bottom: 1.5rem;">
            <div style="font-size: 48px;">🪸</div>
            <div style="font-size: 20px; font-weight: 700; color: #b4e2e9;">Coral Health</div>
            <div style="font-size: 10px; letter-spacing: 2px; color: #95c6ce;">CLASSIFICATION DASHBOARD</div>
        </div>
        """, unsafe_allow_html=True)

        tabs = {
            "overview": "Overview",
            "hotspots": "Hotspot Mapping",
            "prediction": "Prediction",
            "heatmap": "Prediction Explainability",
            "chatbot": "Reef Assistant",
            "report": "Export Report",
        }

        for tab_id, tab_label in tabs.items():
            is_active = st.session_state.active_tab == tab_id
            if st.button(tab_label, key=f"nav_{tab_id}", use_container_width=True,
                        type="primary" if is_active else "secondary"):
                st.session_state.active_tab = tab_id
                st.rerun()

        st.markdown("---")
        st.caption("📍 Koh Tao Marine Sanctuary")
        st.caption("🌊 Gulf of Thailand")

    render_header()

    if st.session_state.active_tab == "overview":
        render_overview()
    elif st.session_state.active_tab == "prediction":
        render_prediction()
    elif st.session_state.active_tab == "hotspots":
        render_hotspots()
    elif st.session_state.active_tab == "heatmap":
        render_xai()
    elif st.session_state.active_tab == "chatbot":
        render_chatbot()
    elif st.session_state.active_tab == "report":
        render_report()

    render_footer()

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()