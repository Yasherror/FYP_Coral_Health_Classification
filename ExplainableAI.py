"""
ExplainableAI.py
The "Heat Map Overlay" tab (render_xai). It explains WHERE the
EfficientNet-B3 model looked when it made its prediction, using Grad-CAM,
and blends that heat map over the uploaded coral image so a non-technical
viewer can literally see which parts of the reef drove the decision.
"""

import streamlit as st

from data_utils import (
    compute_gradcam, bytes_to_pil_image,
    ALL_LABELS, HEALTH_COLORS,
)


def _init_state():
    if "selected_image_bytes" not in st.session_state:
        st.session_state.selected_image_bytes = None
    if "gradcam_result" not in st.session_state:
        st.session_state.gradcam_result = None
    if "gradcam_autorun" not in st.session_state:
        st.session_state.gradcam_autorun = False


def _compute(target_label):
    """Run Grad-CAM on the currently uploaded image and cache the result."""
    pil_image = bytes_to_pil_image(st.session_state.selected_image_bytes)
    with st.spinner("🔥 Computing heat map..."):
        result = compute_gradcam(pil_image, target_label=target_label)
    # Remember which class this heat map explains.
    result["target_label"] = target_label if target_label is not None else result["class_label"]
    st.session_state.gradcam_result = result


def _render_no_image():
    st.markdown("### Heat Map Overlay (Grad-CAM)")
    st.info(
        "No image loaded yet. Head to the **Prediction** tab, upload a coral "
        "image and run the prediction."
    )
    if st.button("➡️ Go to Prediction tab", type="primary"):
        st.session_state.active_tab = "prediction"
        st.rerun()


def render_xai():
    """Render the Grad-CAM heat map tab."""
    _init_state()

    if not st.session_state.selected_image_bytes:
        _render_no_image()
        return

    st.markdown("### Heat Map Overlay")
    st.caption(
        "This is a **Grad-CAM** heat map. It highlights the regions of the coral "
        "image that most influenced the AI's decision. **Red / yellow = strong "
        "influence**, blue = little influence. It's a way to check the AI is looking "
        "at the coral itself and not, say, the background water."
    )

    # --- Which class to explain ------------------------------------------ #
    analysis = st.session_state.get("active_analysis")
    default_label = analysis["primary_health"] if analysis else ALL_LABELS[0]
    default_idx = ALL_LABELS.index(default_label) if default_label in ALL_LABELS else 0

    col_pick, col_btn = st.columns([2, 1])
    with col_pick:
        target_label = st.selectbox(
            "Explain the AI's decision for which condition?",
            ALL_LABELS,
            index=default_idx,
            help="By default this shows the winning prediction. You can also see which "
                 "regions push the image toward the other conditions.",
        )
    with col_btn:
        st.write("")
        st.write("")
        run_clicked = st.button(" Generate Heat Map", type="primary", use_container_width=True)

    # --- Decide whether to (re)compute ----------------------------------- #
    autorun = st.session_state.get("gradcam_autorun", False)
    cached = st.session_state.get("gradcam_result")
    need_run = run_clicked or autorun or (cached is None)
    # If the cached heat map is for a different class than requested, recompute.
    if cached is not None and cached.get("target_label") != target_label and not autorun:
        need_run = run_clicked  # only recompute on explicit click when just switching class

    if autorun:
        # The Prediction-tab button asked us to auto-run: use the winning class.
        target_label = default_label
        st.session_state.gradcam_autorun = False  # consume the flag so it runs once

    if need_run:
        try:
            _compute(target_label if not autorun else default_label)
        except Exception as e:
            st.error(f"❌ Could not compute the heat map: {e}")
            return

    result = st.session_state.get("gradcam_result")
    if result is None:
        st.info("Press **🔥 Generate Heat Map** to compute the overlay.")
        return

    # --- Plain-language headline ----------------------------------------- #
    explained = result.get("target_label", result["class_label"])
    conf = result.get("class_conf", 0.0) * 100
    color = HEALTH_COLORS.get(explained, "#0EA5E9")
    st.markdown(
        f"<div style='background:{color}1f;border-left:6px solid {color};border-radius:12px;"
        f"padding:0.8rem 1.1rem;margin:0.5rem 0 1rem 0;font-size:15px;color:#eafcff;'>"
        f"🔎 Heat map for <b>{explained}</b> — the coloured areas below are the parts of the "
        f"image that pushed the AI toward this reading"
        f"</div>",
        unsafe_allow_html=True,
    )

    # --- The three views: original / heat map / overlay ------------------ #
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(result["original"], use_container_width=True, caption="① Original image")
    with c2:
        st.image(result["heatmap"], use_container_width=True, caption="② Heat map (raw)")
    with c3:
        st.image(result["overlay"], use_container_width=True,
                 caption="③ Overlay — original + heat map")

    st.markdown(
        "**How to read this:** look at picture ③. Wherever it glows **red or yellow**, "
        "that's a part of the coral the AI paid the most attention to when deciding. "
    )
