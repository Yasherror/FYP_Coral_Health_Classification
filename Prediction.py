"""
Prediction.py
Single tab: image upload -> underwater enhancement -> EfficientNet-B3
multi-label inference -> results breakdown -> human-in-the-loop annotation
and save to the live database.
"""
import io                   # Input/output operations (bytes, streams)
import time                 # Time functions (delays, performance timing)
from datetime import date   # Date handling (current date, date objects)
import streamlit as st      # Web UI framework

from data_utils import (
    predict_image, load_dataset, load_model, clear_model_cache,
    bytes_to_pil_image, image_bytes_fingerprint, predicted_label_vector,
    enhance_pil_image, enhance_available,
    CSV_PATH, HEALTH_LABELS, STRESSOR_LABELS, PRED_THRESHOLD,
    HEALTH_COLORS, STRESSOR_COLORS, ALL_LABELS, STRESSOR_MAP,
    HEALTH_RECOMMENDATIONS, STRESSOR_RECOMMENDATIONS,
    save_patch_record, db_counts,
    get_site_options, get_site_defaults, get_surveys_for_site, next_surveyid,
    parse_patch_filename, get_survey_defaults_from_filename,
)


def _run_inference():
    st.session_state.is_loading = True
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    try:
        status_placeholder.info("Loading EfficientNet-B3 model...")
        progress_placeholder.progress(15, text="Loading model...")
        time.sleep(0.2)

        status_placeholder.info("Enhancing image (underwater correction)...")
        progress_placeholder.progress(40, text="Enhancing + preprocessing...")
        pil_original = bytes_to_pil_image(st.session_state.selected_image_bytes)
        # Enhance BEFORE prediction so inference matches the training distribution.
        pil_for_model = enhance_pil_image(pil_original)   # depth unknown here -> default weight
        # Keep the enhanced image (this is what we save into coral_images_processed).
        buf = io.BytesIO()
        pil_for_model.save(buf, format="JPEG", quality=95)
        st.session_state.processed_image_bytes = buf.getvalue()
        time.sleep(0.2)

        status_placeholder.info("Running inference...")
        progress_placeholder.progress(70, text="Running inference...")
        start_time = time.time()
        probs = predict_image(pil_for_model, debug=True)
        inference_time = time.time() - start_time

        progress_placeholder.progress(100, text="Analysis complete")
        status_placeholder.success(f"Inference completed in {inference_time:.2f} seconds")
        time.sleep(0.4)
    except Exception as e:
        st.session_state.error_text = f"Model inference failed: {e}"
        status_placeholder.error(f"Error: {e}")
        st.session_state.is_loading = False
        return

    progress_placeholder.empty()
    status_placeholder.empty()

    if probs is not None:
        health_probs = {k: probs[k] for k in HEALTH_LABELS}
        stressor_probs = {k: probs[k] for k in STRESSOR_LABELS}
        primary_health = max(health_probs, key=health_probs.get)
        predicted_labels = [l for l, p in probs.items() if p >= PRED_THRESHOLD]
        dataset_context = None
        try:
            df_ctx = load_dataset(CSV_PATH)
            dataset_context = {
                "pct_healthy": 100 * (df_ctx["primary_health"] == "Healthy").mean(),
                "pct_compromised": 100 * (df_ctx["primary_health"] == "Compromised").mean(),
                "pct_dead": 100 * (df_ctx["primary_health"] == "Dead").mean(),
            }
        except Exception:
            pass
        _model, _device, fingerprint, load_report = load_model()
        image_hash = image_bytes_fingerprint(st.session_state.selected_image_bytes)
        st.session_state.active_analysis = {
            "probs": probs, "health_probs": health_probs, "stressor_probs": stressor_probs,
            "primary_health": primary_health, "primary_health_conf": health_probs[primary_health],
            "predicted_labels": predicted_labels, "dataset_context": dataset_context,
            "inference_time": inference_time, "model_fingerprint": fingerprint,
            "load_report": load_report, "image_hash": image_hash,
            "enhanced": enhance_available(),
        }
        st.session_state.gradcam_result = None
        st.session_state.save_done = None
    else:
        st.error(st.session_state.error_text)

    st.session_state.is_loading = False
    st.rerun()


def _clear_all():
    st.session_state.selected_image_bytes = None
    st.session_state.processed_image_bytes = None
    st.session_state.file_name = None
    st.session_state.active_analysis = None
    st.session_state.error_text = None
    st.session_state.is_loading = False
    st.session_state.gradcam_result = None
    st.session_state.gradcam_autorun = False
    st.session_state.save_done = None
    st.rerun()


# def _render_diagnostics_panel():
#     with st.expander("Diagnostics — model load status"):
#         col1, col2 = st.columns([3, 1])
#         with col2:
#             if st.button("Clear Model Cache & Reload", use_container_width=True):
#                 clear_model_cache()
#                 st.session_state.active_analysis = None
#                 st.rerun()
#         with col1:
#             try:
#                 _model, _device, fingerprint, load_report = load_model()
#                 st.success(
#                     f"Loaded **{load_report['architecture']}** ({load_report['prefix']}) — "
#                     f"**{load_report['matched']}/{load_report['total']}** tensors "
#                     f"({load_report['match_pct']:.1f}%) on **{load_report['device']}**. "
#                     f"Fingerprint: `{fingerprint}`")
#             except Exception as e:
#                 st.error(f"Model failed to load: {e}")


def _render_upload_section():
    st.markdown("### Upload Coral Image")
    if not enhance_available():
        st.caption("Note: OpenCV not detected — images will be classified without underwater "
                   "enhancement. Install with `pip install opencv-python` to match training.")
    #_render_diagnostics_panel()
    uploaded_file = st.file_uploader("Choose a coral image", type=["png", "jpg", "jpeg"],
                                     label_visibility="collapsed")
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        pil_image = bytes_to_pil_image(file_bytes)
        st.image(pil_image, caption=uploaded_file.name, use_container_width=True)
        st.session_state.selected_image_bytes = file_bytes
        st.session_state.file_name = uploaded_file.name
        col1, col2, col3 = st.columns([1, 2, 1])
    else:
        if st.session_state.selected_image_bytes is not None:
            _clear_all()

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.selected_image_bytes is not None:
            if st.button("Run Prediction", type="primary", use_container_width=True):
                _run_inference()
        else:
            st.button("Run Prediction", disabled=True, use_container_width=True)


def _pill(label):
    c = HEALTH_COLORS.get(label, STRESSOR_COLORS.get(label, "#3B82F6"))
    return (f'<span style="display:inline-block; margin:2px 4px 2px 0; padding:0.15rem 0.6rem; '
            f'border-radius:20px; background:{c}22; color:{c}; font-size:0.75rem; font-weight:bold;">'
            f'{label}</span>')


PLAIN_VERDICT = {
    "Healthy": ("This coral looks HEALTHY.",
                "The AI sees mostly living, healthy coral tissue in this image."),
    "Compromised": ("This coral looks STRESSED / COMPROMISED.",
                    "The AI sees signs of stress (e.g. bleaching or damage). It is alive but not healthy."),
    "Dead": ("This coral appears to be DEAD.",
             "The AI sees mostly dead coral — bare skeleton or algae-covered rock."),
}


def _render_label_readout(r):
    rows = predicted_label_vector(r["probs"])
    top = next(x for x in rows if x["is_top"])
    headline, sub = PLAIN_VERDICT.get(top["label"], (f"Top condition: {top['label']}.", ""))
    top_color = HEALTH_COLORS.get(top["label"], "#0EA5E9")

    st.markdown("### What the AI Detected")
    st.markdown(
        f"<div style='background:{top_color}1f;border-left:6px solid {top_color};"
        f"border-radius:14px;padding:1rem 1.25rem;margin:0.25rem 0 1rem 0;'>"
        f"<div style='font-size:22px;font-weight:800;color:#eafcff;'>{headline}</div>"
        f"<div style='font-size:14px;color:#cfeef4;margin-top:0.25rem;'>{sub}</div>",
        unsafe_allow_html=True)
    st.caption("Each coral condition is checked on its own. A tick = the AI flagged it. "
               "The bar shows how confident the AI is in each one.")

    card_html = ["<div style='display:flex;flex-direction:column;gap:0.6rem;'>"]
    for x in rows:
        label, prob, bit = x["label"], x["prob"], x["bit"]
        color = HEALTH_COLORS.get(label, STRESSOR_COLORS.get(label, "#3B82F6"))
        pct = prob * 100
        detected = bit == 1
        badge_bg = f"{color}22" if detected else "rgba(255,255,255,0.06)"
        badge_col = color if detected else "#95c6ce"
        badge_txt = "DETECTED" if detected else "Not detected"
        border = color if detected else "rgba(255,255,255,0.08)"
        card_html.append(f"""
        <div style="background:rgba(255,255,255,0.05);border-radius:14px;padding:0.9rem 1.1rem;
                    border:1px solid {border};">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                <div style="font-size:17px;font-weight:700;color:#e8f6f9;">
                    <span style="display:inline-block;width:11px;height:11px;border-radius:50%;
                                 background:{color};margin-right:8px;"></span>{label}</div>
                <div style="display:flex;align-items:center;gap:0.75rem;">
                    <span style="font-family:monospace;font-size:18px;font-weight:800;color:{badge_col};">{bit}</span>
                    <span style="padding:0.2rem 0.7rem;border-radius:20px;background:{badge_bg};
                                 color:{badge_col};font-size:0.72rem;font-weight:700;">{badge_txt}</span>
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.12);border-radius:8px;height:12px;overflow:hidden;">
                <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:8px;"></div>
            </div>
            <div style="text-align:right;font-size:12px;color:#95c6ce;margin-top:0.25rem;">{pct:.1f}% confidence</div>
        </div>""")
    card_html.append("</div>")
    st.markdown("".join(card_html), unsafe_allow_html=True)

    vector = [x["bit"] for x in rows]
    chips = "&nbsp;".join(
        f"<span style='font-family:monospace;font-size:15px;font-weight:800;"
        f"color:{HEALTH_COLORS.get(x['label'], '#b4e2e9')};'>{x['label']}={x['bit']}</span>"
        for x in rows)
    st.markdown(
        f"<div style='margin-top:1rem;background:rgba(255,255,255,0.04);border-radius:12px;"
        f"padding:0.85rem 1.1rem;'>"
        f"<div style='font-size:13px;color:#95c6ce;margin-bottom:0.35rem;'><b>Prediction vector</b> "
        f"(per-label / Hamming view)</div>"
        f"<div style='font-family:monospace;font-size:26px;font-weight:800;color:#b4e2e9;'>"
        f"[{', '.join(map(str, vector))}]</div>"
        f"<div style='margin-top:0.4rem;'>{chips}</div>"
        f"<div style='font-size:12px;color:#95c6ce;margin-top:0.5rem;'>"
        f"Read left-to-right as <b>Healthy, Compromised, Dead</b>. A <b>1</b> means present, "
        f"a <b>0</b> means not. Each slot is judged on its own — the Hamming (per-label) view.</div></div>",
        unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# HUMAN REVIEW & SAVE (annotation + write to live database)
# --------------------------------------------------------------------------- #
def _render_annotation_and_save(r):
    st.markdown("---")
    st.markdown("### Human Review & Save to Dataset")
    n_surveys, n_patches, n_anno = db_counts()
    st.caption(f"Confirm or correct the AI result, choose the site and survey date "
               f"(details auto-fill), then save.  ·  Stored so far: "
               f"{n_patches:,} patches · {n_surveys} surveys · {n_anno} annotations.")

    ai_health = r.get("predicted_labels") or [r["primary_health"]]

    # Auto-fill from the uploaded image filename, e.g.
    # 'CBK_0039_00_20240126_0038_35.jpg' -> site CBK, date 20240126, survey 0039.
    upload_name = st.session_state.get("file_name") or ""
    fn_info = parse_patch_filename(upload_name)          # {} or {patchid, site, ...}
    fn_auto = get_survey_defaults_from_filename(upload_name)  # metadata-joined defaults
    fn_site = fn_info.get("site", "")
    fn_date = fn_info.get("survey_date", "")
    if fn_info.get("patchid"):
        st.info(f"Patch ID from filename: **{fn_info['patchid']}** — this is used as "
                f"the annotation key. Details below are auto-filled from the name; edit if needed.")

    # Step 1 — label source
    mode = st.radio("1. Coral health labels", ["Accept AI prediction", "Correct manually"],
                    horizontal=True, key="anno_mode")

    # Step 2 — site location (nine trained sites only; no new-site option)
    site_opts = get_site_options()  # [(code, fullname), ...]
    if not site_opts:
        st.warning("No survey sites found in surveys_metadata.csv. Load the metadata file "
                   "(the nine Koh Tao sites) before saving.")
        return
    site_choices = [f"{code} — {fn}" if fn else code for code, fn in site_opts]
    code_by_choice = {(f"{code} — {fn}" if fn else code): code for code, fn in site_opts}
    # default the selector to the site parsed from the filename, if we have it
    site_index = 0
    if fn_site:
        for i, (code, _fn) in enumerate(site_opts):
            if str(code).upper() == fn_site.upper():
                site_index = i
                break
    site_pick = st.selectbox("2. Site location (trained sites only)", site_choices,
                             index=site_index, key="anno_site")
    site_code = code_by_choice.get(site_pick, "")
    site_def = get_site_defaults(site_code)

    # Step 3 — existing survey date, or a new one
    existing_dates = []
    subs = get_surveys_for_site(site_code)
    if subs is not None and not subs.empty and "survey_date" in subs.columns:
        existing_dates = sorted({str(d) for d in subs["survey_date"] if str(d).strip()})

    picked_row = {}
    if existing_dates:
        # if the filename date is already on record, default to picking it
        default_datemode = 0 if (fn_date and fn_date in existing_dates) else 1
        date_mode = st.radio("3. Survey date", ["Existing survey date", "New survey date"],
                             index=default_datemode, horizontal=True, key="anno_datemode")
    else:
        date_mode = "New survey date"
        st.caption("This site has no recorded survey dates yet — enter a new one below.")

    if date_mode == "Existing survey date" and existing_dates:
        date_index = existing_dates.index(fn_date) if fn_date in existing_dates else 0
        picked_date = st.selectbox("Choose existing survey date", existing_dates,
                                   index=date_index, key="anno_date")
        match = subs[subs["survey_date"].astype(str) == str(picked_date)]
        picked_row = match.iloc[0].to_dict() if not match.empty else {}
    else:
        # default the date picker to the filename date when we have one
        try:
            default_date = (date(int(fn_date[:4]), int(fn_date[4:6]), int(fn_date[6:8]))
                            if len(fn_date) == 8 else date.today())
        except Exception:
            default_date = date.today()
        picked_date = st.date_input("New survey date", value=default_date,
                                    key="anno_newdate").strftime("%Y%m%d")

    # merged auto-fill: site defaults, then filename-joined metadata, then the picked survey row
    af = dict(site_def)
    af.update({k: str(v) for k, v in fn_auto.items() if str(v).strip()})
    af.update({k: str(v) for k, v in picked_row.items() if str(v).strip()})
    af["site"] = site_code or af.get("site", "")
    af["survey_date"] = picked_date

    # Step 4 — the form (auto-filled, editable)
    with st.form("save_patch_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            health = st.multiselect(
                "Coral health condition(s)", ALL_LABELS, default=ai_health,
                disabled=(mode == "Accept AI prediction"),
                help="The AI's health prediction. Switch to 'Correct manually' to edit.")
        with c2:
            stressors = st.multiselect(
                "Stressor(s) — human-added", list(STRESSOR_MAP.values()), default=[],
                help="The model does not predict stressors; add them here from your own review.")

        default_surveyid = af.get("surveyid", "")
        if not default_surveyid and date_mode == "New survey date":
            default_surveyid = next_surveyid()

        st.markdown("**Survey details** — auto-filled from your site/date choice; edit if needed")
        m1, m2, m3 = st.columns(3)
        surveyid = m1.text_input("Survey ID", value=str(default_surveyid))
        transectid = m2.text_input("Transect ID", value=str(af.get("transectid", "00")))
        annotated_by = m3.text_input("Annotated by", value="researcher")

        s1, s2, s3 = st.columns(3)
        site = s1.text_input("Site code", value=str(af.get("site", "")))
        site_fullname = s2.text_input("Site full name", value=str(af.get("site_fullname", "")))
        camera = s3.text_input("Camera", value=str(af.get("camera", "Olympus TG-6")))

        g1, g2, g3, g4 = st.columns(4)
        lat = g1.text_input("Lat start", value=str(af.get("lat_start", "")))
        lng = g2.text_input("Lng start", value=str(af.get("lng_start", "")))
        depth = g3.text_input("Depth", value=str(af.get("depth", "")), help="e.g. 9.2m")
        temp = g4.text_input("Temp", value=str(af.get("temp", "")), help="e.g. 28.3℃")

        st.caption(f"Survey date: **{picked_date}**  ·  "
                   f"{'existing' if date_mode == 'Existing survey date' else 'new'} date")
        submitted = st.form_submit_button("Save patch to database",
                                          type="primary", use_container_width=True)

    if submitted:
        final_health = ai_health if mode == "Accept AI prediction" else health
        if not final_health:
            st.error("Please select at least one coral health label before saving.")
            return
        if not site.strip():
            st.error("Please choose a site location before saving.")
            return
        # Save the ENHANCED image (falls back to the original upload if needed).
        image_bytes = st.session_state.get("processed_image_bytes") or st.session_state.get("selected_image_bytes")
        if not image_bytes:
            st.error("No image is loaded to save.")
            return
        sd = picked_date
        folder = f"{sd}_{site.strip()}" if site.strip() else sd
        meta = dict(surveyid=(surveyid.strip() or sd), transectid=transectid.strip(),
                    survey_date=sd, site=site.strip(), site_fullname=site_fullname.strip(),
                    folder_name=folder, lat_start=lat.strip(), lng_start=lng.strip(),
                    camera=camera.strip(), depth=depth.strip(), temp=temp.strip())
        src = "ai_accepted" if mode == "Accept AI prediction" else "human_corrected"
        try:
            res = save_patch_record(final_health, stressors, meta, image_bytes,
                                    source_name=st.session_state.get("file_name"),
                                    ai_health_labels=ai_health, source=src,
                                    annotated_by=(annotated_by.strip() or "researcher"))
            st.session_state.save_done = res
        except Exception as e:
            st.error(f"Could not save: {e}")
            return

    done = st.session_state.get("save_done")
    if done:
        st.success(f"Saved **{done['image_filename']}** (patch {done['patchid']}, "
                   f"annotation {done['annotation_id']}) to the live database.")


def _render_results_section():
    r = st.session_state.active_analysis
    status = r["primary_health"]
    status_color = HEALTH_COLORS.get(status, "#3B82F6")

    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<span style="font-size: 22px; font-weight:bold;">AI Classification Outcome</span>',
                    unsafe_allow_html=True)
    with col2:
        if st.button("New Analysis", use_container_width=True):
            _clear_all()

    if "inference_time" in r:
        note = "Inference on the enhanced image" if r.get("enhanced") else "Inference (enhancement unavailable)"
        st.success(f"{note} — completed in {r['inference_time']:.2f} seconds")

    # with st.expander("Diagnostics for this prediction"):
    #     lr = r.get("load_report", {})
    #     st.markdown(
    #         f"- **Backbone used:** {lr.get('architecture', 'n/a')} ({lr.get('prefix', 'n/a')}), "
    #         f"{lr.get('match_pct', 0):.1f}% of weights matched the checkpoint\n"
    #         f"- **Classifier weight fingerprint:** `{r.get('model_fingerprint')}`\n"
    #         f"- **This image's byte hash:** `{r.get('image_hash')}`")

    _render_label_readout(r)

    st.markdown("---")
    col_img, col_info = st.columns(2)
    with col_img:
        show_bytes = st.session_state.get("processed_image_bytes") or st.session_state.get("selected_image_bytes")
        if show_bytes:
            cap = "Enhanced image (used for prediction)" if r.get("enhanced") else "Uploaded image"
            st.image(bytes_to_pil_image(show_bytes), use_container_width=True, caption=cap)
    with col_info:
        labels_html = "".join(_pill(l) for l in r["predicted_labels"]) or \
            "<i style='color:#95c6ce;'>No label crossed the 50% threshold</i>"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); border-radius: 16px; padding: 1.5rem;">
            <div style="display: inline-block; padding: 0.25rem 1rem; border-radius: 9999px;
                        background: {status_color}20; color: {status_color}; font-weight: bold;">
                {status.upper()}
            </div>
            <div style="margin-top: 1rem;">
                <div style="font-size: 12px; color: #95c6ce;">Primary Health Confidence</div>
                <div style="font-size: 24px; font-weight: bold;">{(r['primary_health_conf'] * 100):.1f}%</div>
            </div>
            <div style="margin-top: 1rem;">
                <div style="font-size: 12px; color: #95c6ce;">Detected Labels (>= 50%)</div>
                <div style="margin-top: 0.35rem;">{labels_html}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if r.get("dataset_context"):
        ctx = r["dataset_context"]
        st.markdown("### How This Compares to the Sanctuary Dataset")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Dataset % Healthy", f"{ctx['pct_healthy']:.1f}%",
                   f"{r['health_probs']['Healthy'] * 100 - ctx['pct_healthy']:+.1f} pts vs this patch")
        cc2.metric("Dataset % Compromised", f"{ctx['pct_compromised']:.1f}%",
                   f"{r['health_probs']['Compromised'] * 100 - ctx['pct_compromised']:+.1f} pts vs this patch")
        cc3.metric("Dataset % Dead", f"{ctx['pct_dead']:.1f}%",
                   f"{r['health_probs']['Dead'] * 100 - ctx['pct_dead']:+.1f} pts vs this patch")

    st.markdown("---")
    st.markdown("### Recommendations")
    st.markdown(f"- {HEALTH_RECOMMENDATIONS.get(status, '')}")
    top_stressors = [l for l in r["predicted_labels"] if l in STRESSOR_LABELS]

    # Grad-CAM launcher
    st.markdown("---")
    st.markdown("### See Where the AI Looked")
    st.caption("Generate a heat-map overlay (Grad-CAM) highlighting the regions that drove the "
               "AI's decision. Red / yellow = the areas that mattered most.")
    lc, cc, rc = st.columns([1, 2, 1])
    with cc:
        if st.button("Show Heat Map Overlay", type="primary", use_container_width=True):
            st.session_state.gradcam_autorun = True
            st.session_state.gradcam_result = None
            st.session_state.active_tab = "heatmap"
            st.rerun()

    # Human review + save to live database
    _render_annotation_and_save(r)


def render_prediction():
    for key, default in [("selected_image_bytes", None), ("processed_image_bytes", None),
                         ("file_name", None), ("active_analysis", None), ("error_text", None),
                         ("is_loading", False), ("gradcam_result", None),
                         ("gradcam_autorun", False), ("save_done", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    _render_upload_section()
    if st.session_state.active_analysis:
        _render_results_section()
    if st.session_state.get('is_loading', False):
        st.info("Processing your image... Please wait.")