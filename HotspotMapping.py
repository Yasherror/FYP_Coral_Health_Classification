"""
HotspotMapping.py
Interactive folium-based geospatial hotspot map: click a survey site to
see its GPS-derived location, health breakdown, environmental stats, and
sample coral imagery pulled from the processed image archive.
"""

import os                                       # File system operations (paths, directories)
import numpy as np                              # Numerical operations (arrays, math)
import pandas as pd                             # Data manipulation (DataFrames, CSV)
import streamlit as st                          # Web UI framework
import matplotlib.pyplot as plt                # Static plotting
import folium                                   # Interactive map creation
from folium.plugins import (                   # Folium plugins for enhanced maps
    MarkerCluster,                             # Clusters markers for dense data
    HeatMap,                                   # Heatmap overlay
    Fullscreen,                                # Fullscreen map toggle
    MiniMap                                    # Miniature overview map
)
from streamlit_folium import st_folium         # Folium integration with Streamlit

from data_utils import (
    load_dataset, load_image_zip, get_image_from_zip, build_site_geo,
    CSV_PATH, ZIP_PATH, HEALTH_COLORS, HEALTH_MARKER_COLORS, HEALTH_MAP,
)


def _active_labels_for_row(row):
    """
    All health labels actually present on this image (from the has_healthy /
    has_compromised / has_dead multi-hot columns), not just the single
    worst-case 'primary_health' bucket -- a patch can genuinely show more
    than one condition at once.
    """
    return [name for name in HEALTH_MAP.values() if row.get(f"has_{name.lower()}", 0) == 1]


def _label_pills(labels):
    if not labels:
        return '<span style="color:#95c6ce; font-style:italic; font-size:12px;">Unlabeled</span>'
    return "".join(
        f'<span style="display:inline-block; margin:2px 4px 2px 0; padding:0.15rem 0.6rem; '
        f'border-radius:20px; background:{HEALTH_COLORS.get(l, "#3B82F6")}22; '
        f'color:{HEALTH_COLORS.get(l, "#3B82F6")}; font-size:0.72rem; font-weight:bold;">{l}</span>'
        for l in labels
    )


def _render_sample_gallery(site_images, site_name, zf, zip_index):
    st.markdown(f"### Sample Report — Coral Imagery at {site_name}")

    labeled = site_images.copy()
    labeled["_active_labels"] = labeled.apply(_active_labels_for_row, axis=1)
    labeled["_n_labels"] = labeled["_active_labels"].apply(len)

    total = len(labeled)
    mixed_count = int((labeled["_n_labels"] > 1).sum())
    mixed_pct = 100 * mixed_count / total if total else 0

    # --- Headline summary stats, report-style ---
    s1, s2, s3 = st.columns(3)
    s1.metric("Images at this site", f"{total:,}")
    s2.metric("Show mixed conditions", f"{mixed_pct:.0f}%",
              help="Percentage of images flagged with more than one health label at once "
                   "(e.g. partially Healthy and partially Compromised in the same patch).")
    single_label_pct = 100 - mixed_pct
    s3.metric("Show a single clear condition", f"{single_label_pct:.0f}%")

    if labeled.empty:
        st.caption("No labeled samples available at this site.")
        return

    # --- Sample selection: mix of single-label and mixed-label examples so the
    #     gallery actually illustrates both cases, not just the common one. ---
    n_target = min(6, total)
    mixed_examples = labeled[labeled["_n_labels"] > 1]
    single_examples = labeled[labeled["_n_labels"] <= 1]

    n_mixed = min(len(mixed_examples), max(2, n_target // 2)) if not mixed_examples.empty else 0
    n_single = n_target - n_mixed

    picks = []
    if n_mixed:
        picks.append(mixed_examples.sample(n_mixed, random_state=7))
    if n_single and not single_examples.empty:
        picks.append(single_examples.sample(min(n_single, len(single_examples)), random_state=7))
    sample = pd.concat(picks).sample(frac=1, random_state=3).reset_index(drop=True) if picks else labeled.head(0)

    if sample.empty:
        st.caption("No sample images available.")
        return

    st.caption(f"Showing {len(sample)} sample images — including both single-condition and "
               f"mixed-condition patches where available.")

    cols_per_row = 3
    rows = [sample.iloc[i:i + cols_per_row] for i in range(0, len(sample), cols_per_row)]

    for row_chunk in rows:
        cols = st.columns(len(row_chunk))
        for c, (_, r) in zip(cols, row_chunk.iterrows()):
            with c:
                img = get_image_from_zip(zf, zip_index, r["image_url"]) if zf is not None else None
                border_color = HEALTH_COLORS.get(r["primary_health"], "#3B82F6")
                st.markdown(
                    f'<div style="border:2px solid {border_color}66; border-radius:12px; '
                    f'padding:0.5rem; margin-bottom:0.5rem;">',
                    unsafe_allow_html=True,
                )
                if img is not None:
                    st.image(img, use_container_width=True)
                else:
                    st.caption("Image not found in archive.")
                st.markdown(_label_pills(r["_active_labels"]), unsafe_allow_html=True)
                temp_val = r.get("temp")
                if pd.notna(temp_val):
                    st.caption(f"🌡️ {temp_val:.1f}°C")
                st.markdown("</div>", unsafe_allow_html=True)


def render_hotspots():
    st.markdown("### Geospatial Reef Hotspot Mapping — Koh Tao, Thailand")

    # ---- Load data ---------------------------------------------------- #
    try:
        df = load_dataset(CSV_PATH)
    except Exception as e:
        st.error(f"Could not load the coral dataset from `{CSV_PATH}`: {e}")
        return

    zf, zip_index = None, {}
    if os.path.exists(ZIP_PATH):
        try:
            zf, zip_index = load_image_zip(ZIP_PATH)
        except Exception:
            zf, zip_index = None, {}

    site_geo = build_site_geo(df)
    site_geo = site_geo[site_geo["lat"].notna() & site_geo["lon"].notna()].reset_index(drop=True)

    if site_geo.empty:
        st.warning("No valid GPS coordinates could be parsed from `lat_start` / `lng_start`.")
        return

    # ---- KPI summary (computed live from the dataset) ----------------- #
    n_reefs = len(site_geo)
    n_healthy_sites = int((site_geo["dominant_health"] == "Healthy").sum())
    n_compromised_sites = int((site_geo["dominant_health"] == "Compromised").sum())
    n_dead_sites = int((site_geo["dominant_health"] == "Dead").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reef Sites", n_reefs)
    col2.metric("🟢 Dominantly Healthy", n_healthy_sites)
    col3.metric("🟠 Dominantly Compromised", n_compromised_sites)
    col4.metric("🔴 Dominantly Dead", n_dead_sites)

    st.markdown("---")

    # ---- Map controls --------------------------------------------------- #
    map_col, control_col = st.columns([3, 1])
    with control_col:
        st.markdown("**Map Controls**")
        tile_choice = st.radio("Base layer", ["Satellite", "Street"], index=0)
        show_heat = st.checkbox("Bleaching risk heatmap", value=True)
        show_cluster = st.checkbox("Cluster markers", value=False)
        marker_scale = st.slider("Marker size scale", 0.5, 2.0, 1.0, 0.1)
        st.caption("Click a marker on the map, or use the dropdown below, "
                   "to inspect a site in detail.")

    # ---- Build the folium map ------------------------------------------- #
    center_lat, center_lon = site_geo["lat"].mean(), site_geo["lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None,
                    control_scale=True)

    if tile_choice == "Satellite":
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery", name="Satellite", overlay=False,
        ).add_to(m)
        folium.TileLayer("OpenStreetMap", name="Street", overlay=False).add_to(m)
    else:
        folium.TileLayer("OpenStreetMap", name="Street", overlay=False).add_to(m)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery", name="Satellite", overlay=False,
        ).add_to(m)

    Fullscreen(position="topleft").add_to(m)
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)

    marker_target = MarkerCluster().add_to(m) if show_cluster else m
    max_total = site_geo["total"].max()

    for _, row in site_geo.iterrows():
        color = HEALTH_MARKER_COLORS.get(row["dominant_health"], "#3B82F6")
        last_survey_str = (row["last_survey"].strftime("%Y-%m-%d")
                            if pd.notna(row["last_survey"]) else "Unknown")
        popup_html = f"""
        <div style="font-family: sans-serif; width: 230px;">
            <h4 style="margin: 0 0 6px 0;">{row['site_fullname']} ({row['site']})</h4>
            <b>Health Score:</b> {row['health_score']:.2f} / 1.00<br>
            <b>Dominant Condition:</b>
                <span style="color:{color}; font-weight:bold;">{row['dominant_health']}</span><br>
            <b>Healthy:</b> {row['pct_healthy']:.1f}% &nbsp;
            <b>Compromised:</b> {row['pct_compromised']:.1f}% &nbsp;
            <b>Dead:</b> {row['pct_dead']:.1f}%<br>
            <b>Mean Depth:</b> {row['mean_depth']:.1f} m<br>
            <b>Mean Temp:</b> {row['mean_temp']:.1f} °C<br>
            <b>Images Surveyed:</b> {row['total']:,}<br>
            <b>Last Survey:</b> {last_survey_str}
        </div>
        """
        radius = (8 + (row["total"] / max_total) * 14) * marker_scale
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color, weight=2, fill=True, fill_color=color, fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=row["site_fullname"],
        ).add_to(marker_target)

    if show_heat:
        heat_data = [[row["lat"], row["lon"], row["pct_dead"] / 100] for _, row in site_geo.iterrows()]
        HeatMap(heat_data, radius=45, blur=35, min_opacity=0.25,
                gradient={0.2: "#10B981", 0.5: "#F59E0B", 0.8: "#EF4444"}).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)

    with map_col:
        map_state = st_folium(m, width=None, height=520,
                               returned_objects=["last_object_clicked_tooltip"],
                               key="hotspot_map")

    # ---- Resolve selected site: map click takes priority over dropdown --- #
    clicked_name = map_state.get("last_object_clicked_tooltip") if map_state else None
    site_names = site_geo["site_fullname"].tolist()
    default_idx = site_names.index(clicked_name) if clicked_name in site_names else 0

    selected_name = st.selectbox("📍 Select a reef site for detailed inspection",
                                  site_names, index=default_idx)
    site_row = site_geo[site_geo["site_fullname"] == selected_name].iloc[0]

    # ---- Site detail panel ------------------------------------------------ #
    st.markdown("---")
    badge_color = HEALTH_MARKER_COLORS.get(site_row["dominant_health"], "#3B82F6")
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.75rem;">
        <h3 style="margin:0;">{site_row['site_fullname']} <span style="opacity:0.6; font-size:0.8em;">({site_row['site']})</span></h3>
        <span style="padding:0.2rem 0.7rem; border-radius:20px; background:{badge_color}22;
                     color:{badge_color}; font-weight:bold; font-size:0.8rem;">
            {site_row['dominant_health']}
        </span>
    </div>
    """, unsafe_allow_html=True)

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Health Score", f"{site_row['health_score']:.2f}")
    d2.metric("Mean Depth", f"{site_row['mean_depth']:.1f} m",
              f"{site_row['min_depth']:.1f}–{site_row['max_depth']:.1f} m range")
    d3.metric("Mean Temp", f"{site_row['mean_temp']:.1f} °C",
              f"{site_row['min_temp']:.1f}–{site_row['max_temp']:.1f} °C range")
    d4.metric("Images Surveyed", f"{site_row['total']:,}")
    d5.metric("Last Survey", site_row["last_survey"].strftime("%Y-%m-%d")
              if pd.notna(site_row["last_survey"]) else "N/A")

    # Health breakdown bar for the selected site
    fig, ax = plt.subplots(figsize=(8, 1.4))
    left = 0
    for name, pct in [("Healthy", site_row["pct_healthy"]),
                      ("Compromised", site_row["pct_compromised"]),
                      ("Dead", site_row["pct_dead"])]:
        ax.barh([0], [pct], left=left, color=HEALTH_COLORS[name], height=0.6,
                edgecolor="white", linewidth=1)
        if pct > 4:
            ax.text(left + pct / 2, 0, f"{pct:.0f}%", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        left += pct
    ax.set_xlim(0, 100)
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ---- Sample coral images for this site (report-style, mixed-label aware) --- #
    st.markdown("---")
    site_images = df[df["site"] == site_row["site"]]

    if zf is not None and not site_images.empty:
        _render_sample_gallery(site_images, site_row["site_fullname"], zf, zip_index)
    else:
        st.info(f"Image archive not found or empty (expected at `{ZIP_PATH}`).")

    # ---- Site ranking table -------------------------------------------------- #
    with st.expander(" Full site ranking table"):
        st.dataframe(
            site_geo[["rank", "site_fullname", "total", "pct_healthy", "pct_compromised",
                      "pct_dead", "mean_depth", "mean_temp", "health_score"]]
            .rename(columns={"site_fullname": "Site", "total": "Images",
                              "pct_healthy": "% Healthy", "pct_compromised": "% Compromised",
                              "pct_dead": "% Dead", "mean_depth": "Mean Depth (m)",
                              "mean_temp": "Mean Temp (°C)", "health_score": "Health Score"})
            .round(2).set_index("rank"),
            use_container_width=True,
        )