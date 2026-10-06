"""
Overview.py
Dashboard OVERVIEW tab — rewritten to read like a plain-language reef-health
REPORT for non-technical readers rather than a wall of EDA charts.
"""

import os                # File system operations (paths, directories)
import pandas as pd      # Data manipulation (DataFrames, CSV)
import streamlit as st   # Web UI framework

from data_utils import (
    load_dataset, load_image_zip, get_image_from_zip,
    build_site_geo, CSV_PATH, ZIP_PATH,
    HEALTH_MAP, STRESSOR_MAP, HEALTH_COLORS, STRESSOR_COLORS,
)

HEALTH_ORDER = ["Healthy", "Compromised", "Dead"]
HEALTH_ICON = {"Healthy": "🟢", "Compromised": "🟠", "Dead": "🔴"}


# --------------------------------------------------------------------------- #
# Small pure helpers (unit-testable without Streamlit)
# --------------------------------------------------------------------------- #
def _true_labels(row):
    """
    Return (health_labels, stressor_labels) actually present for one image,
    read from the has_* columns -- i.e. the ground-truth MULTI-LABEL set, not
    just the single 'primary' health label.
    """
    health = [name for name in HEALTH_MAP.values()
              if int(row.get(f"has_{name.lower()}", 0) or 0) == 1]
    stressors = [name for name in STRESSOR_MAP.values()
                 if int(row.get(f"has_{name.lower()}", 0) or 0) == 1]
    return health, stressors


def _health_grade(score01):
    """Map a 0-1 health score to a (label, color) grade."""
    if score01 >= 0.60:
        return "Good", "#27ae60"
    if score01 >= 0.40:
        return "Fair", "#e67e22"
    return "Poor", "#c0392b"


# --------------------------------------------------------------------------- #
# HTML building blocks
# --------------------------------------------------------------------------- #
def _pill(text, color):
    return (f"<span style='display:inline-block;margin:2px 4px 2px 0;padding:0.15rem 0.6rem;"
            f"border-radius:20px;background:{color}26;color:{color};font-size:0.72rem;"
            f"font-weight:700;'>{text}</span>")


def _bar_row(label, pct, count, color):
    icon = HEALTH_ICON.get(label, "•")
    return f"""
    <div style="margin-bottom:0.7rem;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem;">
        <span style="font-size:15px;font-weight:700;color:#e8f6f9;">{icon} {label}</span>
        <span style="font-size:13px;color:#cfeef4;">{pct:.1f}% &nbsp;·&nbsp; {count:,} images</span>
      </div>
      <div style="background:rgba(255,255,255,0.12);border-radius:8px;height:14px;overflow:hidden;">
        <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:8px;"></div>
      </div>
    </div>"""


def _metric_card(idx, title, value, desc, icon, color, progress):
    progress = max(0, min(100, progress))
    return f"""
    <div style="background:rgba(255,255,255,0.06);border:1px solid rgba(180,226,233,0.18);
                border-left:5px solid {color};border-radius:18px;padding:1.1rem 1.2rem;height:100%;
                box-shadow:0 6px 18px rgba(0,0,0,0.18);">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div style="font-size:0.62rem;font-family:monospace;font-weight:700;letter-spacing:0.12em;
                    color:#7fd0c2;text-transform:uppercase;">Metric 0{idx}</div>
        <div style="font-size:1.5rem;">{icon}</div>
      </div>
      <div style="font-size:0.8rem;color:#cfeef4;font-weight:600;margin-top:0.2rem;">{title}</div>
      <div style="font-size:1.9rem;font-weight:800;color:#eafcff;letter-spacing:-0.02em;margin-top:0.25rem;">{value}</div>
      <div style="font-size:0.72rem;color:#95c6ce;line-height:1.3;margin-top:0.25rem;">{desc}</div>
      <div style="background:rgba(255,255,255,0.10);border-radius:9999px;height:6px;margin-top:0.6rem;overflow:hidden;">
        <div style="width:{progress}%;height:100%;background:{color};border-radius:9999px;"></div>
      </div>
    </div>"""


def _section(title, subtitle=""):
    sub = (f"<div style='font-size:0.8rem;color:#95c6ce;margin-top:0.15rem;'>{subtitle}</div>"
           if subtitle else "")
    st.markdown(
        f"<div style='margin:1.9rem 0 0.9rem 0;'>"
        f"<div style='font-size:1.15rem;font-weight:800;color:#b4e2e9;'>{title}</div>{sub}"
        f"<div style='height:2px;background:rgba(180,226,233,0.2);margin-top:0.5rem;'></div></div>",
        unsafe_allow_html=True,
    )


def _inject_css():
    st.markdown("""
    <style>
      .ov-banner{
        background:linear-gradient(135deg,#0d5566 0%,#0a3d49 100%);
        border:1px solid rgba(149,198,206,0.25);border-radius:22px;
        padding:1.4rem 1.6rem;margin-bottom:1.2rem;box-shadow:0 8px 24px rgba(0,0,0,0.25);
      }
      .ov-index{
        display:flex;align-items:center;gap:1.25rem;border-radius:20px;padding:1.1rem 1.4rem;
        margin-bottom:0.6rem;border:1px solid rgba(255,255,255,0.12);
      }
      .stTabs [data-baseweb="tab-list"]{gap:8px;}
      .stTabs [data-baseweb="tab"]{background:rgba(255,255,255,0.05);border-radius:8px;padding:8px 16px;font-weight:600;}
      .stTabs [aria-selected="true"]{background:rgba(14,165,233,0.25);border-bottom:3px solid #0EA5E9;}
    </style>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _render_executive_summary(df, site_geo, nums):
    best = site_geo.iloc[0] if len(site_geo) else None
    worst = site_geo.iloc[-1] if len(site_geo) else None
    grade, gcolor = _health_grade(nums["overall_score"])

    best_txt = (f"The healthiest reef is <b>{best['site_fullname']}</b> "
                f"({best['pct_healthy']:.0f}% healthy)."
                if best is not None else "")
    worst_txt = (f"The most at-risk is <b>{worst['site_fullname']}</b> "
                 f"({worst['pct_dead']:.0f}% dead coral)."
                 if worst is not None and len(site_geo) > 1 else "")
    temp_txt = (f"Average water temperature across surveys is "
                f"<b>{nums['mean_temp']:.1f}°C</b>." if pd.notna(nums["mean_temp"]) else "")


def _render_metric_cards(nums):
    cards = [
        (1, "Reefs Monitored", f"{nums['n_sites']} sites",
         "GPS-tracked survey locations around Koh Tao.", "📍", "#3B82F6",
         min(100, nums["n_sites"] * 10)),
        (2, "Healthy Coral", f"{nums['pct_healthy']:.0f}%",
         "Share of images showing healthy coral.", "🛡️", "#27ae60",
         nums["pct_healthy"]),
        (3, "Avg Water Temp", f"{nums['mean_temp']:.1f}°C" if pd.notna(nums["mean_temp"]) else "N/A",
         "Warmer water raises bleaching risk.", "🌡️", "#F59E0B",
         min(100, (nums["mean_temp"] or 0) * 3)),
        (4, "Coral Needing Attention", f"{nums['n_attention']:,}",
         "Images labelled stressed or dead.", "⚠️", "#EF4444",
         min(100, 100 * nums["n_attention"] / max(1, nums["total"]))),
    ]
    cols = st.columns(4)
    for c, card in zip(cols, cards):
        with c:
            st.markdown(_metric_card(*card), unsafe_allow_html=True)


def _render_overall_health(nums):
    _section("🩺 Overall Coral Health",
             "How the sanctuary's coral splits across the three conditions.")
    rows = "".join(
        _bar_row(name, nums[f"pct_{key}"], nums[f"n_{key}"], HEALTH_COLORS[name])
        for name, key in [("Healthy", "healthy"), ("Compromised", "comp"), ("Dead", "dead")]
    )
    st.markdown(f"<div>{rows}</div>", unsafe_allow_html=True)

    # A plain-language "key finding".
    if nums["pct_dead"] >= 25:
        note, col = "⚠️ A large share of coral is dead — restoration should be prioritised.", "#c0392b"
    elif nums["pct_healthy"] >= 60:
        note, col = "✅ Most coral is healthy — keep up current protection measures.", "#27ae60"
    else:
        note, col = "🟠 Coral is under noticeable stress — increased monitoring is advised.", "#e67e22"
    st.markdown(
        f"<div style='background:{col}1f;border-left:4px solid {col};border-radius:10px;"
        f"padding:0.6rem 0.9rem;margin-top:0.5rem;font-size:14px;color:#eafcff;'>{note}</div>",
        unsafe_allow_html=True,
    )


def _render_by_site(df, site_geo):
    if "site" not in df.columns or df["site"].nunique() == 0:
        return
    _section("🗺️ Reef-by-Reef Comparison",
             "Compare how each survey site is doing. Toggle the view or open a site for detail.")

    site_health = df.groupby("site")[["has_healthy", "has_compromised", "has_dead"]].sum()
    site_health.columns = HEALTH_ORDER
    view = st.radio("View", ["Percentage", "Count"], horizontal=True, key="ov_site_view")
    if view == "Percentage":
        data = site_health.div(site_health.sum(axis=1).replace(0, pd.NA), axis=0) * 100
        st.caption("Each bar = 100% of that site's images, split by condition.")
    else:
        data = site_health
        st.caption("Number of images per condition at each site.")
    st.bar_chart(data, color=["#27ae60", "#e67e22", "#c0392b"], height=340)

    # Best / worst callouts
    if len(site_geo) >= 2:
        best, worst = site_geo.iloc[0], site_geo.iloc[-1]
        cc1, cc2 = st.columns(2)
        cc1.markdown(
            f"<div style='background:#27ae601f;border-left:4px solid #27ae60;border-radius:10px;"
            f"padding:0.7rem 1rem;'>🏆 <b>Healthiest reef</b><br>"
            f"<span style='font-size:15px;color:#eafcff;'>{best['site_fullname']}</span><br>"
            f"<span style='font-size:12px;color:#95c6ce;'>{best['pct_healthy']:.0f}% healthy ",
            unsafe_allow_html=True)
        cc2.markdown(
            f"<div style='background:#c0392b1f;border-left:4px solid #c0392b;border-radius:10px;"
            f"padding:0.7rem 1rem;'>🚨 <b>Most at-risk reef</b><br>"
            f"<span style='font-size:15px;color:#eafcff;'>{worst['site_fullname']}</span><br>"
            f"<span style='font-size:12px;color:#95c6ce;'>{worst['pct_dead']:.0f}% dead ",
            unsafe_allow_html=True)

    # Drill-down
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    pick = st.selectbox("🔍 Open a reef site for detail", site_geo["site"].tolist(),
                        format_func=lambda s: site_geo.loc[site_geo["site"] == s, "site_fullname"].iloc[0]
                        if (site_geo["site"] == s).any() else s)
    row = site_geo[site_geo["site"] == pick].iloc[0]
    grade, gcol = _health_grade(row["health_score"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Images", f"{int(row['total']):,}")
    m2.metric("Healthy", f"{row['pct_healthy']:.0f}%")
    m3.metric("Avg depth", f"{row['mean_depth']:.1f} m" if pd.notna(row["mean_depth"]) else "N/A")
    m4.metric("Avg temp", f"{row['mean_temp']:.1f}°C" if pd.notna(row["mean_temp"]) else "N/A")
    bars = "".join(
        _bar_row(n, row[f"pct_{k}"], int(row[f"n_{k2}"]), HEALTH_COLORS[n])
        for n, k, k2 in [("Healthy", "healthy", "healthy"),
                         ("Compromised", "compromised", "compromised"),
                         ("Dead", "dead", "dead")]
    )
    st.markdown(
        f"<div style='background:{gcol}14;border-radius:12px;padding:0.9rem 1.1rem;margin-top:0.4rem;'>"
        f"<div style='font-weight:700;color:#eafcff;margin-bottom:0.5rem;'>"
        f"{row['site_fullname']} — health grade <span style='color:{gcol};'>{grade}</span></div>"
        f"{bars}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("📋 Full site ranking table (sortable)"):
        show = site_geo[["rank", "site_fullname", "total", "pct_healthy",
                         "pct_compromised", "pct_dead", "mean_depth", "mean_temp", "health_score"]].copy()
        show.columns = ["Rank", "Site", "Images", "% Healthy", "% Compromised",
                        "% Dead", "Avg depth (m)", "Avg temp (°C)", "Health index"]
        show["Health index"] = (show["Health index"] * 100).round(0)
        st.dataframe(show.round(1).set_index("Rank"), use_container_width=True, height=320)


def _render_threats(df):
    stressor_names = list(STRESSOR_MAP.values())
    have = [n for n in stressor_names if f"has_{n.lower()}" in df.columns]
    if not have:
        return
    counts = {n: int(df[f"has_{n.lower()}"].sum()) for n in have}
    counts = {k: v for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True) if v > 0}
    if not counts:
        return
    _section("⚠️ Top Threats to the Reef",
             "What's stressing the coral, counted across all survey images.")
    total = len(df)
    top = max(counts.values())
    html = ["<div>"]
    for name, val in counts.items():
        pct = 100 * val / total if total else 0
        width = 100 * val / top if top else 0
        col = STRESSOR_COLORS.get(name, "#2980b9")
        html.append(
            f"<div style='margin-bottom:0.55rem;'>"
            f"<div style='display:flex;justify-content:space-between;font-size:14px;color:#e8f6f9;'>"
            f"<span style='font-weight:700;'>{name}</span>"
            f"<span style='color:#cfeef4;'>{val:,} images ({pct:.1f}%)</span></div>"
            f"<div style='background:rgba(255,255,255,0.12);border-radius:8px;height:12px;margin-top:3px;overflow:hidden;'>"
            f"<div style='width:{width:.1f}%;height:100%;background:{col};border-radius:8px;'></div></div></div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_samples(df, zf, zip_index):
    _section("🖼️ Sample Images (with their true labels)",
             "Real dataset images tagged with every ground-truth label — health and stressors.")
    if zf is None or "image_url" not in df.columns:
        st.info(f"ℹ️ Image archive not found or `image_url` column missing "
                f"(expected at `{ZIP_PATH}`). Sample gallery skipped.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        pick = st.radio("Show condition", ["All"] + HEALTH_ORDER, horizontal=True, key="ov_sample_cond")
    with c2:
        n_per = st.slider("Images per condition", 2, 6, 4, key="ov_sample_n")
    if "ov_seed" not in st.session_state:
        st.session_state.ov_seed = 42
    if st.button("🔀 Shuffle samples"):
        st.session_state.ov_seed += 1
        st.rerun()

    st.markdown(
        "<div style='font-size:12px;color:#95c6ce;margin:0.2rem 0 0.6rem 0;'>"
        "Pills under each image are the <b>true labels</b> from the dataset: "
        + " ".join(_pill(n, HEALTH_COLORS[n]) for n in HEALTH_ORDER)
        + " &nbsp;plus stressors like " + _pill("Disease", STRESSOR_COLORS["Disease"])
        + _pill("Predation", STRESSOR_COLORS["Predation"]) + "</div>",
        unsafe_allow_html=True,
    )

    conditions = HEALTH_ORDER if pick == "All" else [pick]
    for health_name in conditions:
        subset = df[df["primary_health"] == health_name]
        if subset.empty:
            continue
        st.markdown(f"**{HEALTH_ICON.get(health_name,'•')} {health_name}** "
                    f"— {len(subset):,} images in dataset")
        sample = subset.sample(min(n_per, len(subset)), random_state=st.session_state.ov_seed)
        cols = st.columns(len(sample))
        for col, (_, row) in zip(cols, sample.iterrows()):
            img = get_image_from_zip(zf, zip_index, row["image_url"])
            with col:
                if img is not None:
                    st.image(img, use_container_width=True)
                else:
                    st.markdown("<div style='height:120px;display:flex;align-items:center;"
                                "justify-content:center;background:rgba(255,255,255,0.05);"
                                "border-radius:10px;color:#95c6ce;font-size:12px;'>image not in archive</div>",
                                unsafe_allow_html=True)
                health, stressors = _true_labels(row)
                pills = "".join(_pill(h, HEALTH_COLORS[h]) for h in health)
                pills += "".join(_pill(s, STRESSOR_COLORS.get(s, "#2980b9")) for s in stressors)
                if not stressors:
                    pills += _pill("No stressors", "#7f8c8d")
                site = row.get("site", "N/A")
                st.markdown(
                    f"<div style='font-size:11px;color:#95c6ce;margin-top:2px;'>Site: {site}</div>"
                    f"<div style='margin-top:2px;'>{pills}</div>",
                    unsafe_allow_html=True,
                )


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def render_overview():
    _inject_css()

    # Header banner
    st.markdown(
        """
        <div class="ov-banner">
          <div style="font-family:monospace;font-size:0.75rem;letter-spacing:0.12em;color:#7fd0c2;
                      text-transform:uppercase;">Koh Tao Reef Watch · Overview Report</div>
          <div style="font-size:1.5rem;font-weight:800;color:#eafcff;margin-top:0.2rem;">
            🌊 Coral Reef Health at a Glance</div>
          <div style="font-size:0.9rem;color:#cfeef4;max-width:46rem;margin-top:0.3rem;">
            Summary of coral health across the Koh Tao marine sanctuary</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load data
    try:
        df = load_dataset(CSV_PATH)
    except Exception as e:
        st.warning(f"⚠️ Could not load the coral dataset from `{CSV_PATH}`: {e}")
        st.info("Please check the CSV file exists and contains the required columns.")
        return
    if df is None or len(df) == 0:
        st.warning("⚠️ The dataset loaded but is empty.")
        return

    zf, zip_index = (None, {})
    if os.path.exists(ZIP_PATH):
        try:
            zf, zip_index = load_image_zip(ZIP_PATH)
        except Exception:
            zf, zip_index = None, {}

    try:
        site_geo = build_site_geo(df)
    except Exception:
        site_geo = pd.DataFrame()

    # Derived numbers
    total = len(df)
    n_healthy = int(df["has_healthy"].sum())
    n_comp = int(df["has_compromised"].sum())
    n_dead = int(df["has_dead"].sum())
    pct_healthy = 100 * n_healthy / total
    pct_comp = 100 * n_comp / total
    pct_dead = 100 * n_dead / total
    nums = {
        "total": total, "n_sites": df["site"].nunique() if "site" in df.columns else 0,
        "n_healthy": n_healthy, "n_comp": n_comp, "n_dead": n_dead,
        "pct_healthy": pct_healthy, "pct_comp": pct_comp, "pct_dead": pct_dead,
        "mean_temp": df["temp"].mean() if "temp" in df.columns else float("nan"),
        "n_attention": n_comp + n_dead,
        "overall_score": (pct_healthy * 1.0 + pct_comp * 0.5) / 100,
    }

    _render_executive_summary(df, site_geo, nums)
    _render_metric_cards(nums)
    _render_overall_health(nums)
    if not site_geo.empty:
        _render_by_site(df, site_geo)
    _render_threats(df)
    _render_samples(df, zf, zip_index)

    # Call to action
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,rgba(7,55,66,0.35),rgba(14,165,233,0.14));
                    border:1px solid rgba(149,198,206,0.3);border-radius:20px;padding:1.3rem;
                    margin-top:1.8rem;text-align:center;">
          <div style="font-weight:800;color:#b4e2e9;font-size:1.1rem;">🔬 Have a coral photo to check?</div>
          <div style="font-size:0.9rem;color:#95c6ce;max-width:640px;margin:0.3rem auto 0;">
            Upload an underwater image and the AI will tell you whether the coral looks healthy,
            stressed, or dead — and show you where it looked.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    lc, cc, rc = st.columns([1, 2, 1])
    with cc:
        if st.button("🚀 Go to Image Uploader", key="upload_btn", use_container_width=True):
            st.session_state.active_tab = "prediction"
            st.rerun()