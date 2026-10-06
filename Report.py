"""
Report.py
"Export Report" tab — turns the dashboard's data into PDF reports
"""

import io              # I/O operations
import os          # File system ops
import base64     # Base64 encoding/decoding
import tempfile    # Temporary file handling
from datetime import datetime    # Timestamps

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import (
    load_dataset, build_site_geo, CSV_PATH,
    HEALTH_MAP, STRESSOR_MAP, HEALTH_COLORS, STRESSOR_COLORS,
)

HEALTH_ORDER = ["Healthy", "Compromised", "Dead"]
TEAL = (13, 85, 102)
ACCENT = (14, 165, 233)
INK = (31, 59, 65)
MUTED = (110, 124, 130)


# --------------------------------------------------------------------------- #
# Shared computation (mirrors the Overview report)
# --------------------------------------------------------------------------- #
def _compute_nums(df):
    total = len(df)
    n_healthy = int(df["has_healthy"].sum())
    n_comp = int(df["has_compromised"].sum())
    n_dead = int(df["has_dead"].sum())
    pct_h = 100 * n_healthy / total if total else 0
    pct_c = 100 * n_comp / total if total else 0
    pct_d = 100 * n_dead / total if total else 0
    return {
        "total": total, "n_sites": df["site"].nunique() if "site" in df.columns else 0,
        "n_healthy": n_healthy, "n_comp": n_comp, "n_dead": n_dead,
        "pct_healthy": pct_h, "pct_comp": pct_c, "pct_dead": pct_d,
        "mean_temp": df["temp"].mean() if "temp" in df.columns else float("nan"),
        "mean_depth": df["depth"].mean() if "depth" in df.columns else float("nan"),
        "n_attention": n_comp + n_dead,
        "pct_attention": (100 * (n_comp + n_dead) / total) if total else 0,
    }


def _threat_counts(df):
    out = {}
    for name in STRESSOR_MAP.values():
        col = f"has_{name.lower()}"
        if col in df.columns:
            v = int(df[col].sum())
            if v > 0:
                out[name] = v
    return dict(sorted(out.items(), key=lambda x: x[1], reverse=True))


def _narrative(nums, site_geo):
    parts = [
        f"This brief summarises coral-reef health across the Koh Tao marine sanctuary, "
        f"based on {nums['total']:,} survey images collected across {nums['n_sites']} sites. "
        f"Overall, {nums['pct_healthy']:.0f}% of the coral recorded is healthy, "
        f"{nums['pct_comp']:.0f}% is stressed (compromised), and {nums['pct_dead']:.0f}% is dead."
    ]
    if len(site_geo) >= 1:
        b = site_geo.iloc[0]
        parts.append(f"The healthiest reef surveyed is {b['site_fullname']} "
                     f"({b['pct_healthy']:.0f}% healthy).")
    if len(site_geo) >= 2:
        w = site_geo.iloc[-1]
        parts.append(f"The most at-risk reef is {w['site_fullname']}, where "
                     f"{w['pct_dead']:.0f}% of coral is dead and immediate attention is advised.")
    if pd.notna(nums["mean_temp"]):
        parts.append(f"Average water temperature across surveys is {nums['mean_temp']:.1f} degC; "
                     f"sustained warming raises the risk of further bleaching.")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Chart images (matplotlib -> temp PNG paths) for the PDF
# --------------------------------------------------------------------------- #
def _save_fig(fig):
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return tmp.name


def _chart_health_split(nums):
    """
    Horizontal bar chart showing each health category on its own row, with
    the percentage + image count always labelled just outside the bar's
    end. Unlike a stacked bar, every category gets a guaranteed-visible
    label regardless of how small its share is (e.g. a small "Dead" slice
    no longer disappears).
    """
    vals = [nums["pct_healthy"], nums["pct_comp"], nums["pct_dead"]]
    counts = [nums["n_healthy"], nums["n_comp"], nums["n_dead"]]

    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    y_pos = list(range(len(HEALTH_ORDER)))
    ax.barh(
        y_pos, vals,
        color=[HEALTH_COLORS[n] for n in HEALTH_ORDER],
        edgecolor="white", linewidth=1.5, height=0.55,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(HEALTH_ORDER, fontsize=10.5, fontweight="bold", color="#2c3e50")
    ax.invert_yaxis()  # Healthy on top, Dead on bottom

    max_v = max(vals) if max(vals) > 0 else 1
    ax.set_xlim(0, max_v * 1.35 + 4)

    for i, (v, c) in enumerate(zip(vals, counts)):
        ax.text(
            v + max_v * 0.03 + 0.6, i,
            f"{v:.1f}%  ({c:,} images)",
            va="center", ha="left",
            fontsize=10, fontweight="bold", color="#2c3e50",
        )

    ax.set_xlabel("Share of images (%)", fontsize=9, color="#555555")
    ax.set_title(
        "Overall Coral Health Distribution",
        fontsize=13,
        fontweight="bold",
        pad=16,
          color="#1a1a2e",
          loc="left",
          )
    total_images = nums.get("total", 0)

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, color="#cccccc")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8, colors="#555555")
    ax.tick_params(axis="y", length=0)

    plt.tight_layout()
    return _save_fig(fig)


def _chart_threats(threats):
    if not threats:
        return None
    fig, ax = plt.subplots(figsize=(6.6, max(1.6, 0.42 * len(threats))))
    names = list(threats.keys())[::-1]
    vals = [threats[n] for n in names]
    colors = [STRESSOR_COLORS.get(n, "#2980b9") for n in names]
    ax.barh(names, vals, color=colors, edgecolor="white")
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v:,}", va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Images affected", fontsize=9)
    ax.set_title("Top threats to the reef", fontsize=10, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return _save_fig(fig)


# --------------------------------------------------------------------------- #
# PDF builder (fpdf2)
# --------------------------------------------------------------------------- #
_REPL = {"–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
         "≥": ">=", "≤": "<=", "•": "-", "→": "->", "·": "-", "×": "x"}


def _san(s):
    s = str(s)
    for k, v in _REPL.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "ignore").decode("latin-1")


def build_pdf(meta, nums, site_geo, threats, chart_paths):
    from fpdf import FPDF

    class Brief(FPDF):
        def header(self):
            # No running header band — keeps continuation pages clean under
            # full-width charts. Page context lives in the footer instead.
            return

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(*MUTED)
            self.cell(0, 5, _san(f"Generated {meta['date']}  -  Koh Tao Reef Watch  -  "
                                 f"Model: EfficientNet-B3 (tuned)"))
            self.set_y(-12)
            self.cell(0, 5, _san(f"Page {self.page_no()}"), align="R")

    pdf = Brief()
    pdf.set_auto_page_break(True, margin=16)
    pdf.set_margins(12, 12, 12)
    pdf.add_page()
    M = 12
    usable = pdf.w - 2 * M

    # --- Title block ---
    pdf.set_fill_color(*TEAL)
    pdf.rect(0, 0, pdf.w, 34, "F")
    pdf.set_xy(M, 8)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 9, _san(meta["title"]))
    pdf.set_xy(M, 18)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _san(meta["subtitle"]))
    pdf.set_xy(M, 24)
    pdf.set_font("Helvetica", "I", 8)
    prepared = f"Prepared by: {meta['author'] or 'N/A'}"
    if meta.get("org"):
        prepared += f"  |  {meta['org']}"
    pdf.cell(0, 6, _san(f"{prepared}   |   Date: {meta['date']}"))
    pdf.ln(28)
    pdf.set_text_color(*INK)

    # --- Executive summary ---
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*TEAL)
    pdf.cell(0, 6, "Executive Summary")
    pdf.ln(7)
    pdf.set_text_color(*INK)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(usable, 4.6, _san(_narrative(nums, site_geo)))
    pdf.ln(3)

    # --- Key figures table ---
    def section(title):
        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*TEAL)
        pdf.cell(0, 7, _san(title))
        pdf.ln(8)
        pdf.set_text_color(*INK)

    def kv_table(rows, col_w):
        pdf.set_font("Helvetica", "", 9)
        for label, value in rows:
            pdf.set_x(M)
            pdf.set_fill_color(240, 246, 248)
            pdf.cell(col_w[0], 7, _san(label), border=1, fill=True)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(col_w[1], 7, _san(value), border=1)
            pdf.set_font("Helvetica", "", 9)
            pdf.ln(7)

    section("Key Figures")
    temp_txt = f"{nums['mean_temp']:.1f} degC" if pd.notna(nums["mean_temp"]) else "N/A"
    depth_txt = f"{nums['mean_depth']:.1f} m" if pd.notna(nums["mean_depth"]) else "N/A"
    kv_table([
        ("Survey sites monitored", f"{nums['n_sites']}"),
        ("Coral images analysed", f"{nums['total']:,}"),
        ("Healthy coral", f"{nums['pct_healthy']:.1f}%  ({nums['n_healthy']:,} images)"),
        ("Compromised (stressed) coral", f"{nums['pct_comp']:.1f}%  ({nums['n_comp']:,} images)"),
        ("Dead coral", f"{nums['pct_dead']:.1f}%  ({nums['n_dead']:,} images)"),
        ("Coral needing attention", f"{nums['n_attention']:,} images"),
        ("Average water temperature", temp_txt),
        ("Average survey depth", depth_txt),
    ], [95, 85])

    # --- Overall health chart ---
    if chart_paths.get("split"):
        pdf.ln(2)
        pdf.image(chart_paths["split"], x=M, w=usable)

    # --- Reef-by-reef table ---
    if not site_geo.empty:
        section("Reef-by-Reef Summary")
        headers = ["Rank", "Site", "Images", "%Healthy", "%Compr.", "%Dead"]
        widths = [12, 74, 26, 26, 26, 24]
        pdf.set_x(M)
        pdf.set_fill_color(*TEAL)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)
        for h, w in zip(headers, widths):
            pdf.cell(w, 7, _san(h), border=1, align="C", fill=True)
        pdf.ln(7)
        pdf.set_text_color(*INK)
        pdf.set_font("Helvetica", "", 8)
        for i, (_, r) in enumerate(site_geo.iterrows()):
            if pdf.get_y() > pdf.h - 24:
                pdf.add_page()
            pdf.set_x(M)
            fill = (245, 250, 251) if i % 2 else (255, 255, 255)
            pdf.set_fill_color(*fill)
            cells = [str(int(r["rank"])), _san(str(r["site_fullname"]))[:42],
                     f"{int(r['total']):,}", f"{r['pct_healthy']:.0f}%",
                     f"{r['pct_compromised']:.0f}%", f"{r['pct_dead']:.0f}%"]
            aligns = ["C", "L", "R", "R", "R", "R"]
            for c, w, a in zip(cells, widths, aligns):
                pdf.cell(w, 6.5, _san(c), border=1, align=a, fill=True)
            pdf.ln(6.5)
        pdf.ln(2)

    # --- Threats ---
    if threats:
        section("Top Threats to the Reef")
        if chart_paths.get("threats"):
            pdf.image(chart_paths["threats"], x=M, w=usable)

    # --- Methodology / notes ---
    section("About This Brief")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.multi_cell(usable, 4.6, _san(
        "Health condition (Healthy / Compromised / Dead) is derived from expert survey labels in "
        "the cleaned dataset. The dashboard's image classifier is a fine-tuned EfficientNet-B3 "
        "multi-label model. This brief is generated automatically from the live dashboard data "
        "for reporting and planning purposes; figures reflect the dataset loaded at generation "
        "time."))
    return bytes(pdf.output())


# --------------------------------------------------------------------------- #
# PDF preview (inline, before download)
# --------------------------------------------------------------------------- #
def _render_pdf_preview(pdf_bytes, height=680):
    """Embed the generated PDF inline via a base64 data URI so the user can
    review the brief before deciding to download it."""
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    st.markdown("#### Preview PDF")
    st.markdown(
        f"""
        <iframe
            src="data:application/pdf;base64,{b64}"
            width="100%" height="{height}"
            style="border:1px solid #cfd8dc;border-radius:12px;"
            type="application/pdf">
        </iframe>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def _intro():
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0d5566 0%,#0a3d49 100%);
                    border:1px solid rgba(149,198,206,0.25);border-radius:22px;
                    padding:1.3rem 1.6rem;margin-bottom:1rem;box-shadow:0 8px 24px rgba(0,0,0,0.25);">
          <div style="font-family:monospace;font-size:0.72rem;letter-spacing:0.12em;color:#7fd0c2;
                      text-transform:uppercase;">Export &amp; Reporting</div>
          <div style="font-size:1.4rem;font-weight:800;color:#eafcff;margin-top:0.2rem;">
            📤 Take the data with you</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='display:flex;gap:0.6rem;flex-wrap:wrap;margin-bottom:0.4rem;'>"
        + "".join(
            f"<span style='background:rgba(255,255,255,0.06);border:1px solid rgba(180,226,233,0.25);"
            f"border-radius:20px;padding:0.25rem 0.7rem;font-size:0.75rem;color:#dff6f9;'>{who}</span>"
            for who in ["🔬 Marine researchers", "🌱 Conservation organizations",
                        "🎓 Academic researchers", "🏛️ Policy makers"]
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def render_report():
    _intro()

    try:
        df = load_dataset(CSV_PATH)
    except Exception as e:
        st.warning(f"⚠️ Could not load the dataset from `{CSV_PATH}`: {e}")
        return
    if df is None or len(df) == 0:
        st.warning("⚠️ The dataset is empty — nothing to export.")
        return

    try:
        site_geo_all = build_site_geo(df)
    except Exception:
        site_geo_all = pd.DataFrame()

    # ---- Step 1: scope ---------------------------------------------------- #
    st.markdown("### 1. Choose what to include")
    scope_opts = ["All sites"]
    if "site" in df.columns:
        scope_opts += sorted(df["site"].dropna().unique().tolist())
    scope = st.selectbox("Report scope", scope_opts,
                         help="Export everything, or focus the report on a single reef site.")
    if scope == "All sites":
        scope_df = df
        site_geo = site_geo_all
        scope_label = "All sites"
    else:
        scope_df = df[df["site"] == scope]
        site_geo = site_geo_all[site_geo_all["site"] == scope] if not site_geo_all.empty else site_geo_all
        fullname = (site_geo["site_fullname"].iloc[0]
                    if not site_geo.empty and "site_fullname" in site_geo else scope)
        scope_label = f"Site: {fullname}"

    nums = _compute_nums(scope_df)
    threats = _threat_counts(scope_df)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Sites", nums["n_sites"])
    g2.metric("Images", f"{nums['total']:,}")
    g3.metric("Healthy", f"{nums['pct_healthy']:.0f}%")
    g4.metric("Needs attention", f"{nums['pct_attention']:.0f}%")

    st.divider()

    # ---- Step 2: CSV ------------------------------------------------------ #
    st.markdown("### 2. Download the raw data (CSV)")
    st.caption("For researchers who want to run their own statistics. Opens in Excel, "
               "Python, R — anything.")
    stamp = datetime.now().strftime("%Y%m%d")
    tag = "all" if scope == "All sites" else str(scope)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            " Download Full dataset (per-image)",
            data=scope_df.to_csv(index=False).encode("utf-8"),
            file_name=f"coral_dataset_{tag}_{stamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"{len(scope_df):,} rows · every survey image + labels.")
    with c2:
        summary_csv = (site_geo.to_csv(index=False).encode("utf-8")
                       if not site_geo.empty else b"")
        st.download_button(
            "Download Per-site summary",
            data=summary_csv,
            file_name=f"coral_site_summary_{tag}_{stamp}.csv",
            mime="text/csv",
            disabled=site_geo.empty,
            use_container_width=True,
        )
        st.caption(f"{len(site_geo)} site rows · health %, depth, temp.")

    with st.expander("Preview per-site summary"):
        if not site_geo.empty:
            st.dataframe(site_geo.round(2), use_container_width=True, height=260)
        else:
            st.info("No per-site summary available for this scope.")

    st.divider()

    # ---- Step 3: PDF ------------------------------------------------------ #
    st.markdown("### 3. Generate a shareable PDF brief")
    st.caption("A professional, ready-to-send summary for stakeholders — executive summary, "
               "key figures, reef-by-reef table, and top threats, with charts.")

    with st.container():
        pc1, pc2 = st.columns(2)
        author = pc1.text_input("Prepared by (name)", value="",
                                placeholder="e.g. Dr. A. Marine")
        org = pc2.text_input("Organisation (optional)", value="",
                             placeholder="e.g. Koh Tao Reef Watch")
        title = st.text_input("Report title",
                              value="Coral Reef Health Brief — Koh Tao Marine Sanctuary")

    # Is fpdf available?
    try:
        import fpdf  # noqa: F401
        have_fpdf = True
    except Exception:
        have_fpdf = False

    if not have_fpdf:
        st.warning(" PDF export needs the **fpdf2** library. Install it once with "
                   "`pip install fpdf2`, then restart the app. (CSV export above works without it.)")
        return

    gen = st.button("Generate PDF brief", type="primary", use_container_width=True)
    if gen:
        try:
            with st.spinner("Building your PDF brief…"):
                chart_paths = {
                    "split": _chart_health_split(nums),
                    "threats": _chart_threats(threats),
                }
                meta = {
                    "title": title.strip() or "Coral Reef Health Brief",
                    "subtitle": f"{scope_label}  ·  Koh Tao Marine Sanctuary, Gulf of Thailand",
                    "author": author.strip(),
                    "org": org.strip(),
                    "date": datetime.now().strftime("%d %B %Y"),
                }
                pdf_bytes = build_pdf(meta, nums, site_geo, threats, chart_paths)
                for p in chart_paths.values():
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
            st.session_state.report_pdf = pdf_bytes
            st.session_state.report_pdf_name = f"coral_health_brief_{tag}_{stamp}.pdf"
            st.success("PDF Ready — preview it below, then download.")
        except Exception as e:
            st.error(f"Could not generate the PDF: {e}")

    if st.session_state.get("report_pdf"):
        _render_pdf_preview(st.session_state.report_pdf)
        st.download_button(
            "Download PDF brief",
            data=st.session_state.report_pdf,
            file_name=st.session_state.get("report_pdf_name", "coral_health_brief.pdf"),
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
        st.caption("Tip: re-generate any time after changing the scope or details above.")