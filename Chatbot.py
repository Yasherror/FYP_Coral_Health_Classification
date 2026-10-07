"""
Chatbot.py
"Reef Assistant" tab — a friendly, aesthetically-designed chat interface,
fronted by **Nong Tao** ("Little Turtle"), a Thai sea-turtle
mascot. Koh Tao literally means "Turtle Island" in Thai, so the mascot is
a nod to the island itself. Nong Tao answers questions about the coral
dataset and the user's most recent prediction result.

"""
# ============================================================================
# Required Imports for Overview Functionality
# ============================================================================
# Core Libraries
import streamlit as st                         # Web UI framework
import pandas as pd                            # Data processing
import google.generativeai as genai            # Gemini AI integration
from data_utils import load_dataset, build_site_geo, CSV_PATH  # Local helpers

# --------------------------------------------------------------------------- #
# Gemini credentials 
# --------------------------------------------------------------------------- #
# GEMINI_API_KEY = "Add you own Gemini API key"

# Preferred models, best-first. Do NOT hard-code a single name — 
# key actually supports and uses the first that works.
GEMINI_MODEL_PREFERENCES = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.5-pro",
    "gemini-1.5-flash", 
]

# --------------------------------------------------------------------------- #
# DATA PATHS
#   Main cleaned dataset comes from data_utils.CSV_PATH.
#   Survey metadata is the second file the user pointed at.
# --------------------------------------------------------------------------- #
METADATA_CSV_PATH = r"C:\Users\Yashreen\Downloads\coral-health-classification\src\components\data\surveys_metadata.csv"

MASCOT_NAME = "Nong Tao"
MASCOT_EMOJI = "🐢"

# Example questions shown as clickable chips (Koh Tao coral reef focused).
SAMPLE_PROMPTS = [
    "Which survey site has the healthiest coral?",
    "Which sites have the most dead coral?",
    "How does water temperature relate to coral death here?",
    "Give me a plain-English summary of Koh Tao's reef health.",
    "Which site should conservation efforts prioritise, and why?",
    "What does my most recent image prediction mean?",
]
import base64

# --------------------------------------------------------------------------- #
# Nong Tao mascot — external SVG file, embedded unchanged.
# --------------------------------------------------------------------------- #
MASCOT_SVG_PATH = r"C:\Users\Yashreen\Downloads\coral-health-classification\src\components\data\turtle_mascot_chatbot.svg"


@st.cache_data(show_spinner=False)
def _mascot_svg_data_uri(path=MASCOT_SVG_PATH):
    """Read the SVG once and return it as a base64 data URI (cached)."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _mascot_svg(width=200):
    """
    Return an <img> tag showing the mascot SVG *unchanged* at the given width.
    The size is applied on the <img>, so the SVG file itself is never edited.
    Falls back to the 🐢 emoji if the file can't be read.
    """
    try:
        uri = _mascot_svg_data_uri()
    except Exception:
        return f"<span style='font-size:{width}px;line-height:1;'>🐢</span>"
    return (f"<img src='{uri}' width='{width}' alt='Nong Tao mascot' "
            f"style='display:block;' />")

# --------------------------------------------------------------------------- #
# What the columns / codes actually MEAN — so Gemini reads the data correctly.
# --------------------------------------------------------------------------- #
DATASET_SCHEMA_NOTE = """DATA DICTIONARY — how to read the numbers below:
- The project surveys coral around Koh Tao (Gulf of Thailand). Each row of the
  main dataset (coral_dataset_cleaned.csv) is ONE coral survey image.
- Coral health label IDs: 1 = Healthy, 2 = Compromised (stressed/bleaching but
  alive), 3 = Dead. "primary_health" is the worst condition present in an image.
- Stressor label IDs: 4 = Rubble, 5 = Competition, 6 = Disease, 7 = Predation,
  8 = Physical damage. An image can have several stressors, or none.
- Per-SITE columns in the summary table:
    total          = number of survey images at that site
    pct_healthy / pct_compromised / pct_dead = % of that site's images in each
                     health condition (these three add up to ~100%)
    mean_depth     = average water depth in METRES
    mean_temp      = average water temperature in °C
    health_score   = (pct_healthy*1.0 + pct_compromised*0.5) / 100, range 0-1;
                     higher = healthier reef. Above ~0.6 is good, below ~0.4 poor.
- There are a handful of named survey sites around the island.
- The second table (survey metadata) holds per-survey logistics/context; use it
  for questions about the surveys themselves. Join it to the main data on any
  shared key (e.g. a site or survey id) if one is present."""


SYSTEM_PROMPT_TEMPLATE = """You are "Nong Tao" (Little Turtle), a warm, friendly Thai sea-turtle \
mascot and marine-biology assistant embedded in a coral reef health monitoring dashboard for \
Koh Tao, Thailand ("Turtle Island").

Your job is to help users understand THIS dashboard's coral reef data and their most recent \
AI image prediction. Follow these rules:
- Answer using ONLY the data below (the schema note, the per-site summary, the survey metadata, \
and any prediction context). Interpret the numbers using the data dictionary.
- If asked something the data doesn't cover, say so honestly rather than guessing at numbers.
- Politely steer clearly off-topic questions back to Koh Tao coral reefs / this dashboard.
- Keep answers clear and friendly for a non-technical audience. Use plain language, short \
paragraphs, and cite the specific site names and numbers from the data when relevant.
- You may use an occasional warm touch (e.g. a friendly "Sawadeekap!") but stay concise and factual.

{schema_note}

=== MAIN DATASET — per-survey-site summary (coral_dataset_cleaned.csv) ===
{site_summary}

=== SURVEY METADATA (surveys_metadata.csv) ===
{metadata_summary}

=== MOST RECENT UPLOADED-IMAGE PREDICTION (if any) ===
{prediction_summary}
"""


@st.cache_data(show_spinner=False)
def _load_site_summary(csv_path=CSV_PATH):
    """Per-site summary of the MAIN cleaned dataset. Cached so it's cheap per turn."""
    try:
        df = load_dataset(csv_path)
        site_geo = build_site_geo(df)
        return site_geo[[
            "site_fullname", "total", "pct_healthy", "pct_compromised",
            "pct_dead", "mean_depth", "mean_temp", "health_score",
        ]].round(1).to_string(index=False)
    except Exception as e:
        return f"(main dataset unavailable: {e})"


@st.cache_data(show_spinner=False)
def _load_metadata_summary(path=METADATA_CSV_PATH, max_rows=60, char_cap=6000):
    """
    Read surveys_metadata.csv and turn it into a compact text block Gemini can
    reason over: shape, column names, numeric summary, and the table itself
    (whole thing if small, otherwise the first `max_rows`). Robust to the file
    being missing — returns a note instead of crashing the chat.
    """
    try:
        md = pd.read_csv(path)
    except Exception as e:
        return (f"(survey metadata not loaded from {path}: {e}. "
                f"If the path is wrong, update METADATA_CSV_PATH at the top of Chatbot.py.)")

    parts = [f"{len(md)} rows x {len(md.columns)} columns.",
             "Columns: " + ", ".join(map(str, md.columns))]

    num = md.select_dtypes("number")
    if not num.empty:
        parts.append("Numeric column summary:\n" + num.describe().round(2).to_string())

    if len(md) <= max_rows:
        parts.append("Full metadata table:\n" + md.to_string(index=False))
    else:
        parts.append(f"First {max_rows} of {len(md)} rows:\n"
                     + md.head(max_rows).to_string(index=False))

    text = "\n\n".join(parts)
    if len(text) > char_cap:
        text = text[:char_cap] + "\n... (metadata truncated to keep the prompt compact)"
    return text


def _build_chat_context():
    """Assemble the full grounding prompt: schema + main data + metadata + prediction."""
    site_summary = _load_site_summary()
    metadata_summary = _load_metadata_summary()

    active_analysis = st.session_state.get("active_analysis")
    if active_analysis:
        r = active_analysis
        prediction_summary = (
            f"Primary health: {r['primary_health']} "
            f"({r['primary_health_conf'] * 100:.1f}% confidence)\n"
            f"Predicted labels: {', '.join(r['predicted_labels']) or 'none above threshold'}"
        )
    else:
        prediction_summary = "No image has been analyzed yet in this session."

    return SYSTEM_PROMPT_TEMPLATE.format(
        schema_note=DATASET_SCHEMA_NOTE,
        site_summary=site_summary,
        metadata_summary=metadata_summary,
        prediction_summary=prediction_summary,
    )


def _inject_css():
    st.markdown("""
    <style>
      .nt-header {
        display:flex; align-items:center; gap:1.25rem;
        background:linear-gradient(135deg,#0d5566 0%,#0a3d49 100%);
        border:1px solid rgba(149,198,206,0.25);
        border-radius:22px; padding:1.2rem 1.5rem; margin-bottom:1rem;
        box-shadow:0 8px 24px rgba(0,0,0,0.25);
      }
      .nt-bubble {
        position:relative; background:rgba(255,255,255,0.10);
        border:1px solid rgba(180,226,233,0.25);
        border-radius:16px; padding:0.9rem 1.15rem; color:#eafcff; flex:1;
      }
      .nt-bubble:before{
        content:""; position:absolute; left:-10px; top:28px;
        border-top:8px solid transparent; border-bottom:8px solid transparent;
        border-right:10px solid rgba(255,255,255,0.10);
      }
      .nt-name{ font-size:20px; font-weight:800; color:#b4e2e9; }
      .nt-tag{ font-size:12px; letter-spacing:1px; color:#7fd0c2; text-transform:uppercase; }
      /* example-question chips */
      div[data-testid="stExpander"] .stButton>button{
        background:rgba(255,255,255,0.06);
        border:1px solid rgba(180,226,233,0.30);
        border-radius:14px; color:#dff6f9; font-weight:500;
        text-align:left; white-space:normal; line-height:1.3; padding:0.6rem 0.9rem;
      }
      div[data-testid="stExpander"] .stButton>button:hover{
        background:rgba(79,212,176,0.18); border-color:#4fd4b0;
      }
    </style>
    """, unsafe_allow_html=True)


def _render_header():
    greeting = (
        f"<div class='nt-tag'>Reef Assistant • Koh Tao 🇹🇭</div>"
        f"<div class='nt-name'>Sawadeekap! 🙏 I'm {MASCOT_NAME}, your reef guide.</div>"
        f"<div style='margin-top:0.35rem;font-size:14px;color:#cfeef4;'>"
        f"Ask me about the survey sites, coral health trends, or your most recent image scan. "
        f"I answer straight from this dashboard's live data — so I won't make numbers up.</div>"
    )
    st.markdown(
        f"<div class='nt-header'>{_mascot_svg(170)}<div class='nt-bubble'>{greeting}</div></div>",
        unsafe_allow_html=True,
    )


def _render_sample_prompts():
    has_history = bool(st.session_state.get("chat_history"))
    with st.expander("💡 Example questions to ask about Koh Tao's reefs",
                     expanded=not has_history):
        cols = st.columns(2)
        for i, q in enumerate(SAMPLE_PROMPTS):
            if cols[i % 2].button(q, key=f"nt_sample_{i}", use_container_width=True):
                st.session_state.pending_prompt = q
                st.rerun()


def _model_candidates():
    """
    Model names to try, best-first. A model that already worked this session
    goes first; then the preferred models the key actually exposes (via
    list_models); then the raw preference list as a fallback if listing fails.
    """
    candidates = []
    winner = st.session_state.get("gemini_model_name")
    if winner:
        candidates.append(winner)

    available = set()
    try:
        available = {
            m.name.split("/")[-1]
            for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        }
    except Exception:
        pass  # offline / older SDK — fall back to the raw preference list

    if available:
        for name in GEMINI_MODEL_PREFERENCES:
            if name in available and name not in candidates:
                candidates.append(name)
        for name in sorted(available):  # any other flash model the key has
            if "flash" in name and name not in candidates:
                candidates.append(name)
    else:
        for name in GEMINI_MODEL_PREFERENCES:
            if name not in candidates:
                candidates.append(name)

    return candidates or list(GEMINI_MODEL_PREFERENCES)


def _process_message(prompt):
    """Append the user turn, call Gemini (auto-picking a working model), append reply."""
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Rebuilt every turn so Nong Tao always sees the CURRENT dataset + prediction.
    system_instruction = _build_chat_context()

    # Prior turns as Gemini history (assistant -> "model"), excluding the latest
    # user message, which we send() to get the reply.
    history = [
        {"role": "model" if m["role"] == "assistant" else "user",
         "parts": [m["content"]]}
        for m in st.session_state.chat_history[:-1]
    ]
    user_msg = st.session_state.chat_history[-1]["content"]

    reply, last_err = None, None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        with st.spinner(f"{MASCOT_EMOJI} {MASCOT_NAME} is looking through the reef data..."):
            for name in _model_candidates():
                try:
                    model = genai.GenerativeModel(name, system_instruction=system_instruction)
                    chat = model.start_chat(history=history)
                    response = chat.send_message(user_msg)
                    reply = response.text
                    st.session_state.gemini_model_name = name  # remember the winner
                    break
                except Exception as e:
                    last_err = e
                    continue  # 404 / unsupported model → try the next candidate
    except Exception as e:
        last_err = e

    if reply is None:
        reply = (f"⚠️ Chatbot error: {last_err}\n\n"
                 f"If this mentions an invalid API key, regenerate one at "
                 f"https://aistudio.google.com/app/apikey and update GEMINI_API_KEY "
                 f"at the top of Chatbot.py. If it mentions models/quota, none of the "
                 f"tried models are available on your free tier — check "
                 f"https://aistudio.google.com/ for current model names.")

    st.session_state.chat_history.append({"role": "assistant", "content": reply})


def render_chatbot():
    """Main chatbot render function. No API key prompt — Gemini is baked in."""
    _inject_css()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    _render_header()

    _render_sample_prompts()

    # Render the running conversation with Nong Tao as the assistant avatar.
    for msg in st.session_state.chat_history:
        avatar = MASCOT_EMOJI if msg["role"] == "assistant" else "🧑‍🔬"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # A typed message OR a clicked example chip both feed the same handler.
    typed = st.chat_input(f"Ask {MASCOT_NAME} about the reef data...")
    pending = st.session_state.pop("pending_prompt", None)
    prompt = typed or pending

    if prompt:
        _process_message(prompt)
        st.rerun()

    if st.session_state.chat_history:
        if st.button(" Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()
