"""Visual theme for DSP Studio: dark navy lab-instrument look (Stitch redesign)."""
import plotly.graph_objects as go
import streamlit as st

# ---- palette (from the redesign's DESIGN.md) ----
BG, PANEL, MODULE, WELL, RULE = "#0B1120", "#0F172A", "#162032", "#1E293B", "#334155"
PRIMARY, PRIMARY_DARK = "#38BDF8", "#0284C7"
SECONDARY, WARN, CRIT = "#2DD4BF", "#F59E0B", "#F43F5E"
TEXT, BODY, MUTED = "#F8FAFC", "#94A3B8", "#64748B"
MONO = "JetBrains Mono, Consolas, monospace"
TRACE_COLORS = [PRIMARY, SECONDARY, WARN, CRIT]
SPECTROGRAM_SCALE = [[0.0, BG], [0.35, PRIMARY_DARK], [0.7, SECONDARY], [1.0, WARN]]

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"] { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
.stApp { background: #0B1120; }
[data-testid="stHeader"] { background: rgba(11, 17, 32, 0.85); }
footer, #MainMenu { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 1400px; }

[data-testid="stSidebar"] { background: #0F172A; border-right: 1px solid #1E293B; }
[data-testid="stSidebar"] h1 { font-size: 1.3rem; }

h1, h2, h3 { color: #F8FAFC; font-weight: 600; letter-spacing: -0.015em; }
h3 { font-size: 1.05rem; }
[data-testid="stCaptionContainer"], .stCaption { color: #64748B; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; }

/* metric cards */
[data-testid="stMetric"] { background: #162032; border: 1px solid #334155; border-radius: 4px; padding: 10px 14px; }
[data-testid="stMetricLabel"] p { color: #64748B; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.06em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #38BDF8; font-family: 'JetBrains Mono', monospace; font-weight: 600;
    font-variant-numeric: tabular-nums; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid #1E293B; }
.stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; letter-spacing: 0.03em;
    color: #94A3B8; border-radius: 4px 4px 0 0; padding: 8px 16px; }
.stTabs [aria-selected="true"] { color: #38BDF8; background: #1E293B; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #38BDF8; }

/* charts as instrument modules */
[data-testid="stPlotlyChart"] { background: #162032; border: 1px solid #334155; border-radius: 4px; padding: 4px;
    box-shadow: inset 0 0 12px rgba(56, 189, 248, 0.05); }

/* inputs and buttons */
[data-baseweb="select"] > div, [data-baseweb="input"] > div { background: #0B1120; border-color: #334155; border-radius: 4px; }
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSliderThumbValue"] { font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; }
.stButton > button, .stDownloadButton > button { background: #0284C7; color: #F8FAFC; border: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: 4px; font-weight: 600; }
.stButton > button:hover, .stDownloadButton > button:hover { background: #38BDF8; border-color: #38BDF8; color: #0B1120; }
[data-testid="stAlert"] { border-radius: 4px; }
[data-testid="stFileUploaderDropzone"] { background: #0B1120; border: 1px dashed #334155; border-radius: 4px; }

/* header banner */
.dsp-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
    padding: 12px 16px; margin-bottom: 14px; background: #0F172A; border: 1px solid #1E293B; border-radius: 4px; }
.dsp-title { font-size: 1.5rem; font-weight: 600; color: #F8FAFC; letter-spacing: -0.02em; }
.dsp-sub { margin-left: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.12em; color: #38BDF8; }
.dsp-chip { display: inline-block; margin-left: 8px; padding: 2px 8px; border-radius: 4px; background: #162032;
    border: 1px solid #334155; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #2DD4BF; }
"""


def apply_theme():
    """Inject fonts and component styling. Call once, right after set_page_config."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def header():
    st.markdown(
        '<div class="dsp-header"><div><span class="dsp-title">DSP Studio</span>'
        '<span class="dsp-sub">DIGITAL LAB SUITE</span></div>'
        '<div><span class="dsp-chip">DSP Mini Project</span>'
        '<span class="dsp-chip">ECE · LBS College of Engineering</span></div></div>',
        unsafe_allow_html=True)


def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def style_fig(fig, height=None):
    """Apply the dark instrument look to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor=MODULE, plot_bgcolor=BG,
        font=dict(family=MONO, size=11, color=BODY),
        colorway=TRACE_COLORS,
        legend=dict(orientation="h", y=1.14, font=dict(color=BODY)),
        hoverlabel=dict(bgcolor=WELL, bordercolor=PRIMARY, font=dict(family=MONO, color=TEXT)),
    )
    axis = dict(gridcolor=WELL, griddash="dash", zerolinecolor=RULE, linecolor=RULE,
                tickfont=dict(family=MONO, color=BODY), title_font=dict(family=MONO, color=MUTED))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if height:
        fig.update_layout(height=height)
    return fig


def show_chart(fig, **kwargs):
    """st.plotly_chart without Streamlit's own theme overriding ours."""
    kwargs.setdefault("width", "stretch")
    kwargs.setdefault("theme", None)
    st.plotly_chart(fig, **kwargs)
