"""
DSP Studio — Part 1: Signal Input & Frequency Analysis

Run with:
    streamlit run app.py
"""

import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from signals.generator import GENERATORS, generate
from signals.analysis import compute_fft, dominant_frequency

st.set_page_config(page_title="DSP Studio", layout="wide")

# ---------- Sidebar: signal source ----------
st.sidebar.title("DSP Studio")
st.sidebar.caption("Part 1 · Signal Input & Frequency Analysis")

source = st.sidebar.radio("Signal source", ["Generate", "Upload"])

sample_rate = 8000  # Hz, fixed for consistency between generated and uploaded signals

if source == "Generate":
    signal_type = st.sidebar.selectbox("Waveform", list(GENERATORS.keys()))
    freq = st.sidebar.slider("Frequency (Hz)", 1, 500, 50)
    amplitude = st.sidebar.slider("Amplitude", 0.1, 5.0, 1.0)
    duration = st.sidebar.slider("Duration (s)", 0.1, 2.0, 1.0)
    t, y = generate(signal_type, freq, amplitude, duration, sample_rate)
    st.sidebar.caption(f"Sample rate: {sample_rate} Hz")

else:
    uploaded = st.sidebar.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded is not None:
        import scipy.io.wavfile as wavfile

        sr, data = wavfile.read(io.BytesIO(uploaded.read()))
        if data.ndim > 1:
            data = data.mean(axis=1)  # collapse stereo to mono
        y = data.astype(np.float64)
        y /= np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
        sample_rate = sr
        t = np.linspace(0, len(y) / sr, len(y), endpoint=False)
    else:
        st.info("Upload a WAV file from the sidebar, or switch to 'Generate' to try a test signal.")
        st.stop()

# ---------- Main area ----------
st.title("Signal Input & Frequency Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Time-Domain Waveform")
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color="#2E74B5")))
    fig_time.update_layout(
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        margin=dict(l=40, r=20, t=20, b=40), height=380,
    )
    st.plotly_chart(fig_time, use_container_width=True)

with col2:
    st.subheader("Frequency Spectrum (FFT)")
    freqs, magnitude = compute_fft(y, sample_rate)
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(x=freqs, y=magnitude, mode="lines", line=dict(color="#1F3864")))
    fig_freq.update_layout(
        xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",
        margin=dict(l=40, r=20, t=20, b=40), height=380,
    )
    st.plotly_chart(fig_freq, use_container_width=True)

peak_freq = dominant_frequency(y, sample_rate)
st.metric("Dominant frequency detected", f"{peak_freq:.1f} Hz")
