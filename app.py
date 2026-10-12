"""
DSP Studio — Parts 1 & 2: Signal Input, Frequency Analysis, Filtering, Voice

Run with:
    streamlit run app.py
"""

import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from signals.generator import GENERATORS, generate
from signals.analysis import compute_fft, dominant_frequency
from signals.filters import apply_filter, frequency_response
from signals.voice import load_wav, wav_bytes, telephone_effect, reduce_noise, snr_db
from signals.spectrogram import compute_spectrogram
from scipy.signal import chirp

st.set_page_config(page_title="DSP Studio", layout="wide")

BLUE, NAVY, ORANGE = "#2E74B5", "#1F3864", "#E8833A"


def line_fig(series, xtitle, ytitle, height=340):
    """series: list of (name, x, y, color). One Plotly line chart."""
    fig = go.Figure()
    for name, x, y_, color in series:
        fig.add_trace(go.Scatter(x=x, y=y_, mode="lines", name=name,
                                 line=dict(color=color)))
    fig.update_layout(xaxis_title=xtitle, yaxis_title=ytitle, height=height,
                      margin=dict(l=40, r=20, t=20, b=40),
                      legend=dict(orientation="h", y=1.12))
    return fig


def spec_fig(f, t, s_db, fmax=None, height=360):
    """Spectrogram heatmap. Strides the data so huge clips stay responsive."""
    if fmax:
        keep = f <= fmax
        f, s_db = f[keep], s_db[keep]
    ft = max(1, len(t) // 600)
    ff = max(1, len(f) // 400)
    fig = go.Figure(go.Heatmap(x=t[::ft], y=f[::ff], z=s_db[::ff, ::ft],
                               colorscale="Viridis",
                               colorbar=dict(title="dB")))
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
                      height=height, margin=dict(l=40, r=20, t=20, b=40))
    return fig


# ---------- Sidebar: signal source ----------
st.sidebar.title("DSP Studio")
st.sidebar.caption("Analysis · Filtering · Spectrogram · Voice")

source = st.sidebar.radio("Signal source", ["Generate", "Upload"])

sample_rate = 8000  # Hz, fixed for consistency between generated and uploaded signals
t = y = None

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

# ---------- Main area ----------
st.title("DSP Studio")
tab_analysis, tab_filter, tab_spec, tab_voice = st.tabs(
    ["Signal & Spectrum", "Filtering", "Spectrogram", "Voice Demo"]
)

NO_SIGNAL_MSG = ("Upload a WAV file from the sidebar, or switch to 'Generate' "
                 "to try a test signal.")

# ===== Tab 1: Part 1 (unchanged behaviour) =====
with tab_analysis:
    if y is None:
        st.info(NO_SIGNAL_MSG)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Time-Domain Waveform")
            st.plotly_chart(line_fig([("signal", t, y, BLUE)], "Time (s)", "Amplitude"),
                            width="stretch")
        with col2:
            st.subheader("Frequency Spectrum (FFT)")
            freqs, magnitude = compute_fft(y, sample_rate)
            st.plotly_chart(line_fig([("signal", freqs, magnitude, NAVY)],
                                     "Frequency (Hz)", "Magnitude"),
                            width="stretch")
        st.metric("Dominant frequency detected",
                  f"{dominant_frequency(y, sample_rate):.1f} Hz")

# ===== Tab 2: FIR / IIR filtering =====
with tab_filter:
    if y is None:
        st.info(NO_SIGNAL_MSG)
    else:
        nyq = int(sample_rate / 2)
        c1, c2, c3 = st.columns(3)
        family = c1.radio("Filter family", ["fir", "iir"], horizontal=True,
                          format_func=str.upper)
        ftype = c2.selectbox("Filter type", ["lowpass", "highpass", "bandpass"])
        zero_phase = c3.checkbox("Zero-phase (forward-backward)", value=True,
                                 help="Removes phase delay; doubles the attenuation.")

        if ftype == "bandpass":
            lo = st.slider("Low cutoff (Hz)", 10, nyq - 50, min(200, nyq - 60))
            hi = st.slider("High cutoff (Hz)", lo + 10, nyq - 10,
                           min(lo + 300, nyq - 10))
            cutoff = (lo, hi)
        else:
            cutoff = st.slider("Cutoff (Hz)", 10, nyq - 10, min(200, nyq - 20))

        window, numtaps, order = "hamming", 101, 4
        if family == "fir":
            w1, w2 = st.columns(2)
            window = w1.selectbox("Window", ["hamming", "hanning", "blackman"])
            numtaps = w2.slider("Taps (longer = sharper)", 21, 501, 101, step=2)
        else:
            order = st.slider("Butterworth order", 1, 10, 4)

        try:
            y_f = apply_filter(y, sample_rate, family, ftype, cutoff, order=order,
                               numtaps=numtaps, window=window, zero_phase=zero_phase)
            fr, mag_db = frequency_response(family, ftype, cutoff, sample_rate,
                                            order=order, numtaps=numtaps, window=window)
        except ValueError as e:
            st.error(str(e))
        else:
            a, b = st.columns(2)
            with a:
                st.subheader("Time domain: before vs after")
                st.plotly_chart(line_fig([("original", t, y, BLUE),
                                          ("filtered", t, y_f, ORANGE)],
                                         "Time (s)", "Amplitude"),
                                width="stretch")
            with b:
                st.subheader("Spectrum: before vs after")
                f0, m0 = compute_fft(y, sample_rate)
                f1, m1 = compute_fft(y_f, sample_rate)
                st.plotly_chart(line_fig([("original", f0, m0, BLUE),
                                          ("filtered", f1, m1, ORANGE)],
                                         "Frequency (Hz)", "Magnitude"),
                                width="stretch")
            st.subheader("Filter frequency response")
            st.plotly_chart(line_fig([("response", fr, mag_db, NAVY)],
                                     "Frequency (Hz)", "Gain (dB)", height=280),
                            width="stretch")
            m1_, m2_ = st.columns(2)
            m1_.metric("Dominant frequency (original)",
                       f"{dominant_frequency(y, sample_rate):.1f} Hz")
            m2_.metric("Dominant frequency (filtered)",
                       f"{dominant_frequency(y_f, sample_rate):.1f} Hz")

# ===== Tab 3: spectrogram =====
with tab_spec:
    st.subheader("Spectrogram (frequency vs time)")
    st.caption("A longer window shows frequencies more precisely but blurs timing; "
               "a shorter window does the opposite.")
    spec_source = st.radio("Signal", ["Current signal", "Linear chirp demo (200 Hz to near Nyquist)"],
                           horizontal=True)
    if spec_source == "Current signal":
        x_s, fs_s = y, sample_rate
    else:
        fs_s = sample_rate
        t_c = np.arange(int(2 * fs_s)) / fs_s
        x_s = chirp(t_c, f0=200, f1=0.9 * fs_s / 2, t1=2, method="linear")

    if x_s is None:
        st.info(NO_SIGNAL_MSG)
    else:
        c1, c2, c3 = st.columns(3)
        win_len = c1.select_slider("Window length (samples)", [128, 256, 512, 1024], value=256)
        overlap = c2.slider("Overlap (%)", 0, 90, 75) / 100
        win_type = c3.selectbox("Window", ["hann", "hamming", "blackman"], key="spec_win")
        try:
            f_sp, t_sp, s_db = compute_spectrogram(x_s, fs_s, nperseg=win_len,
                                                   overlap=overlap, window=win_type)
        except ValueError as e:
            st.error(str(e))
        else:
            fmax = st.slider("Max frequency shown (Hz)", 100, int(fs_s / 2), int(fs_s / 2))
            st.plotly_chart(spec_fig(f_sp, t_sp, s_db, fmax=fmax), width="stretch")
            st.caption(f"Frequency resolution: about {fs_s / min(win_len, len(x_s)):.1f} Hz per bin; "
                       f"time step: {1000 * (min(win_len, len(x_s)) * (1 - overlap)) / fs_s:.1f} ms.")

# ===== Tab 4: voice demo =====
with tab_voice:
    st.subheader("Voice filtering demo")
    voice_file = st.file_uploader("Voice clip (WAV)", type=["wav"], key="voice_clip")
    if voice_file is None:
        st.info("Upload a short WAV voice clip. For noise reduction, start the clip "
                "with about half a second of silence so the noise can be measured.")
    else:
        x, fs = load_wav(io.BytesIO(voice_file.getvalue()))
        effect = st.radio("Effect", ["Telephone effect", "Noise reduction"],
                          horizontal=True)

        x_in = x
        add_noise = False
        if effect == "Noise reduction":
            n1, n2 = st.columns(2)
            add_noise = n1.checkbox("Add test noise (so SNR can be measured)", value=True)
            noise_level = n2.slider("Noise level", 0.05, 1.0, 0.3, disabled=not add_noise)
            p1, p2 = st.columns(2)
            noise_seconds = p1.slider("Noise-only lead-in (s)", 0.1, 2.0, 0.5)
            strength = p2.slider("Reduction strength", 0.5, 3.0, 1.5)
            if add_noise:
                rng = np.random.default_rng(0)
                x_in = x + noise_level * np.std(x) * rng.standard_normal(len(x))
            y_v = reduce_noise(x_in, fs, noise_seconds=noise_seconds,
                               over_subtract=strength)
        else:
            y_v = telephone_effect(x, fs)

        left, right = st.columns(2)
        left.markdown("**Original**" if not add_noise else "**Noisy input**")
        left.audio(wav_bytes(x_in, fs), format="audio/wav")
        right.markdown("**Processed**")
        right.audio(wav_bytes(y_v, fs), format="audio/wav")
        right.download_button("Download processed WAV", wav_bytes(y_v, fs),
                              file_name="processed.wav", mime="audio/wav")

        if effect == "Noise reduction" and add_noise:
            before, after = snr_db(x, x_in), snr_db(x, y_v)
            s1, s2, s3 = st.columns(3)
            s1.metric("SNR before", f"{before:.1f} dB")
            s2.metric("SNR after", f"{after:.1f} dB")
            s3.metric("Improvement", f"{after - before:+.1f} dB")

        fa, ma = compute_fft(x_in, fs)
        fb, mb = compute_fft(y_v, fs)
        st.plotly_chart(line_fig([("before", fa, ma, BLUE), ("after", fb, mb, ORANGE)],
                                 "Frequency (Hz)", "Magnitude"),
                        width="stretch")

        st.subheader("Spectrogram: before vs after")
        try:
            f_a, t_a, s_a = compute_spectrogram(x_in, fs, nperseg=512)
            f_b, t_b, s_b = compute_spectrogram(y_v, fs, nperseg=512)
            ref = max(s_a.max(), s_b.max())  # same colour scale on both
            s_a = np.maximum(s_a, ref - 80)
            s_b = np.maximum(s_b, ref - 80)
            g1, g2 = st.columns(2)
            g1.markdown("**Before**")
            g1.plotly_chart(spec_fig(f_a, t_a, s_a), width="stretch")
            g2.markdown("**After**")
            g2.plotly_chart(spec_fig(f_b, t_b, s_b), width="stretch")
        except ValueError as e:
            st.warning(f"Spectrogram unavailable: {e}")