"""
DSP Studio — signal analysis, filtering, spectrogram, operations and voice demo
(dark "lab instrument" theme: see ui_theme.py and .streamlit/config.toml)

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
from ui_theme import (apply_theme, header, style_fig, show_chart, rgba,
                      PRIMARY, SECONDARY, WARN, SPECTROGRAM_SCALE)
from signals.operations import (
    moving_average_kernel, echo_kernel, convolve_signals, cross_correlation,
    estimate_delay, estimate_period, am_modulate, am_demodulate, fm_modulate,
    fm_demodulate, recovery_error)
from signals.windowing import WINDOWS, windowed_spectrum, to_db
from signals.loaders import numeric_columns
from scipy.signal import chirp

st.set_page_config(page_title="DSP Studio", layout="wide")
apply_theme()

BLUE, NAVY, ORANGE = PRIMARY, SECONDARY, WARN


def line_fig(series, xtitle, ytitle, height=340):
    """series: list of (name, x, y, color). One Plotly line chart."""
    fig = go.Figure()
    for name, x, y_, color in series:
        fig.add_trace(go.Scatter(
            x=x, y=y_, mode="lines", name=name, line=dict(color=color, width=1.5),
            fill="tozeroy" if len(series) == 1 else None,
            fillcolor=rgba(color, 0.10) if len(series) == 1 else None))
    fig.update_layout(xaxis_title=xtitle, yaxis_title=ytitle, height=height,
                      margin=dict(l=40, r=20, t=20, b=40))
    return style_fig(fig)


def thin(*arrays, max_points=20000):
    """Stride long arrays so Plotly stays responsive."""
    k = max(1, len(arrays[0]) // max_points)
    return [a[::k] for a in arrays]


def spec_fig(f, t, s_db, fmax=None, height=360):
    """Spectrogram heatmap. Strides the data so huge clips stay responsive."""
    if fmax:
        keep = f <= fmax
        f, s_db = f[keep], s_db[keep]
    ft = max(1, len(t) // 600)
    ff = max(1, len(f) // 400)
    fig = go.Figure(go.Heatmap(x=t[::ft], y=f[::ff], z=s_db[::ff, ::ft],
                               colorscale=SPECTROGRAM_SCALE,
                               colorbar=dict(title="dB")))
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
                      height=height, margin=dict(l=40, r=20, t=20, b=40))
    return style_fig(fig)


# ---------- Sidebar: signal source ----------
st.sidebar.title("DSP Studio")
st.sidebar.caption("Analysis · Filtering · Spectrogram · Operations · Voice")

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
    uploaded = st.sidebar.file_uploader("Upload a WAV or CSV file", type=["wav", "csv"])
    if uploaded is not None and uploaded.name.lower().endswith(".csv"):
        try:
            table = numeric_columns(io.BytesIO(uploaded.getvalue()))
        except ValueError as e:
            st.sidebar.error(str(e))
        else:
            col = st.sidebar.selectbox("Column", list(table.columns))
            sample_rate = float(st.sidebar.number_input("Sample rate (Hz)", 1.0, 100000.0, 1000.0))
            remove_dc = st.sidebar.checkbox("Remove DC offset", value=True)
            y = table[col].to_numpy(dtype=float)
            if remove_dc:
                y = y - y.mean()
            y = y / (np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0)
            t = np.arange(len(y)) / sample_rate
    elif uploaded is not None:
        import scipy.io.wavfile as wavfile

        sr, data = wavfile.read(io.BytesIO(uploaded.getvalue()))
        if data.ndim > 1:
            data = data.mean(axis=1)  # collapse stereo to mono
        y = data.astype(np.float64)
        y /= np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
        sample_rate = sr
        t = np.linspace(0, len(y) / sr, len(y), endpoint=False)

# ---------- Main area ----------
header()
if y is not None:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sampling frequency", f"{sample_rate:,.0f} Hz")
    k2.metric("Nyquist limit", f"{sample_rate / 2:,.0f} Hz")
    k3.metric("Samples", f"{len(y):,}")
    k4.metric("Duration", f"{len(y) / sample_rate:.2f} s")
tab_analysis, tab_filter, tab_spec, tab_ops, tab_voice = st.tabs(
    ["Signal & Spectrum", "Filtering", "Spectrogram", "Operations", "Voice Demo"]
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
            show_chart(line_fig([("signal", t, y, BLUE)], "Time (s)", "Amplitude"),
                            width="stretch")
        with col2:
            st.subheader("Frequency Spectrum (FFT)")
            freqs, magnitude = compute_fft(y, sample_rate)
            show_chart(line_fig([("signal", freqs, magnitude, NAVY)],
                                     "Frequency (Hz)", "Magnitude"),
                            width="stretch")
        st.metric("Dominant frequency detected",
                  f"{dominant_frequency(y, sample_rate):.1f} Hz")

# ===== Tab 2: FIR / IIR filtering =====
with tab_filter:
    if y is None:
        st.info(NO_SIGNAL_MSG)
    elif sample_rate < 200:
        st.info("The filter controls need a sample rate of at least 200 Hz. Raise it in the sidebar.")
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
                show_chart(line_fig([("original", t, y, BLUE),
                                          ("filtered", t, y_f, ORANGE)],
                                         "Time (s)", "Amplitude"),
                                width="stretch")
            with b:
                st.subheader("Spectrum: before vs after")
                f0, m0 = compute_fft(y, sample_rate)
                f1, m1 = compute_fft(y_f, sample_rate)
                show_chart(line_fig([("original", f0, m0, BLUE),
                                          ("filtered", f1, m1, ORANGE)],
                                         "Frequency (Hz)", "Magnitude"),
                                width="stretch")
            st.subheader("Filter frequency response")
            show_chart(line_fig([("response", fr, mag_db, NAVY)],
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
            fmax = st.slider("Max frequency shown (Hz)", max(1, min(100, int(fs_s / 2) - 1)), int(fs_s / 2), int(fs_s / 2))
            show_chart(spec_fig(f_sp, t_sp, s_db, fmax=fmax), width="stretch")
            st.caption(f"Frequency resolution: about {fs_s / min(win_len, len(x_s)):.1f} Hz per bin; "
                       f"time step: {1000 * (min(win_len, len(x_s)) * (1 - overlap)) / fs_s:.1f} ms.")

# ===== Tab 4: convolution / correlation / modulation / windowing =====
with tab_ops:
    if y is None:
        st.info(NO_SIGNAL_MSG)
    elif sample_rate < 800:
        st.info("These tools need a sample rate of at least 800 Hz. Raise it in the sidebar.")
    else:
        fs_o = float(sample_rate)
        nyq_o = fs_o / 2
        x_o = y[: int(5 * fs_o)]
        if len(y) > len(x_o):
            st.caption("Using the first 5 seconds of the signal for these tools.")
        op = st.radio("Operation", ["Convolution", "Correlation", "Modulation", "Windowing"],
                      horizontal=True)

        # ---- convolution ----
        if op == "Convolution":
            kernel_name = st.selectbox("Kernel", ["Moving average", "Echo", "Differentiator"])
            if kernel_name == "Moving average":
                n_k = st.slider("Averaging length (samples)", 2, 200, 10)
                h = moving_average_kernel(n_k)
            elif kernel_name == "Echo":
                e1, e2 = st.columns(2)
                d_ms = e1.slider("Echo delay (ms)", 5, 500, 100)
                gain = e2.slider("Echo gain", 0.1, 1.0, 0.5)
                h = echo_kernel(fs_o, d_ms / 1000, gain)
            else:
                h = np.array([1.0, -1.0])
            y_c = convolve_signals(x_o, h, mode="same")
            a, b = st.columns(2)
            with a:
                st.subheader("Kernel h[n]")
                kx, ky = thin(np.arange(len(h)), h)
                show_chart(line_fig([("h[n]", kx, ky, NAVY)], "Sample n", "Weight"),
                                width="stretch")
            with b:
                st.subheader("Signal: before vs after")
                tx, ox, cx = thin(t[: len(x_o)], x_o, y_c)
                show_chart(line_fig([("original", tx, ox, BLUE), ("convolved", tx, cx, ORANGE)],
                                         "Time (s)", "Amplitude"), width="stretch")
            f0, m0 = compute_fft(x_o, fs_o)
            f1, m1 = compute_fft(y_c, fs_o)
            st.subheader("Spectrum: before vs after")
            show_chart(line_fig([("original", f0, m0, BLUE), ("convolved", f1, m1, ORANGE)],
                                     "Frequency (Hz)", "Magnitude", height=300), width="stretch")
            st.caption("Convolution in time multiplies the spectrum by the kernel's frequency "
                       "response: a moving average is a low-pass filter, an echo adds ripples, "
                       "and a differentiator boosts high frequencies.")

        # ---- correlation ----
        elif op == "Correlation":
            ctype = st.radio("Correlation type",
                             ["Cross-correlation (find a delay)", "Autocorrelation"],
                             horizontal=True)
            if ctype.startswith("Cross"):
                p1, p2, p3 = st.columns(3)
                probe = p1.radio("Probe signal", ["Random noise", "Current signal"])
                max_ms = max(2, min(500, int(1000 * len(x_o) / fs_o / 2)))
                d_ms = p2.slider("True delay (ms)", 1, max_ms, min(25, max_ms))
                noise_lvl = p3.slider("Noise added to delayed copy", 0.0, 2.0, 0.3)
                rng = np.random.default_rng(0)
                base = rng.standard_normal(len(x_o)) * np.std(x_o) if probe == "Random noise" else x_o
                d = int(round(d_ms * fs_o / 1000))
                delayed = np.concatenate([np.zeros(d), base])[: len(base)]
                delayed = delayed + noise_lvl * np.std(base) * rng.standard_normal(len(base))
                lags, c = cross_correlation(base, delayed, fs_o)
                est_ms = 1000 * estimate_delay(base, delayed, fs_o)
                m1_, m2_ = st.columns(2)
                m1_.metric("True delay", f"{d * 1000 / fs_o:.1f} ms")
                m2_.metric("Estimated delay", f"{est_ms:.1f} ms")
                lx, cy = thin(lags * 1000, c)
                show_chart(line_fig([("correlation", lx, cy, NAVY)], "Lag (ms)",
                                         "Normalised correlation"), width="stretch")
                if probe == "Current signal":
                    st.caption("A periodic signal correlates with itself at every multiple of its "
                               "period, so the delay is only known modulo the period. "
                               "Switch to random noise for a single clear peak.")
            else:
                lags, c = cross_correlation(x_o, x_o, fs_o)
                keep = lags >= 0
                lx, cy = thin(lags[keep] * 1000, c[keep])
                show_chart(line_fig([("autocorrelation", lx, cy, NAVY)], "Lag (ms)",
                                         "Normalised correlation"), width="stretch")
                per = estimate_period(x_o, fs_o)
                m1_, m2_ = st.columns(2)
                if per:
                    m1_.metric("Period from autocorrelation", f"{per * 1000:.2f} ms")
                    m2_.metric("Implied frequency", f"{1 / per:.1f} Hz")
                else:
                    m1_.info("No clear repeating pattern (non-periodic signal).")

        # ---- modulation ----
        elif op == "Modulation":
            scheme = st.radio("Modulation scheme", ["AM", "FM"], horizontal=True)
            msg = x_o / max(np.max(np.abs(x_o)), 1e-12)
            c1, c2, c3 = st.columns(3)
            fc = c1.slider("Carrier frequency (Hz)", 100, int(nyq_o) - 100,
                           min(1000, int(nyq_o / 2)))
            dev_max = int(min(fc, nyq_o - fc) * 0.9)
            if scheme == "AM":
                depth = c2.slider("Modulation depth", 0.1, 1.0, 0.8)
            elif dev_max < 20:
                st.error("The carrier is too close to 0 Hz or Nyquist for FM. "
                         "Move it toward the middle.")
            else:
                dev = c2.slider("Frequency deviation (Hz)", 10, dev_max, min(200, dev_max))
            noise_lvl = c3.slider("Channel noise", 0.0, 1.0, 0.1)
            bw = st.slider("Receiver low-pass (Hz)", 20, int(nyq_o * 0.9), min(500, int(nyq_o * 0.4)))
            if scheme == "AM" or dev_max >= 20:
                if scheme == "AM":
                    s = am_modulate(msg, fs_o, fc, depth)
                else:
                    s = fm_modulate(msg, fs_o, fc, dev)
                rx = s + noise_lvl * np.std(s) * np.random.default_rng(0).standard_normal(len(s))
                m_hat = (am_demodulate(rx, fs_o, depth, bw) if scheme == "AM"
                         else fm_demodulate(rx, fs_o, fc, dev, bw))
                win_s = st.slider("Time window shown (s)", 0.01,
                                  float(max(0.05, len(x_o) / fs_o)),
                                  float(min(0.1, max(0.05, len(x_o) / fs_o))))
                k = int(win_s * fs_o)
                a, b = st.columns(2)
                with a:
                    st.subheader(f"{scheme} signal received")
                    show_chart(line_fig([("received", t[:k], rx[:k], ORANGE)],
                                             "Time (s)", "Amplitude"), width="stretch")
                with b:
                    st.subheader("Spectrum of received signal")
                    fr_, mr_ = compute_fft(rx, fs_o)
                    show_chart(line_fig([("spectrum", fr_, mr_, NAVY)],
                                             "Frequency (Hz)", "Magnitude"), width="stretch")
                st.subheader("Message: original vs recovered")
                show_chart(line_fig([("original", t[:k], msg[:k], BLUE),
                                          ("recovered", t[:k], m_hat[:k], ORANGE)],
                                         "Time (s)", "Normalised amplitude", height=300),
                                width="stretch")
                st.metric("Recovery error (RMS, % of message)",
                          f"{100 * recovery_error(msg, m_hat):.1f} %")
                if fc < 4 * dominant_frequency(x_o, fs_o):
                    st.warning("The carrier should be well above the message frequency.")

        # ---- windowing ----
        else:
            w1, w2 = st.columns(2)
            f_tone = w1.slider("Test tone frequency (Hz)", 20.0, min(400.0, nyq_o - 50), 50.5, step=0.5)
            names = w2.multiselect("Windows to compare", list(WINDOWS),
                                   default=["Rectangular (none)", "Hanning"])
            t_w = np.arange(int(fs_o)) / fs_o
            tone_w = np.sin(2 * np.pi * f_tone * t_w)
            fig_w = go.Figure()
            for name in names:
                fw, mw = windowed_spectrum(tone_w, fs_o, name)
                fig_w.add_trace(go.Scatter(x=fw, y=to_db(mw), mode="lines", name=name))
            fig_w.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)",
                                xaxis_range=[0, min(500, nyq_o)], height=380,
                                margin=dict(l=40, r=20, t=20, b=40))
            style_fig(fig_w)
            show_chart(fig_w, width="stretch")
            st.caption("A tone that doesn't land exactly on an FFT bin leaks energy into "
                       "neighbouring bins. Windows taper the edges to reduce the leakage, "
                       "at the cost of a wider main peak.")

# ===== Tab 5: voice demo =====
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
        show_chart(line_fig([("before", fa, ma, BLUE), ("after", fb, mb, ORANGE)],
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
            with g1:
                show_chart(spec_fig(f_a, t_a, s_a))
            g2.markdown("**After**")
            with g2:
                show_chart(spec_fig(f_b, t_b, s_b))
        except ValueError as e:
            st.warning(f"Spectrogram unavailable: {e}")
