"""Convolution, correlation and AM/FM modulation (Week 4)."""
import numpy as np
from scipy import signal
from .filters import apply_filter


def _normalise(m):
    m = np.asarray(m, dtype=float)
    peak = np.max(np.abs(m))
    return m / peak if peak > 0 else m


# ---------- convolution ----------
def moving_average_kernel(n):
    if n < 1:
        raise ValueError("Kernel length must be at least 1")
    return np.ones(int(n)) / int(n)


def echo_kernel(fs, delay_s, gain):
    d = int(round(delay_s * fs))
    if d < 1:
        raise ValueError("Echo delay must be at least one sample")
    h = np.zeros(d + 1)
    h[0], h[d] = 1.0, gain
    return h


def convolve_signals(x, h, mode="full"):
    """FFT-based linear convolution. mode: 'full' or 'same'."""
    x, h = np.asarray(x, dtype=float), np.asarray(h, dtype=float)
    if len(x) == 0 or len(h) == 0:
        raise ValueError("Signal and kernel must not be empty")
    if mode not in ("full", "same"):
        raise ValueError("mode must be 'full' or 'same'")
    return signal.fftconvolve(x, h, mode=mode)


# ---------- correlation ----------
def cross_correlation(x, y, fs):
    """Normalised cross-correlation. A positive peak lag means y lags x."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    c = signal.correlate(y, x, mode="full", method="fft")
    lags = signal.correlation_lags(len(y), len(x), mode="full")
    norm = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    return lags / fs, (c / norm if norm > 0 else c)


def estimate_delay(x, y, fs):
    """Delay of y relative to x in seconds (peak of the cross-correlation)."""
    lags, c = cross_correlation(x, y, fs)
    return float(lags[np.argmax(c)])


def estimate_period(x, fs, min_height=0.3):
    """Period (s) from the first autocorrelation peak after lag 0, or None."""
    lags, c = cross_correlation(x, x, fs)
    keep = lags >= 0
    lags, c = lags[keep], c[keep]
    peaks, _ = signal.find_peaks(c, height=min_height)
    return float(lags[peaks[0]]) if len(peaks) else None


# ---------- modulation ----------
def _check_carrier(fc, fs):
    if not 0 < fc < fs / 2:
        raise ValueError(f"Carrier must be between 0 and Nyquist ({fs / 2} Hz)")


def am_modulate(m, fs, fc, depth=0.8):
    """s(t) = (1 + depth*m(t)) cos(2*pi*fc*t), with m scaled to [-1, 1]."""
    _check_carrier(fc, fs)
    if not 0 < depth <= 1:
        raise ValueError("Modulation depth must be in (0, 1]")
    m = _normalise(m)
    t = np.arange(len(m)) / fs
    return (1 + depth * m) * np.cos(2 * np.pi * fc * t)


def am_demodulate(s, fs, depth=0.8, bandwidth=None):
    """Envelope detection (Hilbert), then optional low-pass of `bandwidth` Hz."""
    m_hat = (np.abs(signal.hilbert(s)) - 1) / depth
    if bandwidth:
        m_hat = apply_filter(m_hat, fs, "iir", "lowpass", bandwidth, zero_phase=True)
    return m_hat


def fm_modulate(m, fs, fc, deviation=200.0):
    """s(t) = cos(2*pi*fc*t + 2*pi*deviation*integral(m)); m scaled to [-1, 1]."""
    _check_carrier(fc, fs)
    if deviation <= 0 or fc - deviation <= 0 or fc + deviation >= fs / 2:
        raise ValueError("Carrier +/- deviation must stay between 0 and Nyquist")
    m = _normalise(m)
    t = np.arange(len(m)) / fs
    return np.cos(2 * np.pi * fc * t + 2 * np.pi * deviation * np.cumsum(m) / fs)


def fm_demodulate(s, fs, fc, deviation=200.0, bandwidth=None):
    """Instantaneous frequency from the analytic signal, then optional low-pass."""
    phase = np.unwrap(np.angle(signal.hilbert(s)))
    inst = np.append(np.diff(phase), 0.0) * fs / (2 * np.pi)
    inst[-1] = inst[-2]
    m_hat = (inst - fc) / deviation
    if bandwidth:
        m_hat = apply_filter(m_hat, fs, "iir", "lowpass", bandwidth, zero_phase=True)
    return m_hat


def recovery_error(m, m_hat, trim=0.05):
    """RMS error of the recovered message vs the original (both scaled to
    [-1, 1]), ignoring `trim` of each end where the Hilbert transform rings."""
    m, m_hat = _normalise(m), np.asarray(m_hat, dtype=float)
    k = int(len(m) * trim)
    a, b = m[k:len(m) - k], m_hat[k:len(m) - k]
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(a ** 2)))
