"""FIR / IIR filter design and application (Week 2)."""
import numpy as np
from scipy import signal

_WINDOWS = {"hamming": "hamming", "hanning": "hann", "hann": "hann",
            "blackman": "blackman", "rectangular": "boxcar"}


def _check_cutoff(filter_type, cutoff, fs):
    nyq = fs / 2
    if filter_type not in ("lowpass", "highpass", "bandpass"):
        raise ValueError(f"Unknown filter type: {filter_type}")
    vals = np.atleast_1d(cutoff).astype(float)
    need = 2 if filter_type == "bandpass" else 1
    if len(vals) != need:
        raise ValueError(f"{filter_type} needs {need} cutoff value(s)")
    if np.any(vals <= 0) or np.any(vals >= nyq):
        raise ValueError(f"Cutoff must be between 0 and Nyquist ({nyq} Hz)")
    if need == 2 and vals[0] >= vals[1]:
        raise ValueError("Band-pass needs low cutoff < high cutoff")
    return vals if need == 2 else float(vals[0])


def design_fir(filter_type, cutoff, fs, numtaps=101, window="hamming"):
    """Window-method FIR. Returns taps b. numtaps is forced odd."""
    cutoff = _check_cutoff(filter_type, cutoff, fs)
    if numtaps % 2 == 0:
        numtaps += 1
    win = _WINDOWS.get(window.lower())
    if win is None:
        raise ValueError(f"Unknown window: {window}")
    return signal.firwin(numtaps, cutoff, window=win,
                         pass_zero=(filter_type == "lowpass"), fs=fs)


def design_iir(filter_type, cutoff, fs, order=4):
    """Butterworth IIR as second-order sections (numerically stable)."""
    cutoff = _check_cutoff(filter_type, cutoff, fs)
    return signal.butter(order, cutoff, btype=filter_type, fs=fs, output="sos")


def apply_filter(x, fs, family, filter_type, cutoff, order=4,
                 numtaps=101, window="hamming", zero_phase=False):
    """Filter x. family: 'fir' or 'iir'. zero_phase=True uses forward-backward
    filtering (no phase delay, but doubles the attenuation)."""
    x = np.asarray(x, dtype=float)
    if family == "fir":
        b = design_fir(filter_type, cutoff, fs, numtaps, window)
        return signal.filtfilt(b, [1.0], x) if zero_phase else signal.lfilter(b, [1.0], x)
    if family == "iir":
        sos = design_iir(filter_type, cutoff, fs, order)
        return signal.sosfiltfilt(sos, x) if zero_phase else signal.sosfilt(sos, x)
    raise ValueError("family must be 'fir' or 'iir'")


def frequency_response(family, filter_type, cutoff, fs, order=4,
                       numtaps=101, window="hamming", n_points=1024):
    """Return (freqs_hz, magnitude_db) for plotting the filter itself."""
    if family == "fir":
        b = design_fir(filter_type, cutoff, fs, numtaps, window)
        w, h = signal.freqz(b, worN=n_points, fs=fs)
    else:
        sos = design_iir(filter_type, cutoff, fs, order)
        w, h = signal.sosfreqz(sos, worN=n_points, fs=fs)
    return w, 20 * np.log10(np.maximum(np.abs(h), 1e-12))
