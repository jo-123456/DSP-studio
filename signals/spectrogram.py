"""Short-time spectrogram helper (Week 3)."""
import numpy as np
from scipy import signal

_WINDOWS = {"hann": "hann", "hanning": "hann", "hamming": "hamming",
            "blackman": "blackman"}


def compute_spectrogram(x, fs, nperseg=256, overlap=0.75, window="hann",
                        dynamic_range_db=80.0):
    """Return (freqs_hz, times_s, power_db) with shape (len(freqs), len(times)).

    power_db is clipped to `dynamic_range_db` below its maximum so the colour
    scale is readable. Longer nperseg gives finer frequency detail but blurrier
    timing, and the other way round.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 16:
        raise ValueError("Signal is too short for a spectrogram")
    if window.lower() not in _WINDOWS:
        raise ValueError(f"Unknown window: {window}")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    nperseg = int(min(nperseg, len(x)))
    noverlap = min(int(nperseg * overlap), nperseg - 1)
    f, t, s = signal.spectrogram(x, fs, window=_WINDOWS[window.lower()],
                                 nperseg=nperseg, noverlap=noverlap,
                                 scaling="spectrum", mode="psd")
    s_db = 10 * np.log10(s + 1e-20)
    s_db = np.maximum(s_db, s_db.max() - dynamic_range_db)
    return f, t, s_db
