"""
signals/analysis.py

Frequency-domain analysis: computes the FFT of a signal and returns
a one-sided magnitude spectrum ready for plotting.
"""

import numpy as np


def compute_fft(y: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the one-sided FFT magnitude spectrum of a real-valued signal.

    Returns:
        freqs: array of frequency bins (Hz), 0 .. Nyquist
        magnitude: normalized magnitude at each frequency bin
    """
    n = len(y)
    fft_vals = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Normalize so amplitude reads consistently regardless of signal length,
    # and double all bins except DC/Nyquist to account for the folded negative-frequency energy.
    magnitude = np.abs(fft_vals) / n
    if n > 1:
        magnitude[1:-1] *= 2

    return freqs, magnitude


def dominant_frequency(y: np.ndarray, sample_rate: int) -> float:
    """Return the frequency (Hz) with the highest magnitude in the spectrum (ignoring DC)."""
    freqs, magnitude = compute_fft(y, sample_rate)
    if len(freqs) <= 1:
        return 0.0
    idx = np.argmax(magnitude[1:]) + 1  # skip the DC bin at index 0
    return float(freqs[idx])
