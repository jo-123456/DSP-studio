"""
signals/windowing.py

Window functions and their effect on the FFT spectrum (spectral leakage demo).
"""

import numpy as np
from scipy import signal as sp_signal

WINDOWS = {
    "Rectangular (none)": "boxcar",
    "Hamming": "hamming",
    "Hanning": "hann",
    "Blackman": "blackman",
}


def apply_window(y: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Multiply y by the chosen window. Returns (windowed_signal, window)."""
    if name not in WINDOWS:
        raise ValueError(f"Unknown window '{name}'. Choose from {list(WINDOWS)}.")
    w = sp_signal.get_window(WINDOWS[name], len(y))
    return y * w, w


def windowed_spectrum(y: np.ndarray, sample_rate: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    """One-sided magnitude spectrum of y after applying the chosen window (amplitude-corrected)."""
    yw, w = apply_window(y, name)
    spectrum = np.abs(np.fft.rfft(yw)) / np.sum(w)
    spectrum[1:-1] *= 2
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sample_rate)
    return freqs, spectrum


def to_db(magnitude: np.ndarray, floor_db: float = -120.0) -> np.ndarray:
    """Convert magnitude to dB, clipped at a floor so log(0) never appears."""
    return np.maximum(20 * np.log10(np.maximum(magnitude, 1e-12)), floor_db)
