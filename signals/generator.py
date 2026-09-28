"""
signals/generator.py

Generates standard test signals used across DSP Studio:
sine, square, sawtooth, triangle, and white noise.
"""

import numpy as np
from scipy import signal as sp_signal


def time_vector(duration: float, sample_rate: int) -> np.ndarray:
    """Return a time axis array for the given duration and sample rate."""
    n_samples = int(duration * sample_rate)
    return np.linspace(0, duration, n_samples, endpoint=False)


def generate_sine(freq: float, amplitude: float, duration: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    t = time_vector(duration, sample_rate)
    y = amplitude * np.sin(2 * np.pi * freq * t)
    return t, y


def generate_square(freq: float, amplitude: float, duration: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    t = time_vector(duration, sample_rate)
    y = amplitude * sp_signal.square(2 * np.pi * freq * t)
    return t, y


def generate_sawtooth(freq: float, amplitude: float, duration: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    t = time_vector(duration, sample_rate)
    y = amplitude * sp_signal.sawtooth(2 * np.pi * freq * t)
    return t, y


def generate_triangle(freq: float, amplitude: float, duration: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    t = time_vector(duration, sample_rate)
    # width=0.5 turns the sawtooth wave generator into a symmetric triangle wave
    y = amplitude * sp_signal.sawtooth(2 * np.pi * freq * t, width=0.5)
    return t, y


def generate_white_noise(amplitude: float, duration: float, sample_rate: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    t = time_vector(duration, sample_rate)
    rng = np.random.default_rng(seed)
    y = amplitude * rng.normal(0, 1, size=t.shape)
    return t, y


GENERATORS = {
    "Sine": generate_sine,
    "Square": generate_square,
    "Sawtooth": generate_sawtooth,
    "Triangle": generate_triangle,
}


def generate(signal_type: str, freq: float, amplitude: float, duration: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the right generator by name. Raises ValueError on unknown type."""
    if signal_type not in GENERATORS:
        raise ValueError(f"Unknown signal type '{signal_type}'. Choose from {list(GENERATORS)}.")
    return GENERATORS[signal_type](freq, amplitude, duration, sample_rate)
