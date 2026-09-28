"""
Basic sanity tests for Part 1 (signal generation + FFT).
Run with: python -m pytest tests/
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.generator import generate
from signals.analysis import compute_fft, dominant_frequency


def test_sine_dominant_frequency():
    t, y = generate("Sine", freq=50, amplitude=1.0, duration=1.0, sample_rate=1000)
    peak = dominant_frequency(y, 1000)
    assert abs(peak - 50) < 1.0


def test_sine_amplitude_recovered():
    t, y = generate("Sine", freq=100, amplitude=2.0, duration=1.0, sample_rate=2000)
    _, magnitude = compute_fft(y, 2000)
    assert abs(max(magnitude) - 2.0) < 0.05


def test_signal_length_matches_duration():
    t, y = generate("Square", freq=10, amplitude=1.0, duration=0.5, sample_rate=1000)
    assert len(y) == 500


def test_unknown_signal_type_raises():
    try:
        generate("NotAWaveform", 1, 1, 1, 1000)
        assert False, "Expected ValueError"
    except ValueError:
        pass
