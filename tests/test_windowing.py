import numpy as np
import pytest
from signals.windowing import WINDOWS, apply_window, windowed_spectrum, to_db


def test_window_reduces_leakage_for_off_bin_tone():
    fs = 1000
    t = np.arange(1000) / fs
    y = np.sin(2 * np.pi * 50.5 * t)  # between FFT bins -> leakage
    f, rect = windowed_spectrum(y, fs, "Rectangular (none)")
    _, hann = windowed_spectrum(y, fs, "Hanning")
    far = f > 200
    assert hann[far].max() < rect[far].max()


@pytest.mark.parametrize("name", list(WINDOWS))
def test_window_amplitude_recovered(name):
    fs = 1000
    t = np.arange(1000) / fs
    _, mag = windowed_spectrum(2.0 * np.sin(2 * np.pi * 100 * t), fs, name)
    assert abs(mag.max() - 2.0) < 0.05


def test_unknown_window_raises():
    with pytest.raises(ValueError):
        apply_window(np.ones(10), "Nope")


def test_to_db_floor_never_minus_infinity():
    assert np.all(np.isfinite(to_db(np.zeros(5))))
