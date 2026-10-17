import io
import numpy as np
import pytest
from signals.operations import (
    moving_average_kernel, echo_kernel, convolve_signals, cross_correlation,
    estimate_delay, estimate_period, am_modulate, am_demodulate, fm_modulate,
    fm_demodulate, recovery_error)
from signals.loaders import numeric_columns

FS = 8000


def tone(freq, dur=1.0):
    t = np.arange(int(FS * dur)) / FS
    return np.sin(2 * np.pi * freq * t)


def peak_hz(x):
    return np.fft.rfftfreq(len(x), 1 / FS)[np.argmax(np.abs(np.fft.rfft(x)))]


# ----- convolution -----
def test_convolution_matches_numpy_and_length():
    x, h = np.random.default_rng(0).standard_normal((2, 200))
    y = convolve_signals(x, h)
    assert len(y) == len(x) + len(h) - 1
    assert np.allclose(y, np.convolve(x, h))


def test_impulse_kernel_is_identity_and_shift_is_delay():
    x = np.random.default_rng(1).standard_normal(100)
    assert np.allclose(convolve_signals(x, [1.0], "same"), x)
    delayed = convolve_signals(x, [0, 0, 0, 1.0])
    assert np.allclose(delayed[3:3 + len(x)], x)


def test_moving_average_attenuates_high_frequency():
    x = tone(100) + tone(3000)
    y = convolve_signals(x, moving_average_kernel(16), "same")
    spec = np.abs(np.fft.rfft(y))
    assert spec[3000] < 0.1 * np.abs(np.fft.rfft(x))[3000]
    assert spec[100] > 0.8 * np.abs(np.fft.rfft(x))[100]


def test_echo_kernel_shape_and_bad_input():
    h = echo_kernel(FS, 0.1, 0.5)
    assert len(h) == 801 and h[0] == 1 and h[800] == 0.5
    with pytest.raises(ValueError):
        echo_kernel(FS, 0, 0.5)
    with pytest.raises(ValueError):
        convolve_signals([], [1.0])


# ----- correlation -----
def test_cross_correlation_recovers_delay_in_noise():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(FS)
    d = 200  # 25 ms
    y = np.concatenate([np.zeros(d), x])[:FS] + 0.5 * rng.standard_normal(FS)
    assert abs(estimate_delay(x, y, FS) - d / FS) <= 1 / FS


def test_autocorrelation_peak_is_one_and_period_found():
    lags, c = cross_correlation(tone(100), tone(100), FS)
    assert lags[np.argmax(c)] == 0 and c.max() == pytest.approx(1.0)
    assert estimate_period(tone(100), FS) == pytest.approx(0.01, abs=2 / FS)
    assert estimate_period(np.random.default_rng(3).standard_normal(FS), FS) is None


# ----- modulation -----
def test_am_spectrum_has_carrier_and_sidebands():
    s = am_modulate(tone(50), FS, 1000, depth=0.8)
    spec = np.abs(np.fft.rfft(s))
    for f in (950, 1000, 1050):
        assert spec[f] > 0.05 * spec[1000]
    assert peak_hz(s) == 1000


def test_am_roundtrip():
    m = tone(50)
    s = am_modulate(m, FS, 1000, depth=0.8)
    assert recovery_error(m, am_demodulate(s, FS, 0.8, bandwidth=500)) < 0.05


def test_fm_roundtrip():
    m = tone(50)
    s = fm_modulate(m, FS, 1500, deviation=300)
    assert recovery_error(m, fm_demodulate(s, FS, 1500, 300, bandwidth=500)) < 0.05


def test_am_demodulation_with_noise_still_close():
    rng = np.random.default_rng(4)
    m = tone(50)
    s = am_modulate(m, FS, 1000, 0.8)
    s = s + 0.1 * np.std(s) * rng.standard_normal(len(s))
    assert recovery_error(m, am_demodulate(s, FS, 0.8, bandwidth=200)) < 0.25


def test_modulation_bad_parameters_raise():
    with pytest.raises(ValueError):
        am_modulate(tone(50), FS, FS, 0.5)
    with pytest.raises(ValueError):
        am_modulate(tone(50), FS, 1000, 1.5)
    with pytest.raises(ValueError):
        fm_modulate(tone(50), FS, 3900, deviation=300)


# ----- CSV import -----
def test_csv_loader_keeps_numeric_columns():
    rows = "time,temp,label\n" + "\n".join(f"{i},{20 + i % 3},a" for i in range(50))
    nums = numeric_columns(io.StringIO(rows))
    assert list(nums.columns) == ["time", "temp"] and len(nums) == 50


def test_csv_loader_rejects_bad_files():
    with pytest.raises(ValueError):
        numeric_columns(io.StringIO("a,b\nx,y\n"))
    with pytest.raises(ValueError):
        numeric_columns(io.StringIO(""))
