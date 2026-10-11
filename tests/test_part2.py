import numpy as np
import pytest
from signals.filters import apply_filter, design_fir
from signals.voice import telephone_effect, reduce_noise, snr_db

FS = 8000


def tone(freq, dur=1.0, amp=1.0):
    t = np.arange(int(FS * dur)) / FS
    return amp * np.sin(2 * np.pi * freq * t)


def peak_db(x, freq):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) / len(x)
    k = int(round(freq * len(x) / FS))
    return 20 * np.log10(spec[k - 2:k + 3].max() + 1e-12)


@pytest.mark.parametrize("family", ["fir", "iir"])
def test_lowpass_removes_500hz(family):
    x = tone(50) + tone(500)
    y = apply_filter(x, FS, family, "lowpass", 100, numtaps=401, zero_phase=True)
    assert peak_db(x, 500) - peak_db(y, 500) > 40
    assert abs(peak_db(x, 50) - peak_db(y, 50)) < 1.0


@pytest.mark.parametrize("family", ["fir", "iir"])
def test_highpass_removes_50hz(family):
    x = tone(50) + tone(1000)
    y = apply_filter(x, FS, family, "highpass", 300, numtaps=401, zero_phase=True)
    assert peak_db(x, 50) - peak_db(y, 50) > 40


def test_bandpass_keeps_middle_tone():
    x = tone(50) + tone(500) + tone(3000)
    y = apply_filter(x, FS, "iir", "bandpass", (300, 800), zero_phase=True)
    assert peak_db(x, 50) - peak_db(y, 50) > 40
    assert peak_db(x, 3000) - peak_db(y, 3000) > 40
    assert abs(peak_db(x, 500) - peak_db(y, 500)) < 1.0


def test_fir_length_forced_odd():
    assert len(design_fir("highpass", 300, FS, numtaps=100)) % 2 == 1


def test_invalid_cutoff_raises():
    with pytest.raises(ValueError):
        apply_filter(tone(50), FS, "iir", "lowpass", FS)
    with pytest.raises(ValueError):
        apply_filter(tone(50), FS, "fir", "bandpass", 500)


def test_telephone_effect_limits_band():
    x = tone(100) + tone(1000) + tone(3900)
    y = telephone_effect(x, FS)
    assert peak_db(x, 100) - peak_db(y, 100) > 10
    assert abs(peak_db(x, 1000) - peak_db(y, 1000)) < 1.0


def test_noise_reduction_improves_snr():
    rng = np.random.default_rng(0)
    clean = np.concatenate([np.zeros(FS // 2), tone(440, 2.0, 0.5)])
    noisy = clean + 0.1 * rng.standard_normal(len(clean))
    assert snr_db(clean, reduce_noise(noisy, FS)) > snr_db(clean, noisy)
