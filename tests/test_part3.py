import numpy as np
import pytest
from scipy.signal import chirp
from signals.spectrogram import compute_spectrogram
from signals.voice import reduce_noise

FS = 8000


def test_tone_appears_at_correct_frequency():
    t = np.arange(FS) / FS
    f, tt, s = compute_spectrogram(np.sin(2 * np.pi * 1000 * t), FS, nperseg=512)
    peak = f[np.argmax(s.mean(axis=1))]
    assert abs(peak - 1000) <= FS / 512  # within one frequency bin


def test_chirp_frequency_rises_over_time():
    t = np.arange(2 * FS) / FS
    x = chirp(t, f0=200, f1=3000, t1=2, method="linear")
    f, tt, s = compute_spectrogram(x, FS, nperseg=256)
    early = f[np.argmax(s[:, 2])]
    late = f[np.argmax(s[:, -3])]
    assert late > early + 1000


def test_output_shape_and_dynamic_range():
    x = np.random.default_rng(1).standard_normal(FS)
    f, t, s = compute_spectrogram(x, FS, dynamic_range_db=60)
    assert s.shape == (len(f), len(t))
    assert s.max() - s.min() <= 60 + 1e-9


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        compute_spectrogram(np.zeros(5), FS)
    with pytest.raises(ValueError):
        compute_spectrogram(np.zeros(FS), FS, window="nope")
    with pytest.raises(ValueError):
        compute_spectrogram(np.zeros(FS), FS, overlap=1.0)


def test_noise_reduction_darkens_noise_only_region():
    rng = np.random.default_rng(0)
    t = np.arange(3 * FS) / FS
    clean = np.where(t > 0.5, 0.5 * np.sin(2 * np.pi * 440 * t), 0.0)
    noisy = clean + 0.1 * rng.standard_normal(len(t))
    cleaned = reduce_noise(noisy, FS, noise_seconds=0.5)
    _, tt, s_noisy = compute_spectrogram(noisy, FS, nperseg=512)
    _, _, s_clean = compute_spectrogram(cleaned, FS, nperseg=512)
    # compare noise-only frames, using the same colour reference for both
    quiet = tt < 0.45
    assert s_clean[:, quiet].mean() < s_noisy[:, quiet].mean() - 5
