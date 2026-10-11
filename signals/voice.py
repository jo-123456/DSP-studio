"""Voice demos: telephone effect, noise reduction, WAV helpers (Week 2)."""
import io
import numpy as np
from scipy import signal
from scipy.io import wavfile
from .filters import apply_filter


def load_wav(file_or_path):
    """Read a WAV (path or uploaded file) -> (mono float array in [-1,1], fs)."""
    fs, data = wavfile.read(file_or_path)
    if data.dtype.kind in "iu":
        data = data.astype(float) / np.iinfo(data.dtype).max
        if data.dtype.kind == "u":
            data = data * 2 - 1
    data = data.astype(float)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, fs


def wav_bytes(x, fs):
    """Float array -> 16-bit WAV bytes (for st.download_button)."""
    buf = io.BytesIO()
    wavfile.write(buf, fs, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return buf.getvalue()


def telephone_effect(x, fs, low=300.0, high=3400.0, order=4):
    """Band-limit to the classic telephone band."""
    high = min(high, 0.45 * fs)
    return apply_filter(x, fs, "iir", "bandpass", (low, high), order=order)


def reduce_noise(x, fs, noise_seconds=0.5, nperseg=512, over_subtract=1.5, floor=0.05):
    """Spectral subtraction. Assumes the first `noise_seconds` is noise only."""
    x = np.asarray(x, dtype=float)
    f, t, Z = signal.stft(x, fs, nperseg=nperseg)
    hop = nperseg // 2
    n_frames = max(1, int(noise_seconds * fs / hop))
    mag, phase = np.abs(Z), np.angle(Z)
    noise = mag[:, :n_frames].mean(axis=1, keepdims=True)
    clean = np.maximum(mag - over_subtract * noise, floor * mag)
    _, y = signal.istft(clean * np.exp(1j * phase), fs, nperseg=nperseg)
    y = y[:len(x)]
    return np.pad(y, (0, len(x) - len(y)))


def snr_db(reference, test):
    """SNR of `test` against a clean `reference`, in dB."""
    n = min(len(reference), len(test))
    ref, tst = np.asarray(reference[:n]), np.asarray(test[:n])
    noise = tst - ref
    return 10 * np.log10(np.sum(ref**2) / max(np.sum(noise**2), 1e-20))
