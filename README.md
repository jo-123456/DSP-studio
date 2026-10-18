# DSP Studio

**An interactive digital signal processing and analysis platform.**
DSP Mini Project, Department of Electronics and Communication Engineering, LBS College of Engineering, Kasaragod.

**Live app:** https://dsp-studio-dxuuk29ltkxyjveinb7cw5.streamlit.app/

> The app runs on Streamlit Community Cloud's free tier. If it has been idle it may be asleep: open the link and click the wake-up button, then allow a minute for it to start.

DSP Studio turns DSP theory into something you can see and hear. Generate or import a signal, process it, and watch the time-domain waveform, spectrum and spectrogram change as you move the controls.

## Features

| Tab | What it does |
|---|---|
| **Signal & Spectrum** | Generate standard test waveforms, or upload a WAV file or a CSV of sensor data (choose the column and sample rate, optional DC-offset removal). Shows the waveform, FFT spectrum and the dominant frequency. |
| **Filtering** | FIR (window method: Hamming, Hanning, Blackman) and IIR (Butterworth) low-pass, high-pass and band-pass filters. Adjustable cutoff, taps or order, optional zero-phase filtering. Shows before/after waveforms and spectra plus the filter's own frequency response. |
| **Spectrogram** | Time-frequency view with adjustable window length, overlap and window type, and a linear-chirp demo. Displays the resulting frequency resolution and time step. |
| **Operations** | **Convolution** (moving average, echo, differentiator), **Correlation** (cross-correlation delay estimation, autocorrelation period detection), **Modulation** (AM and FM with channel noise, receiver low-pass and recovery error) and **Windowing** (spectral-leakage comparison of window functions). |
| **Voice Demo** | Upload a WAV voice clip. Apply a telephone effect (300-3400 Hz band-pass) or spectral-subtraction noise reduction, compare before/after audio, spectra and spectrograms, and download the result. With test noise enabled, it reports SNR before, after and the improvement. |

## Project structure

```
dsp-studio/
├── app.py                  Streamlit interface (presentation layer)
├── ui_theme.py             Dark "lab instrument" theme and chart styling
├── .streamlit/config.toml  Theme colours
├── signals/                Signal-processing core (no Streamlit code)
│   ├── generator.py        Test-signal generation
│   ├── analysis.py         FFT and dominant-frequency analysis
│   ├── filters.py          FIR / IIR design and application
│   ├── spectrogram.py      Short-time spectrogram
│   ├── operations.py       Convolution, correlation, AM/FM modulation
│   ├── windowing.py        Window functions and spectral leakage
│   ├── voice.py            Telephone effect, noise reduction, WAV helpers
│   └── loaders.py          CSV sensor-data import
├── tests/                  Automated tests (pytest)
├── requirements.txt        Runtime dependencies
└── requirements-dev.txt    Runtime + test dependencies
```

The signal-processing core is kept separate from the interface, so new algorithms and visualisations can be added without disturbing existing ones.

## Run locally

Requires Python 3.11.

```
git clone https://github.com/jo-123456/DSP-studio.git
cd DSP-studio
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Tests

```
pip install -r requirements-dev.txt
python -m pytest tests -v
```

The suite (61 tests) covers:
- signal generation and FFT accuracy,
- FIR/IIR filter attenuation and input validation,
- spectrogram frequency and time accuracy,
- convolution, correlation and delay estimation,
- AM/FM modulation round trips, including with noise,
- window leakage and amplitude recovery,
- CSV import,
- end-to-end tests that run the whole Streamlit app and exercise every tab.

## Using the Voice Demo

- For noise reduction, start the clip with about half a second of silence. The algorithm estimates the noise from that opening segment.
- Keep "Add test noise" ticked to measure SNR improvement on a clean recording. Untick it to process a real noisy recording as it is.
- On the hosted free tier, short clips (under about 20 seconds, ideally 16 kHz mono) work best.

## Technologies

Python, NumPy, SciPy, Plotly, pandas and Streamlit, tested with pytest and deployed on Streamlit Community Cloud. Every push to `main` redeploys the live app.

## Future work

- Real-time, hardware-in-the-loop processing
- Adaptive filtering demonstrations
- Additional visualisation modes