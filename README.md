# DSP Studio — Part 1: Signal Input & Frequency Analysis

This is the first working slice of DSP Studio: generate or upload a signal,
view its waveform, and see its frequency spectrum via FFT.

## Setup

1. Make sure you have Python 3.10+ installed.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Run

```
streamlit run app.py
```

This opens the app in your browser (usually at http://localhost:8501).

## What's included

- `signals/generator.py` — generates sine, square, sawtooth, and triangle waves,
  plus white noise.
- `signals/analysis.py` — computes the FFT and returns a one-sided magnitude
  spectrum; also finds the dominant frequency.
- `app.py` — the Streamlit UI: pick a waveform (or upload a WAV file), and see
  the time-domain plot and frequency spectrum update live.
- `tests/test_part1.py` — basic checks that the generator and FFT are correct
  (e.g., a 50 Hz sine wave is correctly detected at 50 Hz).

## Next steps (Part 2 & Part 3, by teammates)

- Part 2 will add filtering (FIR/IIR) and the voice filtering module, likely
  reusing `signals/analysis.py` for before/after spectrum comparisons.
- Part 3 will add the spectrogram view and unify everything into one
  polished interface — this `app.py` is meant to be extended with more
  pages/tabs rather than replaced.

## Folder structure

```
dsp-studio/
├── signals/
│   ├── generator.py
│   └── analysis.py
├── tests/
│   └── test_part1.py
├── app.py
├── requirements.txt
└── README.md
```
