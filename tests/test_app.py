"""End-to-end smoke tests: run the real Streamlit app headlessly and drive its widgets."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def fresh():
    return AppTest.from_file(APP, default_timeout=60).run()


def widget(items, label):
    return next(w for w in items if w.label == label)


def assert_clean(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_app_starts_with_five_tabs():
    at = fresh()
    assert_clean(at)
    assert [t.label for t in at.tabs] == [
        "Signal & Spectrum", "Filtering", "Spectrogram", "Operations", "Voice Demo"]


def test_default_sine_reports_50hz():
    at = fresh()
    peak = widget(at.metric, "Dominant frequency detected")
    assert peak.value == "50.0 Hz"


@pytest.mark.parametrize("family", ["FIR", "IIR"])
@pytest.mark.parametrize("ftype", ["lowpass", "highpass", "bandpass"])
def test_every_filter_combination_runs(family, ftype):
    at = fresh()
    widget(at.radio, "Filter family").set_value(family).run()
    widget(at.selectbox, "Filter type").set_value(ftype).run()
    assert_clean(at)


def test_highpass_changes_dominant_frequency():
    at = fresh()
    widget(at.selectbox, "Filter type").set_value("highpass").run()
    widget(at.slider, "Cutoff (Hz)").set_value(300).run()
    assert_clean(at)
    # a 50 Hz sine through a 300 Hz high-pass must not still read "50 Hz"
    filtered = widget(at.metric, "Dominant frequency (filtered)")
    assert filtered.value != "50.0 Hz"


@pytest.mark.parametrize("win", [128, 256, 512, 1024])
def test_spectrogram_window_sizes(win):
    at = fresh()
    widget(at.select_slider, "Window length (samples)").set_value(win).run()
    assert_clean(at)


def test_spectrogram_chirp_demo():
    at = fresh()
    widget(at.radio, "Signal").set_value(
        "Linear chirp demo (200 Hz to near Nyquist)").run()
    assert_clean(at)


def test_upload_mode_without_file_shows_info_not_crash():
    at = fresh()
    widget(at.radio, "Signal source").set_value("Upload").run()
    assert_clean(at)
    assert at.info  # the "upload a WAV" hint is shown


def choose_operation(name):
    at = fresh()
    widget(at.radio, "Operation").set_value(name).run()
    assert_clean(at)
    return at


@pytest.mark.parametrize("kernel", ["Moving average", "Echo", "Differentiator"])
def test_convolution_kernels(kernel):
    at = choose_operation("Convolution")
    widget(at.selectbox, "Kernel").set_value(kernel).run()
    assert_clean(at)


def test_cross_correlation_finds_the_true_delay():
    at = choose_operation("Correlation")
    true = widget(at.metric, "True delay").value
    assert widget(at.metric, "Estimated delay").value == true


def test_autocorrelation_reports_period_of_default_sine():
    at = choose_operation("Correlation")
    widget(at.radio, "Correlation type").set_value("Autocorrelation").run()
    assert_clean(at)
    assert widget(at.metric, "Implied frequency").value == "50.0 Hz"


@pytest.mark.parametrize("scheme", ["AM", "FM"])
def test_modulation_schemes_recover_the_message(scheme):
    at = choose_operation("Modulation")
    widget(at.radio, "Modulation scheme").set_value(scheme).run()
    assert_clean(at)
    err = widget(at.metric, "Recovery error (RMS, % of message)").value
    assert float(err.rstrip(" %")) < 15


def test_windowing_demo_runs():
    at = choose_operation("Windowing")
    assert_clean(at)
