"""Headless UI tests (Streamlit AppTest): every screen renders, empty states are clear."""

import pytest
from streamlit.testing.v1 import AppTest

from src.config import load_config
from src.utils import PROJECT_ROOT
from src.video_io import validate_video_file

WEIGHTS = PROJECT_ROOT / "models" / "yolo11n.pt"


def _app(page: str | None = None, **state) -> AppTest:
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=90)
    for k, v in state.items():
        at.session_state[k] = v
    if page:
        at.session_state["page"] = page
    at.run()
    return at


def test_home_renders_without_exception():
    at = _app()
    assert not at.exception
    assert any("Choose a video" in s.value for s in at.subheader)
    assert any("System status" in m.value or "Compute device" in m.value for m in at.markdown)


@pytest.mark.parametrize("page,text", [
    ("Configure", "No video selected"),
    ("Process", "Nothing is being processed"),
    ("Results", "No results yet"),
])
def test_empty_states(page, text):
    at = _app(page)
    assert not at.exception
    assert any(text in i.value for i in at.info)


def test_navigation_buttons_from_empty_state():
    at = _app("Configure")
    at.button[0].click().run()
    assert at.session_state["page"] == "Home"


def test_configure_with_video(make_video):
    info = validate_video_file(make_video("ui.mp4", frames=20, size=(320, 240)), load_config().video)
    at = _app("Configure", video=info, video_name="ui.mp4", video_is_temp=False)
    assert not at.exception
    assert any("Counting line" in s.value for s in at.subheader)
    if not WEIGHTS.is_file() and not any((PROJECT_ROOT / "models").glob("*.pt")):
        # No model: start must be blocked with an explanation.
        assert any("valid detection model" in w.value for w in at.warning)


def test_configure_rejects_degenerate_line(make_video):
    info = validate_video_file(make_video("ui2.mp4", frames=20, size=(320, 240)), load_config().video)
    at = _app("Configure", video=info, video_name="ui2.mp4", video_is_temp=False,
              lx1=50.0, ly1=50.0, lx2=51.0, ly2=50.0)
    assert not at.exception
    assert any("at least" in w.value for w in at.warning)
    start = [b for b in at.button if b.label == "Start processing"][0]
    assert start.disabled


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="models/yolo11n.pt not present")
def test_start_processing_navigates_and_runs(make_video):
    info = validate_video_file(make_video("ui4.mp4", frames=15, size=(320, 240)), load_config().video)
    at = _app("Configure", video=info, video_name="ui4.mp4", video_is_temp=False)
    start = [b for b in at.button if b.label == "Start processing"][0]
    assert not start.disabled
    start.click().run()
    assert not at.exception
    assert at.session_state["page"] == "Process"
    job = at.session_state["job"]
    job.join(120)
    snap = job.snapshot()
    assert snap.status.value == "completed", snap.error
    assert snap.result.frames_processed == 15


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="models/yolo11n.pt not present")
def test_generic_model_warning_shown(make_video):
    info = validate_video_file(make_video("ui3.mp4", frames=20, size=(320, 240)), load_config().video)
    at = _app("Configure", video=info, video_name="ui3.mp4", video_is_temp=False)
    assert not at.exception
    assert any("generic COCO model" in w.value for w in at.warning)
