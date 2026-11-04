"""Headless UI smoke test: the Streamlit app renders without exceptions."""

from streamlit.testing.v1 import AppTest

from src.utils import PROJECT_ROOT


def _run(page: str | None = None) -> AppTest:
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)
    at.run()
    if page:
        at.sidebar.radio[0].set_value(page).run()
    return at


def test_home_renders_without_exception():
    at = _run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Brick" in text and "System status" in " ".join(s.value for s in at.subheader)


def test_missing_weights_shows_helpful_warning():
    at = _run()
    # Repository ships without weights, so a guidance warning must be shown.
    assert any("Model weights required" in w.value for w in at.warning)


def test_unimplemented_screens_are_labelled_honestly():
    for page in ("Configure video", "Process", "Results"):
        at = _run(page)
        assert not at.exception
        assert any("not implemented yet" in i.value for i in at.info)
