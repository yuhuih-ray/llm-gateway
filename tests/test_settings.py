from llm_gateway.config import get_settings


def test_settings_are_cached(monkeypatch, tmp_path):  # Isolated settings environment.
    get_settings.cache_clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "12")
    try:
        first = get_settings()
        monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "25")
        assert get_settings() is first
        assert get_settings().gemini_timeout_seconds == 12
    finally:
        get_settings.cache_clear()
