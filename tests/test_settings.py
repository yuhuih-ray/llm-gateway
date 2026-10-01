from llm_gateway.config import get_settings


def test_settings_are_cached(monkeypatch, tmp_path):  # Isolated settings environment.
    get_settings.cache_clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "first")
    try:
        first = get_settings()
        monkeypatch.setenv("DATABASE_URL", "second")
        assert get_settings() is first
        assert get_settings().database_url == "first"
    finally:
        get_settings.cache_clear()
