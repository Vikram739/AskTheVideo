import pytest

from askthevideo import config


@pytest.fixture(autouse=True)
def no_env_file(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ENV_PATH", tmp_path / ".env")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)


def test_missing_api_key_raises():
    with pytest.raises(config.MissingApiKeyError):
        config.load_settings()


def test_blank_api_key_raises(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "   ")
    with pytest.raises(config.MissingApiKeyError):
        config.load_settings()


def test_default_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    settings = config.load_settings()
    assert settings.groq_api_key == "test-key"
    assert settings.groq_model == "openai/gpt-oss-20b"


def test_model_override(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    assert config.load_settings().groq_model == "llama-3.1-8b-instant"


def test_values_read_from_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env_file)
    assert config.load_settings().groq_api_key == "file-key"
