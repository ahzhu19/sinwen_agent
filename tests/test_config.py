"""Config and BaseLLM configuration tests."""

from core.config import Config
from core.llm import BaseLLM


def test_config_defaults_to_current_model() -> None:
    config = Config()

    assert config.default_model == "gpt-4o-mini"


def test_config_from_env_reads_llm_settings(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL_ID", "env-model")
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_TIMEOUT", "12")

    config = Config.from_env()

    assert config.default_model == "env-model"
    assert config.llm_api_key == "env-key"
    assert config.llm_base_url == "https://example.com/v1"
    assert config.llm_timeout == 12


def test_base_llm_uses_config_values() -> None:
    config = Config(
        default_model="config-model",
        llm_api_key="config-key",
        llm_base_url="https://example.com/v1",
        llm_timeout=7,
    )

    llm = BaseLLM(config=config)

    assert llm.model == "config-model"


def test_base_llm_explicit_arguments_override_config() -> None:
    config = Config(
        default_model="config-model",
        llm_api_key="config-key",
        llm_base_url="https://example.com/v1",
    )

    llm = BaseLLM(
        model="explicit-model",
        api_key="explicit-key",
        base_url="https://override.example.com/v1",
        config=config,
    )

    assert llm.model == "explicit-model"
