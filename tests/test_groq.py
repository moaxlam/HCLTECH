import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

from assa.config import get_settings, reset_settings_cache
from assa.llm import LLMClient


def test_groq_config_auto_detection(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_123")
    monkeypatch.delenv("ASSA_LLM_PROVIDER", raising=False)
    reset_settings_cache()
    settings = get_settings()
    assert settings.groq_api_key == "gsk_test_key_123"
    assert settings.llm_provider == "groq"
    assert settings.groq_model == "llama-3.3-70b-versatile"
    reset_settings_cache()


def test_groq_health_checks(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setenv("ASSA_LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    reset_settings_cache()

    client = LLMClient()
    ok, detail = client.health()
    assert not ok
    assert "GROQ_API_KEY is not set" in detail

    # Health check with valid mock response
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_123")
    reset_settings_cache()
    client = LLMClient()

    mock_resp = io.BytesIO(json.dumps({"data": [{"id": "llama-3.3-70b-versatile"}]}).encode("utf-8"))
    with patch("urllib.request.urlopen", return_value=mock_resp):
        ok, detail = client.health()
        assert ok
        assert "Groq API" in detail

    # Health check with 401 Unauthorized
    http_error = urllib.error.HTTPError(
        url="https://api.groq.com/openai/v1/models",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=io.BytesIO(b'{"error": {"message": "Invalid API Key"}}'),
    )
    with patch("urllib.request.urlopen", side_effect=http_error):
        ok, detail = client.health()
        assert not ok
        assert "401 Unauthorized" in detail

    reset_settings_cache()


def test_groq_generate_success(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setenv("ASSA_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_valid_mock_key")
    reset_settings_cache()

    client = LLMClient()
    mock_payload = {
        "id": "chatcmpl-123",
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"category": "policy_fact"}',
                }
            }
        ],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 10,
        },
    }

    mock_resp = io.BytesIO(json.dumps(mock_payload).encode("utf-8"))
    with patch("urllib.request.urlopen", return_value=mock_resp):
        reply = client.generate("What is the policy?", json_mode=True)
        assert reply is not None
        assert reply.backend == "groq"
        assert reply.model == "llama-3.3-70b-versatile"
        assert reply.tokens == 35
        assert reply.text == '{"category": "policy_fact"}'

    reset_settings_cache()


def test_groq_json_mode_safeguard(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setenv("ASSA_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_key")
    reset_settings_cache()

    client = LLMClient()
    captured_request = {}

    def fake_urlopen(req, timeout=None):
        captured_request["body"] = json.loads(req.data.decode("utf-8"))
        captured_request["headers"] = dict(req.headers)
        return io.BytesIO(json.dumps({
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }).encode("utf-8"))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        # Prompt without the word 'json'
        client.generate("Give me an object", json_mode=True)

    assert "body" in captured_request
    messages = captured_request["body"]["messages"]
    # Check that 'json' instruction was automatically inserted for Groq compliance
    assert any("json" in m["content"].lower() for m in messages)
    assert captured_request["headers"]["Authorization"] == "Bearer gsk_mock_key"

    reset_settings_cache()
