"""LLM access.

Primary path: a local Ollama server (``llama3.1:8b`` or ``qwen2.5:7b-instruct`` by
default). The client uses the plain HTTP API through ``urllib`` so no extra
package is needed, and it asks for JSON output where the caller needs structure.

``MOCK_LLM=true`` disables the model entirely. Every node in the workflow has a
deterministic path that runs in that mode, which is what makes the pipeline
testable without a GPU: classification, tool planning, answer composition and
grounding verification all work with ``MOCK_LLM=true``.

A cloud model can be enabled **only** through ``ASSA_CLOUD_FALLBACK=true`` with
explicit endpoint settings, and is disclosed in the README.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .config import get_settings


@dataclass
class LLMReply:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    backend: str = "ollama"

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMStats:
    calls: int = 0
    tokens: int = 0
    model: str = ""
    backend: str = ""
    fallbacks: list[str] = field(default_factory=list)

    def record(self, reply: LLMReply) -> None:
        self.calls += 1
        self.tokens += reply.tokens
        self.model = reply.model
        self.backend = reply.backend


class LLMUnavailable(RuntimeError):
    pass


class LLMClient:
    """Thin wrapper that never raises into the workflow: callers get ``None``."""

    def __init__(self) -> None:
        self.settings = get_settings()
        active_model = (
            self.settings.groq_model
            if self.settings.llm_provider == "groq"
            else self.settings.ollama_model
        )
        self.stats = LLMStats(
            model=active_model,
            backend=self.settings.llm_provider,
        )

    # -- public -------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return not self.settings.mock_llm

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.0,
        timeout: int | None = None,
    ) -> LLMReply | None:
        if not self.enabled:
            return None
        timeout = timeout or self.settings.llm_timeout_s
        reply: LLMReply | None = None

        if self.settings.llm_provider == "groq":
            try:
                reply = self._groq(prompt, system, json_mode, temperature, timeout)
            except Exception as exc:
                self.stats.fallbacks.append(f"groq failed: {exc}")
                reply = self._cloud(prompt, system, json_mode, temperature, timeout)
        else:
            try:
                reply = self._ollama(prompt, system, json_mode, temperature, timeout)
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
                self.stats.fallbacks.append(f"ollama unavailable: {exc}")
                if self.settings.groq_api_key:
                    try:
                        reply = self._groq(prompt, system, json_mode, temperature, timeout)
                    except Exception as g_exc:
                        self.stats.fallbacks.append(f"groq fallback failed: {g_exc}")
                if reply is None:
                    reply = self._cloud(prompt, system, json_mode, temperature, timeout)

        if reply is None:
            return None
        self.stats.record(reply)
        return reply

    def generate_json(
        self, prompt: str, system: str | None = None, temperature: float = 0.0
    ) -> dict[str, Any] | None:
        reply = self.generate(prompt, system=system, json_mode=True, temperature=temperature)
        if reply is None:
            return None
        return _parse_json(reply.text)

    def health(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "MOCK_LLM=true (model not used)"

        if self.settings.llm_provider == "groq":
            if not self.settings.groq_api_key:
                return False, "GROQ_API_KEY is not set"
            try:
                request = urllib.request.Request(
                    f"{self.settings.groq_base_url.rstrip('/')}/models",
                    headers={
                        "Authorization": f"Bearer {self.settings.groq_api_key}",
                        "User-Agent": "ASSA/1.0",
                    },
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                models = [m.get("id", "") for m in payload.get("data", [])]
                if self.settings.groq_model and models and self.settings.groq_model not in models:
                    return True, f"{self.settings.groq_model} on Groq API (warning: model not found in account models)"
                return True, f"{self.settings.groq_model} via Groq API"
            except urllib.error.HTTPError as exc:
                if exc.code == 401:
                    return False, "Groq API error: Invalid API key (401 Unauthorized)"
                return False, f"Groq API error: HTTP {exc.code}"
            except Exception as exc:
                return False, f"Groq API unreachable at {self.settings.groq_base_url}: {exc}"

        try:
            request = urllib.request.Request(f"{self.settings.ollama_host}/api/tags")
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            names = [model.get("name", "") for model in payload.get("models", [])]
            if self.settings.ollama_model not in names:
                return False, f"{self.settings.ollama_model} not pulled (available: {', '.join(names) or 'none'})"
            return True, f"{self.settings.ollama_model} on {self.settings.ollama_host}"
        except Exception as exc:
            return False, f"Ollama unreachable at {self.settings.ollama_host}: {exc}"

    # -- backends -----------------------------------------------------------
    def _groq(
        self, prompt: str, system: str | None, json_mode: bool, temperature: float, timeout: int
    ) -> LLMReply | None:
        """Call Groq Cloud LLM via OpenAI-compatible endpoint."""
        if not self.settings.groq_api_key:
            self.stats.fallbacks.append("groq api key is not set")
            return None

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Groq requires 'json' to be present in messages when response_format is json_object
        if json_mode:
            has_json = any("json" in m["content"].lower() for m in messages)
            if not has_json:
                messages.insert(0, {"role": "system", "content": "You must respond with a valid JSON object."})

        payload: dict[str, Any] = {
            "model": self.settings.groq_model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        req_url = f"{self.settings.groq_base_url.rstrip('/')}/chat/completions"
        request = urllib.request.Request(
            req_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.groq_api_key}",
                "User-Agent": "ASSA/1.0",
            },
        )
        try:
            start_time = time.perf_counter()
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            latency_ms = int((time.perf_counter() - start_time) * 1000)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            self.stats.fallbacks.append(f"groq error {exc.code}: {err_body}")
            return None
        except Exception as exc:
            self.stats.fallbacks.append(f"groq request failed: {exc}")
            return None

        choices = body.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = body.get("usage", {})

        return LLMReply(
            text=str(content or ""),
            model=str(body.get("model", self.settings.groq_model)),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_ms=latency_ms,
            backend="groq",
        )

    def _ollama(
        self, prompt: str, system: str | None, json_mode: bool, temperature: float, timeout: int
    ) -> LLMReply:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        request = urllib.request.Request(
            f"{self.settings.ollama_host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return LLMReply(
            text=str(body.get("message", {}).get("content", "")),
            model=str(body.get("model", self.settings.ollama_model)),
            prompt_tokens=int(body.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(body.get("eval_count", 0) or 0),
            backend="ollama",
        )

    def _cloud(
        self, prompt: str, system: str | None, json_mode: bool, temperature: float, timeout: int
    ) -> LLMReply | None:
        """Optional, explicitly configured fallback. Disabled by default."""
        if not (
            self.settings.cloud_fallback_enabled
            and self.settings.cloud_base_url
            and self.settings.cloud_api_key
            and self.settings.cloud_model
        ):
            return None
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload: dict[str, Any] = {
            "model": self.settings.cloud_model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.settings.cloud_base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.cloud_api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            self.stats.fallbacks.append(f"cloud fallback failed: {exc}")
            return None
        self.stats.fallbacks.append("cloud fallback was used")
        return LLMReply(
            text=str(body.get("choices", [{}])[0].get("message", {}).get("content", "")),
            model=self.settings.cloud_model,
            prompt_tokens=int(body.get("usage", {}).get("prompt_tokens", 0) or 0),
            completion_tokens=int(body.get("usage", {}).get("completion_tokens", 0) or 0),
            backend="cloud-fallback",
        )


def _parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None
