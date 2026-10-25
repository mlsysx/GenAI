"""Provider-neutral transport; no credentials are printed or stored in results.

Reference REST layouts: Hugging Face Inference Providers chat completion,
Google Gemini generateContent, and Ollama /api/chat (see PROVIDERS.md).
No backend is called on import. No model fallback or automatic retry occurs.
"""
from __future__ import annotations
import base64
from dataclasses import dataclass, field
import mimetypes
import os
from pathlib import Path
import time
from urllib.parse import quote, urlparse
import requests


class ModelError(RuntimeError):
    pass


@dataclass
class Reply:
    text: str
    model: str
    elapsed_seconds: float = 0.0
    usage: dict = field(default_factory=lambda: {"input_tokens": None, "output_tokens": None})
    finish_reason: str | None = None
    is_mock: bool = False


def image_payload(path: str | Path) -> tuple[str, str]:
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError("Use PNG, JPEG, or WebP images.")
    if path.stat().st_size > 8_000_000:
        raise ValueError("Resize images above 8 MB before sending them.")
    return mime, base64.b64encode(path.read_bytes()).decode("ascii")


class ChatBackend:
    """One configured model. Use a second instance for a separate VLM.

    Model IDs must be explicit; model hosting/access is not inferred from its name.
    ``enable_live`` must be deliberately enabled. Unknown capabilities are tested
    with a small smoke test, never silently replaced with another model.
    """
    def __init__(self, provider: str, model: str, *, enable_live: bool = False,
                 temperature: float = 0.0, max_output_tokens: int = 800,
                 timeout_seconds: float = 45.0, base_url: str | None = None,
                 api_key_env: str | None = None, json_mode: bool = True):
        if provider not in {"huggingface", "gemini", "ollama"}:
            raise ValueError("provider must be huggingface, gemini, or ollama")
        if not model or not model.strip():
            raise ValueError("Specify an exact accessible model ID in config.json.")
        if max_output_tokens < 1 or not 0 < timeout_seconds <= 600:
            raise ValueError("Invalid output-token or timeout setting.")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        self.provider, self.model = provider, model
        self.enable_live = enable_live
        self.temperature, self.max_output_tokens = temperature, max_output_tokens
        self.timeout_seconds, self.json_mode = timeout_seconds, json_mode
        self.api_key_env = api_key_env or {
            "huggingface": "HF_TOKEN", "gemini": "GEMINI_API_KEY", "ollama": ""
        }[provider]
        defaults = {
            "huggingface": "https://router.huggingface.co/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta",
            "ollama": "http://localhost:11434",
        }
        self.base_url = (base_url or defaults[provider]).rstrip("/")
        u = urlparse(self.base_url)
        if u.username or u.password or u.query or u.fragment:
            raise ValueError("Do not put credentials, query strings, or fragments in base_url.")
        if provider == "ollama":
            if u.hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("This starter's local backend only permits loopback hosts.")
        elif self.base_url != defaults[provider]:
            raise ValueError("Hosted adapters use their official endpoint; add a reviewed adapter for others.")

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "base_url": self.base_url,
                "temperature": self.temperature, "max_output_tokens": self.max_output_tokens,
                "timeout_seconds": self.timeout_seconds, "json_mode": self.json_mode}

    def generate(self, messages: list[dict], *, images: list[str | Path] | None = None,
                 json_mode: bool | None = None) -> Reply:
        if not self.enable_live:
            raise ModelError("Live calls are disabled. Configure a model and enable_live explicitly.")
        if not messages or any(m.get("role") not in {"system", "user", "assistant"}
                               or not isinstance(m.get("content"), str) for m in messages):
            raise ValueError("Messages must have a system/user/assistant role and string content.")
        images = images or []
        if images and messages[-1]["role"] != "user":
            raise ValueError("Images attach to the final user message.")
        assets = [image_payload(p) for p in images]
        want_json = self.json_mode if json_mode is None else json_mode
        headers = {"Content-Type": "application/json"}
        if self.provider != "ollama":
            key = os.environ.get(self.api_key_env, "")
            if not key:
                raise ModelError(f"Set the {self.api_key_env} environment variable or notebook secret.")
        if self.provider == "huggingface":
            url = self.base_url + "/chat/completions"
            headers["Authorization"] = "Bearer " + key
            mm = [dict(m) for m in messages]
            if assets:
                mm[-1]["content"] = [{"type": "text", "text": mm[-1]["content"]}] + [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
                    for mime, data in assets]
            body = {"model": self.model, "messages": mm, "stream": False,
                    "temperature": self.temperature, "max_tokens": self.max_output_tokens}
            if want_json:
                body["response_format"] = {"type": "json_object"}
        elif self.provider == "gemini":
            model = self.model.removeprefix("models/")
            url = self.base_url + "/models/" + quote(model, safe="-._") + ":generateContent"
            headers["x-goog-api-key"] = key  # Never put a key in a URL.
            body = {"contents": [], "generationConfig": {
                "temperature": self.temperature, "maxOutputTokens": self.max_output_tokens}}
            instructions = [m["content"] for m in messages if m["role"] == "system"]
            if instructions:
                body["systemInstruction"] = {"parts": [{"text": "\n\n".join(instructions)}]}
            for m in messages:
                if m["role"] != "system":
                    body["contents"].append({"role": "model" if m["role"] == "assistant" else "user",
                                             "parts": [{"text": m["content"]}]})
            if not body["contents"]:
                raise ValueError("Gemini requires a non-system message.")
            body["contents"][-1]["parts"] += [
                {"inlineData": {"mimeType": mime, "data": data}} for mime, data in assets]
            if want_json:
                body["generationConfig"]["responseMimeType"] = "application/json"
        else:
            url = self.base_url + "/api/chat"
            mm = [dict(m) for m in messages]
            if assets:
                mm[-1]["images"] = [data for _, data in assets]
            body = {"model": self.model, "messages": mm, "stream": False,
                    "options": {"temperature": self.temperature, "num_predict": self.max_output_tokens}}
            if want_json:
                body["format"] = "json"
        start = time.perf_counter()
        try:
            response = requests.post(url, headers=headers, json=body,
                                     timeout=self.timeout_seconds, allow_redirects=False)
        except requests.RequestException:
            # Raw request exceptions can contain URLs/credentials; do not leak them.
            raise ModelError(f"{self.provider} request failed or timed out; no retry was made.") from None
        if response.status_code != 200:
            raise ModelError(f"{self.provider} HTTP {response.status_code}; check access, quota, model ID, and capabilities.")
        try:
            data = response.json()
            if self.provider == "huggingface":
                choice = data["choices"][0]
                text = choice["message"]["content"]
                usage = data.get("usage") or {}
                model, finish = data.get("model", self.model), choice.get("finish_reason")
                counts = {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}
            elif self.provider == "gemini":
                candidate = data["candidates"][0]
                text = "".join(p.get("text", "") for p in candidate["content"]["parts"]
                               if not p.get("thought", False))
                usage = data.get("usageMetadata", {})
                model, finish = data.get("modelVersion", self.model), candidate.get("finishReason")
                counts = {"input_tokens": usage.get("promptTokenCount"),
                          "output_tokens": usage.get("candidatesTokenCount"),
                          "thinking_tokens": usage.get("thoughtsTokenCount")}
            else:
                text = data["message"]["content"]
                model, finish = data.get("model", self.model), data.get("done_reason")
                counts = {"input_tokens": data.get("prompt_eval_count"), "output_tokens": data.get("eval_count")}
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty response")
        except (ValueError, TypeError, KeyError, IndexError):
            raise ModelError(f"{self.provider} returned no usable text; inspect safety/length/capability settings without exposing secrets.") from None
        return Reply(text, model, time.perf_counter() - start, counts, finish)


class ScriptedBackend:
    """OFFLINE TEST DOUBLE, not a language model and not empirical model evidence."""
    model = "OFFLINE_SCRIPTED_TEST_DOUBLE"
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
    def describe(self):
        return {"provider": "scripted", "model": self.model, "is_mock": True}
    def generate(self, messages, *, images=None, json_mode=None):
        self.calls.append({"messages": messages, "images": [str(p) for p in (images or [])]})
        try:
            text = next(self.responses)
        except StopIteration:
            raise ModelError("Scripted test double ran out of responses.") from None
        if isinstance(text, Exception):
            raise text
        return Reply(text=text, model=self.model, is_mock=True)
