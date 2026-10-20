"""Provider-neutral chat interface used by the agent loop.

Backends
  openai_compat  any OpenAI-compatible server (vLLM, Ollama, llama.cpp, TGI). This is how the
                 open-weight models with early knowledge cutoffs required by the leakage
                 mitigation are served.
  anthropic      Claude via the official Anthropic SDK (late cutoff: reported as an upper bound).
  mock           scripted responses for tests and dry runs.

The agent keeps a neutral transcript:
  {"role": "user", "content": str}
  {"role": "assistant", "text": str, "tool_calls": [ToolCall], "raw": <backend payload>}
  {"role": "tool", "results": [{"id", "name", "content", "is_error"}]}
and each backend converts it to its wire format. Assistant turns carry the backend's raw
payload so provider-specific blocks (e.g. Claude thinking blocks) are replayed unchanged.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.calls += other.calls

    def cost(self, price_in: float, price_out: float) -> float:
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1e6

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""
    raw: Any = None
    model: str = ""


class LLMClient:
    supports_native_tools = True

    def __init__(self, profile: dict):
        self.profile = profile
        self.model = profile["model"]
        self.max_tokens = int(profile.get("max_tokens", 4096))

    def chat(self, system: str, messages: list[dict], tools: list[dict] | None = None) -> AssistantTurn:
        raise NotImplementedError


# ---------------------------------------------------------------------- OpenAI-compatible
class OpenAICompatClient(LLMClient):
    def __init__(self, profile: dict):
        super().__init__(profile)
        from openai import OpenAI

        key = os.environ.get(profile.get("api_key_env", "OPENAI_API_KEY"), "") or "not-needed"
        # A stalled local server must not hold a run hostage: fail the request and let the SDK
        # retry it rather than waiting out a ten-minute default.
        self.client = OpenAI(base_url=profile.get("base_url"), api_key=key,
                             timeout=float(profile.get("timeout", 240)),
                             max_retries=int(profile.get("max_retries", 3)))
        self.temperature = float(profile.get("temperature", 0.0))
        self.seed = profile.get("seed")

    @staticmethod
    def _tool_spec(tool: dict) -> dict:
        return {"type": "function", "function": {"name": tool["name"],
                                                 "description": tool["description"],
                                                 "parameters": tool["input_schema"]}}

    def _wire(self, system: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg: dict = {"role": "assistant", "content": m.get("text") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": c.id, "type": "function",
                                          "function": {"name": c.name,
                                                       "arguments": json.dumps(c.arguments)}}
                                         for c in m["tool_calls"]]
                out.append(msg)
            elif m["role"] == "tool":
                for r in m["results"]:
                    out.append({"role": "tool", "tool_call_id": r["id"], "content": r["content"]})
        # Some chat templates require strict user/assistant alternation.
        merged: list[dict] = []
        for msg in out:
            if merged and msg["role"] == "user" and merged[-1]["role"] == "user":
                merged[-1] = {"role": "user",
                              "content": merged[-1]["content"] + "\n\n" + msg["content"]}
            else:
                merged.append(msg)
        return merged

    def chat(self, system, messages, tools=None) -> AssistantTurn:
        kwargs: dict = {"model": self.model, "messages": self._wire(system, messages),
                        "max_tokens": self.max_tokens, "temperature": self.temperature}
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        if tools:
            kwargs["tools"] = [self._tool_spec(t) for t in tools]
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        calls = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__invalid_json__": tc.function.arguments}
            calls.append(ToolCall(tc.id or uuid.uuid4().hex[:12], tc.function.name, args))
        u = resp.usage
        usage = Usage(getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0,
                      0, 1)
        return AssistantTurn(choice.message.content or "", calls, usage,
                             choice.finish_reason or "", None, resp.model or self.model)


# ---------------------------------------------------------------------- Anthropic
class AnthropicClient(LLMClient):
    def __init__(self, profile: dict):
        super().__init__(profile)
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=int(profile.get("max_retries", 4)))
        self.effort = profile.get("effort")

    def _wire(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                # Replay the exact content blocks (thinking + text + tool_use).
                out.append({"role": "assistant", "content": m["raw"]})
            elif m["role"] == "tool":
                out.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"],
                     **({"is_error": True} if r.get("is_error") else {})} for r in m["results"]]})
        # Merge consecutive user turns (tool results followed by an instruction) into one.
        merged: list[dict] = []
        for msg in out:
            if merged and msg["role"] == "user" and merged[-1]["role"] == "user":
                prev = merged[-1]["content"]
                prev = prev if isinstance(prev, list) else [{"type": "text", "text": prev}]
                cur = msg["content"]
                cur = cur if isinstance(cur, list) else [{"type": "text", "text": cur}]
                merged[-1] = {"role": "user", "content": prev + cur}
            else:
                merged.append(msg)
        return merged

    def chat(self, system, messages, tools=None) -> AssistantTurn:
        kwargs: dict = {"model": self.model, "max_tokens": self.max_tokens,
                        "system": system, "messages": self._wire(messages)}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if tools:
            kwargs["tools"] = [{"name": t["name"], "description": t["description"],
                                "input_schema": t["input_schema"]} for t in tools]
        # Streaming avoids HTTP timeouts on long agent turns.
        with self.client.messages.stream(**kwargs) as stream:
            resp = stream.get_final_message()
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        u = resp.usage
        usage = Usage(u.input_tokens + (getattr(u, "cache_creation_input_tokens", 0) or 0),
                      u.output_tokens, getattr(u, "cache_read_input_tokens", 0) or 0, 1)
        raw = [b.model_dump(exclude_none=True) for b in resp.content]
        return AssistantTurn(text, calls, usage, resp.stop_reason or "", raw, resp.model)


# ---------------------------------------------------------------------- Mock
class MockClient(LLMClient):
    """Replays scripted turns. A script entry is either an AssistantTurn, a string (plain text
    reply) or a callable(system, messages, tools) -> AssistantTurn | str."""

    def __init__(self, profile: dict, script: list | Callable | None = None):
        super().__init__({"model": "mock", **profile})
        self.script = script if script is not None else profile.get("script", [])
        self._i = 0
        self._lock = threading.Lock()
        self.supports_native_tools = bool(profile.get("native_tools", True))

    def chat(self, system, messages, tools=None) -> AssistantTurn:
        with self._lock:
            if callable(self.script):
                item = self.script(system, messages, tools)
            else:
                if self._i >= len(self.script):
                    item = "[]"
                else:
                    item = self.script[self._i]
                    self._i += 1
        if callable(item):
            item = item(system, messages, tools)
        if isinstance(item, str):
            item = AssistantTurn(item)
        item.usage = Usage(len(system) // 4 + sum(len(str(m)) for m in messages) // 4,
                           len(item.text) // 4 + 1, 0, 1)
        return item


def make_client(profile: dict, **kw: Any) -> LLMClient:
    backend = profile.get("backend", "openai_compat")
    if backend == "openai_compat":
        return OpenAICompatClient(profile)
    if backend == "anthropic":
        return AnthropicClient(profile)
    if backend == "mock":
        return MockClient(profile, **kw)
    raise ValueError(f"Unknown backend: {backend}")
