"""Anthropic inference facade for cross-evaluator validation (Phase 5).

Mirrors `InferenceFacade.create_chat_completion_async` so `main_benchmark_rag_result.py`
can swap evaluators by model-name prefix without changing call sites. Only the
non-streaming async path is implemented because that is all the benchmark needs.
"""
import re
from typing import List, Dict, Optional
from logging import getLogger

from constants import ANTHROPIC_API_KEY
from data_classes.inference_parameters import InferenceParameters


logger = getLogger(__name__)


class AnthropicInferenceFacade:
    """Minimal Anthropic facade with the same async-completion contract as InferenceFacade."""

    def __init__(self, model: str, base_url: Optional[str] = None):
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is empty. Add it to <repo_root>/.env "
                "(KEY=VALUE format) or export it in your shell."
            )

        from anthropic import AsyncAnthropic

        self.model = model
        # max_retries=10 mirrors the OpenAI client patch — absorbs brief 429
        # storms without crashing long benchmark sweeps.
        client_kwargs = {"api_key": ANTHROPIC_API_KEY, "max_retries": 10}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.async_client = AsyncAnthropic(**client_kwargs)

        logger.info(f"Initialized AnthropicInferenceFacade with model {model}")

    @staticmethod
    def _split_system_and_messages(messages: List[Dict[str, str]]):
        """Anthropic takes `system` as a top-level parameter, not as a role in messages."""
        system_parts: List[str] = []
        chat: List[Dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if role == "system":
                if content:
                    system_parts.append(content)
            elif role in ("user", "assistant"):
                chat.append({"role": role, "content": content})
            else:
                # Unknown role — fall through as user content to stay permissive.
                chat.append({"role": "user", "content": content})
        system_text = "\n\n".join(system_parts) if system_parts else None
        return system_text, chat

    # Minimum max_tokens floor for Claude. Sonnet 4.6 routinely emits a short
    # preamble even when told "JSON only", and a 100-token cap from
    # OpenAI-tuned callers chops off the response before any JSON is produced.
    # 1024 is safe and cheap: actual eval responses are ~200 tokens.
    _CLAUDE_MAX_TOKENS_FLOOR = 1024

    @staticmethod
    def _extract_trailing_json(text: str) -> str:
        """Pull the last balanced `{...}` block out of `text`.

        Claude tends to emit a brief preamble ("The RAG system answer is...")
        followed by the requested JSON. The benchmark caller does
        `json.loads(response.strip())`, which fails on prose. We strip the
        prose by returning only the last top-level `{...}` block.

        Returns the unchanged text if no balanced block is found — the caller's
        parse will fail with the same error it would have had originally.
        """
        if not text or "{" not in text:
            return text
        last_open = text.rfind("{")
        depth = 0
        for i in range(last_open, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[last_open : i + 1]
        # Unbalanced — try the broadest greedy regex match as a last resort.
        m = re.search(r"\{[^{}]*\}", text)
        return m.group(0) if m else text

    async def create_chat_completion_async(
        self,
        messages: List[Dict[str, str]],
        *,
        inference_parameters: Optional[InferenceParameters] = None,
        **kwargs,
    ) -> str:
        """Generate a complete response. Returns the text content as a single string.

        When `inference_parameters.response_format == {"type": "json_object"}`
        we floor `max_tokens` at 1024 (Claude tends to emit preamble before
        the JSON, and small token caps tuned for OpenAI's strict-JSON mode
        chop the response off before any JSON appears) and then post-extract
        the trailing balanced `{...}` block before returning, so the caller's
        `json.loads` works without modification.
        """
        if inference_parameters and inference_parameters.system_prompt:
            messages = [m for m in messages if m.get("role", "").lower() != "system"]
            messages = [
                {"role": "system", "content": inference_parameters.system_prompt}
            ] + messages

        system_text, chat = self._split_system_and_messages(messages)

        if inference_parameters:
            temperature = (
                inference_parameters.temperature
                if inference_parameters.temperature is not None
                else 0.0
            )
            max_tokens = (
                inference_parameters.max_tokens
                if inference_parameters.max_tokens is not None
                else 1024
            )
            top_p = inference_parameters.top_p
            stop_words = inference_parameters.stop_words
            response_format = inference_parameters.response_format
        else:
            temperature = kwargs.get("temperature", 0.0)
            max_tokens = kwargs.get("max_tokens", 1024)
            top_p = kwargs.get("top_p", None)
            stop_words = kwargs.get("stop_words", None)
            response_format = kwargs.get("response_format", None)

        wants_json = (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
        )
        if wants_json and max_tokens < self._CLAUDE_MAX_TOKENS_FLOOR:
            max_tokens = self._CLAUDE_MAX_TOKENS_FLOOR

        api_params: Dict = {
            "model": self.model,
            "messages": chat,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_text is not None:
            api_params["system"] = system_text
        if top_p is not None:
            api_params["top_p"] = top_p
        if stop_words:
            api_params["stop_sequences"] = stop_words

        response = await self.async_client.messages.create(**api_params)

        # Anthropic returns a list of content blocks; we want the concatenated text.
        parts = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                parts.append(getattr(block, "text", ""))
        text = "".join(parts)

        if wants_json:
            text = self._extract_trailing_json(text)

        return text
