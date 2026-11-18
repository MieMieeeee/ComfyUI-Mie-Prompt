"""Ideogram 4 structured-caption prompt generator.

Expands a plain user idea into validated, compact Ideogram 4 JSON via LLM.
Formatting and schema validation are built in — invalid LLM output raises ``ValueError``,
consistent with other MieNodes prompt generator nodes.
"""

from __future__ import annotations

import hashlib

from mie_llm.utils import mie_log

from mie_prompt.ideogram4_prompts import COMPOSITION_MODES, COMPOSITION_MODE_TOOLTIPS, build_ideogram4_messages, resolve_aspect_ratio


MY_CATEGORY = "\ud83d\udc11 MieNodes/\ud83d\udc11 Prompt Generator"

_MAX_TOKENS = 16384


from mie_prompt.ideogram4_prompt_formatter import format_ideogram4_caption


def postprocess_caption(raw_text: str) -> str:
    """Parse and validate LLM output into compact JSON. Raises on parse/validation failure."""
    prompt, log = format_ideogram4_caption(raw_text)
    if log and log != "OK: no fixes needed":
        mie_log(f"[Ideogram4PromptGenerator] {log}")
    return prompt


class Ideogram4PromptEnhancer:
    """Expand a user idea into Ideogram 4 JSON via LLM."""

    def __init__(
        self,
        llm_service_connector,
        *,
        temperature: float = 1.0,
        timeout: int = 120,
        composition_mode: str = "simple",
    ):
        self.llm = llm_service_connector
        self.temperature = temperature
        self.timeout = timeout
        self.composition_mode = composition_mode

    def _invoke(self, messages: list[dict], seed=None) -> str:
        prev_timeout = getattr(self.llm, "timeout", None)
        try:
            if prev_timeout is not None:
                self.llm.timeout = self.timeout
            return self.llm.invoke(
                messages,
                seed=seed,
                temperature=self.temperature,
                max_tokens=_MAX_TOKENS,
            )
        finally:
            if prev_timeout is not None:
                self.llm.timeout = prev_timeout

    def __call__(
        self,
        user_prompt: str,
        aspect_ratio: str = "1:1",
        seed=None,
    ) -> str:
        prompt = (user_prompt or "").strip()
        if not prompt:
            raise ValueError("user_prompt is empty")

        ar = resolve_aspect_ratio(aspect_ratio)
        messages = build_ideogram4_messages(
            prompt, ar, composition_mode=self.composition_mode
        )
        raw = self._invoke(messages, seed=seed)
        return postprocess_caption(raw)
