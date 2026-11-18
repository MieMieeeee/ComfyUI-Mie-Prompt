"""Krea-2 prompt-enhancer ComfyUI node.

Expands a plain user idea into a single-paragraph, Krea-2-flavoured
text-to-image prompt by sending the upstream ``expansion.txt`` system
prompt to the configured ``LLMServiceConnector``. No images, no JSON
contract -- the LLM emits one cohesive paragraph per upstream rule 6.

Sits next to ``Ideogram4PromptGenerator`` and ``BerniniPromptGenerator``
in the same Prompt Generator category and follows the same I/O shape:
single STRING output, ``is_changed`` hash, per-call timeout override.
"""
from __future__ import annotations

import hashlib
import re
import time

from mie_llm.utils import mie_log


from mie_prompt.krea2_prompts import DEFAULT_ASPECT_RATIO, build_krea2_messages, resolve_aspect_ratio


MY_CATEGORY = "\U0001F411 MieNodes/\U0001F411 Prompt Generator"

# Default token budget. Krea docs explicitly say long detailed prompts
# yield best results, so we leave generous headroom. Reasoning models
# (M3, DeepSeek-R1, GLM-5.x) also emit a chain-of-thought block inside
# this budget before the final paragraph; 4096 is the lowest number
# that comfortably covers both. Mirrors the project's pattern of
# 2048-8192 across the LLM node family.
_MAX_TOKENS_DEFAULT = 4096
_MIN_MAX_TOKENS = 64
_MAX_MAX_TOKENS = 32768

# Default sampling knobs. ``temperature=0.7`` matches the Bernini / Flux2
# family; Krea-2's own examples range from photoreal to stylised, so a
# mild default is a reasonable middle ground.
_DEFAULT_TEMPERATURE = 0.7

# Default timeout. Krea-2 expansion runs as one LLM call (no vision,
# no JSON parsing), so 120 s is comfortably above the typical 10-30 s
# response window for most backends.
_DEFAULT_TIMEOUT = 120

# Strip a leading ``<think>...</think>`` block that reasoning models
# may emit before the visible answer. The Krea-2 system prompt tells
# the LLM to keep planning internal (rule 3) and to write one cohesive
# paragraph (rule 6), so any leaked thinking wrapper must be removed
# before the node returns the string.
_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


# --------------------------------------------------------------------------- #
# Response postprocess
# --------------------------------------------------------------------------- #
def postprocess_paragraph(raw_text: str) -> str:
    """Strip a leading ``<think>...</think>`` block (if any) and return
    the visible paragraph. Whitespace-only inputs collapse to ``""``.

    Per upstream rule 6 the visible answer is one cohesive paragraph
    with no bullets / JSON / markdown; reasoning models may still emit
    a leading think wrapper, so we sanitize it here.
    """
    if not raw_text:
        return ""
    text = raw_text.strip()
    text = _THINK_BLOCK_RE.sub("", text, count=1).strip()
    return text


# --------------------------------------------------------------------------- #
# Enhancer
# --------------------------------------------------------------------------- #
class Krea2PromptEnhancer:
    """Expand a user idea into a Krea-2 paragraph via the configured LLM.

    Mirrors the I/O shape of ``Ideogram4PromptEnhancer`` so the node
    can be swapped into existing workflows with minimal wiring.
    """

    def __init__(
        self,
        llm_service_connector,
        *,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _MAX_TOKENS_DEFAULT,
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        self.llm = llm_service_connector
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        # Per-call timeout override. ``None`` would mean "leave the
        # connector's own timeout alone"; the node always wires a
        # dropdown default, so this is always set in practice.
        self._timeout_override = int(timeout) if timeout else None

    def _invoke(self, messages: list[dict], seed=None) -> str:
        prev_timeout = getattr(self.llm, "timeout", None)
        try:
            if self._timeout_override is not None and prev_timeout is not None:
                self.llm.timeout = self._timeout_override
            t0 = time.perf_counter()
            out = self.llm.invoke(
                messages,
                seed=seed,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            elapsed = time.perf_counter() - t0
            model_name = getattr(self.llm, "model", "?")
            if not out:
                mie_log(
                    f"Krea2: model={model_name} returned empty after {elapsed:.2f}s"
                )
                return ""
            mie_log(
                f"Krea2: model={model_name} ok in {elapsed:.2f}s response_chars={len(out)}"
            )
            return out
        finally:
            if prev_timeout is not None:
                self.llm.timeout = prev_timeout

    def __call__(
        self,
        user_prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        seed=None,
    ) -> str:
        """Run one LLM call and return the postprocessed paragraph.

        Raises ``ValueError`` when ``user_prompt`` is empty -- the
        upstream expansion prompt is meaningless without an idea to
        expand, and silent fall-through to the LLM would burn tokens.
        """
        prompt = (user_prompt or "").strip()
        if not prompt:
            raise ValueError("Krea2PromptEnhancer: user_prompt is empty")

        ar = resolve_aspect_ratio(aspect_ratio)
        messages = build_krea2_messages(prompt, ar)
        raw = self._invoke(messages, seed=seed)
        return postprocess_paragraph(raw)


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
