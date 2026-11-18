"""SCAIL-2 prompt-enhancer ComfyUI node.

Two-stage LLM pipeline mirroring
https://github.com/zai-org/SCAIL-2/blob/wan-scail2/prompt_enhancer.py
for replacement mode, plus a MieNodes-original motion-transfer variant.

The node is registered under the same ``Prompt Generator`` category as
``BerniniPromptGenerator`` and ``Ideogram4PromptGenerator`` and follows
the same output shape (single ``STRING``).
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional

from mie_llm.media import image_tensor_batch_to_data_urls
from mie_llm.utils import mie_log

from mie_prompt.scail2_prompts import MAX_EXAMPLE_CHARS, TASK_CODES, TASK_TYPES, bundled_examples_motion_transfer, bundled_examples_replacement, caption_motion_transfer_prompt, caption_replacement_prompt, enhance_motion_transfer_prompt, enhance_replacement_prompt, load_bundled_examples, parse_task_code

from mie_llm.budget import invoke_with_budget_escalation


MY_CATEGORY = "\ud83d\udc11 MieNodes/\ud83d\udc11 Prompt Generator"

# Per-stage token budgets. Stage 1 writes a long English paragraph;
# stage 2 is constrained to 80-140 words by the prompt templates.
# NOTE: reasoning models (MiniMax-M3, DeepSeek-R1, GLM-5.x) emit their
# chain-of-thought INSIDE the token budget before the final answer. If the
# <think> chain consumes the whole budget the model returns an empty
# content (the #1 cause of "Scail2 ...: returned empty" logs — M3 thinking
# for tens of seconds and exhausting the budget before the answer). So both
# budgets are sized to hold the think chain AND the ~140-word answer with
# headroom. Giving more than needed costs nothing — the model stops as soon
# as the answer is done; only worst-case latency grows. (512 was the
# original default and was too tight; 2048 still edge-case-exhausted on long
# driving-video captions; 4096 leaves comfortable room.) _invoke also
# auto-escalates on finish_reason='length' (shared invoke_budget helper),
# so even a budget that still edge-case-exhausts self-heals with a doubled
# retry instead of failing the queue.
_DEFAULT_MAX_TOKENS_CAPTION = 4096
_DEFAULT_MAX_TOKENS_ENHANCE = 4096

# Default number of frames sampled from the driving video for stage 1.
# Matches upstream ``prompt_enhancer.py`` default of 8.
DEFAULT_NUM_FRAMES = 8
MIN_NUM_FRAMES = 1
MAX_NUM_FRAMES = 16

# --------------------------------------------------------------------------- #
# Frame sampling (mirrors bernini_prompt_generator._sample_indices / _sample_urls)
# --------------------------------------------------------------------------- #
def _sample_indices(total: int, n: int) -> list[int]:
    """Return ``n`` unique indices in ``[0, total)`` sampled as evenly as possible.

    ``n`` is clamped to ``[1, total]`` before sampling. When ``n == 1`` the
    middle index is returned; when ``n == 2`` the endpoints are returned
    so a driving video gets both the first and last frame in its caption;
    for larger ``n`` the indices are spaced uniformly using
    ``round(i * (total - 1) / (n - 1))``.
    """
    if total <= 0:
        return []
    n = max(1, min(n, total))
    if n == 1:
        return [total // 2]
    if n == 2:
        return [0, total - 1]
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]


def _sample_urls(urls: list[str], n: int) -> list[str]:
    """Apply ``_sample_indices`` to a list of URLs, preserving order and
    dropping duplicate indices (can happen at the endpoints)."""
    if not urls:
        return []
    idx = _sample_indices(len(urls), n)
    seen: set[int] = set()
    out: list[str] = []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(urls[i])
    return out


# --------------------------------------------------------------------------- #
# Enhancer
# --------------------------------------------------------------------------- #
class Scail2PromptEnhancer:
    """Two-stage SCAIL-2 prompt enhancer that talks to the project's LLMServiceConnector.

    Stage 1 captions the driving video; stage 2 writes the final positive
    description of the animated / replaced video. Each stage is one
    ``llm.invoke`` call against the same connector.

    Mirrors the structure of ``BerniniPromptEnhancer`` so the two LLM
    nodes can share the project's logging + timeout patterns.
    """

    def __init__(
        self,
        llm_service_connector: Any,
        *,
        num_frames: int = DEFAULT_NUM_FRAMES,
        image_detail: str = "auto",
        temperature: float = 0.4,
        max_tokens_caption: int = _DEFAULT_MAX_TOKENS_CAPTION,
        max_tokens_enhance: int = _DEFAULT_MAX_TOKENS_ENHANCE,
        timeout: Optional[int] = None,
    ):
        self.llm = llm_service_connector
        self.num_frames = max(MIN_NUM_FRAMES, min(int(num_frames), MAX_NUM_FRAMES))
        self.image_detail = image_detail
        self.temperature = float(temperature)
        self.max_tokens_caption = int(max_tokens_caption)
        self.max_tokens_enhance = int(max_tokens_enhance)
        # Per-call timeout override; None means leave the connector's
        # own timeout in place. Saved and restored around invoke() so
        # the connector object is safe to share with other nodes.
        self._timeout_override = int(timeout) if timeout else None

    # ------------------------------------------------------------------ #
    # Internal: one LLM call with timeout / log plumbing
    # ------------------------------------------------------------------ #
    def _invoke(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        seed: Optional[int],
        stage: str,
    ) -> str:
        prev_timeout = getattr(self.llm, "timeout", None)
        try:
            if self._timeout_override is not None:
                self.llm.timeout = self._timeout_override
            return invoke_with_budget_escalation(
                self.llm,
                messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
                seed=seed,
                stage_label=f"Scail2 {stage}",
            )
        finally:
            if prev_timeout is not None:
                self.llm.timeout = prev_timeout

    @staticmethod
    def _build_messages(
        system_prompt: str,
        user_text: str,
        image_urls: list[str],
        image_detail: str,
    ) -> list[dict]:
        """Build a ``[system, user[...]]`` pair in OpenAI-chat format.

        Mirrors ``bernini_prompt_generator._build_messages``: each image
        gets a small ``[Image N]:`` caption preceding the image part so
        vision models that look at the textual layout of the conversation
        know which image is which.
        """
        parts: list[dict] = []
        for i, url in enumerate(image_urls or []):
            parts.append({"type": "text", "text": f"\n[Image {i}]:"})
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": image_detail},
                }
            )
        if user_text:
            parts.append({"type": "text", "text": user_text})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": parts},
        ]

    # ------------------------------------------------------------------ #
    # Per-task caption + enhance pipeline
    # ------------------------------------------------------------------ #
    def _caption(
        self,
        task_code: str,
        frame_urls: list[str],
        user_prompt: str,
        seed: Optional[int],
    ) -> str:
        """Stage 1: caption the driving video."""
        if task_code == "character_replacement":
            system = caption_replacement_prompt()
            user_text = (
                f"User replacement instruction: {user_prompt.strip()}\n"
                f"The following images are {len(frame_urls)} sampled frames in chronological order."
            )
        elif task_code == "motion_transfer":
            system = caption_motion_transfer_prompt()
            user_text = (
                f"User hint (may be empty): {user_prompt.strip() or '(no user hint)'}\n"
                f"The following images are {len(frame_urls)} sampled frames in chronological order."
            )
        else:
            raise ValueError(f"Scail2: unknown task code: {task_code!r}")
        messages = self._build_messages(
            system, user_text, frame_urls, self.image_detail
        )
        caption = self._invoke(
            messages,
            max_tokens=self.max_tokens_caption,
            seed=seed,
            stage=f"caption[{task_code}]",
        )
        mie_log(
            f"Scail2 caption[{task_code}]: {len(caption)} chars: {caption[:120]!r}"
        )
        return caption

    def _enhance(
        self,
        task_code: str,
        ref_urls: list[str],
        user_prompt: str,
        caption: str,
        seed: Optional[int],
    ) -> str:
        """Stage 2: rewrite the caption as a positive SCAIL-2 prompt."""
        if task_code == "character_replacement":
            examples = bundled_examples_replacement()
            user_text = enhance_replacement_prompt(
                instruction=user_prompt,
                caption=caption,
                examples=examples,
            )
            system = "You are a prompt enhancer for SCAIL-2 character replacement."
        elif task_code == "motion_transfer":
            examples = bundled_examples_motion_transfer()
            user_text = enhance_motion_transfer_prompt(
                caption=caption,
                user_hint=user_prompt,
                examples=examples,
            )
            system = "You are a prompt enhancer for SCAIL-2 character animation / motion transfer."
        else:
            raise ValueError(f"Scail2: unknown task code: {task_code!r}")
        messages = self._build_messages(
            system, user_text, ref_urls, self.image_detail
        )
        return self._invoke(
            messages,
            max_tokens=self.max_tokens_enhance,
            seed=seed,
            stage=f"enhance[{task_code}]",
        )

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        task_type: str,
        user_prompt: str,
        *,
        driving_video: Any = None,
        reference_images: Any = None,
        seed: Optional[int] = None,
    ) -> str:
        """Run the two-stage pipeline and return the final SCAIL-2 prompt.

        Returns ``user_prompt`` unchanged on missing media (matching
        Bernini's graceful-degradation behavior), on an LLM error, or
        on a ``character_replacement`` task with an empty user_prompt.

        ``motion_transfer`` accepts an empty user_prompt: in that case
        the stage-2 template falls back to deriving a motion description
        purely from the driving-video caption.
        """
        code = parse_task_code(task_type)
        if code not in TASK_CODES:
            mie_log(
                f"Scail2: unknown task {task_type!r} (parsed code={code!r}); returning original"
            )
            return user_prompt

        # Empty user_prompt rules:
        #   - character_replacement: required -> return original (empty) prompt.
        #   - motion_transfer: optional -> auto-derive from media.
        if not user_prompt or not user_prompt.strip():
            if code == "character_replacement":
                mie_log(
                    "Scail2: character_replacement requires a non-empty user_prompt; returning original"
                )
                return user_prompt

        # Both tasks need a driving video + at least one reference image.
        driving_urls = image_tensor_batch_to_data_urls(driving_video)
        ref_urls = image_tensor_batch_to_data_urls(reference_images)
        if not driving_urls:
            mie_log(f"Scail2: no driving_video frames provided (task={code}); returning original")
            return user_prompt
        if not ref_urls:
            mie_log(f"Scail2: no reference images provided (task={code}); returning original")
            return user_prompt

        frame_urls = _sample_urls(driving_urls, self.num_frames)
        mie_log(
            f"Scail2: task={code} driving_frames={len(driving_urls)}->{len(frame_urls)} "
            f"ref_imgs={len(ref_urls)} detail={self.image_detail} "
            f"temperature={self.temperature}"
        )

        caption = self._caption(code, frame_urls, user_prompt or "", seed=seed)
        if not caption:
            mie_log(
                f"Scail2: stage-1 caption returned empty (task={code}); returning original"
            )
            return user_prompt

        enhanced = self._enhance(code, ref_urls, user_prompt or "", caption, seed=seed)
        return enhanced or user_prompt


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
