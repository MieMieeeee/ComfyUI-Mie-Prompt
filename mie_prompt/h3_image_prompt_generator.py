"""MiniMax-H3-Image prompt-enhancer ComfyUI node.

Rewrites a rough user draft into the prompt **MiniMax-H3-Image** consumes.
H3-Image is the *image* model; its video sibling is the separate
``MiniMaxH3PromptGenerator`` node, and this one must not be confused with
it.

Four modes, one dropdown:

* ``t2i``        -- text-to-image. The short brief becomes an English
  caption **plus** an aspect ratio, delivered as one JSON object with two
  keys. Driven by the official PE task prompt, substituted with
  ``.format(user_text)``.
* ``edit``       -- image editing. The edit instruction plus the reference
  images become a three-part caption plus a target ratio. The official edit
  task prompt has NO placeholder, so the instruction and the ratio travel
  in the last text part of a multi-part user message, and every reference
  image is preceded by its own ``Reference Media: <Picture N> ...`` part.
* ``t2i_guide``  -- the official prompt-writing skill: a free-text prompt
  block plus an ``Aspect ratio: W:H`` line.
* ``edit_guide`` -- the same skill for editing, with the reference images.

The node owns three things the model does not:

* the **aspect ratio** -- the user's choice wins over the model's, and for
  ``t2i`` it is clamped to the ten ratios the official PE may pick;
* the **reference-image metadata** -- each image's exact ratio is computed
  from its pixels here, never guessed by the model;
* the **output contract** -- the reply is parsed and normalized (key names,
  ratio, empty caption, dropped quoted text). The reply is never passed
  through raw.

``edit`` / ``edit_guide`` need a vision-capable connector: the official
prompt requires the model to look at the sources (``<Picture N>``,
``<Subject N>``, transcribed source text). A text-only connector would make
the node invent what it supposedly saw, so the node refuses instead of
silently degrading.

Wire ``prompt`` straight into the H3-Image conditioning node; wire
``aspect_ratio`` into the resolution control. H3-Image runs CFG 1 with no
negative slot -- this node produces no negative prompt.
"""
from __future__ import annotations

import hashlib
import time
from typing import Optional

from mie_prompt.h3_image_prompts import ADAPTIVE, ALLOWED_T2I_RATIOS, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, GUIDE_MODES, IMAGE_MODES, MAX_REFERENCE_IMAGES, MODES, build_messages, image_ratio, is_image_mode, parse_expansion_reply, parse_guide_reply, parse_mode
from mie_llm.budget import invoke_with_budget_escalation
from mie_llm.media import image_tensor_batch_to_data_urls
from mie_llm.utils import mie_log


MY_CATEGORY = "\U0001F411 MieNodes/\U0001F411 Prompt Generator"

# Per-call timeout override (seconds).
_DEFAULT_TIMEOUT = 300

# The node's own cap on how many reference images it forwards. The official
# skill allows 16; the ComfyUI-side prompt-assist nodes top out around 9.
# Beyond the cap the extra images are dropped and the count is logged --
# the user owns the sampler configuration and we do not fail their run.
_DEFAULT_REFERENCE_CAP = 8

# How many times an unusable reply is re-asked. One corrective retry, the
# same shape as the MOSS layout parser: the assistant turn and a short
# corrective user turn are appended and the call is made again.
_MAX_PARSE_ATTEMPTS = 2

_CORRECTIVE_USER_TEXT = (
    "Your previous reply could not be used. Reply with ONE JSON object and "
    "nothing else: no prose, no code fence, no thinking tags. The object has "
    "exactly two string keys: expanded_caption and aspect_ratio. Every word "
    "the user put in quotes must appear verbatim in expanded_caption."
)

# The guide modes do not answer with JSON -- they answer with the prompt in
# a text block followed by an "Aspect ratio: W:H" line -- so their corrective
# turn has to ask for that shape, not for a JSON object.
_CORRECTIVE_USER_TEXT_GUIDE = (
    "Your previous reply could not be used. Reply with the prompt in a text "
    "block, followed by a line 'Aspect ratio: W:H' -- nothing else, no "
    "thinking tags. Every word the user put in quotes must appear verbatim, "
    "exactly as written."
)


def _image_count(reference_images) -> int:
    """How many images are in the batch (0 when nothing is connected)."""
    if reference_images is None:
        return 0
    try:
        return int(reference_images.shape[0])
    except (AttributeError, IndexError, TypeError):
        return 0


def _reference_ratios(reference_images) -> list[str]:
    """The exact ratio of every reference image, computed from its pixels.

    The official edit prompt wants the caller's own ratio for each image
    ("the caller has already reduced the image's original display width and
    height to their exact lowest integer ratio"), and the model must not
    estimate it from the preview. ComfyUI hands the node an IMAGE tensor
    whose frames are already upright, so the tensor's width and height are
    the display size the ratio comes from.
    """
    if reference_images is None:
        return []
    try:
        _, height, width = reference_images.shape[:3]
    except (AttributeError, ValueError, TypeError):
        return []
    ratio = image_ratio(int(width), int(height))
    return [ratio] * _image_count(reference_images)


# --------------------------------------------------------------------------- #
# Enhancer
# --------------------------------------------------------------------------- #
class H3ImagePromptEnhancer:
    """Run one H3-Image enhancement call and turn the reply into a prompt."""

    def __init__(
        self,
        llm_service_connector,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        self.llm = llm_service_connector
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        # Per-call timeout override, restored after the call. ``None`` would
        # mean "leave the connector's own timeout alone".
        self._timeout_override = int(timeout) if timeout else None

    def _invoke(self, messages: list[dict], *, seed=None, stage: str) -> str:
        prev_timeout = getattr(self.llm, "timeout", None)
        prev_fallback = getattr(self.llm, "allow_reasoning_fallback", None)
        try:
            if self._timeout_override is not None and prev_timeout is not None:
                self.llm.timeout = self._timeout_override
            # The caption is the answer. A provider that returns only its
            # reasoning channel hands back text with no caption; refusing
            # that fallback turns the failure into the empty-reply error
            # below instead of a silent unusable string.
            self.llm.allow_reasoning_fallback = False
            t0 = time.perf_counter()
            out = invoke_with_budget_escalation(
                self.llm,
                messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=seed,
                stage_label=f"H3Image {stage}",
            )
            elapsed = time.perf_counter() - t0
            model_name = getattr(self.llm, "model", "?")
            if not out or not out.strip():
                raise RuntimeError(
                    f"H3Image: LLM returned an empty reply after "
                    f"{elapsed:.2f}s (model={model_name}). Reasoning models "
                    f"often exhaust the budget before writing the caption -- "
                    f"raise max_tokens / 模型返回空回复，推理模型可能已用尽 "
                    f"token 预算，请调高 max_tokens"
                )
            mie_log(
                f"H3Image: model={model_name} ok in {elapsed:.2f}s "
                f"response_chars={len(out)}"
            )
            return out
        finally:
            if prev_timeout is not None:
                self.llm.timeout = prev_timeout
            if prev_fallback is not None:
                self.llm.allow_reasoning_fallback = prev_fallback

    def __call__(
        self,
        *,
        mode: str,
        user_text: str,
        image_urls: Optional[list[str]] = None,
        reference_ratios: Optional[list[str]] = None,
        aspect_ratio: str = ADAPTIVE,
        seed=None,
    ) -> tuple[str, str, str]:
        """Enhance ``user_text``; return ``(prompt, aspect_ratio, raw_reply)``."""
        messages = build_messages(
            mode,
            user_text,
            image_urls=image_urls,
            reference_ratios=reference_ratios,
            aspect_ratio=aspect_ratio,
        )
        stage = f"enhance[{mode}]"
        corrective = (
            _CORRECTIVE_USER_TEXT_GUIDE
            if mode in GUIDE_MODES
            else _CORRECTIVE_USER_TEXT
        )
        last_error: Optional[Exception] = None
        last_raw = ""
        for attempt in range(1, _MAX_PARSE_ATTEMPTS + 1):
            try:
                raw = self._invoke(messages, seed=seed, stage=stage)
            except RuntimeError as exc:
                # An empty reply is the reasoning model thinking itself out
                # of the answer. It is worth one more call before the run
                # dies -- the same call often lands.
                last_error = exc
                mie_log(
                    f"H3Image: {stage} got an empty reply "
                    f"(attempt {attempt}/{_MAX_PARSE_ATTEMPTS}): {exc}"
                )
                if attempt < _MAX_PARSE_ATTEMPTS:
                    continue
                break
            last_raw = raw
            try:
                return self._parse(
                    raw, mode, user_text, aspect_ratio, image_urls, reference_ratios
                )
            except Exception as exc:  # noqa: BLE001 - re-raised below
                last_error = exc
                mie_log(
                    f"H3Image: {stage} got no usable reply "
                    f"(attempt {attempt}/{_MAX_PARSE_ATTEMPTS}): {exc} "
                    f"raw_head={raw[:300]!r}"
                )
                if attempt < _MAX_PARSE_ATTEMPTS:
                    messages = messages + [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": corrective},
                    ]
        raise RuntimeError(
            f"H3Image: {stage} failed after {_MAX_PARSE_ATTEMPTS} attempts: "
            f"{last_error} raw_head={last_raw[:300]!r}"
        )

    @staticmethod
    def _parse(
        raw: str,
        mode: str,
        user_text: str,
        aspect_ratio: str,
        image_urls: Optional[list[str]],
        reference_ratios: Optional[list[str]] = None,
    ) -> tuple[str, str, str]:
        """Parse and normalize the reply into ``(prompt, ratio, raw)``."""
        ratios = list(reference_ratios or [])
        if mode in GUIDE_MODES:
            result = parse_guide_reply(
                raw,
                requested_ratio=aspect_ratio,
                # The editing guide keeps the input's framing when nothing
                # asks for a change, so the reference's own ratio is the
                # fallback when the reply has no usable ratio line.
                fallback_ratio=ratios[0] if ratios else "",
                user_text=user_text,
            )
        else:
            result = parse_expansion_reply(
                raw,
                mode=mode,
                requested_ratio=aspect_ratio,
                # In adaptive mode the official edit prompt preserves the
                # main reference's own ratio -- the node knows it, so the
                # fallback is the reference ratio, not a blind 1:1.
                fallback_ratio=ratios[0] if ratios else "",
                user_text=user_text,
                reference_count=len(image_urls or []),
            )
        for warning in result.warnings:
            # The warnings already carry the "H3Image: " prefix.
            mie_log(warning)
        # A caption that names no <Picture N> at all in an editing mode
        # means the connector probably never saw the images.
        if image_urls and "<Picture" not in result.caption:
            mie_log(
                "H3Image: the caption names no <Picture N>; the connector may "
                "not be vision-capable / caption 没有引用任何参考图，连接器可能"
                "不支持视觉"
            )
        ratio = result.aspect_ratio
        # The user's choice wins over the model's. For t2i the value stays
        # inside the ten ratios the official PE may pick; for the edit
        # modes any positive-integer W:H is legal.
        wanted = (aspect_ratio or "").strip()
        if wanted and wanted.lower() != ADAPTIVE:
            if mode in {"t2i", "t2i_guide"}:
                if wanted in ALLOWED_T2I_RATIOS and ratio != wanted:
                    mie_log(
                        f"H3Image: the model returned {ratio}, the requested "
                        f"{wanted} wins"
                    )
                    ratio = wanted
            elif ratio != wanted:
                mie_log(
                    f"H3Image: the model returned {ratio}, the requested "
                    f"{wanted} wins"
                )
                ratio = wanted
        return result.caption, ratio, result.raw


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
