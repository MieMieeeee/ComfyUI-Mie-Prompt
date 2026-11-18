"""MOSS-Image prompt-enhancement ComfyUI node.

MOSS-Image is a 5B single-stream DiT trained on full scene captions, and a
one-word request fed straight into it produces a visibly weaker image than
the 200-500 word English caption it was trained on. The official workflow
therefore ships an expander: the model's own text encoder (Qwen3.5-9B) is
run as an LLM over the request before the image model sees it.

This node reproduces that behaviour over HTTP instead of in-process. Upstream
MOSS has **no** image-to-image mode, so there are only two modes:

* ``t2i`` -- the request is expanded into an English caption, using the
  official upsampler system prompt verbatim.
* ``t2i_layout`` -- the request is expanded into a spatial Layout: a JSON
  draft with absolute-pixel boxes, compiled by the node into the one-line
  compact JSON the image model consumes (``abs_centralized`` / ``yxyx``).

Unlike Klein, MOSS needs no separate upsampler weights, so the node occupies
no VRAM, works with any OpenAI-compatible endpoint, and accepts Chinese
input.

Two things make the layout mode different from every other Layout node:

* The three coordinate spaces are kept apart. ``CreateBoundingBoxes`` emits a
  0-1000 ``[y_min, x_min, y_max, x_max]`` grid; the model reads absolute
  pixels in ``xyxy`` from the top-left; the model finally consumes centered
  ``yxyx``. The node converts between them, because the official prompt says
  to use the given coordinates as they are and never convert them.
* The official docs warn that the prompt upsampler rewrites the layout. So
  the user's drawn boxes are authoritative by default: the node keeps the
  user's pixels and takes only the descriptions from the model.

Sits beside ``Flux3KleinPromptUpsampler`` in the Prompt Generator category
and keeps the family conventions: bilingual dropdown labels, per-call timeout
override, ``is_changed`` hashing.
"""
from __future__ import annotations

import hashlib
import json
import time

from mie_llm.media import build_multimodal_user_content, image_tensor_batch_to_data_urls
from mie_llm.utils import mie_log


from mie_prompt.moss_image_prompts import CAPTION_MODE, LAYOUT_MODE, MODES, build_user_message, compile_layout_draft, elements_to_boxes, enforce_supplied_boxes, load_system_prompt, parse_layout_reply, parse_mode_label


MY_CATEGORY = "\U0001F411 MieNodes/\U0001F411 Prompt Generator"

# Official MOSS caps the prompt at roughly 4K tokens, so the *model* never
# needs a bigger budget than that. The node's budget is a different thing:
# reasoning models (M3, DeepSeek-R1, GLM-5.x) spend part of it thinking
# before the caption, and a reply that runs out of budget mid-caption is
# the common failure. The default therefore sits well above upstream's
# figure, matching the rest of this project's LLM nodes.
_MAX_TOKENS_DEFAULT = 16384
_MIN_MAX_TOKENS = 64
_MAX_MAX_TOKENS = 32768

# Official decoding parameters (MOSS template): temperature 0.7, top_p 0.95,
# top_k 64, min_p 0.05, repetition_penalty 1.0, presence_penalty 0.
# Only temperature is exposed -- no connector in this project forwards the
# rest, and the bundled service layer deliberately does not carry them.
_DEFAULT_TEMPERATURE = 0.7

# One call, no JSON round-trips. 300 s covers a long caption from a
# reasoning model on a busy backend.
_DEFAULT_TIMEOUT = 300

# The official template feeds a single ``ref_image`` to the upsampler. The
# node accepts an IMAGE batch, so extra frames are capped rather than
# silently dropped.
_MAX_REFERENCE_IMAGES = 4

# How many samples to take when the layout reply carries no parseable JSON
# draft. Verified live: models habitually fence the object or add a sentence
# of prose around it, and one resample usually lands the official shape.
_LAYOUT_PARSE_ATTEMPTS = 2


# --------------------------------------------------------------------------- #
# Enhancer
# --------------------------------------------------------------------------- #
class MossImagePromptEnhancer:
    """Run one MOSS enhancement call and turn the reply into a prompt."""

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
        # Per-call timeout override, restored after the call. ``None`` would
        # mean "leave the connector's own timeout alone".
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
                # The caption is the answer; a provider that returns only its
                # reasoning channel would hand back text with no caption and
                # no layout. Refusing that fallback turns the failure into the
                # empty-reply error below, which names the real cause.
                allow_reasoning_fallback=False,
            )
            elapsed = time.perf_counter() - t0
            model_name = getattr(self.llm, "model", "?")
            if not out or not out.strip():
                # The connector returns "" rather than raising on an empty
                # 200. Reasoning models usually hit this by spending the
                # whole budget thinking, so say so instead of returning "".
                raise RuntimeError(
                    f"MossImage: LLM returned an empty reply after "
                    f"{elapsed:.2f}s (model={model_name}). Reasoning models "
                    f"often exhaust the budget before writing the caption -- "
                    f"raise max_tokens / 模型返回空回复，推理模型可能已用尽 "
                    f"token 预算，请调高 max_tokens"
                )
            mie_log(
                f"MossImage: model={model_name} ok in {elapsed:.2f}s "
                f"response_chars={len(out)}"
            )
            return out
        finally:
            if prev_timeout is not None:
                self.llm.timeout = prev_timeout

    def __call__(
        self,
        *,
        mode: str,
        prompt: str,
        width: int,
        height: int,
        boxes: list[dict] | None = None,
        reference_images=None,
        keep_layout_verbatim: bool = True,
        seed=None,
    ) -> tuple[str, str, str]:
        """Enhance ``prompt`` and return ``(upsampled_prompt, caption, layout_json)``.

        In the caption mode the reply is the caption and the layout fields
        are empty. In the layout mode the reply is a JSON draft that the
        node compiles into the one-line compact JSON the image model reads;
        ``layout_json`` is the readable draft, kept so the layout can be
        edited and re-compiled.
        """
        system_prompt = load_system_prompt(mode)
        boxes = boxes or []

        if mode == CAPTION_MODE:
            user_text = build_user_message(mode, prompt)
            messages = self._messages(system_prompt, user_text, reference_images)
            raw = self._invoke(messages, seed=seed)
            caption = raw.strip()
            return caption, caption, ""

        user_text = build_user_message(
            mode, prompt, width=width, height=height, boxes=boxes
        )
        messages = self._messages(system_prompt, user_text, reference_images)

        last_error: Exception | None = None
        for attempt in range(1, _LAYOUT_PARSE_ATTEMPTS + 1):
            raw = self._invoke(messages, seed=seed)
            try:
                draft = parse_layout_reply(raw)

                # The canvas belongs to the image workflow, not to the model.
                # If the model picked a different one, its boxes would
                # describe a different picture, so the requested canvas wins.
                if draft.get("width") != width or draft.get("height") != height:
                    mie_log(
                        f"MossImage: model returned canvas "
                        f"{draft.get('width')}x{draft.get('height')}, using the "
                        f"requested {width}x{height}"
                    )
                    draft["width"] = width
                    draft["height"] = height

                if boxes and keep_layout_verbatim:
                    # The official docs warn that the upsampler rewrites the
                    # layout. The user's pixels win; only the descriptions
                    # come from the model.
                    draft = enforce_supplied_boxes(draft, boxes)

                compiled = compile_layout_draft(draft)
            except ValueError as exc:
                last_error = exc
                mie_log(
                    f"MossImage: layout mode got no usable JSON draft: {exc} "
                    f"(attempt {attempt}/{_LAYOUT_PARSE_ATTEMPTS}); "
                    f"raw_tail={raw[-300:]!r}"
                )
                if attempt < _LAYOUT_PARSE_ATTEMPTS:
                    messages = messages + [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Your previous reply could not be used as a "
                                "JSON Layout draft: "
                                f"{exc}\n"
                                "Reply with ONE JSON object and nothing else: "
                                "no prose, no code fence. Keep every bbox "
                                "inside the canvas."
                            ),
                        },
                    ]
                continue

            summary = draft.get("summary")
            caption = (
                str(summary.get("caption") or "")
                if isinstance(summary, dict)
                else ""
            )
            readable = json.dumps(draft, ensure_ascii=False, indent=2)
            return compiled, caption, readable

        raise last_error

    @staticmethod
    def _messages(system_prompt: str, user_text: str, reference_images) -> list[dict]:
        """Build the chat messages, attaching a reference image when given.

        The reference image is only ever shown to the LLM -- MOSS-Image does
        not take a reference image -- so it rides in the user turn ahead of
        the text, the way the official template feeds ``ref_image``.
        """
        urls = (
            image_tensor_batch_to_data_urls(reference_images)
            if reference_images is not None
            else []
        )
        if len(urls) > _MAX_REFERENCE_IMAGES:
            mie_log(
                f"MossImage: {len(urls)} reference images exceed the official "
                f"single-image reference; using the first "
                f"{_MAX_REFERENCE_IMAGES} / 参考图超过官方上限，已截断"
            )
            urls = urls[:_MAX_REFERENCE_IMAGES]
        messages = [{"role": "system", "content": system_prompt}]
        if urls:
            messages.append(
                {
                    "role": "user",
                    "content": build_multimodal_user_content(user_text, urls),
                }
            )
        else:
            messages.append({"role": "user", "content": user_text})
        return messages


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
