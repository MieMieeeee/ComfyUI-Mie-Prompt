"""MiniMax-H3-Image prompt-enhancer helpers (modes, official prompts, reply parsing).

MiniMax-H3-Image is the *image* model. Its video sibling lives in the ``h3``
family, so this one is ``h3_image``.

The official artifacts are shipped verbatim under ``prompts/h3_image/`` and are
never edited (see ``prompts/h3_image/UPSTREAM.md`` for the sha256 digests):

* ``_pe_t2i_system.txt``       -- the only system prompt MiniMax published;
                                  the ComfyUI edit template reuses it too.
* ``_pe_t2i_task.txt``         -- t2i task prompt; one ``{}`` slot, so it is
                                  substituted with ``.format(user_text)``.
* ``_pe_t2i_request_params.json`` -- official sampling parameters + the ten
                                  allowed aspect ratios.
* ``_pe_edit_task.txt``        -- edit task prompt. It has NO ``{}`` slot: a
                                  71-character JSON example sits inside it, so
                                  ``.format()`` would raise ``KeyError``. The
                                  instruction and the ratio travel in the LAST
                                  text part of a multi-part user message, and
                                  every image is preceded by its own
                                  "Reference Media: <Picture N> ..." part.

Two modes, single-select dropdown -- the task axis mirrors the two official
templates (there is no separate reference-generation template: the edit task
natively takes the picture being edited plus up to 16 reference images):

======================  ==========================================  ==========
code                    what it does                                LLM
======================  ==========================================  ==========
``t2i``                 short brief -> English caption + ratio     yes
``edit``                edit instruction + references -> caption     yes (vision)
======================  ==========================================  ==========

The official prompt-writing skill (``_skill_t2i.txt`` / ``_skill_edit.txt`` /
``_skill_main.md``) stays shipped verbatim as reference material; the
``_guide`` modes that once drove it were removed from the dropdown on
2026-11-22 (user ruling: the mode axis is the task axis).

Everything here is deterministic and connector-free: no torch, no ComfyUI,
no network. The node layer wires the sockets and makes the call.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from math import gcd

from .prompts.loader import load_prompt_dict, load_prompt_text
from .aspect_ratio import ratio_from_size


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
# Display strings use the literal separator " - " so ``parse_mode`` can split
# them back into the short code -- the same convention as the other prompt
# generator families.
MODES: tuple[str, ...] = (
    "t2i - 文生图增强",
    "edit - 图像编辑增强",
)

MODE_CODES: tuple[str, ...] = (
    "t2i",
    "edit",
)

# Both modes' contract is a JSON object with two string values (kept as a
# named constant because the reply parser is written against it).
JSON_MODES = frozenset({"t2i", "edit"})

# The modes that need the LLM to *see* the reference images. A text-only model
# cannot run them: the official edit prompt requires the model to look at the
# sources (<Picture N>, <Subject N>, transcribed source text).
IMAGE_MODES = frozenset({"edit"})

# The official reference-image ceiling (the skill says "up to 16 reference
# images"). The node default is lower -- see modes.md -- and this module
# only enforces the hard ceiling.
MAX_REFERENCE_IMAGES = 16


def parse_mode(label: str) -> str:
    """Return the short mode code for a dropdown label ("t2i - 文生图增强" -> "t2i")."""
    return (label or "").split(" - ", 1)[0].strip()


def is_image_mode(mode: str) -> bool:
    """True when the mode needs a vision model to see the reference images."""
    return mode in IMAGE_MODES


# --------------------------------------------------------------------------- #
# Official text
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = "h3_image/_pe_t2i_system"
_TASK_PROMPTS: dict[str, str] = {
    "t2i": "h3_image/_pe_t2i_task",
    "edit": "h3_image/_pe_edit_task",
}
_REQUEST_PARAMS = "h3_image/_pe_t2i_request_params"


def load_system_prompt() -> str:
    """The official system prompt, verbatim.

    MiniMax published exactly one system prompt, for t2i; the ComfyUI edit
    template reuses the same file, so the edit mode uses it too. The trailing
    newline is stripped, exactly like the official ``expand_prompt.py``.
    """
    return load_prompt_text(_SYSTEM_PROMPT).rstrip("\n")


def load_task_prompt(mode: str) -> str:
    """The official task prompt for ``mode``, verbatim.

    The text is never substituted here -- ``t2i`` is the only mode with a
    ``{}`` slot, and the caller gets it through ``build_user_message`` so
    the ``.format()`` call stays visible at the call site.
    """
    try:
        logical_name = _TASK_PROMPTS[mode]
    except KeyError:
        raise ValueError(
            f"H3Image: unknown mode {mode!r}; expected one of "
            f"{', '.join(MODE_CODES)}"
        ) from None
    return load_prompt_text(logical_name)


def load_request_params() -> dict:
    """The official sampling parameters, verbatim.

    Only ``temperature`` and ``max_tokens`` are carried by the project
    connector; the rest (``top_p``, ``top_k``, ``reasoning_effort``,
    ``enable_thinking``, the JSON schema mode) are documented as not
    forwarded. The node parses and validates the JSON itself.
    """
    return load_prompt_dict(_REQUEST_PARAMS)


# The ten ratios the official PE may pick, verbatim from
# ``_pe_t2i_request_params.json``. The t2i mode never emits anything else;
# ``tests/test_h3_image_prompts.py`` asserts that this tuple still equals
# the enum in the official file.
ALLOWED_T2I_RATIOS: tuple[str, ...] = (
    "1:1",
    "5:4",
    "4:3",
    "3:2",
    "16:9",
    "21:9",
    "4:5",
    "3:4",
    "2:3",
    "9:16",
)

# The ratio that means "let the model decide".
ADAPTIVE = "adaptive"

# The official PE's own values. The node's defaults differ on purpose: see
# ``DEFAULT_TEMPERATURE`` / ``DEFAULT_MAX_TOKENS``.
OFFICIAL_TEMPERATURE = 1.0
OFFICIAL_MAX_TOKENS = 4096

# The node's defaults. The temperature follows the official value (1.0 --
# MOSS uses 0.7 because MOSS's official value is 0.7; do not copy that
# number here). The token budget is raised well above the official 4096:
# a reasoning model spends part of the budget thinking before the caption
# starts, and a reply that runs out mid-caption is the common failure.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 16384
MAX_TOKENS_CEILING = 32768


# --------------------------------------------------------------------------- #
# Message building
# --------------------------------------------------------------------------- #
def build_user_message(mode: str, user_text: str) -> str:
    """The user-side text for one H3-Image enhancement call.

    ``t2i`` is the only mode with a placeholder in the official task prompt:
    ``input_text: {}``, with the JSON examples written as ``{{...}}``. One
    ``.format()`` substitutes the user's text AND unescapes the doubled
    braces -- the official ``expand_prompt.py`` does exactly this, so
    ``.format()`` is the contract, not ``.replace()``.

    ``edit`` is never built here: its prompt has no slot, and the
    instruction travels in the last text part of a multi-part message --
    see :func:`build_user_content`.
    """
    text = (user_text or "").strip()
    if mode == "t2i":
        if not text:
            raise ValueError(
                "H3Image: the prompt is empty; there is nothing to "
                "enhance / prompt 为空，无内容可增强"
            )
        return load_task_prompt(mode).format(text)
    raise ValueError(
        f"H3Image: mode {mode!r} has no single-part user message; use "
        f"build_user_content"
    )


def reference_media_line(index: int, ratio: str) -> str:
    """The "Reference Media" text part that precedes one reference image.

    The official edit prompt names the shape exactly: "Each input image is
    immediately preceded by a Reference Media text part containing its
    ``<Picture N>`` label and aspect_ratio W:H", and the ratio is the
    image's own exact ratio -- or the word ``unavailable`` when the pixel
    size is unknown.
    """
    return f"Reference Media: <Picture {index}> aspect_ratio {ratio or 'unavailable'}"


def build_edit_final_part(instruction: str, aspect_ratio: str) -> str:
    """The LAST text part of an edit user message.

    The official edit prompt says the instruction is supplied "as User
    Editing Instruction (JSON string)" and that the ratio is supplied in the
    same final part as "User Aspect Ratio Selection".
    """
    text = (instruction or "").strip()
    if not text:
        raise ValueError(
            "H3Image: the editing instruction is empty; there is nothing to "
            "enhance / 编辑指令为空，无内容可增强"
        )
    return (
        f"User Editing Instruction: {json.dumps(text, ensure_ascii=False)}\n"
        f"User Aspect Ratio Selection: {aspect_ratio or ADAPTIVE}"
    )


def build_user_content(
    mode: str,
    user_text: str,
    *,
    image_urls: list[str] | None = None,
    reference_ratios: list[str] | None = None,
    aspect_ratio: str = ADAPTIVE,
) -> str | list[dict]:
    """The user turn for ``mode``.

    The text-only mode (``t2i``) returns a plain string. The image mode
    returns an OpenAI-style content list, because the official edit prompt
    requires the interleaved shape: a "Reference Media" text part, then the
    image it describes, and the instruction last.
    """
    urls = list(image_urls or [])
    if mode not in IMAGE_MODES:
        if urls:
            raise ValueError(
                f"H3Image: mode {mode!r} does not take reference images"
            )
        return build_user_message(mode, user_text)

    if len(urls) > MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"H3Image: {len(urls)} reference images exceed the official "
            f"limit of {MAX_REFERENCE_IMAGES} / 参考图超过官方上限"
        )

    ratios = list(reference_ratios or [])
    parts: list[dict] = []
    if mode == "edit":
        # The official edit task prompt IS the instruction set the model
        # has to follow, so it rides in the user turn ahead of the images
        # -- exactly what the official ComfyUI edit template does (its
        # task prompt is the node's user text). The instruction itself
        # goes in the LAST text part, which is what the prompt asks for.
        parts.append({"type": "text", "text": load_task_prompt("edit")})
    for index, url in enumerate(urls, start=1):
        ratio = ratios[index - 1] if index - 1 < len(ratios) else ""
        parts.append({"type": "text", "text": reference_media_line(index, ratio)})
        parts.append(
            {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
        )
    parts.append(
        {
            "type": "text",
            "text": build_edit_final_part(user_text, aspect_ratio),
        }
    )
    return parts


def build_system_prompt(mode: str) -> str:
    """The system message for ``mode``.

    Both modes use MiniMax's single official system prompt (published for
    t2i; the ComfyUI edit template reuses the same file).
    """
    if mode not in JSON_MODES:
        raise ValueError(
            f"H3Image: unknown mode {mode!r}; expected one of "
            f"{', '.join(MODE_CODES)}"
        )
    return load_system_prompt()


def build_messages(
    mode: str,
    user_text: str,
    *,
    image_urls: list[str] | None = None,
    reference_ratios: list[str] | None = None,
    aspect_ratio: str = ADAPTIVE,
) -> list[dict]:
    """The full chat payload for one H3-Image enhancement call.

    ``t2i`` is a plain [system, user] pair. The edit mode carries the
    interleaved user content the official edit prompt asks for.
    """
    content = build_user_content(
        mode,
        user_text,
        image_urls=image_urls,
        reference_ratios=reference_ratios,
        aspect_ratio=aspect_ratio,
    )
    return [
        {"role": "system", "content": build_system_prompt(mode)},
        {"role": "user", "content": content},
    ]


# --------------------------------------------------------------------------- #
# Reference-image metadata (H12)
# --------------------------------------------------------------------------- #
def image_ratio(width: int, height: int) -> str:
    """The exact ``W:H`` of a reference image, reduced to lowest terms.

    The node computes this; the LLM is never asked to guess it. ComfyUI's
    IMAGE input is a tensor whose pixels are already upright, so the EXIF
    orientation has been applied upstream and the tensor's width and height
    are the display size the ratio must come from.
    """
    return ratio_from_size(width, height)


# --------------------------------------------------------------------------- #
# Reply parsing
# --------------------------------------------------------------------------- #
class H3ImageError(ValueError):
    """Base class for the H3-Image helper errors."""


class H3ImageReplyError(H3ImageError):
    """The model reply could not be used as an expansion."""


@dataclass(frozen=True)
class Expansion:
    """One parsed enhancement.

    ``caption`` is the English description the image model consumes,
    ``aspect_ratio`` the normalised ratio, ``raw`` the untouched reply (for
    the debug port), ``warnings`` the soft findings and
    ``missing_literals`` the user's quoted text that did not survive.
    """

    caption: str
    aspect_ratio: str
    raw: str = ""
    warnings: tuple[str, ...] = ()
    missing_literals: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when the caption can go straight to the sampler."""
        return bool(self.caption) and not self.missing_literals


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", re.DOTALL)
_CJK_RE = re.compile(
    r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)
_MARKER_RE = re.compile(r"<\s*(Picture|Subject)\s+(\d+)\s*>", re.IGNORECASE)
_BACKTICKED_MARKER_RE = re.compile(r"`\s*<\s*(?:Picture|Subject)\s+\d+\s*>`", re.IGNORECASE)
_BARE_GRID_RE = re.compile(r"\d+\s*[×✕]\s*\d+")
_LOOSE_GRID_RE = re.compile(r"\d+\s*[xX]\s*\d+(?!\s*\(\s*rows)")


def strip_reasoning(text: str) -> str:
    """Drop the thinking block and the code-fence markers from a model reply.

    The official system prompt asks for the JSON object and nothing else,
    but a reasoning model still wraps its answer in ``<think>...</think>``
    and habitually fences it. Both are removed here so the parser only ever
    sees the object.

    The fence *markers* go, their content stays: a reply may carry prose
    around the fenced object, and dropping everything outside the first
    fence would lose it.
    """
    out = _THINK_BLOCK_RE.sub("", text or "")
    out = _OPEN_THINK_RE.sub("", out)
    out = _FENCE_RE.sub(lambda m: m.group(1), out)
    return out.strip()


def _straight_single_quoted(text: str) -> list[str]:
    """The single-quoted spans that are quotes, not apostrophes.

    A straight ``'`` opens a literal when it does not follow a word
    character, and closes one when it is not followed by one -- so
    ``it's`` is not a literal while ``'quoted'`` is.
    """
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if text[index] != "'" or (index and text[index - 1].isalnum()):
            index += 1
            continue
        cursor = index + 1
        while cursor < length:
            if text[cursor] == "'" and not (
                cursor + 1 < length and text[cursor + 1].isalnum()
            ):
                literal = text[index + 1 : cursor]
                if literal.strip() and "\n" not in literal:
                    out.append(literal)
                index = cursor + 1
                break
            cursor += 1
        else:
            index += 1
    return out


def _loads_first_object(text: str) -> tuple[dict, str]:
    """Parse the first JSON object in ``text``; also return the trailing text."""
    start = text.find("{")
    if start < 0:
        raise H3ImageReplyError(
            "H3Image: the reply carries no JSON object / 回复里没有 JSON 对象"
        )
    try:
        obj, end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise H3ImageReplyError(
            f"H3Image: the reply is not valid JSON ({exc}) / 回复不是合法 JSON"
        ) from exc
    if not isinstance(obj, dict):
        raise H3ImageReplyError(
            "H3Image: the reply's JSON is not an object / 回复的 JSON 不是对象"
        )
    return obj, text[start + end :].strip()


def extract_quoted_literals(text: str) -> list[str]:
    """The literals the user put in quotes, in the order they appear.

    The official guides require every word that appears in the image to be
    quoted and kept exactly as given, so the quoted spans of the user's
    draft are the ones that must survive into the caption. Straight single
    quotes are only read as quotes when they are not an apostrophe (``it's``
    is not a literal).
    """
    out: list[str] = []
    for opener, closer in (('"', '"'), ("\u201c", "\u201d"), ("\u2018", "\u2019")):
        index = 0
        while True:
            start = text.find(opener, index)
            if start < 0:
                break
            end = text.find(closer, start + 1)
            if end < 0:
                break
            literal = text[start + 1 : end]
            index = end + 1
            if literal.strip() and "\n" not in literal:
                out.append(literal)
    out.extend(_straight_single_quoted(text or ""))
    return out


def missing_literals(user_text: str, caption: str) -> list[str]:
    """The user's quoted literals that did not make it into the caption.

    This is the headline failure of the upstream PE: a model that rewords
    the copy it was told to keep renders the wrong words on the picture.
    """
    caption_text = caption or ""
    missing: list[str] = []
    for literal in extract_quoted_literals(user_text or ""):
        if literal not in caption_text and literal not in missing:
            missing.append(literal)
    return missing


def _cjk_outside_quotes(text: str) -> str:
    """The text with the quoted spans removed."""
    out = text or ""
    for opener, closer in (('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019")):
        out = re.sub(
            re.escape(opener) + r"[^" + re.escape(opener + closer) + r"]*"
            + re.escape(closer),
            " ",
            out,
        )
    return out


def _normalize_ratio_pair(text: str) -> str | None:
    """``"4:3"`` -> ``"4:3"`` reduced to lowest terms; ``None`` if it is not a ratio."""
    raw = (text or "").strip().replace(" ", "")
    if not raw or raw.lower() == ADAPTIVE or ":" not in raw:
        return None
    left, _, right = raw.partition(":")
    try:
        w, h = int(left), int(right)
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    g = gcd(w, h)
    return f"{w // g}:{h // g}"


def _ratio_value(text: str) -> float:
    """The numeric value of a ``W:H`` string, for comparing ratios."""
    pair = _normalize_ratio_pair(text)
    if not pair:
        return 0.0
    w, h = (int(part) for part in pair.split(":"))
    return w / h


def _snap_to_allowed(text: str) -> str | None:
    """The allowed ratio closest to ``text``; ``None`` if it is not a ratio."""
    if not text:
        return None
    wanted = _ratio_value(text)
    if not wanted:
        return None
    return min(
        ALLOWED_T2I_RATIOS,
        key=lambda ratio: abs(_ratio_value(ratio) - wanted),
    )


def normalize_ratio(
    value: str,
    mode: str,
    *,
    requested_ratio: str = ADAPTIVE,
    fallback_ratio: str = "",
) -> tuple[str, list[str]]:
    """Normalise the model's ratio to the mode's contract.

    ``t2i`` may only return one of the ten official ratios, so anything else
    is snapped to the nearest one (never to a value outside the enum).
    ``edit`` takes any positive-integer ``W:H``, reduced to lowest terms;
    ``adaptive`` never reaches the output -- it is replaced by the
    requested ratio, else by ``fallback_ratio`` (the main reference's own
    ratio, which is what the official edit prompt preserves in adaptive
    mode), else by 1:1.
    """
    warnings: list[str] = []
    raw = (value or "").strip()

    if mode == "t2i":
        if raw in ALLOWED_T2I_RATIOS:
            return raw, warnings
        snapped = _snap_to_allowed(raw)
        if snapped:
            warnings.append(
                f"H3Image: the model returned {raw!r}, which is not one of the "
                f"ten official ratios; using {snapped} / 模型返回的比例不在官方十值内"
            )
            return snapped, warnings
        fallback = raw if raw in ALLOWED_T2I_RATIOS else None
        if fallback is None:
            wanted = _normalize_ratio_pair(requested_ratio)
            fallback = (
                wanted
                if wanted in ALLOWED_T2I_RATIOS
                else ALLOWED_T2I_RATIOS[0]
            )
        warnings.append(
            f"H3Image: the model returned an unusable ratio {raw!r}; using "
            f"{fallback} / 模型返回的比例无法解析"
        )
        return fallback, warnings

    # edit: any positive-integer W:H, reduced.
    normalized = _normalize_ratio_pair(raw)
    if normalized:
        if normalized != raw:
            warnings.append(
                f"H3Image: the ratio {raw!r} was reduced to {normalized} / "
                f"比例已约分"
            )
        return normalized, warnings

    wanted = _normalize_ratio_pair(requested_ratio)
    fallback = wanted or _normalize_ratio_pair(fallback_ratio) or ALLOWED_T2I_RATIOS[0]
    warnings.append(
        f"H3Image: the model returned {raw!r}; using {fallback} instead of "
        f"adaptive / 模型返回了 adaptive，已替换"
    )
    return fallback, warnings


def _picture_numbers(text: str) -> set[int]:
    """The ``<Picture N>`` numbers that appear in ``text``."""
    return {
        int(match.group(2))
        for match in _MARKER_RE.finditer(text or "")
        if match.group(1).lower() == "picture"
    }


def parse_expansion_reply(
    raw: str,
    *,
    mode: str = "t2i",
    requested_ratio: str = ADAPTIVE,
    fallback_ratio: str = "",
    user_text: str = "",
    reference_count: int = 0,
    strict: bool = True,
) -> Expansion:
    """Parse and validate one JSON-mode reply.

    The rules are the ones the official artifacts imply:

    * one JSON object with ``expanded_caption`` and a ratio key (H1);
    * ``target_aspect_ratio`` and ``aspect_ratio`` are both accepted, and
      the former wins -- MiniMax's own edit prompt and system prompt
      disagree on the key name (H2);
    * the ratio obeys the mode's contract (H3 / H4);
    * the caption is a non-empty string (H5);
    * the prose is English, apart from the literals the user asked to
      render (H6);
    * every literal the user quoted survives verbatim (H7);
    * the ``<Picture N>`` / ``<Subject N>`` markers stay bare (H8);
    * every reference image is referenced, numbered from 1 (H13);
    * a grid is named by rows and columns (H14).

    ``strict`` turns H7 into a hard failure; otherwise the missing
    literals are reported on the result and the caption is kept.
    """
    text = strip_reasoning(raw)
    obj, trailing = _loads_first_object(text)

    warnings: list[str] = []
    if trailing:
        # H10: no prose outside the JSON object.
        warnings.append(
            "H3Image: the reply carries text outside the JSON object / "
            "回复在 JSON 之外还有文字"
        )

    if "expanded_caption" not in obj:
        raise H3ImageReplyError(
            "H3Image: the reply has no expanded_caption key / 回复缺少 "
            "expanded_caption"
        )
    caption = obj["expanded_caption"]
    if not isinstance(caption, str):
        raise H3ImageReplyError(
            "H3Image: expanded_caption is not a string / expanded_caption "
            "不是字符串"
        )
    caption = caption.strip()
    if not caption:
        raise H3ImageReplyError(
            "H3Image: expanded_caption is empty / 扩写内容为空"
        )

    if "target_aspect_ratio" in obj:
        ratio_value = obj["target_aspect_ratio"]
    elif "aspect_ratio" in obj:
        ratio_value = obj["aspect_ratio"]
    else:
        raise H3ImageReplyError(
            "H3Image: the reply has no aspect_ratio / target_aspect_ratio "
            "key / 回复缺少比例字段"
        )
    if not isinstance(ratio_value, str):
        raise H3ImageReplyError(
            "H3Image: the ratio is not a string / 比例不是字符串"
        )

    extra = sorted(
        key for key in obj if key not in {"expanded_caption", "aspect_ratio", "target_aspect_ratio"}
    )
    if extra:
        warnings.append(
            f"H3Image: the reply carries extra key(s) {', '.join(extra)} / "
            f"回复含多余字段"
        )

    ratio, ratio_warnings = normalize_ratio(
        ratio_value,
        mode,
        requested_ratio=requested_ratio,
        fallback_ratio=fallback_ratio,
    )
    warnings.extend(ratio_warnings)

    # H6: the prose is English; CJK may only survive inside a literal the
    # user asked to render.
    stray_cjk = _CJK_RE.search(_cjk_outside_quotes(caption))
    if stray_cjk:
        warnings.append(
            f"H3Image: the caption contains non-English text "
            f"({stray_cjk.group(0)!r}) outside the quoted literals / 正文出现非英文"
        )

    # H7: the user's quoted literals must survive.
    lost = missing_literals(user_text, caption)
    if lost:
        message = (
            f"H3Image: the caption dropped {len(lost)} quoted literal(s) the "
            f"user asked to keep: {'; '.join(repr(item) for item in lost)} / "
            f"用户要求逐字保留的文字被改写"
        )
        if strict:
            raise H3ImageReplyError(message)
        warnings.append(message)

    # H8: the markers stay bare.
    if _BACKTICKED_MARKER_RE.search(caption):
        warnings.append(
            "H3Image: a <Picture N> / <Subject N> marker is wrapped in "
            "backticks / 标记被加了反引号"
        )

    # H13: every reference image is referenced, numbered from 1.
    if reference_count > 0:
        seen = _picture_numbers(caption)
        absent = [n for n in range(1, reference_count + 1) if n not in seen]
        if absent:
            warnings.append(
                f"H3Image: the caption does not reference "
                f"{', '.join('<Picture %d>' % n for n in absent)} / "
                f"caption 未引用全部参考图"
            )

    # H14: a grid is named by rows and columns.
    if _BARE_GRID_RE.search(caption) or _LOOSE_GRID_RE.search(caption):
        warnings.append(
            "H3Image: the caption writes a grid as '2x3'; name the rows and "
            "the columns / 网格应写明行列"
        )

    return Expansion(
        caption=caption,
        aspect_ratio=ratio,
        raw=raw or "",
        warnings=tuple(warnings),
        missing_literals=tuple(lost),
    )
