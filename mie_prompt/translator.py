"""Translation helpers (moved from MieNodes, batch 2c).

The language table and the message assembly: pure text, no connector
and no ComfyUI. The node shell in MieNodes calls ``build_messages``.
"""

from __future__ import annotations


TARGET_LANGUAGE_MAP = {
    "zh": "Chinese",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "it": "Italian",
    "pt": "Portuguese",
}


def build_messages(text: str, target_language: str) -> list:
    """The two-message exchange for a translation request."""
    from .prompts.loader import load_prompt_text

    language_name = TARGET_LANGUAGE_MAP.get(target_language, target_language)
    return [
        {
            "role": "system",
            "content": load_prompt_text("translator/system_template").format(
                language_name=language_name
            ),
        },
        {"role": "user", "content": text},
    ]
