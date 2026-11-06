"""The core's prompt families.

The prompts live as plain text and JSON files under this package directory.
:func:`prompt_path` resolves a logical name against it, so a consumer -
including the MieNodes tests that check the prompt and its documented copy
stay in step - can find a file without knowing the layout.
"""

from __future__ import annotations

from pathlib import Path

_PROMPTS_ROOT = Path(__file__).resolve().parent


def prompt_root() -> Path:
    """The directory the prompt families live in."""
    return _PROMPTS_ROOT


def prompt_path(relative: str, suffix: str = "") -> Path:
    """Resolve a prompt file under this package directory.

    ``relative`` is a path relative to the prompt root, for example
    ``"h3_storyboard/system_asset.txt"``. A suffix may be given for a
    family that stores the extension separately.
    """
    name = relative.replace("\\", "/")
    if suffix and not name.endswith(suffix):
        name += suffix
    return _PROMPTS_ROOT / name


__all__ = ["prompt_path", "prompt_root"]
