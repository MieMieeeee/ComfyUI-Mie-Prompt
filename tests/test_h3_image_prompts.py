"""Tests for the H3-Image prompt family.

Two layers:

* the *provenance* layer -- the seven official files must stay byte-identical
  to upstream. The digests below are the ones recorded in the official
  ``prompt/manifest.json`` (four files) and the upstream GitHub skill files
  (three files), so a re-download can be diffed against them.
* the *normalizer* layer -- the rules the node must hold no matter what the
  model says. None of these tests need a model.

Run with ``pytest tests/``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mie_prompt import h3_image_prompts as h3
from mie_prompt.prompts import prompt_path
from mie_prompt.prompts.loader import list_usable_builtin_prompts


# --------------------------------------------------------------------------- #
# Provenance: the official files, byte for byte
# --------------------------------------------------------------------------- #
# sha256 of the four PE files as recorded in the official
# MiniMaxAI/MiniMax-H3-Image-PE prompt/manifest.json, plus the three
# prompt-writing skill files from MiniMax-AI/MiniMax-H3-Image.
OFFICIAL_DIGESTS = {
    "_pe_t2i_system.txt": "818ab3f652ec36b3b489b55c29f072f87e14af703cd11a4f3bff75f423d0beac",
    "_pe_t2i_task.txt": "890213ad61e143bbd5b7906a802430296e82ca5c61ba16df717dcef5477bbcf4",
    "_pe_t2i_request_params.json": "dfc608bf0c02dd6ce7463be636aa8b148aef200513fa4c46e653ee2e5affbe3e",
    "_pe_edit_task.txt": "c5695dbfc07a9e65555cef67d92467c613371cc2039b96f057022246a30a57b2",
    "_skill_main.md": "d76fb522208149078d284cdab11260c11d974ec12db921f25e3c0fa76aa3f9cf",
    "_skill_t2i.txt": "a0fe54167532464d5c4f40517c07e350c9631fc97fc1299b56f9f1d74b05a78b",
    "_skill_edit.txt": "4f08883dcf390d388fe077959d077834e6514eed73c5385aa71f9e962521aff3",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name,digest", sorted(OFFICIAL_DIGESTS.items()))
def test_official_files_are_verbatim(name: str, digest: str) -> None:
    """The shipped file is byte-identical to upstream."""
    path = prompt_path(f"h3_image/{name}")
    assert path.is_file(), path
    assert _sha256(path) == digest, f"{name} was edited -- restore the upstream bytes"


def test_the_license_copy_travels_with_the_files() -> None:
    """The upstream agreement ships next to the files it covers."""
    path = prompt_path("h3_image/_LICENSE-MiniMax-H3-Image.txt")
    assert path.is_file()
    assert _sha256(path) == (
        "0ec3bb5e6d9e885fa2aaf15e84689e50fd51f42b53fa24d3a64ed9d8c76d423b"
    )
    notice = prompt_path("h3_image/NOTICE")
    assert notice.is_file()
    assert "MiniMax-H3-Image Community License" in notice.read_text(encoding="utf-8")


def test_package_data_ships_the_family() -> None:
    """The wheel must carry the family -- package-data lists every folder."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert '"h3_image/*"' in pyproject.read_text(encoding="utf-8")


def test_the_family_never_leaks_into_the_generic_dropdown() -> None:
    """The PE files are build blocks, not usable system prompts.

    ``_pe_edit_task.txt`` carries a JSON example, not a ``{}`` placeholder,
    so the placeholder filter would not catch it -- the ``_`` prefix does.
    """
    usable = list_usable_builtin_prompts()
    assert not [name for name in usable if name.startswith("h3_image/")]


# --------------------------------------------------------------------------- #
# The official contract, as the code must honour it
# --------------------------------------------------------------------------- #
def test_t2i_task_prompt_has_exactly_one_placeholder() -> None:
    """``.format(user_text)`` substitutes the draft and unescapes the JSON."""
    template = h3.load_task_prompt("t2i")
    assert template.count("{}") == 1
    assert "input_text: {}" in template
    message = h3.build_user_message("t2i", "a cat on a roof")
    assert "input_text: a cat on a roof" in message
    # The doubled braces are gone: the JSON example is now real JSON.
    assert "{{" not in message
    assert '{"expanded_caption"' in message


def test_edit_task_prompt_must_not_be_formatted() -> None:
    """The edit prompt has a JSON example; ``.format()`` would raise."""
    template = h3.load_task_prompt("edit")
    with pytest.raises(KeyError):
        template.format("a cat on a roof")
    # There is no single-part user message for the edit mode: the prompt
    # and the instruction travel as separate parts.
    with pytest.raises(ValueError):
        h3.build_user_message("edit", "make it blue")
    content = h3.build_user_content("edit", "make it blue")
    assert content[0]["text"] == template
    assert "make it blue" in content[1]["text"]


def test_the_guide_modes_send_the_draft_as_the_user_turn() -> None:
    """The guide is the system prompt; the user turn is the brief."""
    assert h3.build_user_message("t2i_guide", "a red cup") == "a red cup"
    system = h3.build_system_prompt("t2i_guide")
    assert "Text-to-image prompt writing guide" in system
    assert "a red cup" not in system


def test_allowed_ratios_match_the_official_request_params() -> None:
    """The code-side enum is the official enum."""
    params = h3.load_request_params()
    official = params["response_format"]["json_schema"]["schema"]["properties"][
        "aspect_ratio"
    ]["enum"]
    assert tuple(official) == h3.ALLOWED_T2I_RATIOS
    assert len(h3.ALLOWED_T2I_RATIOS) == 10


def test_the_system_prompt_is_the_official_one() -> None:
    """One published system prompt, reused by the edit mode."""
    system = h3.load_system_prompt()
    assert "expanded_caption and aspect_ratio" in system
    assert not system.endswith("\n")
    assert h3.build_system_prompt("t2i") == h3.build_system_prompt("edit")


def test_the_guide_contract_is_lifted_from_the_official_skill() -> None:
    """The output format comes from ``SKILL.md``, not from this module."""
    contract = h3.load_guide_output_contract()
    assert "Aspect ratio: 3:4" in contract
    assert "between 1:3 and 3:1" in contract


# --------------------------------------------------------------------------- #
# Message building
# --------------------------------------------------------------------------- #
def test_edit_user_message_interleaves_the_reference_media_parts() -> None:
    """Each image is preceded by its label; the instruction comes last."""
    content = h3.build_user_content(
        "edit",
        "make the cup blue",
        image_urls=["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"],
        reference_ratios=["3:2", "4:3"],
        aspect_ratio="16:9",
    )
    assert isinstance(content, list)
    kinds = [part["type"] for part in content]
    # The official edit task prompt comes first, then a text/image pair per
    # reference, then the instruction.
    assert kinds == ["text", "text", "image_url", "text", "image_url", "text"]
    assert content[0]["text"] == h3.load_task_prompt("edit")
    assert content[1]["text"] == "Reference Media: <Picture 1> aspect_ratio 3:2"
    assert content[3]["text"] == "Reference Media: <Picture 2> aspect_ratio 4:3"
    last = content[5]["text"]
    assert last.startswith("User Editing Instruction:")
    assert "make the cup blue" in last
    assert last.rstrip().endswith("User Aspect Ratio Selection: 16:9")


def test_edit_user_message_marks_an_unknown_ratio() -> None:
    """The official text says the metadata reads 'unavailable'."""
    content = h3.build_user_content(
        "edit", "edit it", image_urls=["data:image/jpeg;base64,AAA"]
    )
    assert content[1]["text"].endswith("aspect_ratio unavailable")


def test_image_modes_reject_images_they_cannot_use() -> None:
    with pytest.raises(ValueError):
        h3.build_user_content("t2i", "a cat", image_urls=["data:image/jpeg;base64,A"])


def test_more_than_sixteen_references_is_an_error() -> None:
    with pytest.raises(ValueError):
        h3.build_user_content(
            "edit",
            "edit it",
            image_urls=[f"data:image/jpeg;base64,{n}" for n in range(17)],
        )


def test_an_empty_draft_is_refused() -> None:
    with pytest.raises(ValueError):
        h3.build_user_message("t2i", "   ")


# --------------------------------------------------------------------------- #
# H1 - one JSON object, two string values
# --------------------------------------------------------------------------- #
def test_thinking_and_fences_are_stripped() -> None:
    raw = (
        "<think>let me look at the brief</think>\n"
        "```json\n"
        '{"expanded_caption": "A red cup on a table.", "aspect_ratio": "4:3"}\n'
        "```"
    )
    result = h3.parse_expansion_reply(raw)
    assert result.caption == "A red cup on a table."
    assert result.aspect_ratio == "4:3"


def test_a_reply_without_json_is_an_error() -> None:
    with pytest.raises(h3.H3ImageReplyError):
        h3.parse_expansion_reply("I am afraid I cannot do that.")


def test_a_reply_without_a_caption_is_an_error() -> None:
    with pytest.raises(h3.H3ImageReplyError):
        h3.parse_expansion_reply('{"aspect_ratio": "4:3"}')
    with pytest.raises(h3.H3ImageReplyError):
        h3.parse_expansion_reply('{"expanded_caption": "   ", "aspect_ratio": "4:3"}')


def test_extra_keys_are_reported_but_do_not_throw_the_caption_away() -> None:
    result = h3.parse_expansion_reply(
        '{"expanded_caption": "A red cup.", "aspect_ratio": "4:3", "notes": "x"}'
    )
    assert result.caption == "A red cup."
    assert any("notes" in warning for warning in result.warnings)


# --------------------------------------------------------------------------- #
# H2 - the two key names
# --------------------------------------------------------------------------- #
def test_both_key_names_are_accepted() -> None:
    a = h3.parse_expansion_reply(
        '{"expanded_caption": "A red cup.", "aspect_ratio": "4:3"}', mode="t2i"
    )
    b = h3.parse_expansion_reply(
        '{"expanded_caption": "A red cup.", "target_aspect_ratio": "4:3"}', mode="edit"
    )
    assert a.aspect_ratio == b.aspect_ratio == "4:3"


def test_target_aspect_ratio_wins_when_both_are_present() -> None:
    result = h3.parse_expansion_reply(
        '{"expanded_caption": "A red cup.", "aspect_ratio": "1:1",'
        ' "target_aspect_ratio": "4:3"}',
        mode="edit",
    )
    assert result.aspect_ratio == "4:3"


def test_a_reply_without_any_ratio_key_is_an_error() -> None:
    with pytest.raises(h3.H3ImageReplyError):
        h3.parse_expansion_reply('{"expanded_caption": "A red cup."}')


# --------------------------------------------------------------------------- #
# H3 / H4 - the ratio contract
# --------------------------------------------------------------------------- #
def test_t2i_ratio_outside_the_enum_is_snapped_into_it() -> None:
    result = h3.parse_expansion_reply(
        '{"expanded_caption": "A red cup.", "aspect_ratio": "2:1"}', mode="t2i"
    )
    assert result.aspect_ratio in h3.ALLOWED_T2I_RATIOS
    assert result.warnings


def test_t2i_never_leaves_the_official_enum() -> None:
    for value in ("7:3", "banana", "1024:768", "21:9"):
        result = h3.parse_expansion_reply(
            json.dumps({"expanded_caption": "A red cup.", "aspect_ratio": value}),
            mode="t2i",
        )
        assert result.aspect_ratio in h3.ALLOWED_T2I_RATIOS


def test_edit_ratio_is_reduced_to_lowest_terms() -> None:
    result = h3.parse_expansion_reply(
        '{"expanded_caption": "A red cup.", "target_aspect_ratio": "1920:1080"}',
        mode="edit",
    )
    assert result.aspect_ratio == "16:9"


def test_edit_ratio_must_be_two_positive_integers() -> None:
    result = h3.parse_expansion_reply(
        '{"expanded_caption": "A red cup.", "target_aspect_ratio": "4:0"}', mode="edit"
    )
    assert result.aspect_ratio != "4:0"


def test_adaptive_never_reaches_the_output() -> None:
    result = h3.parse_expansion_reply(
        '{"expanded_caption": "A red cup.", "target_aspect_ratio": "adaptive"}',
        mode="edit",
        requested_ratio="16:9",
    )
    assert result.aspect_ratio == "16:9"
    assert "adaptive" not in result.aspect_ratio


def test_adaptive_falls_back_to_the_reference_ratio() -> None:
    """In adaptive mode the official edit prompt keeps the reference's ratio."""
    result = h3.parse_expansion_reply(
        '{"expanded_caption": "A blue cup.", "target_aspect_ratio": "adaptive"}',
        mode="edit",
        fallback_ratio="16:9",
    )
    assert result.aspect_ratio == "16:9"
    assert any("adaptive" in warning for warning in result.warnings)


# --------------------------------------------------------------------------- #
# H5 / H6 / H7 - the caption
# --------------------------------------------------------------------------- #
def test_a_quote_inside_the_draft_must_survive() -> None:
    draft = 'Design a dashboard titled "Quarterly sales" with the total "39".'
    good = (
        '{"expanded_caption": "A clean dashboard titled \\"Quarterly sales\\" '
        'showing the total \\"39\\".", "aspect_ratio": "16:9"}'
    )
    result = h3.parse_expansion_reply(good, user_text=draft)
    assert result.missing_literals == ()
    assert result.ok


def test_a_dropped_literal_is_a_hard_failure() -> None:
    draft = 'Design a dashboard titled "Quarterly sales" with the total "39".'
    bad = (
        '{"expanded_caption": "A clean dashboard showing the sales of the '
        'quarter.", "aspect_ratio": "16:9"}'
    )
    with pytest.raises(h3.H3ImageReplyError):
        h3.parse_expansion_reply(bad, user_text=draft)
    lenient = h3.parse_expansion_reply(bad, user_text=draft, strict=False)
    assert lenient.missing_literals == ("Quarterly sales", "39")
    assert not lenient.ok


def test_an_apostrophe_is_not_a_literal() -> None:
    assert h3.extract_quoted_literals("it's a 'quoted' word") == ["quoted"]


def test_curly_quotes_count_as_literals() -> None:
    assert h3.extract_quoted_literals("\u201cQuarterly sales\u201d") == ["Quarterly sales"]


def test_non_english_prose_is_reported() -> None:
    result = h3.parse_expansion_reply(
        '{"expanded_caption": "\u4e00\u53ea\u7ea2\u676f\u5b50\u5728\u684c\u4e0a\u3002",'
        ' "aspect_ratio": "4:3"}'
    )
    assert any("non-English" in warning for warning in result.warnings)


def test_a_cjk_literal_the_user_asked_for_is_fine() -> None:
    result = h3.parse_expansion_reply(
        '{"expanded_caption": "A sign reading \\"\u4f60\u597d\\" on the wall.",'
        ' "aspect_ratio": "4:3"}',
        user_text='a sign saying "\u4f60\u597d"',
    )
    assert not any("non-English" in warning for warning in result.warnings)


# --------------------------------------------------------------------------- #
# H8 / H13 / H14 - the markers
# --------------------------------------------------------------------------- #
def test_a_backticked_marker_is_reported() -> None:
    result = h3.parse_expansion_reply(
        '{"expanded_caption": "A cup `<Picture 1>` on a table.", "aspect_ratio": "4:3"}'
    )
    assert any("backticks" in warning for warning in result.warnings)


def test_every_reference_image_must_be_referenced() -> None:
    result = h3.parse_expansion_reply(
        '{"expanded_caption": "A cup from <Picture 1>.", "aspect_ratio": "4:3"}',
        reference_count=2,
    )
    assert any("<Picture 2>" in warning for warning in result.warnings)


def test_a_bare_grid_is_reported() -> None:
    result = h3.parse_expansion_reply(
        '{"expanded_caption": "A 3x3 grid of tiles.", "aspect_ratio": "1:1"}'
    )
    assert any("grid" in warning for warning in result.warnings)


def test_a_named_grid_is_accepted() -> None:
    result = h3.parse_expansion_reply(
        '{"expanded_caption": "A 3x3 (rows x columns) grid of tiles.",'
        ' "aspect_ratio": "1:1"}'
    )
    assert not any("grid" in warning for warning in result.warnings)


# --------------------------------------------------------------------------- #
# The guide modes
# --------------------------------------------------------------------------- #
def test_the_guide_reply_is_split_into_prompt_and_ratio() -> None:
    raw = (
        "```text\nA red cup on a wooden table.\n```\n\nAspect ratio: 3:4"
    )
    result = h3.parse_guide_reply(raw)
    assert result.caption == "A red cup on a wooden table."
    assert result.aspect_ratio == "3:4"


def test_a_guide_ratio_outside_the_official_range_is_reported() -> None:
    raw = "A tall poster.\n\nAspect ratio: 1:6"
    result = h3.parse_guide_reply(raw)
    assert any("1:3" in warning for warning in result.warnings)


def test_a_guide_ratio_line_may_carry_emphasis_or_a_period() -> None:
    raw = "A tall poster.\n\n**Aspect ratio:** 1:6."
    result = h3.parse_guide_reply(raw)
    assert result.aspect_ratio != "adaptive"


def test_a_guide_ratio_written_as_a_latex_fraction_is_read() -> None:
    """Models do write ``\\frac{5}{4}`` -- it is still 5:4."""
    raw = "A tall poster.\n\nAspect ratio: $ \\frac{5}{4} $"
    result = h3.parse_guide_reply(raw)
    assert result.aspect_ratio == "5:4"


def test_a_guide_reply_without_a_ratio_line_falls_back_to_the_reference() -> None:
    result = h3.parse_guide_reply(
        "A tall poster.", fallback_ratio="16:9"
    )
    assert result.aspect_ratio == "16:9"
    assert any("Aspect ratio" in warning for warning in result.warnings)


def test_a_guide_reply_with_several_variants_keeps_the_first() -> None:
    raw = (
        "First variant.\n\nAspect ratio: 4:3\n\nSecond variant.\n\nAspect ratio: 1:1"
    )
    result = h3.parse_guide_reply(raw)
    assert result.aspect_ratio == "4:3"
    assert result.caption == "First variant."
    assert any("variants" in warning for warning in result.warnings)


def test_a_guide_reply_without_a_ratio_line_is_reported() -> None:
    result = h3.parse_guide_reply("A tall poster.")
    assert result.aspect_ratio == "adaptive"
    assert any("Aspect ratio" in warning for warning in result.warnings)


def test_text_outside_the_json_object_is_reported() -> None:
    result = h3.parse_expansion_reply(
        '{"expanded_caption": "A red cup.", "aspect_ratio": "4:3"} Hope this helps!'
    )
    assert result.caption == "A red cup."
    assert any("outside the JSON" in warning for warning in result.warnings)


# --------------------------------------------------------------------------- #
# H12 - the reference-image metadata
# --------------------------------------------------------------------------- #
def test_the_reference_ratio_is_computed_from_the_pixels() -> None:
    assert h3.image_ratio(1920, 1080) == "16:9"
    assert h3.image_ratio(1000, 1000) == "1:1"
    assert h3.image_ratio(3024, 4032) == "3:4"


# --------------------------------------------------------------------------- #
# The mode table
# --------------------------------------------------------------------------- #
def test_every_mode_maps_to_a_label_and_back() -> None:
    assert len(h3.MODES) == len(h3.MODE_CODES) == 4
    for code, label in zip(h3.MODE_CODES, h3.MODES):
        assert h3.parse_mode(label) == code
        assert h3.load_task_prompt(code)


def test_the_image_modes_are_the_two_edit_modes() -> None:
    assert h3.is_image_mode("edit")
    assert h3.is_image_mode("edit_guide")
    assert not h3.is_image_mode("t2i")


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError):
        h3.load_task_prompt("nope")
