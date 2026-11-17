# Upstream provenance — MiniMax-H3-Image prompt enhancer

Snapshot date: **2026-11-17** (Asia/Shanghai).

Family code: **`h3_image`** (new family). The existing `h3` family in
`mie_prompt/prompts/` is the **video** model (T2VA/I2VA guides,
`system_t2v.txt`); this one is the **image** model and must not collide with it.

## Files

All seven files below are **official, verbatim — never edit**. Nothing in this
folder is MieNodes-authored.

| File | Source | Status |
|---|---|---|
| `_pe_t2i_system.txt` | `MiniMaxAI/MiniMax-H3-Image-PE` → `prompt/t2i/system_prompt.txt` | **official, verbatim** |
| `_pe_t2i_task.txt` | `MiniMaxAI/MiniMax-H3-Image-PE` → `prompt/t2i/task_prompt.txt` | **official, verbatim** |
| `_pe_t2i_request_params.json` | `MiniMaxAI/MiniMax-H3-Image-PE` → `prompt/t2i/request_params.json` | **official, verbatim** |
| `_pe_edit_task.txt` | `MiniMaxAI/MiniMax-H3-Image-PE` → `prompt/edit/task_prompt.txt` | **official, verbatim** |
| `_skill_main.md` | `MiniMax-AI/MiniMax-H3-Image` → `skills/h3-image-prompt-writing/SKILL.md` | **official, verbatim** |
| `_skill_t2i.txt` | `MiniMax-AI/MiniMax-H3-Image` → `skills/h3-image-prompt-writing/references/t2i-en.txt` | **official, verbatim** |
| `_skill_edit.txt` | `MiniMax-AI/MiniMax-H3-Image` → `skills/h3-image-prompt-writing/references/edit-en.txt` | **official, verbatim** |

### Why the leading underscore

`mie_prompt/prompts/loader.py` skips `_`-prefixed files when building the
builtin-prompt list, and `list_usable_builtin_prompts()` drops anything
matching `\{[a-zA-Z_]*\}`. `_pe_t2i_task.txt` contains `{}` (matched → dropped),
but `_pe_edit_task.txt` contains only a 71-char JSON example
(`{"expanded_caption":...}`) that the placeholder regex does **not** match —
it would leak into the generic `CustomSystemPromptGenerator` dropdown and ship
an unsubstituted template to a user. The `_` prefix keeps the whole PE bundle
out of that dropdown. Keep it.

## Digests

Byte-level, verified against the official `prompt/manifest.json` shipped in the
same PE repo (four-way MATCH):

| File | bytes | chars | sha256 |
|---|---|---|---|
| `_pe_t2i_system.txt` | 327 | 327 | `818ab3f652ec36b3b489b55c29f072f87e14af703cd11a4f3bff75f423d0beac` |
| `_pe_t2i_task.txt` | 15821 | 15813 | `890213ad61e143bbd5b7906a802430296e82ca5c61ba16df717dcef5477bbcf4` |
| `_pe_t2i_request_params.json` | 1034 | 1034 | `dfc608bf0c02dd6ce7463be636aa8b148aef200513fa4c46e653ee2e5affbe3e` |
| `_pe_edit_task.txt` | 38489 | 38411 | `c5695dbfc07a9e65555cef67d92467c613371cc2039b96f057022246a30a57b2` |
| `_skill_main.md` | 8416 | 8413 | `d76fb522208149078d284cdab11260c11d974ec12db921f25e3c0fa76aa3f9cf` |
| `_skill_t2i.txt` | 9061 | 9047 | `a0fe54167532464d5c4f40517c07e350c9631fc97fc1299b56f9f1d74b05a78b` |
| `_skill_edit.txt` | 10296 | 10282 | `4f08883dcf390d388fe077959d077834e6514eed73c5385aa71f9e962521aff3` |

The first four sha256 values are the ones recorded in the official
`prompt/manifest.json`; assert against those in the MieNodes test.

## Two official variants of the same task prompt — pick the PE one

The ComfyUI templates ship a **modified** copy of the PE task prompts:

- `templates/image_minimax_h3_image_t2i.json` → subgraph node 20, 15823 chars
- `templates/image_minimax_h3_image_edit.json` → node 39, 38449 chars

Diff against the PE repo:

| | PE repo (shipped here) | ComfyUI template |
|---|---|---|
| t2i user-input slot | `input_text: {}` | `input_text: {input text}` |
| edit tail | *(none)* | appends `User input: {input text}` + `output_text:` |
| brace style | `{{...}}` (escaped for `.format()`) | `{{...}}` (left doubled, **never unescaped**) |

The PE repo variant is the one the official script
(`scripts/expand_prompt.py`) drives, and it is the only one whose braces are
consistent. **Ship the PE variant and substitute with `.format(user_text)`.**
Do not copy the ComfyUI template's task prompt, and do not switch to
`.replace("{input text}", ...)` on the PE variant: `.format()` is what turns
`{{` into `{` and `{}` into the user's text in one step.

### `scripts/expand_prompt.py` — the official reference behaviour

`MiniMaxAI/MiniMax-H3-Image-PE/scripts/expand_prompt.py` (5,808 B) is the
upstream reference for how the PE is driven. It does:

```python
system_prompt = (PROMPT_DIR / "system_prompt.txt").read_text(encoding="utf-8").rstrip("\n")
template      = (PROMPT_DIR / "task_prompt.txt").read_text(encoding="utf-8")
body["messages"] = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": template.format(prompt)},   # <- .format(), not .replace()
]
```

and validates the reply (`parse_response`):

- exactly one choice; `finish_reason == "stop"` (else *"generation did not finish"*)
- `reasoning_content` must be non-empty (else *"no thinking section; start the
  server with `--reasoning-parser qwen3`")
- `content` must be JSON with **exactly** the keys `expanded_caption` and
  `aspect_ratio`
- `aspect_ratio` must be in the enum from `request_params.json`
- `expanded_caption` must be a non-empty string (stripped)

## The one thing the official artifacts disagree about

`_pe_edit_task.txt` (edit PE) uses the key **`target_aspect_ratio`** (7
occurrences, 0 occurrences of `"aspect_ratio"`), while `_pe_t2i_system.txt`
(the only system prompt MiniMax published, and the one the ComfyUI edit
template reuses verbatim) says *"exactly one JSON object with exactly two
keys: expanded_caption and aspect_ratio"*.

So the official edit PE carries an internal key conflict. The ComfyUI template
sidesteps it by extracting only `expanded_caption`
(`JsonExtractString`, widget = `expanded_caption`).

**Node rule: accept both key names and normalise to a canonical pair.** See
`modes.md` § output contract, rule H2. Do not ship a patched edit system
prompt — accepting both keys is code-side and keeps 100 % of the official text.

## Allowed aspect ratios (from `_pe_t2i_request_params.json`)

```
1:1  5:4  4:3  3:2  16:9  21:9  4:5  3:4  2:3  9:16
```

Ten values, and the t2i PE task prompt says *"Choose exactly one of these
allowed ratios"*. The **edit** PE task prompt explicitly does **not** use a
preset list: *"Any positive-integer W:H ratio is allowed; it is not limited to
a preset list"* and *"Do not snap to preset ratios or clamp to a supported
range."* — it only requires W:H, two positive integers, no spaces, reduced.

## Public links

- <https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE>
- <https://huggingface.co/MiniMaxAI/MiniMax-H3-Image>
- <https://github.com/MiniMax-AI/MiniMax-H3-Image>
- <https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/main/skills/h3-image-prompt-writing/SKILL.md>
- <https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_t2i.json>
- <https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_edit.json>
- <https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/main/LICENSE>

## Licence note — the files are vendored, with the terms attached

The PE repo and the prompt-writing skill ship under the
**MiniMax-H3-Image Community License Agreement**, whose *Applicable Territory*
is "worldwide, excluding the Excluded Territories" — **the European Union, the
United Kingdom, the Republic of Korea and the United States of America** — and
whose §IV.1 requires separate written authorisation above USD 1,000,000 annual
revenue or for providing a hosted service to third parties. §V.4 forbids use,
copying, modification, distribution or display **outside** the Applicable
Territory.

These are documentation-side text files rather than model weights, but the
agreement's own definitions put them inside its scope: "Documentation" means
the publicly released specifications and documentation, "Materials" means
MiniMax-H3-Image **and the Documentation**, and "MiniMax-H3-Image Works" is
the Materials. They are licensed text, not free text.

**Decision (project owner, 2026-11-17): vendor them, with the terms
attached.** This folder therefore ships, next to the seven files:

* `_LICENSE-MiniMax-H3-Image.txt` — the agreement, copied verbatim
  (sha256 `0ec3bb5e6d9e885fa2aaf15e84689e50fd51f42b53fa24d3a64ed9d8c76d423b`),
  so every recipient gets a copy of it (§III.1);
* `NOTICE` — the notice §III.4 requires ("MiniMax-H3-Image is licensed under
  the MiniMax-H3-Image Community License Agreement, Copyright © 2026
  MiniMax. All Rights Reserved."), plus the file-to-source map, the digests,
  the attribution §IV.2 asks of a commercial product, and the territorial
  terms of §V.4 and §IV.1.

The alternative — loading the text at runtime from the user's own HF
checkout, which pushes the licence onto the end user — was considered and
rejected: it costs the user a manual download step, and it does not change
the fact that the node ships MiniMax's text either way. The files stay as
separate, byte-identical files rather than embedded in the Python source,
because the sha256 digests above are what prove they are the official
artifacts; embedding the same bytes in source would not change the licence
position and would cost that proof.

Nothing in the seven files is edited, so §III.2 (modified files must carry a
notice) does not apply to them.

> The §IV.1 threshold is a question for the owner, not for this folder: if
> the annualised revenue (affiliates included) exceeds USD 1,000,000, or a
> hosted service is offered to third parties, a separate written
> authorisation must be obtained from MiniMax at
> <https://platform.minimax.io/console/license> — vendoring or not.

---

## What the node composes at runtime

The seven files above are shipped byte-identical and are never edited. The
**composition** around them is decided in code (`mie_prompt/h3_image_prompts.py`)
and was verified live on 2026-11-17 against the local ComfyUI with a real
model (SiliconFlow, `Qwen/Qwen3.8-27B` for text, `Qwen/Qwen3-VL-32B-Thinking`
for the editing modes).

| mode | system message | user message |
|---|---|---|
| `t2i` | `_pe_t2i_system.txt` | `_pe_t2i_task.txt`.format(draft) — one `.format()`, which also un-escapes the doubled braces |
| `edit` | `_pe_t2i_system.txt` | `[ _pe_edit_task.txt, ("Reference Media: <Picture N> aspect_ratio W:H", image) per image, final part: "User Editing Instruction: <json string>" + "User Aspect Ratio Selection: <ratio>" ]` |
| `t2i_guide` | SKILL.md "Writing rules" + `_skill_t2i.txt` (without its `## Examples` section) + SKILL.md "Output format" | the draft |
| `edit_guide` | SKILL.md "Writing rules" + `_skill_edit.txt` (without its `## Examples` section) + SKILL.md "Output format" | the same interleaved parts as `edit`, without the task prompt |

Three of those decisions are worth spelling out, because they are not in
`modes.md` and each of them was forced by a live failure:

1. **The edit task prompt is the first text part of the user turn.** The
   official ComfyUI edit template passes the task prompt as the node's user
   text (node 39 → `TextGenerateV2.prompt`), and the official prompt itself
   says the instruction travels in the *final* text part. Sending only the
   reference-media parts drops the 38 KB of rules and the model answers with
   a couple of words instead of a caption. (Verified: with the prompt
   missing, the reply was 91 characters; with it, 975.)
2. **The guide modes send the rules, not the example gallery.** The guide
   files end with an `## Examples` section (a pendant photograph, a data
   dashboard). A model that sees them imitates *them*: a coffee-poster
   brief came back as the guide's pendant photo, and the user's quoted
   title was dropped. The rules above the examples are the instruction; the
   examples are reference.
3. **The guide modes send the draft as the user turn.** The guide is the
   system prompt; the draft is the request.

Ratio handling (code side, never the model):

* the user's choice wins over the model's;
* `t2i` never leaves the ten official ratios — anything else is snapped
  into the enum;
* `edit` / the guides: when the model answers `adaptive` (or nothing
  parsable), the fallback is the **reference image's own ratio**, because
  that is what the official edit prompt preserves in adaptive mode — not a
  blind 1:1;
* `adaptive` never reaches the output.

Two rules fail the run rather than shipping a wrong caption:

* **H7** — every literal the user quoted must survive verbatim. A reply
  that drops one triggers one corrective retry, and then a loud error.
* **H8 / H13** — the `<Picture N>` / `<Subject N>` markers stay bare and
  every reference image is referenced.

An empty reply (a reasoning model that thinks itself out of the answer) is
retried once before the run dies, with the error naming `max_tokens`.
