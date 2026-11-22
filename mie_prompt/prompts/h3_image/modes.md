# Modes, output contract and input spec — MiniMax-H3-Image prompt enhancer

Family code: `h3_image`. Shell: `nodes/llm/h3_image_prompt_generator.py`
(+ shim `h3_image_prompts.py`).

## 1. Model identity and consumer

| Item | Value |
|---|---|
| Model | `MiniMax-H3-Image` (MiniMax, Nanonoble Pte. Ltd.), weights 2026-11-10 |
| Official PE | `MiniMaxAI/MiniMax-H3-Image-PE` — Qwen3.8-27B FP4 (17.9 GB) |
| ComfyUI PE substitute | `qwen3.5_9b_w4a8_gptq.safetensors` (7.34 GB), from `Comfy-Org/Qwen3.5` |
| Enhanced prompt feeds | `MiniMax H3 Image Conditioning` → `KSampler` (euler / simple / 6 steps / CFG 1 / denoise 1 / **no negative**) |
| Official templates | `image_minimax_h3_image_t2i.json`, `image_minimax_h3_image_edit.json` (Comfy-Org/workflow_templates) |
| ComfyUI version | 0.45.0+ , template pack 0.11.104+ |

The node produces **prompt text and an aspect ratio**. It does **not** produce
pixels, and it does **not** produce a negative prompt — H3-Image runs CFG=1
and the official templates carry no negative slot.

Two facts the node must respect:

- **A local server does not expand prompts.** Only the MiniMax API does
  (H3-Image-Context-IR). This node *is* that expansion step for local use, so
  a short Chinese prompt must be expanded before it reaches the sampler.
- **The edit PE has no published weights.** MiniMax published only the edit
  task prompt. The edit mode of this node therefore drives an external LLM with
  that prompt; it never needs the PE checkpoint itself.

## 2. Modes

Two modes, single-select dropdown, mutually exclusive. The mode axis is the
task axis — it mirrors the two official templates one to one. There is no
separate reference-generation template: the edit task natively takes the
picture being edited plus up to 16 reference images, so "参考生图" is `edit`.

| code | display (zh / en) | what it does | LLM call | official artifact |
|---|---|---|---|---|
| `t2i` | 文生图增强 / Text-to-image | short brief → English caption **+ aspect ratio** (JSON) | yes | `_pe_t2i_task.txt` + `_pe_t2i_system.txt` |
| `edit` | 图像编辑增强 / Image edit | short edit instruction + reference images → English caption **+ target ratio** (JSON) | yes (multimodal) | `_pe_edit_task.txt` + `_pe_t2i_system.txt` |

> 2026-11-22 (user ruling): the `t2i_guide` / `edit_guide` modes were removed
> from the dropdown. They drove the official prompt-writing skill
> (`_skill_t2i.txt` / `_skill_edit.txt` + `_skill_main.md`, free text plus an
> `Aspect ratio: W:H` line) — the same task with a different output format,
> which read as a second task axis and confused users. The skill files stay
> shipped verbatim as reference material; the code that composed them into
> system prompts and parsed their replies was deleted with the modes.

### `t2i` mode

Messages:

- system = `_pe_t2i_system.txt` (rstrip trailing `\n`, exactly like
  `expand_prompt.py`)
- user = `_pe_t2i_task.txt`.format(user_text)

`.format()` — **not** `.replace()`. The task prompt contains exactly one
single-brace token (`{}`) and four escaped JSON examples (`{{...}}`); `.format()`
resolves the former to the user's text and the latter to real single braces.
Verified: `_pe_t2i_task.txt`.format("USER")` succeeds and yields
`{"expanded_caption":"ONE_DETAILED_ENGLISH_CAPTION","aspect_ratio":"ALLOWED_RATIO"}`.

### `edit` mode

The official edit task prompt describes a **multi-part user message**, not a
single template with a placeholder. Reproduce that structure:

```
user turn =
  [ text part: "Reference Media: <Picture 1> aspect_ratio 3:2" ]
  [ image part: reference 1 ]
  [ text part: "Reference Media: <Picture 2> aspect_ratio 4:3" ]
  [ image part: reference 2 ]
  …
  [ text part: final, containing the User Editing Instruction (JSON string)
                and the User Aspect Ratio Selection ]
```

The prompt is explicit about this:

> "Each input image is immediately preceded by a Reference Media text part
> containing its `<Picture N>` label and aspect_ratio W:H."
>
> "One short editing instruction … The final text part supplies it as User
> Editing Instruction (JSON string)."
>
> "User Aspect Ratio Selection: any concrete positive-integer W:H ratio, or
> adaptive, supplied in the final text part."

**Do not** `.format()` `_pe_edit_task.txt`: it contains a 71-character JSON
example `{"expanded_caption":"COMPLETE_DESCRIPTION","target_aspect_ratio":"W:H"}`
and `.format()` raises `KeyError: '"expanded_caption"'`. The instruction goes in
the final text part instead — which is what the prompt asks for.

The ComfyUI template instead appends `User input: {input text}\noutput_text:`
to the task prompt and uses a `StringReplace` node. That is ComfyUI's
adaptation of the same idea. Either is defensible; the multi-part form is the
one the official text describes, so prefer it. If the tail form is used
instead, it must be built at runtime and must not be shipped as a modified copy
of the official file.

**The reference images go to the LLM only.** The image model receives them
separately through `MiniMaxH3ImageConditioning`; `image_1` is the picture being
edited and the rest are references. The LLM needs them because the edit prompt
requires the model to *look* at the sources (`<Picture N>`, `<Subject N>`,
transcribed source text). A text-only model cannot run the `edit` mode.

### The retired `_guide` path (reference only)

The official skill files remain shipped and are still worth reading as
prompt-writing reference:

- `_skill_main.md` (SKILL.md): the writing rules — task before style, quote
  every word that appears, get the data right, grids by rows and columns;
  ratios between 1:3 and 3:1.
- `_skill_t2i.txt` / `_skill_edit.txt`: the guides with worked examples.

Note the tag convention the guides use: the **user side** refers to inputs as
`@Picture 1` (`edit-en.txt` examples: *"Reference image: `@Picture 1` shows a
red cup…"*), while the **model side** uses `<Picture 1>` (the same file's
prompt blocks). Do not let the node rewrite one into the other.

## 3. Output contract

Every rule is marked **H** = must always hold (enforced in code, never depends
on the LLM) or **P** = preference (system prompt only).

### Must always hold (code-side normalizer / validator)

| # | Rule | Where it is enforced |
|---|---|---|
| H1 | The reply is exactly one JSON object with exactly two string values. Strip a ```` ``` ```` fence and any `<think>…</think>` block first. | reply parser |
| H2 | **Key normalisation.** Accept `expanded_caption` + `aspect_ratio` **or** `expanded_caption` + `target_aspect_ratio`; canonical output pair is `(expanded_caption, aspect_ratio)`. `target_aspect_ratio` wins if both are present. | normalizer |
| H3 | `t2i`: the ratio is one of the ten official values — `1:1 5:4 4:3 3:2 16:9 21:9 4:5 3:4 2:3 9:16`. Reject or snap; never silently invent a value outside the enum. | validator |
| H4 | `edit`: the ratio is `W:H`, two positive integers, no spaces, reduced to lowest terms; `adaptive` must never reach the output. | normalizer + validator |
| H5 | `expanded_caption` is a non-empty string; strip it. | validator |
| H6 | The prose is English. CJK characters may appear only inside a quoted literal the user asked to render. | validator (warning) |
| H7 | **Every literal the user put in quotes must survive verbatim** — same characters, same spaces, same case, same punctuation, same language. | validator (hard) |
| H8 | `<Picture N>` / `<Subject N>` markers stay bare — never wrapped in backticks, never renamed. | validator |
| H9 | `edit` output has the three-part shape: Part 1 element definitions → Part 2 a sentence starting `The target image` plus one independent relationship clause per Subject → Part 3 changed/new details only. | validator (structural) |
| H10 | No markdown fence, no thinking tags, no prose outside the JSON object. | reply parser |
| H11 | No negative prompt is produced or emitted (H3-Image has no negative slot). | node has no such output |
| H12 | Each reference image's aspect-ratio metadata is computed **by the node** from its pixel size: EXIF orientation applied, then reduced to the lowest integer W:H; "unavailable" if unknown. | node code, never asked of the LLM |
| H13 | Every input image is referenced as `<Picture N>`, N counted from 1 in input order, with no gaps. | validator |
| H14 | Grid notation names rows and columns: `2 rows and 3 columns` or `3x3 (rows x columns)`. A bare `2×3` is a warning. | validator (warning) |

### Preference (system prompt only)

- **Task before style. Layout before details.** Open with what the image is and
  what it is for; give the number of subjects, their positions, the viewpoint
  and the empty space; style last. (official SKILL.md)
- **Style through visible details** — colour, material, light, texture instead
  of "premium" / "high-end". (official SKILL.md)
- **Quote every word that appears** — put on-image copy in quotes, in its own
  language, with its position, size hierarchy and line breaks. (The *hard* half
  of this rule is H7: whatever the user already quoted must survive.)
- **Say when there is no text** — "The image contains no text, letters or
  numbers." for t2i; "no other lettering is added" for an edit. (official SKILL.md)
- **Get data right** — exact values, units, relationships, time range; mark
  example data as illustrative. (official SKILL.md)
- **Give each reference a role** — say what each image contributes, what stays
  consistent, what may change; when only a colour is borrowed, say the shape
  and material are not.
- **Separate what changes from what stays.**
- **One clear goal per round** in multi-round editing.
- Dense text / charts / diagrams: the official guide says render **two to four**
  and keep the best. That is a node-level workflow hint (tooltip), not a prompt
  rule.

## 4. Input spec

| Item | Spec |
|---|---|
| User draft | Free-form: a sentence, a paragraph, multi-line. Any language (Chinese is the normal case). |
| Verbatim contract | Text inside single or double quotes is preserved character-for-character (H7). The guides use both `'` and `"`; treat both as literal. |
| Images | `edit`: 0–16 reference images, official hard limit 16. Node default cap **8** (the ComfyUI-side prompt-assist nodes top out around 9), exposed as a control. `t2i`: no image needed; an optional image may be used as a *reference description* source (phase 2). |
| Aspect-ratio control | `t2i`: `adaptive` + the ten official values only — matches the PE's own enum. `edit`: `adaptive` + **any** positive-integer W:H — the edit PE explicitly refuses a preset list. |
| Seed | Provide a seed control so the same draft + same seed reproduces the same enhancement. The upstream PE is non-deterministic across runs; the node cannot promise bit-identical output, but a fixed seed + fixed model is the best available lever. |
| Output ports | `prompt` (STRING), `aspect_ratio` (STRING), `raw_reply` (STRING, for debugging), and for `edit` a `subjects` count or the `<Subject N>` list is optional — not required for v1. |

### Aspect ratio, pixels and canvas — the node must not confuse them

The PE chooses a **ratio**, never pixels. Pixel selection belongs to the image
workflow (`ResolutionSelector` / `MiniMax H3 Image Conditioning`), and:

- the effective working range is about **4 MP to 20 MP**, best around **16 MP**;
- 1:1 → 4096×4096, 4:3 → 4736×3552, 3:2 → 5024×3360, 16:9 → 5472×3072,
  21:9 → 6272×2688; every edge is a multiple of **32 px**;
- the official template deliberately starts at **4 MP**, below the native
  resolution, so low-VRAM machines do not OOM.

Put this in the node tooltip, not in the prompt.

## 5. LLM call facts

### Official PE sampling parameters (`_pe_t2i_request_params.json`, verbatim)

```json
{
  "reasoning_effort": "xhigh",
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 0.0,
  "repetition_penalty": 1.0,
  "chat_template_kwargs": { "enable_thinking": true, "preserve_thinking": true },
  "max_tokens": 4096,
  "response_format": { "type": "json_schema", "strict": true, … }
}
```

### What the MieNodes connector actually forwards

No connector in this project forwards anything beyond `temperature` and
`max_tokens`, and the bundled service layer deliberately does not carry the
rest. So:

| Official parameter | Forwarded? | Node action |
|---|---|---|
| `temperature: 1.0` | yes | default the node's temperature to **1.0** (note: MOSS defaults to 0.7 because MOSS's official value is 0.7 — do not copy that number here) |
| `max_tokens: 4096` | yes | default to **16384** like the other nodes in this project: reasoning models spend part of the budget thinking before the caption, and a reply that runs out mid-caption is the common failure. Expose as a control (min 64, max 32768). |
| `top_p` 0.95 / `top_k` 20 / `min_p` 0 / `presence_penalty` 0 / `repetition_penalty` 1 | **no** | document in the node tooltip / README that the official values are not carried by the connector |
| `reasoning_effort: xhigh` | **no** | same note |
| `enable_thinking` / `preserve_thinking` | **no** | same note |
| `response_format: json_schema` | **no** | the node parses and validates the JSON itself (H1–H5) instead of relying on the provider's schema mode |

### Empty replies and truncation

- Keep `allow_reasoning_fallback=False` on the connector call (as
  `MossImagePromptEnhancer` does). A provider that returns only its reasoning
  channel hands back text with no caption; refusing that fallback turns the
  failure into the explicit empty-reply error instead of a silent empty string.
- The error message must name the real cause (budget exhausted while thinking)
  and point at `max_tokens`, in the style of the MOSS node.
- The project's budget assistant (auto-doubling on a truncated reply) should
  cover this node too — **confirm** that the node is registered with it.
- Upstream's own check is `finish_reason == "stop"` plus a non-empty
  `reasoning_content`. The MieNodes connector does not expose `reasoning_content`
  for every provider, so treat a non-empty `content` as the pass condition and
  note the difference.

### Retry on an unusable reply

One corrective retry, same shape as the MOSS layout parser: on a parse
failure, append the assistant turn plus a short corrective user turn
("Reply with ONE JSON object and nothing else: no prose, no code fence") and
call again. Cap at 2 attempts.

## 6. Hard constraints to encode in the node

1. **No negative prompt output** (H11).
2. **6 steps / CFG 1 are fixed by the sampler, not by this node** — the node
   must not offer a steps control; it may state the pair in the tooltip.
3. **`image_1` is the picture being edited**; the slot order sets the
   `<Picture n>` numbers. If the node also serves the ComfyUI side, this
   ordering must be documented in the tooltip so users connect the right image.
4. **The edit prompt must not be `.format()`ed** (see §2).
5. **Reference images are visible to the LLM in `edit` mode**; a text-only model
   makes `edit` meaningless — surface this as an error or a warning, not a
   silent degradation.
6. **Never rewrite `<Picture N>` into `@Picture N`** or vice versa.
