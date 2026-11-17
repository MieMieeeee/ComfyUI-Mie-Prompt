# Examples — MiniMax-H3-Image prompt enhancer

Five groups, each marking **must-change** (assertable in a test) versus
**style** (not assertable, do not assert).

Sources marked *official* are byte-copies from the upstream repositories
(verification in `UPSTREAM.md`). The last group is community evidence and is
**not** an upstream artifact.

---

## A. Official T2I pair (upstream `examples/` directory)

*Official — `MiniMax-AI/MiniMax-H3-Image/examples/`.*

Input (`instruction.t2i.txt`, 177 B):

> A rustic, overhead shot of a farm-to-table feast, showcasing a variety of
> colorful, seasonal vegetables, artisanal cheeses, and freshly baked bread on
> a weathered wooden table.

Expected output (`revised_prompt.t2i.txt`, 1,504 B) — the official PE run:

> The image is an overhead, top-down photograph of a rustic farm-to-table feast
> spread across a weathered, dark brown wooden table with visible grain, knots,
> and a worn finish. In the center, a large woven basket holds a freshly baked
> rustic loaf of bread with a cracked golden-brown crust and scattered flour,
> alongside a halved loaf. To the left, a large dark ceramic plate presents an
> assortment of colorful seasonal vegetables, including bright orange carrots,
> deep red heirloom tomatoes, purple cabbage, and a mound of vibrant green leafy
> kale and arugula. On the right side, a wooden board displays artisanal
> cheeses: a wedge of sharp aged cheddar, a round of creamy brie with a rind, a
> soft blue cheese, and a few scattered almonds and walnuts. Below the bread in
> the lower center, a shallow bowl contains roasted root vegetables, including
> golden potatoes and beets, sprinkled with fresh rosemary sprigs. At the upper
> right, a glass jar of golden olive oil stands beside a smaller jar of honey
> with a wooden dipper. In the lower left, a rustic cloth napkin holds a silver
> knife and fork. At the upper left, a bunch of purple and red grapes rests
> near a small bowl of black olives. Scattered herbs, including fresh thyme and
> sprigs of oregano, decorate the table. The lighting is natural and warm,
> coming from the upper left and casting soft, long shadows to the lower right,
> enhancing the texture of the bread, the moisture of the vegetables, and the
> overall cozy, bountiful atmosphere of the meal.

Upstream ran this at **4736×3552 (4:3)**.

| | |
|---|---|
| **Must change** | English prose; one continuous paragraph; concrete visible inventory (bread basket / vegetable plate / cheese board / herb bowl / oil jar / honey jar / napkin / grapes / olives / scattered herbs); light direction named ("from the upper left", "shadows to the lower right"); no markdown, no headings |
| **Style** | the exact food list, the exact ordering, the exact adjectives |
| **Assert** | H1 (one JSON object), H5 (non-empty), H6 (English), H3 (ratio in the enum) |

---

## B. Official EDIT pair (upstream `examples/` directory)

*Official — `MiniMax-AI/MiniMax-H3-Image/examples/`.*

Input (`instruction.edit.txt`, 305 B):

> Take the over-ear headphones off the man on the left and fill in his hair
> where the band flattened it. Have both hosts lean toward each other and
> fist-bump directly above the "MONEY TALKS" desk sign, knuckles touching,
> elbows lifted off the desk, so the bump sits dead centre between the two
> microphones.

Expected output (`revised_prompt.edit.txt`, 2,914 B) — the official PE run:

> The man on the left in a dark sweater and wristwatch in `<Picture 1>` is
> `<Subject 1>`. The woman on the right in a beige sweater and hoop earrings in
> `<Picture 1>` is `<Subject 2>`. The pair of black microphones on stands in
> front of the hosts in `<Picture 1>` is `<Subject 3>`. The wooden desk sign
> with raised lettering reading 'MONEY TALKS' in `<Picture 1>` is `<Subject 4>`.
> The black mug on the left with text reading 'INVEST', 'IN', and 'YOURSELF' on
> three lines in `<Picture 1>` is `<Subject 5>`. The black mug on the right with
> text reading 'PLAN', 'BUILD', and 'WEALTH' on three lines in `<Picture 1>` is
> `<Subject 6>`. The upper-right overlay text reading 'EP. 47' in `<Picture 1>`
> is `<Subject 7>`. The framed wall poster on the right reading 'COMPOUND',
> 'CONSISTENCY', and 'CONFIDENCE' on three lines in `<Picture 1>` is
> `<Subject 8>`. The small framed shelf sign in the center background reading
> 'DISCIPLINE', '>', and 'MOTIVATION' on three lines in `<Picture 1>` is
> `<Subject 9>`. The podcast studio environment, including the wooden desk,
> shelves, lamps, books, and decor in `<Picture 1>`, is `<Subject 10>`. The
> target image shows `<Subject 1>` without the over-ear headphones from
> `<Picture 1>` and with fuller hair, as `<Subject 1>` and `<Subject 2>` lean
> inward to fist-bump directly above `<Subject 4>` between `<Subject 3>`.
> `<Subject 1>` is partially preserved; his identity, facial features,
> expression, dark sweater, and wristwatch remain, while his headphones are
> removed, his crown hair is filled in, and his posture changes from resting
> his forearms on the desk to leaning toward `<Subject 2>` with lifted elbows
> for a central fist-bump. `<Subject 2>` is partially preserved; her identity,
> facial features, expression, hair, earrings, and beige sweater remain, while
> her posture changes from crossed arms on the desk to leaning toward
> `<Subject 1>` with lifted elbows for a central fist-bump. `<Subject 3>` is
> fully preserved. `<Subject 4>` is fully preserved. `<Subject 5>` is fully
> preserved. `<Subject 6>` is fully preserved. `<Subject 7>` is fully preserved.
> `<Subject 8>` is fully preserved. `<Subject 9>` is fully preserved.
> `<Subject 10>` is fully preserved. On `<Subject 1>`, the headband indentation
> across the top of his head is gone, replaced by natural, voluminous wavy
> brown curls that match the texture and warm highlights of his existing hair,
> leaving his ears and temples completely uncovered where the earcups and cords
> used to be. Both `<Subject 1>` and `<Subject 2>` angle their torsos slightly
> inward toward the center of the frame. Their elbows are raised clear off the
> desk surface, bringing their inner forearms up into the space behind
> `<Subject 3>` and directly above `<Subject 4>`. Their hands are clenched into
> fists with knuckles touching dead center between the two microphones of
> `<Subject 3>`, casting soft contact shadows onto each other's knuckles and
> subtle warm light reflections from the studio lamps across their sleeves and
> raised arms.

Upstream ran this at **4096×4096 (1:1)**.

| | |
|---|---|
| **Must change** | `<Subject N>` definitions first, one per source element, each naming its `<Picture N>`; then a sentence starting `The target image`; then **one independent relationship clause per Subject** (`<Subject N> is fully preserved.` / `is partially preserved; …`); then Part 3 describing only the changed/new content; single paragraph; no headings; literal source text in **single quotes** ('MONEY TALKS', 'EP. 47') |
| **Style** | which elements were promoted to Subjects, the wording of the relationship clauses, how much new detail Part 3 invents |
| **Assert** | H9 (three-part shape, one clause per Subject, "The target image" present), H2 (`target_aspect_ratio` accepted), H8 (bare `<Subject N>`, no backticks), H13 |

**Note the quoting asymmetry** — this is worth a dedicated test:

- the *source* literal text is transcribed in **single quotes** ('MONEY TALKS');
- the *user's* literal text is reproduced in the double quotes the user typed
  ("MONEY TALKS" in the instruction above);
- the output JSON escapes internal double quotes as `\"`.

The node must not "normalise" the user's quotes into the model's single-quote
convention — H7 applies to the user's own literals.

---

## C. Official `_skill_t2i.txt` worked examples (guide modes)

*Official — `skills/h3-image-prompt-writing/references/t2i-en.txt`.*

### C1 — physical contact

Query:

> A woman potter shaping a bowl on a wheel in a sunlit workshop, seen from her
> left side. Emphasize her hands working the wet clay. No text.

Expected prompt (the official block):

> A naturalistic photograph of a woman potter seated at a pottery wheel in a
> sunlit workshop, viewed from her left side at approximately hand level. A
> medium composition includes her profile, shoulders, forearms, hands, and the
> complete bowl on the wheel, with enough of the wheel and stool visible to make
> her working posture clear. She leans slightly forward with relaxed shoulders
> and looks down at the clay. Capture one precise moment as her left hand
> supports the bowl's outer wall while the fingertips of her right hand shape
> the inner surface near the rim. Her hands oppose each other gently across the
> same section of clay; the bowl stays centered on the wheel head. Wet slip
> coats her fingertips and gathers in fine concentric ridges on the rounded
> bowl. The rim is smooth, open, and clearly separated from her fingers. Soft
> sunlight enters from a side window, catching the moisture on the clay and
> casting gentle shadows beneath her wrists. A worn wooden workbench and quiet
> shelves recede out of focus, keeping attention on the hands and their contact
> with the bowl. Warm earth tones, believable skin texture, and subtle
> workshop wear give the scene a tactile, unposed quality. No lettering or
> additional people appear.

Note "**No lettering or additional people appear.**" — this is the guide's form
of the "say when there is no text" rule, applied to a t2i prompt.

### C2 — a finished data dashboard (contains a quoted literal)

Query:

> Design a 16:9 dashboard titled "Quarterly sales". Products A, B, and C sold
> 12, 18, and 9 units respectively. Include a bar chart, a compact data table,
> the total, and the best-performing product. Keep it clean and do not invent
> additional sales data.

Expected prompt (official block, abridged only in the trailing prose):

> A clean 16:9 sales dashboard on a warm white background uses a restrained
> charcoal-and-teal palette, crisp sans-serif typography, and a consistent
> alignment grid. The title "Quarterly sales" sits at the upper left. Two
> compact summary cards occupy the top row beneath it: "Total units" with the
> value "39", and "Best-performing product" with "B — 18 units". These values
> summarize the supplied three-product dataset.
>
> The lower left contains a large vertical bar chart. Its horizontal axis is
> labeled "Product", with categories "A", "B", and "C" in that order; its
> vertical axis is labeled "Units" and has evenly spaced ticks at 0, 5, 10, 15,
> and 20. All bars share the same zero baseline and equal widths. Their heights
> correspond exactly to 12, 18, and 9, and each value is printed clearly above
> its bar. Product B is teal; A and C use the same muted gray. Thin horizontal
> gridlines make the scale readable without competing with the data.
>
> The lower right contains a compact table with the column headings "Product"
> and "Units", followed by the rows "A | 12", "B | 18", and "C | 9". Its B row
> carries a subtle teal accent matching the highlighted bar. Balanced spacing
> separates the summary cards, chart, and table. There are no invented dates,
> forecasts, percentage changes, or additional data series.

| | |
|---|---|
| **Must change** | every quoted literal survives verbatim — "Quarterly sales", "Total units", "39", "Best-performing product", "B — 18 units", "Product", "Units", "A", "B", "C", "A | 12", "B | 18", "C | 9"; the computed total 39 is derived, not invented; no extra data series |
| **Style** | palette wording, layout order, axis phrasing |
| **Assert** | **H7 (hard)** — each quoted literal present as an exact substring; H6 — no CJK in the prose |

C2 is the test that catches the most common failure: a rewritten or paraphrased
literal. It should be the first regression test in the suite.

---

## D. Official `_skill_edit.txt` worked examples (guide modes)

*Official — `skills/h3-image-prompt-writing/references/edit-en.txt`.*

### D1 — local colour edit

Reference (user side): `@Picture 1` shows a red cup on the left and a lemon on
the right, on a tabletop.

Query:

> In @Picture 1, change only the red cup to blue. Leave everything else
> unchanged.

Expected prompt (official block):

> The red cup on the left in `<Picture 1>` is `<Subject 1>`. The lemon on the
> right in `<Picture 1>` is `<Subject 2>`. The target image shows `<Subject 1>`
> with a blue finish. `<Subject 1>` is partially preserved; its shape, material,
> position, and scale remain while its surface color changes. `<Subject 2>` is
> fully preserved. The blue finish follows the existing surface shading and
> highlights of `<Subject 1>`, with the rest of the source image unchanged.

### D2 — a new poster with an exact property transfer

Reference (user side): `@Picture 1` shows a front-facing beige coat against a
plain backdrop; `@Picture 2` shows a coloured glazed vase.

Query:

> Create a portrait fashion poster using the coat in @Picture 1. Match its color
> exactly to the vase's body color in @Picture 2, keep its cut and fabric, and
> write a short English headline. No brand names or product claims.

Expected prompt (official block):

> The front-facing beige coat in `<Picture 1>` is `<Subject 1>`. The body color
> of the vase in `<Picture 2>` is `<Subject 2>`. The target image is a portrait
> fashion poster featuring `<Subject 1>` beneath a short headline.
> `<Subject 1>` is partially preserved; its cut, fabric texture, folds, and
> front-facing presentation remain while its color, scale, and placement change.
> `<Subject 2>` supplies the exact body color transferred to `<Subject 1>`. The
> transferred color follows the existing garment shading without importing the
> vase's glaze or shape. `<Subject 1>` is centered in the lower two-thirds of
> the poster, with its full hem and sleeves visible. A new warm off-white
> background extends evenly to the edges, leaving generous space around the
> garment. The headline 'A quieter kind of bold.' appears above it in large
> charcoal serif lettering, arranged on two centered lines: 'A quieter kind'
> followed by 'of bold.' The composition contains no other lettering, brand
> names, or decorative props.

| | |
|---|---|
| **Must change** | `<Picture N>` on the model side even though the query used `@Picture N>`; one relationship clause per Subject; a **transfer** clause for `<Subject 2>` ("supplies the exact body color transferred to …"); the authored headline 'A quieter kind of bold.' present verbatim, including the period |
| **Style** | which elements get Subjects; how much of the new background is described |
| **Assert** | H8 (no `@Picture` in the output), H9, H7 for the authored headline |

D2 is also the case where the model must **author** requested copy. The
upstream PE edit prompt is explicit that a missing wording is not a reason to
leave the content unspecified: *"Complete every textual deliverable the user
asks you to write… Requested wording is the deliverable itself, not forbidden
restatement of the user's instruction."* The node must not strip a requested
headline out of the prompt.

---

## E. Community failure evidence — the cases the node must not create

*Not an upstream artifact. Recorded so the regression suite has teeth.*

| Failure | Source | What happened | What the node must prevent |
|---|---|---|---|
| **Garbage text** | bilibili `BV1kqWk6nEwh` (张大叔) | The prompt asked for "书页内文宋体一段约100公字的公司简介" — a *description* of text, not the text. The output was meaningless pseudo-characters. The same author traced it to the PE step: the expansion was truncated or the small model looped, so the prompt that reached the sampler was already broken. Rewriting the request with the actual 100 characters made the image reproduce them. | H7 (never paraphrase a literal), H5/H1 (reject a broken expansion), and the "say when there is no text" preference. |
| **Truncated expansion** | same | The PE's output budget plus forced long thinking produced a cut-off prompt; raising the length again broke it further. | `max_tokens` default 16384, the budget assistant's auto-doubling, and the empty-reply error naming the cause. |
| **Reasoning instead of a caption** | upstream `expand_prompt.py` | The official script refuses a reply with empty `reasoning_content` and tells you to start the server with `--reasoning-parser qwen3`. | `allow_reasoning_fallback=False` + the explicit error message. |
| **PE drags the image style** | bilibili `BV1GEWy6GEoH` (Smthem) | On small VRAM the PE falls back to CPU (10 minutes) and the expansion pulled the image style with it. | The `_guide` modes exist for exactly this case — offer them. |
| **Aspect-ratio key mismatch** | upstream artifacts | The edit PE writes `target_aspect_ratio`; the only published system prompt says `aspect_ratio`. | H2 (accept both). |
| **Reference ratio guessed, not computed** | upstream PE edit text | The prompt insists the caller supplies each reference's exact reduced integer ratio and that the model must not estimate it from the preview. | H12 (node computes it). |

---

## F. Local reproduction (this project's own run, 2026-11-16)

*Not upstream. Included because it is the cross-check that makes the official
rules believable.*

18/18 official showcase reproductions succeeded locally on ComfyUI 0.45.0 with
`minimax_h3_image_pruned_int8_convrot`, 16 MP, seed 42. Three findings line up
with the official rules and with the community evidence above:

1. **Text that is spelled out in the prompt comes out right** — 11 Chinese
   component labels, the Chinese main title, the "入选/未入选" column headers.
2. **Text the model has to invent comes out as garbage** — explosion-view part
   names, academic axis labels.
3. **Content that has to be reasoned out comes out wrong** — a staff-selection
   puzzle whose official answer was 丙戊己 came back as 甲乙戊.

That is the same rule the upstream PE task prompt encodes as *"Preserve exact
requested text, named entities, counts, formulas, data values…"* and the same
rule the community states as *"它只会照着写，不会自己想"* — the node's job is to
make sure every literal the user meant to appear actually reaches the sampler
in full.

**Node-level consequence:** the `t2i`/`edit` modes cannot fix a draft that
never contained the text. The node should not silently invent it. If the user
asked for on-image copy without giving it, the right behaviour is to leave the
request for the user to supply — or, in guide mode, to follow the official
*"Quote every word that appears. When the user wants you to write the copy,
write the actual words."* rule and author it, clearly marked as authored.
