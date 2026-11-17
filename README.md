# ComfyUI-Mie-Prompt

**Mie's prompt layer.** The prompt templates and the loader that reads the
prompt families.

```bash
pip install -e <this directory>
```

It sits on top of `ComfyUI-Mie-LLM`, which it uses for the plugin config
and the token resolution.

---

## What is in it

| Module | |
|---|---|
| `prompts/loader.py` | resolves a prompt by logical name, verbatim, never applying `.format()` |
| `prompts/h3/`, `prompts/h3_loop/`, `prompts/h3_storyboard/` | the prompt families, as plain text |
| `prompts/h3_image/` | the MiniMax-H3-Image family: the official PE prompts and the prompt-writing skill, byte for byte |
| `h3_prompts.py` | the H3 image and video prompt templates |
| `h3_image_prompts.py` | the H3-Image modes, the official-text loaders and the reply normalizer |
| `storyboard_prompts.py` | the storyboard prompt templates and their parsing |

## One thing worth knowing

The prompt families and the code that reads them belong together. Moving one
without the other breaks the loader, and the "two copies of the system
instruction must stay in step" discipline becomes a cross-repository check.

The H3-Image family adds a second rule: its seven prompt files are official
MiniMax artifacts and are never edited. `tests/test_h3_image_prompts.py`
asserts their sha256 digests, so an accidental edit fails the build instead
of silently shipping a modified prompt.

## Third-party files

`prompts/h3_image/` ships seven official MiniMax artifacts (the H3-Image PE
prompts and the prompt-writing skill), byte for byte, together with a copy of
the MiniMax-H3-Image Community License Agreement and a `NOTICE` file. They are
covered by that agreement, not by this package's GPL-3.0 licence --
`prompts/h3_image/NOTICE` carries the digests, the upstream links, the
attribution and the territorial terms.

## Licence

GPL-3.0. See `LICENSE`.
