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
| `h3_prompts.py` | the H3 image and video prompt templates |
| `storyboard_prompts.py` | the storyboard prompt templates and their parsing |

## One thing worth knowing

The prompt families and the code that reads them belong together. Moving one
without the other breaks the loader, and the "two copies of the system
instruction must stay in step" discipline becomes a cross-repository check.

## Licence

GPL-3.0.
