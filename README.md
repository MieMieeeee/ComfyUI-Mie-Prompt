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


## 发布 / Releasing

```
python -m build
python -m twine upload dist/*
python tools/notify_mirrors.py --wait 300     # 让国内 PyPI 镜像收录
```

**第三步是发布流程的一部分, 不是可选项。** 国内 PyPI 镜像 (阿里云 / 华为云 / 清华)
对 PyPI 是镜像同步, 新版本通常要等它们自己的同步周期才出现; 期间按
`requirements.txt` 里的版本号去装会拿到 `No matching distribution found`。
这些镜像**有按需回源机制** —— 请求一次该包的 simple index 就能触发它们去 PyPI
拉取, 约 1 分钟内收录 (2026-11-18 实测: 请求后 61 秒华为云就从 0.1.x 变成
0.2.0)。所以发布后跑一次 `tools/notify_mirrors.py`, 把等待从「同步周期」压到
「约一分钟」。

## 依赖声明的规矩

**包只声明依赖, 不指定用哪个镜像源。** `requirements.txt` / `pyproject.toml` 里
只写包名和版本约束, **不写 `-i` / `--index-url` / `extra-index-url`** —— 镜像
是**部署环境**的选择 (启动器那边按用户在「启动与更新页」配的来), 包管不了也不该管。
写出镜像源会导致: 换环境装不上、用户没法用官方源、镜像滞后时把包锁死在旧版本。

同理, 装包的那一侧 (启动器的伴生服务安装链) 负责按配置选镜像, 镜像上没有的
版本由它回退官方 PyPI —— 不需要包自己操心。
