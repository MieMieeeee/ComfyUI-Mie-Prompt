"""A self-test that does not need ComfyUI.

The behavioural tests live in the MieNodes pack and in the H3 engine, which
both consume these packages through their shims. This is the floor: the
package imports, its public surface is there, and the prompt families it
ships are readable.
"""

PACKAGE = "mie_prompt"


def main() -> int:
    module = __import__(PACKAGE)
    assert hasattr(module, "__version__"), "no version"
    print(PACKAGE, module.__version__)

    from mie_prompt.prompts import prompt_path, prompt_root
    from mie_prompt.h3_prompts import system_t2v_prompt
    from mie_prompt.storyboard_prompts import STYLE_ADVICE
    assert prompt_root().is_dir(), prompt_root()
    assert prompt_path("h3/system_t2v.txt").is_file()
    assert prompt_path("h3_loop/shot_system_addendum.txt").is_file()
    assert prompt_path("h3_storyboard/system_asset.txt").is_file()

    print(PACKAGE, "self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
