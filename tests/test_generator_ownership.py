import importlib
import unittest

FAMILIES = ["bernini_prompt_generator","flux3_klein_prompt_generator","h3_image_prompt_generator","h3_prompt_generator","ideogram4_prompt_generator","krea2_prompt_generator","ltx25_prompt_generator","moss_image_prompt_generator","qwen_image_21_prompt_generator","scail2_prompt_generator"]

class GeneratorOwnershipTests(unittest.TestCase):
    def test_each_family_has_a_host_neutral_implementation(self):
        for name in FAMILIES:
            with self.subTest(family=name):
                module = importlib.import_module("mie_prompt." + name)
                self.assertFalse(any(hasattr(value, "INPUT_TYPES") for value in vars(module).values() if isinstance(value, type)))
