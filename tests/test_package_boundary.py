"""Shared packages must never import their ComfyUI consumers."""
import ast
import importlib
from pathlib import Path
import unittest

PACKAGE = "mie_prompt"
ROOT = Path(__file__).resolve().parents[1] / PACKAGE
HOSTS = {"_mienodes_internal", "folder_paths", "comfy", "server", "nodes", "core"}

class PackageBoundaryTests(unittest.TestCase):
    def test_no_host_or_reverse_imports(self):
        forbidden = HOSTS | ({"mie_craft"} if PACKAGE == "mie_prompt" else set())
        violations = []
        for source in ROOT.rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                for name in names:
                    if name.split(".")[0] in forbidden:
                        violations.append(f"{source.name}:{node.lineno}: {name}")
        self.assertEqual([], violations, "consumer imports leak into shared package")

    def test_all_modules_import_without_host(self):
        for source in ROOT.rglob("*.py"):
            parts = source.relative_to(ROOT.parent).with_suffix("").parts
            module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            importlib.import_module(module)

if __name__ == "__main__":
    unittest.main()
