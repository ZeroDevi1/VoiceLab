import os
import importlib.util
import sys
import unittest
from pathlib import Path


class TestGptSovitsBootstrap(unittest.TestCase):
    def _load_module(self):
        tools_dir = (
            Path(__file__).resolve().parents[1] / "workflows" / "gpt_sovits" / "tools"
        )
        old = list(sys.path)
        try:
            if str(tools_dir) not in sys.path:
                sys.path.insert(0, str(tools_dir))
            module_path = tools_dir / "voicelab_bootstrap.py"
            spec = importlib.util.spec_from_file_location(
                "voicelab_bootstrap_gpt_test", module_path
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        finally:
            sys.path[:] = old

    def test_default_pretrained_s2d_matches_s2g_name(self) -> None:
        gpt_bootstrap = self._load_module()
        s2g = gpt_bootstrap.default_pretrained_s2g("v1")
        s2d = gpt_bootstrap.default_pretrained_s2d("v1")
        self.assertEqual(s2d.name, s2g.name.replace("s2G", "s2D", 1))

    def test_shared_assets_root_env_override(self) -> None:
        old_env = os.environ.get("GPT_SOVITS_ASSETS_DIR")
        try:
            os.environ["GPT_SOVITS_ASSETS_DIR"] = "D:/tmp/gpt-assets"
            gpt_bootstrap = self._load_module()
            self.assertEqual(
                gpt_bootstrap.shared_assets_root(),
                Path("D:/tmp/gpt-assets").resolve(),
            )
        finally:
            if old_env is None:
                os.environ.pop("GPT_SOVITS_ASSETS_DIR", None)
            else:
                os.environ["GPT_SOVITS_ASSETS_DIR"] = old_env


if __name__ == "__main__":
    unittest.main()
