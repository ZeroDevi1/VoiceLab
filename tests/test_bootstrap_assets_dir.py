import os
import unittest
from pathlib import Path

from voicelab.bootstrap import repo_root, resolve_assets_dir


class TestBootstrapAssetsDir(unittest.TestCase):
    def test_default_assets_dir_is_project_local_cache(self) -> None:
        old = os.environ.pop("VOICELAB_ASSETS_DIR", None)
        try:
            got = resolve_assets_dir(None)
        finally:
            if old is not None:
                os.environ["VOICELAB_ASSETS_DIR"] = old

        self.assertEqual(got, (repo_root() / ".cache" / "voicelab" / "assets").resolve())

    def test_env_assets_dir_overrides_default(self) -> None:
        old = os.environ.get("VOICELAB_ASSETS_DIR")
        os.environ["VOICELAB_ASSETS_DIR"] = "D:/tmp/custom-assets"
        try:
            got = resolve_assets_dir(None)
        finally:
            if old is None:
                os.environ.pop("VOICELAB_ASSETS_DIR", None)
            else:
                os.environ["VOICELAB_ASSETS_DIR"] = old

        self.assertEqual(got, Path("D:/tmp/custom-assets").resolve())
