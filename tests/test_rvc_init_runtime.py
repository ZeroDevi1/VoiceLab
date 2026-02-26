import sys
import tempfile
import unittest
from pathlib import Path


def _import_rvc_tools() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tools = repo_root / "workflows" / "rvc" / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))


class TestRvcInitRuntime(unittest.TestCase):
    def test_runtime_has_core_assets(self) -> None:
        _import_rvc_tools()
        from rvc_init_runtime import _runtime_has_core_assets

        with tempfile.TemporaryDirectory() as td:
            rt = Path(td)
            self.assertFalse(_runtime_has_core_assets(rt))

            (rt / "assets" / "hubert").mkdir(parents=True, exist_ok=True)
            (rt / "assets" / "rmvpe").mkdir(parents=True, exist_ok=True)
            (rt / "assets" / "hubert" / "hubert_base.pt").write_bytes(b"x")
            self.assertFalse(_runtime_has_core_assets(rt))

            (rt / "assets" / "rmvpe" / "rmvpe.pt").write_bytes(b"x")
            self.assertTrue(_runtime_has_core_assets(rt))

