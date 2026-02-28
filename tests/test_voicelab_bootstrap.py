import os
import unittest


class TestGitMirrorPrefix(unittest.TestCase):
    def test_no_prefix_returns_original(self) -> None:
        from voicelab.bootstrap import apply_git_mirror_prefix

        self.assertEqual(
            apply_git_mirror_prefix("https://github.com/org/repo", None),
            "https://github.com/org/repo",
        )

    def test_only_applies_to_github_https(self) -> None:
        from voicelab.bootstrap import apply_git_mirror_prefix

        self.assertEqual(
            apply_git_mirror_prefix("https://example.com/org/repo", ""),
            "https://example.com/org/repo",
        )

    def test_trailing_slash_handling(self) -> None:
        from voicelab.bootstrap import apply_git_mirror_prefix

        expected = "https://ghproxy.com/https://github.com/org/repo"
        self.assertEqual(
            apply_git_mirror_prefix(
                "https://github.com/org/repo", "https://ghproxy.com"
            ),
            expected,
        )
        self.assertEqual(
            apply_git_mirror_prefix(
                "https://github.com/org/repo", "https://ghproxy.com/"
            ),
            expected,
        )

    def test_idempotent_if_already_mirrored(self) -> None:
        from voicelab.bootstrap import apply_git_mirror_prefix

        self.assertEqual(
            apply_git_mirror_prefix(
                "https://github.com/org/repo",
                "",
            ),
            "https://github.com/org/repo",
        )


class TestParseWorkflows(unittest.TestCase):
    def test_parse_and_dedupe(self) -> None:
        from voicelab.bootstrap import parse_workflows

        self.assertEqual(parse_workflows("rvc,msst,rvc"), ["rvc", "msst"])

    def test_empty_uses_default(self) -> None:
        from voicelab.bootstrap import parse_workflows

        self.assertEqual(parse_workflows(""), ["cosyvoice", "rvc", "msst"])

    def test_invalid_raises(self) -> None:
        from voicelab.bootstrap import parse_workflows

        with self.assertRaises(ValueError):
            parse_workflows("rvc,unknown")


class TestAssetsDir(unittest.TestCase):
    def test_assets_dir_env_override(self) -> None:
        from pathlib import Path

        from voicelab.bootstrap import resolve_assets_dir

        old = os.environ.get("VOICELAB_ASSETS_DIR")
        try:
            os.environ["VOICELAB_ASSETS_DIR"] = "/tmp/voicelab-assets-test"
            self.assertEqual(
                resolve_assets_dir(None), Path("/tmp/voicelab-assets-test").resolve()
            )
        finally:
            if old is None:
                os.environ.pop("VOICELAB_ASSETS_DIR", None)
            else:
                os.environ["VOICELAB_ASSETS_DIR"] = old


class TestEnsureSymlink(unittest.TestCase):
    def test_directory_link_or_copy_is_usable(self) -> None:
        import tempfile
        from pathlib import Path

        from voicelab.bootstrap import ensure_symlink

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            src = base / "src_dir"
            dst = base / "dst_dir"
            src.mkdir(parents=True, exist_ok=True)
            (src / "a.txt").write_text("x", encoding="utf-8")

            ensure_symlink(src=src, dst=dst, force=True)

            # 无论是符号链接还是 copy fallback，dst 都必须可作为“目录”正常访问。
            self.assertTrue(dst.exists())
            self.assertTrue(dst.is_dir())
            self.assertIn("a.txt", [p.name for p in dst.iterdir()])
