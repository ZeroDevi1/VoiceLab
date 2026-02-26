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

        self.assertEqual(
            apply_git_mirror_prefix("https://github.com/org/repo", "https://ghproxy.com"),
            "https://github.com/org/repo",
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
        from voicelab.bootstrap import resolve_assets_dir

        old = os.environ.get("VOICELAB_ASSETS_DIR")
        try:
            os.environ["VOICELAB_ASSETS_DIR"] = "/tmp/voicelab-assets-test"
            self.assertEqual(str(resolve_assets_dir(None)), "/tmp/voicelab-assets-test")
        finally:
            if old is None:
                os.environ.pop("VOICELAB_ASSETS_DIR", None)
            else:
                os.environ["VOICELAB_ASSETS_DIR"] = old

