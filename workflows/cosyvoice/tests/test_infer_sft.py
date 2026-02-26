import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import infer_sft  # noqa: E402


class TestInferSftCli(unittest.TestCase):
    def test_cli_requires_text_or_text_file(self) -> None:
        parser = infer_sft.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--model_dir", "m", "--spk_id", "s"])

    def test_cli_rejects_both_text_and_text_file(self) -> None:
        parser = infer_sft.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--model_dir", "m", "--spk_id", "s", "--text", "hi", "--text_file", "input.txt"]
            )

    def test_cli_removed_split_related_flags_are_rejected(self) -> None:
        parser = infer_sft.build_parser()
        for flag in ["--no_split", "--concat", "--prompt_scope", "--prompt_preset"]:
            with self.assertRaises(SystemExit, msg=f"expected {flag} to be rejected"):
                parser.parse_args(["--model_dir", "m", "--spk_id", "s", "--text", "hi", flag])

    def test_cli_accepts_inline_text(self) -> None:
        parser = infer_sft.build_parser()
        args = parser.parse_args(["--model_dir", "m", "--spk_id", "s", "--text", "  hi  "])
        resolved = infer_sft.resolve_text(args)
        self.assertEqual(resolved.text, "hi")
        self.assertEqual(resolved.text_input, "inline")
        self.assertEqual(resolved.text_file, "")

    def test_cli_accepts_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "input.txt"
            p.write_text("  hi  \n", encoding="utf-8")

            parser = infer_sft.build_parser()
            args = parser.parse_args(["--model_dir", "m", "--spk_id", "s", "--text_file", str(p)])
            resolved = infer_sft.resolve_text(args)
            self.assertEqual(resolved.text, "hi")
            self.assertEqual(resolved.text_input, "file")
            self.assertEqual(resolved.text_file, str(p))

    def test_out_wav_resolves_out_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_wav = Path(td) / "nested" / "full.wav"
            parser = infer_sft.build_parser()
            args = parser.parse_args(["--model_dir", "m", "--spk_id", "s", "--text", "hi", "--out_wav", str(out_wav)])

            out_dir, out_wav_path = infer_sft.resolve_output(args)
            self.assertEqual(out_dir, out_wav.parent)
            self.assertEqual(out_wav_path, out_wav)

    def test_strip_endofprompt(self) -> None:
        self.assertEqual(infer_sft._strip_endofprompt("abc<|endofprompt|>def"), "abcdef")
        self.assertEqual(infer_sft._strip_endofprompt("<|endofprompt|>"), "")


class TestInferSftResolveText(unittest.TestCase):
    def test_resolve_text_file_not_found(self) -> None:
        args = argparse.Namespace(text_file="/no/such/file.txt", text=None)
        with self.assertRaises(infer_sft._TextResolutionError) as ctx:
            infer_sft.resolve_text(args)
        self.assertEqual(ctx.exception.code, 2)

    def test_resolve_inline_empty_is_error(self) -> None:
        args = argparse.Namespace(text_file=None, text="   ")
        with self.assertRaises(infer_sft._TextResolutionError) as ctx:
            infer_sft.resolve_text(args)
        self.assertEqual(ctx.exception.code, 3)

    def test_resolve_out_wav_rejects_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(out_wav=td, out_dir="out_wav")
            with self.assertRaises(infer_sft._TextResolutionError) as ctx:
                infer_sft.resolve_output(args)
            self.assertEqual(ctx.exception.code, 17)


class TestInferSftRunJson(unittest.TestCase):
    def test_run_json_does_not_store_raw_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            args = argparse.Namespace(
                model_dir="model",
                spk_id="dream",
                llm_ckpt="",
                flow_ckpt="",
                speed=1.0,
                stream=False,
                text_frontend=True,
                seed=1986,
                temperature=1.0,
                top_p=0.6,
                top_k=10,
                win_size=10,
                tau_r=1.0,
            )
            infer_sft._write_run_metadata(
                out_dir,
                args=args,
                text="SECRET RAW TEXT",
                prompt_text="PROMPT",
                text_input="inline",
                text_file="",
                prompt_strategy="inject",
                prompt_inject_text="PROMPT",
                guide_text="",
            )

            run = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
            self.assertNotIn("SECRET RAW TEXT", json.dumps(run, ensure_ascii=False))
            self.assertEqual(run["text_input"], "inline")
            self.assertEqual(run["text_file"], "")
            self.assertEqual(run["text_chars"], len("SECRET RAW TEXT"))
            self.assertTrue(run["text_sha1"])
            self.assertEqual(run["split_mode"], "none")


if __name__ == "__main__":
    unittest.main()
