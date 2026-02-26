import tempfile
import unittest
from pathlib import Path


class TestListAnnotations(unittest.TestCase):
    def test_parse_list_tolerant(self) -> None:
        from voicelab.list_annotations import parse_list

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.list"
            p.write_text(
                "\n".join(
                    [
                        "",
                        "# comment",
                        " /abs/x.wav|Spk|ZH|hello world ",
                        "only_audio.wav",
                        "a.wav|Spk|ZH|text|with|pipes",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = parse_list(p)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0].audio, "/abs/x.wav")
            self.assertEqual(rows[0].speaker, "Spk")
            self.assertEqual(rows[0].lang, "ZH")
            self.assertEqual(rows[0].text, "hello world")
            self.assertEqual(rows[1].audio, "only_audio.wav")
            self.assertIsNone(rows[1].speaker)
            self.assertIsNone(rows[1].lang)
            self.assertIsNone(rows[1].text)
            self.assertEqual(rows[2].text, "text|with|pipes")

    def test_find_same_name_list(self) -> None:
        from voicelab.list_annotations import find_same_name_list

        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "Xuan"
            d.mkdir()
            self.assertIsNone(find_same_name_list(d))

            (d / "xuan.list").write_text("x.wav|x|ZH|t\n", encoding="utf-8")
            # same-name lower() should match.
            self.assertEqual(find_same_name_list(d), d / "xuan.list")

    def test_resolve_audio_for_dataset_prefers_local_basename(self) -> None:
        from voicelab.list_annotations import resolve_audio_for_dataset

        with tempfile.TemporaryDirectory() as td:
            dataset = Path(td) / "dataset"
            dataset.mkdir()
            (dataset / "a.wav").write_bytes(b"x")

            # Even if list has absolute path, prefer dataset_dir basename.
            resolved = resolve_audio_for_dataset("/mnt/c/AIGC/whatever/a.wav", dataset)
            self.assertEqual(resolved, dataset / "a.wav")

    def test_resolve_audio_for_dataset_by_stem(self) -> None:
        from voicelab.list_annotations import resolve_audio_for_dataset

        with tempfile.TemporaryDirectory() as td:
            dataset = Path(td) / "dataset"
            dataset.mkdir()
            (dataset / "abc.wav").write_bytes(b"x")
            # list says .mp3 but local has .wav
            resolved = resolve_audio_for_dataset("abc.mp3", dataset)
            self.assertEqual(resolved, dataset / "abc.wav")


if __name__ == "__main__":
    unittest.main()

