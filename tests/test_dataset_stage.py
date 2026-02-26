import tempfile
import unittest
from pathlib import Path


class TestDatasetStage(unittest.TestCase):
    def test_ensure_list_present_from_annotation_dir(self) -> None:
        from voicelab.dataset_stage import ensure_list_present

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "Xuan"
            dst = td / "dst" / "Xuan"
            ann = td / "标注文件"
            src.mkdir(parents=True)
            dst.mkdir(parents=True)
            ann.mkdir(parents=True)

            # annotation has lower-case list
            (ann / "xuan.list").write_text("x.wav|x|ZH|t\n", encoding="utf-8")
            copied = ensure_list_present(src, dst, annotation_dir=ann)
            self.assertEqual(copied, dst / "xuan.list")
            self.assertTrue((dst / "xuan.list").exists())

    def test_ensure_list_present_keeps_existing(self) -> None:
        from voicelab.dataset_stage import ensure_list_present

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "XingTong"
            dst = td / "dst" / "XingTong"
            ann = td / "标注文件"
            src.mkdir(parents=True)
            dst.mkdir(parents=True)
            ann.mkdir(parents=True)

            (dst / "XingTong.list").write_text("a\n", encoding="utf-8")
            (ann / "XingTong.list").write_text("b\n", encoding="utf-8")
            kept = ensure_list_present(src, dst, annotation_dir=ann)
            self.assertEqual(kept, dst / "XingTong.list")
            self.assertEqual((dst / "XingTong.list").read_text(encoding="utf-8").strip(), "a")


if __name__ == "__main__":
    unittest.main()

