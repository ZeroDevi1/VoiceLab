import argparse
import shutil
from pathlib import Path


def _copytree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"out_model_dir already exists: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble SFT model dir by copying base model and overriding llm/flow weights.")
    parser.add_argument("--base_model_dir", type=str, default="pretrained_models/Fun-CosyVoice3-0.5B")
    parser.add_argument("--out_model_dir", type=str, default="pretrained_models/Fun-CosyVoice3-0.5B-xuan-sft")
    parser.add_argument("--llm_pt", type=str, default="exp/xuan_sft/llm/torch_ddp/llm_avg.pt")
    parser.add_argument("--flow_pt", type=str, default="exp/xuan_sft/flow/torch_ddp/flow_avg.pt")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_model_dir)
    out_dir = Path(args.out_model_dir)
    llm_pt = Path(args.llm_pt)
    flow_pt = Path(args.flow_pt)

    if not base_dir.exists():
        print(f"[assemble] base_model_dir not found: {base_dir}")
        return 2

    try:
        _copytree(base_dir, out_dir, overwrite=bool(args.overwrite))
    except Exception as exc:
        print(f"[assemble] failed to copy base model dir: {exc}")
        return 3

    if llm_pt.exists():
        shutil.copy2(llm_pt, out_dir / "llm.pt")
        print(f"[assemble] llm.pt <- {llm_pt}")
    else:
        print(f"[assemble] WARN: llm_pt not found, keep base llm.pt: {llm_pt}")

    if flow_pt.exists():
        shutil.copy2(flow_pt, out_dir / "flow.pt")
        print(f"[assemble] flow.pt <- {flow_pt}")
    else:
        print(f"[assemble] WARN: flow_pt not found, keep base flow.pt: {flow_pt}")

    print(f"[assemble] done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

