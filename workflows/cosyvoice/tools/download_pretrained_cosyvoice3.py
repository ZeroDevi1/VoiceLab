import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Fun-CosyVoice3 pretrained model to pretrained_models/.")
    parser.add_argument(
        "--source",
        choices=["modelscope", "hf"],
        default="modelscope",
        help="Download source: modelscope (recommended in CN) or hf (HuggingFace).",
    )
    parser.add_argument("--repo_id", type=str, default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--local_dir", type=str, default="pretrained_models/Fun-CosyVoice3-0.5B")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "modelscope":
        from modelscope import snapshot_download  # type: ignore

        snapshot_download(args.repo_id, local_dir=str(local_dir))
    else:
        from huggingface_hub import snapshot_download  # type: ignore

        snapshot_download(args.repo_id, local_dir=str(local_dir))

    print(f"[download] done: {local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

