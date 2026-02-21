import argparse
from pathlib import Path


def _torch_load(path: Path):
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create CosyVoice spk2info.pt from spk2embedding.pt for SFT inference.")
    parser.add_argument("--spk2embedding_pt", type=str, default="data/xuan_sft/train/spk2embedding.pt")
    parser.add_argument("--spk_id", type=str, default="xuan")
    parser.add_argument("--out_spk2info_pt", type=str, default="pretrained_models/Fun-CosyVoice3-0.5B-xuan-sft/spk2info.pt")
    args = parser.parse_args()

    spk2embedding_path = Path(args.spk2embedding_pt)
    if not spk2embedding_path.exists():
        print(f"[spk2info] spk2embedding.pt not found: {spk2embedding_path}")
        return 2

    spk2embedding = _torch_load(spk2embedding_path)
    if not isinstance(spk2embedding, dict):
        print("[spk2info] spk2embedding is not a dict")
        return 3

    if args.spk_id not in spk2embedding:
        print(f"[spk2info] spk_id not found in spk2embedding: {args.spk_id}")
        print(f"[spk2info] available_spk_ids={list(spk2embedding.keys())}")
        return 4

    import torch

    vec = spk2embedding[args.spk_id]
    emb = torch.tensor(vec, dtype=torch.float32).reshape(1, -1)
    spk2info = {args.spk_id: {"embedding": emb}}

    out_path = Path(args.out_spk2info_pt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(spk2info, str(out_path))
    print(f"[spk2info] Wrote {out_path} (dim={emb.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

