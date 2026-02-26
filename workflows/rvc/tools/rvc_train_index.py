from __future__ import annotations

import argparse
import os
from pathlib import Path

from rvc_init_runtime import init_runtime


def _iter_feature_files(feature_dir: Path) -> list[Path]:
    return sorted([p for p in feature_dir.glob("*.npy") if p.is_file()])


def _load_npy(path: Path):
    import numpy as np

    return np.load(str(path))


def _sample_training_frames(
    files: list[Path],
    *,
    max_train_frames: int,
    seed: int,
):
    """
    Sample up to `max_train_frames` frames across all npy files, roughly proportional
    to each file's frame count. This avoids concatenating all features into RAM.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    counts: list[int] = []
    for p in files:
        arr = _load_npy(p)
        counts.append(int(arr.shape[0]))

    total = int(sum(counts))
    if total <= 0:
        raise SystemExit("No feature frames found (empty 3_feature*.npy files).")

    train_frames = min(int(max_train_frames), total)

    # Allocate proportional sample sizes, then adjust to exact sum.
    raw = [train_frames * c / total for c in counts]
    ks = [int(x) for x in raw]
    # Fix rounding drift.
    drift = train_frames - sum(ks)
    if drift != 0:
        # Distribute remaining frames to the largest fractional parts.
        frac = sorted(
            [(i, raw[i] - ks[i]) for i in range(len(ks))],
            key=lambda t: t[1],
            reverse=(drift > 0),
        )
        for i, _ in frac[: abs(drift)]:
            ks[i] += 1 if drift > 0 else -1
    assert sum(ks) == train_frames

    samples: list["np.ndarray"] = []
    for p, k in zip(files, ks):
        if k <= 0:
            continue
        arr = _load_npy(p)
        n = int(arr.shape[0])
        if n <= k:
            samples.append(arr)
            continue
        idx = rng.choice(n, size=k, replace=False)
        samples.append(arr[idx])

    return np.concatenate(samples, axis=0), total, train_frames


def _add_all_frames(index, files: list[Path], *, batch_size: int) -> int:
    import numpy as np

    added = 0
    for p in files:
        arr = _load_npy(p)
        if arr.size == 0:
            continue
        # Ensure float32 for faiss.
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        for i in range(0, arr.shape[0], batch_size):
            chunk = arr[i : i + batch_size]
            index.add(chunk)
            added += int(chunk.shape[0])
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description="Train RVC faiss index (streaming, low RAM).")
    ap.add_argument("--exp-name", default="xingtong_v2_48k_f0")
    ap.add_argument("--version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--max-train-frames", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nprobe", type=int, default=1)
    ap.add_argument("--batch-size-add", type=int, default=8192)
    ap.add_argument("--force", action="store_true", help="Overwrite existing index outputs.")
    args = ap.parse_args()

    rt = init_runtime(force=False, assets_src=None, download_missing=False, hf_base="https://hf-mirror.com")
    exp = rt / "logs" / args.exp_name
    feature_dir = exp / ("3_feature256" if args.version == "v1" else "3_feature768")
    if not feature_dir.exists():
        raise SystemExit(f"Feature dir not found: {feature_dir}\nRun: uv run python tools/rvc_train.py")

    files = _iter_feature_files(feature_dir)
    if not files:
        raise SystemExit(f"No .npy feature files found under: {feature_dir}")

    try:
        import faiss  # type: ignore
        import numpy as np
    except Exception as e:
        raise SystemExit(
            f"Missing dependencies for index training: {e}\n"
            "Run: cd workflows/rvc && uv sync"
        )

    dim = 256 if args.version == "v1" else 768
    train_x, total_frames, train_frames = _sample_training_frames(
        files, max_train_frames=int(args.max_train_frames), seed=int(args.seed)
    )
    if train_x.dtype != np.float32:
        train_x = train_x.astype(np.float32)

    n_ivf = min(int(16 * (train_frames**0.5)), train_frames // 39)
    if n_ivf <= 0:
        n_ivf = 1

    out_dir = rt / "indices"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_added = out_dir / f"added_IVF{n_ivf}_Flat_nprobe_{int(args.nprobe)}_{args.exp_name}_{args.version}.index"
    out_short = out_dir / f"{args.exp_name}.index"

    if out_added.exists() and not args.force:
        print(f"[rvc] index exists (skip, use --force to overwrite): {out_added}")
    else:
        if args.force:
            out_added.unlink(missing_ok=True)

        index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = int(args.nprobe)

        print(f"[rvc] train frames: {train_frames} (sampled from total {total_frames})")
        print(f"[rvc] training IVF{n_ivf},Flat (dim={dim}, nprobe={index_ivf.nprobe}) ...")
        index.train(train_x)

        print("[rvc] adding all frames (streaming) ...")
        added = _add_all_frames(index, files, batch_size=int(args.batch_size_add))
        print(f"[rvc] added frames: {added}")

        faiss.write_index(index, str(out_added))
        print(f"[rvc] wrote index: {out_added}")

    # Create/refresh short name symlink for convenience.
    if out_short.exists() or out_short.is_symlink():
        out_short.unlink(missing_ok=True)
    os.symlink(str(out_added), str(out_short))
    print(f"[rvc] short index: {out_short} -> {out_added.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
