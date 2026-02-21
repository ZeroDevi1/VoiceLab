import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Print onnxruntime providers diagnostics.")
    parser.add_argument("--require-cuda", action="store_true", help="Exit non-zero if CUDAExecutionProvider not found.")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except Exception as exc:
        print(f"[diag_ort] Failed to import onnxruntime: {exc}")
        return 2

    version = getattr(ort, "__version__", "unknown")
    try:
        providers = ort.get_available_providers()
    except Exception as exc:
        print(f"[diag_ort] Failed to query providers: {exc}")
        return 3

    print(f"onnxruntime={version}")
    print("providers=" + ",".join(providers))

    if args.require_cuda and "CUDAExecutionProvider" not in providers:
        print("[diag_ort] ERROR: CUDAExecutionProvider not available.")
        return 10
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

