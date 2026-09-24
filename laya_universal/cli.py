"""Command-line interface for laya-universal."""

import argparse
import json
import sys


def _utf8_streams():
    # Windows consoles often default to a legacy code page; answers can contain
    # any language, so make printing them reliable instead of crashing on cp1252.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def main():
    _utf8_streams()
    parser = argparse.ArgumentParser(
        prog="laya-universal",
        description="Laya typed decisions — cross-platform inference (ONNX/MLX)",
    )
    sub = parser.add_subparsers(dest="command")

    # predict command
    p_predict = sub.add_parser("predict", help="Run prediction on a state with questions")
    p_predict.add_argument("--model", type=str, default="convaiinnovations/laya",
                           help="Model ID or local path")
    p_predict.add_argument("--state", type=str, help="State text (inline)")
    p_predict.add_argument("--state-file", type=str, help="Path to JSON state file")
    p_predict.add_argument("--questions", type=str, required=True,
                           help="Path to JSON questions file")
    p_predict.add_argument("--dtype", type=str, default="float16",
                           choices=["float32", "float16"])
    p_predict.add_argument("--backend", type=str, default=None,
                           help="Force backend: onnx, dml, cuda, coreml, cpu, mlx, or a "
                                "registered name. Default: auto (format-aware).")
    p_predict.add_argument("--device", type=str, default=None,
                           help="Device: cpu or gpu (ONNX honors cpu; MLX honors cpu/gpu/metal)")

    # convert command
    p_convert = sub.add_parser("convert", help="Export a PyTorch checkpoint to ONNX")
    p_convert.add_argument("--model", type=str, default="convaiinnovations/laya",
                           help="HuggingFace model ID or local path")
    p_convert.add_argument("--output", type=str, default="./laya-onnx",
                           help="Output directory")
    p_convert.add_argument("--dtype", type=str, default="float32",
                           choices=["float32", "float16"])

    # info command
    sub.add_parser("info", help="Show available backends on this machine")

    args = parser.parse_args()

    if args.command == "info":
        _show_info()
    elif args.command == "convert":
        _run_convert(args)
    elif args.command == "predict":
        _run_predict(args)
    else:
        parser.print_help()


def _show_info():
    from .backends import _BACKENDS, available_backends

    avail = set(available_backends())
    print("laya-universal backends:")
    for name in sorted(_BACKENDS):
        print(f"  {name}: {'available' if name in avail else 'not available'}")


def _run_convert(args):
    from .convert import export_to_onnx

    export_to_onnx(args.model, args.output, args.dtype)


def _run_predict(args):
    import laya_universal as laya

    if args.state:
        state = args.state
    elif args.state_file:
        try:
            with open(args.state_file, encoding="utf-8") as f:
                state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
            print(f"ERROR: Could not read state file {args.state_file!r}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("ERROR: Provide --state or --state-file", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.questions, encoding="utf-8") as f:
            questions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        print(f"ERROR: Could not read questions file {args.questions!r}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        agent = laya.load(args.model, dtype=args.dtype, device=args.device, backend=args.backend)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded with backend: {agent.backend_name}", file=sys.stderr)

    try:
        result = agent.predict(state, questions)
    except Exception as e:
        print(f"ERROR: Prediction failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
