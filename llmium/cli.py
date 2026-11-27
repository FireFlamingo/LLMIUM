import argparse
import json
import sys
from pathlib import Path

from .core import repair, scan


def main():
    parser = argparse.ArgumentParser(prog="llmium")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("scan", help="Inspect a safetensors file or shard directory")
    inspect.add_argument("path")
    inspect.add_argument("--reference")
    inspect.add_argument("--output", required=True)
    restore = commands.add_parser("repair", help="Restore named tensors from a trusted reference")
    restore.add_argument("path")
    restore.add_argument("--reference", required=True)
    restore.add_argument("--output", required=True)
    restore.add_argument("--tensor", action="append", required=True)
    demo = commands.add_parser("demo")
    demo.add_argument("--output", default="runs/demo")
    serve = commands.add_parser("serve")
    serve.add_argument("--root", default="runs")
    serve.add_argument("--port", type=int, default=8765)
    commands.add_parser("mcp", help="Run the read-only stdio agent interface")
    args = parser.parse_args()
    try:
        if args.command == "scan":
            result = scan(args.path, args.reference)
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps({"report": str(target), "tensors": len(result["tensors"]), "findings": len(result["findings"])}))
        elif args.command == "repair":
            print(json.dumps(repair(args.path, args.reference, args.output, args.tensor)))
        elif args.command == "demo":
            from .demo import run_demo
            result = run_demo(args.output)
            print(json.dumps({"report": str(Path(args.output) / "demo.json"), "recovered": result["exact_forward_recovery"]}))
        elif args.command == "serve":
            from .server import serve
            serve(args.root, args.port)
        else:
            from .mcp import run
            run()
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
